"""Check explicit LLaDA model partitioning and full recurrent LoRA gradients.

Default execution is CPU-only. To exercise real inter-GPU copies explicitly use
``--devices cuda:0 cuda:1`` under the intended CUDA_VISIBLE_DEVICES mapping.
The test uses four tiny blocks and a 64-token vocabulary, not 8B weights.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))

import torch
from transformers import AutoConfig, AutoModel

from experiment import add_lora, answer_loss, load_fp32_adapter, recurrent_logits
from llada_backbone import native_llada, partition_llada


def gradient_record(model, prompt, labels, rounds, checkpointing):
    model.zero_grad(set_to_none=True)
    trace = []
    logits = recurrent_logits(model, prompt, mask_id=63, latent_slots=4,
                              answer_slots=3, rounds=rounds, trace=trace,
                              checkpointing=checkpointing)
    loss = answer_loss(logits, labels.to(logits.device))
    loss.backward()
    gradients = {name: value.grad.detach().float().cpu().clone()
                 for name, value in model.named_parameters()
                 if value.requires_grad and value.grad is not None}
    if rounds > 1:
        assert trace[0].grad is not None and trace[0].grad.float().norm().item() > 0
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
    return logits.detach().float().cpu(), float(loss), gradients


def optimizer_step(model):
    parameters = [value for value in model.parameters() if value.requires_grad]
    before = [value.detach().clone() for value in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    optimizer.step()
    changed_devices = {str(value.device) for value, previous in zip(parameters, before)
                       if not torch.equal(value, previous)}
    assert changed_devices == {str(value.device) for value in parameters}
    weights = {name: value.detach().float().cpu().clone()
               for name, value in model.named_parameters() if value.requires_grad}
    return float(norm), weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/tiny-random-llada')
    parser.add_argument('--devices', nargs='+', default=['cpu'])
    parser.add_argument('--rounds', nargs='+', type=int, default=[1, 2, 8])
    parser.add_argument('--max-memory-gib', type=float, default=0.5,
                        help='Per-device PyTorch allocation cap for this tiny test; excludes CUDA context')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    devices = [torch.device(value) for value in args.devices]
    if not devices or args.max_memory_gib <= 0 or min(args.rounds) < 1:
        raise ValueError('Require devices, positive memory limit and positive round counts')
    if len(set(devices)) != len(devices):
        raise ValueError('Use distinct model devices')
    if any(d.type not in ('cpu', 'cuda') for d in devices):
        raise ValueError('Only CPU and CUDA are supported')
    if len(devices) > 1 and any(d.type != 'cuda' for d in devices):
        raise ValueError('Multi-device tests require CUDA devices')
    if any(d.type == 'cuda' and d.index is None for d in devices):
        raise ValueError('Use explicitly indexed CUDA devices')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for device in devices:
        if device.type == 'cuda':
            if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
                raise RuntimeError(f'CUDA device unavailable: {device}')
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, args.max_memory_gib * 2**30 / total), device)
            torch.cuda.reset_peak_memory_stats(device)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for name, value in dict(d_model=32, n_heads=4, n_kv_heads=4, n_layers=4,
                            mlp_hidden_size=64, vocab_size=64, embedding_size=64,
                            max_sequence_length=64, mask_token_id=63,
                            eos_token_id=2, pad_token_id=2, init_device='cpu',
                            embedding_dropout=0.0, attention_dropout=0.0,
                            residual_dropout=0.0).items():
        setattr(config, name, value)
    prompt = torch.tensor([[4, 5, 6]], device=devices[0])
    labels = torch.tensor([[7, 2, -100]])
    checks, placements, reload_checks = [], [], []
    for dtype in (torch.float32, torch.bfloat16):
        template = add_lora(AutoModel.from_config(config, trust_remote_code=True,
                                                init_params=True).to(dtype), rank=4).train()
        # Exercise A and B gradients, including gradients through recurrent state.
        with torch.no_grad():
            for name, parameter in template.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=0.01)
        state_keys = set(template.state_dict())
        for rounds in args.rounds:
            for checkpointing in (False, True):
                reference = copy.deepcopy(template).to(devices[0])
                candidate = copy.deepcopy(template)
                placement = partition_llada(candidate, devices)
                assert set(candidate.state_dict()) == state_keys
                assert native_llada(candidate).model.transformer.wte.weight.device == devices[0]
                assert next(native_llada(candidate).model.transformer.ln_f.parameters()).device == devices[-1]
                old = gradient_record(reference, prompt, labels, rounds, checkpointing)
                new = gradient_record(candidate, prompt, labels, rounds, checkpointing)
                atol, rtol = ((3e-6, 3e-4) if dtype == torch.float32 else (5e-3, 5e-2))
                torch.testing.assert_close(new[0], old[0], atol=atol, rtol=rtol)
                torch.testing.assert_close(torch.tensor(new[1]), torch.tensor(old[1]), atol=atol, rtol=rtol)
                assert new[2].keys() == old[2].keys()
                for name in old[2]:
                    torch.testing.assert_close(new[2][name], old[2][name], atol=atol, rtol=rtol)
                # Check that optimization works for trainable tensors on both cards.
                old_norm, old_weights = optimizer_step(reference)
                new_norm, new_weights = optimizer_step(candidate)
                torch.testing.assert_close(torch.tensor(new_norm), torch.tensor(old_norm), atol=atol, rtol=rtol)
                for name in old_weights:
                    torch.testing.assert_close(new_weights[name], old_weights[name], atol=atol, rtol=rtol)
                active_devices = {str(value.device) for value in candidate.parameters()
                                  if value.requires_grad and value.grad is not None
                                  and value.grad.float().abs().sum().item() > 0}
                assert active_devices == {str(device) for device in devices}
                checks.append(dict(dtype=str(dtype), rounds=rounds, checkpointing=checkpointing,
                                   trainable_gradient_tensors=len(new[2]),
                                   max_logit_error=float((new[0] - old[0]).abs().max()),
                                   max_gradient_error=max(float((new[2][name] - old[2][name]).abs().max())
                                                          for name in old[2]),
                                   active_gradient_devices=sorted(active_devices),
                                   optimizer_update_parity=True))
                placements.append(placement)
                del reference, candidate
        # Loading a new named adapter must preserve both FP32 precision and each
        # layer's CUDA placement, rather than gathering adapters onto one GPU.
        restored = copy.deepcopy(template)
        partition_llada(restored, devices)
        gradient_record(restored, prompt, labels, rounds=2, checkpointing=True)
        optimizer_step(restored)
        restored.eval()
        with torch.no_grad():
            expected = recurrent_logits(restored, prompt, 63, 4, 3, 2).detach().cpu()
        with tempfile.TemporaryDirectory() as directory:
            restored.save_pretrained(directory, save_embedding_layers=False)
            load_fp32_adapter(restored, directory)
            restored.eval()
            named = dict(restored.named_parameters())
            for name, parameter in named.items():
                if '.default.' in name and 'lora_' in name:
                    loaded = named[name.replace('.default.', '.roundtrip.')]
                    assert loaded.device == parameter.device
                    torch.testing.assert_close(loaded, parameter, atol=0, rtol=0)
            with torch.no_grad():
                actual = recurrent_logits(restored, prompt, 63, 4, 3, 2).detach().cpu()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        reload_checks.append(dict(dtype=str(dtype), exact_logits=True,
                                  exact_adapter_weights=True, layer_devices_preserved=True))
        del restored
    memory = {}
    for device in devices:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            memory[str(device)] = dict(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                                       reserved_mib=torch.cuda.memory_reserved(device) / 2**20)
    result = dict(status='passed', devices=[str(device) for device in devices],
                  real_cross_device_test=len(devices) > 1, checks=checks,
                  adapter_reload=reload_checks,
                  placement=placements[-1], pytorch_memory=memory,
                  note='Tiny implementation check, not an 8B memory/speed measurement. '
                       'CUDA context memory is not included in PyTorch allocator statistics.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
