"""Check recurrence gradients, BF16, real input layout, and adapter round-trip."""
import copy
import argparse
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache' / 'modules'))

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel
from experiment import add_lora, answer_loss, load_fp32_adapter, recurrent_logits
from llada_backbone import native_llada


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/tiny-random-llada')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True,
                                        local_files_only=True)
    for name, value in dict(d_model=32, n_heads=4, n_kv_heads=4, n_layers=2,
                            mlp_hidden_size=64, vocab_size=64, embedding_size=64,
                            max_sequence_length=64, mask_token_id=63,
                            eos_token_id=2, pad_token_id=2, init_device='cpu').items():
        setattr(config, name, value)
    def new_base():
        return AutoModel.from_config(config, trust_remote_code=True, init_params=True)
    base = new_base()
    base_state = copy.deepcopy(base.state_dict())
    model = add_lora(base, rank=4).train()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    prompt, labels = torch.tensor([[4, 5, 6]]), torch.tensor([[7, 2, -100]])
    settings = dict(mask_id=63, latent_slots=4, answer_slots=3, rounds=2)
    trace = []
    logits = recurrent_logits(model, prompt, trace=trace, **settings)
    assert logits.shape == (1, 3, 64)
    answer_loss(logits, labels).backward()
    assert trace[0].grad is not None and trace[0].grad.abs().sum() > 0
    first_grad_norm = trace[0].grad.norm().item()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)

    # Compare complete parameter gradients, not just forward outputs.
    for rounds in (1, 2, 3):
        gradients, outputs = [], []
        for use_checkpoint in (False, True):
            model.zero_grad(set_to_none=True)
            out = recurrent_logits(model, prompt, **dict(settings, rounds=rounds),
                                   checkpointing=use_checkpoint)
            answer_loss(out, labels).backward()
            outputs.append(out.detach())
            gradients.append({n: p.grad.detach().clone() for n, p in model.named_parameters()
                              if p.requires_grad and p.grad is not None})
        torch.testing.assert_close(outputs[0], outputs[1])
        assert gradients[0].keys() == gradients[1].keys()
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-4, atol=2e-7)

    captured, hidden = [], []
    def capture(module, positional, kwargs):
        captured.append(positional[0].detach().clone())
    handle = native_llada(model).model.transformer.emb_drop.register_forward_pre_hook(capture, with_kwargs=True)
    out = recurrent_logits(model, prompt, checkpointing=False, trace=hidden, **settings)
    handle.remove()
    embedding = model.get_input_embeddings()
    mask = embedding(torch.tensor([[63]])).detach()
    for x in captured:
        torch.testing.assert_close(x[:, :3], embedding(prompt))
        torch.testing.assert_close(x[:, -3:], mask.expand(-1, 3, -1))
    torch.testing.assert_close(captured[0][:, 3:7], mask.expand(-1, 4, -1))
    expected = hidden[0].detach().float()
    expected = expected / expected.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    expected = expected * mask.float().square().mean(-1, keepdim=True).sqrt()
    torch.testing.assert_close(captured[1][:, 3:7], expected)
    changed_padding = out.detach().clone()
    changed_padding[:, -1] = torch.randn_like(changed_padding[:, -1]) * 100
    torch.testing.assert_close(answer_loss(out, labels), answer_loss(changed_padding, labels))
    torch.testing.assert_close(answer_loss(out, labels), torch.nn.functional.cross_entropy(
        out[:, :2].reshape(-1, 64), labels[:, :2].reshape(-1)))

    initial = answer_loss(recurrent_logits(model, prompt, **settings), labels).item()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = answer_loss(recurrent_logits(model, prompt, **settings), labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_logits = recurrent_logits(model, prompt, **settings)
        final = answer_loss(expected_logits, labels).item()
    assert final < initial
    for name, param in model.named_parameters():
        if name in frozen:
            assert torch.equal(frozen[name], param)
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory, save_embedding_layers=False)
        restored_base = new_base()
        restored_base.load_state_dict(base_state, strict=True)
        restored = PeftModel.from_pretrained(restored_base, directory).eval()
        with torch.no_grad():
            actual = recurrent_logits(restored, prompt, **settings)
        torch.testing.assert_close(actual, expected_logits, rtol=0, atol=0)
    mixed = add_lora(new_base().to(torch.bfloat16), rank=4).train()
    mixed_trace = []
    mixed_loss = answer_loss(recurrent_logits(mixed, prompt, trace=mixed_trace, **settings), labels)
    mixed_loss.backward()
    assert torch.isfinite(mixed_loss)
    assert mixed_trace[0].grad is not None and mixed_trace[0].grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in mixed.parameters() if p.grad is not None)
    optimizer = torch.optim.AdamW([p for p in mixed.parameters() if p.requires_grad], lr=0.002)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    answer_loss(recurrent_logits(mixed, prompt, **settings), labels).backward()
    optimizer.step()
    mixed.eval()
    with torch.no_grad():
        before = recurrent_logits(mixed, prompt, **settings)
    with tempfile.TemporaryDirectory() as directory:
        mixed.save_pretrained(directory, save_embedding_layers=False)
        load_fp32_adapter(mixed, directory)
        parameters = dict(mixed.named_parameters())
        for name, value in parameters.items():
            if '.default.' in name and 'lora_' in name:
                torch.testing.assert_close(value, parameters[name.replace('.default.', '.roundtrip.')],
                                           rtol=0, atol=0)
        mixed.eval()
        with torch.no_grad():
            torch.testing.assert_close(recurrent_logits(mixed, prompt, **settings), before, rtol=0, atol=0)
    result = dict(status='passed', device='cpu', random_tiny_model=True,
                  first_hidden_gradient_norm=first_grad_norm, initial_ce=initial, final_ce=final,
                  checkpoint_gradient_parity_rounds=[1, 2, 3],
                  input_layout_and_masked_answers='passed', ignored_padding_loss='passed',
                  same_position_loss='passed', frozen_base='passed', adapter_reload='exact',
                  bf16_backward='passed', bf16_adapter_reload='exact', optimizer_steps=30,
                  note='Implementation checks; not evidence of reasoning ability.')
    (ROOT / 'training_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
