"""Compare native full-LM recurrence with the head-free path, including gradients."""
import json
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))
import torch
from transformers import AutoConfig, AutoModel
from experiment import add_lora, answer_loss, recurrent_logits
from llada_backbone import native_llada


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/tiny-random-llada')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    config = AutoConfig.from_pretrained(args.model,
                                        trust_remote_code=True, local_files_only=True)
    config.vocab_size = config.embedding_size = 64
    config.mask_token_id, config.eos_token_id, config.pad_token_id = 63, 2, 2
    config.max_sequence_length = 64
    prompt, labels = torch.tensor([[4, 5, 6]]), torch.tensor([[7, 2, -100]])
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        model = add_lora(AutoModel.from_config(config, trust_remote_code=True,
                                               init_params=True).to(dtype), rank=4).train()
        # Nonzero B weights make A and feedback-dependent gradients observable.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    param.normal_(std=0.01)
        for rounds in (1, 2, 3):
            for checkpointing in (False, True):
                records = []
                for readout in ('all_logits', 'final_only'):
                    model.zero_grad(set_to_none=True)
                    output = recurrent_logits(model, prompt, 63, 4, 3, rounds,
                                               checkpointing=checkpointing, readout=readout)
                    loss = answer_loss(output, labels)
                    loss.backward()
                    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                             if p.requires_grad and p.grad is not None}
                    records.append((output.detach(), loss.detach(), grads))
                old, new = records
                atol, rtol = ((2e-6, 2e-4) if dtype == torch.float32 else (2e-3, 2e-2))
                torch.testing.assert_close(new[0], old[0], atol=atol, rtol=rtol)
                torch.testing.assert_close(new[1], old[1], atol=atol, rtol=rtol)
                assert new[2].keys() == old[2].keys()
                grad_error = 0.0
                for name in old[2]:
                    torch.testing.assert_close(new[2][name], old[2][name], atol=atol, rtol=rtol)
                    grad_error = max(grad_error, float((new[2][name] - old[2][name]).abs().max()))
                checks.append(dict(dtype=str(dtype), rounds=rounds, checkpointing=checkpointing,
                                   max_logit_error=float((new[0] - old[0]).abs().max()),
                                   max_gradient_error=grad_error))
        core = native_llada(model).model
        readouts, backbone_calls = {}, {}
        for path in ('all_logits', 'final_only'):
            tokens, blocks = [], []
            head_hook = core.transformer.ff_out.register_forward_pre_hook(
                lambda module, inputs: tokens.append(inputs[0].shape[1]))
            block_hook = core.transformer.blocks[0].register_forward_pre_hook(
                lambda module, inputs: blocks.append(inputs[0].shape[1]))
            model.eval()
            with torch.no_grad():
                recurrent_logits(model, prompt, 63, 4, 3, 2, readout=path)
            head_hook.remove()
            block_hook.remove()
            readouts[path], backbone_calls[path] = tokens, blocks
        assert readouts == {'all_logits': [10, 10], 'final_only': [3]}
        assert backbone_calls == {'all_logits': [10, 10], 'final_only': [10, 10]}
    result = dict(status='passed', device='cpu', comparisons=checks,
                  lm_head_input_lengths=readouts, first_block_input_lengths=backbone_calls,
                  note='Same Transformer iterations; intermediate LM head and full-layer hidden collection removed.')
    (ROOT / 'readout_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
