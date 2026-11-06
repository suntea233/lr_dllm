"""CPU checks for all-position continuous hidden feedback, without token commits.

Tests inspect actual recurrence inputs, rather than treating the schedule alone
as evidence that hidden states are updated correctly. Every latent position is
fed back after the first pass, including when block_size is supplied. The larger
case uses the formal 240-latent + 16-answer geometry with a tiny random backbone.
"""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))

import torch
from transformers import AutoConfig, AutoModel

import experiment


def gradient_snapshot(model, prompt, labels, settings, checkpointing):
    model.zero_grad(set_to_none=True)
    logits = experiment.recurrent_logits(model, prompt, checkpointing=checkpointing,
                                         **settings)
    loss = experiment.answer_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {name: value.grad.detach().float().clone()
                 for name, value in model.named_parameters()
                 if value.requires_grad and value.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
    return logits.detach().float(), loss.detach().float(), gradients


def check_actual_feedback(model, prompt, labels, settings, expected_counts):
    """Inspect inputs and state gradients on a single, non-checkpointed graph."""
    model.zero_grad(set_to_none=True)
    inputs, trace = [], []
    original_backbone = experiment.backbone_hidden
    original_head = experiment.answer_logits

    def capture_backbone(current_model, embeddings):
        inputs.append(embeddings.detach().clone())
        return original_backbone(current_model, embeddings)

    with patch.object(experiment, 'backbone_hidden', side_effect=capture_backbone) as backbone, \
            patch.object(experiment, 'answer_logits', wraps=original_head) as head:
        logits = experiment.recurrent_logits(model, prompt, checkpointing=False,
                                             trace=trace, **settings)
        assert backbone.call_count == settings['rounds']
        assert head.call_count == 1
        experiment.answer_loss(logits, labels).backward()

    assert len(inputs) == len(trace) == len(expected_counts)
    start, latent_slots, answer_slots = prompt.shape[1], settings['latent_slots'], settings['answer_slots']
    assert expected_counts == [0] + [latent_slots] * (settings['rounds'] - 1)
    embedding = model.get_input_embeddings()
    prompt_embedding = embedding(prompt).detach()
    mask = embedding(torch.tensor([[settings['mask_id']]], device=prompt.device)).detach()
    scale = mask.float().square().mean(-1, keepdim=True).sqrt()
    update_detected = False
    for step, (actual, active) in enumerate(zip(inputs, expected_counts)):
        assert actual.shape[1] == start + latent_slots + answer_slots
        torch.testing.assert_close(actual[:, :start], prompt_embedding, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, start + latent_slots:],
                                   mask.expand(-1, answer_slots, -1), atol=0, rtol=0)
        if active:
            previous = trace[step - 1].detach().float()
            expected = previous * scale / previous.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            torch.testing.assert_close(actual[:, start:start + active],
                                       expected[:, :active].to(actual.dtype), atol=0, rtol=0)
        if active < latent_slots:
            torch.testing.assert_close(actual[:, start + active:start + latent_slots],
                                       mask.expand(-1, latent_slots - active, -1), atol=0, rtol=0)
        if step > 1 and expected_counts[step - 1]:
            old_active = expected_counts[step - 1]
            update_detected |= not torch.equal(actual[:, start:start + old_active],
                                               inputs[step - 1][:, start:start + old_active])
    assert update_detected, 'The test did not observe historical hidden states being refreshed'

    feedback_gradients = []
    minimum_slot_gradients = []
    for step, state in enumerate(trace[:-1]):
        active_next = expected_counts[step + 1]
        assert active_next == latent_slots
        assert state.grad is not None
        gradient = state.grad.float()
        norm = gradient.norm().item()
        assert norm > 0 and torch.isfinite(gradient).all()
        minimum_slot_gradient = gradient.norm(dim=-1).min().item()
        assert minimum_slot_gradient > 0, \
            'Every reasoning position must receive a cross-round feedback gradient'
        feedback_gradients.append(norm)
        minimum_slot_gradients.append(minimum_slot_gradient)
    # The final latent output is not a loss readout and is not fed back again.
    assert trace[-1].grad is None
    return dict(input_active_latents=expected_counts,
                input_mask_positions=[latent_slots + answer_slots - n for n in expected_counts],
                backbone_calls=settings['rounds'], answer_head_calls=1,
                historical_hidden_states_refreshed=True,
                feedback_gradient_norms=feedback_gradients,
                minimum_per_position_feedback_gradient_norms=minimum_slot_gradients,
                all_reasoning_positions_fed_back_after_first_pass=True,
                answer_inputs_always_masked=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/tiny-random-llada')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)

    formal_counts = [0] + [240] * 7
    assert experiment.hidden_feedback_counts(240, 16, 8, 16) == formal_counts
    assert experiment.hidden_feedback_counts(240, 16, 8, None) == formal_counts
    assert experiment.hidden_feedback_counts(24, 4, 4, 4) == [0, 24, 24, 24]
    assert experiment.hidden_feedback_counts(24, 4, 4, None) == [0, 24, 24, 24]

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for name, value in dict(d_model=32, n_heads=4, n_kv_heads=4, n_layers=2,
                            mlp_hidden_size=64, vocab_size=64, embedding_size=64,
                            max_sequence_length=512, mask_token_id=63,
                            eos_token_id=2, pad_token_id=2, init_device='cpu',
                            embedding_dropout=0.0, attention_dropout=0.0,
                            residual_dropout=0.0).items():
        setattr(config, name, value)
    prompt = torch.tensor([[4, 5, 6]])
    cases = [('small', dict(latent_slots=24, answer_slots=4, rounds=4, block_size=4),
              [0, 24, 24, 24]),
             ('formal_geometry', dict(latent_slots=240, answer_slots=16, rounds=8, block_size=16),
              formal_counts)]
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        model = experiment.add_lora(AutoModel.from_config(
            config, trust_remote_code=True, init_params=True).to(dtype), rank=4).train()
        with torch.no_grad():
            # Exercise both LoRA factors, rather than relying only on zero-B initialization.
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=0.01)
        for case_name, values, counts in cases:
            settings = dict(mask_id=63, **values)
            labels = torch.tensor([[7, 2] + [-100] * (values['answer_slots'] - 2)])
            semantics = check_actual_feedback(model, prompt, labels, settings, counts)
            direct = gradient_snapshot(model, prompt, labels, settings, checkpointing=False)
            checkpointed = gradient_snapshot(model, prompt, labels, settings, checkpointing=True)
            torch.testing.assert_close(checkpointed[0], direct[0], rtol=1e-4, atol=2e-7)
            torch.testing.assert_close(checkpointed[1], direct[1], rtol=1e-4, atol=2e-7)
            assert checkpointed[2].keys() == direct[2].keys()
            for name in direct[2]:
                torch.testing.assert_close(checkpointed[2][name], direct[2][name], rtol=1e-4, atol=2e-7)
            records.append(dict(case=case_name, dtype=str(dtype), **semantics,
                                trainable_gradient_tensors=len(direct[2]),
                                checkpoint_max_logit_error=float((checkpointed[0] - direct[0]).abs().max()),
                                checkpoint_max_gradient_error=max(
                                    float((checkpointed[2][name] - direct[2][name]).abs().max())
                                    for name in direct[2])))
    result = dict(status='passed', device='cpu', formal_input_active_latents=formal_counts,
                  checks=records,
                  note='Tiny random-model implementation checks, not reasoning or 8B stability results. '
                       'Forward counts exclude training checkpoint recomputation.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
