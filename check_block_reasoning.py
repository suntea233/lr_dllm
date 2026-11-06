"""CPU checks of confidence-stopped, sequential hidden blocks on a tiny LLaDA.

Only confidence probes are mocked. Backbone computation, final answer logits,
cross-block gradients, and checkpoint recomputation use the real tiny model.
These checks establish implementation semantics, not reasoning quality.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))

import torch
from transformers import AutoConfig, AutoModel

import block_reasoning
from experiment import add_lora, answer_loss


class ControlledProbes:
    """Return prescribed no-grad probe distributions, preserving the final head."""

    def __init__(self, patterns, block_size, final_grad_enabled=True):
        self.patterns = list(patterns)
        self.block_size = block_size
        self.final_grad_enabled = final_grad_enabled
        self.original = block_reasoning.answer_logits
        self.calls = 0
        self.inputs = []
        self.confidences = []

    def __call__(self, model, hidden):
        number = self.calls
        self.calls += 1
        if number == len(self.patterns):
            assert torch.is_grad_enabled() == self.final_grad_enabled
            return self.original(model, hidden)
        assert number < len(self.patterns), 'Unexpected extra vocabulary projection'
        assert not torch.is_grad_enabled(), 'Confidence probes must not build a loss graph'
        assert hidden.shape[1] == self.block_size
        self.inputs.append(hidden.detach().clone())
        logits = torch.zeros((*hidden.shape[:-1], 64), device=hidden.device, dtype=torch.float32)
        pattern = self.patterns[number]
        if pattern in ('high', 'one_low'):
            # EOS remains in the probability space: no hidden special-token filter.
            logits[..., 2] = 20.0
            if pattern == 'one_low':
                logits[:, 0] = 0.0
        else:
            assert pattern == 'low'
        self.confidences.append(logits.softmax(-1).max(-1).values)
        return logits


def assert_stats(stats, expected_rounds, expected_reasons):
    assert stats['block_rounds'] == expected_rounds, stats
    assert [block['iterations'] for block in stats['blocks']] == expected_rounds
    assert [block['stop_reason'] for block in stats['blocks']] == expected_reasons
    assert stats['actual_backbone_calls'] == sum(expected_rounds) + 1
    assert stats['confidence_head_calls'] == sum(expected_rounds)
    assert stats['answer_head_calls'] == 1


def assert_trace_gradients(trace):
    minimum_raw, minimum_feedback = [], []
    for record in trace:
        for key, minima in (('raw_hidden', minimum_raw), ('feedback_hidden', minimum_feedback)):
            gradient = record[key].grad
            assert gradient is not None, f'Missing {key} gradient in {record["block_index"]}'
            assert torch.isfinite(gradient).all()
            minimum = gradient.float().norm(dim=-1).min().item()
            assert minimum > 0, f'A {key} position lost its answer-loss gradient'
            minima.append(minimum)
    return dict(minimum_raw_position_gradient=min(minimum_raw),
                minimum_feedback_position_gradient=min(minimum_feedback))


def inspect_inputs(model, prompt, labels, settings, patterns, expected_rounds, reasons):
    model.zero_grad(set_to_none=True)
    model.train()
    inputs, outputs, trace, stats = [], [], [], {}
    original_backbone = block_reasoning.backbone_hidden
    probes = ControlledProbes(patterns, settings['block_size'])

    def capture_backbone(current_model, embeddings):
        inputs.append(embeddings.detach().clone())
        result = original_backbone(current_model, embeddings)
        outputs.append(result.detach().clone())
        return result

    with patch.object(block_reasoning, 'backbone_hidden', side_effect=capture_backbone), \
            patch.object(block_reasoning, 'answer_logits', side_effect=probes):
        logits = block_reasoning.block_recurrent_logits(
            model, prompt, checkpointing=False, trace=trace, stats=stats,
            special_token_ids=[2, 63], **settings)
        assert len(inputs) == sum(expected_rounds) + 1
        assert probes.calls == sum(expected_rounds) + 1
        assert_stats(stats, expected_rounds, reasons)
        loss = answer_loss(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()

    embedding = model.get_input_embeddings()
    prompt_embed = embedding(prompt).detach()
    mask = embedding(torch.tensor([[settings['mask_id']]])).detach()
    scale = mask.float().square().mean(-1, keepdim=True).sqrt()
    completed = []
    index = 0
    width = settings['block_size']
    for block, iterations in enumerate(expected_rounds):
        current = mask.expand(-1, width, -1)
        for iteration in range(iterations):
            # This exact input excludes all future blocks and all answer slots.
            expected = torch.cat([prompt_embed, *completed, current], dim=1)
            torch.testing.assert_close(inputs[index], expected, rtol=0, atol=0)
            start = prompt.shape[1] + block * width
            raw = outputs[index][:, start:start + width]
            torch.testing.assert_close(probes.inputs[index], raw, rtol=0, atol=0)
            record = trace[index]
            assert record['block_index'] == block and record['round_index'] == iteration + 1
            torch.testing.assert_close(record['raw_hidden'].detach(), raw, rtol=0, atol=0)
            normalized = (raw.float() * scale /
                          raw.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)).to(mask.dtype)
            torch.testing.assert_close(record['feedback_hidden'].detach(), normalized, rtol=0, atol=0)
            assert not torch.equal(raw, normalized), 'Probe test must distinguish pre/post RMS hidden'
            assert record['stop_reason'] == (reasons[block] if iteration == iterations - 1 else None)
            current = normalized
            index += 1
        completed.append(current)
    assert len(trace) == index == sum(expected_rounds)
    final_inputs = torch.cat([prompt_embed, *completed,
                              mask.expand(-1, settings['answer_slots'], -1)], dim=1)
    torch.testing.assert_close(inputs[-1], final_inputs, rtol=0, atol=0)
    assert inputs[-1].shape[1] == prompt.numel() + settings['latent_slots'] + settings['answer_slots']
    gradients = assert_trace_gradients(trace)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.requires_grad and p.grad is not None)
    for index, pattern in enumerate(patterns):
        if pattern == 'one_low':
            # Mean confidence passes; minimum confidence fails. This distinguishes
            # the requested all-position rule from a mean or any-position rule.
            assert probes.confidences[index].mean().item() > settings['confidence_threshold']
            assert probes.confidences[index].min().item() < settings['confidence_threshold']
        if pattern == 'high':
            assert probes.confidences[index].min().item() > settings['confidence_threshold']
    return dict(backbone_calls=len(inputs), confidence_head_calls=len(patterns),
                answer_head_calls=1, block_rounds=expected_rounds, stop_reasons=reasons,
                input_sequence_lengths=[x.shape[1] for x in inputs],
                future_blocks_absent=True, completed_persistent_hidden_unchanged=True,
                confidence_checked_before_feedback_rms=True,
                **gradients)


def gradient_snapshot(model, prompt, labels, settings, patterns, checkpointing):
    model.train()
    model.zero_grad(set_to_none=True)
    stats, trace = {}, []
    probes = ControlledProbes(patterns, settings['block_size'])
    with patch.object(block_reasoning, 'answer_logits', side_effect=probes):
        logits = block_reasoning.block_recurrent_logits(
            model, prompt, checkpointing=checkpointing, trace=trace, stats=stats,
            special_token_ids=[2, 63], **settings)
        loss = answer_loss(logits, labels)
        loss.backward()
        assert probes.calls == len(patterns) + 1, 'Backward must not repeat halting probes'
    assert_trace_gradients(trace)
    gradients = {name: parameter.grad.detach().float().clone()
                 for name, parameter in model.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    assert gradients
    return logits.detach().float(), loss.detach().float(), gradients, stats


def checkpoint_parity(model, prompt, labels, settings, patterns, expected_rounds, reasons):
    direct = gradient_snapshot(model, prompt, labels, settings, patterns, False)
    checkpointed = gradient_snapshot(model, prompt, labels, settings, patterns, True)
    for value in (direct, checkpointed):
        assert_stats(value[3], expected_rounds, reasons)
    torch.testing.assert_close(checkpointed[0], direct[0], atol=2e-7, rtol=1e-4)
    torch.testing.assert_close(checkpointed[1], direct[1], atol=2e-7, rtol=1e-4)
    assert direct[2].keys() == checkpointed[2].keys()
    for name in direct[2]:
        torch.testing.assert_close(checkpointed[2][name], direct[2][name], atol=2e-7, rtol=1e-4)
    # Identical confidence rule in eval mode, including the no-grad final readout.
    model.eval()
    stats = {}
    probes = ControlledProbes(patterns, settings['block_size'], final_grad_enabled=False)
    with torch.no_grad(), patch.object(block_reasoning, 'answer_logits', side_effect=probes):
        inference = block_reasoning.block_recurrent_logits(
            model, prompt, checkpointing=True, stats=stats, **settings)
    assert_stats(stats, expected_rounds, reasons)
    torch.testing.assert_close(inference.float(), direct[0], atol=2e-7, rtol=1e-4)
    return dict(trainable_gradient_tensors=len(direct[2]),
                max_logit_error=float((direct[0] - checkpointed[0]).abs().max()),
                max_gradient_error=max(float((direct[2][name] - checkpointed[2][name]).abs().max())
                                       for name in direct[2]),
                evaluation_uses_same_halting_rule=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'checkpoints/tiny-random-llada')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    signature = inspect.signature(block_reasoning.block_recurrent_logits)
    assert not {'labels', 'answer_ids', 'gold'}.intersection(signature.parameters)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for name, value in dict(d_model=32, n_heads=4, n_kv_heads=4, n_layers=2,
                            mlp_hidden_size=64, vocab_size=64, embedding_size=64,
                            max_sequence_length=512, mask_token_id=63,
                            eos_token_id=2, pad_token_id=2, init_device='cpu',
                            embedding_dropout=0.0, attention_dropout=0.0,
                            residual_dropout=0.0).items():
        setattr(config, name, value)
    prompt = torch.tensor([[4, 5, 6]])
    small = dict(mask_id=63, latent_slots=32, answer_slots=3, block_size=16,
                 confidence_threshold=0.9, min_block_rounds=1, max_block_rounds=3)
    labels = torch.tensor([[7, 2, -100]])
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        model = add_lora(AutoModel.from_config(config, trust_remote_code=True,
                                              init_params=True).to(dtype), rank=4).train()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=0.01)
        patterns = ['one_low', 'high', 'high']
        rounds, reasons = [2, 1], ['confidence', 'confidence']
        semantics = inspect_inputs(model, prompt, labels, small, patterns, rounds, reasons)
        parity = checkpoint_parity(model, prompt, labels, small, patterns, rounds, reasons)
        records.append(dict(case='mixed_confidence_schedule', dtype=str(dtype), **semantics, **parity))
        forced = dict(small, max_block_rounds=2, confidence_threshold=1 / 64)
        # Uniform confidence equals this threshold exactly; the rule is strict >.
        records.append(dict(case='strict_threshold_and_budget_cap', dtype=str(dtype),
                            **inspect_inputs(model, prompt, labels, forced, ['low'] * 4,
                                             [2, 2], ['max_rounds'] * 2)))
        minimum = dict(small, min_block_rounds=2)
        records.append(dict(case='minimum_rounds', dtype=str(dtype),
                            **inspect_inputs(model, prompt, labels, minimum, ['high'] * 4,
                                             [2, 2], ['confidence'] * 2)))
        if dtype == torch.float32:
            formal = dict(small, latent_slots=240, answer_slots=16, max_block_rounds=8)
            formal_labels = torch.tensor([[7, 2] + [-100] * 14])
            records.append(dict(case='formal_15_blocks_worst_case', dtype=str(dtype),
                                **inspect_inputs(model, prompt, formal_labels, formal, ['low'] * 120,
                                                 [8] * 15, ['max_rounds'] * 15)))
    result = dict(status='passed', device='cpu', checks=records,
                  note='Tiny-model correctness checks only. No 8B memory, latency, confidence calibration, '
                       'or reasoning-quality claim. Backbone counts exclude backward recomputation.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
