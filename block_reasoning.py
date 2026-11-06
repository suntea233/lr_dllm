"""Sequential latent blocks with confidence-based, bounded iteration counts.

Only the current block's raw final hidden states are inspected by the vocabulary
head. The probabilities decide when to advance; no predicted token is fed back.
Completed blocks keep their persistent hidden inputs, with gradients attached.
Their intermediate backbone activations are recomputed, and may attend to the
current block. Future blocks and answer queries enter only when their turn starts.
"""
from collections import Counter
import math

import torch
from torch.utils.checkpoint import checkpoint

from llada_backbone import answer_logits, backbone_hidden


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _special_ids(model, mask_id, special_token_ids):
    if special_token_ids is None:
        ids = [mask_id]
        eos = getattr(getattr(model, 'config', None), 'eos_token_id', None)
        if eos is not None:
            ids.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    else:
        ids = list(special_token_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
        raise ValueError('special_token_ids must contain nonnegative integer IDs')
    return frozenset(ids)


@torch.no_grad()
def _confidence_probe(model, raw_hidden, special_ids):
    """Original vocabulary, native logit scaling, temperature 1, FP32 softmax."""
    logits = answer_logits(model, raw_hidden)
    confidence, predicted = logits.float().softmax(dim=-1).max(dim=-1)
    if not torch.isfinite(confidence).all().item():
        raise FloatingPointError('Non-finite block confidence')
    minimum, mean = torch.stack((confidence.min(), confidence.mean())).tolist()
    predicted_ids = predicted[0].tolist()
    counts = Counter(str(token) for token in predicted_ids if token in special_ids)
    return dict(confidence_min=minimum, confidence_mean=mean,
                special_top1_count=sum(counts.values()),
                probed_positions=len(predicted_ids),
                special_token_fraction=sum(counts.values()) / len(predicted_ids),
                special_token_counts=dict(counts))


def block_recurrent_logits(model, prompt_ids, mask_id, latent_slots=240, answer_slots=16,
                           block_size=16, confidence_threshold=0.9, min_block_rounds=1,
                           max_block_rounds=8, checkpointing=True, stats=None, trace=None,
                           special_token_ids=None):
    """Return final answer logits after processing all reasoning blocks in order.

    A block stops when *every* position's maximum vocabulary probability is
    strictly greater than confidence_threshold, after min_block_rounds, or when
    max_block_rounds is reached. All special-token probabilities participate in
    this comparison. special_token_ids only selects IDs to count in diagnostics.

    Inputs contain prompt + completed persistent states + the current block.
    Each block starts as MASK; every subsequent iteration uses all of its own
    previous hidden states, scaled to MASK embedding RMS. No later reasoning
    blocks or answer slots are supplied during this block's iterations. Completed
    persistent states stay fixed as inputs, but are never detached from autograd.

    After the last reasoning block, one additional backbone pass reads all saved
    states and MASK answer queries. Only this final answer head contributes to
    the supervised gradient. Probe heads and threshold decisions run in no_grad.
    Training and inference apply the same stopping rule.

    stats, when supplied, is updated with JSON-serializable per-call diagnostics.
    Forward counts and model_positions exclude activation-checkpoint recomputation
    during backward. trace, when supplied, receives one dict per block iteration:
    block_index (zero based), round_index (one based), raw_hidden, feedback_hidden,
    and stop_reason (None, 'confidence', or 'max_rounds'). Tensors in trace retain
    gradients; this diagnostic option consumes extra memory. It is NOT a sequence
    of full-canvas iterations and must not be used for cross-block round cosine.
    """
    for name, value in (('latent_slots', latent_slots), ('answer_slots', answer_slots),
                        ('block_size', block_size), ('min_block_rounds', min_block_rounds),
                        ('max_block_rounds', max_block_rounds)):
        _positive_integer(name, value)
    if latent_slots % block_size:
        raise ValueError('latent_slots must be divisible by block_size')
    if min_block_rounds > max_block_rounds:
        raise ValueError('min_block_rounds must not exceed max_block_rounds')
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError('confidence_threshold must be finite and between 0 and 1')
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError('Sequential latent blocks require one nonempty, unpadded prompt')
    if stats is not None and not isinstance(stats, dict):
        raise TypeError('stats must be a dict when supplied')
    if trace is not None and not isinstance(trace, list):
        raise TypeError('trace must be a list when supplied')

    special_ids = _special_ids(model, mask_id, special_token_ids)
    embedding = model.get_input_embeddings()
    prompt = embedding(prompt_ids)
    mask = embedding(torch.full((1, 1), mask_id, device=prompt_ids.device, dtype=torch.long))
    scale = mask.float().square().mean(-1, keepdim=True).sqrt().detach()
    use_checkpoint = checkpointing and model.training and torch.is_grad_enabled()
    completed = []
    blocks = []
    backbone_calls = confidence_calls = model_positions = 0
    final_special_counts = Counter()
    all_probe_special_count = all_probed_positions = 0

    def feedback(raw_hidden):
        normalized = raw_hidden.float() * scale.to(raw_hidden.device) / raw_hidden.float().square().mean(
            -1, keepdim=True).sqrt().clamp_min(1e-6)
        return normalized.to(device=prompt.device, dtype=prompt.dtype)

    for block_index in range(latent_slots // block_size):
        start = prompt.shape[1] + block_index * block_size
        current = mask.expand(-1, block_size, -1)
        history = []

        # Bound slice indices survive loop advancement when checkpoint recomputes
        # this function in backward. Returning a clone also releases prefix storage.
        def current_hidden_pass(inputs, slice_start=start, width=block_size):
            return backbone_hidden(model, inputs)[:, slice_start:slice_start + width].clone()

        for iteration in range(1, max_block_rounds + 1):
            inputs = torch.cat([prompt, *completed, current], dim=1)
            raw_hidden = (checkpoint(current_hidden_pass, inputs, use_reentrant=False)
                          if use_checkpoint else current_hidden_pass(inputs))
            backbone_calls += 1
            model_positions += inputs.shape[1]
            probe = _confidence_probe(model, raw_hidden, special_ids)
            confidence_calls += 1
            all_probe_special_count += probe['special_top1_count']
            all_probed_positions += probe['probed_positions']
            threshold_met = probe['confidence_min'] > confidence_threshold
            eligible = iteration >= min_block_rounds
            reason = ('confidence' if eligible and threshold_met else
                      'max_rounds' if iteration == max_block_rounds else None)
            current = feedback(raw_hidden)
            if trace is not None:
                if raw_hidden.requires_grad:
                    raw_hidden.retain_grad()
                if current.requires_grad:
                    current.retain_grad()
                trace.append(dict(block_index=block_index, round_index=iteration,
                                  raw_hidden=raw_hidden, feedback_hidden=current,
                                  stop_reason=reason))
            history.append(dict(round=iteration, threshold_met=threshold_met,
                                minimum_rounds_met=eligible, **probe))
            if reason is not None:
                completed.append(current)
                blocks.append(dict(block_index=block_index,
                                   latent_start=block_index * block_size,
                                   latent_end=(block_index + 1) * block_size,
                                   iterations=iteration, stop_reason=reason,
                                   round_history=history, **probe))
                final_special_counts.update(probe['special_token_counts'])
                break

    answer_start = prompt.shape[1] + latent_slots
    final_inputs = torch.cat([prompt, *completed, mask.expand(-1, answer_slots, -1)], dim=1)

    def answer_hidden_pass(inputs, slice_start=answer_start):
        return backbone_hidden(model, inputs)[:, slice_start:].clone()

    final_hidden = (checkpoint(answer_hidden_pass, final_inputs, use_reentrant=False)
                    if use_checkpoint else answer_hidden_pass(final_inputs))
    backbone_calls += 1
    model_positions += final_inputs.shape[1]
    logits = answer_logits(model, final_hidden)
    if stats is not None:
        forced = sum(block['stop_reason'] == 'max_rounds' for block in blocks)
        stats.update(dict(
            mode='block_confidence', blocks=blocks,
            block_rounds=[block['iterations'] for block in blocks],
            blocks_count=len(blocks), confidence_stops=len(blocks) - forced,
            forced_stops=forced, forced_stop_fraction=forced / len(blocks),
            total_block_rounds=confidence_calls,
            final_confidence_min_sum=sum(block['confidence_min'] for block in blocks),
            final_confidence_mean_sum=sum(block['confidence_mean'] for block in blocks),
            special_top1_count=all_probe_special_count, probed_positions=all_probed_positions,
            confidence_threshold=confidence_threshold,
            min_block_rounds=min_block_rounds, max_block_rounds=max_block_rounds,
            actual_backbone_calls=backbone_calls, confidence_head_calls=confidence_calls,
            answer_head_calls=1, model_positions=model_positions,
            forward_counts_exclude_backward_recomputation=True,
            special_token_ids=sorted(special_ids),
            final_special_token_counts=dict(final_special_counts),
            final_special_token_fraction=sum(final_special_counts.values()) / latent_slots,
            full_vocabulary_confidence=True, confidence_temperature=1.0,
            completed_persistent_states_fixed=True,
            prefix_activations_recomputed=True, future_blocks_visible=False,
            answer_queries_visible_during_reasoning=False,
            final_answer_backbone_calls=1, threshold_decision_has_gradient=False))
    return logits
