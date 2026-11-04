"""Head-free LLaDA execution for this unpadded, single-device LoRA experiment.

Uses the loaded checkpoint's original block/norm modules (including injected LoRA).
No monkey-patching, changed state_dict keys, cached KV, or vocabulary projections.
Deliberately rejects architectures/modes outside the tested path.
"""
import math

import torch.nn.functional as F


def native_llada(model):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if getattr(base.config, 'model_type', None) != 'llada':
        raise ValueError('Head-free path supports LLaDA only')
    return base


def backbone_hidden(model, inputs_embeds):
    """Return post-final-norm hidden states; all supplied positions are visible.

    This is the no-mask/no-cache/block_group_size=1 branch of native LLaDA forward.
    Do not use this helper for padded batches, FSDP, or automatic device offloading.
    Outer non-reentrant checkpointing is applied by the recurrence caller.
    """
    core = native_llada(model).model
    config = core.config
    if config.alibi or not config.rope or config.block_group_size != 1:
        raise ValueError('Expected RoPE, no ALiBi, and ungrouped LLaDA blocks')
    if core.activation_checkpointing_strategy is not None:
        raise ValueError('Use outer recurrence checkpointing, not internal checkpointing')
    x = inputs_embeds
    if config.input_emb_norm:
        x = x * math.sqrt(config.d_model)
    x = core.transformer.emb_drop(x)
    for block in core.transformer.blocks:
        x, _ = block(x, attention_bias=None, layer_past=None, use_cache=False)
    return core.transformer.ln_f(x)


def answer_logits(model, answer_hidden):
    """Project only answer slots, with native tying/scaling semantics preserved."""
    core = native_llada(model).model
    if core.config.weight_tying:
        logits = F.linear(answer_hidden, core.transformer.wte.weight)
    else:
        logits = core.transformer.ff_out(answer_hidden)
    if core.config.scale_logits:
        logits = logits * (1 / math.sqrt(core.config.d_model))
    return logits
