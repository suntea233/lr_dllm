"""Head-free LLaDA execution, optionally partitioned over explicit CUDA devices.

Uses the loaded checkpoint's original block/norm modules (including injected LoRA).
No monkey-patching, changed state_dict keys, cached KV, or vocabulary projections.
Manual model parallelism moves each module once and copies activations across
contiguous block partitions. It does not use inference-only device-map hooks.
Deliberately rejects architectures/modes outside the tested path.
"""
import math

import torch
import torch.nn.functional as F


def native_llada(model):
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if getattr(base.config, 'model_type', None) != 'llada':
        raise ValueError('Head-free path supports LLaDA only')
    return base


def _validate_backbone(core):
    config = core.config
    if config.alibi or not config.rope or config.block_group_size != 1:
        raise ValueError('Expected RoPE, no ALiBi, and ungrouped LLaDA blocks')
    if core.activation_checkpointing_strategy is not None:
        raise ValueError('Use outer recurrence checkpointing, not internal checkpointing')


def _module_device(module, fallback):
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else fallback


def partition_llada(model, devices):
    """Place a LLaDA in contiguous block partitions, in place, and report placement.

    Call after injecting LoRA and before constructing the optimizer. A single
    CPU or CUDA device preserves the existing single-device path. Multiple
    devices must be distinct, explicitly indexed CUDA devices; CPU offloading,
    Accelerate dispatch hooks, tied input/output embeddings and nonzero dropout
    are intentionally unsupported for multi-device recurrent training.

    Only ``backbone_hidden``/``answer_logits`` implement inter-device transfers;
    the checkpoint's unmodified full-LM ``forward`` is not model-parallel aware.
    No base weights are copied or tied weights untied. State-dict keys stay fixed.
    The returned JSON-serializable dict is metadata, not a replacement model.
    """
    if isinstance(devices, (str, torch.device)):
        devices = [devices]
    devices = [torch.device(device) for device in devices]
    devices = [torch.device('cpu') if device.type == 'cpu' else device for device in devices]
    if not devices:
        raise ValueError('At least one model device is required')
    if any(device.type not in ('cpu', 'cuda') for device in devices):
        raise ValueError('Only CPU and CUDA model devices are supported')
    if any(device.type == 'cuda' and device.index is None for device in devices):
        raise ValueError('CUDA model devices must have explicit indices, such as cuda:0')
    if len(set(devices)) != len(devices):
        raise ValueError('Model devices must be distinct')
    if len(devices) > 1 and any(device.type != 'cuda' for device in devices):
        raise ValueError('Multiple model devices must all be CUDA; CPU offload is unsupported')
    base = native_llada(model)
    core = base.model
    _validate_backbone(core)
    blocks = core.transformer.blocks
    if len(devices) > len(blocks):
        raise ValueError('Cannot allocate more model devices than Transformer blocks')
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError('Load real checkpoint tensors before manual partitioning')
    if getattr(base, 'hf_device_map', None) or any(
            getattr(module, '_hf_hook', None) is not None for module in model.modules()):
        raise ValueError('Remove Accelerate dispatch/offload hooks before manual partitioning')
    if len(devices) > 1:
        if core.config.weight_tying:
            raise ValueError('Multi-device partitioning requires untied input/output embeddings')
        stochastic = any(float(getattr(core.config, name, 0.0)) != 0.0 for name in (
            'embedding_dropout', 'attention_dropout', 'residual_dropout'))
        stochastic = stochastic or any(isinstance(module, torch.nn.Dropout) and module.p != 0
                                       for module in model.modules())
        if stochastic:
            raise ValueError('Multi-device recurrence checkpointing currently requires dropout=0')

    first, last = devices[0], devices[-1]
    core.transformer.wte.to(first)
    core.transformer.emb_drop.to(first)
    assignments = []
    for index, block in enumerate(blocks):
        device = devices[index * len(devices) // len(blocks)]
        block.to(device)
        assignments.append(str(device))
    core.transformer.ln_f.to(last)
    if not core.config.weight_tying:
        core.transformer.ff_out.to(last)

    parameter_bytes, trainable_bytes = {}, {}
    for parameter in model.parameters():
        key = str(parameter.device)
        size = parameter.numel() * parameter.element_size()
        parameter_bytes[key] = parameter_bytes.get(key, 0) + size
        if parameter.requires_grad:
            trainable_bytes[key] = trainable_bytes.get(key, 0) + size
    if set(parameter_bytes) - {str(device) for device in devices}:
        raise RuntimeError('Some parameters were not covered by the LLaDA partition')
    return dict(mode='manual_contiguous_blocks', devices=[str(d) for d in devices],
                embedding_device=str(first), final_norm_device=str(last),
                output_device=str(first if core.config.weight_tying else last),
                block_devices=assignments, parameter_bytes=parameter_bytes,
                trainable_parameter_bytes=trainable_bytes,
                cpu_offload=False, duplicates_base_weights=False)


def backbone_hidden(model, inputs_embeds):
    """Return post-final-norm hidden states; all supplied positions are visible.

    This is the no-mask/no-cache/block_group_size=1 branch of native LLaDA forward.
    Do not use this helper for padded batches, FSDP, or automatic device offloading.
    Outer non-reentrant checkpointing is applied by the recurrence caller.
    """
    core = native_llada(model).model
    config = core.config
    _validate_backbone(core)
    x = inputs_embeds.to(core.transformer.wte.weight.device)
    if config.input_emb_norm:
        x = x * math.sqrt(config.d_model)
    x = core.transformer.emb_drop(x)
    for block in core.transformer.blocks:
        # Tensor.to is differentiable. Same-device calls return the same tensor,
        # while partition boundaries copy the activation without detaching it.
        x = x.to(_module_device(block, x.device))
        x, _ = block(x, attention_bias=None, layer_past=None, use_cache=False)
    x = x.to(_module_device(core.transformer.ln_f, x.device))
    return core.transformer.ln_f(x)


def answer_logits(model, answer_hidden):
    """Project only answer slots, with native tying/scaling semantics preserved."""
    core = native_llada(model).model
    if core.config.weight_tying:
        answer_hidden = answer_hidden.to(core.transformer.wte.weight.device)
        logits = F.linear(answer_hidden, core.transformer.wte.weight)
    else:
        answer_hidden = answer_hidden.to(_module_device(core.transformer.ff_out,
                                                       answer_hidden.device))
        logits = core.transformer.ff_out(answer_hidden)
    if core.config.scale_logits:
        logits = logits * (1 / math.sqrt(core.config.d_model))
    return logits
