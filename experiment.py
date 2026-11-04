"""Minimal continuous-state LLaDA experiment; no token decoding in latent slots."""
import argparse
import json
import random
import time
import platform
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, PeftConfig, get_peft_model
from llada_backbone import answer_logits, backbone_hidden


def add_lora(model, rank=8):
    # LLaDA sequential and llama block variants; fail if architecture differs.
    targets = sorted({name.rsplit('.', 1)[-1] for name, module in model.named_modules()
                      if isinstance(module, torch.nn.Linear)
                      and name.rsplit('.', 1)[-1] in
                      {'att_proj', 'q_proj', 'k_proj', 'v_proj', 'attn_out',
                       'ff_proj', 'up_proj', 'ff_out'}
                      and '.blocks.' in name})
    if not targets:
        raise ValueError('No supported LLaDA projection modules found')
    # Explicit full paths avoid adapting the vocabulary output head (also ff_out).
    names = [name for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and '.blocks.' in name
             and name.rsplit('.', 1)[-1] in targets]
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
                                          target_modules=names, lora_dropout=0.0))


def load_fp32_adapter(model, directory, adapter_name='roundtrip'):
    """Keep saved FP32 LoRA weights exact when adding an adapter to a BF16 base.

    PEFT 0.15.2 load_adapter casts new adapter parameters to FP32 *after* copying
    weights. Pre-create FP32 destinations to avoid a BF16 round-trip on load.
    """
    model.add_adapter(adapter_name, PeftConfig.from_pretrained(directory))
    found = 0
    for module in model.modules():
        for attr in ('lora_A', 'lora_B'):
            adapters = getattr(module, attr, None)
            if adapters is not None and adapter_name in adapters:
                adapters[adapter_name].float()
                found += 1
    if not found:
        raise ValueError('No LoRA A/B destinations found')
    loaded = model.load_adapter(directory, adapter_name=adapter_name)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f'Adapter load was incomplete: {loaded}')
    model.set_adapter(adapter_name)
    return model


def recurrent_logits(model, prompt_ids, mask_id, latent_slots=16, answer_slots=16,
                     rounds=2, checkpointing=True, trace=None, readout='final_only'):
    """Labels deliberately excluded. All answer inputs remain MASK every round.

    RMS-rescale final hidden to MASK embedding magnitude before the next pass.
    This fixed interface is an experimental choice, not a proven optimal mapping.
    """
    if min(latent_slots, answer_slots, rounds) < 1:
        raise ValueError('slots and rounds must be positive')
    if readout not in ('final_only', 'all_logits'):
        raise ValueError('readout must be final_only or all_logits')
    embedding = model.get_input_embeddings()
    prompt = embedding(prompt_ids)
    mask = embedding(torch.full((prompt_ids.shape[0], 1), mask_id,
                                device=prompt_ids.device, dtype=torch.long))
    latent = mask.expand(-1, latent_slots, -1)
    answers = mask.expand(-1, answer_slots, -1)
    start = prompt.shape[1]
    scale = mask.float().square().mean(-1, keepdim=True).sqrt().detach()

    def forward_pass(x):
        out = model(inputs_embeds=x, use_cache=False,
                    output_hidden_states=True, return_dict=True)
        # Clone slices so outputs don't retain full-sequence logit storage.
        return (out.logits[:, start + latent_slots:].clone(),
                out.hidden_states[-1][:, start:start + latent_slots].clone())

    def hidden_pass(x):
        return backbone_hidden(model, x)

    for step in range(rounds):
        x = torch.cat((prompt, latent, answers), dim=1)
        use_checkpoint = checkpointing and model.training and torch.is_grad_enabled()
        if readout == 'final_only':
            full_hidden = (checkpoint(hidden_pass, x, use_reentrant=False)
                           if use_checkpoint else hidden_pass(x))
            hidden = full_hidden[:, start:start + latent_slots]
            if step == rounds - 1:
                logits = answer_logits(model, full_hidden[:, start + latent_slots:])
        else:
            logits, hidden = (checkpoint(forward_pass, x, use_reentrant=False)
                              if use_checkpoint else forward_pass(x))
        if trace is not None:
            trace.append(hidden)
            if hidden.requires_grad:
                hidden.retain_grad()
        if step + 1 < rounds:
            normalized = hidden.float() * scale / hidden.float().square().mean(
                -1, keepdim=True).sqrt().clamp_min(1e-6)
            latent = normalized.to(prompt.dtype)  # No detach; backprop through rounds.
    return logits


def answer_loss(logits, labels):
    # Masked LLM predicts the same position; no autoregressive label shift.
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100)


def prepare(args):
    from datasets import Dataset, load_dataset
    if args.arrow:
        dataset = Dataset.from_file(args.arrow)
        source = str(Path(args.arrow).resolve())
    else:
        dataset = load_dataset('openai/gsm8k', 'main', split='train')
        source = 'openai/gsm8k:main:train'
    unique = {}
    for i, row in enumerate(dataset):
        question = row['question'].strip()
        if '####' not in row['answer']:
            raise ValueError(f'Missing final answer at source row {i}')
        unique.setdefault(question, dict(source_index=i, question=question,
                                        answer=row['answer'].rsplit('####', 1)[1].strip()))
    rows = list(unique.values())
    random.Random(args.seed).shuffle(rows)
    if len(rows) < 1128:
        raise ValueError('Need 1128 unique training questions')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    splits = {'val128': rows[:128], 'train64': rows[128:192],
              'train1000': rows[128:1128]}
    for name, subset in splits.items():
        (root / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in subset))
    (root / 'manifest.json').write_text(json.dumps(
        dict(source=source, seed=args.seed, counts={k: len(v) for k, v in splits.items()},
             note='Validation split from official train; official test untouched.'), indent=2))
    print(json.dumps({k: len(v) for k, v in splits.items()}))


def encode_rows(path, tokenizer, device, max_prompt, answer_slots, limit=None):
    accepted, skipped = [], 0
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        text = row['question'] + '\nReturn only the final numerical answer.'
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                              tokenize=True, add_generation_prompt=True)
        answer = tokenizer.encode(row['answer'], add_special_tokens=False)
        answer.append(tokenizer.eos_token_id)
        if len(prompt) > max_prompt or len(answer) > answer_slots:
            skipped += 1
            continue
        labels = answer + [-100] * (answer_slots - len(answer))
        accepted.append((torch.tensor([prompt], device=device),
                         torch.tensor([labels], device=device), row['answer']))
        if limit is not None and len(accepted) >= limit:
            break
    if not accepted:
        raise ValueError(f'No examples fit limits: {path}')
    print(json.dumps(dict(file=str(path), accepted=len(accepted), skipped=skipped)), flush=True)
    return accepted


def train(args):
    from transformers import AutoModel, AutoTokenizer
    import transformers
    import peft
    if min(args.steps, args.accumulation, args.rank, args.max_prompt,
           args.latent_slots, args.answer_slots, args.rounds, args.cpu_threads) < 1:
        raise ValueError('Steps, accumulation, rank, lengths, rounds and threads must be positive')
    if args.lr <= 0 or args.min_free_gib < 0:
        raise ValueError('lr must be positive and min-free-gib nonnegative')
    if args.max_memory_gib is not None and args.max_memory_gib <= 0:
        raise ValueError('max-memory-gib must be positive')
    if any(v is not None and v < 1 for v in (args.train_limit, args.eval_limit)):
        raise ValueError('Sample limits must be positive')
    for name in ('model', 'data', 'validation'):
        setattr(args, name, str(Path(getattr(args, name)).resolve(strict=True)))
    args.output = str(Path(args.output).resolve())
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU training requested but CUDA is unavailable')
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info()
        if free < args.min_free_gib * 2**30:
            raise RuntimeError(f'Only {free / 2**30:.1f} GiB free; not loading model')
        if args.max_memory_gib is not None:
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_memory_gib * 2**30 / total), device)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                             local_files_only=True)
    base, loading = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
        output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint did not load exactly: {loading}')
    if getattr(base.config, 'model_type', None) != 'llada':
        raise ValueError('This first experiment supports LLaDA only')
    mask_id = getattr(base.config, 'mask_token_id', None)
    if mask_id is None:
        mask_id = tokenizer.mask_token_id
    if mask_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Checkpoint must define MASK and EOS IDs')
    if not 0 <= mask_id < base.get_input_embeddings().num_embeddings:
        raise ValueError('MASK ID is outside embedding vocabulary')
    if args.max_prompt + args.latent_slots + args.answer_slots > base.config.max_sequence_length:
        raise ValueError('Configured total sequence exceeds model context')
    model = add_lora(base, args.rank).to(device)
    model.print_trainable_parameters()
    training = encode_rows(args.data, tokenizer, device, args.max_prompt,
                           args.answer_slots, args.train_limit)
    validation = encode_rows(args.validation, tokenizer, device, args.max_prompt,
                             args.answer_slots, args.eval_limit)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    settings = dict(mask_id=mask_id, latent_slots=args.latent_slots,
                    answer_slots=args.answer_slots, rounds=args.rounds, readout=args.readout)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, peft=peft.__version__,
        device=str(device), dtype=args.dtype,
        mask_id=mask_id, eos_id=tokenizer.eos_token_id,
        train_count=len(training), validation_count=len(validation),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        loading=loading), indent=2))

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct, losses = 0, []
        for prompt, labels, answer in validation:
            logits = recurrent_logits(model, prompt, **settings)
            losses.append(answer_loss(logits, labels).item())
            ids = logits.argmax(-1)[0].tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id)]
            prediction = tokenizer.decode(ids, skip_special_tokens=False).strip()
            correct += prediction.replace(',', '') == answer.replace(',', '')
        return dict(val_ce=sum(losses) / len(losses), val_answer_em=correct / len(losses))

    print(json.dumps(dict(stage='initial_validation')), flush=True)
    initial = evaluate()
    print(json.dumps(dict(step=0, **initial)), flush=True)
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (output / 'metrics.jsonl').open('w') as log:
        log.write(json.dumps(dict(step=0, **initial)) + '\n')
        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(args.accumulation):
                prompt, labels, _ = rng.choice(training)
                loss = answer_loss(recurrent_logits(model, prompt, **settings), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                (loss / args.accumulation).backward()
                total_loss += loss.item() / args.accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                       error_if_nonfinite=True)
            optimizer.step()
            synchronize()
            metrics = dict(step=step, train_ce=total_loss,
                           grad_norm=float(grad_norm),
                           elapsed_train_s=time.monotonic() - started,
                           peak_allocated_gib=(torch.cuda.max_memory_allocated() / 2**30
                                               if device.type == 'cuda' else None))
            log.write(json.dumps(metrics) + '\n')
            log.flush()
            print(json.dumps(metrics), flush=True)
        final = evaluate()
        log.write(json.dumps(dict(step=args.steps, **final)) + '\n')
    model.save_pretrained(output / 'adapter')
    tokenizer.save_pretrained(output / 'adapter')
    print(json.dumps(final), flush=True)
    if args.verify_reload:
        prompt = validation[0][0]
        model.eval()
        with torch.no_grad():
            before = recurrent_logits(model, prompt, **settings)
        # Reload under a fresh adapter name and switch the active adapter; avoids
        # a second full base allocation and stale PEFT metadata after unload().
        load_fp32_adapter(model, output / 'adapter')
        model.eval()
        parameters = dict(model.named_parameters())
        for name, value in parameters.items():
            if '.default.' in name and 'lora_' in name:
                torch.testing.assert_close(value, parameters[name.replace('.default.', '.roundtrip.')],
                                           rtol=0, atol=0)
        with torch.no_grad():
            after = recurrent_logits(model, prompt, **settings)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        (output / 'reload_check.json').write_text(json.dumps(
            dict(status='passed', exact_logits=True, exact_adapter_weights=True)))
        print(json.dumps(dict(adapter_reload='exact')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--arrow', help='Optional cached GSM8K train Arrow file; default downloads official train')
    prep.add_argument('--output', default='data')
    prep.add_argument('--seed', type=int, default=42)
    run = subs.add_parser('train')
    for name in ('model', 'data', 'validation', 'output'):
        run.add_argument('--' + name, required=True)
    for name, default in [('steps', 100), ('accumulation', 4), ('latent-slots', 16),
                          ('answer-slots', 16), ('rounds', 2), ('rank', 8),
                          ('max-prompt', 256), ('seed', 42)]:
        run.add_argument('--' + name, type=int, default=default)
    run.add_argument('--device', default='cuda:0', help='cpu for integration debugging')
    run.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    run.add_argument('--cpu-threads', type=int, default=4)
    run.add_argument('--train-limit', type=int)
    run.add_argument('--eval-limit', type=int)
    run.add_argument('--verify-reload', action='store_true')
    run.add_argument('--readout', choices=('final_only', 'all_logits'), default='final_only',
                     help='final_only skips vocabulary projection until the last answer slots; '
                          'all_logits retains the old full-LM reference for parity checks')
    run.add_argument('--lr', type=float, default=1e-4)
    run.add_argument('--min-free-gib', type=float, default=40,
                     help='Conservative launch guard, not a measured memory requirement')
    run.add_argument('--max-memory-gib', type=float,
                     help='Optional PyTorch allocator cap; excludes CUDA context/external allocations')
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
