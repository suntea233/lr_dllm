"""Minimal continuous-state LLaDA experiment; no token decoding in latent slots."""
import argparse
import json
import math
import os
import re
import shutil
import uuid
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


def encode_rows(path, tokenizer, device, max_prompt, answer_slots, limit=None, strict=False):
    accepted, skipped = [], 0
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        text = row['question'] + '\nReturn only the final numerical answer.'
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                              tokenize=True, add_generation_prompt=True)
        answer = tokenizer.encode(row['answer'], add_special_tokens=False)
        answer.append(tokenizer.eos_token_id)
        if len(prompt) > max_prompt or len(answer) > answer_slots:
            if strict:
                raise ValueError(f'Example exceeds configured lengths in {path}: '
                                 f'prompt={len(prompt)}/{max_prompt}, '
                                 f'answer_with_eos={len(answer)}/{answer_slots}; '
                                 'strict-lengths forbids dropping examples')
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


def training_groups(count, accumulation, seed, *, steps=None, epochs=None):
    """Yield (epoch, examples_seen_in_epoch, indices) for each optimizer update."""
    if count < 1 or accumulation < 1 or (steps is None) == (epochs is None):
        raise ValueError('Choose a positive sample count/accumulation and exactly one budget')
    rng = random.Random(seed)
    if epochs is not None:
        if epochs < 1:
            raise ValueError('epochs must be positive')
        for epoch in range(1, epochs + 1):
            indices = list(range(count))
            rng.shuffle(indices)
            for start in range(0, count, accumulation):
                group = indices[start:start + accumulation]
                yield epoch, start + len(group), group
    else:
        if steps < 1:
            raise ValueError('steps must be positive')
        for _ in range(steps):
            yield None, None, [rng.randrange(count) for _ in range(accumulation)]


def check_evaluation_split(training_path, evaluation_path, role='validation'):
    """Check source provenance and question disjointness before loading the model."""
    if role not in ('validation', 'test'):
        raise ValueError('validation-role must be validation or test')
    training_rows = [json.loads(line) for line in Path(training_path).read_text().splitlines()]
    evaluation_rows = [json.loads(line) for line in Path(evaluation_path).read_text().splitlines()]
    if any(str(row.get('source_split', '')).lower() == 'test' for row in training_rows):
        raise ValueError('Training data contains rows from an official test split')
    overlap = {row['question'].strip() for row in training_rows}.intersection(
        row['question'].strip() for row in evaluation_rows)
    if overlap:
        raise ValueError(f'Training/evaluation question overlap: {len(overlap)}; '
                         'choose an independent validation/test split')
    test_name = any(re.fullmatch(r'test[0-9]*', part)
                    for part in re.split(r'[_\-.]', Path(evaluation_path).stem.lower()))
    test_provenance = any(str(row.get('source_split', '')).lower() == 'test'
                          for row in evaluation_rows)
    if role == 'validation' and (test_name or test_provenance):
        raise ValueError('Official/test-named data cannot select checkpoints. '
                         'Use an independent validation split, or --validation-role test '
                         'for final-only test evaluation without checkpoint selection')
    return dict(role=role, train_file_rows=len(training_rows),
                evaluation_file_rows=len(evaluation_rows), question_overlap=0,
                official_test_provenance=test_provenance,
                note='Whole-file disjointness checked before optional sample limits')


def save_adapter_checkpoint(model, tokenizer, directory):
    """Replace one bounded LoRA checkpoint without copying base model weights."""
    directory = Path(directory)
    staging = directory.with_name('.' + directory.name + '.pending-' + uuid.uuid4().hex)
    previous = directory.with_name('.' + directory.name + '.previous')
    try:
        model.save_pretrained(staging, save_embedding_layers=False)
        tokenizer.save_pretrained(staging)
        if previous.exists():
            if directory.exists():
                shutil.rmtree(previous)
            else:
                previous.rename(directory)
        if directory.exists():
            directory.rename(previous)
        staging.rename(directory)
        if previous.exists():
            shutil.rmtree(previous)
    except BaseException:
        if previous.exists() and not directory.exists():
            previous.rename(directory)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


class BestCheckpoint:
    """Select trained adapters by example-mean validation CE; ties keep the earlier one."""
    metric_definition = ('Mean of per-example answer-token cross-entropies; '
                         'includes EOS, ignores labels equal to -100')

    def __init__(self, output, validation_path, validation_count):
        self.output = Path(output)
        self.validation_path = str(validation_path)
        self.validation_count = validation_count
        self.best = None

    def consider(self, model, tokenizer, metrics):
        value = float(metrics['val_ce'])
        if metrics['step'] <= 0 or not math.isfinite(value):
            return False
        if self.best is not None and value >= self.best['value']:
            return False
        candidate = dict(metric='val_ce', mode='min', metric_definition=self.metric_definition,
                         value=value, step=metrics['step'], epoch=metrics.get('epoch'),
                         validation=self.validation_path, validation_count=self.validation_count,
                         adapter='best_adapter', predictions_file=metrics['predictions_file'],
                         metrics=metrics)
        save_adapter_checkpoint(model, tokenizer, self.output / 'best_adapter')
        write_json_atomic(self.output / 'best_checkpoint.json', candidate)
        self.best = candidate
        return True


@torch.no_grad()
def evaluate_validation(model, validation, tokenizer, settings, predictions_path, synchronize):
    """One forward per example; diagnostics use the already-computed answer logits."""
    was_training = model.training
    synchronize()
    started = time.monotonic()
    try:
        model.eval()
        correct = eos_count = token_correct = token_count = 0
        loss_sum = nll_sum = confidence_sum = entropy_sum = 0.0
        latent_rms_sum = latent_cosine_sum = latent_change_sum = 0.0
        latent_pairs = 0
        with Path(predictions_path).open('w') as predictions:
            for example_index, (prompt, labels, answer) in enumerate(validation):
                trace = []
                logits = recurrent_logits(model, prompt, trace=trace, **settings)
                latent_rms = trace[-1].float().square().mean().sqrt().item()
                latent_rms_sum += latent_rms
                round_cosines, round_changes = [], []
                for previous, current in zip(trace, trace[1:]):
                    previous, current = previous.float(), current.float()
                    round_cosines.append(F.cosine_similarity(previous, current, dim=-1).mean().item())
                    round_changes.append(((current - previous).norm(dim=-1) /
                                          previous.norm(dim=-1).clamp_min(1e-6)).mean().item())
                latent_cosine_sum += sum(round_cosines)
                latent_change_sum += sum(round_changes)
                latent_pairs += len(round_cosines)
                del trace
                valid = labels != -100
                log_probs = F.log_softmax(logits.float()[valid], dim=-1)
                target = labels[valid]
                nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                ce = nll.mean().item()
                if not math.isfinite(ce):
                    raise FloatingPointError(f'Non-finite validation loss at example {example_index}')
                probabilities = log_probs.exp()
                confidence = probabilities.max(-1).values
                entropy = -(probabilities * log_probs).sum(-1)
                token_count += target.numel()
                token_correct += (log_probs.argmax(-1) == target).sum().item()
                loss_sum += ce
                nll_sum += nll.sum().item()
                confidence_sum += confidence.sum().item()
                entropy_sum += entropy.sum().item()
                ids = logits.argmax(-1)[0].tolist()
                eos_present = tokenizer.eos_token_id in ids
                eos_position = ids.index(tokenizer.eos_token_id) if eos_present else None
                decoded_ids = ids[:eos_position] if eos_present else ids
                prediction = tokenizer.decode(decoded_ids, skip_special_tokens=False).strip()
                exact_match = prediction.replace(',', '') == answer.replace(',', '')
                correct += exact_match
                eos_count += eos_present
                predictions.write(json.dumps(dict(
                    example_index=example_index, gold=answer, prediction=prediction,
                    exact_match=exact_match, ce=ce, target_tokens=target.numel(),
                    mean_confidence=confidence.mean().item(), mean_entropy=entropy.mean().item(),
                    latent_rms=latent_rms, latent_round_cosines=round_cosines,
                    latent_round_relative_changes=round_changes,
                    eos_present=eos_present, eos_position=eos_position,
                    predicted_token_ids=ids)) + '\n')
        synchronize()
        elapsed = time.monotonic() - started
        return dict(val_ce=loss_sum / len(validation), val_token_ce=nll_sum / token_count,
                    val_token_accuracy=token_correct / token_count,
                    val_answer_em=correct / len(validation), val_eos_rate=eos_count / len(validation),
                    val_mean_confidence=confidence_sum / token_count,
                    val_mean_entropy=entropy_sum / token_count,
                    val_target_tokens=token_count, val_examples=len(validation),
                    val_latent_rms=latent_rms_sum / len(validation),
                    val_latent_round_cosine=latent_cosine_sum / latent_pairs if latent_pairs else None,
                    val_latent_relative_change=latent_change_sum / latent_pairs if latent_pairs else None,
                    eval_s=elapsed, eval_samples_per_s=len(validation) / max(elapsed, 1e-9))
    finally:
        model.train(was_training)


def train(args):
    from transformers import AutoModel, AutoTokenizer
    import transformers
    import peft
    from telemetry import TrainingTelemetry, ensure_tensorboard_available
    if args.epochs is not None and args.steps is not None:
        raise ValueError('Choose either epochs or steps')
    if args.epochs is None and args.steps is None:
        args.steps = 100
    if any(v is not None and v < 1 for v in (args.steps, args.epochs)):
        raise ValueError('Steps and epochs must be positive')
    if min(args.accumulation, args.rank, args.max_prompt,
           args.latent_slots, args.answer_slots, args.rounds, args.cpu_threads) < 1:
        raise ValueError('Steps, accumulation, rank, lengths, rounds and threads must be positive')
    if args.lr <= 0 or args.min_free_gib < 0:
        raise ValueError('lr must be positive and min-free-gib nonnegative')
    if args.eval_every < 0:
        raise ValueError('eval-every must be nonnegative')
    if args.epochs is not None and args.eval_every:
        raise ValueError('Epoch training evaluates only at epoch boundaries; omit --eval-every')
    if args.validation_role == 'test' and (args.initial_eval or args.eval_every):
        raise ValueError('Test evaluation is final-only: omit --initial-eval and --eval-every')
    if args.max_memory_gib is not None and args.max_memory_gib <= 0:
        raise ValueError('max-memory-gib must be positive')
    if any(v is not None and v < 1 for v in (args.train_limit, args.eval_limit)):
        raise ValueError('Sample limits must be positive')
    for name in ('model', 'data', 'validation'):
        setattr(args, name, str(Path(getattr(args, name)).resolve(strict=True)))
    args.output = str(Path(args.output).resolve())
    split_audit = check_evaluation_split(args.data, args.validation, args.validation_role)
    if not args.no_tensorboard:
        ensure_tensorboard_available()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
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
                           args.answer_slots, args.train_limit, strict=args.strict_lengths)
    validation = encode_rows(args.validation, tokenizer, device, args.max_prompt,
                             args.answer_slots, args.eval_limit, strict=args.strict_lengths)
    if args.epochs is not None:
        args.steps = args.epochs * ((len(training) + args.accumulation - 1) // args.accumulation)
    args.sampling = 'shuffled_without_replacement' if args.epochs is not None else 'with_replacement'
    args.evaluation_schedule = ('final_test_only' if args.validation_role == 'test' else
                                'every_epoch' if args.epochs is not None else 'step_budget')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    settings = dict(mask_id=mask_id, latent_slots=args.latent_slots,
                    answer_slots=args.answer_slots, rounds=args.rounds, readout=args.readout)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir()
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (output / 'environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, peft=peft.__version__,
        device=str(device), dtype=args.dtype,
        mask_id=mask_id, eos_id=tokenizer.eos_token_id,
        train_count=len(training), validation_count=len(validation),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        loading=loading, evaluation_split=split_audit,
        selection_metric_definition=BestCheckpoint.metric_definition), indent=2))
    telemetry = TrainingTelemetry(output, config=vars(args), enabled=not args.no_tensorboard)
    selection = BestCheckpoint(output, args.validation, len(validation))
    wall_started = time.monotonic()

    def evaluate(step, epoch, reason):
        predictions_path = output / 'predictions' / f'step{step:06d}.jsonl'
        result = evaluate_validation(model, validation, tokenizer, settings,
                                     predictions_path, synchronize)
        if args.validation_role == 'test':
            result = {('test_' + key[4:] if key.startswith('val_') else key): value
                      for key, value in result.items()}
        return dict(step=step, epoch=epoch, eval_reason=reason,
                    wall_time=time.time(), elapsed_wall_s=time.monotonic() - wall_started,
                    predictions_file=str(predictions_path.relative_to(output)), **result)

    def save_latest(step, epoch, coverage, samples_seen):
        # Save first so a failed validation cannot discard the completed epoch.
        save_adapter_checkpoint(model, tokenizer, output / 'adapter')
        write_json_atomic(output / 'last_checkpoint.json', dict(
            adapter='adapter', step=step, epoch=epoch, samples_seen=samples_seen,
            wall_time=time.time(), validation_role=args.validation_role))
        write_json_atomic(output / 'data_coverage.json', dict(
            sampling=args.sampling, samples_seen=samples_seen,
            train_count=len(training), optimizer_steps=step,
            epochs={str(k): dict(unique_examples=len(v), complete=len(v) == len(training))
                    for k, v in coverage.items()}))

    try:
        with (output / 'metrics.jsonl').open('w') as log, \
                (output / 'sample_order.jsonl').open('w') as sample_log:
            def record(metrics):
                log.write(json.dumps(metrics, allow_nan=False) + '\n')
                log.flush()
                telemetry.log(metrics)
                print(json.dumps(metrics, allow_nan=False), flush=True)

            initial_eval = (args.validation_role == 'validation' and not args.skip_initial_eval
                            and (args.initial_eval or args.epochs is None))
            if initial_eval:
                record(evaluate(0, 0 if args.epochs is not None else None, 'initial'))
            synchronize()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            elapsed_train_s = 0.0
            samples_seen = 0
            coverage = {}
            epoch_totals = dict(loss=0.0, nll=0.0, samples=0, tokens=0, seconds=0.0)
            groups = training_groups(len(training), args.accumulation, args.seed,
                                     epochs=args.epochs,
                                     steps=args.steps if args.epochs is None else None)
            model.train()
            for step, (epoch, epoch_samples_seen, indices) in enumerate(groups, 1):
                step_started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                total_loss = total_nll = 0.0
                target_tokens = prompt_tokens = 0
                for index in indices:
                    prompt, labels, _ = training[index]
                    loss = answer_loss(recurrent_logits(model, prompt, **settings), labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss at step {step}')
                    (loss / len(indices)).backward()
                    loss_value = loss.item()
                    targets = (labels != -100).sum().item()
                    total_loss += loss_value / len(indices)
                    total_nll += loss_value * targets
                    target_tokens += targets
                    prompt_tokens += prompt.numel()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                           error_if_nonfinite=True)
                optimizer.step()
                synchronize()
                step_train_s = time.monotonic() - step_started
                elapsed_train_s += step_train_s
                samples_seen += len(indices)
                sample_log.write(json.dumps(dict(step=step, epoch=epoch, indices=indices)) + '\n')
                sample_log.flush()
                epoch_end = epoch is not None and epoch_samples_seen == len(training)
                if epoch is not None:
                    seen = coverage.setdefault(epoch, set())
                    if seen.intersection(indices):
                        raise AssertionError(f'Repeated training sample in epoch {epoch}')
                    seen.update(indices)
                    epoch_totals['loss'] += total_loss * len(indices)
                    epoch_totals['nll'] += total_nll
                    epoch_totals['samples'] += len(indices)
                    epoch_totals['tokens'] += target_tokens
                    epoch_totals['seconds'] += step_train_s
                denominator = max(step_train_s, 1e-9)
                metrics = dict(step=step, train_ce=total_loss, train_token_ce=total_nll / target_tokens,
                               lr=optimizer.param_groups[0]['lr'],
                               train_target_tokens=target_tokens, train_prompt_tokens=prompt_tokens,
                               samples_per_s=len(indices) / denominator,
                               answer_tokens_per_s=target_tokens / denominator,
                               model_positions_per_s=(prompt_tokens + len(indices) *
                                                      (args.latent_slots + args.answer_slots)) *
                                                     args.rounds / denominator,
                               samples_seen=samples_seen, microbatches=len(indices),
                               epoch=epoch, epoch_samples_seen=epoch_samples_seen,
                               grad_norm=float(grad_norm),
                               step_train_s=step_train_s, elapsed_train_s=elapsed_train_s,
                               wall_time=time.time(), elapsed_wall_s=time.monotonic() - wall_started,
                               allocated_gib=(torch.cuda.memory_allocated() / 2**30
                                              if device.type == 'cuda' else None),
                               reserved_gib=(torch.cuda.memory_reserved() / 2**30
                                             if device.type == 'cuda' else None),
                               peak_allocated_gib=(torch.cuda.max_memory_allocated() / 2**30
                                                   if device.type == 'cuda' else None))
                if epoch_end:
                    metrics.update(epoch_train_ce=epoch_totals['loss'] / epoch_totals['samples'],
                                   epoch_train_token_ce=epoch_totals['nll'] / epoch_totals['tokens'],
                                   epoch_train_s=epoch_totals['seconds'],
                                   epoch_sample_count=epoch_totals['samples'],
                                   epoch_target_tokens=epoch_totals['tokens'],
                                   epoch_samples_per_s=epoch_totals['samples'] /
                                                       max(epoch_totals['seconds'], 1e-9))
                record(metrics)
                final_step = step == args.steps
                if epoch_end or final_step:
                    save_latest(step, epoch, coverage, samples_seen)
                should_evaluate = (final_step if args.validation_role == 'test' else
                                   epoch_end if args.epochs is not None else
                                   final_step or bool(args.eval_every and step % args.eval_every == 0))
                if should_evaluate:
                    reason = 'epoch_end' if epoch_end and args.validation_role == 'validation' else (
                        'final_step' if final_step else 'step_interval')
                    validation_metrics = evaluate(step, epoch, reason)
                    if args.validation_role == 'validation':
                        improved = selection.consider(model, tokenizer, validation_metrics)
                        validation_metrics['is_best'] = improved
                        if selection.best is not None:
                            validation_metrics.update(best_val_ce=selection.best['value'],
                                                      best_step=selection.best['step'],
                                                      best_epoch=selection.best['epoch'])
                    record(validation_metrics)
                if epoch_end:
                    epoch_totals = dict(loss=0.0, nll=0.0, samples=0, tokens=0, seconds=0.0)
        if args.verify_reload:
            prompt = validation[0][0]
            model.eval()
            with torch.no_grad():
                before = recurrent_logits(model, prompt, **settings)
            # Reload into FP32 LoRA destinations; keep one shared base allocation.
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
    finally:
        telemetry.close()


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
    budget = run.add_mutually_exclusive_group()
    budget.add_argument('--steps', type=int,
                        help='Optimizer steps with sampling with replacement; defaults to 100')
    budget.add_argument('--epochs', type=int,
                        help='Complete shuffled passes without replacement; keeps the final partial update')
    for name, default in [('accumulation', 4), ('latent-slots', 16),
                          ('answer-slots', 16), ('rounds', 2), ('rank', 8),
                          ('max-prompt', 256), ('seed', 42)]:
        run.add_argument('--' + name, type=int, default=default)
    run.add_argument('--device', default='cuda:0', help='cpu for integration debugging')
    run.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    run.add_argument('--cpu-threads', type=int, default=4)
    run.add_argument('--train-limit', type=int)
    run.add_argument('--eval-limit', type=int)
    run.add_argument('--eval-every', type=int, default=0,
                     help='Step-mode only: optional evaluation interval; epoch mode evaluates every epoch')
    initial = run.add_mutually_exclusive_group()
    initial.add_argument('--initial-eval', action='store_true',
                         help='Opt into initial validation in epoch mode; diagnostic only, never selects best')
    initial.add_argument('--skip-initial-eval', action='store_true',
                         help='Suppress legacy step-mode initial validation; epoch mode skips it by default')
    run.add_argument('--validation-role', choices=('validation', 'test'), default='validation',
                     help='validation selects best per epoch; test evaluates once at the end without selection')
    run.add_argument('--no-tensorboard', action='store_true',
                     help='Disable TensorBoard events; JSONL metrics are always retained')
    run.add_argument('--strict-lengths', action='store_true',
                     help='Fail instead of dropping examples beyond prompt/answer length limits')
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
