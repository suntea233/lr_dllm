"""Prepare, but never launch, a two-GPU confidence-gated hidden-block smoke run.

Only runtime source files are copied. Original GSM8K examples and the shared
LLaDA weights remain external files referenced through symlinks. This one-update
probe is an implementation/resource check, not a full-data performance run.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from prepare_k8_campaign import (checked_model, cpu_preflight, digest,
                                launcher_module, select_gpus, write_json)


SOURCE_FILES = ('experiment.py', 'llada_backbone.py', 'telemetry.py',
                'prompt_profiles.py', 'block_reasoning.py',
                'launch_when_free.py', 'requirements.txt')


def checked_probe_inputs(data, train_probe, validation_probe):
    """Verify two disjoint two-row probes against original CoLaR train/val.

    The official test file is not opened or linked. Both CoLaR files originate
    from the original GSM8K training split, distinguished by source_index.
    """
    manifest = json.loads((data / 'manifest.json').read_text())
    holdout = manifest.get('validation_holdout', {})
    if (manifest.get('source') != 'openai/gsm8k'
            or manifest.get('configuration') != 'main'
            or manifest.get('augmented') is not False
            or manifest.get('counts') != {'train': 6726, 'val': 747, 'test': 1319}
            or holdout.get('source_split') != 'train'
            or holdout.get('policy') != 'tail'
            or holdout.get('validation_source_indices') != list(range(6726, 7473))):
        raise ValueError('Expected original GSM8K CoLaR 6726/747 tail holdout')
    full, integrity = {}, {}
    for split, count, offset in (('train', 6726, 0), ('val', 747, 6726)):
        path = (data / (split + '.jsonl')).resolve(strict=True)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        actual = dict(rows=len(rows), bytes=path.stat().st_size, sha256=digest(path))
        if actual != manifest['files'][split + '.jsonl'] or len(rows) != count:
            raise ValueError(f'Full source hash or count mismatch: {path}')
        if any(row.get('source_split') != 'train' or row.get('source_index') != offset + i
               for i, row in enumerate(rows)):
            raise ValueError(f'Unexpected original GSM8K source ordering: {path}')
        full[split] = {row['source_index']: row for row in rows}
        integrity['full_' + split] = dict(path=str(path), **actual)
    questions = {}
    for split, path in (('train', train_probe), ('val', validation_probe)):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        indices = [row.get('source_index') for row in rows]
        if len(rows) != 2 or len(set(indices)) != 2:
            raise ValueError(f'Expected exactly two distinct original examples: {path}')
        if any(row.get('source_split') != 'train'
               or row != full[split].get(row.get('source_index')) for row in rows):
            raise ValueError(f'Probe is not an exact subset of the declared {split} split: {path}')
        questions[split] = {row['question'].strip() for row in rows}
        integrity[split + '_probe'] = dict(path=str(path), rows=len(rows),
                                          source_indices=indices, source_split='train',
                                          colar_role=split, bytes=path.stat().st_size,
                                          sha256=digest(path))
    if questions['train'] & questions['val']:
        raise ValueError('Training and validation probes overlap')
    full_train_questions = {row['question'].strip() for row in full['train'].values()}
    if questions['val'] & full_train_questions:
        raise ValueError('Validation probe overlaps full training/few-shot source')
    return dict(**integrity, training_validation_question_overlap=0,
                official_test_opened=False, official_test_used=False,
                copied_dataset_bytes=0, purpose='Technical smoke run only; not a performance estimate')


def trainer_argv(args, campaign, model):
    values = {
        'model': model, 'data': campaign / 'data' / 'train_probe.jsonl',
        'validation': campaign / 'data' / 'val_probe.jsonl',
        'fewshot-data': campaign / 'data' / 'fewshot_train.jsonl',
        'output': campaign / 'train', 'validation-role': 'validation',
        'model-parallel-devices': 'cuda:0,cuda:1', 'device': 'cuda:0',
        'dtype': 'bfloat16', 'cpu-threads': 4,
        'reasoning-mode': 'block_confidence', 'confidence-threshold': 0.9,
        'min-block-rounds': 1, 'max-block-rounds': 8,
        'block-size': 16, 'latent-slots': 240, 'answer-slots': 16,
        'generation-length': 256, 'steps': 1, 'accumulation': 4,
        'rank': 8, 'lr': 1e-4, 'seed': 42,
        'prompt-profile': 'answer_only', 'eval-prompt-profile': 'llada4shot',
        'max-prompt': 256, 'max-eval-prompt': 1536,
        'eval-every': 0, 'readout': 'final_only',
        'min-free-gib': args.min_free_gib, 'max-memory-gib': args.cap_gib,
    }
    argv = [args.python, '-u', str(campaign / 'source' / 'experiment.py'), 'train']
    for flag, value in values.items():
        argv.extend(['--' + flag, str(value)])
    argv.extend(['--skip-initial-eval', '--strict-lengths', '--verify-reload'])
    return argv


def method_definition():
    return dict(
        kind='confidence_gated_sequential_hidden_blocks',
        purpose='One-update technical smoke run; no automatic full-data training follows',
        generation_length=256, latent_slots=240, answer_slots=16,
        block_size=16, reasoning_blocks=15,
        confidence_threshold=0.9, comparison='strictly greater than',
        stop_condition='Every current-block position must have max vocabulary probability > 0.9',
        confidence_input='Raw post-final-norm hidden before RMS feedback scaling',
        vocabulary='Original full vocabulary; special tokens are not excluded',
        probe_gradients=False, minimum_rounds_per_block=1, maximum_rounds_per_block=8,
        budget_exhaustion='Force advance and record budget stop separately from confidence stop',
        persistent_completed_hidden_fixed=True, completed_hidden_detached=False,
        prefix_intermediate_activations='Recomputed by the backbone; persistent saved states are not overwritten',
        attention_visibility='Prompt plus completed hidden blocks plus current block; no future blocks or answer slots',
        final_answer_readout='One additional backbone pass on prompt plus all completed hidden plus answer MASKs',
        answer_readout_passes=1, minimum_backbone_passes_per_example=16,
        maximum_backbone_passes_per_example=121,
        training_inference_policy='Same confidence/budget branching policy',
        loss='Final numerical answer plus EOS CE only; no reasoning token supervision',
        reasoning_tokens_committed=False, feedback='RMS-scaled continuous hidden states',
        checkpointing='Non-reentrant activation checkpointing; full cross-block and cross-round BPTT',
        stopping_rule_differentiable=False, official_test_used=False,
        confidence_stop_does_not_prove_reasoning_correctness=True)


def prepare(args):
    repo = args.repo_root.resolve(strict=True)
    campaign = args.campaign.absolute()
    if campaign.exists() or campaign.is_symlink():
        raise FileExistsError(f'Campaign must be a new directory: {campaign}')
    for key in ('min_free_gib', 'cap_gib'):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if args.min_free_gib < args.cap_gib:
        raise ValueError('Free-memory threshold must be at least the allocator cap')
    python = shutil.which(args.python)
    if python is None:
        raise ValueError(f'Python executable not found: {args.python}')
    args.python = os.path.abspath(python)
    model = (args.model or repo / 'checkpoints' / 'llada15-local').resolve(strict=True)
    shared = args.shared_model_root.resolve(strict=True)
    data = (args.data or repo / 'data' / 'gsm8k_colar_split').resolve(strict=True)
    diagnostics = repo / 'runs' / 'diagnostics' / 'model_parallel_20261106'
    train_probe = (args.train_probe or diagnostics / 'longest_train_probe.jsonl').resolve(strict=True)
    val_probe = (args.validation_probe or diagnostics / 'longest_fourshot_val_probe.jsonl').resolve(strict=True)
    for name in SOURCE_FILES:
        if not (repo / name).is_file():
            raise FileNotFoundError(f'Missing runtime source: {repo / name}')
    integrity = checked_probe_inputs(data, train_probe, val_probe)
    weights = checked_model(model, shared)
    launcher = launcher_module(repo)
    gpus = launcher.query_gpus()
    selected, policy = select_gpus(gpus, args.gpus)
    if any(args.cap_gib >= gpu['total_gib'] for gpu in selected):
        raise ValueError('Allocator cap must be below the total memory of every candidate')
    environment = dict(HF_MODULES_CACHE=str(repo / '.cache' / 'modules'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                       PYTHONDONTWRITEBYTECODE='1')
    argv = trainer_argv(args, campaign, model)
    spec = launcher.validate_spec(dict(
        argv=argv, cwd=str(repo), env=environment,
        candidate_gpus=[gpu['uuid'] for gpu in selected], gpu_count=2,
        min_free_gib=args.min_free_gib, poll_seconds=30,
        stable_polls=3, max_wait_seconds=86400))
    campaign.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + campaign.name + '.preparing-', dir=campaign.parent))
    try:
        source = stage / 'source'
        source.mkdir()
        hashes = {}
        for name in SOURCE_FILES:
            original = repo / name
            before = digest(original)
            shutil.copyfile(original, source / name)
            if digest(original) != before or digest(source / name) != before:
                raise RuntimeError(f'Runtime source changed while snapshotting: {name}')
            hashes[name] = dict(sha256=before, bytes=(source / name).stat().st_size)
        preflight = cpu_preflight(args.python, source, argv, environment)
        write_json(stage / 'launch_spec.json', spec)
        subprocess.run([args.python, str(source / 'launch_when_free.py'), '--spec',
                        str(stage / 'launch_spec.json'), '--output', str(campaign / 'launcher'),
                        '--validate-only'], check=True, capture_output=True, text=True,
                       timeout=30, env=dict(os.environ, **environment, CUDA_VISIBLE_DEVICES=''))
        (stage / 'data').mkdir()
        for name, target in {'train_probe.jsonl': train_probe, 'val_probe.jsonl': val_probe,
                             'fewshot_train.jsonl': data / 'train.jsonl'}.items():
            (stage / 'data' / name).symlink_to(target)
        revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=False)
        status = subprocess.run(['git', '-C', str(repo), 'status', '--short'],
                                capture_output=True, text=True, check=False)
        write_json(stage / 'source_manifest.json', dict(
            git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            git_status=status.stdout.splitlines() if status.returncode == 0 else None,
            source_root=str(repo), snapshot_root=str(campaign / 'source'), files=hashes,
            preparation_helper_sha256=digest(Path(__file__).resolve()),
            immutable_by='Read-only runtime source files and directory; exact SHA256 recorded',
            external_data_and_weights_remain_symlinks=True))
        write_json(stage / 'dataset_integrity.json', integrity)
        write_json(stage / 'shared_weights.json', weights)
        write_json(stage / 'method_definition.json', method_definition())
        write_json(stage / 'cpu_preflight.json', preflight)
        write_json(stage / 'gpu_snapshot.json', dict(
            recorded_at=datetime.now(timezone.utc).isoformat(), all_gpus=gpus,
            selected_candidates=selected, policy=policy,
            memory_reserved=False, launch_requires_fresh_stable_polls=True,
            memory_limits_are_unmeasured_for_this_new_method=True))
        launcher_argv = [args.python, '-u', str(campaign / 'source' / 'launch_when_free.py'),
                         '--spec', str(campaign / 'launch_spec.json'),
                         '--output', str(campaign / 'launcher')]
        write_json(stage / 'campaign_plan.json', dict(
            prepared_at=datetime.now(timezone.utc).isoformat(), prepared_only=True,
            gpu_job_started=False, purpose='8B technical smoke run only',
            train_output=str(campaign / 'train'), launcher_output=str(campaign / 'launcher'),
            launcher_argv=launcher_argv, training_argv=argv,
            optimizer_steps=1, gradient_accumulation=4, train_examples=2,
            validation_examples=2, initial_eval=False,
            evaluation_schedule='Once after the sole optimizer update; adapter reload checked afterwards',
            checkpoint_note='Any selected adapter is probe-only, not a full-data best checkpoint',
            official_test_used=False, full_training_automatically_started=False,
            copied_dataset_bytes=0, copied_weight_bytes=0,
            note='Preparation never launches a GPU job. Start the recorded launcher separately.'))
        for path in source.iterdir():
            path.chmod(0o444)
        source.chmod(0o555)
        campaign.mkdir()
        for path in stage.iterdir():
            path.rename(campaign / path.name)
        stage.rmdir()
    except BaseException:
        if (stage / 'source').exists():
            (stage / 'source').chmod(0o755)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return dict(campaign=str(campaign), prepared_only=True, gpu_job_started=False,
                candidate_gpus=spec['candidate_gpus'], launcher_argv=launcher_argv,
                full_training_automatically_started=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpus', nargs='+', default=['1', '3', '4'])
    parser.add_argument('--min-free-gib', type=float, default=16)
    parser.add_argument('--cap-gib', type=float, default=14)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--shared-model-root', type=Path, default=Path('/mnt/public/models/LLaDA-1.5'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--train-probe', type=Path)
    parser.add_argument('--validation-probe', type=Path)
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(prepare(parse_args()), indent=2))
