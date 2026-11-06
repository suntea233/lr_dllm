"""Prepare a reproducible two-GPU hidden-reasoning campaign; never start it.

Run this with the intended training environment's Python. Memory thresholds and
the audited evaluation prompt limit are deliberately required arguments. Source
files are snapshotted read-only; datasets and shared weights are never copied.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


SOURCE_FILES = ('experiment.py', 'llada_backbone.py', 'telemetry.py',
                'prompt_profiles.py', 'requirements.txt', 'launch_when_free.py')
COUNTS = {'train': 6726, 'val': 747, 'test': 1319}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_data(directory):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('source') != 'openai/gsm8k'
            or manifest.get('configuration') != 'main'
            or manifest.get('augmented') is not False
            or manifest.get('counts') != COUNTS):
        raise ValueError('Expected original GSM8K main, CoLaR 6726/747, and 1319 official test')
    holdout = manifest.get('validation_holdout', {})
    if (holdout.get('source_split') != 'train' or holdout.get('policy') != 'tail'
            or holdout.get('validation_source_indices') != list(range(6726, 7473))):
        raise ValueError('Validation must be the final 747 original training rows')
    integrity, questions = {}, {}
    for split, count in COUNTS.items():
        name = split + '.jsonl'
        path = (directory / name).resolve(strict=True)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        expected = manifest['files'][name]
        actual = dict(rows=len(records), bytes=path.stat().st_size, sha256=digest(path))
        if actual != expected or len(records) != count:
            raise ValueError(f'Dataset hash, size, or row count mismatch: {path}')
        offset = 6726 if split == 'val' else 0
        source_split = 'test' if split == 'test' else 'train'
        if any(row.get('source_index') != offset + i or row.get('source_split') != source_split
               for i, row in enumerate(records)):
            raise ValueError(f'Unexpected source order or split: {path}')
        questions[split] = {row['question'].strip() for row in records}
        integrity[name] = dict(path=str(path), **actual)
    if questions['train'] & questions['val']:
        raise ValueError('Training and validation questions overlap')
    return manifest, dict(manifest_sha256=digest(manifest_path), files=integrity,
                          copied_dataset_bytes=0, training_validation_question_overlap=0,
                          official_test_used_by_training_or_selection=False)


def checked_model(directory, shared_root):
    index_path = directory / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index['weight_map'].values()))
    if not shard_names:
        raise ValueError('Model weight index has no shards')
    shards = []
    for name in shard_names:
        if Path(name).name != name:
            raise ValueError(f'Unexpected weight shard filename: {name}')
        path = directory / name
        if not path.is_symlink():
            raise ValueError(f'Weight shards must reuse shared storage through symlinks: {path}')
        target = path.resolve(strict=True)
        if not target.is_file() or not target.is_relative_to(shared_root):
            raise ValueError(f'Weight target is outside shared model storage: {target}')
        blob_digest = target.name if re.fullmatch(r'[0-9a-f]{64}', target.name) else None
        shards.append(dict(name=name, symlink=str(path), link_value=os.readlink(path),
                           target=str(target), bytes=target.stat().st_size,
                           expected_sha256_from_blob_name=blob_digest,
                           content_hash_recomputed=False))
    metadata = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in shard_names:
            metadata[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return dict(model_directory=str(directory), shared_root=str(shared_root), shards=shards,
                metadata=metadata, total_weight_bytes=sum(item['bytes'] for item in shards),
                copied_weight_bytes=0,
                verification='Shard symlink targets and sizes checked; small metadata SHA256 recomputed. '
                             'Shard SHA256 values, when present, come from blob filenames; '
                             'weight contents are not rehashed.')


def launcher_module(repo):
    spec = importlib.util.spec_from_file_location('campaign_launcher', repo / 'launch_when_free.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_gpus(gpus, requested):
    if requested is None:
        selected = sorted(gpus, key=lambda gpu: (-gpu['free_gib'], gpu['index']))[:2]
        policy = 'Two GPUs with greatest free memory at preparation time; UUIDs pinned in launch spec'
    else:
        selected = []
        for candidate in requested:
            matches = [gpu for gpu in gpus
                       if str(gpu['index']) == str(candidate) or gpu['uuid'] == str(candidate)]
            if len(matches) != 1:
                raise ValueError(f'GPU candidate not found uniquely: {candidate}')
            if matches[0]['uuid'] not in {gpu['uuid'] for gpu in selected}:
                selected.append(matches[0])
        policy = 'Explicit candidate pool; launcher chooses two qualifying GPUs by free memory'
    if len(selected) < 2:
        raise ValueError('This campaign needs at least two distinct visible GPU candidates')
    return sorted(selected, key=lambda gpu: gpu['index']), policy


def trainer_argv(args, campaign, model, data):
    values = {
        'model': model, 'data': data / 'train.jsonl', 'validation': data / 'val.jsonl',
        'output': campaign / 'train', 'validation-role': 'validation',
        'model-parallel-devices': 'cuda:0,cuda:1', 'device': 'cuda:0',
        'dtype': 'bfloat16', 'cpu-threads': args.cpu_threads,
        'rounds': args.rounds, 'latent-slots': args.latent_slots,
        'answer-slots': args.answer_slots, 'block-size': args.block_size,
        'generation-length': args.latent_slots + args.answer_slots,
        'prompt-profile': args.prompt_profile, 'eval-prompt-profile': args.eval_prompt_profile,
        'max-prompt': args.max_prompt, 'max-eval-prompt': args.max_eval_prompt,
        'epochs': args.epochs, 'rank': args.rank, 'lr': args.lr,
        'accumulation': args.accumulation, 'seed': args.seed, 'eval-every': 0,
        'readout': 'final_only', 'min-free-gib': args.min_free_gib,
        'max-memory-gib': args.cap_gib,
    }
    argv = [args.python, '-u', str(campaign / 'source' / 'experiment.py'), 'train']
    for flag, value in values.items():
        argv.extend(['--' + flag, str(value)])
    argv.extend(['--skip-initial-eval', '--strict-lengths', '--verify-reload'])
    return argv


def cpu_preflight(python, snapshot, argv, environment):
    # --help exits in argparse before model/data loading or GPU initialization.
    command = [python, str(snapshot / 'experiment.py'), 'train', '--help']
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            timeout=120, cwd=snapshot,
                            env=dict(os.environ, **environment, CUDA_VISIBLE_DEVICES=''))
    required = {arg for arg in argv if arg.startswith('--')}
    available = set(re.findall(r'--[a-z][a-z0-9-]*', result.stdout))
    missing = sorted(required - available)
    if missing:
        raise ValueError(f'Trainer CLI is not ready for this campaign; missing flags: {missing}')
    return dict(command=command, cuda_visible_devices='', exit_code=result.returncode,
                checked_cli_flags=sorted(required), gpu_work_performed=False,
                help_sha256=hashlib.sha256(result.stdout.encode()).hexdigest())


def method_definition(args):
    blocks = math.ceil(args.latent_slots / args.block_size)
    active_slots = [0] + [args.latent_slots] * (args.rounds - 1)
    mask_slots = [args.latent_slots + args.answer_slots] + [args.answer_slots] * (args.rounds - 1)
    return dict(
        kind='continuous_hidden_reasoning_all_positions',
        native_llada_sampler=False, rounds=args.rounds,
        generation_length=args.latent_slots + args.answer_slots,
        latent_slots=args.latent_slots, answer_slots=args.answer_slots,
        block_size=args.block_size, reasoning_blocks=blocks,
        active_hidden_slots_per_round=active_slots,
        mask_input_slots_per_round=mask_slots,
        hidden_feedback='From round 2 onward, ALL latent positions receive hidden states '
                        'from the preceding round and are updated on every subsequent round.',
        reasoning_tokens_decoded=False, answer_readout='Final iteration answer slots only',
        committed_or_frozen_reasoning_positions=False,
        answer_input='MASK at every iteration; labels never enter recurrence',
        block_semantics='Layout and official-setting reference metadata only; block size '
                        'does not control hidden feedback or the update schedule.',
        masking='Round 1: all latent and answer slots are MASK. Later rounds: all latent '
                'slots receive hidden feedback and only answer slots remain MASK inputs.',
        training_prompt_profile=args.prompt_profile,
        validation_prompt_profile=args.eval_prompt_profile,
        loss='Answer cross entropy only; no explicit reasoning-token targets',
        checkpoint_selection='Minimum validation CE after each completed epoch',
        initial_evaluation=False, official_test_used=False,
        comparability='Official generation length and evaluation shot count are reference settings; '
                      'block width is recorded only as layout metadata. '
                      'K global hidden passes are not native blockwise sampling steps; '
                      'this is not a reproduction of official LLaDA-1.5 accuracy.')


def prepare(args):
    repo = args.repo_root.resolve(strict=True)
    campaign = args.campaign.absolute()
    if campaign.exists() or campaign.is_symlink():
        raise FileExistsError(f'Campaign must be a new path: {campaign}')
    for name in ('rounds', 'latent_slots', 'answer_slots', 'block_size', 'max_prompt',
                 'max_eval_prompt', 'epochs', 'rank', 'accumulation', 'cpu_threads'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('lr', 'min_free_gib', 'cap_gib'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if args.latent_slots % args.block_size or args.answer_slots % args.block_size:
        raise ValueError('Layout requires latent and answer slots divisible by block-size')
    if args.min_free_gib < args.cap_gib:
        raise ValueError('The free-memory launch threshold must be at least the allocator cap')
    python = shutil.which(args.python)
    if python is None:
        raise ValueError(f'Python executable not found: {args.python}')
    # Do not resolve Python symlinks: a virtual environment may depend on argv[0].
    args.python = os.path.abspath(python)
    model = (args.model or repo / 'checkpoints' / 'llada15-local').resolve(strict=True)
    data = (args.data or repo / 'data' / 'gsm8k_colar_split').resolve(strict=True)
    shared_root = args.shared_model_root.resolve(strict=True)
    for name in SOURCE_FILES:
        if not (repo / name).is_file():
            raise FileNotFoundError(f'Required runtime source is missing: {repo / name}')
    manifest, integrity = checked_data(data)
    weights = checked_model(model, shared_root)
    launcher = launcher_module(repo)
    current_gpus = launcher.query_gpus()
    selected, selection_policy = select_gpus(current_gpus, args.gpus)
    if any(args.cap_gib >= gpu['total_gib'] for gpu in selected):
        raise ValueError('Allocator cap must be below every candidate GPU total memory')
    environment = {
        'HF_MODULES_CACHE': str(repo / '.cache' / 'modules'),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': str(args.cpu_threads),
        'PYTHONDONTWRITEBYTECODE': '1',
    }
    argv = trainer_argv(args, campaign, model, campaign / 'data')
    spec = launcher.validate_spec(dict(
        argv=argv, cwd=str(repo), env=environment,
        candidate_gpus=[gpu['uuid'] for gpu in selected], gpu_count=2,
        min_free_gib=args.min_free_gib, poll_seconds=args.poll_seconds,
        stable_polls=args.stable_polls, max_wait_seconds=args.max_wait_seconds))
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
            after = digest(original)
            if before != after or digest(source / name) != before:
                raise RuntimeError(f'Source changed while snapshotting; retry preparation: {original}')
            hashes[name] = dict(sha256=before, bytes=(source / name).stat().st_size)
        preflight = cpu_preflight(args.python, source, argv, environment)
        # Also validate the snapshotted launcher's actual CLI, without waiting or launching.
        write_json(stage / 'launch_spec.json', spec)
        subprocess.run([args.python, str(source / 'launch_when_free.py'), '--spec',
                        str(stage / 'launch_spec.json'), '--output', str(campaign / 'launcher'),
                        '--validate-only'], check=True, capture_output=True, text=True, timeout=30,
                       env=dict(os.environ, **environment, CUDA_VISIBLE_DEVICES=''))
        (stage / 'data').mkdir()
        for split in COUNTS:
            (stage / 'data' / (split + '.jsonl')).symlink_to(data / (split + '.jsonl'))
        write_json(stage / 'data' / 'manifest.json', manifest)
        revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=False)
        status = subprocess.run(['git', '-C', str(repo), 'status', '--short'],
                                capture_output=True, text=True, check=False)
        write_json(stage / 'source_manifest.json', dict(
            git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            git_status=status.stdout.splitlines() if status.returncode == 0 else None,
            source_root=str(repo), snapshot_root=str(campaign / 'source'), files=hashes,
            immutable_by='Read-only source files and directory; SHA256 recorded',
            external_data_and_weights_remain_symlinks=True))
        write_json(stage / 'dataset_integrity.json', integrity)
        write_json(stage / 'shared_weights.json', weights)
        write_json(stage / 'gpu_snapshot.json', dict(
            recorded_at=datetime.now(timezone.utc).isoformat(), all_gpus=current_gpus,
            selected_candidates=selected, policy=selection_policy,
            memory_reserved=False, launch_requires_fresh_stable_polls=True))
        write_json(stage / 'method_definition.json', method_definition(args))
        write_json(stage / 'cpu_preflight.json', preflight)
        launcher_argv = [args.python, '-u', str(campaign / 'source' / 'launch_when_free.py'),
                         '--spec', str(campaign / 'launch_spec.json'),
                         '--output', str(campaign / 'launcher')]
        write_json(stage / 'campaign_plan.json', dict(
            prepared_at=datetime.now(timezone.utc).isoformat(), prepared_only=True,
            gpu_job_started=False, train_output=str(campaign / 'train'),
            launcher_output=str(campaign / 'launcher'), launcher_argv=launcher_argv,
            training_argv=argv, initial_eval=False, evaluation_schedule='Every epoch',
            checkpoint_selection='Minimum validation CE', official_test_used=False,
            copied_dataset_bytes=0, copied_weight_bytes=0,
            note='Launcher must be started separately. Preserve its output directory: it prevents repeat launches.'))
        for path in source.iterdir():
            if path.is_file():
                path.chmod(0o444)
        source.chmod(0o555)
        # mkdir is exclusive: an existing campaign can never be replaced, even in a race.
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
                candidate_gpus=spec['candidate_gpus'], launcher_argv=launcher_argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True, help='New campaign directory; never overwritten')
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--python', default=sys.executable, help='Training environment Python; used only for CPU help checks')
    parser.add_argument('--model', type=Path, help='Defaults to repo checkpoints/llada15-local')
    parser.add_argument('--shared-model-root', type=Path, default=Path('/mnt/public/models/LLaDA-1.5'))
    parser.add_argument('--data', type=Path, help='Defaults to repo data/gsm8k_colar_split')
    parser.add_argument('--gpus', nargs='+', help='Candidate physical indices or UUIDs; default queries and pins two most free GPUs')
    parser.add_argument('--min-free-gib', type=float, required=True)
    parser.add_argument('--cap-gib', type=float, required=True, help='Per-device PyTorch allocator cap')
    parser.add_argument('--max-eval-prompt', type=int, required=True, help='Audited validation prompt length cap')
    for name, default in [('max-prompt', 256), ('rounds', 8), ('latent-slots', 240),
                          ('answer-slots', 16), ('block-size', 16), ('epochs', 3),
                          ('rank', 8), ('accumulation', 4), ('seed', 42), ('cpu-threads', 4)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--prompt-profile', choices=('answer_only', 'llada4shot'), default='answer_only')
    parser.add_argument('--eval-prompt-profile', choices=('answer_only', 'llada4shot'), default='llada4shot')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--stable-polls', type=int, default=3)
    parser.add_argument('--max-wait-seconds', type=float, default=86400)
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(prepare(parse_args()), indent=2))
