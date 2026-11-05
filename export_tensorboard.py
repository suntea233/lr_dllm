"""Export existing JSONL logs into an idempotent, separate TensorBoard view.

Example: python export_tensorboard.py runs/run_a runs/run_b --output runs/tb_history
Old runs evaluated on official GSM8K test are labelled test/*, never val/*.
The exporter does not create checkpoints or change original training artifacts.
"""
import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path
import shutil
import tempfile
import uuid

from telemetry import TrainingTelemetry, ensure_tensorboard_available


MARKER = '.lr_dllm_tensorboard_export.json'
EXPORT_VERSION = 1


def infer_eval_split(config, requested='auto'):
    if requested != 'auto':
        return requested
    if config.get('validation_role') == 'test':
        return 'test'
    validation = config.get('validation')
    if not validation:
        return 'val'
    path = Path(validation)
    if path.stem.lower() in ('test', 'gsm8k_test'):
        return 'test'
    if path.is_file():
        with path.open() as handle:
            first_line = handle.readline()
        if first_line.strip():
            first = json.loads(first_line)
            if first.get('source_split') == 'test' or first.get('split') == 'test':
                return 'test'
    return 'val'


def _validate_owned_destination(destination, source):
    marker = destination / MARKER
    if destination.is_symlink() or marker.is_symlink() or not marker.is_file():
        raise FileExistsError(f'Refusing to replace an unmanaged directory: {destination}')
    previous = json.loads(marker.read_text())
    if previous.get('exporter') != 'lr_dllm' or previous.get('source') != str(source):
        raise FileExistsError(f'Export destination belongs to another source: {destination}')
    for entry in destination.iterdir():
        if entry.name == MARKER:
            continue
        if entry.is_symlink() or not entry.is_file() or not entry.name.startswith('events.out.tfevents.'):
            raise FileExistsError(f'Refusing to replace unexpected content: {entry}')
    return previous


def export_run(source, output, eval_split='auto'):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    destination = output / source.name
    if destination == source or source in destination.parents:
        raise ValueError('Export output must be outside the original training run')
    metrics_bytes = (source / 'metrics.jsonl').read_bytes()
    config_path = source / 'config.json'
    config_bytes = config_path.read_bytes() if config_path.exists() else b'{}'
    config = json.loads(config_bytes)
    split = infer_eval_split(config, eval_split)
    records = [json.loads(line) for line in metrics_bytes.splitlines() if line.strip()]
    if not records:
        raise ValueError(f'No metrics found in {source}')
    metadata = dict(exporter='lr_dllm', export_version=EXPORT_VERSION, source=str(source),
                    metrics_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    eval_split=split, record_count=len(records),
                    timestamped_record_count=sum(isinstance(r.get('wall_time'), Real)
                                                 for r in records),
                    wall_time_policy='Use recorded wall_time when available; otherwise event '
                                     'timestamps are export time. Use STEP and recorded '
                                     'elapsed_train_s for historical training progress, not '
                                     'the TensorBoard WALL axis.',
                    note='Derived view only; original artifacts unchanged. Historical validation '
                         'on the official test split is labelled test. No best checkpoint inferred.')
    if destination.exists() or destination.is_symlink():
        previous = _validate_owned_destination(destination, source)
        if previous == metadata:
            return dict(source=str(source), output=str(destination), eval_split=split,
                        records=len(records), status='unchanged')
    ensure_tensorboard_available()
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + source.name + '-export-', dir=output))
    backup = None
    try:
        with TrainingTelemetry(source, config=dict(config, tensorboard_export=metadata),
                               eval_split=split, event_dir=staging) as logger:
            for record in records:
                logger.log(record)
        (staging / MARKER).write_text(json.dumps(metadata, indent=2) + '\n')
        if destination.exists():
            backup = output / ('.' + source.name + '-previous-' + uuid.uuid4().hex)
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup is not None:
                backup.rename(destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return dict(source=str(source), output=str(destination), eval_split=split,
                    records=len(records), status='exported')
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='Training run directories containing metrics.jsonl')
    parser.add_argument('--output', required=True, help='Separate root for managed derived event runs')
    parser.add_argument('--eval-split', choices=('auto', 'val', 'test'), default='auto',
                        help='auto checks validation_role, filename, and source_split provenance')
    args = parser.parse_args()
    names = [Path(run).resolve().name for run in args.runs]
    if len(names) != len(set(names)):
        parser.error('Run directory names must be unique within an export root')
    for run in args.runs:
        print(json.dumps(export_run(run, args.output, args.eval_split)), flush=True)


if __name__ == '__main__':
    main()
