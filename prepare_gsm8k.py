"""Prepare original GSM8K, optionally reserving a reproducible validation holdout."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path


EXPECTED_COUNTS = {'train': 7473, 'test': 1319}


def validation_holdout(rows, size, seed, policy='random'):
    """Partition original train rows without changing order within each partition."""
    if not 0 <= size < len(rows):
        raise ValueError('validation-size must be nonnegative and smaller than the training split')
    if policy not in ('random', 'tail'):
        raise ValueError('validation-policy must be random or tail')
    if policy == 'tail':
        boundary = len(rows) - size
        return rows[:boundary], rows[boundary:]
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    held_out = set(indices[:size])
    return ([row for i, row in enumerate(rows) if i not in held_out],
            [row for i, row in enumerate(rows) if i in held_out])


def write_checked(path, data):
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f'Existing file differs; choose a fresh output directory: {path}')
    if not path.exists():
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/gsm8k_full'))
    parser.add_argument('--cache-dir', type=Path, help='Optional Hugging Face datasets cache')
    parser.add_argument('--revision', help='Optional pinned Hugging Face dataset revision')
    parser.add_argument('--offline', action='store_true', help='Use an existing datasets cache only')
    parser.add_argument('--validation-size', type=int, default=0,
                        help='Reserve this many original train rows for validation; 0 preserves all 7473')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Reproducible train/validation partition seed')
    parser.add_argument('--validation-policy', choices=('random', 'tail'), default='random',
                        help='random: seeded holdout; tail: last N original train rows (CoLaR uses 747)')
    args = parser.parse_args()
    if not 0 <= args.validation_size < EXPECTED_COUNTS['train']:
        parser.error('--validation-size must be between 0 and 7472')
    if args.offline:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    from datasets import load_dataset

    kwargs = {}
    if args.cache_dir is not None:
        kwargs['cache_dir'] = str(args.cache_dir.expanduser())
    if args.revision is not None:
        kwargs['revision'] = args.revision
    dataset = load_dataset('openai/gsm8k', 'main', **kwargs)
    converted, questions, sources = {}, {}, {}
    for split, expected in EXPECTED_COUNTS.items():
        data = dataset[split]
        if len(data) != expected:
            raise ValueError(f'Expected original GSM8K {split} count {expected}, got {len(data)}')
        rows = []
        for i, row in enumerate(data):
            if '####' not in row['answer']:
                raise ValueError(f'Missing final answer delimiter: {split}[{i}]')
            answer = row['answer'].rsplit('####', 1)[1].strip()
            if not answer:
                raise ValueError(f'Empty final answer: {split}[{i}]')
            rows.append(dict(source_index=i, source_split=split, question=row['question'],
                             answer=answer, solution=row['answer']))
        converted[split] = rows  # Preserve original order and every row, including duplicates.
        questions[split] = {row['question'].strip() for row in rows}
        sources[split] = dict(fingerprint=data._fingerprint,
                             cache_files=[dict(path=f['filename'],
                                               sha256=hashlib.sha256(Path(f['filename']).read_bytes()).hexdigest())
                                          for f in data.cache_files])
    overlap = questions['train'] & questions['test']
    if overlap:
        raise ValueError(f'Official splits contain {len(overlap)} identical stripped questions')

    original_duplicate_counts = {s: len(converted[s]) - len(questions[s]) for s in EXPECTED_COUNTS}
    if args.validation_size:
        converted['train'], converted['val'] = validation_holdout(
            converted['train'], args.validation_size, args.split_seed, args.validation_policy)
        train_questions = {row['question'].strip() for row in converted['train']}
        val_questions = {row['question'].strip() for row in converted['val']}
        if train_questions & val_questions:
            raise ValueError('Train/validation holdout contains overlapping questions')

    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in converted.items():
        # Escape Unicode line separators too: the training reader uses splitlines().
        content = ''.join(json.dumps(row) + '\n' for row in rows).encode('utf-8')
        path = args.output / (split + '.jsonl')
        write_checked(path, content)
        files[path.name] = dict(rows=len(rows), bytes=len(content),
                               sha256=hashlib.sha256(content).hexdigest())
    counts = {split: len(rows) for split, rows in converted.items()}
    manifest = dict(source='openai/gsm8k', configuration='main',
                    requested_revision=args.revision or 'main (default)',
                    counts=counts, augmented=False,
                    split_policy=(f'{args.validation_size} original train rows reserved for validation; '
                                  'remaining original train rows used for training. Official test untouched.'
                                  if args.validation_size else
                                  'All original train rows; official test is separate. No holdout, resampling or deduplication.'),
                    training_target='Final answer after ####; solution preserves original text for provenance only.',
                    stripped_question_overlap=0,
                    duplicate_questions=original_duplicate_counts,
                    dataset_sources=sources, files=files)
    if args.validation_size:
        manifest['validation_holdout'] = dict(
            source_split='train', size=args.validation_size, policy=args.validation_policy,
            seed=args.split_seed if args.validation_policy == 'random' else None,
            train_validation_question_overlap=0,
            validation_source_indices=[row['source_index'] for row in converted['val']],
            official_counts=EXPECTED_COUNTS)
    write_checked(args.output / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(dict(output=str(args.output.resolve()), counts=counts,
                          augmented=False, stripped_question_overlap=0), indent=2))


if __name__ == '__main__':
    main()
