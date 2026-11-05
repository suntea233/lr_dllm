"""Prepare every row of the original GSM8K train/test splits, without augmentation."""
import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_COUNTS = {'train': 7473, 'test': 1319}


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
    args = parser.parse_args()
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

    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in converted.items():
        # Escape Unicode line separators too: the training reader uses splitlines().
        content = ''.join(json.dumps(row) + '\n' for row in rows).encode('utf-8')
        path = args.output / (split + '.jsonl')
        write_checked(path, content)
        files[path.name] = dict(rows=len(rows), bytes=len(content),
                               sha256=hashlib.sha256(content).hexdigest())
    manifest = dict(source='openai/gsm8k', configuration='main',
                    requested_revision=args.revision or 'main (default)',
                    counts=EXPECTED_COUNTS, augmented=False,
                    split_policy='All original train rows; official test is separate. No holdout, resampling or deduplication.',
                    training_target='Final answer after ####; solution preserves original text for provenance only.',
                    stripped_question_overlap=0,
                    duplicate_questions={s: len(converted[s]) - len(questions[s]) for s in EXPECTED_COUNTS},
                    dataset_sources=sources, files=files)
    write_checked(args.output / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(dict(output=str(args.output.resolve()), counts=EXPECTED_COUNTS,
                          augmented=False, stripped_question_overlap=0), indent=2))


if __name__ == '__main__':
    main()
