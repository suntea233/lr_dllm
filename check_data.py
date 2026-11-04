"""Audit real-tokenizer length filtering and disjoint GSM8K subsets."""
import hashlib
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from experiment import encode_rows


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=root / 'checkpoints/tiny-random-llada')
    parser.add_argument('--data', type=Path, default=root / 'data')
    parser.add_argument('--max-prompt', type=int, default=256)
    parser.add_argument('--answer-slots', type=int, default=16)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    counts, questions = {}, {}
    for name in ('train64', 'train1000', 'val128'):
        path = args.data / (name + '.jsonl')
        raw = [json.loads(x) for x in path.read_text().splitlines()]
        questions[name] = {r['question'] for r in raw}
        rows = encode_rows(path, tokenizer, 'cpu', args.max_prompt, args.answer_slots)
        assert len(questions[name]) == len(raw)
        counts[name] = dict(raw=len(raw), accepted=len(rows), skipped=len(raw) - len(rows),
                           max_accepted_prompt_tokens=max(x.shape[1] for x, y, a in rows),
                           max_answer_tokens_including_eos=max(int((y != -100).sum()) for x, y, a in rows),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert not questions['train1000'] & questions['val128']
    assert questions['train64'] <= questions['train1000']
    (args.data / 'tokenization_audit.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
