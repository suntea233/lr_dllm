"""Audit real-tokenizer length filtering and GSM8K split membership."""
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
    parser.add_argument('--splits', nargs='+', default=['train64', 'train1000', 'val128'],
                        help='JSONL file stems; use train test for full official GSM8K')
    parser.add_argument('--require-all', action='store_true', help='Fail if any example exceeds the length limits')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    counts, questions = {}, {}
    for name in args.splits:
        path = args.data / (name + '.jsonl')
        raw = [json.loads(x) for x in path.read_text().splitlines()]
        questions[name] = {r['question'].strip() for r in raw}
        prompt_lengths, answer_lengths = [], []
        for row in raw:
            text = row['question'] + '\nReturn only the final numerical answer.'
            prompt_lengths.append(len(tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True)))
            answer_lengths.append(len(tokenizer.encode(row['answer'], add_special_tokens=False)) + 1)
        rows = encode_rows(path, tokenizer, 'cpu', args.max_prompt, args.answer_slots)
        if name in ('train64', 'train1000', 'val128'):
            assert len(questions[name]) == len(raw)
        counts[name] = dict(raw=len(raw), accepted=len(rows), skipped=len(raw) - len(rows),
                           duplicate_questions=len(raw) - len(questions[name]),
                           max_raw_prompt_tokens=max(prompt_lengths),
                           max_raw_answer_tokens_including_eos=max(answer_lengths),
                           max_accepted_prompt_tokens=max(x.shape[1] for x, y, a in rows),
                           max_answer_tokens_including_eos=max(int((y != -100).sum()) for x, y, a in rows),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if {'train1000', 'val128'} <= questions.keys():
        assert not questions['train1000'] & questions['val128']
    if {'train64', 'train1000'} <= questions.keys():
        assert questions['train64'] <= questions['train1000']
    if {'train', 'test'} <= questions.keys():
        assert not questions['train'] & questions['test']
    (args.data / 'tokenization_audit.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    if args.require_all and any(row['skipped'] for row in counts.values()):
        raise ValueError('Some examples exceed length limits; inspect tokenization_audit.json and increase limits')


if __name__ == '__main__':
    main()
