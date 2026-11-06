"""Training-sourced four-shot prompts for the continuous-state LLaDA experiment.

This is an explicit adaptation, not a byte-identical OpenCompass prompt. The
four Question/Answer demonstrations are placed inside one user message and
rendered with the loaded LLaDA chat template. No current-example solution or
answer is accepted by the prompt builders.
"""
import hashlib
import json
import re
from pathlib import Path


DEFAULT_INDICES = (669, 1708, 6724, 0, 1)
PROFILE_NAME = 'llada_train_four_shot'


def _read_rows(path):
    path = Path(path).resolve(strict=True)
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty prompt source: {path}')
    return path, content, rows


def load_fewshot_pool(trainpath, indices=DEFAULT_INDICES):
    """Load five fixed training examples; the fifth replaces any self-example.

    Returns a JSON-serializable provenance record, including the exact source
    file hash and demonstration contents. Source indices refer to the original
    GSM8K training split, not the current file's row numbers.
    """
    indices = tuple(indices)
    if len(indices) < 5 or len(set(indices)) != len(indices) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError('Provide at least five distinct nonnegative training source indices')
    path, content, rows = _read_rows(trainpath)
    if any(re.fullmatch(r'(?:val|validation|dev|test)[0-9]*', part)
           for part in re.split(r'[_\-.]', path.stem.lower())):
        raise ValueError('Few-shot demonstrations must come from the training file')
    by_index = {}
    for row in rows:
        index = row.get('source_index')
        if index not in indices:
            continue
        if index in by_index:
            raise ValueError(f'Duplicate source_index in training source: {index}')
        if row.get('source_split') != 'train':
            raise ValueError(f'Demonstration {index} is not from the original training split')
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'Demonstration {index} has no question')
        solution = row.get('solution')
        if not isinstance(solution, str) or '####' not in solution:
            raise ValueError(f'Demonstration {index} must retain its full GSM8K solution')
        answer = row.get('answer')
        if not isinstance(answer, str) or solution.rsplit('####', 1)[1].strip() != answer.strip():
            raise ValueError(f'Demonstration {index} has an inconsistent final answer')
        by_index[index] = dict(source_index=index, source_split='train',
                               question=row['question'].strip(), solution=solution.strip(),
                               answer=answer.strip())
    missing = set(indices) - set(by_index)
    if missing:
        raise ValueError(f'Training source lacks demonstration indices: {sorted(missing)}')
    examples = [by_index[index] for index in indices]
    if len({row['question'] for row in examples}) != len(examples):
        raise ValueError('Few-shot pool questions must be distinct')
    return dict(profile=PROFILE_NAME, source_file=str(path),
                source_sha256=hashlib.sha256(content).hexdigest(),
                source_rows=len(rows), source_indices=list(indices), shots=4,
                examples=examples,
                current_question_policy='exclude exact stripped question before selecting first four',
                rendering='Question/Answer text inside a single user chat-template message',
                note='Training-sourced replacement pool; not the official fixed four-shot examples')


def select_examples(question, pool):
    """Return four demonstrations without including the current question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must be a nonempty string')
    question = question.strip()
    examples = [row for row in pool['examples'] if row['question'].strip() != question][:4]
    if len(examples) != 4 or len({row['question'].strip() for row in examples}) != 4:
        raise ValueError('Four distinct non-self demonstrations are required')
    return examples


def build_prompt(question, pool):
    """Build four full CoT demonstrations followed by an unanswered question."""
    examples = select_examples(question, pool)
    blocks = [f"Question: {row['question']}\nLet's think step by step\nAnswer: {row['solution']}"
              for row in examples]
    blocks.append(f"Question: {question.strip()}\nLet's think step by step\nAnswer:")
    return '\n\n'.join(blocks)


def encode_prompt(tokenizer, question, pool):
    """Return untruncated chat-template token IDs, ending at the assistant prefix."""
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': build_prompt(question, pool)}],
        tokenize=True, add_generation_prompt=True)


def audit_fewshot_prompts(tokenizer, split_paths, pool, *, max_prompt=None):
    """Audit prompt lengths and forbid demonstration overlap with held-out data.

    ``split_paths`` is a mapping such as {"train": train_file, "val": val_file}.
    Names starting with ``train`` permit self-examples because the prompt builder
    excludes those examples. All other splits must be disjoint from the pool.
    This function never truncates or deletes a row.
    """
    if max_prompt is not None and max_prompt < 1:
        raise ValueError('max_prompt must be positive')
    source_path = Path(pool['source_file'])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != pool['source_sha256']:
        raise ValueError('Few-shot training source changed after pool creation')
    demo_questions = {row['question'].strip() for row in pool['examples']}
    result = dict(profile=pool['profile'], pool_source_sha256=pool['source_sha256'],
                  source_indices=pool['source_indices'], max_prompt=max_prompt, splits={})
    for name, filename in split_paths.items():
        path, content, rows = _read_rows(filename)
        overlap = sum(row['question'].strip() in demo_questions for row in rows)
        if not name.lower().startswith('train') and overlap:
            raise ValueError(f'Few-shot pool overlaps held-out split {name}: {overlap} questions')
        lengths = [len(encode_prompt(tokenizer, row['question'], pool)) for row in rows]
        ordered = sorted(lengths)
        oversized = sum(length > max_prompt for length in lengths) if max_prompt is not None else 0
        result['splits'][name] = dict(
            file=str(path), source_sha256=hashlib.sha256(content).hexdigest(), count=len(rows),
            prompt_min=min(lengths), prompt_mean=sum(lengths) / len(lengths),
            prompt_p95=ordered[int(.95 * (len(ordered) - 1))], prompt_max=max(lengths),
            over_max_prompt=oversized, pool_question_overlap=overlap,
            self_examples_excluded_from_prompts=overlap)
        if oversized:
            raise ValueError(f'{name}: {oversized} prompts exceed {max_prompt}; maximum is {max(lengths)}')
    return result
