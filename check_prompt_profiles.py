"""CPU checks for training-only, self-excluding four-shot prompt construction."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prompt_profiles import (audit_fewshot_prompts, build_prompt, encode_prompt,
                             load_fewshot_pool, select_examples)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return list(messages[0]['content'].encode('utf-8')) + [0]


class PromptChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [dict(source_index=i, source_split='train', question=f'Question marker {i}?',
                          answer=str(i), solution=f'Unique reasoning marker {i}.\n#### {i}')
                     for i in range(7)]
        self.train = self.root / 'train.jsonl'
        self.write(self.train, self.rows)
        self.pool = load_fewshot_pool(self.train, indices=[0, 1, 2, 3, 4])

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_provenance_and_full_solutions(self):
        self.assertEqual(self.pool['source_sha256'], hashlib.sha256(self.train.read_bytes()).hexdigest())
        self.assertEqual(self.pool['source_rows'], 7)
        self.assertEqual(self.pool['source_indices'], [0, 1, 2, 3, 4])
        text = build_prompt('Unseen query?', self.pool)
        for row in self.rows[:4]:
            self.assertIn(row['solution'], text)
        self.assertNotIn(self.rows[4]['solution'], text)
        self.assertTrue(text.endswith("Question: Unseen query?\nLet's think step by step\nAnswer:"))
        self.assertEqual(text.count('Question:'), 5)

    def test_self_demonstrations_are_replaced_without_query_gold(self):
        for row in self.rows[:5]:
            selected = select_examples('  ' + row['question'] + '  ', self.pool)
            self.assertEqual(len(selected), 4)
            self.assertNotIn(row['source_index'], [x['source_index'] for x in selected])
            text = build_prompt(row['question'], self.pool)
            self.assertNotIn(row['solution'], text)
            self.assertEqual(text.count(row['question']), 1)
        # The builder accepts a question, not the current row/answer/solution.
        with self.assertRaises(TypeError):
            build_prompt(question='Unseen query?', pool=self.pool, answer='SECRET')

    def test_encoding_is_untruncated_and_uses_chat_template(self):
        question = 'A distinct evaluation question?'
        ids = encode_prompt(FakeTokenizer(), question, self.pool)
        self.assertEqual(ids, list(build_prompt(question, self.pool).encode('utf-8')) + [0])

    def test_nontraining_missing_duplicate_and_inconsistent_sources_rejected(self):
        validation = self.root / 'val.jsonl'
        self.write(validation, self.rows)
        with self.assertRaisesRegex(ValueError, 'training file'):
            load_fewshot_pool(validation, [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'lacks demonstration'):
            load_fewshot_pool(self.train, [0, 1, 2, 3, 99])
        bad = [dict(row) for row in self.rows]
        bad[0]['source_split'] = 'test'
        self.write(self.train, bad)
        with self.assertRaisesRegex(ValueError, 'original training split'):
            load_fewshot_pool(self.train, [0, 1, 2, 3, 4])
        bad[0]['source_split'] = 'train'
        bad[0]['answer'] = 'WRONG'
        self.write(self.train, bad)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            load_fewshot_pool(self.train, [0, 1, 2, 3, 4])
        self.write(self.train, self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate source_index'):
            load_fewshot_pool(self.train, [0, 1, 2, 3, 4])

    def test_audit_checks_heldout_overlap_lengths_and_source_mutation(self):
        validation = self.root / 'val.jsonl'
        self.write(validation, [self.rows[6]])
        audit = audit_fewshot_prompts(FakeTokenizer(), {'train': self.train, 'val': validation}, self.pool)
        self.assertEqual(audit['splits']['train']['count'], 7)
        self.assertEqual(audit['splits']['train']['pool_question_overlap'], 5)
        self.assertEqual(audit['splits']['val']['pool_question_overlap'], 0)
        self.assertEqual(audit['splits']['val']['count'], 1)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            audit_fewshot_prompts(FakeTokenizer(), {'val': validation}, self.pool, max_prompt=10)
        self.write(validation, [self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'overlaps held-out'):
            audit_fewshot_prompts(FakeTokenizer(), {'val': validation}, self.pool)
        self.write(self.train, self.rows[:-1])
        with self.assertRaisesRegex(ValueError, 'source changed'):
            audit_fewshot_prompts(FakeTokenizer(), {'train': self.train}, self.pool)


if __name__ == '__main__':
    unittest.main(verbosity=2)
