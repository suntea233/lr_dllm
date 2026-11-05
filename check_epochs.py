"""CPU checks for full-epoch coverage, tail-update scaling and final-only evaluation."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

import experiment


def run_cli(output, budget, extra=()):
    argv = ['experiment.py', 'train', '--model', str(ROOT / 'checkpoints/tiny-random-llada'),
            '--data', str(ROOT / 'data/train64.jsonl'), '--validation', str(ROOT / 'data/val128.jsonl'),
            '--output', str(output), '--device', 'cpu', '--dtype', 'float32', '--cpu-threads', '2',
            '--train-limit', '5', '--eval-limit', '2', '--accumulation', '4',
            '--strict-lengths', '--verify-reload', *budget, *extra]
    with patch.object(sys, 'argv', argv):
        experiment.main()


def main():
    torch.set_num_threads(2)
    groups = list(experiment.training_groups(5, 4, 42, epochs=2))
    assert [len(g[2]) for g in groups] == [4, 1, 4, 1]
    for epoch in (1, 2):
        assert sorted(i for e, _, indices in groups if e == epoch for i in indices) == list(range(5))
    assert groups == list(experiment.training_groups(5, 4, 42, epochs=2))
    assert groups != list(experiment.training_groups(5, 4, 43, epochs=2))
    legacy = list(experiment.training_groups(5, 4, 42, steps=2))
    rng = random.Random(42)
    assert [i for _, _, indices in legacy for i in indices] == [rng.choice(range(5)) for _ in range(8)]

    with tempfile.TemporaryDirectory(prefix='lr_dllm_epochs_') as directory:
        output = Path(directory) / 'epoch'
        run_cli(output, ['--epochs', '1'], ['--skip-initial-eval'])
        records = [json.loads(x) for x in (output / 'metrics.jsonl').read_text().splitlines()]
        training = [r for r in records if 'train_ce' in r]
        validation = [r for r in records if 'val_ce' in r]
        assert [r['microbatches'] for r in training] == [4, 1]
        assert [r['samples_seen'] for r in training] == [4, 5]
        assert [r['epoch_samples_seen'] for r in training] == [4, 5]
        assert [r['step'] for r in validation] == [2]
        assert not (output / 'predictions/step000000.jsonl').exists()
        order = [json.loads(x) for x in (output / 'sample_order.jsonl').read_text().splitlines()]
        assert [r['indices'] for r in order] == [g[2] for g in groups[:2]]
        coverage = json.loads((output / 'data_coverage.json').read_text())
        assert coverage['samples_seen'] == coverage['train_count'] == 5
        assert coverage['epochs']['1'] == {'unique_examples': 5, 'complete': True}
        assert json.loads((output / 'config.json').read_text())['steps'] == 2

        # Independent optimizer updates: the singleton tail must use its full loss,
        # not one quarter. Compare pre-clipping gradient norms as clipping could
        # otherwise conceal an incorrectly scaled singleton update.
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'checkpoints/tiny-random-llada',
                                                 trust_remote_code=True, local_files_only=True)
        base = AutoModel.from_pretrained(ROOT / 'checkpoints/tiny-random-llada',
                                        trust_remote_code=True, local_files_only=True,
                                        torch_dtype=torch.float32, low_cpu_mem_usage=True)
        reference = experiment.add_lora(base, rank=8).train()
        examples = experiment.encode_rows(ROOT / 'data/train64.jsonl', tokenizer, 'cpu',
                                          256, 16, limit=5, strict=True)
        optimizer = torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], lr=1e-4)
        expected_norms = []
        for _, _, indices in groups[:2]:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in indices:
                prompt, labels, _ = examples[index]
                logits = experiment.recurrent_logits(reference, prompt, base.config.mask_token_id)
                losses.append(experiment.answer_loss(logits, labels))
            torch.stack(losses).mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
            expected_norms.append(float(norm))
            optimizer.step()
        torch.testing.assert_close(torch.tensor([r['grad_norm'] for r in training]),
                                   torch.tensor(expected_norms), rtol=1e-5, atol=1e-6)
        saved = load_file(output / 'adapter/adapter_model.safetensors')
        expected = get_peft_model_state_dict(reference)
        assert saved.keys() == expected.keys()
        for name in saved:
            torch.testing.assert_close(saved[name], expected[name], rtol=1e-5, atol=1e-7)

        try:
            experiment.encode_rows(ROOT / 'data/train64.jsonl', tokenizer, 'cpu', 1, 16, strict=True)
        except ValueError as error:
            assert 'strict-lengths forbids dropping' in str(error)
        else:
            raise AssertionError('Strict length rejection did not fire')

        old_output = Path(directory) / 'legacy'
        run_cli(old_output, ['--steps', '2'])
        old_records = [json.loads(x) for x in (old_output / 'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in old_records if 'val_ce' in r] == [0, 2]
        assert [r['samples_seen'] for r in old_records if 'train_ce' in r] == [4, 8]
        old_order = [json.loads(x) for x in (old_output / 'sample_order.jsonl').read_text().splitlines()]
        assert [r['indices'] for r in old_order] == [g[2] for g in legacy]

    print(json.dumps(dict(status='passed', coverage='every example once per epoch',
                          partial_group_scaling='matches independent gradient/reference weights',
                          skip_initial_eval='passed', strict_lengths='passed',
                          legacy_with_replacement='passed')))


if __name__ == '__main__':
    main()
