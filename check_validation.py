"""CPU checks for epoch validation, checkpoint selection and leakage safeguards."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

import experiment


def read_records(directory):
    return [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]


def run_cli(output, *extra):
    argv = ['experiment.py', 'train', '--model', str(ROOT / 'checkpoints/tiny-random-llada'),
            '--data', str(ROOT / 'data/train64.jsonl'), '--validation', str(ROOT / 'data/val128.jsonl'),
            '--output', str(output), '--device', 'cpu', '--dtype', 'float32', '--cpu-threads', '2',
            '--train-limit', '5', '--eval-limit', '2', '--accumulation', '4', '--rounds', '1',
            '--latent-slots', '4', '--strict-lengths', '--verify-reload', '--no-tensorboard', *extra]
    with patch.object(sys, 'argv', argv):
        experiment.main()


def assert_weights(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def check_validation_math(directory):
    logits = [torch.tensor([[[3., 1., 0.], [1., 0., 2.], [0., 0., 4.]]]),
              torch.tensor([[[2., 1., 0.], [0., 3., 1.], [0., 0., 3.]]])]
    labels = [torch.tensor([[0, 2, -100]]), torch.tensor([[1, 1, 2]])]
    examples = [(torch.tensor([[i]]), label, '0' if i == 0 else '1 1')
                for i, label in enumerate(labels)]
    class Tokenizer:
        eos_token_id = 2
        def decode(self, ids, **kwargs):
            return ' '.join(str(i) for i in ids)
    model = torch.nn.Linear(1, 1).train()
    def fake_forward(model, prompt, trace=None, **settings):
        assert not model.training and not torch.is_grad_enabled()
        trace.extend([torch.tensor([[[1., 0.], [0., 1.]]]),
                      torch.tensor([[[0., 1.], [1., 0.]]])])
        return logits[prompt.item()]
    with patch.object(experiment, 'recurrent_logits', fake_forward):
        result = experiment.evaluate_validation(model, examples, Tokenizer(), {},
                                                directory / 'math.jsonl', lambda: None)
    assert model.training
    token_losses = [torch.nn.functional.cross_entropy(x[lab != -100], lab[lab != -100], reduction='none')
                    for x, lab in zip(logits, labels)]
    assert math.isclose(result['val_ce'], sum(x.mean().item() for x in token_losses) / 2, rel_tol=1e-6)
    assert math.isclose(result['val_token_ce'], torch.cat(token_losses).mean().item(), rel_tol=1e-6)
    assert result['val_ce'] != result['val_token_ce']
    assert result['val_target_tokens'] == 5 and result['val_token_accuracy'] == 4/5
    assert result['val_answer_em'] == 1/2 and result['val_eos_rate'] == 1
    assert result['val_latent_round_cosine'] == 0
    assert math.isclose(result['val_latent_relative_change'], math.sqrt(2), rel_tol=1e-6)
    model.eval()
    with patch.object(experiment, 'recurrent_logits', fake_forward):
        experiment.evaluate_validation(model, examples, Tokenizer(), {}, directory / 'eval.jsonl', lambda: None)
    assert not model.training
    model.train()
    with patch.object(experiment, 'recurrent_logits', side_effect=RuntimeError('simulated eval failure')):
        try:
            experiment.evaluate_validation(model, examples, Tokenizer(), {}, directory / 'fail.jsonl', lambda: None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Injected eval failure did not propagate')
    assert model.training


def check_selection_rules(directory):
    best = experiment.BestCheckpoint(directory, '/independent/dev.jsonl', 2)
    def metrics(value, step=1):
        return dict(val_ce=value, step=step, epoch=1, predictions_file='predictions/example.jsonl')
    with patch.object(experiment, 'save_adapter_checkpoint') as save:
        for value in (float('nan'), float('inf'), -float('inf')):
            assert not best.consider(None, None, metrics(value))
        assert not best.consider(None, None, metrics(0.1, step=0))
        assert save.call_count == 0
        assert best.consider(None, None, metrics(2.0))
        recorded = (directory / 'best_checkpoint.json').read_bytes()
        for value in (2.0, 3.0, float('nan')):
            assert not best.consider(None, None, metrics(value, step=2))
        assert recorded == (directory / 'best_checkpoint.json').read_bytes()
        assert best.consider(None, None, metrics(1.0, step=3))
        assert best.best['step'] == 3 and save.call_count == 2


def check_checkpoint_rollback(directory):
    checkpoint = directory / 'rollback_adapter'
    checkpoint.mkdir()
    (checkpoint / 'sentinel').write_text('previous good adapter')
    class Model:
        def save_pretrained(self, target, **kwargs):
            Path(target).mkdir()
            (Path(target) / 'new').write_text('replacement')
    class Tokenizer:
        def save_pretrained(self, target):
            pass
    rename = Path.rename
    def fail_staging(self, target):
        if '.pending-' in self.name:
            raise OSError('simulated rename failure')
        return rename(self, target)
    with patch.object(Path, 'rename', fail_staging):
        try:
            experiment.save_adapter_checkpoint(Model(), Tokenizer(), checkpoint)
        except OSError:
            pass
        else:
            raise AssertionError('Injected checkpoint failure did not propagate')
    assert (checkpoint / 'sentinel').read_text() == 'previous good adapter'
    assert not list(directory.glob('.rollback_adapter.*'))


def check_split_guard(directory):
    train, dev, test = [directory / n for n in ('train.jsonl', 'dev.jsonl', 'test.jsonl')]
    train.write_text(json.dumps(dict(question='one', source_split='train')) + '\n')
    dev.write_text(json.dumps(dict(question='two', source_split='train')) + '\n')
    test.write_text(json.dumps(dict(question='three', source_split='test')) + '\n')
    assert experiment.check_evaluation_split(train, dev)['question_overlap'] == 0
    assert experiment.check_evaluation_split(train, test, 'test')['official_test_provenance']
    for bad, fragment in ((test, 'cannot select'), (train, 'overlap')):
        try:
            experiment.check_evaluation_split(train, bad)
        except ValueError as error:
            assert fragment in str(error)
        else:
            raise AssertionError('Unsafe validation split was accepted')


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='lr_dllm_validation_') as temporary:
        root = Path(temporary)
        check_validation_math(root)
        check_selection_rules(root)
        check_split_guard(root)
        check_checkpoint_rollback(root)
        output = root / 'epochs'
        snapshots = []
        actual_evaluate = experiment.evaluate_validation
        values = iter((1.0, 0.5, 0.8))
        def controlled_validation(model, *args, **kwargs):
            assert model.training
            result = actual_evaluate(model, *args, **kwargs)
            assert model.training
            snapshots.append({key: value.detach().clone()
                              for key, value in get_peft_model_state_dict(model).items()})
            result['val_ce'] = next(values)
            return result
        with patch.object(experiment, 'evaluate_validation', controlled_validation):
            run_cli(output, '--epochs', '3')
        records = read_records(output)
        validation = [record for record in records if 'val_ce' in record]
        training = [record for record in records if 'train_ce' in record]
        assert [record['step'] for record in validation] == [2, 4, 6]
        assert [record['epoch'] for record in validation] == [1, 2, 3]
        assert [record['is_best'] for record in validation] == [True, True, False]
        assert [record['microbatches'] for record in training] == [4, 1, 4, 1, 4, 1]
        assert [record['epoch_sample_count'] for record in training if 'epoch_sample_count' in record] == [5, 5, 5]
        assert not (output / 'predictions/step000000.jsonl').exists()
        selected = json.loads((output / 'best_checkpoint.json').read_text())
        latest = json.loads((output / 'last_checkpoint.json').read_text())
        assert selected['step'] == 4 and selected['epoch'] == 2 and selected['value'] == 0.5
        assert latest['step'] == 6 and latest['epoch'] == 3
        assert_weights(load_file(output / 'best_adapter/adapter_model.safetensors'), snapshots[1])
        assert_weights(load_file(output / 'adapter/adapter_model.safetensors'), snapshots[2])
        assert any(not torch.equal(snapshots[1][key], snapshots[2][key]) for key in snapshots[1])
        assert not list(output.glob('.*.pending-*')) and not list(output.glob('.*.previous'))
        for folder in ('adapter', 'best_adapter'):
            assert not (output / folder / 'model.safetensors').exists()
        test_output = root / 'test_role'
        run_cli(test_output, '--epochs', '2', '--validation-role', 'test', '--train-limit', '1')
        test_records = read_records(test_output)
        assert [r['step'] for r in test_records if 'test_ce' in r] == [2]
        assert not any('val_ce' in r for r in test_records)
        assert not (test_output / 'best_adapter').exists()
        assert not (test_output / 'best_checkpoint.json').exists()
        for extra in (('--epochs', '2', '--eval-every', '1'),
                      ('--epochs', '2', '--validation-role', 'test', '--initial-eval')):
            try:
                run_cli(root / 'invalid', *extra)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid dense/test evaluation schedule was accepted')
    print(json.dumps(dict(status='passed', epoch_validation='one evaluation at each epoch boundary',
                          best_checkpoint='matches exact epoch-2 weights; latest matches epoch 3',
                          nonfinite_and_ties='excluded from replacement',
                          diagnostics='independently checked token weighting and hidden deltas',
                          model_mode='restored on success and failure',
                          test_role='final-only without selection', split_leakage_guards='passed')))


if __name__ == '__main__':
    main()
