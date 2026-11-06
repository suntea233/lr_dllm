"""Low-overhead TensorBoard views of the canonical metrics.jsonl records.

Loss tags keep the trainer's semantics: */loss is mean per-example answer CE,
while */token_loss is token-weighted CE. Only */loss selects a validation best.
"""
import json
from numbers import Real
from pathlib import Path


def ensure_tensorboard_available():
    """Fail before loading an expensive model when TensorBoard is requested."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            'TensorBoard logging requires tensorboard. Install requirements.txt '
            'in the training environment, or explicitly use --no-tensorboard.'
        ) from error
    return SummaryWriter


_TAGS = {
    'train_ce': 'train/loss',
    'train_token_ce': 'train/token_loss',
    'epoch_train_ce': 'train/epoch_loss',
    'epoch_train_token_ce': 'train/epoch_token_loss',
    'epoch_train_s': 'performance/epoch_train_s',
    'epoch_sample_count': 'progress/epoch_sample_count',
    'epoch_target_tokens': 'progress/epoch_target_tokens',
    'epoch_samples_per_s': 'performance/epoch_samples_per_s',
    'lr': 'optimizer/learning_rate',
    'grad_norm': 'optimizer/grad_norm',
    'samples_seen': 'progress/samples_seen',
    'microbatches': 'progress/microbatches',
    'epoch': 'progress/epoch',
    'epoch_samples_seen': 'progress/epoch_samples_seen',
    'train_target_tokens': 'progress/step_target_tokens',
    'train_prompt_tokens': 'progress/step_prompt_tokens',
    'step_train_s': 'performance/step_train_s',
    'elapsed_train_s': 'performance/elapsed_train_s',
    'elapsed_wall_s': 'performance/elapsed_wall_s',
    'samples_per_s': 'performance/samples_per_s',
    'answer_tokens_per_s': 'performance/answer_tokens_per_s',
    'model_positions_per_s': 'performance/model_positions_per_s',
    'allocated_gib': 'gpu/allocated_gib',
    'reserved_gib': 'gpu/reserved_gib',
    'peak_allocated_gib': 'gpu/peak_allocated_gib',
    'best_val_ce': 'checkpoint/best_val_loss',
    'best_step': 'checkpoint/best_step',
    'best_epoch': 'checkpoint/best_epoch',
    'is_best': 'checkpoint/is_best',
}

_EVAL_NAMES = {
    'ce': 'loss',
    'token_ce': 'token_loss',
}


def scalar_tag(name, eval_split='val'):
    """Map stable JSON field names to readable TensorBoard groups."""
    if name.startswith('val_'):
        suffix = name[4:]
        return f'{eval_split}/{_EVAL_NAMES.get(suffix, suffix)}'
    if name.startswith('test_'):
        suffix = name[5:]
        return f'test/{_EVAL_NAMES.get(suffix, suffix)}'
    if name in _TAGS:
        return _TAGS[name]
    if name.startswith('train_'):
        return 'train/' + name[6:]
    if name.startswith('epoch_'):
        return 'epoch/' + name[6:]
    return 'diagnostics/' + name


class TrainingTelemetry:
    """Write metrics without additional model forwards or tensor histograms.

    Native runs use ``output_dir/tensorboard``. ``event_dir`` is only needed by
    the historical exporter, which builds a separate, managed derived view.
    """

    def __init__(self, output_dir, config=None, enabled=True, *, eval_split='val',
                 event_dir=None):
        if eval_split not in ('val', 'test'):
            raise ValueError('eval_split must be val or test')
        self.eval_split = eval_split
        self.writer = None
        self.event_dir = Path(event_dir) if event_dir else Path(output_dir) / 'tensorboard'
        if enabled:
            writer_type = ensure_tensorboard_available()
            self.writer = writer_type(log_dir=str(self.event_dir), max_queue=100,
                                      flush_secs=15)
            if config is not None:
                self.writer.add_text('run/config',
                                     '```json\n' + json.dumps(config, indent=2, default=str) + '\n```',
                                     global_step=0)
            self.writer.add_text(
                'run/metric_definitions',
                '`train/loss`, `val/loss`, `test/loss`: mean per-example answer '
                'cross-entropy (including EOS). `*/token_loss`: token-weighted '
                'cross-entropy. `*/answer_em` and `*/token_accuracy`: fractions, '
                'not percentages. Grad norm is measured before clipping. '
                'No intermediate latent tokens are decoded for logging.',
                global_step=0)
            self.writer.flush()

    def log(self, metrics):
        if self.writer is None:
            return
        step = int(metrics.get('step', 0))
        wall_time = metrics.get('wall_time')
        if not isinstance(wall_time, Real):
            wall_time = None
        split = 'test' if 'test_ce' in metrics else self.eval_split
        for name, value in metrics.items():
            if name == 'gpu_memory_by_device' and isinstance(value, dict):
                for device, fields in value.items():
                    for metric, amount in fields.items():
                        if isinstance(amount, Real):
                            self.writer.add_scalar(
                                f'gpu/{device.replace(":", "_")}/{metric}', float(amount),
                                global_step=step, walltime=wall_time)
                continue
            if name in ('step', 'wall_time') or not isinstance(value, Real):
                continue
            if name in ('eval_s', 'eval_samples_per_s'):
                tag = f'{split}/{name}'
            else:
                tag = scalar_tag(name, self.eval_split)
            self.writer.add_scalar(tag, float(value), global_step=step,
                                   walltime=wall_time)
        if any(key in metrics for key in ('val_ce', 'test_ce', 'epoch_train_ce')):
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
