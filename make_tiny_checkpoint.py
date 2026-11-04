"""Small random model with the REAL tokenizer/vocabulary for CLI integration tests."""
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.cache/modules'))
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'checkpoints/llada15')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/tiny-random-llada')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source, dest = args.source.resolve(), args.output.resolve()
    if dest.exists():
        raise FileExistsError(dest)
    config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    for name, value in dict(d_model=32, n_layers=2, n_heads=4, n_kv_heads=4,
                            mlp_hidden_size=64, max_sequence_length=512, init_device='cpu').items():
        setattr(config, name, value)
    model = AutoModel.from_config(config, trust_remote_code=True, init_params=True)
    # The LLaDA module self-registers with AutoModel; explicitly request copying
    # custom model code when saving, otherwise a fresh process cannot reload it.
    model.register_for_auto_class('AutoModel')
    config.register_for_auto_class('AutoConfig')
    model.save_pretrained(dest)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.save_pretrained(dest)
    print(dest, flush=True)


if __name__ == '__main__':
    main()
