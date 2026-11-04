"""Prepare LLaDA files from Hugging Face or a local directory; weights are optional."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

METADATA_PATTERNS = ['*.json', '*.py', '*.model', '*.txt', '*.tiktoken']
CODE_FILES = ('modeling_llada.py', 'configuration_llada.py')


def resolve_source(source, revision, metadata_only):
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.is_absolute() or source.startswith(('./', '../', '~')):
        raise FileNotFoundError(local)
    from huggingface_hub import snapshot_download
    patterns = METADATA_PATTERNS if metadata_only else METADATA_PATTERNS + ['*.safetensors', '*.bin']
    return Path(snapshot_download(source, revision=revision, allow_patterns=patterns)).resolve()


def validate_weights(source):
    for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = source / index_name
        if index.is_file():
            names = set(json.loads(index.read_text())['weight_map'].values())
            if not names:
                raise ValueError(f'Empty weight index: {index}')
            missing = [name for name in names if not (source / name).is_file()]
            if missing:
                raise FileNotFoundError(f'Missing weight shards: {missing}')
            return
    if not any((source / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        raise FileNotFoundError('No complete model weights; use --metadata-only for tiny-model checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='GSAI-ML/LLaDA-1.5', help='HF repo ID or local model directory')
    parser.add_argument('--revision', default='main', help='HF branch/tag/commit; prefer a commit for fixed experiments')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/llada15'))
    parser.add_argument('--metadata-only', action='store_true', help='Fetch config/code/tokenizer without 8B weights')
    parser.add_argument('--copy-weights', action='store_true', help='Copy weights instead of linking local/cache files')
    parser.add_argument('--code-source', default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Fallback code repo/directory, used only if source lacks model Python files')
    parser.add_argument('--code-revision', default='main')
    args = parser.parse_args()
    source = resolve_source(args.source, args.revision, args.metadata_only)
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'llada':
        raise ValueError('Expected a LLaDA checkpoint')
    if not (source / 'tokenizer_config.json').is_file():
        raise FileNotFoundError('Missing tokenizer_config.json')
    code_source = source
    if not all((source / name).is_file() for name in CODE_FILES):
        code_source = resolve_source(args.code_source, args.code_revision, True)
        fallback_config = json.loads((code_source / 'config.json').read_text())
        # Compare architecture, not incidental version/dtype metadata.
        keys = ('model_type', 'd_model', 'n_layers', 'n_heads', 'n_kv_heads', 'block_type',
                'block_group_size', 'mlp_hidden_size', 'activation_type', 'rope', 'alibi',
                'layer_norm_type', 'weight_tying', 'embedding_size', 'vocab_size')
        mismatches = [key for key in keys if config.get(key) != fallback_config.get(key)]
        if mismatches:
            raise ValueError(f'Fallback code checkpoint has incompatible architecture: {mismatches}')
    for name in CODE_FILES:
        if not (code_source / name).is_file():
            raise FileNotFoundError(code_source / name)
    if not args.metadata_only:
        validate_weights(source)
    output = args.output.resolve()
    if output == source:
        raise ValueError('Choose an output directory different from the source')
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix not in ('.json', '.py', '.model', '.txt', '.tiktoken', '.safetensors', '.bin'):
            continue
        if path.name == 'provenance.json':
            continue
        weight = path.suffix in ('.safetensors', '.bin')
        if weight and args.metadata_only:
            continue
        target = output / path.name
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                if target.resolve() != path.resolve():
                    raise FileExistsError(f'Conflicting existing link: {target}')
            elif weight:
                raise FileExistsError(f'Existing weight file; choose a fresh output directory: {target}')
            elif target.read_bytes() != path.read_bytes():
                raise FileExistsError(f'Conflicting existing metadata: {target}')
        elif weight and not args.copy_weights:
            target.symlink_to(path.resolve())
        else:
            shutil.copyfile(path, target)
        files[path.name] = dict(bytes=target.stat().st_size, linked=target.is_symlink())
    for name in CODE_FILES:
        data = (code_source / name).read_bytes()
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f'Conflicting existing model code: {target}')
        if not target.exists():
            target.write_bytes(data)
        files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    # Prevent a successful command from leaving an unusable complete-checkpoint view.
    if not args.metadata_only:
        validate_weights(output)
    manifest = dict(source=args.source, revision=args.revision, resolved_source=str(source),
                    code_source=str(code_source), metadata_only=args.metadata_only,
                    copy_weights=args.copy_weights, files=files)
    (output / 'provenance.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(output=str(output), metadata_only=args.metadata_only), indent=2))


if __name__ == '__main__':
    main()
