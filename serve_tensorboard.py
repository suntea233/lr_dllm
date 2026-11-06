"""Serve explicitly named experiment logs without moving training artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'tensorboard_runs.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    content = manifest_path.read_bytes()
    manifest = json.loads(content)
    entries, names, sources = [], set(), set()
    for entry in manifest['runs']:
        name = entry['display_name']
        source = (manifest_path.parent / entry['events']).resolve(strict=True)
        label = Path(name)
        if not name or label.is_absolute() or '..' in label.parts or str(label) != name:
            raise ValueError(f'Invalid relative display name: {name!r}')
        if name in names or source in sources:
            raise ValueError(f'Duplicate display name or event directory: {name}')
        if not source.is_dir() or not any(source.glob('events.out.tfevents.*')):
            raise ValueError(f'No TensorBoard event files: {source}')
        names.add(name)
        sources.add(source)
        entries.append((label, source))
    if not entries:
        raise ValueError('The run manifest is empty')
    # Separate, content-addressed views avoid changing original event directories.
    # Directory aliases also avoid logdir_spec appending '/.' to every run name.
    view = manifest_path.parent / '.tensorboard_views' / hashlib.sha256(content).hexdigest()[:16]
    for label, source in entries:
        alias = view / label
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            if not alias.is_symlink() or alias.resolve() != source:
                raise FileExistsError(f'Refusing to replace an unexpected view entry: {alias}')
        else:
            alias.symlink_to(source, target_is_directory=True)
    command = [sys.executable, '-m', 'tensorboard.main', '--logdir', str(view),
               '--host', args.host, '--port', str(args.port), '--load_fast', 'false',
               '--reload_interval', '15', '--samples_per_plugin', 'scalars=100000',
               '--window_title', 'lr_dllm | GSM8K | LLaDA-1.5 8B']
    if args.print_command:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
