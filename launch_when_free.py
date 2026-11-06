"""Wait for one or two GPUs, launch one detached worker once, and record its exit.

This is a free-memory launch gate, not a GPU reservation or allocator limit.
The worker command must implement any desired per-process allocator cap itself.
Run this supervisor under nohup or with start_new_session=True for persistence.
"""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


def timestamps():
    now = datetime.now(ZoneInfo('UTC'))
    return {'utc': now.isoformat(), 'beijing': now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def validate_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError('The spec must be a JSON object')
    allowed = {'argv', 'cwd', 'env', 'candidate_gpus', 'gpu_count', 'min_free_gib',
               'poll_seconds', 'stable_polls', 'max_wait_seconds'}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f'Unknown spec fields: {sorted(unknown)}')
    argv = spec.get('argv')
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or '\x00' in v for v in argv):
        raise ValueError('argv must be a nonempty list of NUL-free strings')
    if not argv[0]:
        raise ValueError('argv[0] must not be empty')
    cwd = Path(spec.get('cwd', '.')).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError('cwd must be a directory')
    env = spec.get('env', {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                      or not k or '=' in k or '\x00' in k or '\x00' in v
                                      for k, v in env.items()):
        raise ValueError('env must map valid environment variable names to strings')
    if 'CUDA_VISIBLE_DEVICES' in env:
        raise ValueError('Use candidate_gpus; the launcher sets CUDA_VISIBLE_DEVICES to the selected UUID')
    candidates = spec.get('candidate_gpus')
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('candidate_gpus must be a nonempty list of indices or GPU UUIDs')
    normalized = []
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError('GPU candidates must be nonnegative indices or GPU UUIDs')
        value = str(value)
        if not (value.isdigit() or (value.startswith('GPU-') and len(value) > 4)):
            raise ValueError(f'Invalid GPU candidate: {value!r}')
        normalized.append(value)
    def positive(name, default, maximum=None):
        value = spec.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a positive number')
        if not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
            raise ValueError(f'{name} must be positive' + (f' and at most {maximum}' if maximum else ''))
        return float(value)
    stable = spec.get('stable_polls', 3)
    if isinstance(stable, bool) or not isinstance(stable, int) or stable < 3:
        raise ValueError('stable_polls must be an integer >= 3')
    gpu_count = spec.get('gpu_count', 1)
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count not in (1, 2):
        raise ValueError('gpu_count must be 1 or 2')
    if len(set(normalized)) < gpu_count:
        raise ValueError('candidate_gpus contains fewer unique candidates than gpu_count')
    return dict(argv=list(argv), cwd=str(cwd), env=dict(env), candidate_gpus=normalized,
                gpu_count=gpu_count,
                min_free_gib=positive('min_free_gib', None),
                poll_seconds=positive('poll_seconds', 30, 60), stable_polls=stable,
                max_wait_seconds=positive('max_wait_seconds', 86400, 86400))


def query_gpus():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=15)
    gpus = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f'Unexpected nvidia-smi field count: {len(row)}')
        index, uuid, free, total, utilization = [part.strip() for part in row]
        free_gib, total_gib = float(free) / 1024, float(total) / 1024
        if not all(math.isfinite(v) for v in (free_gib, total_gib)):
            raise ValueError('Non-finite GPU memory reading')
        gpus.append(dict(index=int(index), uuid=uuid, free_gib=free_gib, total_gib=total_gib,
                         utilization=float(utilization) if utilization.isdigit() else None))
    return gpus


def run_job(spec, output_dir, *, query=query_gpus, popen=subprocess.Popen,
            monotonic=time.monotonic, sleep=time.sleep):
    spec = validate_spec(spec)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'supervisor.lock').open('a') as lock:
        # Both a lock and a persistent exclusive marker are required: even a
        # completed or timed-out supervisor cannot launch again in this directory.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / 'started.json').open('x') as marker:
            json.dump(dict(supervisor_pid=os.getpid(), **timestamps()), marker, indent=2)
        with (output / 'supervisor.log').open('x') as log:
            def record(phase, **fields):
                payload = dict(phase=phase, supervisor_pid=os.getpid(), **timestamps(), **fields)
                write_json(output / 'state.json', payload)
                log.write(json.dumps(payload, allow_nan=False) + '\n')
                log.flush()
                return payload

            # Do not copy arbitrary environment values into process metadata.
            public_spec = {k: v for k, v in spec.items() if k != 'env'}
            public_spec['env_override_keys'] = sorted(spec['env'])
            public_spec['allocator_cap_managed_by_launcher'] = False
            public_spec['spec_sha256'] = hashlib.sha256(
                json.dumps(spec, sort_keys=True).encode()).hexdigest()
            write_json(output / 'launch_plan.json', public_spec)
            started = monotonic()
            streaks = {}
            record('waiting', elapsed_wait_s=0, candidate_gpus=spec['candidate_gpus'],
                   gpu_count=spec['gpu_count'],
                   min_free_gib=spec['min_free_gib'], max_wait_seconds=spec['max_wait_seconds'])
            process = None
            try:
                while process is None:
                    elapsed = monotonic() - started
                    if elapsed >= spec['max_wait_seconds']:
                        record('timed_out', elapsed_wait_s=elapsed, launched=False)
                        return 124
                    try:
                        all_gpus = query()
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        streaks.clear()
                        elapsed = monotonic() - started
                        if elapsed >= spec['max_wait_seconds']:
                            record('timed_out', elapsed_wait_s=elapsed, launched=False,
                                   query_error=str(error))
                            return 124
                        record('waiting', elapsed_wait_s=elapsed, query_error=str(error),
                               stable_counts={}, launched=False)
                        sleep(min(spec['poll_seconds'], spec['max_wait_seconds'] - elapsed))
                        continue
                    candidates = [gpu for gpu in all_gpus
                                  if str(gpu['index']) in spec['candidate_gpus']
                                  or gpu['uuid'] in spec['candidate_gpus']]
                    present = {gpu['uuid'] for gpu in candidates}
                    streaks = {uuid: count for uuid, count in streaks.items() if uuid in present}
                    for gpu in candidates:
                        streaks[gpu['uuid']] = (streaks.get(gpu['uuid'], 0) + 1
                                               if gpu['free_gib'] >= spec['min_free_gib'] else 0)
                    eligible = [gpu for gpu in candidates
                                if streaks[gpu['uuid']] >= spec['stable_polls']]
                    elapsed = monotonic() - started
                    if elapsed >= spec['max_wait_seconds']:
                        record('timed_out', elapsed_wait_s=elapsed, launched=False)
                        return 124
                    if len(eligible) < spec['gpu_count']:
                        record('waiting', elapsed_wait_s=elapsed, gpus=candidates,
                               stable_counts=streaks, launched=False)
                        sleep(min(spec['poll_seconds'], spec['max_wait_seconds'] - elapsed))
                        continue
                    selected = sorted(sorted(eligible, key=lambda gpu: (-gpu['free_gib'], gpu['index']))
                                      [:spec['gpu_count']], key=lambda gpu: gpu['index'])
                    env = dict(os.environ, **spec['env'])
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu['uuid'] for gpu in selected)
                    # Exclusive launch intent survives a crash between Popen and
                    # writing process.json. Never automatically retry that case.
                    with (output / 'launch_intent.json').open('x') as handle:
                        json.dump(dict(gpus=selected, argv=spec['argv'], cwd=spec['cwd'],
                                       elapsed_wait_s=elapsed, **timestamps()), handle, indent=2)
                    with (output / 'worker.log').open('xb') as worker_log:
                        process = popen(spec['argv'], cwd=spec['cwd'], env=env,
                                        stdin=subprocess.DEVNULL, stdout=worker_log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                    metadata = dict(pid=process.pid, argv=spec['argv'], cwd=spec['cwd'],
                                    gpus=selected, cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                                    env_override_keys=sorted(spec['env']), **timestamps())
                    write_json(output / 'process.json', metadata)
                    record('running', worker_pid=process.pid, gpus=selected, elapsed_wait_s=elapsed)
                while True:
                    exit_code = process.poll()
                    if exit_code is not None:
                        metadata.update(exit_code=exit_code, finished=timestamps())
                        write_json(output / 'process.json', metadata)
                        record('completed' if exit_code == 0 else 'failed', worker_pid=process.pid,
                               gpus=selected, exit_code=exit_code, retries=0)
                        return 0 if exit_code == 0 else 1
                    sleep(spec['poll_seconds'])
            except BaseException as error:
                record('supervisor_error', error_type=type(error).__name__, error=str(error),
                       worker_pid=process.pid if process is not None else None,
                       worker_left_running=process is not None, retries=0)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True, help='JSON command and GPU launch conditions')
    parser.add_argument('--output', type=Path, required=True, help='Fresh supervisor records directory')
    parser.add_argument('--validate-only', action='store_true', help='Validate without creating files or querying GPUs')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.validate_only:
        validate_spec(spec)
        print('Launch specification is valid; no process started.')
        return
    raise SystemExit(run_job(spec, args.output))


if __name__ == '__main__':
    main()
