"""CPU-only launcher checks; no NVIDIA queries or real worker processes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from launch_when_free import query_gpus, run_job, validate_spec


class Clock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def gpu(index, free):
    return dict(index=index, uuid=f'GPU-test-{index}', free_gib=free, total_gib=80., utilization=100.)


class LauncherChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = dict(argv=[sys.executable, '-c', 'pass'], cwd=str(self.root),
                         env={'HF_HUB_OFFLINE': '1'}, candidate_gpus=[2, 3], min_free_gib=24,
                         poll_seconds=30, stable_polls=3, max_wait_seconds=180)
        self.clock = Clock()

    def run_mock(self, query, popen, output=None):
        return run_job(self.spec, output or self.root / 'launch', query=query, popen=popen,
                       monotonic=self.clock.monotonic, sleep=self.clock.sleep)

    def test_insufficient_memory_times_out_without_launch(self):
        query, popen = Mock(return_value=[gpu(2, 20), gpu(3, 23)]), Mock()
        self.assertEqual(self.run_mock(query, popen), 124)
        popen.assert_not_called()
        self.assertEqual(query.call_count, 6)
        state = json.loads((self.root / 'launch/state.json').read_text())
        self.assertEqual(state['phase'], 'timed_out')
        self.assertFalse((self.root / 'launch/process.json').exists())

    def test_three_consecutive_readings_launch_exactly_once_and_record_exit(self):
        process = Mock(pid=12345)
        process.poll.side_effect = [None, 0]
        popen = Mock(return_value=process)
        query = Mock(side_effect=[[gpu(2, 25), gpu(3, 26)], [gpu(2, 25), gpu(3, 26)],
                                  [gpu(2, 25), gpu(3, 26)]])
        self.assertEqual(self.run_mock(query, popen), 0)
        popen.assert_called_once()
        self.assertEqual(query.call_count, 3)
        call = popen.call_args.kwargs
        self.assertEqual(call['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test-3')
        self.assertEqual(call['env']['HF_HUB_OFFLINE'], '1')
        self.assertTrue(call['start_new_session'])
        self.assertEqual(json.loads((self.root / 'launch/process.json').read_text())['exit_code'], 0)
        with self.assertRaises(FileExistsError):
            self.run_mock(query, popen)
        popen.assert_called_once()

    def test_low_reading_resets_stability_and_worker_failure_does_not_retry(self):
        process = Mock(pid=999)
        process.poll.return_value = 7
        popen = Mock(return_value=process)
        query = Mock(side_effect=[[gpu(2, value)] for value in [25, 25, 20, 25, 25, 25]])
        self.assertEqual(self.run_mock(query, popen), 1)
        self.assertEqual(query.call_count, 6)
        popen.assert_called_once()
        state = json.loads((self.root / 'launch/state.json').read_text())
        self.assertEqual((state['phase'], state['exit_code'], state['retries']), ('failed', 7, 0))

    def test_query_failure_resets_stability_and_still_times_out(self):
        query = Mock(side_effect=OSError('nvidia unavailable'))
        popen = Mock()
        self.assertEqual(self.run_mock(query, popen), 124)
        popen.assert_not_called()
        self.assertEqual(query.call_count, 6)

    def test_two_gpu_launch_waits_until_both_are_stable(self):
        self.spec['gpu_count'] = 2
        process = Mock(pid=123)
        process.poll.return_value = 0
        popen = Mock(return_value=process)
        query = Mock(side_effect=[[gpu(2, 25), gpu(3, free)] for free in [20, 20, 26, 26, 26]])
        self.assertEqual(self.run_mock(query, popen), 0)
        self.assertEqual(query.call_count, 5)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-test-2,GPU-test-3')
        metadata = json.loads((self.root / 'launch/process.json').read_text())
        self.assertEqual([g['index'] for g in metadata['gpus']], [2, 3])

    def test_bounded_spec_and_launcher_owns_cuda_selection(self):
        for update in [{'max_wait_seconds': 86401}, {'stable_polls': 2},
                       {'env': {'CUDA_VISIBLE_DEVICES': '2'}}, {'min_free_gib': float('nan')},
                       {'gpu_count': 3}, {'gpu_count': 2, 'candidate_gpus': [2]}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_spec(dict(self.spec, **update))

    def test_nvidia_output_is_parsed_without_running_it(self):
        result = Mock(stdout='2, GPU-abc, 25600, 81920, 100\n3, GPU-def, 24064, 81920, N/A\n')
        with patch('launch_when_free.subprocess.run', return_value=result) as run:
            values = query_gpus()
        run.assert_called_once()
        self.assertEqual(values[0]['free_gib'], 25)
        self.assertEqual(values[1]['utilization'], None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
