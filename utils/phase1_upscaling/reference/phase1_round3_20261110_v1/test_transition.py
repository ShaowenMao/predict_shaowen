import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C
import adoption as A
import control
import planning
import transition_operator as T

class Transition(unittest.TestCase):
    def test_science_and_worker_body_identical_to_qualified_v4(self):
        old = C.CODE.parent / 'phase1_pc40_adaptive10_runner_20261030_v4'
        oldlock = C.read(old / 'LOCK.json')
        newlock = C.read(C.CODE / 'LOCK.json')
        self.assertEqual(newlock['science'], oldlock['science'])
        self.assertEqual(newlock['science_bundle_sha256'], oldlock['science_bundle_sha256'])
        for n in ('flow_packet.jl', 'cache_contract.py', 'common.py', 'planning.py', 'slurm.py', 'prepare_packet.m', 'export_packet.py', 'native_packet.jl'):
            self.assertEqual(C.sha(C.CODE / n), oldlock['code'][n], n)
        s = (C.CODE / 'worker.py').read_text().replace("    from adoption import route_legacy_worker\n    route_legacy_worker('worker', sys.argv[1])\n", '')
        self.assertEqual(s, (old / 'worker.py').read_text())
    def test_new_launcher_does_not_use_mutable_run(self):
        s = (C.CODE / 'job.sh').read_text()
        self.assertNotIn('/RUN.json', s)
        self.assertIn('CODE="/orcd/data/juanes/001/shaowen/' + C.CODE.name + '"', s)
    def test_unknown_legacy_job_rejected(self):
        with patch.object(A, 'legacy_assignment', return_value={'launch': {'job': '12'}}):
            with self.assertRaises(RuntimeError):
                A.legacy_target('worker', 'case', '13', '99')
            with self.assertRaises(RuntimeError):
                A.legacy_target('control', 'case', '12', '99')
    def test_new_case_not_legacy(self):
        with patch.object(A, 'legacy_assignment', return_value=None):
            self.assertIsNone(A.legacy_target('worker', 'new', '13', '99'))
    def test_legacy_case_does_not_plan_or_resubmit(self):
        with patch.object(A, 'preserve_legacy_assignment', return_value=True), patch.object(planning, 'prepare_plan') as prep, patch.object(control.S, 'held') as submit:
            control.launch({'purpose': 'production'}, 'legacy')
            prep.assert_not_called()
            submit.assert_not_called()
    def test_new_case_uses_existing_launch_path(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)), patch.object(A, 'preserve_legacy_assignment', return_value=False), patch.object(planning, 'prepare_plan') as prep, patch.object(control.S, 'held', side_effect=['12', '13']) as submit, patch.object(control.S, 'release') as release:
            control.launch({'purpose': 'production'}, 'new')
            self.assertEqual(submit.call_count, 2)
            self.assertEqual(release.call_count, 2)
            self.assertEqual(submit.call_args_list[0].args[0]['array'], '0-16')
            self.assertEqual(submit.call_args_list[0].args[0]['memory'], 18432)
            prep.assert_called_once()
    def test_original_authorization_needs_adoption(self):
        with patch.object(A, 'record', return_value=None):
            self.assertFalse(A.authorized_source({'source_sha256': 'old'}, {}))
    def test_legacy_record_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)):
            p = Path(td) / 'ack.json'
            C.write(p, {'job': '12'})
            a = {'legacy_cases': {'old': {'files': {'ack.json': C.sha(p)}}}}
            with patch.object(A, 'record', return_value=a):
                self.assertIsNotNone(A.legacy_assignment('old'))
                C.write(p, {'job': '13'}, replace=True)
                with self.assertRaises(RuntimeError):
                    A.legacy_assignment('old')
    def test_rolling_replaces_one_not_whole_window(self):
        state = dict(cursor=8, active=list(range(8)))
        launched = []
        planning.advance(state, list(range(12)), lambda c: c == 3, launched.append, lambda s: None)
        self.assertEqual(len(state['active']), 8)
        self.assertEqual(state['cursor'], 9)
        self.assertEqual(state['active'], [0, 1, 2, 4, 5, 6, 7, 8])
    def test_only_pending_handoff_is_held(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)):
            jobs = {'1': {'spec': {'mode': 'control'}}, '2': {'spec': {'mode': 'worker'}}}
            live = {'1': dict(state='PENDING', reason='Dependency'), '2': dict(state='RUNNING', reason='None')}
            calls = []
            def command(args):
                calls.append(args)
                live['1']['reason'] = 'JobHeldUser'
                return ''
            with patch.object(T, 'submissions', return_value=jobs), patch.object(T, 'queue', side_effect=lambda: copy.deepcopy(live)), patch.object(T.S, 'command', side_effect=command):
                T.hold_handoffs()
            self.assertEqual(calls, [['scontrol', 'hold', '1']])
    def test_existing_user_hold_is_not_claimed(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)), patch.object(T, 'submissions', return_value={'1': {'spec': {'mode': 'control'}}}), patch.object(T, 'queue', return_value={'1': dict(state='PENDING', reason='JobHeldUser')}), patch.object(T.S, 'command') as cmd:
            with self.assertRaises(RuntimeError):
                T.hold_handoffs()
            cmd.assert_not_called()
    def test_running_gate_is_not_stopped(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)), patch.object(T, 'submissions', return_value={'1': {'spec': {'mode': 'control'}}}), patch.object(T, 'queue', return_value={'1': dict(state='RUNNING', reason='None')}), patch.object(T.time, 'sleep'), patch.object(T.S, 'command') as cmd:
            with self.assertRaisesRegex(RuntimeError, 'No safe handoff'):
                T.hold_handoffs()
            cmd.assert_not_called()
    def test_release_only_our_gate_holds(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)):
            C.write(Path(td) / 'cache_transition_gate_holds/1.json', {'job': '1'})
            live = {'1': dict(state='PENDING', reason='JobHeldUser'), '2': dict(state='PENDING', reason='JobHeldUser')}
            with patch.object(T, 'queue', return_value=live), patch.object(T.S, 'command') as cmd:
                T.release_handoffs()
                cmd.assert_called_once_with(['scontrol', 'release', '1'])
    def test_unacknowledged_submission_blocks_cutover(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)):
            C.write(Path(td) / 'submissions/x/INTENT.json', {'spec': {}})
            with self.assertRaises(FileNotFoundError):
                T.submissions()
    def test_conflicting_submission_blocks_cutover(self):
        with tempfile.TemporaryDirectory() as td, patch.object(C, 'ROOT', Path(td)):
            C.write(Path(td) / 'submissions/x/INTENT.json', {'spec': {'a': 1}})
            C.write(Path(td) / 'submissions/x/ACK.json', {'job': '1', 'spec': {'a': 2}})
            with self.assertRaises(RuntimeError):
                T.submissions()

if __name__ == '__main__':
    unittest.main()
