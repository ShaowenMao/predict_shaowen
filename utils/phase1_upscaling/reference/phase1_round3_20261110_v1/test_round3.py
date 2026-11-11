import contextlib
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C
import recovery as R
import slurm as S
import timeout_process as T
import test_window as previous_tests


class RoundThreeTests(unittest.TestCase):
    setUp = previous_tests.WindowTests.setUp
    pump_fixture = previous_tests.WindowTests.pump_fixture
    test_more_than_eight_cases = previous_tests.WindowTests.test_more_than_eight_cases_without_raising_worker_ceiling
    test_exact_128 = previous_tests.WindowTests.test_exact_128_is_admitted_but_129_is_not
    test_tails_remain_charged = previous_tests.WindowTests.test_all_original_packets_remain_charged_during_tail
    test_128_single_core_workers = previous_tests.WindowTests.test_small_cases_still_cannot_exceed_128_workers
    test_shared_task_limit = previous_tests.WindowTests.test_shared_submitted_task_guard_unchanged
    test_terminal_case_not_repeated = previous_tests.WindowTests.test_terminal_case_frees_capacity_without_resubmitting_it

    def test_four_hour_budget_retains_finalization_and_exit_reserve(self):
        self.assertEqual(T.Budget(14400, clock=lambda: 0).limit(), 13770)
        self.assertEqual(T.Budget(14400, clock=lambda: 7200).limit(), 6570)
        self.assertEqual(T.Budget(14400, clock=lambda: 13770).limit(finalizing=True), 600)
        with self.assertRaises(T.BudgetExhausted):
            T.Budget(14400, clock=lambda: 13770).limit()

    def test_finish_window_not_widened(self):
        state, _, _ = self.pump_fixture([1]*12, 8, 'finish')
        self.assertEqual(state['cursor'], 8)

    def test_packets_do_not_hide_other_cores_behind_a_straggler(self):
        rows = [dict(slot=i, category='interrupted' if i%2 else 'unstarted') for i in range(1, 40)]
        records = [dict(slot=i) for i in range(1, 40)]
        packets = R.packetize(rows, records, 'solve')
        self.assertEqual(packets, [[r] for r in records])

    def prior_fixture(self, root):
        case = 's01_c003_pc06_ca40_IND_001'; record = dict(case_id=case, slot=1); key=C.key(record)
        t=root/'solve/cases'/case/'TERMINAL.json'
        C.write(t,dict(job='100',tasks={'0':['COMPLETED','0:0']}))
        claim=root/'numerical_claims'/(key+'.json');C.write(claim,dict(record_sha256=C.digest(record)))
        e=dict(original_work=str(root/'solve/runs'/case/'cores'/key),original_job='100',original_packet=0,
               original_state=['COMPLETED','0:0'],terminal_sha256=C.sha(t),previous_claim_sha256=C.sha(claim))
        return record,e,t,claim

    def test_previous_claim_and_terminal_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r,e,t,claim=self.prior_fixture(root);before={p:p.read_bytes() for p in (t,claim)}
            with patch.object(R,'PREVIOUS',root):self.assertEqual(R.validate_prior(e,r),Path(e['original_work']))
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_prior_task_path_claim_or_terminal_change_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r,e,t,claim=self.prior_fixture(root)
            for key,value in [('original_work',str(root/'foreign')),('terminal_sha256','changed'),
                              ('original_job','200'),('original_state',['RUNNING','0:0']),('previous_claim_sha256',None)]:
                changed=dict(e,**{key:value})
                with patch.object(R,'PREVIOUS',root),self.assertRaises(RuntimeError):R.validate_prior(changed,r)

    def test_new_label_does_not_allow_uncertain_resubmission(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'sha',return_value='source'):
            root=Path(d)
            with patch.object(C,'ROOT',root):
                w=S.spec('r3s_case','worker',['case'],array='0',cpus=2,memory=18432,minutes=240)
                C.write(root/'submissions/r3s_case/INTENT.json',dict(spec=w,created='2026-11-10T00:00:00'))
                with patch.object(S,'command',return_value='') as command,self.assertRaisesRegex(RuntimeError,'Uncertain'):
                    S.held(w)
                self.assertTrue(all(call.args[0][0]!='sbatch' for call in command.call_args_list))

    def test_launch_uses_expected_allocation_and_afterany_gate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'ROOT',Path(d)),patch.object(C,'sha',return_value='source'), \
             patch.object(R,'checked_plan',return_value=dict(packets=[[1],[2]])), \
             patch.object(S,'held',side_effect=['100','101']) as held,patch.object(S,'release') as release:
            R.launch('case',dict(phase='solve'))
            w,g=[call.args[0] for call in held.call_args_list]
            self.assertEqual((w['cpus'],w['memory'],w['minutes'],w['array']),(2,18432,240,'0-1'))
            self.assertEqual(g['dependency'],'100')
            self.assertEqual([call.args[0] for call in release.call_args_list],['101','100'])

    def test_old_scope_not_authorized_as_round_three(self):
        with self.assertRaisesRegex(RuntimeError,'Invalid recovery phase'):
            R.authorize_recovery(dict(purpose='round2',phase='solve'))

    def test_missing_saved_archive_blocks_numerical_release(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);C.write(root/'finish/SCOPE.json',dict(cases=['case'],cores=1))
            with patch.object(R,'HOME',root),patch.object(R,'PRODUCTION',root/'production'),patch.object(C,'ROOT',root/'finish'), \
                 patch.object(C,'locked',lambda p:contextlib.nullcontext()),patch.object(C,'read',wraps=C.read) as read, \
                 patch.object(R,'checked_plan',return_value=dict(packets=[[dict(slot=1)]])),patch.object(S,'held') as held:
                real_read=C.read
                read.side_effect=lambda p: {'method_sha256':'method'} if Path(p).name=='LOCK.json' else __import__('json').loads(Path(p).read_text())
                with self.assertRaisesRegex(RuntimeError,'Saved output not accepted'):R.next_phase('finish')
                held.assert_not_called()

    def test_scientific_worker_and_slurm_helpers_unchanged(self):
        parent=C.read(C.CODE/'ROUND2_PARENT_LOCK.json')
        for name in ('worker.py','flow_packet.jl','prepare_packet.m','native_packet.jl','export_packet.py','allocation_guard.py','slurm.py'):
            self.assertEqual(C.sha(C.CODE/name),parent['code'][name])


if __name__ == '__main__':
    unittest.main()
