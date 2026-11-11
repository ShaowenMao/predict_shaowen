import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import allocation_guard as A
import common as C
import worker
from timeout_process import Budget, BudgetExhausted


ENV = dict(SLURM_JOB_ID='100', SLURM_ARRAY_JOB_ID='100', SLURM_ARRAY_TASK_ID='1')
OWN = 'JobId=100 ArrayJobId=100 ArrayTaskId=1 JobState=RUNNING EndTime=2026-11-09T10:30:00'
SIBLING = 'JobId=101 ArrayJobId=100 ArrayTaskId=0 JobState=COMPLETED EndTime=2026-11-09T09:00:00'


class AllocationTests(unittest.TestCase):
    def test_completed_sibling_does_not_supply_deadline(self):
        raw = OWN + '\n' + SIBLING
        old = dict(x.split('=', 1) for x in raw.split() if '=' in x)
        self.assertEqual(old['JobState'], 'COMPLETED')
        self.assertEqual(A.select_record(raw, ENV)['EndTime'], '2026-11-09T10:30:00')

    def test_record_order_does_not_matter(self):
        self.assertEqual(A.select_record(OWN+'\n'+SIBLING, ENV), A.select_record(SIBLING+'\n'+OWN, ENV))

    def test_exact_array_query(self):
        calls = []
        def query(args):
            calls.append(args)
            return OWN
        self.assertEqual(A.read_allocation(ENV, query)['JobId'], '100')
        self.assertEqual(calls, [['scontrol', 'show', 'job', '-o', '100_1']])

    def test_nonarray_query(self):
        self.assertEqual(A.read_allocation({'SLURM_JOB_ID':'42'}, lambda a: 'JobId=42 JobState=RUNNING')['JobId'], '42')

    def test_missing_or_duplicate_record_rejected(self):
        for raw in (SIBLING, OWN+'\n'+OWN):
            with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                A.select_record(raw, ENV)

    def test_wrong_array_identity_rejected(self):
        for raw in (OWN.replace('ArrayTaskId=1', 'ArrayTaskId=0'), OWN.replace('ArrayJobId=100', 'ArrayJobId=99'), OWN.replace('ArrayTaskId=1', 'ArrayTaskId=0-1')):
            with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
                A.select_record(raw, ENV)

    def test_terminal_exact_task_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'not running'):
            A.select_record(OWN.replace('RUNNING', 'COMPLETED'), ENV)

    def test_missing_malformed_environment_rejected(self):
        for env in ({}, {'SLURM_JOB_ID':'x'}, {'SLURM_JOB_ID':'100','SLURM_ARRAY_JOB_ID':'100'}, dict(ENV,SLURM_ARRAY_TASK_ID='1;bad')):
            with self.assertRaises(RuntimeError):
                A.read_allocation(env, lambda _: self.fail('Query must not run'))

    def test_single_line_combination_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Malformed'):
            A.select_record(OWN+' '+SIBLING, ENV)

    def test_array_not_allowed_without_array_context(self):
        with self.assertRaisesRegex(RuntimeError, 'Unexpected array'):
            A.select_record(OWN, {'SLURM_JOB_ID':'100'})

    def test_resource_checks_still_apply(self):
        fields = dict(Account='wrong', QOS='mit_amf_advanced_cpu', Partition='mit_normal', NumCPUs='2', Requeue='0', MinMemoryNode='18G')
        with patch.object(A, 'read_allocation', return_value=fields):
            with self.assertRaisesRegex(RuntimeError, 'Allocation Account'):
                worker.allocation()

    def test_time_reserve_and_expiry_unchanged(self):
        self.assertEqual(Budget(1800, clock=lambda:0).limit(), 1170)
        self.assertEqual(Budget(1800, clock=lambda:0).limit(finalizing=True), 1770)
        with self.assertRaises(BudgetExhausted):
            Budget(1800, clock=lambda:1771).limit(finalizing=True)

    def test_only_worker_allocation_lookup_changed(self):
        old = "    raw=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)\n    d=dict(x.split('=',1) for x in raw.split() if '=' in x)\n"
        new = '    from allocation_guard import read_allocation\n    d=read_allocation()\n'
        text = (C.CODE/'worker.py').read_text()
        self.assertEqual(text.count(new), 1)
        parent = json.loads((C.CODE/'GUARD_PARENT_LOCK.json').read_text())
        self.assertEqual(hashlib.sha256(text.replace(new,old).encode()).hexdigest(), parent['code']['worker.py'])
        self.assertEqual(C.sha(C.CODE/'timeout_process.py'), parent['code']['timeout_process.py'])


if __name__ == '__main__':
    unittest.main()
