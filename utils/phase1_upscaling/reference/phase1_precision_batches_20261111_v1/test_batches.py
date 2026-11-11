import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import common as C
import scope
import publish as P
import driver as D

class Batches(unittest.TestCase):
    def fixture(self,index=0):
        root=scope.PILOT/'output'
        if not root.exists():root=C.CODE.parent/'phase1_precision_recovery_20261111_v1/results/output'
        p=root/scope.PILOT_KEYS[index]
        return p/'verified_core.zip',C.read(p/'work/input/record.json'),C.read(p/'QUALIFIED_PASS.json')
    def test_disjoint_batches(self):
        batches=[scope.batch_keys(str(i)) for i in range(5)]
        self.assertEqual([len(x) for x in batches],[8,8,8,8,3])
        self.assertEqual([x for b in batches for x in b],scope.NEW_KEYS)
        self.assertEqual(len(set(scope.NEW_KEYS)),35)
        self.assertFalse(set(scope.NEW_KEYS)&set(scope.PILOT_KEYS))
        self.assertFalse(any(k.endswith('core229') for k in scope.TARGETS))
    def test_scope_categories(self):
        self.assertEqual(len(scope.TARGETS),38)
        self.assertTrue(all(r['category']=='stable_reference_plateau' and r['prior_scientific_failures']==2 for r in scope.SCOPE['rows'].values()))
    def test_invalid_batch(self):
        for value in ('-1','5','all',0):
            with self.assertRaises(RuntimeError):scope.batch_keys(value)
    def test_pilot_no_recalculation(self):
        with self.assertRaises(RuntimeError):D.preflight(scope.PILOT_KEYS[0])
    def test_existing_work_protected(self):
        with patch.object(Path,'exists',return_value=True):
            with self.assertRaises(RuntimeError):D.preflight(scope.NEW_KEYS[0])
    def test_live_campaign_blocks(self):
        with patch.object(C,'read',return_value=dict(active=['active'],cursor=2592)):
            with self.assertRaises(RuntimeError):P.stopped()
    def test_missing_qualification_blocks(self):
        with patch.object(C,'read',return_value=dict(passed=False,source='bad')):
            with self.assertRaises(RuntimeError):D.compute_gate()
    def test_three_real_archives(self):
        for i in range(3):
            p,r,q=self.fixture(i);self.assertEqual(P.verify_archive(p,r,q),q['receipt'])
    def test_changed_source_rejected(self):
        p,r,q=self.fixture();q['source']='wrong'
        with self.assertRaises(RuntimeError):P.verify_archive(p,r,q)
    def test_changed_draw_rejected(self):
        p,r,q=self.fixture();r['slot']+=1
        with self.assertRaises(RuntimeError):P.verify_archive(p,r,q)
    def test_missing_independent_audit(self):
        p,r,q=self.fixture();q['saved_precision_reaudit']['passed']=False
        with self.assertRaises(RuntimeError):P.verify_archive(p,r,q)
    def test_repeated_call_rejected(self):
        p,r,q=self.fixture();q['completed_calls_repeated']=1
        with self.assertRaises(RuntimeError):P.verify_archive(p,r,q)
    def test_idempotent_promotion(self):
        p,r,q=self.fixture()
        with tempfile.TemporaryDirectory(dir=C.CODE) as root:
            target=Path(root)/'core.zip';P.promote(p,target,r,q);fp=P.fingerprint(target)
            P.promote(p,target,r,q);self.assertEqual(P.fingerprint(target),fp)
    def test_no_overwrite(self):
        p,r,q=self.fixture()
        with tempfile.TemporaryDirectory(dir=C.CODE) as root:
            target=Path(root)/'core.zip';target.write_bytes(b'keep')
            with self.assertRaises(RuntimeError):P.promote(p,target,r,q)
            self.assertEqual(target.read_bytes(),b'keep')
    def test_publisher_cannot_solve(self):
        text=(C.CODE/'publish.py').read_text()
        for token in ('sbatch','MATLAB','R.run','StrictProbe'):self.assertNotIn(token,text)
    def test_numerical_code_identical(self):
        parent=scope.PILOT if scope.PILOT.exists() else C.CODE.parent/'phase1_precision_recovery_20261111_v1'
        for n in ('AccurateAudit.jl','PrecisionRecovery.jl','recover_flow.jl','StrictProbe.jl','ResumeContract.jl'):
            self.assertEqual(C.sha(C.CODE/n),C.sha(parent/n))

if __name__=='__main__':unittest.main()
