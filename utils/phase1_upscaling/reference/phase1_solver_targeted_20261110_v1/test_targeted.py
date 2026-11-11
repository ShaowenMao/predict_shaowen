import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import common as C
import scope
import publish as P
import submit as S
import driver as D

class Targeted(unittest.TestCase):
    def test_fixed_partition(self):
        batches=[scope.batch_keys(i) for i in range(20)]
        self.assertEqual(batches[0],scope.FIRST)
        self.assertEqual([k for b in batches for k in b],scope.TARGETS)
        self.assertTrue(all(len(b)<=8 for b in batches))
        self.assertFalse(set(scope.TARGETS)&scope.EXCLUDED)
    def test_invalid_batch(self):
        for i in [-1,20]:
            with self.assertRaises(RuntimeError):scope.batch_keys(i)
    def test_first_three_gate(self):
        with patch.object(C,'read',return_value=dict(all_passed=False,published=[])):
            with self.assertRaises(RuntimeError):S.admitted_keys(1)
    def test_source_gate(self):
        with patch.object(C,'read',return_value=dict(all_passed=True,published=scope.FIRST,source='wrong')),patch.object(C,'sha',return_value='right'):
            with self.assertRaises(RuntimeError):S.admitted_keys(1)
    def test_uncertain_probe_not_reused(self):
        with patch.object(C,'read',return_value=dict(passed=True,probe_result=False)):
            with self.assertRaises(RuntimeError):scope.probe_proof(scope.FIRST[0])
    def test_prior_output_not_retried(self):
        with patch.object(C,'verify_source'),patch.object(P,'stopped'),patch.object(P,'reconcile'),patch.object(Path,'exists',return_value=True):
            with self.assertRaises(RuntimeError):D.preflight(scope.FIRST[0])
    def test_stopped_gate(self):
        with patch.object(C,'read',return_value=dict(active=['live'],cursor=2592)):
            with self.assertRaises(RuntimeError):P.stopped()
    def fixture(self,index=0):
        q=C.read(C.CODE/('QUALIFIED_378.json' if index==0 else 'QUALIFIED_371.json'))
        path=Path(q['archive'])
        if not path.exists():path=C.CODE.parent/'phase1_solver_recovery_20261110_v2/results/output'/q['key']/'verified_core.zip'
        with zipfile.ZipFile(path) as z:r=json.loads(z.read('input/record.json'))
        return path,r,q
    def test_verified_archives(self):
        for i in (0,1):
            path,r,q=self.fixture(i);self.assertEqual(P.verify_archive(path,r,q),q['receipt'])
    def test_changed_qualification(self):
        path,r,q=self.fixture();q['source']='wrong'
        with self.assertRaises(RuntimeError):P.verify_archive(path,r,q)
    def test_wrong_identity(self):
        path,r,q=self.fixture();r['slot']=379
        with self.assertRaises(RuntimeError):P.verify_archive(path,r,q)
    def test_promote_idempotent(self):
        path,r,q=self.fixture()
        with tempfile.TemporaryDirectory(dir=C.ROOT) as t:
            target=Path(t)/'core.zip';P.promote(path,target,r,q);fp=P.fingerprint(target)
            P.promote(path,target,r,q);self.assertEqual(P.fingerprint(target),fp)
    def test_conflict_never_overwritten(self):
        path,r,q=self.fixture()
        with tempfile.TemporaryDirectory() as t:
            target=Path(t)/'core.zip';target.write_bytes(b'preserve')
            with self.assertRaises(RuntimeError):P.promote(path,target,r,q)
            self.assertEqual(target.read_bytes(),b'preserve')
    def test_publisher_no_numerics(self):
        text=(C.CODE/'publish.py').read_text()
        for word in ['sbatch','MATLAB','recovery.run','StrictProbe']:self.assertNotIn(word,text)
    def test_strict_implementation_unchanged(self):
        parent=C.CODE.parent/'phase1_solver_finish3_20261110_v1'
        self.assertEqual(C.sha(C.CODE/'StrictProbe.jl'),C.sha(parent/'StrictProbe.jl'))
        self.assertEqual(C.sha(C.CODE/'recover_flow.jl'),C.sha(parent/'recover_flow.jl'))

if __name__=='__main__':unittest.main()
