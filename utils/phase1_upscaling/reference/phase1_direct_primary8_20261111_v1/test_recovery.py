import copy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import common as C
import recovery as R
import scope
import slurm as S

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        mock=patch.object(C,'sha',return_value=scope.DIAG_PIN)
        mock.start();self.addCleanup(mock.stop)
    def good(self):
        h=[0.,1e-13,1.,1.,1.,1e-13,1e-13,0.,1.,1e-13]
        a=dict(passed=True,metrics=h[1:9],mass=h[9],independent_difference=0.,independent_metrics=h[1:9],independent_mass=h[9])
        return dict(passed=True,diagnostic_only=True,scientific_acceptance=False,mode='direct',source=scope.DIAG_PIN,
            primary_target=1e-9,reference_target=1e-11,agreement_target=1e-5,
            attempts=[dict(a,reference=False,bits=256,order_sha256='a'),dict(a,reference=True,bits=384,order_sha256='b')],
            flows=[1.,1.],difference=0.,completed_phase_calls_repeated=0,production_writes=0,
            regularization=False,changed_conductances=False)
    def test_saved_success(self):self.assertEqual(R.check_probe(self.good()),1.)
    def test_failed_not_reused(self):
        d=self.good();d['passed']=False
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_old_source(self):
        d=self.good();d['source']='bad'
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_no_tolerance_relaxation(self):
        d=self.good();d['reference_target']=1e-9
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_independent_mass(self):
        d=self.good();d['attempts'][1]['independent_mass']=2e-11
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_distinct_preconditioners(self):
        d=self.good();d['attempts'][1]['order_sha256']='a'
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_no_repeated_phase(self):
        d=self.good();d['completed_phase_calls_repeated']=1
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_immutable_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'test.json';C.write(p,dict(q=1));C.write(p,dict(q=1))
            with self.assertRaises(RuntimeError):C.write(p,dict(q=2))
    def test_closed_scope(self):
        self.assertEqual(len(scope.TARGETS),8)
        self.assertTrue(all(not k.endswith('229') for k in scope.TARGETS))
        with self.assertRaises(RuntimeError):scope.probe_proof('unrelated')
    def test_saved_prefix_duplicate(self):
        row=dict(sample=0,phase='intrinsic',q=1.,info=dict(spanning_cells=2,metrics=[1e-12,1.,1.,1.,0.,0.,0.,1.]))
        line=json.dumps(row)
        with self.assertRaises(RuntimeError):R.saved_rows(line+'\n'+line)
    def test_changed_original_protection(self):
        with self.assertRaises(RuntimeError):R.verify_protection({'p':'new'},{'p':'old'},lambda _: 'new')
    def test_scientific_gates(self):
        for index,value in ((1,2e-11),(3,0.),(5,2e-11),(6,2e-11),(7,-1e-8),(8,1+1e-8),(9,2e-11)):
            a=self.good()['attempts'][1];h=[0]+a['metrics']+[a['mass']];h[index]=value
            with self.assertRaises(RuntimeError):R.strict_history(h,True)

if __name__=='__main__':unittest.main()
