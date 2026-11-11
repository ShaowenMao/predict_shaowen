import copy
import json
from pathlib import Path
import tempfile
import unittest
import common as C
import recovery as R

def success():
    a=[0,1e-12,1.,1.,1.,1e-12,1e-12,0.,1.,1e-12]
    return dict(passed=True,diagnostic_only=True,scientific_acceptance=False,earlier_flow_calls_repeated=0,
        production_writes=0,attempts=[dict(passed=True,tight=False,shift=1e-14,history=[a]),
        dict(passed=True,tight=True,shift=1e-13,history=[a])],primary_q=1.,reference_q=1.,difference=0.)

def call(j,phase):
    return dict(sample=j,phase=phase,q=1.,info=dict(spanning_cells=2,metrics=[0,1,1,1,0,0,0,1]))

class Checks(unittest.TestCase):
    def test_probe(self):self.assertEqual(R.check_probe(success()),1.)
    def test_reference_mass(self):
        d=success();d['attempts'][1]['history'][0][-1]=2e-11
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_independence(self):
        d=success();d['attempts'][1]['shift']=1e-14
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_flux(self):
        d=success();d['reference_q']=1.1
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_nan(self):
        d=success();d['attempts'][0]['history'][0][-1]=float('nan')
        with self.assertRaises(RuntimeError):R.check_probe(d)
    def test_preserved_calls(self):
        rows=R.saved_rows('\n'.join(json.dumps(x) for x in [call(0,'intrinsic'),call(57,'brine')]))
        self.assertEqual(len(rows),2);self.assertEqual(rows[1]['sample'],57)
    def test_duplicate(self):
        with self.assertRaises(RuntimeError):R.saved_rows('\n'.join(json.dumps(x) for x in [call(0,'intrinsic'),call(57,'brine'),call(57,'brine')]))
    def test_no_completed_retry(self):
        with self.assertRaises(RuntimeError):R.saved_rows('\n'.join(json.dumps(x) for x in [call(0,'intrinsic'),call(57,'gas')]))
    def test_original_bad_audit(self):
        bad=call(0,'intrinsic');bad['info']['metrics'][4]=1.
        with self.assertRaises(RuntimeError):R.saved_rows('\n'.join(json.dumps(x) for x in [bad,call(57,'brine')]))
    def test_saved_toml_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.toml';rows=[dict(sample=57,phase='brine',q=3.238796618576338e-27)]
            R.write_toml(p,dict(probe_q=1e-26),rows)
            self.assertEqual(R.tomllib.loads(p.read_text())['calls'],rows)
            with self.assertRaises(FileExistsError):R.write_toml(p,{},rows)
    def test_scope(self):
        self.assertNotIn('s04_c005_pc04_ca40_IND_003_core182',R.TARGETS);self.assertEqual(len(R.TARGETS),154)
    def test_spool_launcher(self):
        script=(C.CODE/'job.sh').read_text()
        self.assertIn('cd /orcd/data/juanes/001/shaowen/phase1_solver_targeted_20261110_v1',script)
        self.assertNotIn('dirname',script)
    def test_private_publication_only(self):
        script=(C.CODE/'recovery.py').read_text()
        self.assertNotIn('scontrol',script);self.assertNotIn('sbatch',script)
        self.assertIn("archive=out/'verified_core.zip'",script)
    def test_earlier_manifest_subset(self):
        protected={'input':'a','started':'b'}
        R.verify_protection(protected,{'input':'a'},lambda p:protected[p])
    def test_earlier_manifest_conflict(self):
        with self.assertRaises(RuntimeError):R.verify_protection({'input':'a'},{'input':'b'},lambda p:'a')
    def test_newly_protected_file_changed(self):
        with self.assertRaises(RuntimeError):R.verify_protection({'input':'a','started':'b'},{'input':'a'},lambda p:'a')

if __name__=='__main__':unittest.main()
