import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import launch_support as S
import dispatcher as D

class AdmissionTests(unittest.TestCase):
    def row(self,id='1',state='RUNNING',cpus='8',mem='18G',reason='None'):
        return [id,state,cpus,mem,reason,S.ACCOUNT,S.ACCOUNT]
    def test_external_running_counted(self):
        a=S.admission([self.row()],set(),2,1024);self.assertEqual(a['used_cpus'],8);self.assertEqual(a['used_mib'],18432)
    def test_external_held_not_counted(self):
        self.assertEqual(S.admission([self.row(state='PENDING',reason='JobHeldUser')],set(),2,1024)['used_cpus'],0)
    def test_own_pending_always_counted(self):
        self.assertEqual(S.admission([self.row(state='PENDING',reason='Dependency')],{'1'},2,1024)['used_cpus'],8)
    def test_eligible_external_pending_counted(self):
        self.assertEqual(S.admission([self.row(state='PENDING',reason='Resources')],set(),0,0)['used_cpus'],8)
    def test_cpu_ceiling(self):self.assertFalse(S.admission([self.row(cpus='500')],set(),1,1)['passed'])
    def test_memory_ceiling(self):self.assertFalse(S.admission([self.row(mem='1984G')],set(),1,1)['passed'])
    def test_duplicates_rejected(self):
        with self.assertRaises(AssertionError):S.admission([self.row(),self.row()],set(),0,0)
    def test_wrong_qos_rejected(self):
        r=self.row();r[-1]='normal'
        with self.assertRaises(AssertionError):S.admission([r],set(),0,0)
    def test_memory_units(self):
        self.assertEqual(S.memory_mib('2T'),2097152);self.assertEqual(S.memory_mib('1025K'),2)

class RoutingTests(unittest.TestCase):
    def row(self,n=60000,clay=.5,span=7):return dict(features=dict(num_cells=n,cartesian_dimensions=[100,n//10000,100],clay_cell_fraction=clay,log10_permeability_span=[span]*3))
    def test_small_routes_to_bounded_class(self):self.assertEqual(S.route(self.row(20000))[0],'mixed_60000')
    def test_clay_not_overallocated_as_mixed(self):self.assertEqual(S.route(self.row(110000,1))[0],'clay_120000')
    def test_large_held(self):self.assertIsNone(S.route(self.row(160000))[0])
    def test_contrast_held(self):self.assertIsNone(S.route(self.row(span=9))[0])
    def test_unknown_geometry_held(self):
        r=self.row();r['features']['cartesian_dimensions']=[50,12,100];self.assertIsNone(S.route(r)[0])
    def test_stage_specific_memory(self):
        p=D.policy('fresh','mixed_150000');self.assertEqual(p['flow']['memory_mib'],20096);self.assertEqual(p['export']['memory_mib'],512)
    def test_json_sorted_keys_do_not_change_dag(self):
        spec=json.loads(json.dumps(D.policy('fresh','mixed_60000'),sort_keys=True));done=set()
        for stage in D.stage_order('fresh'):
            self.assertTrue(set(spec[stage]['deps'])<=done);done.add(stage)
    def test_no_flow_for_completed_curves(self):self.assertEqual(D.stage_order('derive'),['export','native'])
    def test_ramp_requires_success(self):self.assertEqual([S.concurrency(n) for n in [0,1,3,7,15]],[1,2,8,32,250])

class RecoveryTests(unittest.TestCase):
    def test_submission_logging_argument_names(self):
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()) as f:
            D.event('SUBMITTED',stage_name='example',job='123',cpus=2,memory_mib=2816)
        self.assertEqual(json.loads(f.getvalue())['stage_name'],'example')
    def test_uncertain_submit_never_resubmits(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'submissions').mkdir();(root/'submissions/x.intent.json').write_text('{"token":"t"}')
            with patch.object(D,'ROOT',root),patch.dict(D.os.environ,{'USER':'test'}),patch.object(D.subprocess,'check_output',return_value=''):
                with self.assertRaises(AssertionError):D.returned('x')
    def test_completed_submit_returns_same_job(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'submissions').mkdir();(root/'submissions/x.returned.json').write_text('{"job":"123"}')
            with patch.object(D,'ROOT',root):self.assertEqual(D.returned('x'),'123')
    def test_canonical_comparison_checks_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b';x=bytearray(112);y=bytearray(112);x[8]=3;y[8]=7
            a.write_bytes(x);b.write_bytes(y);S.canonical_equal(a,b)
            y[-1]=1;b.write_bytes(y)
            with self.assertRaises(AssertionError):S.canonical_equal(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
