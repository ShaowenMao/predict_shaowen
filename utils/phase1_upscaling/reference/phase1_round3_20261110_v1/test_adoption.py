import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import common as C
import recovery as R
import slurm as S
import adopt_budget as A
from transition_contract import ORIGIN, validate_authority, require_held_gate, retired_gate, validate_pilot_boundary


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.original=dict(approved_by_user=True,source=ORIGIN)
        self.adoption=dict(approved_by_user=True,previous_source=ORIGIN,source='new',method='m',configuration='c')
        self.q=dict(passed=True,source='new',method='m',configuration='c')
        self.lock=dict(method_sha256='m',configuration_sha256='c')
    def validate(self,phase,scope_source,**changes):
        validate_authority(dict(phase=phase,source=scope_source),self.original,dict(self.adoption,**changes),self.q,'new',self.lock)
    def test_original_pilot_authority_is_preserved(self):
        self.validate('pilot',ORIGIN)
        self.assertEqual(self.original['source'],ORIGIN)
    def test_future_scope_must_use_new_source(self):
        self.validate('solve','new')
        with self.assertRaisesRegex(RuntimeError,'Scope/source'):self.validate('solve',ORIGIN)
    def test_unknown_successor_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'successor'):self.validate('pilot',ORIGIN,source='unknown')
    def test_scientific_identity_cannot_change(self):
        with self.assertRaisesRegex(RuntimeError,'Scientific'):self.validate('pilot',ORIGIN,method='changed')
    def test_qualification_required(self):
        self.q['passed']=False
        with self.assertRaisesRegex(RuntimeError,'not qualified'):self.validate('pilot',ORIGIN)
    def test_running_gate_cannot_be_retired(self):
        with self.assertRaisesRegex(RuntimeError,'held/pending'):
            require_held_gate(dict(JobId='1',JobState='RUNNING',Reason='None'),'1','2','script')
    def test_wrong_gate_dependency_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'dependency'):
            require_held_gate(dict(JobId='1',JobState='PENDING',Reason='JobHeldUser',Command='s',Dependency='afterany:9'),'1','2','s')
    def test_existing_solve_scope_blocks_adoption(self):
        with self.assertRaisesRegex(RuntimeError,'already advanced'):
            validate_pilot_boundary(dict(cursor=1,active=[A.CASE]),dict(job=A.PILOT,gate=A.OLD_GATE,packets=2),True,False)
    def test_retirement_requires_exact_cancelled_record(self):
        retired_gate('28458641|CANCELLED by 1|0:0',A.OLD_GATE)
        for raw in ('28458641|RUNNING|0:0','9|CANCELLED|0:0'):
            with self.assertRaises(RuntimeError):retired_gate(raw,A.OLD_GATE)
    def test_retire_then_release_never_cancels_workers(self):
        calls=[]
        def command(args):
            calls.append(args)
            return '' if args[0]=='scancel' else A.OLD_GATE+'|CANCELLED|0:0'
        with patch.object(S,'command',command),patch.object(S,'release',lambda j,w:calls.append(['release',j])):
            A.retire_and_release('new',{})
        self.assertEqual(calls[0],['scancel',A.OLD_GATE])
        self.assertEqual(calls[-1],['release','new'])
        self.assertFalse(any(A.PILOT in x for x in calls))
    def test_ambiguous_retirement_never_releases(self):
        with patch.object(S,'command',return_value=''),patch.object(A.time,'sleep'),patch.object(S,'release') as release:
            with self.assertRaises(RuntimeError):A.retire_and_release('new',{})
            release.assert_not_called()
    def test_future_workers_and_gates_use_successor(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            with patch.object(C,'ROOT',Path(d)),patch.object(C,'sha',return_value='new'), \
                 patch.object(R,'checked_plan',return_value=dict(packets=[[1],[2]])), \
                 patch.object(S,'held',side_effect=lambda w:(calls.append(w) or str(len(calls)))),patch.object(S,'release'):
                R.launch(A.CASE,dict(phase='solve'))
            self.assertEqual(len(calls),2)
            self.assertTrue(all(x['source']=='new' and x['script']==str(C.CODE/'job.sh') for x in calls))
            self.assertEqual((calls[0]['array'],calls[0]['cpus'],calls[0]['memory']),('0-1',2,18432))
            self.assertEqual(calls[1]['dependency'],'1')


if __name__=='__main__':unittest.main()
