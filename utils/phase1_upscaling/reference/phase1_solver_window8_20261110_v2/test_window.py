import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C
import slurm as S
import window as W

KEYS=['key%03d'%i for i in range(154)]

class Window(unittest.TestCase):
    def test_scope_partition(self):
        self.assertEqual([k for b in range(20) for k in W.keys_for(KEYS,b)],KEYS)
        self.assertEqual(len(W.keys_for(KEYS,19)),7)
    def test_invalid_batch(self):
        for b in [-1,20,True]:
            with self.assertRaises(RuntimeError):W.keys_for(KEYS,b)
    def test_fill_eight(self):
        self.assertEqual(W.select_batches(KEYS,{0:{},1:{},2:{}},0),list(range(3,11)))
    def test_no_overfill(self):
        self.assertEqual(W.select_batches(KEYS,{0:{},1:{},2:{}},60),[])
        self.assertEqual(W.select_batches(KEYS,{0:{},1:{},2:{}},64),[])
    def test_one_slow_tail_does_not_block(self):
        self.assertEqual(W.select_batches(KEYS,{i:{} for i in range(11)},1),list(range(11,18)))
    def test_partial_arrays_free_capacity(self):
        # Eight unfinished groups can admit more once eight tasks have ended.
        self.assertEqual(W.select_batches(KEYS,{i:{} for i in range(11)},56),[11])
    def test_final_short_batch(self):
        self.assertEqual(W.select_batches(KEYS,{i:{} for i in range(19)},57),[19])
    def test_attempted_never_selected(self):
        self.assertEqual(W.select_batches(KEYS,{i:{} for i in range(20)},0),[])
    def test_over_ceiling_refused(self):
        with self.assertRaises(RuntimeError):W.select_batches(KEYS,{},65)
    def test_pending_and_held_charged(self):
        d={1:dict(worker='123')};s={f'123_{i}':x for i,x in enumerate(['RUNNING','PENDING','COMPLETING','SUSPENDED','COMPLETED','FAILED','CANCELLED'])}
        self.assertEqual(W.unresolved(KEYS,d,s),5)
    def test_missing_accounting_charged(self):
        self.assertEqual(W.unresolved(KEYS,{1:dict(worker='123')},{}),8)
    def test_parse_states(self):
        text='123_0|COMPLETED|\n123_1|CANCELLED by 55|\n123.batch|FAILED|\n123_[2-7]|PENDING|'
        self.assertEqual(W.parse_states(text),{'123_0':'COMPLETED','123_1':'CANCELLED'})
    def test_canonical_worker_spec(self):
        with patch.object(C,'sha',return_value='new'):
            s=W.specs(3,KEYS[:8])
        self.assertEqual(s,dict(label='targeted003',name='p1q_targeted003',mode='worker',args=['3'],array='0-7',dependency=None,
            cpus=2,memory=18432,minutes=120,root=str(W.OUT),source=W.OLDPIN,script=str(W.OLD/'job.sh')))
    def test_publication_keeps_own_dependency(self):
        with patch.object(C,'sha',return_value='new'):s=W.pubspec(3,'123')
        self.assertEqual(s['root'],str(W.OUT));self.assertEqual(s['dependency'],'123')
        self.assertEqual(s['memory'],2048);self.assertEqual(s['source'],W.OLDPIN)
    def test_handoff_trigger_not_publication_barrier(self):
        with patch.object(C,'sha',return_value='new'):s=W.event_spec(3,'123')
        self.assertEqual(s['mode'],'advance');self.assertEqual(s['dependency'],'123')
        self.assertEqual(s['source'],'new');self.assertEqual(s['cpus'],1)
    def test_uncertain_submission_blocks(self):
        with tempfile.TemporaryDirectory() as t,patch.object(W,'OUT',Path(t)):
            f=Path(t)/'submissions/targeted003/INTENT.json';C.write(f,dict(uncertain=True))
            with self.assertRaises(RuntimeError):W.launches(KEYS)
    def test_saved_result_blocks_admission(self):
        row=dict(key='k',case_id='case',slot=1,unsuccessful_core_attempts=1)
        with patch.object(C,'read',return_value=dict(cores=[row])),patch.object(Path,'exists',return_value=True):
            with self.assertRaises(RuntimeError):W.checks(3,['k'])
    def test_retry_count_not_reset(self):
        row=dict(key='k',case_id='case',slot=1,unsuccessful_core_attempts=2)
        with patch.object(C,'read',return_value=dict(cores=[row])):
            with self.assertRaises(RuntimeError):W.checks(3,['k'])
    def test_strict_reference_isolated(self):
        iso=dict(source=W.OLDPIN,key='k',automatic_retry=False,error='Probe not passed')
        with patch.object(W,'visible',side_effect=[iso,dict(passed=False,failed_stage='reference')]):
            self.assertEqual(W.scientific_hold('k'),'strict_reference')
    def test_unclassified_stops(self):
        iso=dict(source=W.OLDPIN,key='k',automatic_retry=False,error='Input identity mismatch')
        with patch.object(W,'visible',return_value=iso),patch.object(Path,'exists',return_value=False):
            with self.assertRaises(RuntimeError):W.scientific_hold('k')
    def test_false_probe_not_hidden(self):
        iso=dict(source=W.OLDPIN,key='k',automatic_retry=False,error='Probe not passed')
        with patch.object(W,'visible',side_effect=[iso,dict(passed=True)]):
            with self.assertRaises(RuntimeError):W.scientific_hold('k')
    def test_failure_does_not_prevent_other_groups(self):
        states={f'123_{i}':'FAILED' for i in range(8)}
        with patch.object(W,'scientific_hold',return_value='strict_reference'):
            d=W.classify(KEYS,{1:dict(worker='123',publish='124')},states)
        self.assertTrue(d['terminal'][1]);self.assertEqual(len(d['isolated']),8)
    def test_publication_failure_blocks(self):
        with self.assertRaises(RuntimeError):W.classify(KEYS,{1:dict(worker='123',publish='124')},{'124':'FAILED'})
    def test_completed_not_accepted_from_slurm_alone(self):
        with patch.object(W,'visible',return_value=dict(passed=False)):
            with self.assertRaises(RuntimeError):W.classify(KEYS,{1:dict(worker='123',publish='124')},{'123_0':'COMPLETED'})
    def test_read_only_retry(self):
        with patch.object(C,'read',side_effect=[FileNotFoundError(),{'passed':True}]),patch.object(W.time,'sleep'):
            self.assertEqual(W.visible(Path('x')),{'passed':True})
    def test_does_not_retry_parse_error(self):
        with patch.object(C,'read',side_effect=ValueError('corrupt')):
            with self.assertRaises(ValueError):W.visible(Path('x'))
    def test_ack_no_second_submission(self):
        with tempfile.TemporaryDirectory() as t:
            want=dict(root=t,label='x',source='pin');C.write(Path(t)/'submissions/x/ACK.json',dict(job='123',spec=want))
            with patch.object(S,'command',side_effect=AssertionError('must not submit')):
                self.assertEqual(S.held(want),'123')
    def test_ack_conflict_refused(self):
        with tempfile.TemporaryDirectory() as t:
            want=dict(root=t,label='x',source='pin');C.write(Path(t)/'submissions/x/ACK.json',dict(job='123',spec={}))
            with self.assertRaises(RuntimeError):S.held(want)
    def test_intent_without_job_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as t:
            want=dict(root=t,label='x',name='job');C.write(Path(t)/'submissions/x/INTENT.json',dict(spec=want,created='2026-11-10T00:00:00'))
            with patch.object(S,'command',return_value='') as command:
                with self.assertRaises(RuntimeError):S.held(want)
                self.assertEqual(command.call_count,2)
                self.assertTrue(all(c.args[0][0] in ('squeue','sacct') for c in command.call_args_list))
    def test_no_numerical_or_source_edit(self):
        text=(C.CODE/'window.py').read_text()
        for word in ['scancel','shutil.copy','recover_flow.jl','sbatch --wrap','C.write(OLD/']:
            self.assertNotIn(word,text)
    def test_launcher_absolute(self):
        text=(C.CODE/'job.sh').read_text()
        self.assertIn('cd /orcd/data/juanes/001/shaowen/phase1_solver_window8_20261110_v2',text)
        self.assertNotIn('dirname',text)
    def test_actual_sbatch_argv_uses_spec_root(self):
        with tempfile.TemporaryDirectory() as t,patch.object(C,'sha',return_value='pin'):
            want=S.spec('argv','worker',['3'],array='0-7',cpus=2,memory=18432,minutes=120)
            want['root']=t;want['script']='/frozen/numerical/job.sh'
            with patch.object(S,'command',return_value='123') as command:
                self.assertEqual(S.held(want),'123')
                argv=command.call_args.args[0]
                self.assertEqual(argv[-4:],[want['script'],t,'worker','3'])
                self.assertIn('--account=mit_amf_advanced_cpu',argv)
                self.assertIn('--qos=mit_amf_advanced_cpu',argv)
                self.assertNotIn(str(C.ROOT),argv)
    def test_rootcheck_rejects_controller_root(self):
        with patch.dict(W.os.environ,SLURM_JOB_ID='123'),patch.object(C,'ROOT',Path('/wrong')):
            with self.assertRaises(RuntimeError):W.rootcheck()
    def test_repaired_pilot_keeps_prior_records(self):
        with patch.object(C,'sha',return_value='pin'):
            original=W.specs(3,KEYS[:8]);corrected=W.specs(3,KEYS[:8],True)
        self.assertNotEqual(original['label'],corrected['label'])
        for field in ['root','source','script','mode','args','cpus','memory','minutes']:
            self.assertEqual(original[field],corrected[field])

if __name__=='__main__':unittest.main()
