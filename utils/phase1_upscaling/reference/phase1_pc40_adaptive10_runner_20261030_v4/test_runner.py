import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import common as C
import planning as P
import control
import slurm as S
import worker
import os

def record(i=1):return dict(case_id='s01_c001_pc04_ca40_IND_001',slot=i)

class Planning(unittest.TestCase):
    def test_array_index_is_not_packet_zero(self):
        with patch.dict(os.environ,{'SLURM_ARRAY_TASK_ID':'16'}):
            self.assertEqual(worker.packet_index(),16)
            self.assertEqual(worker.packet_index(16),16)
            with self.assertRaises(RuntimeError):worker.packet_index(0)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(RuntimeError):worker.packet_index()
            self.assertEqual(worker.packet_index(5),5)
    def test_522_partition(self):
        ps=P.packets([record(i) for i in range(522,0,-1)])
        self.assertEqual([len(x) for x in ps],[32]*16+[10])
        self.assertEqual([r['slot'] for p in ps for r in p],list(range(1,523)))
    def test_duplicate_and_bad_identity(self):
        with self.assertRaises(RuntimeError):P.packets([record(),record()])
        for r in [dict(case_id='../old',slot=1),record(523),record(0)]:
            with self.assertRaises(RuntimeError):C.key(r)
    def test_rolling_not_batches(self):
        keys=[str(i) for i in range(20)];state=dict(cursor=0,active=[]);launches=[];done=set()
        P.advance(state,keys,lambda x:x in done,launches.append,lambda x:None)
        self.assertEqual(state['active'],keys[:8])
        done.add('3');P.advance(state,keys,lambda x:x in done,launches.append,lambda x:None)
        self.assertEqual(len(state['active']),8);self.assertIn('8',state['active']);self.assertIn('0',state['active'])
        self.assertNotIn('3',state['active'])
    def test_interrupted_assignment_is_reconciled(self):
        state=dict(cursor=0,active=[]);snapshots=[];seen=[]
        def fail(k):raise OSError('interrupted after saved assignment')
        with self.assertRaises(OSError):P.advance(state,['a','b'],lambda c:False,fail,lambda s:snapshots.append(copy.deepcopy(s)))
        self.assertEqual(state,dict(cursor=1,active=['a']))
        P.advance(state,['a','b'],lambda c:False,seen.append,lambda s:None)
        self.assertEqual(seen,['a','b'])
    def test_no_exceed_eight(self):
        with self.assertRaises(RuntimeError):P.advance(dict(cursor=9,active=list('abcdefghi')),list('abcdefghi'),lambda c:False,lambda c:None,lambda s:None)
    def test_completed_cases_skipped_without_preparation(self):
        state=dict(cursor=0,active=[]);launched=[]
        P.advance(state,['done','new'],lambda c:c=='done',launched.append,lambda s:None)
        self.assertEqual(launched,['new']);self.assertEqual(state,dict(cursor=2,active=['new']))
    def test_scope_cannot_launch_campaign(self):
        with self.assertRaises(RuntimeError):control.authorize(dict(purpose='production',cases=list(range(2592)),packet_count=17))
        with self.assertRaises(RuntimeError):control.authorize(dict(purpose='qualification',cases=[1,2,3],packet_count=1))
        with self.assertRaises(RuntimeError):control.authorize(dict(purpose='lifecycle_fixture',cases=list(range(10)),packet_count=2))

class Checkpoints(unittest.TestCase):
    def test_decisions_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td)/'work';w.mkdir();out=Path(td)/'result.zip';r=record()
            self.assertEqual(C.action(w,out,r,'m'),'PREPARE')
            (w/'SOLVE_STARTED.toml').touch();self.assertEqual(C.action(w,out,r,'m'),'HOLD_UNCERTAIN')
            (w/'flow').mkdir();(w/'flow/CORE_PASS.toml').touch();self.assertEqual(C.action(w,out,r,'m'),'FINISH')
            (w/'POSTFLOW_HOLD.json').touch();self.assertEqual(C.action(w,out,r,'m'),'HOLD_POSTFLOW')
    def test_archive_checks_all_members(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td)/'work';w.mkdir();(w/'x').write_bytes(b'checked');out=Path(td)/'result.zip'
            receipt=C.publish(w,record(),'method',out)
            self.assertEqual(C.archive_receipt(out,record(),'method'),receipt)
            self.assertEqual(C.action(w,out,record(),'method'),'DONE')
            with self.assertRaises(RuntimeError):C.verify_archive(out,record(2),'method')
            with self.assertRaises(RuntimeError):C.verify_archive(out,record(),'other')
            bad=Path(td)/'bad.zip'
            with zipfile.ZipFile(out) as z,zipfile.ZipFile(bad,'w') as zz:
                for n in z.namelist():zz.writestr(n,b'corrupt' if n=='x' else z.read(n))
            with self.assertRaises(RuntimeError):C.verify_archive(bad,record(),'method')
    def test_immutable_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.json';C.write(p,dict(a=1));C.write(p,dict(a=1))
            with self.assertRaises(RuntimeError):C.write(p,dict(a=2))
    def test_fixture_cannot_be_scientific(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.zip';r=record()
            with zipfile.ZipFile(p,'w') as z:z.writestr('PASS.json',json.dumps(dict(files={},scientific_pass=False,fixture=True,record_sha256=C.digest(r),key=C.key(r),method='m')))
            with self.assertRaises(RuntimeError):C.verify_archive(p,r,'m')

class Scheduler(unittest.TestCase):
    def test_terminal_including_failure(self):
        text='12_0|COMPLETED|0:0\n12_1|OUT_OF_MEMORY|0:125\n12_1.batch|FAILED|1:0\n'
        self.assertEqual(len(control.terminal_tasks(text,'12',2)),2)
        with self.assertRaises(RuntimeError):control.terminal_tasks(text,'12',17)
        with self.assertRaises(RuntimeError):control.terminal_tasks('12_0|RUNNING|0:0','12',1)
    def test_allocation_and_throttle(self):
        want=dict(cpus=2,name='x',script='/x',memory=18432,array='0-16',minutes=120,dependency=None)
        fields=dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='2',Requeue='0',JobName='x',Command='/x',
                    MinMemoryNode='18G',ArrayTaskId='0-16',TimeLimit='02:00:00',Dependency='(null)')
        S.check(fields,want)
        for k,v in [('Account','mit_general'),('QOS','normal'),('MinMemoryNode','16G'),('ArrayTaskThrottle','8'),('ArrayTaskId','0-15')]:
            with self.subTest(field=k),self.assertRaises(RuntimeError):S.check(dict(fields,**{k:v}),want)
    def test_uncertain_submit_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)):
            want=dict(label='u',name='u',script='/x',args=[],array='0-16',dependency=None,cpus=2,memory=18432,minutes=120,mode='worker')
            C.write(Path(td)/'submissions/u/INTENT.json',dict(spec=want,created='2026-10-30T00:00:00'))
            commands=[]
            def fake(args):commands.append(args);return ''
            with patch.object(S,'command',fake),self.assertRaises(RuntimeError):S.held(want)
            self.assertTrue(commands);self.assertFalse(any(x[0]=='sbatch' for x in commands))
    def test_lost_ack_reconciled_exactly_once(self):
        with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)):
            want=dict(label='u',name='u');C.write(Path(td)/'submissions/u/INTENT.json',dict(spec=want,created='2026-10-30T00:00:00'))
            tag='phase1:'+C.digest(want);commands=[]
            def fake(args):commands.append(args);return '123_0|'+tag
            with patch.object(S,'command',fake):
                self.assertEqual(S.held(want),'123');self.assertEqual(S.held(want),'123')
            self.assertEqual(len(commands),2)
    def test_two_matching_jobs_is_uncertain(self):
        with tempfile.TemporaryDirectory() as td,patch.object(C,'ROOT',Path(td)):
            want=dict(label='u',name='u');C.write(Path(td)/'submissions/u/INTENT.json',dict(spec=want,created='2026-10-30T00:00:00'))
            tag='phase1:'+C.digest(want)
            with patch.object(S,'command',lambda a:'123|'+tag+'\n124|'+tag),self.assertRaises(RuntimeError):S.held(want)

if __name__=='__main__':unittest.main()
