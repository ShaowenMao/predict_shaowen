import contextlib
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from unittest.mock import patch

import common as C
import worker
import timeout_process as T
from cache_contract import verify_cache_evidence, CACHED_PREDECESSORS


class DeadlineTests(unittest.TestCase):
    def test_preparation_time_and_finalization_reserve(self):
        now=[0]
        b=T.Budget(7200,clock=lambda:now[0])
        self.assertEqual(b.limit(),6570)
        now[0]=300
        self.assertEqual(b.limit(),6270)
        now[0]=6570
        with self.assertRaises(T.BudgetExhausted):b.limit()
        self.assertEqual(b.limit(finalizing=True),600)
    def test_finalization_cannot_overrun_allocation(self):
        b=T.Budget(7200,clock=lambda:7180)
        with self.assertRaises(T.BudgetExhausted):b.limit(finalizing=True)
    def test_no_unbounded_timeout_for_long_allocation(self):
        self.assertEqual(T.Budget(20000,clock=lambda:0).limit(),13800)
    def test_bad_end_time_fails_closed(self):
        with self.assertRaises(ValueError):T.Budget.allocation({'EndTime':'Unknown'})
    def test_no_launch_with_zero_budget(self):
        with patch.object(subprocess,'Popen') as popen:
            with self.assertRaises(T.BudgetExhausted):T.process([],Path('unused'),None,timeout=0)
            popen.assert_not_called()
    def test_success_and_nonzero_exit_are_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            for code in (0,7):
                self.assertEqual(T.process([sys.executable,'-c',f'raise SystemExit({code})'],
                                          Path(td)/str(code),None,timeout=10),code)
    @unittest.skipUnless(os.name=='posix','Linux process-group test runs on compute')
    def test_timeout_leaves_no_live_group(self):
        with tempfile.TemporaryDirectory() as td:
            log=Path(td)/'child.log'
            with self.assertRaises(subprocess.TimeoutExpired):
                T.process([sys.executable,'-u','-c','import os,time;print(os.getpid());time.sleep(30)'],log,None,timeout=.6)
            self.assertFalse(T.live_group(int(log.read_text().strip())))
    @unittest.skipUnless(os.name=='posix','Linux descendants test runs on compute')
    def test_parent_exit_does_not_leave_descendant_running(self):
        code=('import os,subprocess,sys,time;print(os.getpid(),flush=True);'
              'subprocess.Popen([sys.executable,"-c",'
              '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"]);time.sleep(30)')
        with tempfile.TemporaryDirectory() as td:
            log=Path(td)/'group.log'
            with self.assertRaises(subprocess.TimeoutExpired):T.process([sys.executable,'-u','-c',code],log,None,timeout=1)
            self.assertFalse(T.live_group(int(log.read_text().strip())))
    def test_failed_termination_does_not_become_recoverable_timeout(self):
        fake=unittest.mock.Mock()
        fake.wait.side_effect=subprocess.TimeoutExpired('child',1)
        with tempfile.TemporaryDirectory() as td,patch.object(subprocess,'Popen',return_value=fake),patch.object(T,'terminate',side_effect=RuntimeError('not terminal')):
            with self.assertRaisesRegex(RuntimeError,'not terminal'):T.process(['child'],Path(td)/'log',None,timeout=1)
    def test_timeout_during_termination_is_not_a_flow_timeout(self):
        fake=unittest.mock.Mock()
        fake.wait.side_effect=subprocess.TimeoutExpired('flow',1)
        with tempfile.TemporaryDirectory() as td,patch.object(subprocess,'Popen',return_value=fake),patch.object(T,'terminate',side_effect=subprocess.TimeoutExpired('termination wait',5)):
            with self.assertRaisesRegex(RuntimeError,'finalization forbidden'):T.process(['flow'],Path(td)/'log',None,timeout=1)


class WorkerFixture:
    """Mock numerical stages, but exercise the candidate's actual run() decisions."""
    def __init__(self, initial=('PREPARE','PREPARE','PREPARE'), failure='timeout'):
        self.initial=initial;self.failure=failure;self.calls=[]
    def __enter__(self):
        self.stack=contextlib.ExitStack();td=self.stack.enter_context(tempfile.TemporaryDirectory())
        self.code=Path(td)/'code';self.root=self.code/'qualification'/'fixture';self.root.mkdir(parents=True)
        self.case='s01_c001_pc04_ca40_IND_001'
        self.records=[dict(case_id=self.case,slot=i+1,configuration_sha256='config',assignment=dict(
            clay_pc_sample_id='4',checkpoint_relative_path='checkpoint',checkpoint_hash='hash',predict_seed=1,source_library_row=1)) for i in range(len(self.initial))]
        self.method='method';self.lock=dict(method_sha256=self.method,science={})
        self.write(self.root/'SCOPE.json',dict(purpose='qualification',cases=[self.case],packet_count=1))
        inventory={worker.PHYSICS+'/x':'hash',worker.MRST+'/x':'hash'}
        for name,value in [('EXECUTABLE_PINS.json',{}),('DATA_PINS.json',{}),('RUNTIME_BUNDLE.json',{}),('runtime_inventory.json',{'inventory':inventory}),('LOCK.json',self.lock)]:self.write(self.code/name,value)
        self.work={r['slot']:self.root/'runs'/self.case/'cores'/C.key(r) for r in self.records}
        self.result=lambda r:self.root/'results'/self.case/f"core{r['slot']:03d}.zip"
        for r,state in zip(self.records,self.initial):
            w=self.work[r['slot']]
            if state=='FINISH':self.flow_pass(w)
            elif state=='HOLD_UNCERTAIN':self.write(w/'SOLVE_STARTED.toml',{})
            elif state=='DONE':self.write(self.result(r),{})
        patcher=lambda *a,**k:self.stack.enter_context(patch.object(*a,**k))
        patcher(C,'CODE',self.code);patcher(C,'ROOT',self.root)
        patcher(C,'verify_source',return_value=self.lock);patcher(C,'sha',return_value='hash')
        patcher(C,'locked',lambda p:contextlib.nullcontext(None))
        patcher(C,'verify_archive',return_value={});patcher(C,'archive_receipt',return_value={'scientific_pass':True})
        patcher(C,'publish',self.publish)
        patcher(worker.planning,'checked_plan',return_value=dict(purpose='qualification',allowed_packets=[0],packets=[self.records]))
        patcher(worker,'allocation',return_value={'EndTime':datetime.datetime.fromtimestamp(time.time()+7200).isoformat()})
        patcher(worker,'stage_science',return_value=self.code/'science')
        patcher(worker,'flow_proof',return_value={});patcher(worker,'selection',lambda r,p:{'slot':r['slot']})
        local=self.root/'scratch';local.mkdir()
        patcher(worker.tempfile,'mkdtemp',return_value=str(local))
        patcher(worker.runtime_snapshot,'stage',return_value={'mapping':{worker.PHYSICS+'/x':'/fixture/physics/x',worker.MRST+'/x':'/fixture/mrst/x'}})
        patcher(worker,'process',self.process)
        self.stack.enter_context(patch.dict(os.environ,{'SLURM_JOB_ID':'12345'}))
        return self
    def __exit__(self,*args):return self.stack.__exit__(*args)
    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    @staticmethod
    def flow_pass(work):
        (work/'flow').mkdir(parents=True,exist_ok=True);(work/'flow/CORE_PASS.toml').write_text('passed = true\n')
    def publish(self,w,r,method,target):
        self.write(target,{});return {'scientific_pass':True}
    def process(self,args,log,fd,env=None,cwd=None,timeout=6900):
        self.calls.append(log.name)
        if log.name=='prepare.log':
            request=json.loads(next((self.root/'runs'/self.case/'attempts').glob('*/prepare.json')).read_text())
            Path(request['loaded']).write_text('')
            for row in request['cores']:self.write(Path(row['status']),dict(passed=True,seed=1,source_row=1))
        if log.name=='flow.log':
            if self.failure=='budget':raise T.BudgetExhausted('expired before launch')
            self.flow_pass(self.work[1]);self.write(self.work[2]/'SOLVE_STARTED.toml',{})
            if self.failure=='timeout':raise subprocess.TimeoutExpired(args,timeout)
            if self.failure=='unknown':raise RuntimeError('unknown flow failure')
        if log.name=='export.log':
            if self.failure=='export':raise subprocess.TimeoutExpired(args,timeout)
            req=C.read(Path(args[-1]))
            for row in req['cases']:
                w=Path(row['work']);self.write(w/'export/EXPORT_PASS.json',dict(passed=True,sha256='hash'))
        if log.name=='native.log':
            req=tomllib.loads(Path(args[-1]).read_text())
            for row in req['cases']:
                p=Path(row['work'])/'NATIVE_PASS.toml'
                worker.toml(p,dict(passed=True,record_sha256=row['record_sha256'],method=self.method,sgof_sha256='hash'),[])
        return 0
    def run(self,**kw):worker.run(self.case,0,**kw)
    def summary(self):return C.read(next((self.root/'runs'/self.case/'attempts').glob('*/SUMMARY.json')))


class WorkerTests(unittest.TestCase):
    def test_timeout_finalizes_only_passed_core(self):
        with WorkerFixture() as f:
            f.run();s=f.summary()
            self.assertTrue(s['flow_timed_out']);self.assertEqual(s['verified'],1)
            self.assertEqual(list(s['final'].values()),['DONE','HOLD_UNCERTAIN','PREPARE'])
            self.assertEqual(f.calls.count('flow.log'),1)
            self.assertEqual(f.calls.count('export.log'),1)
            self.assertTrue((f.work[2]/'SOLVE_STARTED.toml').exists())
    def test_saved_flows_take_priority_without_fresh_calculation(self):
        with WorkerFixture(initial=('FINISH','PREPARE','HOLD_UNCERTAIN')) as f:
            f.run();self.assertTrue(f.summary()['checkpoint_only'])
            self.assertEqual(f.summary()['verified'],1)
            self.assertNotIn('flow.log',f.calls);self.assertNotIn('prepare.log',f.calls)
    def test_finish_only_never_prepares_unstarted(self):
        with WorkerFixture() as f:
            f.run(finish_only=True);self.assertEqual(f.summary()['verified'],0)
            self.assertNotIn('flow.log',f.calls);self.assertNotIn('prepare.log',f.calls)
    def test_already_accepted_is_noop(self):
        with WorkerFixture(initial=('DONE','DONE')) as f:
            f.run();self.assertEqual(f.calls,[])
            self.assertFalse((f.root/'runs'/f.case/'attempts').exists())
    def test_unknown_error_not_swallowed_or_retried(self):
        with WorkerFixture(failure='unknown') as f:
            with self.assertRaisesRegex(RuntimeError,'unknown flow'):f.run()
            self.assertNotIn('export.log',f.calls);self.assertEqual(f.calls.count('flow.log'),1)
            self.assertTrue((f.work[1]/'flow/CORE_PASS.toml').exists())
    def test_export_timeout_preserves_flow_and_does_not_publish(self):
        with WorkerFixture(initial=('FINISH','PREPARE'),failure='export') as f:
            with self.assertRaises(subprocess.TimeoutExpired):f.run()
            self.assertTrue((f.work[1]/'flow/CORE_PASS.toml').exists())
            self.assertFalse(f.result(f.records[0]).exists());self.assertNotIn('native.log',f.calls)
    def test_budget_exhaustion_reaches_finalization(self):
        with WorkerFixture(failure='budget') as f:
            f.run();self.assertTrue(f.summary()['flow_timed_out'])
            self.assertEqual(f.summary()['verified'],0)
    def test_production_scope_forbidden(self):
        with WorkerFixture() as f:
            f.write(f.root/'SCOPE.json',dict(purpose='production'))
            with self.assertRaisesRegex(RuntimeError,'Only isolated qualification'):f.run()
            self.assertEqual(f.calls,[])
    def test_root_outside_candidate_forbidden(self):
        with WorkerFixture() as f,patch.object(C,'CODE',f.code/'different'):
            with self.assertRaisesRegex(RuntimeError,'Only isolated qualification'):f.run()
    def test_flow_identity_failure_prevents_publication(self):
        with WorkerFixture(initial=('FINISH','PREPARE')) as f,patch.object(worker,'flow_proof',side_effect=RuntimeError('checksum mismatch')):
            with self.assertRaisesRegex(RuntimeError,'checksum'):f.run()
            self.assertNotIn('export.log',f.calls);self.assertFalse(f.result(f.records[0]).exists())


class SourceTests(unittest.TestCase):
    def test_only_fixed_cached_predecessors(self):
        from test_cache_integration import CacheContract
        f=CacheContract();f.setUp()
        for source in CACHED_PREDECESSORS:
            self.assertEqual(f.check(source={'runner_source':source}),'verified_cache')
            with self.assertRaises(RuntimeError):f.check(source={'runner_source':source},p={'state_cache_implementation':{}})
    def test_numerical_and_acceptance_sources_unchanged(self):
        parent=C.read(C.CODE/'PARENT_LOCK.json')
        for name in ('export_packet.py','native_packet.jl','common.py','planning.py','slurm.py','control.py','prepare_packet.m'):
            self.assertEqual(C.sha(C.CODE/name),parent['code'][name],name)
        import hashlib
        guard='    prior=joinpath(work,"PRIOR_INPUT_PASS.toml")\n    if isfile(prior)\n        @assert TOML.parsefile(prior)==proof "Round-two replay must match the exact original input"\n    end\n'
        body=(C.CODE/'flow_packet.jl').read_text()
        self.assertEqual(body.count(guard),1)
        self.assertEqual(hashlib.sha256(body.replace(guard,'').encode()).hexdigest(),parent['code']['flow_packet.jl'])
        self.assertEqual(C.sha(C.CODE/'science_bundle.zip'),parent['science_bundle_sha256'])


if __name__=='__main__':unittest.main()
