import contextlib
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C
import recovery as R
import slurm as S
import window_contract as W
import adopt_window as A


class WindowTests(unittest.TestCase):
    def setUp(self):
        env=patch.dict(os.environ, {'USER':'qualification'})
        env.start()
        self.addCleanup(env.stop)

    def pump_fixture(self, sizes, active, phase='solve', queue_count=0):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cases=['case%03d'%i for i in range(len(sizes))]
            scope=dict(phase=phase,cases=cases,source='original')
            C.write(root/'SCOPE.json',scope)
            C.write(root/'STATE.json',dict(cursor=active,active=cases[:active]))
            calls=[]
            plans={c:dict(packets=[[i] for i in range(n)]) for c,n in zip(cases,sizes)}
            with patch.object(C,'ROOT',root),patch.object(C,'verify_source'),patch.object(R,'authorize_recovery'), \
                 patch.object(C,'locked',lambda p:contextlib.nullcontext()), \
                 patch.object(R,'checked_plan',side_effect=lambda c:plans[c]), \
                 patch.object(R,'launch',side_effect=lambda c,s:calls.append(c)), \
                 patch.object(S,'command',return_value='\n'.join(['123']*queue_count)):
                R.pump()
            return C.read(root/'STATE.json'), calls, C.read(root/'SCOPE.json')

    def test_more_than_eight_cases_without_raising_worker_ceiling(self):
        state,calls,scope=self.pump_fixture([10]*15,8)
        self.assertEqual(len(state['active']),12)
        self.assertEqual(state['cursor'],12)
        self.assertEqual(len(set(calls)),12)
        self.assertEqual(scope['source'],'original')

    def test_exact_128_is_admitted_but_129_is_not(self):
        state,_,_=self.pump_fixture([16]*8+[1],7)
        self.assertEqual(len(state['active']),8)
        self.assertEqual(state['cursor'],8)

    def test_all_original_packets_remain_charged_during_tail(self):
        state,_,_=self.pump_fixture([16]*7+[17],7)
        self.assertEqual(state['cursor'],7)

    def test_finish_and_pilot_windows_are_not_expanded(self):
        for phase in ('finish','pilot'):
            state,_,_=self.pump_fixture([1]*12,8,phase)
            self.assertEqual(state['cursor'],8)

    def test_small_cases_still_cannot_exceed_128_workers(self):
        state,_,_=self.pump_fixture([1]*129,8)
        self.assertEqual(state['cursor'],128)

    def test_shared_submitted_task_guard_unchanged(self):
        with self.assertRaisesRegex(RuntimeError,'capacity unavailable'):
            self.pump_fixture([1]*9,8,queue_count=439)

    def test_terminal_case_frees_capacity_without_resubmitting_it(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);C.write(root/'SCOPE.json',dict(phase='solve',cases=['old','new']))
            C.write(root/'STATE.json',dict(cursor=1,active=['old']))
            C.write(root/'cases/old/TERMINAL.json',dict(job='done'))
            calls=[]
            with patch.object(C,'ROOT',root),patch.object(C,'verify_source'),patch.object(R,'authorize_recovery'), \
                 patch.object(C,'locked',lambda p:contextlib.nullcontext()), \
                 patch.object(R,'checked_plan',return_value=dict(packets=[[1]])), \
                 patch.object(R,'launch',side_effect=lambda c,s:calls.append(c)),patch.object(S,'command',return_value=''):
                R.pump()
            self.assertEqual(calls,['new'])
            self.assertEqual(C.read(root/'STATE.json'),dict(cursor=2,active=['new']))

    def authority(self):
        scope=dict(phase='solve',source=W.PARENT_SOURCE,max_cases=8,max_worker_reservations=128)
        a=dict(approved_by_user=True,previous_source=W.PARENT_SOURCE,source='new',scope_sha256='scope',
               max_worker_reservations=128,effective_max_cases=128,method='m',configuration='c')
        q=dict(passed=True,source='new',method='m',configuration='c')
        t=dict(passed=True,source='new');lock=dict(method_sha256='m',configuration_sha256='c')
        return [scope,a,q,t,'new',lock,'scope']

    def test_old_scope_is_preserved_under_append_only_authority(self):
        args=self.authority();old=copy.deepcopy(args[0]);W.validate_window(*args)
        self.assertEqual(args[0],old)

    def test_no_science_scope_or_ceiling_override(self):
        for index,key,value in [(0,'phase','finish'),(0,'source','other'),(0,'max_worker_reservations',129),
                                (1,'scope_sha256','changed'),(1,'method','changed'),(1,'effective_max_cases',256),
                                (1,'max_worker_reservations',129),(1,'approved_by_user',False),
                                (2,'passed',False),(3,'passed',False)]:
            args=self.authority();args[index][key]=value
            with self.assertRaises(RuntimeError):W.validate_window(*args)

    def old_launch_fixture(self, root):
        case='s01_c012_pc06_ca40_IND_001';home=root;phase=home/'solve';label='r2s_'+case
        with patch.object(C,'ROOT',phase),patch.object(C,'sha',return_value='new'):
            w=S.spec(label,'worker',[case],array='0-1',cpus=2,memory=18432,minutes=120)
            w.update(source=W.PARENT_SOURCE,script=str(W.PARENT_CODE/'job.sh'))
            g=S.spec(label+'_gate','recovery',['gate',case],dependency='10',minutes=15)
            g.update(source=W.PARENT_SOURCE,script=str(W.PARENT_CODE/'job.sh'))
            ng=S.spec(label+'_window_v4_gate','recovery',['gate',case],dependency='10',minutes=15)
        C.write(phase/'cases'/case/'LAUNCHED.json',dict(job='10',gate='11',packets=2))
        C.write(phase/'submissions'/label/'ACK.json',dict(job='10',spec=w))
        C.write(phase/'submissions'/label/'RELEASED.json',dict(job='10',spec_sha256=C.digest(w)))
        C.write(phase/'submissions'/(label+'_gate')/'ACK.json',dict(job='11',spec=g))
        C.write(phase/'submissions'/(label+'_window_v4_gate')/'ACK.json',dict(job='12',spec=ng))
        C.write(phase/'submissions'/(label+'_window_v4_gate')/'RELEASED.json',dict(job='12',spec_sha256=C.digest(ng)))
        C.write(home/W.ADOPTION,dict(gates={case:dict(worker='10',old_gate='11',new_gate='12')}))
        return case,phase

    def test_old_workers_are_not_submitted_released_or_cancelled(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);case,phase=self.old_launch_fixture(home)
            before={str(p):p.read_bytes() for p in home.rglob('*.json')}
            with patch.object(C,'ROOT',phase),patch.object(C,'sha',return_value='new'),patch.object(R,'HOME',home), \
                 patch.object(R,'checked_plan',return_value=dict(packets=[[1],[2]])), \
                 patch.object(S,'held') as held,patch.object(S,'release') as release,patch.object(S,'command') as command:
                R.launch(case,dict(phase='solve'))
                R.launch(case,dict(phase='solve'))
                held.assert_not_called();release.assert_not_called();command.assert_not_called()
            self.assertEqual(before,{str(p):p.read_bytes() for p in home.rglob('*.json')})

    def test_uncertain_old_worker_release_blocks_instead_of_resubmits(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);case,phase=self.old_launch_fixture(home)
            (phase/'submissions'/('r2s_'+case)/'RELEASED.json').unlink()
            with patch.object(C,'ROOT',phase),patch.object(C,'sha',return_value='new'), \
                 patch.object(S,'held') as held:
                with self.assertRaises(FileNotFoundError):W.preserved_launch(case,2,home)
                held.assert_not_called()

    def test_changed_old_packet_identity_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);case,phase=self.old_launch_fixture(home)
            with patch.object(C,'ROOT',phase),patch.object(C,'sha',return_value='new'):
                with self.assertRaisesRegex(RuntimeError,'assignment changed'):W.preserved_launch(case,3,home)

    def test_running_old_gate_cannot_be_cancelled(self):
        with patch.object(A,'fields',return_value=dict(JobId='11',JobState='RUNNING',Reason='None')), \
             patch.object(S,'command') as command:
            with self.assertRaises(RuntimeError):A.retire_gates({'case':dict(worker='10',old_gate='11')})
            command.assert_not_called()

    def test_retirement_only_targets_known_control_jobs(self):
        calls=[]
        def fields(job):
            return dict(JobId=job,JobState='PENDING',Reason='JobHeldUser',Command=str(W.PARENT_CODE/'job.sh'),Dependency='afterany:10')
        def command(args):
            calls.append(args)
            return '' if args[0]=='scancel' else args[args.index('-j')+1]+'|CANCELLED|0:0'
        with patch.object(A,'fields',fields),patch.object(S,'command',command):
            A.retire_gates({'c1':dict(worker='10',old_gate='11'),'c2':dict(worker='10',old_gate='12')})
        self.assertEqual([x for x in calls if x[0]=='scancel'],[['scancel','11'],['scancel','12']])

    def test_unknown_retirement_blocks_without_retrying_cancel(self):
        f=dict(JobId='11',JobState='PENDING',Reason='JobHeldUser',Command=str(W.PARENT_CODE/'job.sh'),Dependency='afterany:10')
        calls=[]
        with patch.object(A,'fields',return_value=f),patch.object(A.time,'sleep'), \
             patch.object(S,'command',side_effect=lambda args:(calls.append(args) or '')):
            with self.assertRaises(RuntimeError):A.retire_gates({'c':dict(worker='10',old_gate='11')})
        self.assertEqual(sum(x[0]=='scancel' for x in calls),1)

    def test_worker_and_science_are_byte_identical_to_parent(self):
        parent=C.read(C.CODE/'WINDOW_PARENT_LOCK.json')
        for name in ('worker.py','allocation_guard.py','flow_packet.jl','prepare_packet.m','timeout_process.py',
                     'common.py','slurm.py','export_packet.py','native_packet.jl'):
            self.assertEqual(C.sha(C.CODE/name),parent['code'][name],name)
        self.assertEqual(C.sha(C.CODE/'science_bundle.zip'),parent['science_bundle_sha256'])

    def test_transition_retires_controls_before_authorizing_release(self):
        text=(C.CODE/'adopt_window.py').read_text()
        self.assertLess(text.index('retire_gates(gates)\n'),text.index("C.write(out/'TRANSITION_PASS.json'"))
        self.assertLess(text.index("C.write(out/'TRANSITION_PASS.json'"),text.index('S.release(job,want)'))


if __name__=='__main__':
    unittest.main()
