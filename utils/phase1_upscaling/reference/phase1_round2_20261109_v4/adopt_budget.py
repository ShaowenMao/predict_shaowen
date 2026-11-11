"""One compute-side handoff replacement; numerical jobs are never cancelled."""
import os
import time
from pathlib import Path
import common as C
import slurm as S
import recovery as R
from transition_contract import ORIGIN, require_held_gate, retired_gate, validate_pilot_boundary

CASE='s01_c003_pc06_ca40_IND_001'
OLD_GATE='28458641'
PILOT='28458640'
OLD_CODE=R.BASE/'phase1_round2_20261109_v2'
FIX_CODE=R.BASE/'phase1_round2_budget_fix_20261109_v1'


def fields(job):
    raw=S.command(['scontrol','show','job','-o',job])
    rows=raw.splitlines()
    C.require(len(rows)==1 and rows[0].startswith('JobId='+job+' '),'Ambiguous control job')
    return dict(x.split('=',1) for x in rows[0].split() if '=' in x)


def retire_and_release(job,want):
    # Only the already-held, not-yet-run original control handoff is retired.
    S.command(['scancel',OLD_GATE])
    for attempt in range(12):
        raw=S.command(['sacct','-X','-n','-P','-j',OLD_GATE,'--format=JobID%40,State,ExitCode'])
        try:
            retired_gate(raw,OLD_GATE)
            break
        except RuntimeError:
            if attempt==11:raise
            time.sleep(5)
    S.release(job,want)


def main():
    C.require('SLURM_JOB_ID' in os.environ,'Adoption runs on compute')
    out=R.HOME/'transitions/budget_guard_20261109_v1'
    C.require(C.ROOT==out,'Wrong adoption output')
    lock=C.verify_source();source=C.sha(C.CODE/'LOCK.json')
    q=C.read(C.CODE/'qualification/QUALIFIED_PASS.json')
    C.require(q['passed'] and q['source']==source,'New integration not qualified')
    C.require(C.sha(OLD_CODE/'LOCK.json')==ORIGIN,'Original runner changed')
    original=C.read(R.HOME/'AUTHORIZATION.json')
    C.require(original['approved_by_user'] and original['source']==ORIGIN,'Original authority changed')
    guard=C.read(FIX_CODE/'LOCK.json')
    C.require(C.sha(C.CODE/'allocation_guard.py')==guard['code']['allocation_guard.py'],'Qualified guard changed')
    tested=C.read(FIX_CODE/'qualification/ALLOCATION_ARRAY_PASS.json')
    C.require(tested['passed'] and tested['source']==C.sha(FIX_CODE/'LOCK.json'),'Guard array test missing')
    repaired=C.read(R.PRODUCTION/'round2_budget_fix_20261109_v1/SAVED49_RECOVERED_PASS.json')
    C.require(repaired['passed'] and repaired['verified_archives']==49 and repaired['new_solves']==0,'Actual repaired lifecycle missing')
    C.require(C.read(out/'HOLD_REQUEST.json')['job']==OLD_GATE,'Pending handoff hold evidence missing')
    adoption=R.HOME/'ADOPTION_budget_guard_20261109_v1.json'
    C.require(not adoption.exists(),'Adoption already recorded; reconcile, never repeat')
    with C.locked(R.HOME/'phase_transition.lock'),C.locked(R.HOME/'pilot/control.lock'):
        R.first_pass_terminal()
        phase=R.HOME/'pilot';launch=C.read(phase/'cases'/CASE/'LAUNCHED.json')
        validate_pilot_boundary(C.read(phase/'STATE.json'),launch,(R.HOME/'solve/SCOPE.json').exists(),(R.HOME/'NUMERICAL_PILOT_PASS.json').exists())
        C.require(not list((R.HOME/'solve/submissions').glob('*')),'Unreconciled solve reservations')
        require_held_gate(fields(OLD_GATE),OLD_GATE,PILOT,str(OLD_CODE/'job.sh'))
        old_ack=C.read(phase/'submissions'/('r2p_'+CASE+'_gate')/'ACK.json')
        worker_ack=C.read(phase/'submissions'/('r2p_'+CASE)/'ACK.json')
        C.require(old_ack['job']==OLD_GATE and worker_ack['job']==PILOT,'Pilot submission identity changed')
        C.require(old_ack['spec']['source']==worker_ack['spec']['source']==ORIGIN,'Pilot submission source changed')
        preserve=[R.HOME/'AUTHORIZATION.json',phase/'SCOPE.json',phase/'plans'/(CASE+'.json'),
                  phase/'plans'/(CASE+'.seal.json'),phase/'cases'/CASE/'LAUNCHED.json',
                  R.PRODUCTION/'STATE.json',R.PRODUCTION/'RUN.json',OLD_CODE/'LOCK.json']
        preserve+=list((R.HOME/'numerical_claims').glob('*.json'))
        with R.at_root(phase):
            plan=R.checked_plan(CASE)
            for packet in plan['packets']:
                for r in packet:
                    target=R.PRODUCTION/'results'/CASE/f"core{r['slot']:03d}.zip"
                    if target.exists():
                        C.verify_archive(target,r,lock['method_sha256']);preserve.append(target)
            before={str(p):C.sha(p) for p in preserve}
            w=S.spec('r2p_'+CASE+'_budget_gate_v3','recovery',['gate',CASE],dependency=PILOT,cpus=1,memory=1024,minutes=15)
            job=S.held(w)
            actual=S.check(fields(job),w)
            C.require(actual['JobState']=='PENDING' and actual['Reason']=='JobHeldUser','Replacement must be held')
            spool=S.command(['scontrol','write','batch_script',job,'-'])+'\n'
            C.require(spool==(C.CODE/'job.sh').read_text(),'Replacement spool differs')
            C.write(C.ROOT/'submissions'/w['label']/'ALLOCATION.json',actual)
            record=dict(approved_by_user=True,previous_source=ORIGIN,source=source,
                method=lock['method_sha256'],configuration=lock['configuration_sha256'],
                old_gate=OLD_GATE,new_gate=job,pilot_array=PILOT,qualification_sha256=C.sha(C.CODE/'qualification/QUALIFIED_PASS.json'),
                old_authority_preserved=True,scope='Future recovery submissions only; preserve pilot and accepted calculations')
            C.write(adoption,record)
            retire_and_release(job,w)
        for name,digest in before.items():C.require(C.sha(Path(name))==digest,'Protected authority/result changed')
        C.write(out/'TRANSITION_PASS.json',dict(passed=True,source=source,old_gate_retired=OLD_GATE,
            replacement_gate=job,pilot_array_untouched=PILOT,protected_files=before,
            new_numerical_jobs_submitted=0,completed_calculations_repeated=0))
        print('FUTURE_RUNNER_ADOPTED',job,source,flush=True)


if __name__=='__main__':main()
