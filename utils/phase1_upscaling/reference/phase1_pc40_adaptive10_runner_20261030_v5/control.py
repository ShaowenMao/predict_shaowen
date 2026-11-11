"""Finite case handoffs; one frontier, no daemon and no shared-job scheduler."""
import argparse
import os
import time
import common as C
import planning as P
import slurm as S

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE','REVOKED'}

def terminal_tasks(text,job,count):
    found={}
    for line in text.splitlines():
        cells=line.split('|')
        if len(cells)<3:continue
        name,state,exitcode=cells[:3]
        if name.startswith(job+'_') and name[len(job)+1:].isdigit():
            i=int(name[len(job)+1:]);C.require(i not in found,'Duplicate task accounting')
            found[i]=[state.split()[0].rstrip('+'),exitcode]
    C.require(set(found)==set(range(count)) and all(v[0] in TERMINAL for v in found.values()),'Incomplete terminal array accounting')
    return found

def authorize(scope):
    if scope['purpose']=='production':
        p=C.ROOT/'LAUNCH_AUTHORIZATION.json'
        C.require(p.exists(),'Full campaign launch disabled; explicit user authorization required')
        a=C.read(p)
        from adoption import authorized_source
        C.require(a['approved_by_user'] is True and a['scope_sha256']==C.digest(scope) and authorized_source(a,scope),'Launch authorization identity')
        C.require(scope['configuration_sha256']==C.read(C.CODE/'LOCK.json')['configuration_sha256'] and scope['packet_count']==17,'Production configuration mismatch')
        return
    C.require(scope['purpose'] in ('lifecycle_fixture','qualification'),'Invalid scope')
    if scope['purpose']=='lifecycle_fixture':C.require(len(scope['cases'])==9 and scope['packet_count']==2,'Fixture exceeded fixed bound')
    else:C.require(len(scope['cases'])<=2 and scope['packet_count']==1,'Numerical qualification exceeded fixed bound')

def launch(scope,case):
    from adoption import preserve_legacy_assignment
    if preserve_legacy_assignment(case):return
    f=C.ROOT/'cases'/case
    if scope['purpose']=='lifecycle_fixture':
        w=S.spec('life_'+case,'fixture',[case],array='0-1',cpus=1,memory=256,minutes=5)
    else:
        if scope['purpose']=='production':P.prepare_plan(C.CODE,C.ROOT,case)
        w=S.spec('num_'+case,'worker',[case],array='0-16' if scope['purpose']=='production' else '0',cpus=2,memory=18432,minutes=120)
    job=S.held(w);g=S.spec(w['label']+'_gate','control',['gate',case],dependency=job);follow=S.held(g)
    C.write(f/'LAUNCHED.json',dict(job=job,gate=follow))
    S.release(follow,g);S.release(job,w)

def pump():
    C.verify_source();scope=C.read(C.ROOT/'SCOPE.json');authorize(scope)
    with C.locked(C.ROOT/'control.lock'):
        state=C.read(C.ROOT/'STATE.json')
        def term(c):
            complete=C.ROOT/'complete_cases'/f'{c}.json'
            if complete.exists():
                proof=C.read(complete)
                C.require(proof['case_id']==c and proof['positions']==522 and proof['method']==C.read(C.CODE/'LOCK.json')['method_sha256'],'Completed case identity')
                return True
            return (C.ROOT/'cases'/c/'TERMINAL.json').exists()
        def save(value):C.write(C.ROOT/'STATE.json',value,replace=True)
        before=dict(cursor=state['cursor'],active=list(state['active']))
        P.advance(state,scope['cases'],term,lambda c:launch(scope,c),save)
        event=dict(before=before,after=state,job=os.environ.get('SLURM_JOB_ID','operator'))
        C.write(C.ROOT/'events'/f"{time.time_ns()}.json",event)
        if state['cursor']==len(scope['cases']) and not state['active']:
            C.write(C.ROOT/'FIRST_PASS_TERMINAL.json',dict(purpose=scope['purpose'],cases=len(scope['cases']),scientific_completion=False))

def gate(case):
    scope=C.read(C.ROOT/'SCOPE.json');authorize(scope)
    C.require(case in scope['cases'],'Case outside scope')
    f=C.ROOT/'cases'/case
    launch_record=C.read(f/'LAUNCHED.json');job=launch_record['job']
    for attempt in range(12):
        text=S.command(['sacct','-X','-n','-P','-j',job,'--format=JobID%40,State,ExitCode'])
        try:rows=terminal_tasks(text,job,scope['packet_count']);break
        except RuntimeError:
            if attempt==11:raise
            time.sleep(5)
    C.write(f/'TERMINAL.json',dict(job=job,tasks=rows,purpose=scope['purpose']))
    # Refill immediately; failed/held cores never prevent the next case.
    pump()
    if scope['purpose'] in ('qualification','production'):
        p=C.read(C.ROOT/'plans'/f'{case}.json');expected=p['records']
        receipts=list((C.ROOT/'runs'/case/'done').glob('core*.json'))
        # A pilot packet cannot accidentally close a 522-position full case.
        if len(receipts)==522:
            got={C.read(x)['key']:C.read(x) for x in receipts}
            C.require(set(got)=={C.key(r) for r in expected},'Case receipt coverage')
            for r in expected:C.require(got[C.key(r)]['record_sha256']==C.digest(r) and got[C.key(r)]['scientific_pass'],'Case receipt identity')
            C.write(C.ROOT/'complete_cases'/f'{case}.json',dict(case_id=case,positions=522,plan_sha256=C.sha(C.ROOT/'plans'/f'{case}.json'),method=C.read(C.CODE/'LOCK.json')['method_sha256']))

def fixture(case):
    from adoption import route_legacy_worker
    route_legacy_worker('fixture',case)
    scope=C.read(C.ROOT/'SCOPE.json');authorize(scope)
    C.require(scope['purpose']=='lifecycle_fixture' and case in scope['cases'],'Fixture cannot run science')
    i=int(os.environ['SLURM_ARRAY_TASK_ID']);C.require(i in (0,1),'Fixture task')
    time.sleep(1 if case in ('fixture00','fixture08') else 30)
    C.write(C.ROOT/'fixture_outputs'/case/f'{i}.json',dict(fixture=True,scientific_pass=False,case=case,packet=i))
    if case=='fixture02' and i==1:raise SystemExit(7)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pump','gate']);p.add_argument('case',nargs='?');a=p.parse_args()
    if a.mode=='pump':pump()
    else:gate(a.case)

if __name__=='__main__':main()
