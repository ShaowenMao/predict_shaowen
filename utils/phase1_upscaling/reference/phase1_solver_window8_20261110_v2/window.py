"""Finite Slurm completion handoffs; unchanged numerical/publication workers."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import tomllib
import common as C
import slurm as S

BASE=Path('/orcd/data/juanes/001/shaowen')
OLD=BASE/'phase1_solver_targeted_20261110_v1'
BAD=BASE/'phase1_solver_window8_20261110_v1'
BADPIN='4ac5b8a073c7a89968c60c23442d41894878d5362028b28c078e6919f389ecf9'
BADJOBS={3:('28635657','28635659'),4:('28635664','28635665')}
OUT=OLD/'output'
P=BASE/'phase1_pc40_adaptive10_campaign_20261030_v1'
OLDPIN='846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502'
METHOD='ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'
PILOT=(3,4)
MAX_WORKERS=64
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}

def visible(path,seconds=30):
    end=time.monotonic()+seconds
    while True:
        try:return C.read(path)
        except FileNotFoundError:
            if time.monotonic()>=end:raise
            time.sleep(2)

def source():
    C.verify_source()
    C.require(C.ROOT==C.CODE/'output','Private controller output only')
    C.require(C.sha(OLD/'LOCK.json')==OLDPIN,'Frozen numerical source differs')
    lock=C.read(OLD/'LOCK.json')
    for n,h in lock['code'].items():C.require(C.sha(OLD/n)==h,'Frozen worker source changed: '+n)
    C.require(len(lock['keys'])==154 and len(set(lock['keys']))==154,'Fixed scope differs')
    for path,cursor in [(P/'STATE.json',2592),(P/'round2_20261109_v1/solve/STATE.json',125),(P/'round3_20261110_v1/solve/STATE.json',55)]:
        C.require(C.read(path)==dict(active=[],cursor=cursor),'Old campaign resumed')
    C.require(not (C.ROOT/'STOP.json').exists(),'Controller stopped; no automatic retry')
    C.require(shutil.disk_usage(OUT).free>20*1024**3,'Disk headroom')
    return lock['keys']

def keys_for(keys,batch):
    C.require(type(batch) is int and 0<=batch<20,'Invalid finite batch')
    return keys[:3] if batch==0 else keys[3+8*(batch-1):3+8*batch]

def specs(batch,keys,corrected=False):
    label=f'targeted{batch:03d}'+('_rootfix1' if corrected else '')
    w=S.spec(label,'worker',[str(batch)],array=f'0-{len(keys)-1}',cpus=2,memory=18432,minutes=120)
    w.update(root=str(OUT),source=OLDPIN,script=str(OLD/'job.sh'))
    return w

def pubspec(batch,job,corrected=False):
    label=f'targeted{batch:03d}'+('_rootfix1' if corrected else '')+'_publish'
    g=S.spec(label,'publish',[str(batch)],dependency=job,cpus=1,memory=2048,minutes=30)
    g.update(root=str(OUT),source=OLDPIN,script=str(OLD/'job.sh'))
    return g

def checks(batch,keys):
    """Only never-attempted fixed keys may be newly admitted."""
    rows={r['key']:r for r in C.read(OLD/'AUDIT_SCOPE.json')['cores']}
    for key in keys:
        r=rows[key];case=r['case_id'];slot=r['slot']
        C.require(r['unsuccessful_core_attempts']==1,'Prior attempt count changed')
        paths=[P/'complete_cases'/(case+'.json'),P/'results'/case/f'core{slot:03d}.zip',
            P/'runs'/case/'done'/f'core{slot:03d}.json',OUT/key,OUT/'probes'/key,
            OUT/'claims'/(key+'.json'),OUT/'isolated'/(key+'.json')]
        paths += [P/x/'numerical_claims'/(key+'.json') for x in ('round2_20261109_v1','round3_20261110_v1')]
        C.require(not any(p.exists() for p in paths),'Accepted, isolated, assigned or saved core excluded: '+key)
    return keys

def launches(keys):
    result={}
    for b in range(20):
        folder=OUT/'batches'/f'{b:03d}';launch=folder/'LAUNCHED.json'
        corrected=False
        if b in PILOT and (C.ROOT/'MISROUTE_PASS.json').exists():
            proof=C.read(C.ROOT/'MISROUTE_PASS.json')
            C.require(proof['passed'] and proof['source']==C.sha(C.CODE/'LOCK.json'),'Preflight recovery proof differs')
            launch=C.ROOT/'repairs'/f'{b:03d}'/'LAUNCHED.json';corrected=True
            if not launch.exists():
                C.require(not (OUT/'submissions'/f'targeted{b:03d}_rootfix1'/'INTENT.json').exists(),'Corrected submission uncertain; never duplicate')
                continue
        intent=OUT/'submissions'/f'targeted{b:03d}'/'INTENT.json'
        if not launch.exists():
            C.require(not intent.exists() and not (folder/'ADMITTED.json').exists(),'Incomplete submission; explicit reconciliation needed')
            continue
        d=C.read(launch);ks=keys_for(keys,b)
        C.require(d['keys']==ks and d['source']==OLDPIN,'Existing batch identity differs')
        for want,job in [(specs(b,ks,corrected),d['worker']),(pubspec(b,d['worker'],corrected),d['publish'])]:
            s=OUT/'submissions'/want['label'];ack=C.read(s/'ACK.json')
            C.require(ack==dict(job=job,spec=want),'Canonical ACK mismatch')
            C.require(C.read(s/'INTENT.json')['spec']==want,'Canonical intent mismatch')
            C.require(C.read(s/'RELEASED.json')['job']==job,'Held/unacknowledged reservation requires reconciliation')
        result[b]=d
    return result

def parse_states(text):
    states={}
    for line in text.splitlines():
        fields=line.strip().split('|')
        if len(fields)<2:continue
        job=fields[0].strip()
        if re.fullmatch(r'\d+(?:_\d+)?',job):states[job]=fields[1].strip().split()[0].rstrip('+')
    return states

def queue_states(launched):
    live=parse_states(S.command(['squeue','-h','-r','-u',getpass.getuser(),'-o','%i|%T']))
    if not launched:return live
    ids=','.join(str(v[k]) for v in launched.values() for k in ('worker','publish'))
    past=parse_states(S.command(['sacct','-X','-n','-P','-j',ids,'--format=JobID%64,State%32']))
    past.update(live)
    return past

def unresolved(keys,launched,states):
    count=0
    for b,d in launched.items():
        for i,_ in enumerate(keys_for(keys,b)):
            if states.get(f"{d['worker']}_{i}") not in TERMINAL:count+=1
    return count

def scientific_hold(key):
    iso=visible(OUT/'isolated'/(key+'.json'))
    C.require(iso['source']==OLDPIN and iso['key']==key and not iso['automatic_retry'],'Isolation identity differs')
    probe=OUT/'probes'/key/'PROBE_RESULT.json'
    if iso['error']=='Probe not passed':
        d=visible(probe)
        C.require(not d['passed'] and d.get('failed_stage') in ('primary','reference'),'Unclassified probe failure')
        return 'strict_'+d['failed_stage']
    flow=OUT/key/'flow.log';work=OUT/key/'work/flow'
    if flow.exists() and 'AssertionError: Unchanged strict correction checks failed' in flow.read_text():
        failed=[tomllib.loads(p.read_text()) for p in work.glob('gas*_strict.toml')]
        C.require(any(not d['passed'] and d.get('failed_stage') in ('primary','reference') for d in failed),'Missing failed strict-call evidence')
        return 'later_strict_flow'
    raise RuntimeError('Unclassified recovery error; stop new admission: '+key)

def classify(keys,launched,states):
    terminal={};accepted=[];held=[]
    for b,d in launched.items():
        ks=keys_for(keys,b)
        terminal[b]=all(states.get(f"{d['worker']}_{i}") in TERMINAL for i in range(len(ks)))
        for i,key in enumerate(ks):
            state=states.get(f"{d['worker']}_{i}")
            if state not in TERMINAL:continue
            if state=='COMPLETED':
                q=visible(OUT/key/'QUALIFIED_PASS.json')
                C.require(q['passed'] and q['key']==key and q['source']==OLDPIN and q['method']==METHOD and q['completed_calls_repeated']==0,'Missing or conflicting scientific qualification')
                accepted.append(key)
            elif state in ('OUT_OF_MEMORY','TIMEOUT','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','CANCELLED'):
                held.append(dict(key=key,reason=state))
            else:held.append(dict(key=key,reason=scientific_hold(key)))
        pubstate=states.get(d['publish'])
        if pubstate in TERMINAL:
            C.require(pubstate=='COMPLETED','Publication failed; new admission stopped')
            receipt=visible(OUT/'batches'/f'{b:03d}'/'PASS.json')
            C.require(receipt['passed'] and receipt['source']==OLDPIN and receipt['keys']==ks,'Publication identity differs')
            C.require(set(receipt['published'])|set(receipt['isolated'])==set(ks),'Batch coverage incomplete')
    return dict(terminal=terminal,qualified=accepted,isolated=held)

def snapshot(keys):
    existing=launches(keys);states=queue_states(existing)
    report=classify(keys,existing,states)
    report.update(launched=existing,states=states,outstanding_workers=unresolved(keys,existing,states))
    return report

def select_batches(keys,existing,outstanding,limit=64):
    C.require(0<=outstanding<=limit,'Outstanding worker ceiling exceeded')
    result=[]
    for b in range(20):
        if b in existing:continue
        size=len(keys_for(keys,b))
        if outstanding+size>limit:break
        result.append(b);outstanding+=size
    return result

def event_spec(batch,job):
    return S.spec(f'w8v2_after_{batch:03d}','advance',[str(batch)],dependency=job,cpus=1,memory=2048,minutes=15)

def arm(batch,job):
    want=event_spec(batch,job);follow=S.held(want);S.release(follow,want)
    return follow

def launch(batch,keys,automatic):
    ks=keys_for(keys,batch);checks(batch,ks)
    folder=OUT/'batches'/f'{batch:03d}'
    C.write(folder/'ADMITTED.json',dict(keys=ks,source=OLDPIN,automatic_retry=False))
    corrected=batch in PILOT
    if corrected:
        proof=C.read(C.ROOT/'MISROUTE_PASS.json')
        C.require(proof['passed'] and proof['new_solves']==0 and proof['source']==C.sha(C.CODE/'LOCK.json'),'Unstarted-core proof required')
        folder=C.ROOT/'repairs'/f'{batch:03d}'
    w=specs(batch,ks,corrected);job=S.held(w)
    g=pubspec(batch,job,corrected);publication=S.held(g)
    handoff=arm(batch,job) if automatic else None
    S.release(publication,g);S.release(job,w)
    C.write(folder/'LAUNCHED.json',dict(worker=job,publish=publication,keys=ks,source=OLDPIN))
    C.write(C.ROOT/'admissions'/f'{batch:03d}.json',dict(worker=job,publish=publication,handoff=handoff,keys=ks,source=C.sha(C.CODE/'LOCK.json'),worker_source=OLDPIN,prior_operational_failed_launches=int(corrected),prior_jobs=list(BADJOBS[batch]) if corrected else []))
    print('ADMITTED_BATCH',batch,job,publication,handoff,flush=True)

def qualification():
    d=visible(C.ROOT/'QUALIFIED.json')
    C.require(d['passed'] and d['source']==C.sha(C.CODE/'LOCK.json') and d['worker_source']==OLDPIN,'Compute qualification required')

def rootcheck():
    C.require(os.environ.get('SLURM_JOB_ID') and C.ROOT==OUT,'Actual launcher must receive the numerical output root')
    C.verify_source()
    C.write(C.CODE/'output/ARGUMENT_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),actual_root=str(C.ROOT),job=os.environ['SLURM_JOB_ID'],new_solves=0))

def preflight_failure_proof(keys):
    C.require(C.sha(BAD/'LOCK.json')==BADPIN,'Original failed launcher source differs')
    jobs=','.join(j for pair in BADJOBS.values() for j in pair)
    states=parse_states(S.command(['sacct','-X','-n','-P','-j',jobs,'--format=JobID%64,State%32']))
    files={};cleared=[]
    for b,(job,pub) in BADJOBS.items():
        C.require(states.get(pub)=='FAILED','Failed publisher not terminal/reconciled')
        d=C.read(OUT/'batches'/f'{b:03d}'/'LAUNCHED.json')
        C.require((d['worker'],d['publish'])==(job,pub),'Original failed jobs differ')
        for i,key in enumerate(keys_for(keys,b)):
            C.require(states.get(f'{job}_{i}')=='FAILED','Original worker not terminal failed')
            p=BAD/'output/isolated'/(key+'.json');claim=BAD/'output/claims'/(key+'.json')
            e=C.read(p);c=C.read(claim)
            C.require(e['source']==OLDPIN and e['key']==key and e['error']=='Private output required','Not the proven launch-only failure')
            C.require('probe.py' in e['traceback'] and 'Private output required' in e['traceback'],'Failure location differs')
            C.require(c['key']==key and c['source']==OLDPIN and c['job']==e['job'],'Failed claim differs')
            C.require(not any(f.exists() for f in [BAD/'output'/key,BAD/'output/probes'/key,OUT/key,OUT/'probes'/key,OUT/'claims'/(key+'.json'),OUT/'isolated'/(key+'.json')]),'Saved work exists; no rerun allowed')
            for f in (p,claim):files[str(f)]=C.sha(f)
            cleared.append(key)
    C.write(C.ROOT/'MISROUTE_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),failed_source=BADPIN,keys=cleared,protected=files,new_solves=0,prior_operational_failed_launches_per_core=1))

def qualify():
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute qualification only')
    keys=source()
    a=visible(C.ROOT/'ARGUMENT_PASS.json')
    C.require(a['passed'] and a['source']==C.sha(C.CODE/'LOCK.json') and a['actual_root']==str(OUT),'Cross-root launcher qualification required')
    preflight_failure_proof(keys)
    with C.locked(OUT/'submit.lock'):report=snapshot(keys)
    C.write(C.ROOT/'QUALIFIED.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),worker_source=OLDPIN,
        job=os.environ['SLURM_JOB_ID'],read_only=True,report=report))
    print('READ_ONLY_QUALIFICATION_PASS',flush=True)

def pilot():
    keys=source();qualification()
    with C.locked(OUT/'submit.lock'):
        report=snapshot(keys)
        C.require(all(b not in report['launched'] for b in PILOT),'Pilot already assigned; no duplicate')
        C.require(report['outstanding_workers']+16<=64,'Pilot capacity')
        for b in PILOT:launch(b,keys,False)

def adopt():
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute adoption only')
    keys=source();qualification()
    with C.locked(OUT/'submit.lock'):
        report=snapshot(keys)
        for b in PILOT:
            C.require(report['terminal'].get(b),'Pilot workers not terminal')
            d=visible(OUT/'batches'/f'{b:03d}'/'PASS.json')
            C.require(d['passed'] and d['source']==OLDPIN,'Pilot publication not verified')
            for key in d['published']:
                q=C.read(OUT/key/'QUALIFIED_PASS.json');case,slot=key.rsplit('_core',1)
                C.require(C.read(P/'runs'/case/'done'/f'core{slot}.json')==q['receipt'],'Pilot canonical receipt mismatch')
        C.write(C.ROOT/'ADOPTED.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),worker_source=OLDPIN,pilot=list(PILOT),max_workers=64,job=os.environ['SLURM_JOB_ID']))
    advance('adoption')

def advance(trigger):
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute handoff only')
    keys=source();qualification()
    adopted=visible(C.ROOT/'ADOPTED.json')
    C.require(adopted['passed'] and adopted['source']==C.sha(C.CODE/'LOCK.json'),'Not adopted')
    with C.locked(OUT/'submit.lock'):
        report=snapshot(keys)
        for b,d in report['launched'].items():
            if not report['terminal'][b]:arm(b,d['worker'])
        selected=select_batches(keys,report['launched'],report['outstanding_workers'])
        for b in selected:launch(b,keys,True)
        C.write(C.ROOT/'events'/(os.environ['SLURM_JOB_ID']+'.json'),dict(source=C.sha(C.CODE/'LOCK.json'),trigger=trigger,admitted=selected,before=report,scope_exhausted=len(report['launched'])+len(selected)==20))
        print('WINDOW_ADVANCED',selected,'PREVIOUS_OUTSTANDING',report['outstanding_workers'],flush=True)

def submit_stage(mode):
    C.require(mode in ('rootcheck','qualify','adopt'),'Stage scope')
    source();(C.ROOT/'logs').mkdir(parents=True,exist_ok=True)
    want=S.spec('w8v2_'+mode,mode,[],cpus=1,memory=2048,minutes=30)
    if mode=='rootcheck':want['root']=str(OUT)
    with C.locked(C.ROOT/'stage_submit.lock'):
        job=S.held(want);S.release(job,want);print(mode,job)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit-rootcheck','submit-qualify','submit-adopt','rootcheck','qualify','pilot','adopt','advance']);p.add_argument('trigger',nargs='?',default='operator');a=p.parse_args()
    try:
        if a.mode.startswith('submit-'):submit_stage(a.mode[7:])
        elif a.mode=='rootcheck':rootcheck()
        elif a.mode=='qualify':qualify()
        elif a.mode=='pilot':pilot()
        elif a.mode=='adopt':adopt()
        else:advance(a.trigger)
    except Exception:
        # Never retry an enclosing submission/publication body.
        if a.mode in ('adopt','advance') and os.environ.get('SLURM_JOB_ID'):
            C.write(C.ROOT/'errors'/(os.environ['SLURM_JOB_ID']+'.json'),dict(source=C.sha(C.CODE/'LOCK.json'),traceback=traceback.format_exc()))
            if not (C.ROOT/'STOP.json').exists():C.write(C.ROOT/'STOP.json',dict(job=os.environ['SLURM_JOB_ID'],reason='Unclassified admission/control uncertainty; reconcile before resuming'))
        raise

if __name__=='__main__':main()
