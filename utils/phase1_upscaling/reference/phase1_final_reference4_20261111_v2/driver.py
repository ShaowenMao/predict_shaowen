"""Five finite batches. Solvers and publishers are separate, no resident manager."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import tomllib
import traceback
import common as C
import scope
import slurm as S
import recovery as R
import publish as P

def compute_gate():
    q=C.read(C.ROOT/'qualification/COMPLETE.json')
    C.require(q['passed'] and q['source']==C.sha(C.CODE/'LOCK.json'),'Compute qualification required')

def recovered_gate(batch):
    for key in scope.batch_keys(batch):
        p=C.ROOT/'published'/key/'PASS.json'
        d=C.read(p)
        C.require(d['passed'] and d['source']==C.sha(C.CODE/'LOCK.json'),'Previous batch not published')

def admitted(batch):
    C.require(batch!='pilot','Pilot is publication-only')
    compute_gate()
    d=C.read(scope.PARENT/'output/batches/0.json')
    C.require(d['all_passed'] and d['source']==scope.PARENT_PIN and not d['held'],'Three direct full recoveries must publish first')
    if batch!='0':recovered_gate('0')
    return scope.batch_keys(batch)

def preflight(key):
    C.require(key in scope.NEW_KEYS,'No recalculation of qualified pilot')
    case,slot=key.rsplit('_core',1)
    for p in (R.P/'complete_cases'/(case+'.json'),R.P/'results'/case/('core'+slot+'.zip'),
              R.P/'runs'/case/'done'/('core'+slot+'.json'),C.ROOT/key):
        C.require(not p.exists(),'Existing result/attempt is protected: '+str(p))
    row=scope.SCOPE['rows'][key]
    for p,h in row['metadata'].items():C.require(C.sha(p)==h,'Changed saved failure evidence')

def submit(mode,batch):
    scope.source_proof();P.stopped()
    (C.ROOT/'logs').mkdir(parents=True,exist_ok=True)
    if mode=='qualify':
        w=S.spec('finalref4v2_qualify','qualify',[],cpus=2,memory=4096,minutes=20)
        with C.locked(C.ROOT/'submit.lock'):
            j=S.held(w);S.release(j,w)
        print('SUBMITTED',mode,j,flush=True);return
    compute_gate()
    C.require(mode=='batch','No pilot republication')
    keys=admitted(batch)
    with C.locked(C.ROOT/'submit.lock'):
        # Existing submission evidence is reconciled by held(); never resubmit.
        intent=C.ROOT/'submissions'/('finalref4v2_recover_'+batch)/'INTENT.json'
        if not intent.exists():
            for key in keys:preflight(key)
        w=S.spec('finalref4v2_recover_'+batch,'recover',[batch],array='0-'+str(len(keys)-1),cpus=2,memory=18432,minutes=120)
        job=S.held(w)
        gate=S.spec('finalref4v2_publish_'+batch,'publish',[batch],dependency=job,cpus=1,memory=2048,minutes=120)
        follow=S.held(gate)
        S.release(follow,gate);S.release(job,w)
    print('SUBMITTED batch',batch,'recover',job,'publisher',follow,'cores',len(keys),flush=True)

def qualify():
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute only')
    C.require(C.ROOT==C.CODE/'output' and Path.cwd().resolve()==C.CODE,'Wrong qualification root/launcher')
    scope.source_proof();P.stopped()
    fields=dict(x.split('=',1) for x in S.command(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']]).split() if '=' in x)
    for k,v in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='2',Requeue='0').items():
        C.require(fields.get(k)==v,'Qualification allocation')
    C.require(fields['MinMemoryNode'] in ('4G','4096M'),'Qualification memory')
    out=C.ROOT/'qualification';out.mkdir(parents=True,exist_ok=False)
    local=Path(tempfile.mkdtemp(prefix='p38qual_',dir='/tmp'))
    D=R.external('phase1_precision_batch_staging',scope.DIAG/'driver.py')
    science=D.stage(local)
    env=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_PKG_OFFLINE='true',JULIA_PKG_PRECOMPILE_AUTO='0',
        JULIA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',JULIA_DEPOT_PATH=str(local/'depot')+':'+str(D.DEPOT))
    cmd=[D.JULIA,'--startup-file=no','--history-file=no','--threads=2','--gcthreads=1','--project='+str(D.PROJECT)]
    request=R.write_toml(out/'REQUEST.toml',dict(science=str(science),out=str(out)))
    with (out/'python_tests.log').open('x') as log:
        subprocess.run(['/usr/bin/python3.12','-B','-m','unittest','test_recovery','test_batches','test_primary','test_accounting'],cwd=C.CODE,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    for key in scope.NEW_KEYS:R.authorities(key)
    for script,args in [('test_precision.jl',[]),('test_container.jl',[]),('test_resume.jl',[]),('test_gates.jl',[]),('test_extra.jl',[]),('test_saved.jl',[]),('test_integration.jl',[str(request)])]:
        with (out/(script+'.log')).open('x') as log:
            subprocess.run(cmd+[str(C.CODE/script)]+args,cwd=local,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    C.require(tomllib.loads((out/'QUALIFIED.toml').read_text())['passed'],'Scientific path failed')
    # Read the actual qualified archives, without publishing or re-solving.
    archives=[]
    for key in scope.PILOT_KEYS:
        qpath=scope.PILOT/'output'/key/'QUALIFIED_PASS.json'
        q=C.read(qpath);archive=qpath.parent/'verified_core.zip'
        C.require(q['source']==scope.PILOT_PIN,'Qualified fixture source')
        record=C.read(qpath.parent/'work/input/record.json')
        archives.append(P.verify_archive(archive,record,q))
    C.verify_source()
    C.write(out/'COMPLETE.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],
        qualified_archives=archives,new_production_solves=0,production_writes=0))

def recover(batch):
    keys=admitted(batch);key=keys[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    with C.locked(C.ROOT/'locks'/(key+'.lock')):
        preflight(key);start=time.monotonic()
        try:
            R.run(key)
            C.write(C.ROOT/key/'COMPLETE.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],
                seconds=time.monotonic()-start,qualified_pass_sha256=C.sha(C.ROOT/key/'QUALIFIED_PASS.json')))
        except Exception:
            C.write(C.ROOT/key/'ERROR.json',dict(error=traceback.format_exc(),source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],
                prior_scientific_failures=3,new_bounded_attempt=4,prior_preflight_failures=1,automatic_retry=False,outputs_preserved=True))
            raise

def publish(batch):
    compute_gate();published=[];held=[]
    for key in scope.batch_keys(batch):
        folder,_=scope.qualification_location(key)
        if not (folder/'QUALIFIED_PASS.json').exists() or not (folder/'COMPLETE.json').exists():
            C.require(batch!='pilot','Pilot qualification missing')
            held.append(key);continue
        P.run(key);published.append(key)
    C.write(C.ROOT/'batches'/(batch+'.json'),dict(finished=True,all_passed=not held,published=published,held=held,
        source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],automatic_retry=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit-qualify','submit-batch','qualify','recover','publish'])
    p.add_argument('batch',nargs='?',default='pilot');a=p.parse_args()
    if a.mode.startswith('submit-'):submit(a.mode[7:],a.batch)
    elif a.mode=='qualify':qualify()
    elif a.mode=='recover':recover(a.batch)
    else:publish(a.batch)
