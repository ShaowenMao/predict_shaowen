"""Finite qualification, then three private complete recoveries. No publisher."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time
import tomllib
import traceback
import common as C
import slurm as S
import scope
import recovery as R

def preflight():
    C.verify_source()
    C.require(C.ROOT==C.CODE/'output','Private qualification only')
    for path,cursor in ((R.P/'STATE.json',2592),(R.P/'round2_20261109_v1/solve/STATE.json',125),
                        (R.P/'round3_20261110_v1/solve/STATE.json',55)):
        C.require(C.read(path)==dict(active=[],cursor=cursor),'Campaign state changed')
    for key in scope.TARGETS:
        case,slot=key.rsplit('_core',1)
        for path in (R.P/'complete_cases'/(case+'.json'),R.P/'results'/case/('core'+slot+'.zip'),
                     R.P/'runs'/case/'done'/('core'+slot+'.json')):
            C.require(not path.exists(),'Accepted core excluded')
    return C.sha(C.CODE/'LOCK.json')

def qualify():
    source=preflight()
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute-only qualification')
    fields=dict(x.split('=',1) for x in S.command(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']]).split() if '=' in x)
    for key,value in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='2',Requeue='0').items():
        C.require(fields.get(key)==value,'Qualification allocation mismatch')
    C.require(fields['MinMemoryNode'] in ('4G','4096M'),'Qualification memory mismatch')
    C.require(C.sha(scope.DIAG/'LOCK.json')==scope.DIAG_PIN,'Diagnostic source pin changed')
    for n,h in C.read(scope.DIAG/'LOCK.json')['code'].items():C.require(C.sha(scope.DIAG/n)==h,'Diagnostic dependency changed')
    C.require(C.sha(R.R3/'LOCK.json')==R.R3PIN,'Frozen dependency changed')
    out=C.ROOT/'qualification';out.mkdir(parents=True,exist_ok=False)
    local=Path(tempfile.mkdtemp(prefix='p56qual_',dir='/tmp'))
    D=R.external('phase1_precision_diagnostic_staging',scope.DIAG/'driver.py')
    science=D.stage(local)
    env=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_PKG_OFFLINE='true',JULIA_PKG_PRECOMPILE_AUTO='0',
             JULIA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',JULIA_DEPOT_PATH=str(local/'depot')+':'+str(D.DEPOT))
    cmd=[D.JULIA,'--startup-file=no','--history-file=no','--threads=2','--gcthreads=1','--project='+str(D.PROJECT)]
    req=R.write_toml(out/'REQUEST.toml',dict(science=str(science),out=str(out)))
    for script,args in [('test_precision.jl',[]),('test_resume.jl',[]),('test_gates.jl',[]),
                        ('test_extra.jl',[]),('test_integration.jl',[str(req)])]:
        with (out/(script+'.log')).open('x') as f:
            subprocess.run(cmd+[str(C.CODE/script)]+args,cwd=local,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
    result=tomllib.loads((out/'QUALIFIED.toml').read_text())
    C.require(result['passed'] and result['production_solves']==0,'Qualification failed')
    C.verify_source()
    C.write(out/'COMPLETE.json',dict(passed=True,source=source,job=os.environ['SLURM_JOB_ID'],result=result))
    print('COMPUTE_QUALIFIED',flush=True)

def submit(mode):
    preflight();(C.ROOT/'logs').mkdir(parents=True,exist_ok=True)
    if mode=='recover':
        q=C.read(C.ROOT/'qualification/COMPLETE.json')
        C.require(q['passed'] and q['source']==C.sha(C.CODE/'LOCK.json'),'Compute qualification required')
    want=S.spec('precision3_'+mode,mode,[],array='0-2' if mode=='recover' else None,
                cpus=2,memory=18432 if mode=='recover' else 4096,minutes=120 if mode=='recover' else 20)
    with C.locked(C.ROOT/'submit.lock'):
        job=S.held(want);S.release(job,want)
    print('SUBMITTED',mode,job,flush=True)

def run(mode):
    if mode=='qualify':return qualify()
    preflight()
    key=scope.TARGETS[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    start=time.monotonic()
    try:
        R.run(key)
        C.write(C.ROOT/key/'COMPLETE.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],
            seconds=time.monotonic()-start,source=C.sha(C.CODE/'LOCK.json'),
            qualified_pass_sha256=C.sha(C.ROOT/key/'QUALIFIED_PASS.json')))
    except Exception:
        C.write(C.ROOT/key/'ERROR.json',dict(error=traceback.format_exc(),job=os.environ['SLURM_JOB_ID'],
            source=C.sha(C.CODE/'LOCK.json'),automatic_retry=False,saved_results_preserved=True))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['submit-qualify','submit-recover','qualify','recover'])
    a=p.parse_args()
    if a.mode.startswith('submit-'):submit(a.mode[7:])
    else:run(a.mode)
