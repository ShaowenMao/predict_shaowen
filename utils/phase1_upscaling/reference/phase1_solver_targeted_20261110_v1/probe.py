"""Private, single-phase diagnostic. No result publication or campaign control."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
import common as C
import scope

BASE=Path('/orcd/data/juanes/001/shaowen')
P=BASE/'phase1_pc40_adaptive10_campaign_20261030_v1'
R3=BASE/'phase1_round3_20261110_v1'
AUDIT=BASE/'phase1_held173_audit_20261110_v1/output'
R3_PIN='2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b'
SCOPE_PIN='4ac23c184f042b437863af7c5edf3321b2c5be817bb57a36b664eb87c12a8234'
METHOD='ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'
TARGETS=scope.TARGETS


def last_failed_call(calls):
    C.require(calls and calls[0]['phase']=='intrinsic','Intrinsic log missing')
    seen=set()
    for c in calls:
        k=(c['sample'],c['phase'])
        C.require(k not in seen,'Duplicate saved call')
        seen.add(k)
        C.require(c['q']>=0,'Negative saved flow')
    last=calls[-1]
    C.require(last['phase']=='brine' and (last['sample'],'gas') not in seen,'Failed gas call ambiguous')
    return last['sample']


def external(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def run(index):
    lock=C.verify_source()
    C.require(os.environ.get('SLURM_JOB_ID') and 0<=index<len(TARGETS),'Compute-only bounded scope')
    C.require(C.ROOT==C.CODE/'output','Private output required')
    C.require(C.sha(R3/'LOCK.json')==R3_PIN and C.sha(AUDIT/'SCOPE.json')==SCOPE_PIN,'Authority changed')
    rlock=C.read(R3/'LOCK.json')
    for n,h in rlock['code'].items():C.require(C.sha(R3/n)==h,'Frozen runner changed: '+n)
    sys.path.insert(0,str(R3))
    W=external('phase1_frozen_probe_dependency_worker',R3/'worker.py')
    W.allocation()
    item=next(r for r in C.read(AUDIT/'SCOPE.json')['cores'] if r['key']==TARGETS[index])
    C.require(item['category'] in ('strict_solver_correction_failed','singular_factorization'),'Wrong scope')
    case=item['case_id'];slot=item['slot'];original=Path(item['original_work'])
    C.require(not (P/'complete_cases'/f'{case}.json').exists(),'Complete case excluded')
    C.require(not (P/'results'/case/f'core{slot:03d}.zip').exists(),'Accepted core excluded')
    C.require(not (P/'runs'/case/'done'/f'core{slot:03d}.json').exists(),'Accepted receipt excluded')
    for state in (P/'STATE.json',P/'round2_20261109_v1/solve/STATE.json',P/'round3_20261110_v1/solve/STATE.json'):
        C.require(not C.read(state)['active'],'Campaign still active')
    out=C.ROOT/'probes'/item['key'];out.mkdir(parents=True,exist_ok=False)
    protected={str(original/n):C.sha(original/n) for n in item['original_files']}
    old_protected=C.read(AUDIT/'PROTECTED.json')
    for path,h in protected.items():
        if path in old_protected:C.require(h==old_protected[path],'Original evidence changed')
    record=C.read(original/'input/record.json')
    C.require(C.digest(record)==item['record_sha256'] and C.key(record)==item['key'],'Record mismatch')
    plan=C.read(P/'plans'/f'{case}.json')
    C.require(next(r for r in plan['records'] if r['slot']==slot)==record,'Assignment changed')
    terminal=C.read(P/'cases'/case/'TERMINAL.json')
    C.require(terminal['job']==item['original_job'] and terminal['tasks'],'Original terminal proof missing')
    terminal_states={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    C.require(all(v[0] in terminal_states for v in terminal['tasks'].values()),'Original work not terminal')
    calls=[json.loads(s) for s in (original/'flow/flow_calls.jsonl').read_text().splitlines() if s.strip()]
    sample=last_failed_call(calls)
    C.write(out/'ATTEMPT.json',dict(prior_unsuccessful_core_attempts=1,diagnostic_attempt=1,
        prior_failure_preserved=True,sample=sample,phase='gas',publication_authorized=False,
        key=item['key'],source=C.sha(C.CODE/'LOCK.json')))
    C.write(out/'PROTECTED.json',protected)
    local=Path(tempfile.mkdtemp(prefix='p1_solver_probe_'+os.environ['SLURM_JOB_ID']+'_',dir='/tmp'))
    C.require(shutil.disk_usage(local).free>4*1024**3,'Node-local storage headroom')
    science=local/'science';science.mkdir()
    C.require(C.sha(R3/'science_bundle.zip')==rlock['science_bundle_sha256'],'Science bundle changed')
    with zipfile.ZipFile(R3/'science_bundle.zip') as z:
        C.require(set(z.namelist())==set(rlock['science']),'Science members differ')
        for n,h in rlock['science'].items():
            b=z.read(n);C.require(hashlib.sha256(b).hexdigest()==h,'Science member changed')
            f=science/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
    def stage(args,log,env=None,timeout=2400):
        with (out/log).open('x') as f:
            subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=local,check=True,timeout=timeout)
    for p,h in C.read(R3/'EXECUTABLE_PINS.json').items():C.require(C.sha(p)==h,'Executable changed')
    stage(['/usr/bin/python3.12','-B',str(science/'runtime_check.py')],'runtime.log')
    inventory=C.read(R3/'runtime_inventory.json')['inventory']
    staged=W.runtime_snapshot.stage(C.read(R3/'RUNTIME_BUNDLE.json'),inventory,local/'runtime')
    def staged_root(original):
        name=next(p for p in staged['mapping'] if p.startswith(original+'/'))
        return staged['mapping'][name][:-len(name[len(original):])]
    for p,h in C.read(R3/'DATA_PINS.json').items():C.require(C.sha(W.DATA/p)==h,'Replay data changed')
    cp=W.DATA/record['assignment']['checkpoint_relative_path']
    C.require(C.sha(cp)==record['assignment']['checkpoint_hash'],'Checkpoint changed')
    sel=W.selection(record,cp)
    with (local/'selection.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(sel));writer.writeheader();writer.writerow(sel)
    row=dict(key=item['key'],slot=slot,sample=int(record['assignment']['clay_pc_sample_id']),
             selection=str(local/'selection.csv'),local=str(local/'core'),rawdir=str(local/'raw'),status=str(local/'PREPARE.json'))
    request=out/'prepare.json'
    C.write(request,dict(code=str(R3),science=str(science),physics=staged_root(W.PHYSICS),
        mrst=staged_root(W.MRST),data=str(W.DATA),cores=[row],loaded=str(local/'loaded.txt')))
    env=dict(os.environ,TMPDIR=str(local),TMP=str(local),TEMP=str(local),MATLAB_PREFDIR=str(local/'preferences'),
             MCR_CACHE_ROOT=str(local/'mcr'),OMP_NUM_THREADS='2')
    stage([W.MATLAB,'-batch',f"addpath('{R3}');prepare_packet('{request}');"],'prepare.log',env)
    status=C.read(row['status']);C.require(status['passed'],'Replay failed')
    prior=C.read(original/'input/REPLAY_PASS.json')
    C.require({k:v for k,v in status.items() if k!='seconds'}=={k:v for k,v in prior.items() if k!='seconds'},'Replay not identical')
    stagedpins={v:inventory[k] for k,v in staged['mapping'].items()}
    for name in (local/'loaded.txt').read_text().splitlines():
        f=Path(name)
        if name in stagedpins:C.require(C.sha(f)==stagedpins[name],'Loaded runtime changed')
        elif f.is_relative_to(R3):C.require(C.sha(f)==rlock['code'][f.relative_to(R3).as_posix()],'Loaded runner changed')
        elif f.is_relative_to(science):C.require(C.sha(f)==rlock['science'][f.relative_to(science).as_posix()],'Loaded science changed')
        else:C.require(f.is_relative_to(Path(W.MATLAB).parent.parent),'Unexpected MATLAB source')
    C.write(out/'REPLAY_PASS.json',status)
    flow=dict(key=item['key'],slot=slot,sample=row['sample'],record_sha256=item['record_sha256'],
              raw=str(local/'raw/raw_unplanned.bin'),geometry=str(local/'raw/geometry_unplanned.bin'),rawdir=str(local/'raw'))
    req=W.toml(out/'probe.toml',dict(science=str(science),method=METHOD,out=str(out),
               prior_input=str(original/'input/INPUT_PASS.toml'),failed_sample=sample),[flow])
    env=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_PKG_OFFLINE='true',JULIA_PKG_PRECOMPILE_AUTO='0',
             JULIA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',JULIA_DEPOT_PATH=str(local/'depot')+':'+W.DEPOT)
    try:
        stage([W.JULIA,'--startup-file=no','--history-file=no','--threads=2','--gcthreads=1',
               '--project='+W.PROJECT,str(C.CODE/'probe.jl'),str(req)],'probe.log',env,timeout=2600)
    finally:
        for name in ('raw_planned.bin','geometry_unplanned.bin'):
            f=local/'raw'/name
            if f.exists():shutil.copyfile(f,out/name)
        for path,h in protected.items():C.require(C.sha(path)==h,'Original evidence modified')
        C.require(C.sha(cp)==record['assignment']['checkpoint_hash'],'Checkpoint changed during replay')
        C.write(out/'PRESERVED.json',dict(original_files_unchanged=len(protected),production_writes=0))
    C.verify_source()
    C.write(out/'DIAGNOSTIC_COMPLETE.json',dict(passed=True,probe_result=C.read(out/'PROBE_RESULT.json')['passed'],
             accepted_cores=0,previous_flow_calls_repeated=0,source=C.sha(C.CODE/'LOCK.json'),
             files={n:C.sha(out/n) for n in ('PROTECTED.json','CAPTURE.json','PROBE_RESULT.json','INPUT_MATCH.json','raw_planned.bin','geometry_unplanned.bin','failed_operator.jls')}))


if __name__=='__main__':
    run(int(sys.argv[1]))
