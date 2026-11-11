"""Complete missing calls after a private, strict, independently checked probe."""
import argparse
import hashlib
import importlib.util
import json
import math
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
R3PIN='2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b'
SCOPEPIN='4ac23c184f042b437863af7c5edf3321b2c5be817bb57a36b664eb87c12a8234'
METHOD='ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'
TARGETS=scope.TARGETS
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}

def verify_protection(protected,old,hash_file=C.sha):
    for path,h in protected.items():
        C.require(hash_file(path)==h,'Original evidence changed')
        if path in old:C.require(old[path]==h,'Earlier audit evidence differs')

def external(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def strict_history(h,tight):
    C.require(len(h)==10 and all(math.isfinite(v) for v in h),'Invalid strict audit')
    _,res,_,q,_,balance,energy,pmin,pmax,mass=h
    target=1e-11 if tight else 1e-9
    C.require(q>0 and pmin>=-1e-9 and pmax<=1+1e-9 and res<=min(1e-10,target)
              and max(balance,energy,mass)<=target,'Strict audit failed')
    return q

def check_probe(d):
    C.require(d['passed'] and d['diagnostic_only'] and not d['scientific_acceptance'],'Probe not qualified')
    C.require(d['mode']=='precision' and d['source']==scope.DIAG_PIN,'Wrong precision implementation')
    C.require((d['primary_target'],d['reference_target'],d['agreement_target'])==(1e-9,1e-11,1e-5),'Threshold change')
    success=[a for a in d['attempts'] if a['passed']]
    C.require(len(success)==2 and [a['tight'] for a in success]==[False,True],'Missing independent checks')
    C.require(success[0]['shift']!=success[1]['shift'],'Independent preconditioners required')
    q=[strict_history(a['history'][-1],a['tight']) for a in success]
    C.require(q==[d['primary_q'],d['reference_q']],'Probe flux mismatch')
    diff=abs(q[0]-q[1])/max(abs(q[0]),abs(q[1]))
    C.require(diff<=1e-5 and diff==d['difference'],'Flux agreement failed')
    C.require(d['completed_phase_calls_repeated']==0 and d['production_writes']==0,'Probe side effects')
    audits=d['independent_checks']
    C.require(len(audits)==2,'Missing independent pressure audits')
    for a,tight in zip(audits,(False,True)):
        C.require(a['independent_passed'] and a['independent_q_difference']<=1e-5,'Independent check failed')
        strict_history([0]+a['independent_metrics']+[a['independent_mass']],tight)
    return q[1]

def saved_rows(text):
    rows=[];seen=set()
    for line in text.splitlines():
        if not line.strip():continue
        d=json.loads(line);k=(d['sample'],d['phase']);q=d['q'];info=d['info']
        C.require(k not in seen and math.isfinite(q) and q>=0,'Invalid/duplicate saved call')
        C.require(d['phase'] in ('intrinsic','brine','gas'),'Invalid saved phase')
        seen.add(k)
        m=info['metrics']
        C.require(len(m)==8 and all(math.isfinite(v) for v in m),'Missing saved audit')
        if info['spanning_cells']==0:C.require(q==0,'Disconnected positive flow')
        else:
            C.require(q==m[2] and q>0 and m[0]<=1e-8 and m[4]<=1e-5 and m[5]<=1e-5
                      and m[6]>=-1e-7 and m[7]<=1+1e-7,'Original flow audit failed')
        rows.append(dict(sample=d['sample'],phase=d['phase'],q=q,line_sha256=hashlib.sha256(line.encode()).hexdigest()))
    C.require(rows and (rows[0]['sample'],rows[0]['phase'])==(0,'intrinsic'),'No saved intrinsic')
    C.require(rows[-1]['phase']=='brine' and (rows[-1]['sample'],'gas') not in seen,'Ambiguous failed gas call')
    return rows

def authorities(key):
    C.verify_source()
    C.require(key in TARGETS and C.ROOT==C.CODE/'output','Private bounded scope only')
    C.require(C.sha(R3/'LOCK.json')==R3PIN,'Dependency source changed')
    C.require(C.sha(AUDIT/'SCOPE.json')==SCOPEPIN,'Classification changed')
    for source in (R3,):
        for n,h in C.read(source/'LOCK.json')['code'].items():C.require(C.sha(source/n)==h,'Frozen source changed')
    row=next(r for r in C.read(AUDIT/'SCOPE.json')['cores'] if r['key']==key)
    original=Path(row['original_work']);old=C.read(AUDIT/'PROTECTED.json')
    probe,proof=scope.probe_proof(key)
    protected=C.read(probe/'PROTECTED.json')
    C.require(set(protected)=={str(original/n) for n in row['original_files']},'Original evidence inventory changed')
    verify_protection(protected,old)
    record=C.read(original/'input/record.json')
    C.require(C.key(record)==key and C.digest(record)==row['record_sha256'],'Input record changed')
    C.require(next(r for r in C.read(P/'plans'/(row['case_id']+'.json'))['records'] if r['slot']==row['slot'])==record,'Assignment changed')
    source=C.read(original/'input/SOURCE.json')
    C.require(source['method']==METHOD and source['configuration']==record['configuration_sha256'],'Original scientific identity')
    for f in (P/'STATE.json',P/'round2_20261109_v1/solve/STATE.json',P/'round3_20261110_v1/solve/STATE.json'):
        C.require(not C.read(f)['active'],'Campaign active; reconcile overlapping work')
    term=C.read(P/'cases'/row['case_id']/'TERMINAL.json')
    C.require(term['job']==row['original_job'] and all(v[0] in TERMINAL for v in term['tasks'].values()),'Original work not terminal')
    queue=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i'],text=True)
    C.require(not any(j.split('_')[0]==str(row['original_job']) for j in queue.splitlines()),'Original job live')
    for phase in ('round2_20261109_v1','round3_20261110_v1'):
        C.require(not (P/phase/'numerical_claims'/(key+'.json')).exists(),'Unexpected recovery reservation')
    C.require(not (P/'complete_cases'/(row['case_id']+'.json')).exists(),'Complete case excluded')
    C.require(not (P/'results'/row['case_id']/f"core{row['slot']:03d}.zip").exists(),'Accepted core excluded')
    C.require(not (P/'runs'/row['case_id']/'done'/f"core{row['slot']:03d}.json").exists(),'Accepted receipt excluded')
    d=scope.result(key);capture=C.read(probe/'CAPTURE.json')
    C.require(d['key']==key and d['operator_sha256']==capture['operator_sha256'],'Saved precision operator identity')
    inp=tomllib.loads((original/'input/INPUT_PASS.toml').read_text())
    C.require(inp==C.read(probe/'INPUT_MATCH.json'),'Prepared input proof differs')
    C.require(C.sha(probe/'raw_planned.bin')==inp['raw_sha256'] and C.sha(probe/'geometry_unplanned.bin')==inp['geometry_unplanned_sha256'],'Prepared inputs changed')
    C.require(C.sha(probe/'failed_operator.jls')==capture['operator_sha256'] and capture['parts_hash']==d['parts_hash'],'Captured operator changed')
    for n in ('PROBE_RESULT.json','CAPTURE.json','INPUT_MATCH.json','raw_planned.bin','geometry_unplanned.bin','failed_operator.jls'):
        protected[str(probe/n)]=C.sha(probe/n)
    for n,h in C.read(C.CODE/'SCOPE.json')['proofs'][key].items():
        protected[str(scope.DIAG/'output'/key/n)]=h
    return row,record,original,probe,protected,d,capture

def stage_runtime(local):
    sys.path.insert(0,str(R3));W=external('phase1_solver_recovery_frozen_worker',R3/'worker.py')
    W.allocation()
    lock=C.read(R3/'LOCK.json');science=local/'science';science.mkdir()
    C.require(C.sha(R3/'science_bundle.zip')==lock['science_bundle_sha256'],'Science bundle changed')
    with zipfile.ZipFile(R3/'science_bundle.zip') as z:
        C.require(set(z.namelist())==set(lock['science']),'Science inventory mismatch')
        for n,h in lock['science'].items():
            b=z.read(n);C.require(hashlib.sha256(b).hexdigest()==h,'Scientific source changed')
            f=science/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
    for p,h in C.read(R3/'EXECUTABLE_PINS.json').items():C.require(C.sha(p)==h,'Executable changed')
    return W,science

def write_toml(path,header,rows=None):
    lines=[k+' = '+json.dumps(v,allow_nan=False) for k,v in header.items()]
    for r in rows or []:lines+=['[[calls]]']+[k+' = '+json.dumps(v,allow_nan=False) for k,v in r.items()]
    with path.open('x') as f:f.write('\n'.join(lines)+'\n')
    return path

def validate_final(work,record,rows):
    f=tomllib.loads((work/'flow/CORE_PASS.toml').read_text())
    C.require(f['passed'] and f['method']==METHOD and f['record_sha256']==C.digest(record),'Final flow identity')
    C.require(f['recovery_implementation']==C.sha(C.CODE/'LOCK.json'),'Recovery source changed')
    C.require(f['input']==tomllib.loads((work/'input/INPUT_PASS.toml').read_text()),'Flow input changed')
    for n,h in f['artifacts'].items():C.require(C.sha(work/'flow'/n)==h,'Flow artifact changed')
    s=tomllib.loads((work/'flow/summary.toml').read_text());life=s['native_lifetime']
    C.require(s['mode']=='adaptive10' and s['same_mip_microstates'] and s['pc_points']==100 and 2<=s['kr_states']<=10,'Scientific method changed')
    C.require(s['phase_solver_calls']<=20 and s['intrinsic_solver_calls']==1,'Logical solve budget')
    C.require(s['completed_flow_calls_repeated']==0 and s['saved_flow_calls_reused']==len(rows) and s['probe_calls_reused']==1,'Saved work not fully reused')
    C.require(life['numeric_created']==life['numeric_released'] and life['symbolic_created']==life['symbolic_released'],'Native resource lifetime')
    C.require(s['cached_states_verified']==100 and s['cached_queries_verified']==s['kr_states'] and s['cache_payload_bytes']<=512*1024**2,'Drainage cache proof')
    calls=[json.loads(x) for x in (work/'flow/flow_calls.jsonl').read_text().splitlines()]
    C.require(len(calls)==s['phase_solver_calls']+1==s['new_flow_calls']+len(rows)+1,'Logical/physical count mismatch')
    for old,new in zip(rows,calls):
        C.require(all(old[k]==new[k] for k in ('sample','phase','q')) and new['info']['original_line_sha256']==old['line_sha256'],'Saved flow not identical')
    ex=C.read(work/'export/EXPORT_PASS.json');n=tomllib.loads((work/'NATIVE_PASS.toml').read_text())
    C.require(ex['passed'] and n['passed'] and n['method']==METHOD and n['record_sha256']==C.digest(record),'Native acceptance absent')
    C.require(C.sha(work/'export/sgof.csv')==ex['sha256']==n['sgof_sha256'],'Native table changed')
    C.require(not (work/'POSTFLOW_HOLD.json').exists() and not (work/'NATIVE_ERROR.toml').exists(),'Finalization held')
    return s

def run(key):
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute-only')
    row,record,original,probe,protected,d,capture=authorities(key)
    out=C.ROOT/key;out.mkdir(parents=True,exist_ok=False)
    C.write(out/'PROTECTED.json',protected)
    C.write(out/'ATTEMPT.json',dict(key=key,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],
        prior_scientific_failures=2,passing_precision_diagnostic_reused=True,
        full_recovery_qualification_attempt=1,production_publication=False,automatic_retry=False))
    local=Path(tempfile.mkdtemp(prefix='p1_recovery_'+os.environ['SLURM_JOB_ID']+'_',dir='/tmp'))
    C.require(shutil.disk_usage(local).free>4*1024**3,'Local headroom')
    W,science=stage_runtime(local)
    def stage(args,name,env=None,timeout=5400):
        with (out/name).open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=local,env=env,check=True,timeout=timeout)
    env=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_PKG_OFFLINE='true',JULIA_PKG_PRECOMPILE_AUTO='0',
             JULIA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',JULIA_DEPOT_PATH=str(local/'depot')+':'+W.DEPOT)
    julia=[W.JULIA,'--startup-file=no','--history-file=no','--threads=2','--gcthreads=1','--project='+W.PROJECT]
    try:
        stage(['/usr/bin/python3.12','-B',str(science/'runtime_check.py')],'runtime.log')
        stage(julia+[str(C.CODE/'test_resume.jl')],'test_resume.log',env,timeout=120)
        stage(julia+[str(C.CODE/'test_gates.jl')],'test_gates.log',env,timeout=120)
        stage(julia+[str(C.CODE/'test_extra.jl')],'test_extra.log',env,timeout=120)
        q=check_probe(d);rows=saved_rows((original/'flow/flow_calls.jsonl').read_text())
        C.require(rows[-1]['sample']==capture['sample'],'Failed state mismatch')
        work=out/'work';(work/'input').mkdir(parents=True);(work/'recovery').mkdir()
        for n in ('record.json','SOURCE.json','INPUT_PASS.toml','REPLAY_PASS.json'):shutil.copyfile(original/'input'/n,work/'input'/n)
        shutil.copyfile(original/'flow/flow_calls.jsonl',work/'recovery/original_flow_calls.jsonl')
        shutil.copyfile(original/'FLOW_ERROR.toml',work/'recovery/original_FLOW_ERROR.toml')
        for n in ('CAPTURE.json','PROBE_RESULT.json','INPUT_MATCH.json'):shutil.copyfile(probe/n,work/'recovery'/n)
        precision=work/'recovery/precision';precision.mkdir()
        for n in C.read(C.CODE/'SCOPE.json')['proofs'][key]:
            shutil.copyfile(scope.DIAG/'output'/key/n,precision/n)
        C.write(work/'RECOVERY.json',dict(source=C.sha(C.CODE/'LOCK.json'),probe_source=scope.DIAG_PIN,
            original_input_source_preserved=True,scientific_method=METHOD,scope=key,production_modified=False))
        saved=write_toml(work/'recovery/saved_calls.toml',dict(probe_sample=capture['sample'],probe_q=q,
            parts_hash=capture['parts_hash'],state_hash=capture['state_hash'],probe_sha256=C.sha(precision/'RESULT.toml'),
            precision=str(precision)),rows)
        req=write_toml(out/'recover.toml',dict(science=str(science),work=str(work),key=key,slot=row['slot'],sample=int(record['assignment']['clay_pc_sample_id']),
            record_sha256=C.digest(record),method=METHOD,source=C.sha(C.CODE/'LOCK.json'),saved=str(saved),
            raw=str(probe/'raw_planned.bin'),geometry=str(probe/'geometry_unplanned.bin')))
        stage(julia+[str(C.CODE/'recover_flow.jl'),str(req)],'flow.log',env)
        C.require((work/'flow/CORE_PASS.toml').exists(),'Full flow not complete; saved output preserved')
        ready=[dict(key=key,work=str(work),record_sha256=C.digest(record))]
        C.write(out/'export.json',dict(science=str(science),method=METHOD,cases=ready))
        stage([W.PYTHON,'-B',str(R3/'export_packet.py'),str(out/'export.json')],'export.log')
        C.require((work/'export/EXPORT_PASS.json').exists(),'Export held; never recalculate flow')
        req=W.toml(out/'native.toml',dict(science=str(science),method=METHOD),ready)
        nativeenv=dict(env,JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JULIA_DEPOT_PATH=str(local/'native_depot')+':'+W.NATIVE_D)
        stage([W.NATIVE_J,'--startup-file=no','--threads=1','--project='+W.NATIVE_P,str(R3/'native_packet.jl'),str(req)],'native.log',nativeenv)
        s=validate_final(work,record,rows)
        audit=tomllib.loads((work/'recovery/PRECISION_REAUDIT.toml').read_text())
        C.require(audit['passed'] and audit['new_solves']==0 and audit['reference_q']==q,
                  'Saved pressure re-audit required')
        archive=out/'verified_core.zip';receipt=C.publish(work,record,METHOD,archive)
        C.verify_archive(archive,record,METHOD)
        # A second independent read of every archived member precedes qualification.
        C.require(C.archive_receipt(archive,record,METHOD)==receipt,'Archive verification mismatch')
        authorities(key);C.verify_source()
        C.write(out/'QUALIFIED_PASS.json',dict(passed=True,key=key,source=C.sha(C.CODE/'LOCK.json'),method=METHOD,
            job=os.environ['SLURM_JOB_ID'],archive=str(archive),receipt=receipt,new_flow_calls=s['new_flow_calls'],
            saved_calls_reused=s['saved_flow_calls_reused'],probe_calls_reused=s['probe_calls_reused'],
            completed_calls_repeated=0,saved_precision_solves_repeated=0,production_writes=0,accepted_production_cores=0,
            saved_precision_reaudit=audit,
            native_sha256=C.sha(work/'NATIVE_PASS.toml'),summary=s))
        print('END_TO_END_QUALIFIED',key,flush=True)
    finally:
        for f,h in protected.items():C.require(C.sha(f)==h,'Original or diagnostic evidence modified')
        C.write(out/'PRESERVED.json',dict(files_unchanged=len(protected),production_writes=0))

if __name__=='__main__':
    run(TARGETS[int(sys.argv[1])])
