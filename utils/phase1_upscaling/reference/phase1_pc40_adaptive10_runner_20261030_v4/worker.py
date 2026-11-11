"""One finite packet, warm stages, durable per-core results and no scheduler."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import zipfile
import common as C
import runtime_snapshot
import planning
from cache_contract import verify_cache_evidence

DATA=Path('/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7/inputs/predict')
PHYSICS='/orcd/data/juanes/001/shaowen/predict_shaowen/orchestration/independent_full_fault_v1_04927f5/physics_repo'
MRST='/orcd/pool/006/shaowen/predict_shaowen/software/mrst-2025a/SINTEF-AppliedCompSci-MRST-75749fa'
MATLAB='/orcd/software/core/001/pkg/matlab/R2025b/bin/matlab'
RUNTIME='/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/wrr60_phase1_production_candidate_20260912_v1'
JULIA='/orcd/software/core/001/pkg/julia/1.12.6/bin/julia'
PROJECT=RUNTIME+'/warm_worker_qualification/20260915_v2/environment'
DEPOT=RUNTIME+'/warm_worker_qualification/20260915_v2/runs/cache/depot'
PYTHON=RUNTIME+'/reference_integration_v1/integration_v1/environments/export_py310_v1/bin/python'
NATIVE_J='/orcd/software/community/001/pkg/julia/1.10.4/bin/julia'
NATIVE_P='/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/corrected_hysteresis_15case_20260911_v1/runtime'
NATIVE_D='/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/wrr60_pc100_kr10_campaign_20260916_v1/full_fault_dispatch_v2/depots/production_1104:/home/shaowen/.julia'

def process(args,log,fd,env=None,cwd=None):
    start=time.monotonic()
    with log.open('x') as stream:
        p=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=cwd,pass_fds=(fd,),start_new_session=True)
        try:code=p.wait(timeout=6900)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    print('STAGE_EXIT',log.name,code,'seconds',time.monotonic()-start,flush=True)
    return code

def toml(path,header,rows):
    lines=[k+' = '+json.dumps(v) for k,v in header.items()]
    for r in rows:lines+=['[[cases]]']+[k+' = '+json.dumps(v) for k,v in r.items()]
    with path.open('x') as f:f.write('\n'.join(lines)+'\n')
    return path

def allocation():
    raw=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
    d=dict(x.split('=',1) for x in raw.split() if '=' in x)
    for k,v in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='2',Requeue='0').items():C.require(d.get(k)==v,'Allocation '+k)
    C.require(d['MinMemoryNode'] in ('18G','18432M') and d.get('ArrayTaskThrottle','0')=='0','Memory/throttle')
    return d

def selection(r,checkpoint):
    a=r['assignment'];s=r['geology'];m=r['source_metadata']
    d={k:s[k] for k in ('geology_id','scenario_index','scenario_label','scenario_name','case_index','case_label','faulting_depth_m','sand_vcl','clay_vcl')}
    d.update(task_id=C.key(r),task_key_sha256=C.digest(r),case_id=a['replicate_id'],case_name=r['case_id'],case_category='Probabilistic UQ',
        window=a['window_id'],slice_index=a['slice_id'],assigned_state='independent',sampling_mode='independent',sampling_pool='full_distribution',
        selected_sample_index=a['source_library_row'],exact_replay_seed=a['exact_replay_seed'],source_checkpoint_file=str(checkpoint),
        source_seed_base=m['SeedBase'],source_num_attempts=m['NumAttempts'],source_num_rejected=m['NumRejected'])
    d.update({k:a[k] for k in ('log_kxx','log_kyy','log_kzz')})
    return d

def flow_proof(work,r,method):
    C.require(C.read(work/'input/record.json')==r,'Saved input identity')
    source=C.read(work/'input/SOURCE.json')
    C.require(source['method']==method and source['configuration']==r['configuration_sha256'],'Saved scientific configuration')
    p=tomllib.loads((work/'flow/CORE_PASS.toml').read_text())
    C.require(p['passed'] and p['record_sha256']==C.digest(r) and p['method']==method,'Saved flow identity')
    C.require(p['input']==tomllib.loads((work/'input/INPUT_PASS.toml').read_text()),'Input proof changed')
    C.require(p['input']['sample']==int(r['assignment']['clay_pc_sample_id']) and p['input']['contact_angle_deg']==40,'Saved profile')
    for n,h in p['artifacts'].items():C.require(C.sha(work/'flow'/n)==h,'Saved flow checksum')
    s=tomllib.loads((work/'flow/summary.toml').read_text());life=s['native_lifetime']
    C.require(s['mode']=='adaptive10' and s['same_mip_microstates'] and s['pc_points']==100 and 2<=s['kr_states']<=10,'Flow method')
    C.require(s['phase_solver_calls']<=20 and s['intrinsic_solver_calls']==1,'Solve budget')
    C.require(life['numeric_created']==life['numeric_released'] and life['symbolic_created']==life['symbolic_released'],'Native lifetime')
    verify_cache_evidence(s,p,source,C.sha(C.CODE/'LOCK.json'),C.read(C.CODE/'LOCK.json')['science'])
    return p

def stage_science(local,lock):
    zpath=C.CODE/'science_bundle.zip'
    C.require(C.sha(zpath)==lock['science_bundle_sha256'],'Science bundle changed')
    root=local/'science';root.mkdir()
    with zipfile.ZipFile(zpath) as z:
        C.require(set(z.namelist())==set(lock['science']),'Science bundle inventory')
        for n,h in lock['science'].items():
            b=z.read(n);C.require(hashlib.sha256(b).hexdigest()==h,'Science source changed')
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(b)
    return root

def run(case,index,stop_after=0):
    lock=C.verify_source();method=lock['method_sha256']
    from control import authorize
    scope=C.read(C.ROOT/'SCOPE.json');authorize(scope)
    C.require(case in scope['cases'],'Case outside authorized scope')
    planpath=C.ROOT/'plans'/f'{case}.json'
    plan=planning.checked_plan(C.CODE,C.ROOT,case)
    C.require(0<=index<len(plan['packets']),'Packet outside plan')
    if scope['purpose']=='qualification':C.require(plan['purpose']=='qualification' and index in plan['allowed_packets'],'Qualification packet bound')
    else:C.require(scope['purpose']=='production' and plan['purpose']=='production','Invalid worker scope')
    records=plan['packets'][index];C.require(len(records)<=32 and records,'Packet size')
    b=C.ROOT/'runs'/case;works={C.key(r):b/'cores'/C.key(r) for r in records}
    result=lambda r:C.ROOT/'results'/case/f"core{r['slot']:03d}.zip"
    with C.locked(b/f'packet{index}.lock') as fd:
        allocation()
        actions={C.key(r):C.action(works[C.key(r)],result(r),r,method) for r in records}
        # Recover the small receipt if interrupted after immutable archive publication.
        for r in records:
            if actions[C.key(r)]=='DONE':
                C.write(b/'done'/f"core{r['slot']:03d}.json",C.archive_receipt(result(r),r,method))
        print('INITIAL',json.dumps(actions),flush=True)
        if all(a=='DONE' for a in actions.values()):print('ALREADY_COMPLETE_NO_WORK');return
        attempt=b/'attempts'/f"packet{index}_{os.environ['SLURM_JOB_ID']}";attempt.mkdir(parents=True)
        C.write(attempt/'BEGIN.json',dict(actions=actions,source=C.sha(C.CODE/'LOCK.json')))
        local=Path(tempfile.mkdtemp(prefix='p1_'+os.environ['SLURM_JOB_ID']+'_',dir='/tmp'))
        C.require(shutil.disk_usage(local).free>8*1024**3,'Local disk headroom')
        science=stage_science(local,lock)
        for p,h in C.read(C.CODE/'EXECUTABLE_PINS.json').items():C.require(C.sha(p)==h,'Julia executable changed')
        C.require(process(['/usr/bin/python3.12','-B',str(science/'runtime_check.py')],attempt/'runtime.log',fd)==0,'Pinned runtime changed')
        baseenv=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_PKG_OFFLINE='true',JULIA_PKG_PRECOMPILE_AUTO='0')
        flowenv=dict(baseenv,JULIA_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',JULIA_DEPOT_PATH=str(local/'depot')+':'+DEPOT)
        fresh=[r for r in records if actions[C.key(r)]=='PREPARE']
        candidates=[]
        if fresh:
            for p,h in C.read(C.CODE/'DATA_PINS.json').items():C.require(C.sha(DATA/p)==h,'Replay definition changed: '+p)
            inventory=C.read(C.CODE/'runtime_inventory.json')['inventory']
            stage=runtime_snapshot.stage(C.read(C.CODE/'RUNTIME_BUNDLE.json'),inventory,local/'runtime')
            def staged_root(original):
                name=next(p for p in stage['mapping'] if p.startswith(original+'/'))
                return stage['mapping'][name][:-len(name[len(original):])]
            rows=[];checked={}
            for r in fresh:
                k=C.key(r);w=works[k];(w/'input').mkdir(parents=True,exist_ok=True)
                C.write(w/'input/record.json',r)
                C.write(w/'input/SOURCE.json',dict(runner_source=C.sha(C.CODE/'LOCK.json'),method=method,configuration=r['configuration_sha256']))
                cp=DATA/r['assignment']['checkpoint_relative_path'];h=r['assignment']['checkpoint_hash']
                if str(cp) not in checked:C.require(C.sha(cp)==h,'Checkpoint library changed');checked[str(cp)]=h
                C.require(checked[str(cp)]==h,'Conflicting checkpoint identity')
                rawdir=local/k;rawdir.mkdir()
                sel=selection(r,cp)
                with (rawdir/'selection.csv').open('x',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(sel));writer.writeheader();writer.writerow(sel)
                rows.append(dict(key=k,slot=r['slot'],sample=int(r['assignment']['clay_pc_sample_id']),selection=str(rawdir/'selection.csv'),
                    local=str(rawdir),rawdir=str(rawdir/'raw'),status=str(rawdir/'PREPARE.json')))
            request=attempt/'prepare.json'
            C.write(request,dict(code=str(C.CODE),science=str(science),physics=staged_root(PHYSICS),mrst=staged_root(MRST),data=str(DATA),cores=rows,loaded=str(local/'loaded.txt')))
            env=dict(os.environ,TMPDIR=str(local),TMP=str(local),TEMP=str(local),MATLAB_PREFDIR=str(local/'preferences'),MCR_CACHE_ROOT=str(local/'mcr'),OMP_NUM_THREADS='2')
            code=process([MATLAB,'-batch',f"addpath('{C.CODE}');prepare_packet('{request}');"],attempt/'prepare.log',fd,env,local)
            C.require(code==0 and (local/'loaded.txt').exists(),'Global MATLAB preparation/runtime failure')
            stagedpins={v:inventory[k] for k,v in stage['mapping'].items()}
            for p in (local/'loaded.txt').read_text().splitlines():
                if p in stagedpins:C.require(C.sha(p)==stagedpins[p],'Loaded runtime changed')
                elif Path(p).is_relative_to(C.CODE):
                    name=Path(p).relative_to(C.CODE).as_posix();C.require(name in lock['code'] and C.sha(p)==lock['code'][name],'Unexpected runner function')
                elif Path(p).is_relative_to(science):
                    name=Path(p).relative_to(science).as_posix();C.require(name in lock['science'] and C.sha(p)==lock['science'][name],'Unexpected scientific function')
                else:C.require(Path(p).is_relative_to(Path(MATLAB).parent.parent),'Unexpected loaded MATLAB source: '+p)
            for p,h in checked.items():C.require(C.sha(p)==h,'Checkpoint changed during replay')
            for r,row in zip(fresh,rows):
                w=works[C.key(r)];status=C.read(row['status'])
                if not status['passed']:
                    C.write(w/'HOLD.json',status);continue
                C.require(int(status['seed'])==int(r['assignment']['predict_seed']) and int(status['source_row'])==int(r['assignment']['source_library_row']),'Replay identity')
                prior=w/'input/REPLAY_PASS.json'
                if prior.exists():
                    old=C.read(prior)
                    C.require({k:v for k,v in old.items() if k!='seconds'}=={k:v for k,v in status.items() if k!='seconds'},'Replay changed on unstarted core')
                else:C.write(prior,status)
                candidates.append(dict(key=C.key(r),slot=r['slot'],sample=int(r['assignment']['clay_pc_sample_id']),work=str(w),record_sha256=C.digest(r),
                    raw=str(Path(row['rawdir'])/'raw_unplanned.bin'),geometry=str(Path(row['rawdir'])/'geometry_unplanned.bin'),rawdir=row['rawdir']))
            if candidates:
                req=toml(attempt/'flow.toml',dict(science=str(science),method=method,stop_after=stop_after),candidates)
                process([JULIA,'--startup-file=no','--history-file=no','--threads=2','--gcthreads=1','--project='+PROJECT,str(C.CODE/'flow_packet.jl'),str(req)],attempt/'flow.log',fd,flowenv)
            if stop_after:
                C.write(attempt/'CHECKPOINT_STOP.json',dict(stop_after=stop_after,flow_receipts=sum((w/'flow/CORE_PASS.toml').exists() for w in works.values())))
                return
        ready=[]
        for r in records:
            w=works[C.key(r)]
            if C.action(w,result(r),r,method)=='FINISH':
                flow_proof(w,r,method)
                ready.append(dict(key=C.key(r),work=str(w),record_sha256=C.digest(r)))
        if ready:
            req=attempt/'export.json';C.write(req,dict(science=str(science),method=method,cases=ready))
            C.require(process([PYTHON,'-B',str(C.CODE/'export_packet.py'),str(req)],attempt/'export.log',fd)==0,'Export failed; saved flow retained')
            ready=[r for r in ready if (Path(r['work'])/'export/EXPORT_PASS.json').exists() and not (Path(r['work'])/'POSTFLOW_HOLD.json').exists()]
            req=toml(attempt/'native.toml',dict(science=str(science),method=method),ready)
            env=dict(baseenv,JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JULIA_DEPOT_PATH=str(local/'native_depot')+':'+NATIVE_D)
            if ready:C.require(process([NATIVE_J,'--startup-file=no','--threads=1','--project='+NATIVE_P,str(C.CODE/'native_packet.jl'),str(req)],attempt/'native.log',fd,env)==0,'Native import failed; saved flow retained')
            for r in records:
                w=works[C.key(r)]
                if C.action(w,result(r),r,method)!='FINISH':continue
                if not (w/'NATIVE_PASS.toml').exists():
                    C.write(w/'POSTFLOW_HOLD.json',dict(stage='native',reason='Native receipt absent; saved flow retained'));continue
                flow_proof(w,r,method)
                ex=C.read(w/'export/EXPORT_PASS.json');native=tomllib.loads((w/'NATIVE_PASS.toml').read_text())
                C.require(native['record_sha256']==C.digest(r) and native['method']==method,'Native receipt identity')
                C.require(ex['passed'] and native['passed'] and C.sha(w/'export/sgof.csv')==ex['sha256']==native['sgof_sha256'],'Final import identity')
                proof=C.publish(w,r,method,result(r))
                C.write(b/'done'/f"core{r['slot']:03d}.json",proof)
                # Only this worker's private, fully archived per-core checkpoint is retired.
                C.verify_archive(result(r),r,method)
                C.require(w.resolve().is_relative_to((b/'cores').resolve()) and not w.is_symlink(),'Unsafe private retirement')
                shutil.rmtree(w)
        final={C.key(r):C.action(works[C.key(r)],result(r),r,method) for r in records}
        C.verify_source()
        C.write(attempt/'SUMMARY.json',dict(initial=actions,final=final,verified=sum(x=='DONE' for x in final.values()),expected=len(records),
            method=method,source=C.sha(C.CODE/'LOCK.json'),full_campaign_launched=False))
        print('PACKET_END',json.dumps(final),flush=True)
        if all(x=='DONE' for x in final.values()):
            C.require(local.parent==Path('/tmp') and local.name.startswith('p1_') and not local.is_symlink(),'Unsafe scratch retirement')
            shutil.rmtree(local)

def packet_index(value=None):
    array=os.environ.get('SLURM_ARRAY_TASK_ID')
    C.require(value is not None or array is not None,'Packet index required')
    index=int(array) if value is None else value
    C.require(index>=0 and (array is None or index==int(array)),'Conflicting packet index')
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--packet',type=int,default=None);p.add_argument('--stop-after',type=int,default=0)
    a=p.parse_args();run(a.case,packet_index(a.packet),a.stop_after)
