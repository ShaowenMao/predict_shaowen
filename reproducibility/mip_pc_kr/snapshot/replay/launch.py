"""Campaign wave 1: exact original-draw replay, with numerical dispatch held.

No scientific source modifications, no reservoir submission, no cancellation.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, math, os, re, shutil, signal, sqlite3, struct
import subprocess, sys, time, traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from assignment_plan import plan_case
from temp_paths import TEMP_ROOT, create, validate_environment
from dispatch_checks import memory_gib, unique_job

HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'spec.json').read_text()) if (HERE/'spec.json').exists() else {}
ROOT=Path(SPEC.get('root','/orcd/data/juanes/001/shaowen/predict_shaowen/campaigns/wrr60_phase1_2430_20260914_v1'))
QOS='mit_amf_advanced_cpu'
BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen')
PROD=BASE/'production_runs/independent_full_fault_v1_phase1_20260809_v1'
DATA=BASE/'production_freezes/collapsed_cell_union_20260722_v7/inputs/predict'
PHYSICS=BASE/'orchestration/independent_full_fault_v1_04927f5/physics_repo'
MRST='/orcd/pool/006/shaowen/predict_shaowen/software/mrst-2025a/SINTEF-AppliedCompSci-MRST-75749fa'
MATLAB='/orcd/software/core/001/pkg/matlab/R2025b/bin/matlab'
NY=ZoneInfo('America/New_York')

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x,replace=False):
    p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.partial')
    with tmp.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    if replace:os.replace(tmp,p)
    else:os.link(tmp,p);tmp.unlink()
def event(name,**d):print(json.dumps(dict(event=name,utc=datetime.now(NY).isoformat(),**d)),flush=True)
def verify():
    lock=read(HERE/'source_lock.json')
    for n,h in lock['files'].items():assert sha(HERE/n)==h,n
    return lock
def cutoff_epoch(text):return datetime.fromisoformat(text).replace(tzinfo=NY).timestamp()
def seconds_allowed(now,cutoff,requested=10800,margin=1800):
    return max(0,min(requested,int(cutoff-now-margin)//60*60))
def resource_for(n):
    if not 1<=n<=150000:return None
    return 4 if n<=30000 else 5 if n<=130000 else 6
def batch_records(records,limit=24):
    groups=defaultdict(list)
    for r in records:groups[(r['case_key'],r['assignment']['window'],r['memory_gib'])].append(r)
    batches=[]
    for _,rs in sorted(groups.items()):
        rs.sort(key=lambda r:r['slot'])
        for i in range(0,len(rs),limit):
            part=rs[i:i+limit]
            batches.append(dict(index=len(batches),memory_gib=part[0]['memory_gib'],cpus=2,records=part))
    assert sum(len(b['records']) for b in batches)==len(records)
    return batches
def allocation(job,cpus,mem):
    raw=subprocess.check_output(['scontrol','show','job','-o',str(job)],text=True,timeout=45)
    fields=dict(v.split('=',1) for v in raw.split() if '=' in v)
    for k,v in dict(Account=QOS,QOS=QOS,Partition='mit_normal',NumCPUs=str(cpus),MinMemoryNode=str(mem)+'G',Requeue='0').items():assert fields[k]==v,(k,fields.get(k),v)
    return raw
def memory(job,allocation_gib):
    roots=list(Path('/sys/fs/cgroup/memory/slurm').glob('uid_*/job_'+job))
    assert len(roots)==1
    root=roots[0];peak=int((root/'memory.max_usage_in_bytes').read_text())/1024**3
    limit=int((root/'memory.limit_in_bytes').read_text())/1024**3
    events=int((root/'memory.failcnt').read_text());required=max(peak+1,1.3*peak)
    return dict(peak_gib=peak,limit_gib=limit,events=events,required_gib=required,passed=limit==allocation_gib and events==0 and required<=limit)
def storage_probe(out):
    out.mkdir(parents=True,exist_ok=True);p=out/f'probe_{os.getpid()}.tmp'
    data=os.urandom(1024**2);start=time.monotonic()
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    dest=p.with_suffix('.ok');p.rename(dest)
    assert sha(dest)==hashlib.sha256(data).hexdigest();dest.unlink()
    free=shutil.disk_usage(out).free/1024**3
    assert free>=2048,'Less than 2 TiB project disk headroom'
    return dict(passed=True,seconds=time.monotonic()-start,free_gib=free,probe_bytes=len(data),global_storage_or_quota_certified=False)

def plan():
    verify();job=os.environ['SLURM_JOB_ID'];allocation(job,2,4)
    manifest=read(HERE/'campaign.json')
    assert len(manifest['cases'])==2430 and SPEC['campaign_authorized']
    selected=read(HERE/'SELECTION.json')['entries'];assert len(selected)==12
    assert sha(SPEC['inventory'])==SPEC['inventory_sha256']
    db=sqlite3.connect('file:'+SPEC['inventory']+'?mode=ro&immutable=1',uri=True)
    records=[];held=[];source_hashes={};plans=[]
    frozen_cases={c['case_key']:c for c in manifest['cases']}
    for entry in selected:
        c=entry['case'];assert c==frozen_cases[c['case_key']]
        key=c['case_key'];geo=c['geology_id'];num=c['realization_id']
        src=PROD/f'case_work_manifest/geology_assignments/{geo}.csv'
        rr=[r for r in rows(src) if int(r['case_id'])==num]
        assert len(rr)==522
        inp=PROD/f'case_inputs/cases/{geo}/case{num:02d}'
        receipt=read(inp/'case_inputs.done.json')
        assert receipt['status']=='assembled' and receipt['assignment_count']==522
        assert receipt['case_id']==num and receipt['geology_id']==geo and receipt['case_name']==c['level3_case_name']
        name=f'pc_curve_summary_{key}_ip_full87.csv';summary_file=inp/'inputs'/name
        assert sha(summary_file)==receipt['files'][name]['sha256']
        summaries={(r['Window'],int(r['SliceIndex'])):r for r in rows(summary_file)}
        assert len(summaries)==522 and sha(c['specific_path'])==c['specific_sha256']
        expected=db.execute('SELECT window,slice,region,sample,seed FROM assignments WHERE task=? ORDER BY window,slice',(c['task'],)).fetchall()
        actual=sorted((int(r['window'][4:]),int(r['slice_index']),(int(r['window'][4:])-1)*87+int(r['slice_index'])+8,int(r['selected_sample_index']),int(r['exact_replay_seed'])) for r in rr)
        assert actual==expected
        recs=[]
        for row in sorted(rr,key=lambda r:(r['window'],int(r['slice_index']))):
            slot=(int(row['window'][4:])-1)*87+int(row['slice_index'])
            s=summaries[(row['window'],int(row['slice_index']))]
            recs.append(dict(slot=slot,assignment=row,production_summary=s,checkpoint_sha256=row['checkpoint_sha256']))
        pc=plan_case(c,recs,SPEC['sources'],read(HERE/'policy.json'))
        plans.append(dict(case=c,placement_identity=digest(pc['placements']),placements=pc['placements']))
        name=f'pc_native_curve_points_{key}_ip_full87.csv';native=inp/'inputs'/name
        assert sha(native)==receipt['files'][name]['sha256']
        pressure=defaultdict(set)
        for r in rows(native):pressure[(r['Window'],int(r['SliceIndex']))].add(float(r['PcPa']))
        for r in recs:
            s=r['production_summary'];row=r['assignment'];n=int(s['NumCells']);mem=resource_for(n)
            checkpoint=DATA/row['checkpoint_relative_path']
            assert checkpoint.resolve().is_relative_to(DATA.resolve())
            h=row['checkpoint_sha256']
            if str(checkpoint) not in source_hashes:assert sha(checkpoint)==h;source_hashes[str(checkpoint)]=h
            else:assert source_hashes[str(checkpoint)]==h
            entry=dict(case_key=key,case=c,**r,checkpoint=str(checkpoint),memory_gib=mem,
                prepare_summary=dict(slot=r['slot'],NumCells=n,PoreVolume=float(s['PoreVolume']),BulkVolume=float(s['BulkVolume']),production_pressure_pa=sorted(pressure[(row['window'],int(row['slice_index']))])))
            assert len(entry['prepare_summary']['production_pressure_pa'])>=2
            (records if mem else held).append(entry)
        for p in (src,summary_file,native,inp/'case_inputs.done.json'):source_hashes[str(p)]=sha(p)
        source_hashes[c['specific_path']]=c['specific_sha256']
        event('CASE_PLANNED',case=key,placements=522)
    db.close()
    inv=read(HERE/'runtime_inventory.json')['inventory']
    for p,h in inv.items():assert sha(p)==h,p
    probe=storage_probe(ROOT/'storage_probes'/('plan'+job))
    batches=batch_records(records)
    mem=memory(job,4);assert mem['passed'];verify()
    p=ROOT/'wave01'/'PLAN.json'
    result=dict(case_count=12,placements=len(records)+len(held),admitted_replay_placements=len(records),held=held,
        batches=batches,cases=plans,source_hashes=source_hashes,storage=probe,memory=mem,
        numerical_dispatch_approved=False,root=str(ROOT),source_lock_sha256=sha(HERE/'source_lock.json'),
        intended_scenario='WRR60, physical prebreakthrough, dip-parallel MIP Kr',
        stage='same_draw_geometry_replay',campaign_manifest_sha256=SPEC['campaign_manifest_sha256'])
    atomic(p,result);atomic(p.parent/'PLAN_ACCEPTED.json',dict(job=job,plan_sha256=sha(p),passed=True,source_lock_sha256=sha(HERE/'source_lock.json')))
    event('PLAN_ACCEPTED',batches=len(batches),replay=len(records),held=len(held))

def get_plan():
    p=ROOT/'wave01'/'PLAN.json';done=read(p.parent/'PLAN_ACCEPTED.json')
    assert done['passed'] and sha(p)==done['plan_sha256'] and done['source_lock_sha256']==sha(HERE/'source_lock.json')
    return read(p)
def core_receipt(key,slot):return ROOT/'wave01'/'accepted'/key/f'core{slot:03d}.json'
def accepted_core(r):
    p=core_receipt(r['case_key'],r['slot'])
    if not p.exists():return None
    d=read(p);assert d['record_sha256']==digest(r) and d['source_lock_sha256']==sha(HERE/'source_lock.json')
    assert d['memory']['passed'] and d['complete']
    folder=Path(d['output']);assert folder.resolve().is_relative_to(ROOT.resolve())
    for n,h in d['files'].items():assert sha(folder/n)==h,n
    return d
def run_matlab(expr,log,out,timeout=900):
    start=time.monotonic()
    with log.open('x') as f:
        p=subprocess.Popen([MATLAB,'-batch',expr],cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    assert code==0,str(log)
    return time.monotonic()-start
def worker(index,canary=False):
    verify();job=os.environ['SLURM_JOB_ID'];plan=get_plan();b=plan['batches'][index]
    allocation(job,2,b['memory_gib'])
    out=ROOT/'wave01'/'attempts'/f'batch{index:04d}_job{job}';out.mkdir(parents=True,exist_ok=False)
    paths=create(TEMP_ROOT,job,HERE.name)
    for k,n in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('TEMPDIR','tmp'),('MATLAB_PREFDIR','preferences'),('MCR_CACHE_ROOT','cache')]:os.environ[k]=str(paths[n])
    validate_environment(TEMP_ROOT,job,HERE.name,os.environ)
    inv=read(HERE/'runtime_inventory.json')['inventory']
    try:
        for p,h in inv.items():assert sha(p)==h,p
        inputs={r['checkpoint']:r['checkpoint_sha256'] for r in b['records']}
        for p,h in inputs.items():assert sha(p)==h,p
        probe=storage_probe(out/'probe')
        run_matlab(f"addpath('{HERE}');verify_matlab_startup('{out}/startup.json');",out/'startup.log',out)
        assert read(out/'startup.json')['passed'] and read(out/'startup.json')['matlab_version']==SPEC['matlab_version']
        done=[]
        for r in b['records'][:2] if canary else b['records']:
            old=accepted_core(r)
            if old:done.append(dict(slot=r['slot'],reused=True));continue
            if time.time()>cutoff_epoch(SPEC['maintenance_start'])-1800:
                event('MAINTENANCE_STOP_BETWEEN_CORES');break
            core=out/f'core{r["slot"]:03d}';core.mkdir()
            with (core/'selection.csv').open('x',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(r['assignment']));w.writeheader();w.writerow(r['assignment'])
            atomic(core/'summary.json',r['prepare_summary'])
            event('REPLAY_START',case=r['case_key'],slot=r['slot'],cells=r['prepare_summary']['NumCells'])
            expr=f"addpath('{HERE}/frozen');prepare_slot('{core}/selection.csv','{core}/summary.json','{core}/prepared','{PHYSICS}','{DATA}','{MRST}');addpath('{HERE}');R=jsondecode(fileread('{core}/prepared/preparation.json'));features(R.replay_file,'{core}/features.json');"
            seconds=run_matlab(expr,core/'replay.log',core)
            prep=read(core/'prepared/preparation.json');feat=read(core/'features.json');row=r['assignment']
            assert prep['num_cells']==feat['num_cells']==r['prepare_summary']['NumCells']
            assert prep['sample']==int(row['selected_sample_index']) and prep['seed']==int(row['exact_replay_seed'])
            assert prep['slot']==r['slot'] and prep['geology']==r['case']['geology_id'] and prep['window']==row['window']
            assert prep['max_log10_replay_difference']<=0.005 and len(prep['architecture_sha256'])==64
            with (core/'prepared/geometry_unplanned.bin').open('rb') as f:header=struct.unpack('<7q',f.read(56));length,area=struct.unpack('<2d',f.read(16))
            assert header[:4]==(2026090906,r['slot'],3,prep['num_cells']) and length>0 and area>0
            loaded={}
            for name in (core/'prepared/loaded_matlab_files.txt').read_text().splitlines():
                p=Path(name)
                if p.is_relative_to(HERE):expected=read(HERE/'source_lock.json')['files'][p.relative_to(HERE).as_posix()]
                elif name in inv:expected=inv[name]
                else:assert p.is_relative_to(Path(MATLAB).parent.parent),name;continue
                assert sha(p)==expected,name;loaded[name]=expected
            mem=memory(job,b['memory_gib']);atomic(core/'MEMORY.json',mem)
            assert mem['passed'],'Replay completed; memory headroom requires targeted review'
            files={p.relative_to(core).as_posix():sha(p) for p in core.rglob('*') if p.is_file()}
            record=dict(complete=True,job=job,case=r['case_key'],slot=r['slot'],record_sha256=digest(r),source_lock_sha256=sha(HERE/'source_lock.json'),
                output=str(core),files=files,preparation=prep,features=feat,memory=mem,wall_seconds=seconds,
                stage='REPLAY_ACCEPTED_PENDING_PC_KR',numerical_dispatch_approved=False,loaded_sources=loaded)
            atomic(core_receipt(r['case_key'],r['slot']),record)
            done.append(dict(slot=r['slot'],reused=False,wall_seconds=seconds));event('CORE_ACCEPTED',case=r['case_key'],slot=r['slot'],seconds=seconds)
        for p,h in inputs.items():assert sha(p)==h,p
        for p,h in inv.items():assert sha(p)==h,p
        verify();mem=memory(job,b['memory_gib']);assert mem['passed']
        atomic(out/'COMPLETE.json',dict(job=job,index=index,canary=canary,completed=done,memory=mem,storage=probe,
            source_lock_sha256=sha(HERE/'source_lock.json'),all_batch_cores_accepted=all(core_receipt(r['case_key'],r['slot']).exists() for r in b['records'])))
        event('BATCH_FINISHED',index=index,cores=len(done),canary=canary)
    except BaseException:
        atomic(out/'FAILURE.json',dict(job=job,index=index,error=traceback.format_exc(),memory=memory(job,b['memory_gib'])));raise

def usage():
    limit=subprocess.check_output(['sacctmgr','-nP','show','qos',QOS,'format=Name,MaxTRESPU'],text=True,timeout=45)
    assert limit.strip().split('|')[:2]==[QOS,'cpu=512,mem=2T']
    raw=subprocess.check_output(['squeue','-h','-r','-u','shaowen','-A',QOS,'-o','%i|%C|%m'],text=True,timeout=45)
    c=0;m=0.
    for line in raw.splitlines():
        j,cpus,mem=line.split('|');assert '[' not in j;c+=int(cpus);m+=memory_gib(mem,int(cpus))
    return c,m,raw
def find_job(name):
    q=subprocess.check_output(['squeue','-h','-u','shaowen','-n',name,'-o','%i|%j'],text=True,timeout=45)
    a=subprocess.check_output(['sacct','-XnP','-S','2026-09-14','--name',name,'-o','JobIDRaw,JobName%100,State'],text=True,timeout=45)
    return unique_job(name,(q,a))
def submit(mode,index,cpus,mem,seconds,claim):
    assert seconds>=600
    name=f'w60w1_{HERE.name[:6]}_{mode}_{index}'
    if claim.exists():
        d=read(claim)
        if d.get('job'):return d['job']
        found=find_job(name)
        assert found,'Uncertain submission must be reconciled, not repeated'
        d.update(job=found,reconciled=True);atomic(claim,d,replace=True);return found
    atomic(claim,dict(name=name,mode=mode,index=index,source_lock=sha(HERE/'source_lock.json'),intent_time=time.time()))
    cmd=['sbatch','--parsable',f'--account={QOS}',f'--qos={QOS}','--partition=mit_normal','--nodes=1','--ntasks=1',
        f'--cpus-per-task={cpus}',f'--mem={mem}G',f'--time={math.ceil(seconds/60)}','--no-requeue',f'--job-name={name}',
        f'--output={ROOT}/logs/%j.log',str(HERE/'job.sh'),str(HERE),mode,str(index)]
    job=subprocess.check_output(cmd,text=True,timeout=60).strip().split(';')[0];assert job.isdigit()
    d=read(claim);d.update(job=job,command=cmd,actual=allocation(job,cpus,mem));atomic(claim,d,replace=True)
    event('SUBMITTED',job=job,mode=mode,index=index,cpus=cpus,memory_gib=mem);return job
def controller():
    import fcntl
    verify();plan=get_plan();ROOT.joinpath('logs').mkdir(exist_ok=True)
    with (ROOT/'wave01/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Canary jobs are genuine work; their accepted cores are skipped in bulk.
        for i in (0,len(plan['batches'])//2):
            claim=ROOT/'wave01/submissions'/f'canary{i}.json'
            assert claim.exists()
            job=read(claim)['job'];p=ROOT/'wave01/attempts'/f'batch{i:04d}_job{job}'/'COMPLETE.json'
            d=read(p);assert d['canary'] and len(d['completed'])==2 and d['memory']['passed']
            for r in plan['batches'][i]['records'][:2]:assert accepted_core(r)
        while True:
            budget=seconds_allowed(time.time(),cutoff_epoch(SPEC['maintenance_start']))
            if budget<900:event('CONTROLLER_MAINTENANCE_PAUSE');break
            c,m,raw=usage();submitted=0
            if shutil.disk_usage(ROOT).free<2048*1024**3:raise RuntimeError('Storage backpressure')
            for b in plan['batches']:
                i=b['index'];claim=ROOT/'wave01/submissions'/f'batch{i}.json'
                if claim.exists():
                    if not read(claim).get('job'):submit('worker',i,2,b['memory_gib'],budget,claim)
                    continue
                if all(core_receipt(r['case_key'],r['slot']).exists() for r in b['records']):continue
                if c+2>500 or m+b['memory_gib']>1984:continue
                submit('worker',i,2,b['memory_gib'],budget,claim);c+=2;m+=b['memory_gib'];submitted+=1
            missing=[b for b in plan['batches'] if not (ROOT/'wave01/submissions'/f'batch{b["index"]}.json').exists() and not all(core_receipt(r['case_key'],r['slot']).exists() for r in b['records'])]
            atomic(ROOT/'wave01/controller_status.json',dict(utc=datetime.now(NY).isoformat(),submitted_this_pass=submitted,
                pending_batches=len(missing),reserved_cpus=c,reserved_memory_gib=m,maintenance_budget_seconds=budget,
                numerical_dispatch_approved=False),replace=True)
            event('CONTROLLER_PASS',submitted=submitted,not_yet_submitted=len(missing),cpus=c,memory=m)
            if not missing:break
            time.sleep(60)
def initial():
    verify();ROOT.mkdir(parents=True,exist_ok=True);ROOT.joinpath('logs').mkdir(exist_ok=True)
    c,m,_=usage();assert c+2<=500 and m+4<=1984
    seconds=seconds_allowed(time.time(),cutoff_epoch(SPEC['maintenance_start']),3600)
    submit('plan',0,2,4,seconds,ROOT/'wave01/submissions/planner.json')
def canaries():
    verify();p=get_plan()
    for i in (0,len(p['batches'])//2):
        b=p['batches'][i];c,m,_=usage();assert c+2<=500 and m+b['memory_gib']<=1984
        submit('canary',i,2,b['memory_gib'],seconds_allowed(time.time(),cutoff_epoch(SPEC['maintenance_start']),3600),ROOT/'wave01/submissions'/f'canary{i}.json')
def start_controller():
    verify();c,m,_=usage();assert c+1<=500 and m+2<=1984
    submit('controller',0,1,2,seconds_allowed(time.time(),cutoff_epoch(SPEC['maintenance_start']),9*3600),ROOT/'wave01/submissions/controller.json')

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='plan':plan()
    elif mode in ('worker','canary'):worker(int(sys.argv[2]),mode=='canary')
    elif mode=='controller':controller()
    elif mode=='initial':initial()
    elif mode=='canaries':canaries()
    elif mode=='start-controller':start_controller()
    else:raise ValueError(mode)
