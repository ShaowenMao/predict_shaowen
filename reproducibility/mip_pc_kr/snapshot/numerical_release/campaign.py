"""Replay-receipt-bound WRR60 numerical campaign; scientific kernels unchanged."""
from pathlib import Path
from collections import defaultdict,Counter
import copy,hashlib,json,math,os,subprocess,sys,time,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from policy import QOS,route,canaries,room,claim_action,gate_status,seconds_allowed

HERE=Path(__file__).resolve().parent
S=json.loads((HERE/'spec.json').read_text()) if (HERE/'spec.json').exists() else {}
CAMPAIGN=Path(S.get('campaign','/orcd/data/juanes/001/shaowen/predict_shaowen/campaigns/wrr60_phase1_2430_20260914_v1'))
ROOT=CAMPAIGN/('numerical_wave01_reuse_v1' if S.get('reuse_parent_root') else 'numerical_wave01')
SOURCE=CAMPAIGN/'releases/5748ac68fdc0be88'
LOCK='49ec2e52cf526d8c40f693c07342021825c57b03048117323c5eb8a6fb4d06ce'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def event(e,**kw):print(json.dumps(dict(event=e,utc=datetime.now(ZoneInfo('UTC')).isoformat(),**kw)),flush=True)
def write(p,d,replace=False):
    p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.partial')
    with tmp.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    if replace:os.replace(tmp,p)
    else:os.link(tmp,p);tmp.unlink()
def library():
    assert sha(SOURCE/'source_lock.json')==LOCK
    sys.path.insert(0,str(SOURCE))
    import launch as L
    # The frozen replay launcher used Python3.12; this pinned export environment
    # is Python3.10. Only substitute equivalent streaming SHA256, not worker logic.
    L.sha=sha
    return L
def pins(external=True):
    for n,h in read(HERE/'source_lock.json')['files'].items():assert sha(HERE/n)==h,n
    assert sha(S['closure_path'])==S['closure_sha256']
    if external:
        for p,h in S['external_files'].items():assert sha(p)==h,p
def context():
    sys.path.insert(0,str(HERE));import context as C
    return C
def accepted_path(key):return ROOT/'accepted'/(key+'.json')
def failed_path(key):return ROOT/'held'/(key+'.json')
def plan():
    done=read(ROOT/'PLAN_ACCEPTED.json');p=ROOT/'PLAN.json'
    assert done['source_lock']==sha(HERE/'source_lock.json') and sha(p)==done['sha256']
    return read(p)
def cutoff():return datetime.fromisoformat(S['maintenance_start']).replace(tzinfo=ZoneInfo('America/New_York')).timestamp()
def allocation(job,cpu,mem):return library().allocation(job,cpu,mem)
def claim_path(label):return ROOT/'submissions'/HERE.name/(label+'.json')

def submit(mode,label,cpu,mem,seconds,arg='',deps=()):
    import fcntl
    folder=claim_path(label).parent;folder.mkdir(parents=True,exist_ok=True)
    with (folder/(label+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=claim_path(label);old=read(p) if p.exists() else None
        L=library();name='w60n1_'+HERE.name[:6]+'_'+label
        if old:
            assert old['name']==name and old['source_lock']==sha(HERE/'source_lock.json')
            j=old.get('job') or L.find_job(name)
            assert j,'Uncertain submission must be reconciled, not retried'
            if not old.get('verified'):
                old.update(job=j,actual=allocation(j,cpu,mem),verified=True);write(p,old,True)
            return j
        used,usedmem,_=L.usage();assert room(used,usedmem,cpu,mem),'Capacity changed; retry controller admission only'
        seconds=seconds_allowed(time.time(),cutoff(),seconds);assert seconds>=900
        d=dict(name=name,mode=mode,arg=arg,cpus=cpu,memory_gib=mem,source_lock=sha(HERE/'source_lock.json'))
        write(p,d)
        args=['sbatch','--parsable','--account='+QOS,'--qos='+QOS,'--partition=mit_normal',
              '--nodes=1','--ntasks=1','--cpus-per-task='+str(cpu),'--mem='+str(mem)+'G',
              '--time='+str(math.ceil(seconds/60)),'--no-requeue','--exclude=node1622',
              '--job-name='+name,'--output='+str(CAMPAIGN/'logs/%j.log')]
        if deps:args.append('--dependency=afterok:'+':'.join(deps))
        args += [str(HERE/'job.sh'),str(HERE),mode,label,str(arg)]
        j=subprocess.check_output(args,text=True,timeout=60).strip().split(';')[0];assert j.isdigit()
        d.update(job=j,command=args);write(p,d,True)
        d.update(actual=allocation(j,cpu,mem),verified=True);write(p,d,True)
        event('SUBMITTED',job=j,mode=mode,label=label,cpus=cpu,memory_gib=mem)
        return j

def bound(label):
    p=claim_path(label);end=time.monotonic()+120
    while (not p.exists() or not read(p).get('verified')) and time.monotonic()<end:time.sleep(.2)
    d=read(p);assert d['verified'] and d['job']==os.environ['SLURM_JOB_ID']
    assert d['source_lock']==sha(HERE/'source_lock.json')
    allocation(d['job'],d['cpus'],d['memory_gib'])
    return d

def validate_replay(r,full=False):
    rp=Path(r['replay_receipt']);assert rp.resolve().is_relative_to((CAMPAIGN/'wave01/accepted').resolve())
    assert sha(rp)==r['replay_receipt_sha256'];d=read(rp)
    assert d['complete'] and d['memory']['passed'] and d['source_lock_sha256']==LOCK
    assert d['record_sha256']==r['record_sha256'] and d['case']==r['case'] and d['slot']==r['slot']
    prep=d['preparation'];assert prep['seed']==r['seed'] and prep['sample']==r['sample']
    assert prep['window']==r['window'] and d['features']==r['features']
    out=Path(d['output']);assert out.resolve().is_relative_to(CAMPAIGN.resolve())
    for name,key in [('prepared/raw_unplanned.bin','raw'),('prepared/geometry_unplanned.bin','geometry')]:
        p=out/name;assert str(p)==r[key] and d['files'][name]==r[key+'_hash']
        if full:assert sha(p)==r[key+'_hash'],str(p)
    return d

def prepare(r,out):
    C=context();out.mkdir(parents=True,exist_ok=False);validate_replay(r,True)
    C.config(out/'prepare.toml',dict(frozen=S['frozen'],scenario=S['scenario'],raw=r['raw'],
        raw_hash=r['raw_hash'],geometry=r['geometry'],geometry_hash=r['geometry_hash'],
        out30=str(out/'planned30'),out60=str(out/'planned60')))
    C.run(['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',str(HERE/'prepare.jl'),str(out/'prepare.toml')],out/'prepare.log')
    return dict(slot=r['slot'],raw=str(out/'planned60/raw.bin'),raw_hash=sha(out/'planned60/raw.bin'),
                geometry=str(out/'planned60/geometry.bin'),geometry_hash=sha(out/'planned60/geometry.bin'))

def record(r,rp,d):
    out=Path(d['output']);f=d['features'];a=r['assignment']
    result=dict(key=r['case_key']+f'_core{r["slot"]:03d}',case=r['case_key'],slot=r['slot'],
        region=r['slot']+8,window=a['window'],slice=int(a['slice_index']),
        seed=int(a['exact_replay_seed']),sample=int(a['selected_sample_index']),
        record_sha256=digest(r),replay_receipt=str(rp),replay_receipt_sha256=sha(rp),features=f,
        raw=str(out/'prepared/raw_unplanned.bin'),geometry=str(out/'prepared/geometry_unplanned.bin'),
        raw_hash=d['files']['prepared/raw_unplanned.bin'],geometry_hash=d['files']['prepared/geometry_unplanned.bin'])
    validate_replay(result)
    return result

def verify_accept(key,full=False):
    p=accepted_path(key);d=read(p)
    assert d['complete'] and d['key']==key and d['memory']['passed']
    assert d['source_lock']==sha(HERE/'source_lock.json')
    if full:
        for p,h in d['artifacts'].items():assert sha(p)==h,p
    return d

def reuse_control(r,ref,out):
    C=context();ptr=ref['pointer'];assert sha(ptr['path'])==ptr['sha256']
    old=Path(ptr['path']).parent;d=C.verify_artifacts(old)
    assert d['record']==ref['record'] and d['numerical']==ref['numerical']
    assert r['raw_hash']==ref['record']['inputs'][ref['record']['raw']]
    assert r['geometry_hash']==ref['record']['inputs'][ref['record']['geometry']]
    new=prepare(r,out)
    for n in ('raw.bin','geometry.bin'):assert sha(out/'planned60'/n)==sha(old/'planned60'/n),n
    C.completion(old/'managed/pass01')
    assert ref['memory']['passed']
    # Exact input and frozen-kernel identity permit result reuse, never geometric similarity alone.
    artifacts={str(old/'context.json'):sha(old/'context.json'),ptr['path']:ptr['sha256'],
               S['closure_path']:S['closure_sha256'],r['replay_receipt']:r['replay_receipt_sha256']}
    artifacts.update({str(old/n):h for n,h in d['artifacts'].items()})
    result=dict(complete=True,key=r['key'],case=r['case'],slot=r['slot'],source_lock=sha(HERE/'source_lock.json'),
        memory=ref['memory'],native_passed=True,kind='EXACT_COMPLETED_RESULT_REUSE',
        output=str(old/'managed/pass01'),sgof=str(old/'sgof.csv'),artifacts=artifacts,
        replay_receipt=r['replay_receipt'],replay_receipt_sha256=r['replay_receipt_sha256'],
        planned_inputs=new,report=ref['numerical'],scientific_recomputed=False)
    p=accepted_path(r['key'])
    if p.exists():assert read(p)==result
    else:write(p,result)
    event('KNOWN_RESULT_REUSED',key=r['key'])

def preflight(label):
    if S.get('reuse_parent_root'):
        import reuse_campaign
        return reuse_campaign.preflight(sys.modules[__name__],label)
    job=bound(label);pins();C=context();L=library()
    original=L.get_plan();closure=read(S['closure_path'])
    assert closure['status']=='PASS_BOUNDED_IMPORT_AND_RECOVERY_REUSE' and closure['native']['passed']
    assert len(closure['records'])==6
    run=ROOT/'preflights'/os.environ['SLURM_JOB_ID'];run.mkdir(parents=True,exist_ok=False)
    probe=L.storage_probe(run/'probe')
    refs={x['record']['case']+f'_core{x["record"]["slot"]:03d}':x for x in closure['records']}
    limit=max(max(x['record']['features']['log10_permeability_span']) for x in closure['records'])
    records={};held=[];groups=defaultdict(list);count=0
    for b in original['batches']:
        for row in b['records']:
            rp=L.core_receipt(row['case_key'],row['slot'])
            if not rp.exists():held.append(dict(case=row['case_key'],slot=row['slot'],reason='replay_not_accepted'));continue
            d=read(rp);r=record(row,rp,d);count+=1
            policy,why=route(r['features'],limit)
            if (r['case'],r['slot']) in {(v[0],v[1]) for v in S['special_holds']}:policy=None;why='special_flow_correction_review'
            if policy is None:held.append(dict(case=r['case'],slot=r['slot'],reason=why));continue
            r['resource']=policy;records[r['key']]=r;groups[policy['stratum']].append(r)
            if count%100==0:event('INPUTS_INDEXED',count=count,admitted=len(records))
    for r in original['held']:held.append(dict(case=r['case_key'],slot=r['slot'],reason='replay_resource_oversized'))
    assert set(refs)<=set(records),'Known controls absent from campaign replay'
    for key,ref in refs.items():reuse_control(records[key],ref,run/key)
    strata={k:dict(memory_gib=v[0]['resource']['memory_gib'],canaries=canaries(v),count=len(v)) for k,v in groups.items()}
    p=dict(records=records,strata=strata,held=held,case_count=original['case_count'],
        original_plan_sha256=sha(CAMPAIGN/'wave01/PLAN.json'),source_lock=sha(HERE/'source_lock.json'),
        contrast_limit=limit,reused=list(refs),scenario='WRR60_PHYSICAL_PREBREAKTHROUGH_FIXED100',
        full_case_export_approved=False,all2430_expansion_approved_by_this_receipt=False)
    write(ROOT/'PLAN.json',p)
    mem=C.memory_snapshot();assert mem['passed'] and mem['limit_gib']==job['memory_gib']
    pins();write(ROOT/'PLAN_ACCEPTED.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],
        source_lock=sha(HERE/'source_lock.json'),sha256=sha(ROOT/'PLAN.json'),memory=mem,storage=probe,
        controls_reused=6,admitted=len(records),held=len(held),strata=strata))
    event('PREFLIGHT_PASS',admitted=len(records),held=len(held),strata=len(strata))

def numerical_review(out,r):
    import numpy as np
    C=context();p=out/'managed/pass01';summary=C.completion(p)
    x=np.loadtxt(p/'pc_kr_raw.csv',delimiter=',',skiprows=1)
    assert x.shape==(100,8) and np.isfinite(x).all()
    assert summary['core']==r['slot'] and summary['axis']==3 and summary['points']==100
    assert np.all(np.diff(x[:,0])>=0) and np.all(np.diff(x[:,1])>=-1e-12)
    assert np.all(x[:,2:4]>=-1e-12) and np.all(x[:,2:4]<=1+1e-8)
    assert summary['raw_input_sha256']==sha(out/'planned60/raw.bin')
    assert summary['geometry_sha256']==sha(out/'planned60/geometry.bin')
    flows=[json.loads(s) for s in (p/'flow_calls.jsonl').read_text().splitlines()]
    assert flows and all(math.isfinite(f['q']) and f['q']>=0 for f in flows)
    life=C.toml(out/'managed/pass1_timing.toml')['lifetime']
    assert life['numeric_created']==life['numeric_released'] and life['symbolic_created']==life['symbolic_released']
    return dict(passed=True,summary=summary,phase_calls=len(flows),lifetime=life,
                all_native_flow_corrections_unchanged=True,reference_result_not_used_as_truth=True)

def compute(r,out,j):
        C=context();key=r['key']
        library().storage_probe(ROOT/'storage_probes'/os.environ['SLURM_JOB_ID'])
        item=prepare(r,out)
        C.config(out/'managed.toml',dict(pipeline=S['pipeline'],pipeline_hash=S['pipeline_hash'],
            bridge=S['bridge'],threads=2,passes=1,output=str(out/'managed')))
        with (out/'managed.toml').open('a') as f:
            f.write('\n[[cases]]\n')
            for k,v in item.items():f.write(k+' = '+(json.dumps(v) if isinstance(v,str) else str(v))+'\n')
        C.run(['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',str(HERE/'worker.jl'),str(out/'managed.toml')],out/'managed.log')
        review=numerical_review(out,r);write(out/'NUMERICAL_COMPLETE.json',review)
        exported=C.export(out);write(out/'export_review.json',exported)
        with (out/'native.log').open('x') as f:
            result=subprocess.run(['julia','--startup-file=no','--project='+S['runtime'],str(HERE/'native.jl'),
                str(out/'native'),str(out/'sgof.csv')],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
        native_pass=result.returncode==0 and (out/'native/constitutive_pass.toml').exists()
        if native_pass:assert C.toml(out/'native/constitutive_pass.toml')['passed']
        mem=C.memory_snapshot();write(out/'MEMORY.json',mem)
        assert mem['passed'] and mem['limit_gib']==j['memory_gib'],'Targeted resource review; numerical files preserved'
        validate_replay(r,True);pins()
        artifacts={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
        done=dict(complete=True,key=key,case=r['case'],slot=r['slot'],job=os.environ['SLURM_JOB_ID'],
            source_lock=sha(HERE/'source_lock.json'),memory=mem,native_passed=native_pass,
            kind='NEW_CAMPAIGN_NUMERICAL_RESULT',output=str(out/'managed/pass01'),sgof=str(out/'sgof.csv'),
            artifacts=artifacts,replay_receipt=r['replay_receipt'],replay_receipt_sha256=r['replay_receipt_sha256'],
            report=review,export=exported,full_case_export_approved=False)
        if 'numerical_key' in r:done['numerical_key']=r['numerical_key']
        return done

def worker(label,key):
    j=bound(label);pins();p=plan();r=p['records'][key];C=context()
    assert j['memory_gib']==r['resource']['memory_gib'] and j['cpus']==2
    if accepted_path(key).exists():verify_accept(key,True);event('ALREADY_ACCEPTED',key=key);return
    out=ROOT/'attempts'/(key+'_job'+os.environ['SLURM_JOB_ID'])
    try:
        if S.get('reuse_parent_root'):
            import reuse_campaign
            reuse_campaign.require_qualified(sys.modules[__name__])
            reuse_campaign.verify_group(sys.modules[__name__],p,key)
        done=compute(r,out,j)
        write(accepted_path(key),done)
        event('NUMERICAL_ACCEPTED',key=key,native_passed=done['native_passed'],memory=done['memory'])
    except BaseException:
        error=dict(key=key,job=os.environ['SLURM_JOB_ID'],error=traceback.format_exc(),
                   output=str(out),memory=C.memory_snapshot(),retry_automatically=False)
        write(failed_path(key),error);event('TARGETED_HOLD',key=key,error=error['error']);raise

def terminal_missing(claims,accepted,failed):
    ids={d['job']:k for k,d in claims.items() if k not in accepted and k not in failed and d.get('job')}
    if not ids:return
    raw=subprocess.check_output(['sacct','-XnP','-j',','.join(ids),'-o','JobIDRaw,State'],text=True,timeout=60)
    for line in raw.splitlines():
        j,state,*_=line.split('|')
        if j in ids and state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
            key=ids[j]
            if not failed_path(key).exists():write(failed_path(key),dict(key=key,job=j,scheduler_state=state,retry_automatically=False))

def controller(label):
    import fcntl
    bound(label);pins();p=plan();L=library()
    if S.get('reuse_parent_root'):
        import reuse_campaign
        reuse_campaign.require_qualified(sys.modules[__name__])
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while seconds_allowed(time.time(),cutoff())>=1800:
            accepted={f.stem:read(f) for f in (ROOT/'accepted').glob('*.json')}
            failed={f.stem:read(f) for f in (ROOT/'held').glob('*.json')}
            if S.get('reuse_parent_root'):
                reuse_campaign.publish_mappings(sys.modules[__name__],p,accepted)
            claims={f.stem.removeprefix('core_'):read(f) for f in claim_path('x').parent.glob('core_*.json')}
            terminal_missing(claims,accepted,failed)
            gates={k:gate_status(s['canaries'],accepted,failed) for k,s in p['strata'].items()}
            todo=[]
            for k,s in p['strata'].items():
                if gates[k]=='WAIT':todo.extend(key for key in s['canaries'] if key not in accepted and key not in failed)
                elif gates[k]=='PASS':todo.extend(key for key,r in p['records'].items() if r['resource']['stratum']==k and key not in accepted and key not in failed)
            todo=list(dict.fromkeys(todo));new=0
            for key in todo:
                if key in claims:
                    d=claims[key]
                    if not d.get('job'):submit('worker','core_'+key,2,p['records'][key]['resource']['memory_gib'],10800,key)
                    continue
                cpu,mem,_=L.usage();need=p['records'][key]['resource']['memory_gib']
                if not room(cpu,mem,2,need):continue
                submit('worker','core_'+key,2,need,10800,key);new+=1
            status=dict(observed=datetime.now(ZoneInfo('UTC')).isoformat(),accepted=len(accepted),held=len(failed),
                gates=gates,submitted_this_pass=new,numerical_scope=len(p['records']),
                full_case_export_approved=False,maintenance_stop=cutoff()-1800)
            if 'placements' in p:
                status.update(total_placements=len(p['placements']),unique_numerical_tasks=len(p['records']),
                    mapped_placements=sum(len(p['records'][k]['placements']) for k,d in accepted.items()
                                          if k in p['records'] and d.get('native_passed')))
            write(ROOT/'STATUS.json',status,True);event('CONTROLLER_PASS',**status)
            # A failed stratum is a durable scoped hold, not a reason to repeat its canary.
            if not todo:break
            time.sleep(60)
        event('CONTROLLER_STOP',reason='all_supported_resolved_or_maintenance',claims_preserved=True)

def initial():
    pins(False)
    ROOT.mkdir(parents=True,exist_ok=True)
    if S.get('reuse_parent_root'):
        import reuse_campaign
        return reuse_campaign.start_scan(sys.modules[__name__])
    pre=submit('preflight','preflight',2,4,3600)
    control=submit('controller','controller',1,2,8*3600,deps=[pre])
    event('CHAIN_SUBMITTED',preflight=pre,controller=control)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='initial':initial()
    elif mode=='preflight':preflight(sys.argv[2])
    elif mode=='worker':worker(sys.argv[2],sys.argv[3])
    elif mode=='controller':controller(sys.argv[2])
    elif mode=='reuse-pilot':
        import reuse_campaign
        reuse_campaign.pilot(sys.modules[__name__],sys.argv[2])
    elif mode=='reuse-index':
        import reuse_campaign
        reuse_campaign.index_case(sys.modules[__name__],sys.argv[2],sys.argv[3])
    elif mode=='reuse-launch':
        import reuse_campaign
        reuse_campaign.launch(sys.modules[__name__])
    else:raise ValueError(mode)
