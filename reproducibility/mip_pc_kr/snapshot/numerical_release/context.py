"""Full-curve workers and independent numerical/export review. No time marching."""
from pathlib import Path
import hashlib,json,os,subprocess,time
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import numpy as np
from scipy.io import savemat,loadmat
from build_tables import convert


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def memory_snapshot():
    job=os.environ['SLURM_JOB_ID']
    roots=list(Path('/sys/fs/cgroup/memory/slurm').glob('uid_*/job_'+job));assert len(roots)==1
    r=roots[0];peak=int((r/'memory.max_usage_in_bytes').read_text())/1024**3
    limit=int((r/'memory.limit_in_bytes').read_text())/1024**3
    events=int((r/'memory.failcnt').read_text());required=max(peak+1,1.3*peak)
    return dict(peak_gib=peak,limit_gib=limit,events=events,required_gib=required,passed=events==0 and required<=limit)


def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False)
def toml(p):
    with Path(p).open('rb') as f:return tomllib.load(f)


def verify_artifacts(folder):
    d=read(folder/'context.json')
    for n,h in d['artifacts'].items(): assert sha(folder/n)==h,n
    assert d['status']=='PASS_BROADER_NUMERICAL_CURVE'
    return d


def verify_external(spec):
    for p,h in spec['external_files'].items(): assert sha(p)==h,p


def config(path,d):
    lines=[]
    for k,v in d.items():
        if isinstance(v,str): lines.append(k+' = '+json.dumps(v))
        elif isinstance(v,int):lines.append(k+' = '+str(v))
        else:raise TypeError(k)
    with path.open('x') as f:f.write('\n'.join(lines)+'\n')


def run(args,log):
    start=time.monotonic();print('START',args[0],str(log),flush=True)
    with log.open('x') as f:
        p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=27000)
    write(log.with_suffix('.receipt.json'),dict(args=args,returncode=p.returncode,seconds=time.monotonic()-start))
    assert p.returncode==0,str(log)
    print('END',str(log),flush=True)


def completion(folder):
    c=toml(folder/'completion.toml');assert c['execution_complete'] and c['points']==100
    for n,h in c['outputs'].items():assert sha(folder/n)==h,n
    return toml(folder/'summary.toml')


def compare_arrays(a,b):
    assert a.shape==b.shape==(100,8) and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.array_equal(a[:,[0,6]],b[:,[0,6]])
    delta=np.max(np.abs(a-b),axis=0)
    assert max(delta[[1,4,5,7]])<=16*np.finfo(float).eps,delta
    assert max(delta[2:4])<1e-8,delta
    assert np.all(np.abs(a[:,2:4]-b[:,2:4])<=1e-12+1e-5*np.abs(a[:,2:4]))
    assert np.array_equal(a[:,2:4]==0,b[:,2:4]==0),'Mobility zero pattern changed'
    return delta.tolist()


def compare(out):
    base=out/'baseline';new=out/'managed/pass01'
    a=completion(base);b=completion(new)
    for k in ('core','axis','points','event_counts','events','owner_samples','fast_path',
              'cache_exact_mobility_states','unique_mobility_states','model_audit',
              'raw_input_sha256','geometry_sha256','bridge_sha256','source_tree_sha256',
              'intrinsic_conductance','intrinsic_permeability_m2'):
        assert a[k]==b[k],k
    x=np.loadtxt(base/'pc_kr_raw.csv',delimiter=',',skiprows=1)
    y=np.loadtxt(new/'pc_kr_raw.csv',delimiter=',',skiprows=1)
    delta=compare_arrays(x,y)
    flows=[]
    for p in (base,new):
        rows=[json.loads(s) for s in (p/'flow_calls.jsonl').read_text().splitlines()]
        assert all(np.isfinite(r['q']) and r['q']>=0 for r in rows)
        flows.append(rows)
    assert [(r['phase'],r['sample']) for r in flows[0]]==[(r['phase'],r['sample']) for r in flows[1]]
    # Each native solve already enforces its unchanged strict correction gates.
    # Independently compare every conductance, not only the normalized curve.
    for old,newrow in zip(*flows):
        assert abs(old['q']-newrow['q'])<=a['intrinsic_conductance']*1e-12+1e-5*abs(old['q'])
        assert (old['q']==0)==(newrow['q']==0)
    life=toml(out/'managed/pass1_timing.toml')['lifetime']
    assert life['numeric_created']==life['numeric_released']
    assert life['symbolic_created']==life['symbolic_released']
    return dict(passed=True,max_abs_difference_by_column=delta,phase_calls=len(flows[0]),
        event_count=b['events'],unique_mobility_states=b['unique_mobility_states'],
        mip_seconds=b['mip_and_observation_seconds'],flow_seconds=b['phase_flow_seconds'],
        baseline_wall_seconds=toml(base/'completion.toml')['wall_seconds'],
        managed_wall_seconds=toml(out/'managed/pass1_timing.toml')['seconds'],
        fast_path=b['fast_path'],lifetime=life,repeatable_speedup_claimed=False)


def export(out):
    x=np.loadtxt(out/'managed/pass01/pc_kr_raw.csv',delimiter=',',skiprows=1)
    entry=toml(out/'planned60/scenario_audit.toml')['entry_pressure_pa']
    table,report=convert(x,entry)
    np.savetxt(out/'sgof.csv',table,delimiter=',',header='Sg,Krg,Krw,PcPa',comments='',fmt='%.17g')
    savemat(out/'sgof.mat',dict(table=table),do_compression=True)
    assert np.array_equal(loadmat(out/'sgof.mat')['table'],table)
    assert np.array_equal(np.loadtxt(out/'sgof.csv',delimiter=',',skiprows=1),table)
    report.update(mat_roundtrip_exact=True,native_import_passed=False,full_fault_mapping_test=False)
    return report


def build_context(here,batch,out):
    spec=read(here/'spec.json');verify_external(spec);r=spec['records'][batch]
    for p,h in r['inputs'].items():assert sha(p)==h,p
    config(out/'prepare.toml',dict(frozen=spec['frozen'],scenario=spec['scenario'],
        raw=r['raw'],raw_hash=r['inputs'][r['raw']],geometry=r['geometry'],
        geometry_hash=r['inputs'][r['geometry']],out30=str(out/'planned30'),out60=str(out/'planned60')))
    julia=['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes']
    run(julia+[str(here/'prepare.jl'),str(out/'prepare.toml')],out/'prepare.log')
    item=dict(slot=r['slot'],raw=str(out/'planned60/raw.bin'),geometry=str(out/'planned60/geometry.bin'))
    item.update(raw_hash=sha(item['raw']),geometry_hash=sha(item['geometry']))
    common=dict(pipeline=spec['pipeline'],pipeline_hash=spec['pipeline_hash'],bridge=spec['bridge'])
    config(out/'baseline.toml',dict(**common,**item,output=str(out/'baseline')))
    config(out/'managed.toml',dict(**common,threads=2,passes=1,output=str(out/'managed')))
    with (out/'managed.toml').open('a') as f:
        f.write('\n[[cases]]\n')
        for k,v in item.items():f.write(k+' = '+(json.dumps(v) if isinstance(v,str) else str(v))+'\n')
    run(julia+[str(here/'baseline.jl'),str(out/'baseline.toml')],out/'baseline.log')
    run(julia+[str(here/'worker.jl'),str(out/'managed.toml')],out/'managed.log')
    result=compare(out);write(out/'numerical_review.json',result)
    exported=export(out);write(out/'export_review.json',exported)
    verify_external(spec)
    return dict(status='PASS_BROADER_NUMERICAL_CURVE',record=r,numerical=result,export=exported,
        artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        native_import_passed=False,campaign_launch_ready=False,production_modified=False)
