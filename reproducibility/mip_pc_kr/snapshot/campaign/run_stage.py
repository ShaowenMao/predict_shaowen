"""Small isolated stages: never spend flow-solve memory on the native importer."""
import time
import traceback
import tomllib
from launch_support import *

def worker_config(batch,rows,bootstrap=False):
    m=read(Q3/'manifest.json');lock=guard()
    cfg=batch/('bootstrap.toml' if bootstrap else 'worker.toml')
    top={k:m[k] for k in ['pipeline','backend','bridge','pipeline_hash']}
    top.update(kernel_root=str(Q3),output=str(batch/'curves'),bootstrap=bootstrap,
               memory_mib=read(batch/'spec.json')['flow_mib'] if not bootstrap else 2816,
               launch_lock_sha256=sha(HERE/'LAUNCH_LOCK.json'))
    toml_config(cfg,top,rows,lock['worker_pins']);return cfg

def bootstrap():
    folder=ROOT/'audit/bootstrap';folder.mkdir(exist_ok=False)
    s=storage_probe(os.environ['SLURM_JOB_ID'])
    cfg=worker_config(folder,[],True)
    run(julia_args(HERE/'worker.jl',cfg),folder/'julia.log',julia_env(folder))
    assert 'CAMPAIGN_CACHE_LOAD_PASS' in (folder/'julia.log').read_text()
    write(folder/'PASS.json',dict(passed=True,storage=s,strict_existing_compilation_cache=True))
    return [folder/'PASS.json',folder/'julia.log']

def cache():
    folder=ROOT/'audit/prepare_cache';folder.mkdir(exist_ok=False)
    s=read(NUM/'spec.json');cfg=folder/'config.toml'
    toml_config(cfg,{k:s[k] for k in ['frozen','scenario']})
    depot=ROOT/'cache/depot';depot.mkdir(parents=True,exist_ok=False)
    env=julia_env(folder)
    # Only this job builds the shared preparation cache; science workers remain strict readers.
    env['JULIA_DEPOT_PATH']=':'.join(map(str,[depot,Q3/'cache/depot',read(Q3/'manifest.json')['depot']]))
    args=julia_args(HERE/'prepare_import.jl',cfg)
    prime=[a.replace('--compiled-modules=strict','--compiled-modules=yes').replace('--pkgimages=existing','--pkgimages=yes') for a in args]
    run(prime,folder/'prime.log',env)
    run(args,folder/'strict.log',env)
    assert 'FROZEN_PREPARATION_IMPORT_PASS' in (folder/'strict.log').read_text()
    write(folder/'PASS.json',dict(passed=True,preparation_cache_strict_read_passed=True,
        science_recomputed=False,shared_depot=str(depot),files={str(p):sha(p) for p in depot.rglob('*') if p.is_file()}))
    return [folder/'PASS.json',folder/'prime.log',folder/'strict.log']

def prepare(batch,spec):
    settings=read(NUM/'spec.json');rows=[];identities=[]
    for item in spec['items']:
        identities.append(verify_owner(item));r=item['row'];p=batch/'planned'/item['key']
        rows.append(dict(key=item['key'],raw=r['raw'],raw_hash=r['raw_hash'],geometry=r['geometry'],
            geometry_hash=r['geometry_hash'],out30=str(p/'planned30'),out60=str(p/'planned60')))
    cfg=batch/'prepare.toml';toml_config(cfg,{k:settings[k] for k in ['frozen','scenario']},rows)
    run(julia_args(HERE/'prepare_batch.jl',cfg),batch/'prepare.log',julia_env(batch))
    write(batch/'IDENTITY_PASS.json',dict(passed=True,records=identities))
    files=[batch/'IDENTITY_PASS.json']
    for r in rows:
        p=Path(r['out60']);audit=tomllib.loads((p/'scenario_audit.toml').read_text())
        assert audit['passed'] and audit['clay_swi']==.2 and audit['clay_ng']==2.5 and audit['clay_nw']==3.5
        assert audit['unchanged_geometry_intrinsic_properties_sand_and_sampling']
        files.extend(p/n for n in ['raw.bin','geometry.bin','scenario_audit.toml'])
    return files

def flow(batch,spec):
    stage_receipt(batch.name,'prepare');rows=[]
    for item in spec['items']:
        p=batch/'planned'/item['key']/'planned60'
        rows.append(dict(key=item['key'],slot=item['row']['slot'],raw=str(p/'raw.bin'),
                         geometry=str(p/'geometry.bin'),scenario_audit=str(p/'scenario_audit.toml')))
    cfg=worker_config(batch,rows)
    run(julia_args(HERE/'worker.jl',cfg),batch/'flow.log',julia_env(batch))
    result=tomllib.loads((batch/'curves/BATCH_RESULT.toml').read_text())
    files=[batch/'curves/BATCH_RESULT.toml']
    for r in result['records']:
        if r['passed']:files.append(batch/'curves'/r['key']/'CORE_PASS.toml')
    assert len(files)>1,'No numerical cores passed'
    return files

def export(batch,spec):
    items=[];held=[]
    if spec['kind']=='derive':
        for item in spec['items']:
            try:verify_owner(item);items.append(item)
            except Exception:held.append(dict(key=item['key'],error=traceback.format_exc()))
    else:
        stage_receipt(batch.name,'flow')
        for item in spec['items']:
            p=batch/'curves'/item['key']/'CORE_PASS.toml'
            if p.exists():items.append(dict(item,curve=str(p.parent)))
            else:held.append(dict(key=item['key'],error='Numerical output incomplete; do not rerun completed prefix'))
    config=batch/'export.json';write(config,dict(items=items,out=str(batch/'export')))
    write(batch/'EXPORT_INPUT_HOLDS.json',dict(records=held))
    run([PY,'-B',HERE/'export_sparse.py',config],batch/'export.log')
    r=read(batch/'export/EXPORT_BATCH.json');assert r['passed']
    return [batch/'export/EXPORT_BATCH.json',batch/'EXPORT_INPUT_HOLDS.json',
            *[batch/'export'/x['key']/'EXPORT_PASS.json' for x in r['records']]]

def native(batch,spec):
    stage_receipt(batch.name,'export');r=read(batch/'export/EXPORT_BATCH.json')
    cfg=read(QUAL/'config.json');tables=[batch/'export'/x['key']/'sgof.csv' for x in r['records']]
    for x in r['records']:pins_ok(x['artifacts'])
    depot=batch/'native_depot';depot.mkdir(exist_ok=True)
    env=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_DEPOT_PATH=str(depot)+':/home/shaowen/.julia',
             JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    julia='/orcd/software/community/001/pkg/julia/1.10.4/bin/julia'
    args=[julia,'--startup-file=no','--threads=1','--gcthreads=1',
          '--compiled-modules=existing','--pkgimages=existing','--project='+cfg['runtime'],
          QUAL/'native.jl',batch/'native',*tables]
    run(args,batch/'native.log',env)
    p=batch/'native/constitutive_pass.toml';native=tomllib.loads(p.read_text())
    assert native['passed'] and len(native['records'])==len(tables)
    assert native['julia']=='1.10.4'
    for item,table in zip(native['records'],tables):assert item['sha256']==sha(table)
    return [p]

def main():
    guard();name,stage=sys.argv[1:3];start=time.monotonic()
    spec=read(ROOT/'batches'/name/'spec.json') if name not in ('bootstrap','cache') else {}
    policy=(dict(cpus=2,memory_mib=2816) if stage in ('bootstrap','cache') else spec['stages'][stage])
    actual=allocation(policy['cpus'],policy['memory_mib'])
    batch=ROOT/'batches'/name
    p=receipt_path(name,stage);assert not p.exists()
    files=globals()[stage]() if stage in ('bootstrap','cache') else globals()[stage](batch,spec)
    guard();mem=memory(policy['memory_mib'])
    write(p,dict(passed=True,batch=name,stage=stage,job=os.environ['SLURM_JOB_ID'],
        lock_sha256=sha(HERE/'LAUNCH_LOCK.json'),memory=mem,seconds=time.monotonic()-start,
        actual_scontrol=actual,artifacts={str(p):sha(p) for p in files}))
    event('STAGE_PASS',batch=name,stage=stage,memory=mem)

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'holds'/('job'+os.environ.get('SLURM_JOB_ID','local')+'.json'),
              dict(args=sys.argv[1:],error=traceback.format_exc(),completed_outputs_preserved=True))
        raise
