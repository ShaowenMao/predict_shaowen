"""Freeze orchestration separately from the already completed reconciliation."""
from common_launch import *

verify()
names=['launch_support.py','index_work.py','dispatcher.py','run_stage.py','campaign_job.sh',
       'worker.jl','prepare_batch.jl','prepare_import.jl','recover_launch.py','export_sparse.py','test_launch.py','freeze_launch.py']
m=read(Q3/'manifest.json');s=read(NUM/'spec.json');external={}
for n in ['planning.py','build_tables.py','reuse.py','bounded_io.py','resource.py',
          'native.jl','endpoint_check.jl','endpoint_tests.jl','origin.toml','config.json','COMPLETE_REVIEW.json']:
    external[str(QUAL/n)]=sha(QUAL/n)
for n in ['Kernel.jl','SelectStates.jl']:
    assert sha(Q3/n)==m['sources'][n];external[str(Q3/n)]=m['sources'][n]
roots=[Path(m[k]) for k in ['pipeline','backend','project']]
worker={p:h for p,h in m['pins'].items() if any(Path(p).is_relative_to(root) for root in roots) or p==m['bridge']}
worker.update({str(Q3/n):sha(Q3/n) for n in ['Kernel.jl','SelectStates.jl']})
external.update(worker);external.update(s['external_files'])
frozen=Path(s['frozen']);lock=read(frozen/'source_lock.json')
external.update({str(frozen/n):h for n,h in lock['files'].items()})
external[str(NUM/'spec.json')]=sha(NUM/'spec.json');external[str(Q3/'manifest.json')]=sha(Q3/'manifest.json')
for c in ['mixed_60000','mixed_90000','mixed_120000','mixed_150000','clay_120000']:
    p=WARM/'progressive_launch_v1/class_gates'/(c+'.json');g=read(p)
    assert g['passed'];external[str(p)]=sha(p);external.update(g['evidence'])
tomli=PY.parent.parent/'lib/python3.10/site-packages/pip/_vendor/tomli'
assert tomli.is_dir()
external.update({str(p):sha(p) for p in tomli.glob('*.py')})
parents=[]
for name in ['campaign_launch_v1','campaign_launch_v2']:
    p=HERE.parent/name/'LAUNCH_LOCK.json';old=read(p)
    pins_ok({p.parent/n:h for n,h in old['files'].items()})
    external[str(p)]=sha(p);parents.append(sha(p))
pins_ok(external)
contract=dict(clay_angle=60,clay_swi=.2,clay_endpoint=.8,clay_ng=2.5,clay_nw=3.5,
    sand_endpoint=.95,pc_states=100,kr_states=10,cell_specific_pc=True,wetting_rule='krw>0',
    physical_prebreakthrough=True,axis=3,solver='corrected_direct',threads=2,
    selector_sha256=sha(Q3/'SelectStates.jl'),kernel_sha256=sha(Q3/'Kernel.jl'),
    pipeline_sha256=m['pipeline_hash'],bridge_sha256=sha(m['bridge']))
assert all(read(HERE.parent/n/'LAUNCH_LOCK.json')['method_sha256']==digest(contract) for n in ['campaign_launch_v1','campaign_launch_v2'])
result=dict(base_lock_sha256=sha(HERE/'LOCK.json'),files={n:sha(HERE/n) for n in names},inherited_launch_locks=parents,
            external=external,worker_pins=worker,contract=contract,method_sha256=digest(contract),
            full_design=2430,production_modified=False,reservoir_submission=False)
p=HERE/'LAUNCH_LOCK.json'
with p.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print('LAUNCH_LOCK',sha(p),flush=True)
