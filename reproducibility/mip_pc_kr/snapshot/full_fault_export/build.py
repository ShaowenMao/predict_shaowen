"""Copy qualified unchanged helpers and freeze this separate release."""
from pathlib import Path
import json
import shutil
from common import HERE, sha, write

PARENT=HERE.parent
OLD=Path('D:/codex_gom/studies/wrr60_phase1_production_candidate_20260912_v1')
RELEASE=OLD/'reference_integration_v1/downstream_recovery_v1/releases/3a70d76fd76f0f43'
BASE='/orcd/data/juanes/001/shaowen/predict_shaowen/candidates'
REMOTE=BASE+'/'+PARENT.name+'/'+HERE.name


def main():
    sources={}
    for n in ('planning.py',):sources[n]=PARENT/n
    sources['resource.py']=PARENT/'qualification_v3/resource.py'
    sources['reuse.py']=OLD/'campaign_numerical_v1/reuse.py'
    for n in ('native.jl','endpoint_check.jl','endpoint_tests.jl','origin.toml'):
        sources[n]=PARENT/'native_import_recovery_v2'/n
    for n in ('assemble60.m','build_tables.py','make_specific.m','make_specific60.m','verify_matlab_startup.m',
              'runtime_reference_preflight.jl','contract.jl','hysteresis_validation.jl','snapshot.json',
              'reference/advanced.jl','reference/production_qoi.jl'):
        sources[n]=RELEASE/n
    for n,source in sources.items():
        target=HERE/n;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert sha(target)==sha(source)
        else:shutil.copyfile(source,target)
    spec=json.loads((RELEASE/'integration_spec.json').read_text())
    cfg=dict(remote=REMOTE,old_root=spec['root'],old_release='3a70d76fd76f0f43',
             old_final_sha256='d423c607db208dfd2946f2d1104c049ee90061fd6f34506e246e91bcbc694823',
             pilot=BASE+'/'+PARENT.name+'/qualification_v3',qualification_only=True,campaign_launch=False,
             matlab='/orcd/software/core/001/pkg/matlab/R2025b/bin/matlab',
             julia='/orcd/software/core/001/pkg/julia/1.12.6/bin/julia',
             runtime=json.loads((RELEASE/'snapshot.json').read_text())['runtime'],
             export_python=spec['export_python'],common=spec['common'],baseline_specific=spec['baseline_specific'])
    write(HERE/'config.json',cfg)
    files={str(p.relative_to(HERE)).replace('\\','/'):sha(p) for p in HERE.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.name not in ('LOCK.json','tests.json')}
    write(HERE/'LOCK.json',dict(schema='one_fault_pc100_kr10_qualification_v1',files=files,
                               copied_sources={n:dict(path=str(p),sha256=sha(p)) for n,p in sources.items()},
                               production_modified=False,campaign_launch=False))
    print(REMOTE,sha(HERE/'LOCK.json'))


if __name__=='__main__':main()
