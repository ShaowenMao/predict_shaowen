"""Independent conversion, integrity and exported-table checks for this pilot."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p, value):
    with p.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)
def load_module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'vendor/kernel/{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def inputs():
    import numpy as np
    lock=json.loads((ROOT/'INPUT_LOCK.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in lock['staged_files'].items())
    if sys.platform=='win32':
        assert all(sha(p)==h for p,h in lock['protected_sources'].items())
    atlas=list(csv.DictReader((ROOT/'data/atlas_hg.csv').open()))
    factor=abs(25*math.cos(math.radians(40))/(485*math.cos(math.radians(140))))
    details=[]
    for sid in (4,6):
        a=[r for r in atlas if int(r['sample_id'])==sid]
        sg=np.array([float(r['gas_saturation']) for r in a])
        pc=np.array([float(r['hg_air_pressure_psi']) for r in a])*6894.757*factor
        assert pc[0]<pc[1] and np.all(np.diff(pc)>0)
        for use,end in [('host',.57),('master',.8)]:
            t=np.loadtxt(ROOT/f'data/{use}_sample{sid}.csv',delimiter=',',skiprows=1)
            assert t[0,0]==0 and t[-1,0]==end and np.all(np.diff(t,axis=0)>0)
            assert np.allclose(t[:,1],np.interp(t[:,0],sg,pc),rtol=3e-15,atol=1e-8)
            details.append(dict(sample=sid,use=use,entry_pa=float(t[0,1]),endpoint_sg=end))
    return dict(passed=True,conversion_factor=factor,details=details,immutable_staged_files=len(lock['staged_files']))

def exports():
    import numpy as np
    planning=load_module('planning');tables=load_module('build_tables');records=[]
    for c in json.loads((ROOT/'cases.json').read_text()):
        root=ROOT/'results'/c['key']
        proof=tomllib.loads((root/'CORE_PASS.toml').read_text())
        assert proof['passed']
        assert all(sha(root/p)==h for p,h in proof['artifacts'].items())
        def rows(name):
            return [{k:v if k=='state_hash' else float(v) for k,v in r.items()} for r in csv.DictReader((root/name).open())]
        pc,kr=rows('pc100.csv'),rows('kr.csv')
        ids,_=planning.select_states([r['sg_physical'] for r in pc])
        assert ids==[int(r['sample'])-1 for r in kr]
        dense=np.array([[r['pressure_pa'],r['sg_physical'],0,0,r['trapped_pv_fraction'],r['immobile_pv_fraction'],r['gas_breakthrough'],r['invaded_pv_fraction']] for r in pc])
        dense[:,:4]=planning.paired_grid(pc,kr)
        t,detail=tables.convert(dense,proof['input']['entry_pressure_pa'])
        out=root/'export';out.mkdir(exist_ok=False)
        np.savetxt(out/'sgof.csv',t,delimiter=',',header='Sg,Krg,Krw,PcPa',comments='',fmt='%.17g')
        np.savetxt(out/'paired.csv',dense,delimiter=',',header='pressure_pa,sg_physical,krw,krg,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction',comments='',fmt='%.17g')
        summary=tomllib.loads((root/'summary.toml').read_text())
        assert summary['same_mip_microstates'] and summary['intrinsic_solver_calls']==1
        write_new(out/'EXPORT_PASS.json',dict(passed=True,detail=detail,sha256=sha(out/'sgof.csv')))
        records.append(dict(key=c['key'],final_sg=pc[-1]['sg_physical'],final_krg=kr[-1]['krg'],
            final_krw=kr[-1]['krw'],trapped_pv_fraction=pc[-1]['trapped_pv_fraction'],
            intrinsic_permeability_m2=summary['intrinsic_permeability_m2'],wall_seconds=summary['wall_seconds'],
            peak_rss_kib=summary['memory']['VmHWM_kib'],export=detail))
    for core in {r['key'].rsplit('_pc',1)[0] for r in records}:
        pair=[r for r in records if r['key'].startswith(core+'_')]
        assert pair[0]['intrinsic_permeability_m2']==pair[1]['intrinsic_permeability_m2']
    return dict(passed=True,records=records,full_campaign_launched=False)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='inputs':
        report=inputs();print(json.dumps(report));write_new(ROOT/'UNIT_PASS.json',report)
    elif mode=='export':
        inputs();report=exports();write_new(ROOT/'EXPORT_PASS.json',report);print(json.dumps(report))
    elif mode=='seal':
        inputs()
        pins={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='RELEASE.json'}
        write_new(ROOT/'RELEASE.json',dict(files=pins,pilot_only=True,full_campaign_authorized=False))
        print('SEALED',len(pins))
    elif mode=='release':
        release=json.loads((ROOT/'RELEASE.json').read_text())
        assert all(sha(ROOT/p)==h for p,h in release['files'].items())
        print('RELEASE_PASS',len(release['files']))
    else: raise ValueError(mode)
