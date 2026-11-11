"""Independent adaptive trace, fixed-Pc regression, export and receipt checks."""
import csv
import json
import math
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib
from baseline_verify import sha, inputs, load_module, write_new

ROOT = Path(__file__).resolve().parent

def check(ok, message):
    if not ok:
        raise ValueError(message)

def select_trace(s, query, budget=10):
    check(len(s)>=2 and all(math.isfinite(x) for x in s), 'Invalid Sg grid')
    check(all(b>=a for a,b in zip(s,s[1:])) and budget>=2, 'Grid/budget')
    ids = [0] + [i for i in range(1,len(s)) if s[i]>s[i-1]]
    if len(ids)==1:
        ids = [0,len(s)-1]
    else:
        ids[-1] = len(s)-1
    values, order = {}, []
    def observe(rank):
        v = query(ids[rank]+1)
        check(len(v)==2 and all(math.isfinite(x) for x in v), 'Invalid queried Kr')
        values[rank] = v
        order.append(ids[rank]+1)
    observe(0); observe(len(ids)-1)
    while len(values)<min(budget,len(ids)):
        known=sorted(values)
        intervals=[]
        for a,b in zip(known,known[1:]):
            if b-a>1:
                delta=max(abs(values[b][j]-values[a][j]) for j in (0,1))
                intervals.append((delta,s[ids[b]]-s[ids[a]],-a,a,b))
        *_,a,b=max(intervals)
        observe((a+b)//2)
    return sorted(order),order

def rows(path):
    with path.open() as f:
        return [{k:v if k=='state_hash' else float(v) for k,v in r.items()}
                for r in csv.DictReader(f)]

def release():
    r=json.loads((ROOT/'RELEASE.json').read_text())
    check(r['pilot_only'] and not r['full_campaign_authorized'], 'Scope')
    for p,h in r['files'].items():
        check(sha(ROOT/p)==h, 'Source changed: '+p)
    return r

def exports():
    import numpy as np
    release(); conversion=inputs()
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    planning=load_module('planning');tables=load_module('build_tables');records=[]
    for c in json.loads((ROOT/'cases.json').read_text()):
        key=c['key'];root=ROOT/'results'/key
        receipt=tomllib.loads((root/'CORE_PASS.toml').read_text())
        proof=receipt['input'];summary=tomllib.loads((root/'summary.toml').read_text())
        check(receipt['passed'] and proof['passed'], key)
        check(receipt['method']=='adaptive10_endpoint_delta_v1', 'Wrong method')
        for field in ('selector_sha256','adaptive_curve_sha256'):
            check(receipt[field]==provenance[field], field)
        check(receipt['source_release_sha256']==sha(ROOT/'RELEASE.json'), 'Release identity')
        check(proof['sample']==c['sample'] and proof['contact_angle_deg']==40, 'Wrong clay profile')
        for p,h in receipt['artifacts'].items():
            check(sha(root/p)==h, key+'/'+p)
        pc,kr=rows(root/'pc100.csv'),rows(root/'kr.csv')
        check(len(pc)==100 and 2<=len(kr)<=10, 'State budget')
        observed={int(r['sample']):(r['krw'],r['krg']) for r in kr}
        check(len(observed)==len(kr), 'Repeated state')
        ids,order=select_trace([r['sg_physical'] for r in pc],lambda i:observed[i])
        check(ids==[int(r['sample']) for r in kr]==summary['selected_indices'], 'Selected states')
        check(order==summary['query_order'], 'Adaptive decision trace')
        check(summary['mode']=='adaptive10' and summary['same_mip_microstates'], 'Method evidence')
        check(summary['phase_solver_calls']<=20 and summary['intrinsic_solver_calls']==1, 'Flow budget')
        life=summary['native_lifetime']
        check(life['numeric_created']==life['numeric_released'] and
              life['symbolic_created']==life['symbolic_released'], 'Native lifetime')
        ref=ROOT/'reference/results'/key
        old=tomllib.loads((ref/'CORE_PASS.toml').read_text())
        check(old['passed'], 'Reference receipt')
        for p,h in old['artifacts'].items():
            check(sha(ref/p)==h, 'Reference changed')
        check(proof['raw_sha256']==old['input']['raw_sha256'] and
              proof['geometry_sha256']==old['input']['geometry_sha256'], 'Prepared input changed')
        check(pc==rows(ref/'pc100.csv'), 'Adaptive changed Pc/MIP states')
        refkr={int(r['sample']):r for r in rows(ref/'kr.csv')}
        common=sorted(set(observed)&set(refkr))
        maxerr=max(abs(observed[i][j]-refkr[i][k]) for i in common for j,k in enumerate(('krw','krg')))
        check(maxerr<=1e-8, 'Same-state Kr changed')
        oldsummary=tomllib.loads((ref/'summary.toml').read_text())
        check(math.isclose(summary['intrinsic_permeability_m2'],oldsummary['intrinsic_permeability_m2'],rel_tol=1e-12), 'Intrinsic changed')
        dense=np.array([[r['pressure_pa'],r['sg_physical'],0,0,r['trapped_pv_fraction'],r['immobile_pv_fraction'],r['gas_breakthrough'],r['invaded_pv_fraction']] for r in pc])
        dense[:,:4]=planning.paired_grid(pc,kr)
        t,detail=tables.convert(dense,proof['entry_pressure_pa'])
        out=root/'export';out.mkdir(exist_ok=False)
        np.savetxt(out/'sgof.csv',t,delimiter=',',header='Sg,Krg,Krw,PcPa',comments='',fmt='%.17g')
        np.savetxt(out/'paired.csv',dense,delimiter=',',header='pressure_pa,sg_physical,krw,krg,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction',comments='',fmt='%.17g')
        write_new(out/'EXPORT_PASS.json',dict(passed=True,detail=detail,sha256=sha(out/'sgof.csv')))
        records.append(dict(key=key,cells=summary['cells'],kr_states=len(kr),phase_solver_calls=summary['phase_solver_calls'],
            same_pc100_and_microstates=True,shared_kr_states=common,max_shared_kr_error=maxerr,
            wall_seconds=summary['wall_seconds'],peak_rss_kib=summary['memory']['VmHWM_kib'],
            intrinsic_permeability_m2=summary['intrinsic_permeability_m2']))
    for core in ('low_143','intermediate_167','high_144'):
        pair=[r for r in records if r['key'].startswith(core+'_')]
        check(pair[0]['intrinsic_permeability_m2']==pair[1]['intrinsic_permeability_m2'], 'Sample-dependent intrinsic')
    write_new(ROOT/'EXPORT_PASS.json',dict(passed=True,records=records,conversion_checks=conversion,full_campaign_launched=False))
    print('EXPORT_AND_REGRESSION_PASS',len(records))

def finish():
    release()
    numerical=tomllib.loads((ROOT/'NUMERICAL_PASS.toml').read_text())
    native=tomllib.loads((ROOT/'NATIVE_PASS.toml').read_text())
    exported=json.loads((ROOT/'EXPORT_PASS.json').read_text())
    expected={c['key'] for c in json.loads((ROOT/'cases.json').read_text())}
    check(numerical['passed'] and native['passed'] and exported['passed'], 'Acceptance')
    check(len(expected)==6 and {r['key'] for r in numerical['records']}==expected, 'Six outputs')
    check({r['key'] for r in native['records'] if r['use']=='core'}==expected, 'Native coverage')
    check({r['sample'] for r in native['records'] if r['use']=='host'}=={4,6}, 'Host coverage')
    for key in expected:
        root=ROOT/'results'/key
        h=sha(root/'export/sgof.csv')
        check(h==json.loads((root/'export/EXPORT_PASS.json').read_text())['sha256'], 'Export identity')
        check(h==next(r['sha256'] for r in native['records'] if r.get('key')==key), 'Imported identity')
    write_new(ROOT/'INTEGRATION_PASS.json',dict(passed=True,runs=6,method='adaptive10_endpoint_delta_v1',
        source_release_sha256=sha(ROOT/'RELEASE.json'),samples=[4,6],contact_angle_deg=40,
        native_core_imports=6,native_host_imports=2,full_campaign_launched=False,
        old_campaign_resumed=False,new_campaign_case_credit=False,records=exported['records']))
    print('PC40_ADAPTIVE10_INTEGRATION_PASS')

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='release':
        print('RELEASE_PASS',len(release()['files']))
    elif mode=='export':
        exports()
    elif mode=='finish':
        finish()
    else:
        raise SystemExit(mode)
