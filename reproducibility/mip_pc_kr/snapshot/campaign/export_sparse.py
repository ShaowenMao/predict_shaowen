"""No flow solves: validate saved sparse curves and use the qualified exporter."""
import csv
import sys
import math
try:import tomllib
except ImportError:from pip._vendor import tomli as tomllib
import numpy as np
from common_launch import *
sys.path.insert(0,str(QUAL))
from planning import select_states,paired_grid
from build_tables import convert

NAMES=['pressure_pa','sg_physical','krw','krg','trapped_pv_fraction','immobile_pv_fraction','gas_breakthrough','invaded_pv_fraction']
def rows(p):
    with Path(p).open() as f:return [{k:(v if k=='state_hash' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]

def export(item,out):
    out=Path(out);mark=out/'EXPORT_PASS.json'
    if mark.exists():
        x=read(mark);pins_ok(x['artifacts']);return x
    if item['kind']=='derive':
        old=read(item['receipt']);assert sha(item['receipt'])==item['receipt_sha256']
        assert old['complete'] and old['report']['passed'] and old['numerical_key']==item['old_numerical_key']
        pins_ok(old['artifacts'])
        paths=[Path(p) for p in old['artifacts'] if p.endswith('/pc_kr_raw.csv') and 'baseline' not in p]
        assert paths
        # If a collector included more than one copy, they must agree exactly.
        hashes={sha(p) for p in paths};assert len(hashes)==1
        src=paths[0];pc=rows(src);assert len(pc)==100
        for i,r in enumerate(pc):r['sample']=i+1
        ids,targets=select_states([r['sg_physical'] for r in pc]);kr=[pc[i] for i in ids]
        audits=[Path(p) for p in old['artifacts'] if p.endswith('/scenario_audit.toml')]
        audits=[p for p in audits if ('planned60' in str(p) or 'planned/' in str(p))];assert len(audits)>=1
        audit_values=[tomllib.loads(p.read_text()) for p in audits]
        assert all(a['passed'] and a['clay_swi']==.2 and a['clay_ng']==2.5 and a['clay_nw']==3.5 for a in audit_values)
        entry_values={a['entry_pressure_pa'] for a in audit_values};assert len(entry_values)==1
        entry=entry_values.pop();origin={str(src):sha(src),item['receipt']:sha(item['receipt'])}
        summaries=[Path(p) for p in old['artifacts'] if p.endswith('/summary.toml') and 'baseline' not in p]
        assert summaries
        for p in summaries:
            s=tomllib.loads(p.read_text())
            assert s['points']==100 and s['axis']==3 and s['source_tree_sha256']==read(Q3/'manifest.json')['pipeline_hash']
            assert s['blas_threads']==s['julia_threads']==2
        dense=np.array([[r[k] for k in NAMES] for r in pc]);error_reference=dense.copy()
    else:
        work=Path(item['curve']);passed=tomllib.loads((work/'CORE_PASS.toml').read_text());assert passed['passed']
        pins_ok({work/n:h for n,h in passed['artifacts'].items()})
        pc=rows(work/'pc100.csv');kr=rows(work/'kr.csv');assert len(pc)==100 and 1<=len(kr)<=10
        ids,targets=select_states([r['sg_physical'] for r in pc]);assert [int(r['sample'])-1 for r in kr]==ids
        assert all(r['pressure_pa']==pc[i]['pressure_pa'] and r['sg_physical']==pc[i]['sg_physical'] for r,i in zip(kr,ids))
        summary=tomllib.loads((work/'summary.toml').read_text());assert summary['same_mip_microstates']
        assert summary['phase_solver_calls']<=20 and summary['intrinsic_solver_calls']==1
        flows=[json.loads(s) for s in (work/'flow_calls.jsonl').read_text().splitlines()]
        assert flows and all(math.isfinite(r['q']) and r['q']>=0 for r in flows)
        life=summary['native_lifetime'];assert life['numeric_created']==life['numeric_released'] and life['symbolic_created']==life['symbolic_released']
        entry=passed['entry_pressure_pa'];origin={str(work/'CORE_PASS.toml'):sha(work/'CORE_PASS.toml')}
        dense=np.array([[r['pressure_pa'],r['sg_physical'],0,0,r['trapped_pv_fraction'],r['immobile_pv_fraction'],r['gas_breakthrough'],r['invaded_pv_fraction']] for r in pc]);error_reference=None
    assert np.isfinite(dense).all() and np.all(np.diff(dense[:,0])>=0) and np.all(np.diff(dense[:,1])>=-1e-12)
    dense[:,:4]=paired_grid(pc,kr)
    table,detail=convert(dense,entry)
    out.mkdir(parents=True,exist_ok=False)
    for name,data,header in [('paired.csv',dense,','.join(NAMES)),('sgof.csv',table,'Sg,Krg,Krw,PcPa')]:
        np.savetxt(out/name,data,delimiter=',',header=header,comments='',fmt='%.17g')
        assert np.array_equal(np.loadtxt(out/name,delimiter=',',skiprows=1),data)
    errors=None if error_reference is None else np.abs(error_reference[:,2:4]-dense[:,2:4]).max(axis=0).tolist()
    result=dict(passed=True,key=item['key'],old_numerical_key=item['old_numerical_key'],kind=item['kind'],selected_indices=[i+1 for i in ids],
                actual_sg=[r['sg_physical'] for r in kr],targets=targets,pc100_preserved=True,selected_kr_preserved=True,
                max_abs_sampling_error_brine_gas=errors,sampling_error_status='UNKNOWN_WITHOUT_REFERENCE' if errors is None else 'RECORDED_NOT_A_FAILURE',
                approximation_flag=None if errors is None else max(errors)>.08,export=detail,source=origin,
                artifacts={str(out/n):sha(out/n) for n in ['paired.csv','sgof.csv']})
    write(mark,result);return result

if __name__=='__main__':
    spec=read(sys.argv[1]);result=[];held=[]
    for item in spec['items']:
        try:result.append(export(item,Path(spec['out'])/item['key']))
        except Exception as e:
            import traceback
            held.append(dict(key=item['key'],error=traceback.format_exc()))
    write(Path(spec['out'])/'EXPORT_BATCH.json',dict(passed=bool(result),all_passed=not held,records=result,held=held))
    assert result,'No exported cores passed; preserve existing numerical outputs'
