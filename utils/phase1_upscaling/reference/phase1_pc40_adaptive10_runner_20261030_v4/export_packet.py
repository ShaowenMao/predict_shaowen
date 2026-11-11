"""Validate adaptive decisions and export checked saved flow; never solve."""
import csv
import json
import sys
import traceback
from pathlib import Path
try:import tomllib
except ImportError:from pip._vendor import tomli as tomllib
import numpy as np

req=json.loads(Path(sys.argv[1]).read_text());src=Path(req['science']);sys.path.insert(0,str(src))
from integration_check import select_trace
from baseline_verify import load_module,sha
planning=load_module('planning');tables=load_module('build_tables')
def export_one(r):
    w=Path(r['work']);out=w/'export'
    if (out/'EXPORT_PASS.json').exists():
        proof=json.loads((out/'EXPORT_PASS.json').read_text());assert proof['passed'] and proof['sha256']==sha(out/'sgof.csv')
        return
    if out.exists():raise RuntimeError('Incomplete export retained; do not overwrite')
    receipt=tomllib.loads((w/'flow/CORE_PASS.toml').read_text())
    assert receipt['passed'] and receipt['record_sha256']==r['record_sha256'] and receipt['method']==req['method']
    for name,h in receipt['artifacts'].items():assert sha(w/'flow'/name)==h
    def rows(n):
        return [{k:v if k=='state_hash' else float(v) for k,v in a.items()} for a in csv.DictReader((w/'flow'/n).open())]
    pc,kr=rows('pc100.csv'),rows('kr.csv')
    observed={int(a['sample']):(a['krw'],a['krg']) for a in kr}
    ids,order=select_trace([a['sg_physical'] for a in pc],lambda i:observed[i])
    summary=tomllib.loads((w/'flow/summary.toml').read_text())
    assert ids==summary['selected_indices']==[int(a['sample']) for a in kr] and order==summary['query_order']
    assert summary['mode']=='adaptive10' and summary['same_mip_microstates']
    dense=np.array([[a['pressure_pa'],a['sg_physical'],0,0,a['trapped_pv_fraction'],a['immobile_pv_fraction'],a['gas_breakthrough'],a['invaded_pv_fraction']] for a in pc])
    dense[:,:4]=planning.paired_grid(pc,kr)
    t,detail=tables.convert(dense,receipt['input']['entry_pressure_pa'])
    out.mkdir()
    np.savetxt(out/'sgof.csv',t,delimiter=',',header='Sg,Krg,Krw,PcPa',comments='',fmt='%.17g')
    with (out/'EXPORT_PASS.json').open('x') as f:json.dump(dict(passed=True,sha256=sha(out/'sgof.csv'),detail=detail),f)
    print('EXPORT_PASS',r['key'],flush=True)

for r in req['cases']:
    try:export_one(r)
    except Exception:
        out=Path(r['work'])/'POSTFLOW_HOLD.json'
        if not out.exists():
            with out.open('x') as f:json.dump(dict(stage='export',error=traceback.format_exc()),f)
        print('EXPORT_HOLD',r['key'],flush=True)
