"""Produce private exact-node tables from verified saved flow, without solving."""
import csv
import hashlib
import json
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib
import numpy as np
from exact_export import load_converter, POLICY

request = json.loads(Path(sys.argv[1]).read_text())
science = Path(request['science'])
sys.path.insert(0, str(science))
from integration_check import select_trace
from baseline_verify import load_module
planning = load_module('planning')
tables = load_module('build_tables')
convert = load_converter(science/'vendor/kernel/build_tables.py', request['table_source_sha256'])
for row in request['cases']:
    work = Path(row['work'])
    proof = tomllib.loads((work/'flow/CORE_PASS.toml').read_text())
    assert proof['passed'] and proof['method'] == request['method'] and proof['record_sha256'] == row['record_sha256']
    for name, h in proof['artifacts'].items():
        assert hashlib.sha256((work/'flow'/name).read_bytes()).hexdigest() == h
    def read(name):
        with (work/'flow'/name).open() as f:
            return [{k:v if k == 'state_hash' else float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    pc, kr = read('pc100.csv'), read('kr.csv')
    observed = {int(r['sample']):(r['krw'],r['krg']) for r in kr}
    ids, order = select_trace([r['sg_physical'] for r in pc], lambda i:observed[i])
    summary = tomllib.loads((work/'flow/summary.toml').read_text())
    assert ids == summary['selected_indices'] == [int(r['sample']) for r in kr] and order == summary['query_order']
    dense = np.array([[r['pressure_pa'],r['sg_physical'],0,0,r['trapped_pv_fraction'],r['immobile_pv_fraction'],r['gas_breakthrough'],r['invaded_pv_fraction']] for r in pc])
    dense[:,:4] = planning.paired_grid(pc,kr)
    try:
        tables.convert(dense, proof['input']['entry_pressure_pa'])
    except AssertionError as error:
        import traceback
        assert 'np.diff(t[:, 0]) > 1e-12' in ''.join(traceback.format_exception(type(error),error,error.__traceback__))
    else:
        raise AssertionError('Not a minimum-spacing held core')
    table, detail = convert(dense, proof['input']['entry_pressure_pa'])
    keep = np.r_[0, np.flatnonzero(np.diff(dense[:,1]) > 0)+1]
    assert np.array_equal(table[:,0], dense[keep,1])
    assert np.array_equal(table[1:,3], dense[keep[1:],0])
    assert np.all(np.diff(table[:,0])>0)
    out = work/'export';out.mkdir()
    np.savetxt(out/'sgof.csv',table,delimiter=',',header='Sg,Krg,Krw,PcPa',comments='',fmt='%.17g')
    reread = np.loadtxt(out/'sgof.csv',delimiter=',',skiprows=1)
    assert np.array_equal(table,reread)
    details = dict(passed=True,sha256=hashlib.sha256((out/'sgof.csv').read_bytes()).hexdigest(),detail=detail,export_policy=POLICY,export_source=request['source'],legacy_table_source_sha256=request['table_source_sha256'],minimum_positive_spacing=float(min(np.diff(table[:,0]))),raw_curve_values_unchanged=True,all_distinct_positive_saturation_nodes_preserved=True,native_validation_required=True,new_solves=0)
    with (out/'EXPORT_PASS.json').open('x') as f:json.dump(details,f,indent=2,allow_nan=False)
    print('PRIVATE_EXPORT_PASS',row['key'],flush=True)
