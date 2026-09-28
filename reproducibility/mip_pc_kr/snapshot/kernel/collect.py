"""Export sparse curves without solving or changing existing artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat, loadmat

from build_tables import convert
from planning import paired_grid, select_states

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:
        return [{k:float(v) for k,v in r.items() if k!='state_hash'} for r in csv.DictReader(f)]


def paired_raw(pc, kr):
    t=paired_grid(pc,kr)
    x=np.zeros((100,8));x[:,:4]=t
    x[:,4]=[r['trapped_pv_fraction'] for r in pc]
    x[:,5]=[r['immobile_pv_fraction'] for r in pc]
    x[:,6]=[r['gas_breakthrough'] for r in pc]
    x[:,7]=[r['invaded_pv_fraction'] for r in pc]
    return x


def main():
    out=Path(sys.argv[1]);m=read(HERE/'manifest.json');packet=read(out/'inputs.json')
    assert sha(HERE/'manifest.json')==packet['manifest_sha256']
    report=[]
    for key,c in m['cases'].items():
        pc=rows(c['reference_pc']);full=rows(c['reference_kr'])
        ids,targets=select_states([r['sg_physical'] for r in pc])
        sparse=[full[i] for i in ids]
        raw=paired_raw(pc,sparse);entry=packet['entries'][key]
        table,detail=convert(raw,entry)
        # Identical Pc columns to the original exporter applied to the full100 curve.
        dense=paired_raw(pc,sparse)
        dense[:,2:4]=[[r['krw'],r['krg']] for r in full]
        dense_table,_=convert(dense,entry)
        assert np.array_equal(table[:,[0,3]],dense_table[:,[0,3]])
        checks=[]
        for item in packet['records']:
            if item['key']!=key:continue
            folder=Path(item['receipt']).parent
            actual_pc=rows(folder/'pc100.csv');actual_kr=rows(folder/'kr.csv')
            fresh,review=convert(paired_raw(actual_pc,actual_kr),entry)
            assert fresh.shape==table.shape
            assert np.array_equal(fresh[:,3],table[:,3])
            assert np.all(np.abs(fresh[:,0]-table[:,0]) <= 32*np.finfo(float).eps*np.maximum(np.abs(fresh[:,0]),np.abs(table[:,0])))
            error=float(np.max(np.abs(fresh[:,1:3]-table[:,1:3])))
            assert error<=1e-8
            assert np.all(np.abs(fresh[:,1:3]-table[:,1:3])<=1e-12+1e-5*np.abs(table[:,1:3]))
            assert np.array_equal(fresh[:,1:3]==0,table[:,1:3]==0)
            checks.append(dict(batch=item['batch'],reused=item['reused'],max_abs_difference=error,
                               saturation_grid_max_abs_difference=float(np.max(np.abs(fresh[:,0]-table[:,0]))),
                               receipt=item['receipt'],receipt_sha256=sha(item['receipt'])))
        folder=out/key;folder.mkdir()
        path=folder/'sgof.csv'
        np.savetxt(path,table,delimiter=',',header='Sg,Krg,Krw,PcPa',comments='',fmt='%.17g')
        savemat(folder/'sgof.mat',dict(table=table))
        assert np.array_equal(loadmat(folder/'sgof.mat')['table'],table)
        assert np.array_equal(np.loadtxt(path,delimiter=',',skiprows=1),table)
        report.append(dict(key=key,table_path=str(path),table_sha256=sha(path),
                           exported_rows=len(table),selected_states=len(ids),entry_pressure_pa=entry,
                           selected_sample_ids=[i+1 for i in ids],pc_export_unchanged=True,
                           full100_flow_recomputed=False,fresh_sparse_checks=checks,export_review=detail))
        print('EXPORT_PASS',key,len(table),'rows',flush=True)
    assert len(report)==8 and sum(len(r['fresh_sparse_checks']) for r in report)==12
    with (out/'EXPORT_PASS.json').open('x') as f:
        json.dump(dict(passed=True,cores=report,full_fault_mapping_test=False,
                       native_import_passed=False,production_modified=False),f,indent=2)


if __name__=='__main__':main()
