"""Select ten existing states; keep every original Pc value and placement."""
import csv
from pathlib import Path
import sys
import numpy as np
from common import HERE, read, write, sha, verify, pins_ok
from planning import select_states, paired_grid
from build_tables import convert

NAMES = ['pressure_pa', 'sg_physical', 'krw', 'krg', 'trapped_pv_fraction',
         'immobile_pv_fraction', 'gas_breakthrough', 'invaded_pv_fraction']


def load(p):
    with Path(p).open() as f:
        r = list(csv.DictReader(f))
    assert len(r)==100 and list(r[0])==NAMES
    return [{**{k:float(v) for k,v in row.items()}, 'sample':i+1} for i,row in enumerate(r)]


def one(row):
    folder = HERE/'owners'/row['key']
    mark = folder/'PASS.json'
    if mark.exists():
        r = read(mark)
        assert r['key']==row['key'] and r['source_curve_sha256']==row['curve_sha256']
        assert r['lock_sha256']==sha(HERE/'LOCK.json')
        pins_ok(r['artifacts'])
        return r, True
    pins_ok({row['curve']:row['curve_sha256']})
    source = load(row['curve'])
    ids, targets = select_states([x['sg_physical'] for x in source])
    selected = [source[i] for i in ids]
    dense = np.array([[r[k] for k in NAMES] for r in source])
    paired = dense.copy()
    paired[:,:4] = paired_grid(source, selected)
    assert np.array_equal(paired[:,[0,1,4,5,6,7]], dense[:,[0,1,4,5,6,7]])
    assert np.array_equal(paired[ids,2:4],dense[ids,2:4])
    table, detail = convert(paired,row['entry_pressure_pa'])
    control, _ = convert(dense,row['entry_pressure_pa'])
    assert np.array_equal(table[:,[0,3]],control[:,[0,3]])
    folder.mkdir(parents=True, exist_ok=False)
    paths = {'paired.csv':paired,'selected.csv':dense[ids], 'sgof.csv':table}
    for name, values in paths.items():
        header = ','.join(NAMES) if name!='sgof.csv' else 'Sg,Krg,Krw,PcPa'
        np.savetxt(folder/name,values,delimiter=',',header=header,comments='',fmt='%.17g')
        assert np.array_equal(np.loadtxt(folder/name,delimiter=',',skiprows=1),values)
    errors = np.max(np.abs(paired[:,2:4]-dense[:,2:4]),axis=0)
    result = dict(passed=True,key=row['key'], source_curve=row['curve'],source_curve_sha256=row['curve_sha256'],
                  lock_sha256=sha(HERE/'LOCK.json'), sample_ids=[i+1 for i in ids], targets=targets,
                  selected_actual_sg=[r['sg_physical'] for r in selected], actual_pressure_pa=[r['pressure_pa'] for r in selected],
                  max_abs_krw_error=float(errors[0]), max_abs_krg_error=float(errors[1]),
                  approximation_review_flag=bool(max(errors)>.08), original_pc100_exact=True, selected_kr_exact=True,
                  pc_export_exact=True, flow_solves=0, mip_reruns=0, export_review=detail,
                  artifacts={str(folder/name):sha(folder/name) for name in paths})
    write(mark,result)
    return result,False


def main(stage):
    verify()
    plan = read(HERE/'audit/PLAN.json')
    index = 0 if stage=='prefix' else int(stage.removeprefix('derive'))
    keys = plan['batches'][index]
    if stage=='prefix':
        keys = keys[:2]
    records=[]
    for key in keys:
        row = plan['owners'][key]
        r,reused = one(row)
        records.append(dict(key=key,reused=reused,receipt=str(HERE/'owners'/key/'PASS.json'),
                            receipt_sha256=sha(HERE/'owners'/key/'PASS.json')))
        print('OWNER_PASS',key[:16],'reused',reused,flush=True)
    if stage=='derive000':
        assert len(records)>=2 and all(r['reused'] for r in records[:2])
    write(HERE/'batches'/(stage+'.json'),dict(passed=True,records=records,unfinished_only=True))


if __name__=='__main__':main(sys.argv[1])
