"""Build a separate Pc-only intervention from frozen candidate tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from scipy.io import loadmat, savemat

RAW_SHA = 'ce79f0c5db718157171754845eff40776e25095d81c76f06ae8eab702eb75f92'
TABLE_SHA = 'b88dd2d8a7b9d4f612611f88e35c3ce6c93a94313a4877e513c169ad18a9c481'
SPECIFIC_SHA = '82fac5cd1e40a65ee26167dc6e54224b469ee84adce5c03148082c0dd7c14c71'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def convert(table, raw, pb):
    t = np.asarray(table, dtype=float)
    raw = np.asarray(raw, dtype=float)
    assert np.isfinite(t).all() and np.isfinite(raw).all()
    assert t.shape[1] == 4 and raw.shape[1] == 8
    assert np.all(np.diff(t[:, 0]) > 0) and t[0, 0] == 0
    assert np.all(np.diff(raw[:, 0]) > 0)
    flags = raw[:, 6]
    assert np.isin(flags, [0, 1]).all() and np.all(np.diff(flags) >= 0)
    hits = np.flatnonzero(flags == 1)
    assert len(hits), 'No breakthrough; do not invent an entry threshold'
    b = int(hits[0])
    assert b > 0 and raw[b-1, 0] < pb <= raw[b, 0]
    assert t[0, 3] <= pb and pb > 0
    sb, pfirst = raw[b, :2][::-1]
    assert sb > 0
    j = int(np.searchsorted(t[:, 0], sb))
    assert j < len(t) and t[j, 0] == sb and t[j, 3] == pfirst
    out = t.copy()
    # All zero-prebreakthrough raw rows collapse to the exact (Sg=0, Pc=Pb)
    # anchor. The unresolved onset connects linearly to the first retained
    # post-breakthrough sample. Existing Sg knots preserve both Kr functions.
    pre = t[:, 0] < sb
    out[pre, 3] = pb + (pfirst-pb)*(t[pre, 0]/sb)
    assert np.array_equal(out[:, :3], t[:, :3])
    assert np.array_equal(out[~pre], t[~pre])
    assert np.array_equal(out[-1], t[-1])
    assert np.all(np.diff(out[:, 3]) >= 0)
    assert np.all(out[:, 3] >= t[:, 3])
    assert out[0, 3] == pb
    slopes = np.diff(out[:, 3])/np.diff(out[:, 0])
    assert np.isfinite(slopes).all()
    report = dict(breakthrough_pa=float(pb), first_post_pressure_pa=float(pfirst),
        last_pre_pressure_pa=float(raw[b-1, 0]), first_post_sg=float(sb),
        last_pre_sg=float(raw[b-1, 1]), original_entry_pa=float(t[0, 3]),
        entry_multiplier=float(pb/t[0, 3]), first_post_raw_index=b,
        changed_pc_nodes=int(np.count_nonzero(out[:, 3] != t[:, 3])),
        onset_pressure_bracket_relative_width=float((pfirst-pb)/pb),
        maximum_pc_slope_before=float(np.max(np.diff(t[:, 3])/np.diff(t[:, 0]))),
        maximum_pc_slope_after=float(np.max(slopes)),
        sg_kr_columns_bitwise_equal=True, post_breakthrough_nodes_exact=True,
        all_terminal_values_exact=True)
    return out, report


def text_dataset(f, path, text):
    old = f[path]
    assert old.attrs['MATLAB_class'] in (b'char', 'char')
    attrs = dict(old.attrs)
    parent, name = path.rsplit('/', 1)
    del f[path]
    new = f[parent].create_dataset(name, data=np.array([ord(c) for c in text], dtype='uint16')[:, None])
    for k, v in attrs.items():
        new.attrs[k] = v


def build(raw_path, tables_path, specific_path, breakthrough_path, out):
    assert not out.exists()
    for p, h in [(raw_path, RAW_SHA), (tables_path, TABLE_SHA), (specific_path, SPECIFIC_SHA)]:
        assert sha(p) == h, f'Frozen baseline hash mismatch: {p}'
    entries = {int(r['slot']): r for r in csv.DictReader(breakthrough_path.open())}
    assert set(entries) == set(range(1, 523))
    old = loadmat(tables_path)['exportTables']
    assert old.shape == (6, 87)
    new = np.empty_like(old)
    reports = []
    with h5py.File(raw_path) as f:
        refs = f['candidateRaw/rawCurves']
        assert refs.shape == (87, 6)
        for w in range(6):
            for s in range(87):
                slot = w*87+s+1
                raw = f[refs[s, w]][:].T
                assert float(entries[slot]['entry_pressure_pa']) == old[w, s][0, 3]
                new[w, s], r = convert(old[w, s], raw, float(entries[slot]['breakthrough_pa']))
                r.update(slot=slot, window=w+1, slice=s+1)
                reports.append(r)
    out.mkdir(parents=True)
    target = out/'candidate_specific_pc_breakthrough_s03_c022_case12.mat'
    shutil.copyfile(specific_path, target)
    with h5py.File(target, 'r+') as f:
        sgof = f['fault/fluid_tables/SGOF']
        refs = sgof['tables'][:].reshape(-1)
        indices = sgof['source_curve_linear_indices'][:].reshape(-1).astype(int)
        assert sorted(indices) == list(range(1, 523))
        seen = set()
        for ref, index in zip(refs, indices):
            w, s = (index-1) % 6, (index-1)//6
            ds = f[ref]
            assert ds.name not in seen
            seen.add(ds.name)
            assert np.array_equal(ds[:].T, old[w, s])
            ds[3, :] = new[w, s][:, 3]
            assert np.array_equal(ds[:].T, new[w, s])
        text_dataset(f, 'metadata/pc_curve_treatment',
            'Pc-only sensitivity: Sg=0 before exact core breakthrough; original post-breakthrough branch and Kr(Sg) unchanged')
        text_dataset(f, 'metadata/note',
            'Isolated diagnostic, not production promotion. Pc onset linearly connects (0,Pb) to first sampled post-breakthrough state; see external pc_only_audit.json.')
        text_dataset(f, 'candidate_export/schema', 'candidate_pc_only_breakthrough_test_v1')
        text_dataset(f, 'candidate_export/table_audit_file', 'pc_only_audit.json')
    mat = out/'candidate_tables_pc_breakthrough_s03_c022_case12.mat'
    savemat(mat, dict(exportTables=new, tableSlotMap=np.arange(1, 523).reshape(6, 87),
        schema='candidate_pc_only_breakthrough_test_v1', productionPromotion=False), do_compression=True)
    reread = loadmat(mat)['exportTables']
    assert all(np.array_equal(new[w,s], reread[w,s]) for w in range(6) for s in range(87))
    with h5py.File(target) as f:
        g=f['fault/fluid_tables/SGOF']
        for ref, ind in zip(g['tables'][:].reshape(-1), g['source_curve_linear_indices'][:].reshape(-1).astype(int)):
            assert np.array_equal(f[ref][:].T, new[(ind-1)%6, (ind-1)//6])
    report = dict(passed=True, cores=522, baseline_specific_sha256=SPECIFIC_SHA,
        baseline_tables_sha256=TABLE_SHA, raw_sha256=RAW_SHA,
        breakthrough_sha256=sha(breakthrough_path), specific_sha256=sha(target), table_sha256=sha(mat),
        source_sha256=sha(__file__), all_kr_and_saturation_coordinates_exact=True,
        all_post_breakthrough_pc_nodes_exact=True, all_endpoint_values_exact=True,
        no_new_mip_sweeps=True, no_phase_flow_solves=True,
        zero_prebreakthrough_convention='exact Pb at Sg=0; linear onset to first sampled post-breakthrough state',
        numerical_interpretation='No strict gas-entry switch added to simulator; existing Pc importer and flux formulation unchanged',
        reservoir_export_requires_native_preflight=True, production_promotion=False, records=reports)
    (out/'pc_only_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for p, h in [(raw_path, RAW_SHA), (tables_path, TABLE_SHA), (specific_path, SPECIFIC_SHA)]:
        assert sha(p) == h
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('raw', 'tables', 'specific', 'breakthrough', 'output'):
        p.add_argument(name, type=Path)
    a=p.parse_args()
    build(a.raw,a.tables,a.specific,a.breakthrough,a.output)
