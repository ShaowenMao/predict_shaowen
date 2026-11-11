"""Explicit, bounded conversion of parametric drainage curves to SGOF tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat, loadmat

NOISE_BUDGET = 1e-9


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(raw, entry):
    x = np.asarray(raw, dtype=float)
    assert x.ndim == 2 and x.shape[1] == 8 and len(x) >= 3
    assert np.isfinite(x).all() and np.isfinite(entry) and entry > 0
    p, s, kw, kg = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    assert p[0] == s[0] == 0 and np.all(np.diff(p) > 0)
    assert np.all(np.diff(s) >= 0) and np.all((s >= 0) & (s <= 1))
    assert np.all((x[:, 2:4] >= 0) & (x[:, 2:4] <= 1+NOISE_BUDGET))
    positive = np.flatnonzero(s > 0)
    assert len(positive) >= 2
    first = int(positive[0])
    assert p[first-1] <= entry <= p[first], 'Exact entry is outside the sampled arrival bracket'
    repeats = np.flatnonzero(np.diff(s) == 0)
    assert all(s[i] == 0 or s[i] == s[-1] for i in repeats), 'Interior vertical branch requires separate review'
    assert all(np.array_equal(x[i, 2:4], x[i+1, 2:4]) for i in repeats), 'Kr differs at repeated Sg'
    # Retain the pressure at which the terminal saturation was first sampled.
    # Later pressures at identical Sg/Kr do not change the approach to the endpoint.
    keep = np.r_[0, np.flatnonzero(np.diff(s) > 0) + 1]
    t = x[keep][:, [1, 3, 2, 0]].copy()  # SGOF: Sg, Krg, Krw, Pc(Pa)
    t[0, 3] = entry
    assert np.all(np.diff(t[:, 0]) > 1e-12)
    assert np.all(np.diff(t[:, 3]) >= 0)
    before = t[:, 1:3].copy()
    # One-sided monotone envelopes remove only bounded solver-level reversals.
    # Endpoints and the exact zero/nonzero mobility pattern must stay unchanged.
    t[:, 1] = np.minimum.accumulate(t[::-1, 1])[::-1]
    t[:, 2] = np.minimum.accumulate(t[:, 2])
    delta = np.max(np.abs(t[:, 1:3] - before), axis=0)
    assert max(delta) <= NOISE_BUDGET, ('Kr correction exceeds numerical budget', delta)
    assert np.all((t[:, 1:3]>=0)&(t[:, 1:3]<=1))
    assert np.array_equal(t[:, 1:3] == 0, before == 0), 'Mobility connectivity would change'
    assert np.array_equal(t[[0, -1], 1:3], before[[0, -1]]), 'Kr endpoints would change'
    assert t[0, 1] == 0 and t[0, 2] == 1
    assert t[-1, 2] == 0, 'Positive terminal brine mobility requires a different endpoint policy'
    assert np.all(np.diff(t[:, 1]) >= 0) and np.all(np.diff(t[:, 2]) <= 0)
    assert np.array_equal(t[1:, 0], s[keep[1:]])
    assert np.array_equal(t[1:, 3], p[keep[1:]])
    slopes = np.diff(t[:, 3]) / np.diff(t[:, 0])
    assert np.isfinite(slopes).all()
    terminal = int(keep[-1])
    return t, dict(raw_points=len(x), exported_points=len(t),
        retained_raw_indices_zero_based=keep.tolist(),
        initial_zero_rows=first, terminal_rows_removed=len(x)-terminal-1,
        entry_pressure_pa=float(entry), entry_bracket_pa=[float(p[first-1]), float(p[first])],
        entry_not_core_breakthrough=True,
        first_positive_sg=float(s[first]), maximum_sg=float(s[-1]),
        terminal_pc_pa=float(p[terminal]), final_sweep_pc_pa=float(p[-1]),
        terminal_krg=float(t[-1, 1]), terminal_krw=float(t[-1, 2]),
        max_absolute_krg_change=float(delta[0]), max_absolute_krw_change=float(delta[1]),
        changed_krg_nodes=int(np.count_nonzero(t[:, 1] != before[:, 0])),
        changed_krw_nodes=int(np.count_nonzero(t[:, 2] != before[:, 1])),
        kr_noise_budget=NOISE_BUDGET, kr_zero_pattern_preserved=True,
        positive_saturation_pc_nodes_bitwise_preserved=True,
        max_pc_slope_pa=float(max(slopes)),
        extrapolation='constant endpoint values; not a hard saturation constraint',
        continuous_sampling_accuracy_claimed=False)


def build(rawfile, entryfile, out):
    assert not out.exists()
    entry_rows = list(csv.DictReader(entryfile.open()))
    entries = {int(r['slot']): float(r['entry_pressure_pa']) for r in entry_rows}
    assert set(entries) == set(range(1, 523))
    tables = np.empty((6, 87), dtype=object)
    reports = []
    with h5py.File(rawfile) as f:
        refs = f['candidateRaw/rawCurves']
        assert refs.shape == (87, 6)
        for w in range(6):
            for s in range(87):
                slot = w*87+s+1
                raw = f[refs[s, w]][:].T
                assert raw.shape == (100, 8)
                table, report = convert(raw, entries[slot])
                report.update(slot=slot, window=w+1, slice=s+1)
                tables[w, s] = table
                reports.append(report)
    out.mkdir(parents=True)
    curves = out/'tables'; curves.mkdir()
    for r in reports:
        name = 'slot%03d.csv' % r['slot']
        np.savetxt(curves/name, tables[r['window']-1, r['slice']-1], delimiter=',',
                   header='Sg,Krg,Krw,PcPa', comments='', fmt='%.17g')
    mat = out/'candidate_tables_s03_c022_case12.mat'
    savemat(mat, dict(exportTables=tables,
        entryPressurePa=np.array([entries[i] for i in range(1,523)]).reshape(6,87),
        tableSlotMap=np.arange(1,523).reshape(6,87),
        rawSourceSha256=sha(rawfile), entryMetadataSha256=sha(entryfile),
        schema='candidate_sgof_native_nodes_v1', productionPromotion=False,
        validationReceiptRequired=True), do_compression=True)
    reread = loadmat(mat)['exportTables']
    assert all(np.array_equal(tables[w,s],reread[w,s]) for w in range(6) for s in range(87))
    summary = dict(cores=522, all_structural_checks_passed=True,
        raw_sha256=sha(rawfile), entry_metadata_sha256=sha(entryfile),
        table_mat_sha256=sha(mat), source_sha256=sha(__file__),
        initial_zero_rows_counts={str(n):sum(r['initial_zero_rows']==n for r in reports) for n in (2,3)},
        terminal_plateau_slots=[r['slot'] for r in reports if r['terminal_rows_removed']],
        max_absolute_krg_change=max(r['max_absolute_krg_change'] for r in reports),
        max_absolute_krw_change=max(r['max_absolute_krw_change'] for r in reports),
        cores_with_kr_cleanup=sum(r['changed_krg_nodes']+r['changed_krw_nodes']>0 for r in reports),
        exported_node_range=[min(r['exported_points'] for r in reports),max(r['exported_points'] for r in reports)],
        sampled_endpoint_sg_range=[min(r['maximum_sg'] for r in reports),max(r['maximum_sg'] for r in reports)],
        entry_pressure_pa_range=[min(entries.values()),max(entries.values())],
        runtime_validation_passed=False, production_promotion=False,
        continuous_sampling_accuracy_claimed=False, records=reports)
    (out/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path);parser.add_argument('entries',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();result=build(args.raw,args.entries,args.out)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
