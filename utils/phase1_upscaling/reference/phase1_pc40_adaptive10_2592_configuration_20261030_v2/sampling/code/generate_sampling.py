"""New scenario-independent sampling; all existing inputs are read-only."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import platform
import sys
import tomllib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('D:/codex_gom/UQ_workflow/independent_full_fault_v1_phase1_20260809_v1')
GEOLOGY = Path('D:/codex_gom/UQ_workflow/three_level_sampling_162_collapsed_cell_union/catalog/level1_geology_catalog.csv')
INVENTORY = Path('D:/codex_gom/UQ_workflow/production_freeze_collapsed_cell_union_20260722_v7/inventory/predict_files_sha256.csv')
CURVES = Path('D:/codex_gom/studies/lu2017_six_pc70_table_entry_monotonic_20261027_v1/pc70_table_entry_monotonic_plot_points.csv')
JULIA_SOURCE = Path('D:/Github/predict_shaowen_independent_full_fault_v1/examples/Julia_analyses/UQ_workflow/sampling/lib/sampling_provenance.jl')
WINDOWS = [f'famp{i}' for i in range(1, 7)]
VERSION = 'sha256_scenario_pc_u63_counter_u64_rejection_v1'
OLD_VERSION = 'sha256_identity_u63_counter_u64_rejection_v1'
BASE_SEED = 20261029
DESIGN = 'independent_full_fault_pc40_324_v1'
SAMPLES = (4, 6)
NREP, NSLICE, NPOOL = 8, 87, 2000
FIELDS = ['schema_version', 'design_version', 'geology_id', 'scenario_id',
          'clay_pc_sample_id', 'co2_brine_contact_angle_deg', 'phase', 'case_id',
          'case_type', 'replicate_id', 'window_id', 'slice_id', 'library_id',
          'source_library', 'source_library_row', 'predict_realization_id',
          'predict_seed', 'exact_replay_seed', 'sampling_seed', 'sampling_seed_method',
          'selection_method', 'checkpoint_path', 'checkpoint_relative_path',
          'checkpoint_hash', 'level2_state_path', 'sampling_config_sha256',
          'predict_code_commit', 'predict_method_config_hash', 'log_kxx', 'log_kyy',
          'log_kzz', 'perm_kxx_md', 'perm_kyy_md', 'perm_kzz_md',
          'use_for_probabilistic_uq', 'is_benchmark', 'is_stress_test']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty catalog')
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


@contextmanager
def gzip_csv(path):
    # A fixed gzip header makes the canonical manifest reproducible byte-for-byte.
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=5) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
                w.writeheader()
                yield w


def payload(*fields):
    return b''.join(str(len(b)).encode() + b':' + b + b'|'
                    for b in (str(field).encode('utf-8') for field in fields))


def stable_seed(base, identity, replicate, window, version=VERSION):
    h = hashlib.sha256(payload(version, base, identity, 'phase1',
                               'independent_full', replicate, window)).digest()
    return (int.from_bytes(h[:8], 'big') & ((1 << 63) - 1)) or 1


def uniform_index(seed, draw, n=NPOOL, version=VERSION):
    if n <= 0 or n >= 1 << 64 or draw <= 0:
        raise ValueError('Invalid sampling dimensions')
    threshold = (1 << 64) % n
    attempt = 0
    while True:
        h = hashlib.sha256(payload(version, seed, draw, attempt)).digest()
        value = int.from_bytes(h[:8], 'big')
        if value >= threshold:
            return value % n + 1
        attempt += 1


def checked_integer_array(values, label):
    a = np.asarray(values).reshape(-1)
    if not (np.isfinite(a).all() and np.equal(a, np.floor(a)).all() and (a > 0).all()):
        raise ValueError(f'Invalid integer data: {label}')
    return a.astype(np.int64)


def load_library(row, geo, frozen):
    p = Path(row['checkpoint_path'])
    digest = sha(p)
    if digest != row['checkpoint_hash'] or digest != frozen[row['checkpoint_relative_path']]:
        raise ValueError(f'PREDICT source hash mismatch: {p}')
    state_path = Path(row['level2_state_path'])
    with h5py.File(p, 'r') as f:
        k = np.asarray(f['perms'], dtype=np.float64).T.copy()
        seeds = checked_integer_array(f['meta/AcceptedSeeds'][()], 'AcceptedSeeds')
        ids = checked_integer_array(f['meta/AcceptedAttemptIndices'][()], 'AcceptedAttemptIndices')
        tracking = f.get('meta/AcceptedSeedTracking', f.get('checkpointInfo/AcceptedSeedTracking'))
        if tracking is None or not bool(np.asarray(tracking).item()):
            raise ValueError(f'Replay seeds are not tracked: {p}')
        info = f['checkpointInfo']
        for key, expected in [('FaultingDepth', geo['faulting_depth_m']),
                              ('SandVcl', geo['sand_vcl']), ('ClayVcl', geo['clay_vcl']),
                              ('ScenarioIndex', geo['scenario_index']), ('CaseIndex', geo['case_index'])]:
            if not np.isclose(float(np.asarray(info[key]).item()), float(expected), rtol=0, atol=1e-12):
                raise ValueError(f'Geology mismatch: {p}: {key}')
    if k.shape != (NPOOL, 3) or not np.isfinite(k).all() or not (k > 0).all():
        raise ValueError(f'Invalid joint permeability library: {p}')
    if any(len(a) != NPOOL or len(np.unique(a)) != NPOOL for a in (ids, seeds)):
        raise ValueError(f'Invalid accepted IDs/seeds: {p}')
    if int(row['n_accepted']) != NPOOL:
        raise ValueError('Unexpected library population')
    with h5py.File(state_path, 'r') as f:
        if not np.array_equal(np.asarray(f['raw_perms']).T, k):
            raise ValueError(f'Level-2/source permeability mismatch: {state_path}')
        if not np.allclose(np.asarray(f['log_perms']).T, np.log10(k), rtol=0, atol=2e-14):
            raise ValueError(f'Level-2/source log permeability mismatch: {state_path}')
    return k, seeds, ids, sha(state_path)


def main():
    out = ROOT / 'phase1'
    if out.exists():
        raise SystemExit('Refusing to overwrite an existing phase1 directory')
    geos = read_csv(GEOLOGY)
    if len(geos) != 162 or len({g['geology_id'] for g in geos}) != 162:
        raise ValueError('Expected exactly 162 base geologies')
    geos.sort(key=lambda g: (int(g['scenario_index']), int(g['case_index'])))
    old_config = tomllib.loads((OLD / 'config/effective_sampling_config.toml').read_text())
    frozen = {r['relative_path'].replace('\\', '/'): r['sha256'] for r in read_csv(INVENTORY)}
    old_manifest = OLD / 'phase1/phase1_sampling_manifest_all_geologies.csv'
    protected = [GEOLOGY, INVENTORY, CURVES, JULIA_SOURCE, old_manifest,
                 OLD / 'config/effective_sampling_config.toml']
    protected += [OLD / 'libraries' / f"{g['geology_id']}_library_catalog.csv" for g in geos]
    protected_before = {p.as_posix(): sha(p) for p in protected}
    config = {
        'design_version': DESIGN, 'scope': 'sampling_only', 'base_seed': BASE_SEED,
        'seed_method_version': VERSION, 'seed_identity': 'complete scenario_id including clay Pc sample and angle',
        'base_geologies': 162, 'scenarios': 324, 'realizations_per_scenario': NREP,
        'reservoir_cases': 2592, 'windows': WINDOWS, 'slices_per_window': NSLICE,
        'accepted_library_size': NPOOL, 'clay_pc_samples': list(SAMPLES),
        'co2_brine_contact_angle_deg': 40, 'with_replacement': True,
        'cross_window': 'conditionally_independent', 'along_strike': 'conditionally_independent',
        'cross_pc': 'independently_sampled_not_paired', 'joint_permeability_components_preserved': True,
        'benchmarks_included': False, 'old_reservoir_outcomes_used': False,
        'reference_curve_source': CURVES.as_posix(),
        'reference_curve_source_contact_angle_deg': 70,
        'curve_source_note': 'Source/provenance only. The 40-degree solver inputs are NOT generated here.',
        'clay_pc_application': 'same selected atlas sample for host clay and fault-clay master',
        'pc_kr_upscaling_status': 'not_performed', 'reservoir_simulation_status': 'not_started',
        'source_predict_code_commit': old_config['provenance']['predict_code_commit'],
        'source_predict_method_config_hash': old_config['provenance']['predict_method_config_hash'],
        'source_sampling_config': (OLD / 'config/effective_sampling_config.toml').as_posix(),
        'implementation_sha256': sha(__file__),
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (out / 'manifests').mkdir(parents=True)
    (ROOT / 'validation').mkdir()
    save_json(ROOT / 'CONFIG.json', config)
    save_json(ROOT / 'validation/source_hashes_before.json', protected_before)
    save_csv(out / 'base_geologies.csv', geos)
    cases, scenarios, catalogs, streams, qcs = [], [], [], [], []
    seen_seeds, seen_field_hashes = set(), set()
    all_source = np.empty((2592, 6, NSLICE), dtype=np.int32)
    all_perms = np.empty((2592, 6, NSLICE, 3), dtype=np.float64)
    all_replay = np.empty((2592, 6, NSLICE), dtype=np.int64)
    all_ids = np.empty_like(all_replay)
    all_streams = np.empty((2592, 6), dtype=np.int64)
    nrows = 0
    with gzip_csv(out / 'sampling_manifest_all_scenarios.csv.gz') as combined:
        for gi, geo in enumerate(geos):
            gid = geo['geology_id']
            cat = read_csv(OLD / 'libraries' / f'{gid}_library_catalog.csv')
            by_window = {r['window_id']: r for r in cat}
            if len(cat) != 6 or set(by_window) != set(WINDOWS):
                raise ValueError(f'Incomplete window catalog: {gid}')
            libraries = {}
            for window in WINDOWS:
                row = by_window[window]
                if row['geology_id'] != gid:
                    raise ValueError('Library/geology mismatch')
                k, seeds, ids, state_sha = load_library(row, geo, frozen)
                libraries[window] = (k, seeds, ids)
                catalogs.append({'library_id': f'{gid}_{window}', 'geology_id': gid,
                                 'window_id': window, 'n_accepted': NPOOL,
                                 'checkpoint_path': row['checkpoint_path'],
                                 'checkpoint_relative_path': row['checkpoint_relative_path'],
                                 'checkpoint_hash': row['checkpoint_hash'],
                                 'level2_state_path': row['level2_state_path'],
                                 'level2_state_sha256': state_sha})
            for sample in SAMPLES:
                sid = f'{gid}_pc{sample:02d}_ca40'
                scenarios.append({**geo, 'scenario_id': sid, 'clay_pc_sample_id': sample,
                                  'co2_brine_contact_angle_deg': 40, 'n_stochastic': NREP,
                                  'sampling_design': 'independent_full_fault',
                                  'paired_pc_sampling': False,
                                  'manifest_relative_path': f'manifests/{sid}.csv.gz'})
                with gzip_csv(out / 'manifests' / f'{sid}.csv.gz') as manifest:
                    for rep in range(1, NREP + 1):
                        ci = len(cases)
                        cid = f'{sid}_IND_{rep:03d}'
                        for wi, window in enumerate(WINDOWS):
                            k, seeds, ids = libraries[window]
                            meta = by_window[window]
                            seed = stable_seed(BASE_SEED, sid, rep, window)
                            if seed in seen_seeds:
                                raise ValueError('Sampling stream collision; do not silently re-seed')
                            seen_seeds.add(seed)
                            rows = np.array([uniform_index(seed, j) for j in range(1, NSLICE + 1)], dtype=np.int32)
                            pick = rows - 1
                            all_source[ci, wi], all_perms[ci, wi] = rows, k[pick]
                            all_replay[ci, wi], all_ids[ci, wi] = seeds[pick], ids[pick]
                            all_streams[ci, wi] = seed
                            streams.append({'scenario_id': sid, 'case_id': cid, 'replicate_id': rep,
                                            'window_id': window, 'sampling_seed': seed,
                                            'n_accepted': NPOOL,
                                            'source_rows_sha256': hashlib.sha256(','.join(map(str, rows)).encode()).hexdigest()})
                            for j, ix in enumerate(pick, 1):
                                values = k[ix]
                                row = dict(zip(FIELDS, [
                                    'full_fault_sampling_manifest_pc_v1', DESIGN, gid, sid, sample, 40,
                                    'phase1', cid, 'independent_full', rep, window, j, f'{gid}_{window}',
                                    'full_distribution', int(ix + 1), int(ids[ix]), int(seeds[ix]), int(seeds[ix]),
                                    seed, VERSION, 'uniform_with_replacement', meta['checkpoint_path'],
                                    meta['checkpoint_relative_path'], meta['checkpoint_hash'], meta['level2_state_path'],
                                    config_hash, config['source_predict_code_commit'], config['source_predict_method_config_hash'],
                                    *map(float, np.log10(values)), *map(float, values), 'true', 'false', 'false']))
                                if len(row) != len(FIELDS):
                                    raise ValueError('Manifest schema mismatch')
                                manifest.writerow(row)
                                combined.writerow(row)
                                nrows += 1
                        field_hash = hashlib.sha256(all_source[ci].astype('>i4').tobytes()).hexdigest()
                        if (gid, field_hash) in seen_field_hashes:
                            raise ValueError('Duplicate full-fault draw; retain evidence rather than silently redraw')
                        seen_field_hashes.add((gid, field_hash))
                        cases.append({'case_id': cid, 'scenario_id': sid, 'geology_id': gid,
                                      'clay_pc_sample_id': sample, 'co2_brine_contact_angle_deg': 40,
                                      'replicate_id': rep, 'case_type': 'independent_full',
                                      'use_for_probabilistic_uq': True, 'is_benchmark': False,
                                      'n_assignments': 522, 'array_index_zero_based': ci,
                                      'source_assignment_sha256': field_hash,
                                      'pc_kr_status': 'not_generated_for_new_physics',
                                      'reservoir_status': 'not_started'})
            start = gi * 16
            a, b = all_source[start:start+8], all_source[start+8:start+16]
            qcs.append({'geology_id': gid, 'matched_index_cross_pc_equal_positions': int((a == b).sum()),
                        'compared_positions': int(a.size), 'exact_cross_pc_field_matches':
                        sum(np.array_equal(x, y) for x in a for y in b)})
            if (gi + 1) % 18 == 0:
                print(f'Generated {gi+1}/162 geologies; {len(cases)} cases', flush=True)
    if (len(scenarios), len(cases), nrows, len(seen_seeds)) != (324, 2592, 1353024, 15552):
        raise ValueError('Incorrect final dimensions')
    save_csv(out / 'scenario_catalog.csv', scenarios)
    save_csv(out / 'case_catalog.csv', cases)
    save_csv(out / 'library_catalog.csv', catalogs)
    save_csv(out / 'sampling_streams.csv', streams)
    save_csv(ROOT / 'validation/cross_pc_sampling_check.csv', qcs)
    with h5py.File(out / 'sampled_fields.h5', 'x') as f:
        f.attrs['design_version'] = DESIGN
        f.attrs['sampling_config_sha256'] = config_hash
        f.attrs['array_axes'] = 'case,window,slice,component (component axis only for permeability)'
        f.attrs['window_order'] = ','.join(WINDOWS)
        f.attrs['component_order'] = 'kxx,kyy,kzz'
        f.attrs['permeability_units'] = 'mD'
        f.attrs['source_library_row_index_base'] = 1
        f.attrs['physics_status'] = 'Sampling only; new Pc/Kr and reservoir inputs not generated'
        for name, data in [('source_library_row', all_source), ('predict_seed', all_replay),
                           ('predict_realization_id', all_ids), ('permeability_md', all_perms),
                           ('sampling_seed', all_streams)]:
            f.create_dataset(name, data=data, compression='gzip', shuffle=True)
        f.create_dataset('case_id', data=np.asarray([c['case_id'] for c in cases], dtype=h5py.string_dtype()))
    protected_after = {p.as_posix(): sha(p) for p in protected}
    if protected_before != protected_after:
        raise ValueError('An existing protected input changed during sampling')
    save_json(ROOT / 'validation/source_hashes_after.json', protected_after)
    save_json(ROOT / 'GENERATION.json', {
        'status': 'generated_awaiting_independent_validation', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': config_hash, 'geologies': 162, 'scenarios': 324, 'cases': 2592,
        'assignments': nrows, 'independent_window_streams': len(seen_seeds),
        'all_972_predict_hashes_match_frozen_inventory': True,
        'all_972_joint_permeability_libraries_match_level2': True,
        'protected_existing_files_unchanged': True, 'protected_file_count': len(protected),
        'existing_simulation_results_opened_or_modified': False,
        'python': sys.version, 'platform': platform.platform(),
        'numpy': np.__version__, 'h5py': h5py.__version__,
    })
    print(f'Complete: {nrows:,} assignments in {out}', flush=True)


if __name__ == '__main__':
    main()
