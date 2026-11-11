"""Audit the complete saved sampling package against read-only source libraries."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'phase1'
OLD = Path('D:/codex_gom/UQ_workflow/independent_full_fault_v1_phase1_20260809_v1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    config = json.loads((ROOT / 'CONFIG.json').read_text())
    cfg_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    generation = json.loads((ROOT / 'GENERATION.json').read_text())
    require(cfg_sha == generation['config_sha256'], 'Configuration hash changed')
    require(digest(ROOT / 'code/generate_sampling.py') == config['implementation_sha256'], 'Generator changed')
    scenarios = pd.read_csv(OUT / 'scenario_catalog.csv')
    cases = pd.read_csv(OUT / 'case_catalog.csv')
    libs = pd.read_csv(OUT / 'library_catalog.csv')
    streams = pd.read_csv(OUT / 'sampling_streams.csv', dtype={'sampling_seed': 'int64'})
    require(len(scenarios) == 324 and scenarios.scenario_id.nunique() == 324, 'Scenario count')
    require(scenarios.geology_id.nunique() == 162, 'Base-geology count')
    require(set(scenarios.clay_pc_sample_id) == {4, 6}, 'Pc choices')
    require(scenarios.groupby('geology_id').clay_pc_sample_id.nunique().eq(2).all(), 'Incomplete crossing')
    require(scenarios.co2_brine_contact_angle_deg.eq(40).all(), 'Angle label')
    require(not scenarios.paired_pc_sampling.any(), 'Unexpected paired sampling')
    require(len(cases) == 2592 and cases.case_id.nunique() == 2592, 'Case count')
    require(cases.groupby('scenario_id').size().eq(8).all(), 'Cases per scenario')
    require(cases.groupby('scenario_id').replicate_id.apply(lambda x: set(x) == set(range(1, 9))).all(), 'Replicate IDs')
    require(cases.use_for_probabilistic_uq.all() and not cases.is_benchmark.any(), 'Non-stochastic case present')
    require(np.array_equal(cases.array_index_zero_based, np.arange(2592)), 'Case/array mapping')
    require(len(libs) == 972 and libs.library_id.nunique() == 972, 'Library count')
    require(len(streams) == 15552 and streams.sampling_seed.nunique() == 15552, 'Stream count/collision')
    with h5py.File(OUT / 'sampled_fields.h5', 'r') as f:
        require(list(f['case_id'].asstr()[()]) == list(cases.case_id), 'HDF5 case identities')
        require(f.attrs['sampling_config_sha256'] == cfg_sha, 'HDF5 configuration hash')
        source = f['source_library_row'][()]
        seeds = f['predict_seed'][()]
        ids = f['predict_realization_id'][()]
        perms = f['permeability_md'][()]
        random_seeds = f['sampling_seed'][()]
    require(source.shape == (2592, 6, 87) and perms.shape == (2592, 6, 87, 3), 'Array shapes')
    require(np.isfinite(perms).all() and (perms > 0).all(), 'Invalid permeability')
    require(source.min() >= 1 and source.max() <= 2000, 'Library rows out of bounds')
    require(np.array_equal(random_seeds.ravel(), streams.sampling_seed.to_numpy()), 'Stream catalog mapping')
    window_index = {f'famp{i}': i - 1 for i in range(1, 7)}
    for j, r in enumerate(libs.itertuples(index=False)):
        require(digest(r.checkpoint_path) == r.checkpoint_hash, f'Source changed: {r.library_id}')
        require(digest(r.level2_state_path) == r.level2_state_sha256, f'Level2 changed: {r.library_id}')
        ci = np.flatnonzero(cases.geology_id.to_numpy() == r.geology_id)
        wi = window_index[r.window_id]
        with h5py.File(r.checkpoint_path, 'r') as f:
            original_k = f['perms'][()].T
            original_seed = f['meta/AcceptedSeeds'][()].reshape(-1).astype(np.int64)
            original_id = f['meta/AcceptedAttemptIndices'][()].reshape(-1).astype(np.int64)
        pick = source[ci, wi] - 1
        require(np.array_equal(perms[ci, wi], original_k[pick]), f'Joint permeability changed: {r.library_id}')
        require(np.array_equal(seeds[ci, wi], original_seed[pick]), f'Wrong replay seeds: {r.library_id}')
        require(np.array_equal(ids[ci, wi], original_id[pick]), f'Wrong realization IDs: {r.library_id}')
        if (j + 1) % 324 == 0:
            print(f'Validated source arrays for {j+1}/972 libraries', flush=True)

    expected_combined = hashlib.sha256()
    total_rows = 0
    for si, r in enumerate(scenarios.itertuples(index=False)):
        path = OUT / r.manifest_relative_path
        ci = np.flatnonzero(cases.scenario_id.to_numpy() == r.scenario_id)
        df = pd.read_csv(path, float_precision='round_trip', dtype={'sampling_seed': 'int64', 'predict_seed': 'int64', 'exact_replay_seed': 'int64'})
        require(len(df) == 4176, f'Manifest count: {r.scenario_id}')
        require(df.geology_id.eq(r.geology_id).all() and df.scenario_id.eq(r.scenario_id).all(), 'Manifest identity')
        require(df.clay_pc_sample_id.eq(r.clay_pc_sample_id).all() and df.co2_brine_contact_angle_deg.eq(40).all(), 'Manifest Pc identity')
        require(df.sampling_config_sha256.eq(cfg_sha).all(), 'Manifest configuration hash')
        require(df.use_for_probabilistic_uq.all() and not df.is_benchmark.any() and not df.is_stress_test.any(), 'Case flags')
        require(df.case_id.tolist() == np.repeat(cases.iloc[ci].case_id.to_numpy(), 522).tolist(), 'Manifest case order')
        require(np.array_equal(df.slice_id, np.tile(np.arange(1, 88), 48)), 'Slice order')
        require(df.window_id.tolist() == list(np.tile(np.repeat(list(window_index), 87), 8)), 'Window order')
        for name, expected in [('source_library_row', source[ci].ravel()), ('predict_seed', seeds[ci].ravel()),
                               ('exact_replay_seed', seeds[ci].ravel()), ('predict_realization_id', ids[ci].ravel()),
                               ('sampling_seed', np.repeat(random_seeds[ci].ravel(), 87))]:
            require(np.array_equal(df[name].to_numpy(), expected), f'Manifest/HDF5 mismatch: {name}')
        for component, axis in [('kxx', 0), ('kyy', 1), ('kzz', 2)]:
            require(np.allclose(df[f'perm_{component}_md'], perms[ci, :, :, axis].ravel(), rtol=2e-15, atol=0), 'CSV permeability roundtrip')
            require(np.allclose(df[f'log_{component}'], np.log10(perms[ci, :, :, axis]).ravel(), rtol=0, atol=3e-14), 'CSV log permeability')
        for wi, window in enumerate(window_index):
            lr = libs[(libs.geology_id == r.geology_id) & (libs.window_id == window)].iloc[0]
            block = df[df.window_id == window]
            for name in ['library_id', 'checkpoint_path', 'checkpoint_relative_path', 'checkpoint_hash', 'level2_state_path']:
                require(block[name].eq(lr[name]).all(), f'Manifest source pointer: {name}')
        with gzip.open(path, 'rb') as f:
            header = f.readline()
            if si == 0:
                expected_combined.update(header)
            for b in iter(lambda: f.read(1024 * 1024), b''):
                expected_combined.update(b)
        total_rows += len(df)
        if (si + 1) % 108 == 0:
            print(f'Validated {si+1}/324 scenario manifests', flush=True)
    actual_combined = hashlib.sha256()
    with gzip.open(OUT / 'sampling_manifest_all_scenarios.csv.gz', 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            actual_combined.update(b)
    require(actual_combined.digest() == expected_combined.digest(), 'Combined/scenario manifests differ')
    require(total_rows == 1353024, 'Total assignment count')

    old_seeds, old_matches, old_hashes = set(), 0, {}
    for gid in cases.geology_id.unique():
        path = OLD / 'phase1/manifests' / f'{gid}_sampling_manifest.csv'
        old_hashes[path.as_posix()] = digest(path)
        old = pd.read_csv(path, usecols=['case_id', 'source_library_row', 'sampling_seed'], dtype={'sampling_seed': 'int64'})
        old_seeds.update(int(v) for v in old.sampling_seed.unique() if v != 0)
        signatures = {tuple(x.source_library_row) for _, x in old.groupby('case_id', sort=False)}
        for ci in np.flatnonzero(cases.geology_id.to_numpy() == gid):
            old_matches += tuple(source[ci].ravel()) in signatures
    require(not (set(random_seeds.ravel()) & old_seeds), 'New/old sampling stream collision')
    require(old_matches == 0, 'A new full-fault assignment is identical to an old assignment')
    cross_matches, exact_pairs = 0, 0
    for gid in cases.geology_id.unique():
        ci = np.flatnonzero(cases.geology_id.to_numpy() == gid)
        a, b = source[ci[:8]], source[ci[8:]]
        cross_matches += int((a == b).sum())
        exact_pairs += sum(np.array_equal(x, y) for x in a for y in b)
        require(len({x.tobytes() for x in source[ci]}) == 16, f'Duplicate new field: {gid}')
        for index in ci:
            h = hashlib.sha256(source[index].astype('>i4').tobytes()).hexdigest()
            require(h == cases.iloc[index].source_assignment_sha256, 'Case signature mismatch')
    require(exact_pairs == 0, 'Cross-Pc pairing found')

    tests = subprocess.run([sys.executable, '-B', str(ROOT / 'code/test_sampling.py')], capture_output=True, text=True, check=True)
    oracle = subprocess.run(['julia', '--startup-file=no', '--history-file=no', str(ROOT / 'code/verify_draws.jl')], capture_output=True, text=True, check=True)
    before = json.loads((ROOT / 'validation/source_hashes_before.json').read_text())
    after = json.loads((ROOT / 'validation/source_hashes_after.json').read_text())
    require(before == after, 'Protected old sources changed during generation')
    require(all(digest(p) == h for p, h in before.items()), 'Protected old source changed after generation')
    require(all(digest(p) == h for p, h in old_hashes.items()), 'Old per-geology manifest changed during validation')
    frequencies = np.bincount(source.ravel(), minlength=2001)[1:]
    chi = chisquare(frequencies)
    result = {
        'passed': True, 'scope': 'sampling_only', 'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_geologies': 162, 'scenarios': 324, 'stochastic_cases': 2592,
        'realizations_per_scenario': 8, 'assignments': total_rows, 'library_count': 972,
        'source_population_per_window': 2000, 'independent_window_streams': 15552,
        'all_assignments_match_exact_source_k_tuple_and_replay_seed': True,
        'all_source_predict_hashes_match_frozen_inventory': True,
        'all_scenario_and_combined_manifests_match': True,
        'cross_pc_pairing': False, 'cross_pc_exact_full_field_matches': int(exact_pairs),
        'new_vs_old_exact_full_field_matches': int(old_matches),
        'new_vs_old_stream_seed_collisions': 0,
        'cross_pc_same_position_coincidences': cross_matches,
        'cross_pc_same_position_trials': 162 * 8 * 522,
        'cross_pc_expected_coincidences_under_independence': 162 * 8 * 522 / 2000,
        'uniform_index_chisquare_statistic': float(chi.statistic),
        'uniform_index_chisquare_pvalue_diagnostic_only': float(chi.pvalue),
        'statistical_diagnostics_do_not_trigger_reseeding_or_filtering': True,
        'unit_test_output': tests.stdout + tests.stderr,
        'independent_original_julia_verification': oracle.stdout + oracle.stderr,
        'old_primary_manifest_and_protected_inputs_unchanged': True,
        'old_per_geology_manifest_hashes': old_hashes,
        'old_simulation_results_opened_or_modified': False,
        'pc_kr_upscaling_performed': False, 'reservoir_simulations_launched': False,
        'mc_interpretation': 'Independent draws conditional on the existing finite PREDICT libraries and sampling assumptions; no old QoIs used.',
    }
    with (ROOT / 'validation/VALIDATION.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    inventory = []
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name != 'OUTPUT_SHA256.csv' and '__pycache__' not in p.parts:
            inventory.append({'relative_path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)})
    with (ROOT / 'OUTPUT_SHA256.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'sha256'], lineterminator='\n')
        w.writeheader()
        w.writerows(inventory)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('hashes') and not k.endswith('output')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
