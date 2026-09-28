"""Read-only production inputs; paired full-fault candidate study on Engaging."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import tomllib

SOURCE = Path(__file__).resolve().parent
BASE = Path('/orcd/data/juanes/001/shaowen/predict_shaowen')
CANDIDATES = BASE/'candidates/cell_pc_mip_kr_v1_20260908'
OUT = CANDIDATES/'comparative_3case_wrr30_60_pc_conventions_20260910_v1'
FROZEN = CANDIDATES/'releases/fullfault_julia_1bc6badb9d4d'
FROZEN_HASH = '1bc6badb9d4de2817907a411c7bff6132727f2c870e39df3e1de7d6fad84bb69'
P30 = CANDIDATES/'full_fault_julia_s03_c022_case12_20260909_v1'
P60 = CANDIDATES/'full_fault_wrr60_s03_c022_case12_20260910_v1'
PROD = BASE/'production_runs/independent_full_fault_v1_phase1_20260809_v1'
PHYSICS = BASE/'orchestration/independent_full_fault_v1_04927f5/physics_repo'
PHYSICS_COMMIT = '68351e35f3679317b35532a9ca0533674e0aafb5'
DATA = BASE/'production_freezes/collapsed_cell_union_20260722_v7/inputs/predict'
MRST = Path('/orcd/pool/006/shaowen/predict_shaowen/software/mrst-2025a/SINTEF-AppliedCompSci-MRST-75749fa')
BRIDGE = P30/'bridge/libphase_bridge.so'
GOM = Path('/orcd/data/juanes/001/shaowen/gom_full_production')
CAMPAIGN_NAME = 'step62_phase1_2430_independent_full_fault_v1_effective_globalplateau_qoi4_coordfix_pressurefix_20260814'
CAMPAIGN = GOM/'campaign_manifests'/CAMPAIGN_NAME/'campaign.toml'
CAMPAIGN_HASH = 'adf73c15b1c4edc4b4721c7666a2c2b56aaf800088fff3599a5541dce0ed3dde'
CASES = [('low', 's01_c018_case07', 262), ('intermediate', 's03_c022_case12', 1137),
         ('high', 's03_c003_case08', 848)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def check(root, expected):
    for name, value in expected.items():
        assert sha(Path(root)/name) == value, ('Checksum changed', name)


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def coverage(values, geology, realization):
    assert len(values) == 522
    assert all(x['geology_id'] == geology and int(x['case_id']) == realization for x in values)
    assert {(x['window'], int(x['slice_index'])) for x in values} == {
        (f'famp{w}', s) for w in range(1, 7) for s in range(1, 88)}
    return sorted(values, key=lambda x: (int(x['window'][4:]), int(x['slice_index'])))


def variants():
    return [dict(risk=risk, case=key, task=task, angle=angle, pc_convention=pc,
                 variant=f'{key}_wrr{angle}_{pc}', production_promotion=False)
            for risk, key, task in CASES for angle in (30, 60)
            for pc in ('physical_prebreakthrough', 'zero_before_breakthrough')]


def seal():
    files = hashes(SOURCE)
    assert 'source_lock.json' not in files
    write(SOURCE/'source_lock.json', dict(files=files, sha256=digest(files)))


def verify():
    lock = read(SOURCE/'source_lock.json')
    assert digest(lock['files']) == lock['sha256']
    check(SOURCE, lock['files'])
    frozen = read(FROZEN/'source_lock.json')
    assert frozen['sha256'] == FROZEN_HASH
    check(FROZEN, frozen['files'])
    assert sha(SOURCE/'scenario.jl') == sha(P60/'source/scenario.jl')
    check(OUT/'inputs', read(OUT/'inputs_lock.json'))
    m = read(OUT/'inputs/manifest.json')
    assert m['source_sha256'] == lock['sha256']
    assert sha(BRIDGE) == m['bridge_sha256']
    assert sha(P30/'external_inventory.json') == m['external_inventory_sha256']
    os.environ['FROZEN_SOURCE'] = str(FROZEN)
    return m


def stage():
    assert not OUT.exists()
    lock = read(SOURCE/'source_lock.json')
    check(SOURCE, lock['files'])
    assert sha(CAMPAIGN) == CAMPAIGN_HASH
    campaign = tomllib.loads(CAMPAIGN.read_text())
    selected = {x['case_key']: x for x in campaign['cases']}
    OUT.mkdir()
    for name in ('logs', 'tmp', 'inputs'):
        (OUT/name).mkdir()
    sources = {str(CAMPAIGN): CAMPAIGN_HASH, campaign['common']['path']: campaign['common']['sha256']}
    cases = {}
    for risk, key, task in CASES:
        c = selected[key]
        assert c['task'] == task
        geology, realization = c['geology_id'], c['realization_id']
        values = coverage([x for x in rows(PROD/f'case_work_manifest/geology_assignments/{geology}.csv')
                           if int(x['case_id']) == realization], geology, realization)
        cin = PROD/f'case_inputs/cases/{geology}/case{realization:02d}'
        cres = PROD/f'case_results/cases/{geology}/case{realization:02d}'
        a, b = read(cin/'case_inputs.done.json'), read(cres/'case.done.json')
        assert a['status'] == 'assembled' and b['status'] == 'complete'
        assert b['pc_assignment_count'] == 522 and b['dynamic_kr_representative_count'] == 6
        assert b['physics_commit'] == PHYSICS_COMMIT
        summary = cin/f'inputs/pc_curve_summary_{key}_ip_full87.csv'
        native = cin/f'inputs/pc_native_curve_points_{key}_ip_full87.csv'
        baseline = cres/f'kr/reservoir_ready/fault_properties_{key}_pc_full_slice.mat'
        for p in (summary, native):
            assert sha(p) == a['files'][p.name]['sha256']
            sources[str(p)] = a['files'][p.name]['sha256']
        sources[str(baseline)] = b['files'][baseline.relative_to(cres).as_posix()]['sha256']
        for p in (cin/'case_inputs.done.json', cres/'case.done.json',
                  PROD/f'case_work_manifest/geology_assignments/{geology}.csv'):
            sources[str(p)] = sha(p)
        sources[c['specific_path']] = c['specific_sha256']
        summaries = {(x['Window'], int(x['SliceIndex'])): x for x in rows(summary)}
        pressure = {}
        for x in rows(native):
            pressure.setdefault((x['Window'], int(x['SliceIndex'])), []).append(float(x['PcPa']))
        records = []
        csvwrite(OUT/f'inputs/{key}/assignments.csv', values)
        for slot, x in enumerate(values, 1):
            y = summaries[(x['window'], int(x['slice_index']))]
            assert x['task_id'] == y['TaskId']
            assert int(x['selected_sample_index']) == int(y['SelectedSampleIndex'])
            assert int(x['exact_replay_seed']) == int(y['ReplaySeed'])
            record = dict(slot=slot, assignment=x, production_summary=y,
                          checkpoint=str(DATA/x['checkpoint_relative_path']), checkpoint_sha256=x['checkpoint_sha256'])
            record['calculation_key'] = digest(dict(frozen=FROZEN_HASH, record=record))
            records.append(record)
            folder = OUT/f'inputs/{key}/slot{slot:03d}'
            csvwrite(folder/'selection.csv', [x])
            write(folder/'summary.json', dict(slot=slot, NumCells=int(y['NumCells']),
                  PoreVolume=float(y['PoreVolume']), BulkVolume=float(y['BulkVolume']),
                  production_pressure_pa=sorted(set(pressure[(x['window'], int(x['slice_index']))]))))
        pilots = [(w-1)*87 + int(a['swi_medoid_slices'][f'famp{w}']) for w in range(1, 7)]
        pilots += [min(range(1, 523), key=lambda i: float(values[i-1]['log_kzz'])),
                   max(range(1, 523), key=lambda i: float(values[i-1]['log_kzz']))]
        pilots = sorted(set(pilots))
        remaining = [i for i in range(1, 523) if i not in pilots]
        original = GOM/'campaigns'/CAMPAIGN_NAME/f'tasksets/taskset_0001/payload/cases/{key}/production'
        assert original.is_dir(), original
        cases[key] = dict(risk=risk, production=c, baseline_mat=str(baseline), records=records,
                          pilots=pilots, main_batches=[remaining[i:i+4] for i in range(0, len(remaining), 4)],
                          original_reservoir_output=str(original), production_method_sha256=b['method_config_sha256'])
    matrix = variants()
    for v in matrix:
        if v['case'] == 's03_c022_case12' and v['angle'] == 30:
            physical = v['pc_convention'] == 'physical_prebreakthrough'
            v['reuse_reservoir_root'] = str(CANDIDATES/(('reservoir_comparison_' if physical else 'reservoir_pc_breakthrough_') + 's03_c022_case12_20260910_v1'))
            v['existing_job'] = '22438751_1137' if physical else '22474505_1137'
            v['launch_policy'] = 'reuse_existing_do_not_duplicate'
        else:
            v['launch_policy'] = 'after_property_review_and_matched_native_preflight'
    reused = {}
    for angle, root in ((30, P30), (60, P60)):
        accepted = {p.name: sha(p) for p in sorted((root/'accepted').glob('slot*.json'))}
        assert len(accepted) == 522
        reused[str(angle)] = dict(root=str(root), accepted=accepted)
    for root, names in ((P30, ['inputs/manifest.json', 'inputs_lock.json', 'initialized.json',
                              'study_complete.json', 'recovery_flowfix1/qualified.json']),
                        (P60, ['inputs.json', 'inputs_lock.json', 'source_lock.json',
                              'frozen_source_lock.json', 'qualification_pass.json'])):
        for name in names:
            sources[str(root/name)] = sha(root/name)
    write(OUT/'inputs/manifest.json', dict(source_sha256=lock['sha256'], cases=cases, variants=matrix,
          sources=sources, common=campaign['common'], reused_intermediate=reused,
          bridge_sha256=read(P30/'initialized.json')['bridge']['libphase_bridge.so'],
          external_inventory_sha256=sha(P30/'external_inventory.json'), frozen_source_sha256=FROZEN_HASH,
          production_read_only=True, production_promotion=False, reuse_policy='exact_identity_and_verified_hashes_only'))
    write(OUT/'inputs_lock.json', hashes(OUT/'inputs'))
    write(OUT/'study_registry.json', dict(variants=matrix, original_baselines=3,
          unique_full_fault_upscalings=6, cores_per_full_fault=522, reservoir_inputs_expected=12,
          new_reservoir_jobs_submitted=0, stage='awaiting_qualification'))
    print('STAGED', OUT, '12 variants; 6 datasets; 2 existing reservoir jobs preserved', flush=True)


def command(folder, args, timeout=80000):
    folder.mkdir(parents=True)
    start = time.monotonic()
    with (folder/'process.log').open('x') as f:
        p = subprocess.run(['/usr/bin/time', '-v', '-o', str(folder/'resources.txt')]+args,
                           cwd=folder, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    write(folder/'receipt.json', dict(args=args, returncode=p.returncode, seconds=time.monotonic()-start,
          node=socket.gethostname(), job=os.environ['SLURM_JOB_ID']))
    assert p.returncode == 0, ('Process failed', str(folder), p.returncode)


def jq(mode, raw, geometry, out):
    return ['julia', '--startup-file=no', '--math-mode=ieee', '--check-bounds=yes',
            str(SOURCE/'scenario.jl'), mode, str(raw), str(geometry), str(BRIDGE), str(out)]


def q(value):
    return "'" + str(value).replace("'", "''") + "'"


def load_auditor():
    spec = importlib.util.spec_from_file_location('frozen_audit', FROZEN/'export_audit.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_accepted(path, key=None):
    d = read(path)
    assert d['numerical_checks_passed']
    if key is not None:
        assert d['calculation_key'] == key
    check(d['attempt'], d['outputs'])
    return d


def assembly(out, records, case, angle, baseline):
    assert len(records) == 522 and [r['slot'] for r in records] == list(range(1, 523))
    out.mkdir(parents=True)
    manifest = out/'manifest.json'
    write(manifest, dict(case=case, angle=angle, records=records, baseline_mat=baseline,
                        source=read(SOURCE/'source_lock.json')['sha256']))
    expression = f'addpath({q(SOURCE)});assemble_case({q(manifest)},{q(out/"raw_property")});'
    # Pure data assembly does not require JVM; replay retains JVM for SHA-256.
    command(out/'assembly_process', ['matlab', '-nojvm', '-nodisplay', '-nosplash', '-batch', expression])
    assert read(out/'raw_property/roundtrip.json')['passed']
    write(out/'complete.json', dict(passed=True, files=hashes(out/'raw_property'), reservoir_export_approved=False))


def qualify(m):
    attempt = OUT/'qualification_attempts'/f'job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    attempt.mkdir(parents=True)
    check('/', m['sources'])
    assert shutil.disk_usage(OUT).free > 150*1024**3
    assert subprocess.check_output(['git', '-C', str(PHYSICS), 'rev-parse', 'HEAD'], text=True).strip() == PHYSICS_COMMIT
    for angle, root in ((30, P30), (60, P60)):
        check(root/'accepted', m['reused_intermediate'][str(angle)]['accepted'])
        records = []
        parent = read(P30/'inputs/manifest.json')['records']
        for r in parent:
            slot = r['slot']
            d = verify_accepted(root/f'accepted/slot{slot:03d}.json', r['calculation_key'] if angle == 30 else None)
            if angle == 60:
                assert d['parent_accepted_sha256'] == m['reused_intermediate']['30']['accepted'][f'slot{slot:03d}.json']
                assert d['input_manifest_sha256'] == m['sources'][str(P60/'inputs.json')]
            current = m['cases']['s03_c022_case12']['records'][slot-1]
            assert r['assignment'] == current['assignment']
            a = r['assignment']
            records.append(dict(slot=slot, window=int(a['window'][4:]), slice=int(a['slice_index']),
                  sample=int(a['selected_sample_index']), seed=int(a['exact_replay_seed']),
                  curve=str(Path(d['attempt'])/'numerical/coupled/pc_kr_raw.csv'),
                  audit=str(Path(d['attempt'])/'export_audit.json')))
        write(attempt/f'reuse_wrr{angle}.json', dict(passed=True, verified_results=522, root=str(root)))
        if angle == 60:
            assembly(attempt/'intermediate_wrr60_assembly', records, 's03_c022_case12', 60,
                     m['cases']['s03_c022_case12']['baseline_mat'])
    command(attempt/'unit_tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(SOURCE), '-p', 'test_*.py'])
    check('/', m['sources'])
    write(OUT/'qualification_pass.json', dict(passed=True, attempt=str(attempt),
          files=hashes(attempt), source=read(SOURCE/'source_lock.json')['sha256'],
          existing_runs_untouched=True, no_production_writes=True))
    print('QUALIFIED; existing 1044 core results verified; WRR60 assembly recovered', flush=True)


def acceptance_path(case, angle, slot):
    return OUT/f'datasets/{case}/wrr{angle}/accepted/slot{slot:03d}.json'


def run_core(m, case, slot):
    assert case in ('s01_c018_case07', 's03_c003_case08')
    r = m['cases'][case]['records'][slot-1]
    assert r['slot'] == slot
    done = {}
    for angle in (30, 60):
        p = acceptance_path(case, angle, slot)
        if p.exists():
            done[angle] = verify_accepted(p, r['calculation_key'])
    if len(done) == 2:
        print('VERIFIED_SKIP', case, slot, flush=True)
        return
    lock = OUT/f'locks/{case}_slot{slot:03d}'
    lock.parent.mkdir(exist_ok=True)
    lock.mkdir()
    attempt = OUT/f'attempts/{case}/slot{slot:03d}_job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    attempt.mkdir(parents=True)
    write(attempt/'provenance.json', dict(record=r, source=read(SOURCE/'source_lock.json')['sha256'],
          node=socket.gethostname(), job=os.environ['SLURM_JOB_ID'], production_promotion=False))
    try:
        assert sha(r['checkpoint']) == r['checkpoint_sha256']
        if 30 in done:
            planned = Path(done[30]['attempt'])/'numerical/planned'
        else:
            ready = OUT/f'replay/{case}/slot{slot:03d}.json'
            if ready.exists():
                cached = read(ready)
                assert cached['calculation_key'] == r['calculation_key']
                check(cached['root'], cached['files'])
                prepared = Path(cached['root'])
            else:
                prepared = attempt/'prepared'
                folder = OUT/f'inputs/{case}/slot{slot:03d}'
                expr = f'addpath({q(FROZEN)});prepare_slot({q(folder/"selection.csv")},{q(folder/"summary.json")},{q(prepared)},{q(PHYSICS)},{q(DATA)},{q(MRST)});'
                command(attempt/'replay_process', ['matlab', '-nodisplay', '-nosplash', '-batch', expr])
                write(ready, dict(root=str(prepared), files=hashes(prepared), calculation_key=r['calculation_key']))
            inv = read(P30/'external_inventory.json')
            for name in (prepared/'loaded_matlab_files.txt').read_text().splitlines():
                if name.startswith((str(PHYSICS), str(MRST))):
                    assert name in inv and sha(name) == inv[name], ('External code changed', name)
            a30 = attempt/'wrr30'
            command(attempt/'run30_process', ['julia', '--startup-file=no', '--math-mode=ieee', '--check-bounds=yes',
                    str(FROZEN/'plan_and_run.jl'), str(prepared/'raw_unplanned.bin'),
                    str(prepared/'geometry_unplanned.bin'), str(BRIDGE), str(a30/'numerical'), 'false'])
            done[30] = accept(m, case, 30, r, a30)
            planned = a30/'numerical/planned'
        if 60 not in done:
            a60 = attempt/'wrr60'
            command(attempt/'run60_process', jq('run', planned/'raw.bin', planned/'geometry.bin', a60/'numerical'))
            done[60] = accept(m, case, 60, r, a60)
        before = tomllib.loads((Path(done[30]['attempt'])/'numerical/coupled/summary.toml').read_text())
        after = tomllib.loads((Path(done[60]['attempt'])/'numerical/coupled/summary.toml').read_text())
        assert abs(after['intrinsic_permeability_m2']/before['intrinsic_permeability_m2']-1) <= 1e-5
        assert sha(r['checkpoint']) == r['checkpoint_sha256']
        write(attempt/'paired_pass.json', dict(passed=True, angles=[30, 60], same_draw=True, same_pressure_nodes=True))
        print('PAIRED_COMPLETE', case, slot, flush=True)
    finally:
        lock.rmdir()


def accept(m, case, angle, r, a):
    coupled = a/'numerical/coupled'
    completion = tomllib.loads((coupled/'completion.toml').read_text())
    assert completion['execution_complete']
    check(coupled, completion['outputs'])
    summary = tomllib.loads((coupled/'summary.toml').read_text())
    with (coupled/'pc_kr_raw.csv').open() as f:
        values = list(csv.reader(f))[1:]
    audit = load_auditor().audit([[float(v) for v in row] for row in values])
    write(a/'export_audit.json', audit)
    assert summary['points'] == 100
    canonical = float(r['assignment']['perm_kzz'])*9.869233e-16
    logdiff = abs(math.log10(summary['intrinsic_permeability_m2']/canonical))
    assert logdiff <= .005, ('Intrinsic Kzz replay check', case, r['slot'], logdiff)
    if angle == 60:
        scenario = tomllib.loads((a/'numerical/planned/scenario_audit.toml').read_text())
        assert scenario['passed']
        d30 = verify_accepted(acceptance_path(case, 30, r['slot']), r['calculation_key'])
        root30 = Path(d30['attempt'])/'numerical'
        assert scenario['raw_parent_sha256'] == sha(root30/'planned/raw.bin')
        summary30 = tomllib.loads((root30/'coupled/summary.toml').read_text())
        assert abs(summary['intrinsic_permeability_m2']/summary30['intrinsic_permeability_m2']-1) <= 1e-5
    d = dict(slot=r['slot'], case=case, angle=angle, attempt=str(a), numerical_checks_passed=True,
             calculation_key=r['calculation_key'], source=read(SOURCE/'source_lock.json')['sha256'],
             input_manifest_sha256=sha(OUT/'inputs/manifest.json'), canonical_kzz_log10_difference=logdiff,
             export_flags=audit['flags'], reservoir_export_approved=False, production_promotion=False,
             outputs=hashes(a))
    write(acceptance_path(case, angle, r['slot']), d)
    return d


def gate(m, case):
    good = []
    for slot in m['cases'][case]['pilots']:
        r = m['cases'][case]['records'][slot-1]
        for angle in (30, 60):
            good.append(verify_accepted(acceptance_path(case, angle, slot), r['calculation_key']))
    largest = max(sum(p.stat().st_size for p in Path(d['attempt']).rglob('*') if p.is_file()) for d in good)
    estimate = largest*2*522*1.5
    assert shutil.disk_usage(OUT).free > estimate + 100*1024**3
    write(OUT/f'gates/{case}.json', dict(passed=True, paired_pilot_cores=len(good)//2,
          estimated_remaining_bytes=estimate, flags_are_not_failures=True, reservoir_export_approved=False))


def finish(m, case):
    missing = {str(a): [i for i in range(1, 523) if not acceptance_path(case, a, i).exists()] for a in (30, 60)}
    if any(missing.values()):
        write(OUT/f'reports/{case}_incomplete_{os.environ["SLURM_JOB_ID"]}.json', dict(missing=missing))
        print('INCOMPLETE; preserve successes and retry only listed cores', missing, flush=True)
        return
    for angle in (30, 60):
        records = []
        for r in m['cases'][case]['records']:
            d = verify_accepted(acceptance_path(case, angle, r['slot']), r['calculation_key'])
            x = r['assignment']
            records.append(dict(slot=r['slot'], window=int(x['window'][4:]), slice=int(x['slice_index']),
                  sample=int(x['selected_sample_index']), seed=int(x['exact_replay_seed']),
                  curve=str(Path(d['attempt'])/'numerical/coupled/pc_kr_raw.csv'),
                  audit=str(Path(d['attempt'])/'export_audit.json')))
        assembly(OUT/f'assemblies/{case}/wrr{angle}', records, case, angle, m['cases'][case]['baseline_mat'])
    check('/', m['sources'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['seal', 'stage', 'qualify', 'pilot', 'main', 'gate', 'finish', 'status'])
    p.add_argument('case', nargs='?')
    p.add_argument('index', nargs='?', type=int)
    a = p.parse_args()
    if a.mode == 'seal':
        seal()
        return
    if a.mode == 'stage':
        stage()
        return
    if a.mode == 'status':
        for _, case, _ in CASES:
            print(case, {angle: len(list((OUT/f'datasets/{case}/wrr{angle}/accepted').glob('slot*.json'))) for angle in (30, 60)})
        return
    assert os.environ.get('SLURM_JOB_ID'), 'Compute-node execution required'
    assert subprocess.check_output(['julia', '--version'], text=True).strip() == 'julia version 1.12.6'
    m = verify()
    if a.mode == 'qualify':
        qualify(m)
        return
    assert read(OUT/'qualification_pass.json')['passed']
    if a.mode == 'gate':
        gate(m, a.case)
    elif a.mode == 'finish':
        finish(m, a.case)
    else:
        if a.mode == 'main':
            assert read(OUT/f'gates/{a.case}.json')['passed']
            slots = m['cases'][a.case]['main_batches'][a.index-1]
        else:
            slots = [m['cases'][a.case]['pilots'][a.index-1]]
        failures = []
        for slot in slots:
            try:
                run_core(m, a.case, slot)
            except Exception as e:
                failures.append(dict(slot=slot, error=repr(e)))
                print('FAILED', a.case, slot, repr(e), flush=True)
        if failures:
            write(OUT/f'failures/{a.case}_{a.mode}_{a.index}_{os.environ["SLURM_JOB_ID"]}.json', failures)
            sys.exit(1)


if __name__ == '__main__':
    main()
