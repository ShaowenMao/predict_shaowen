"""Read-only, allowlisted source export; never traverses campaign results.

Run on the source host with an unused /tmp output filename. No sources are
modified. Payload data and installed runtimes remain external dependencies.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


BASE = Path('/orcd/data/juanes/001/shaowen/predict_shaowen')
CAND = BASE / 'candidates'
CAM = CAND / 'wrr60_pc100_kr10_campaign_20260916_v1'
MIP = CAND / 'cell_pc_mip_kr_v1_20260908'
PHASE = CAND / 'wrr60_phase1_production_candidate_20260912_v1'
OLD = BASE / 'campaigns/wrr60_phase1_2430_20260914_v1'
TEXT = {'.py', '.jl', '.m', '.cpp', '.c', '.h', '.hpp', '.sh', '.in',
        '.sbatch', '.toml', '.json', '.md', '.txt', '.csv', '.yml', '.yaml', '.env', '.sha256'}


def read(p):
    return json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert args.output.parent.resolve() == Path('/tmp') and not args.output.exists()
    manifest = read(CAM / 'qualification_v3/manifest.json')
    numerical = read(OLD / 'numerical_releases/6c240610c9f90e81/spec.json')
    launch = read(CAM / 'campaign_launch_v3/LAUNCH_LOCK.json')
    pins = dict(manifest['pins'])
    for key in ('external', 'worker_pins'):
        for p, digest in launch.get(key, {}).items():
            assert p not in pins or pins[p] == digest
            pins[p] = digest
    components = {}
    files = {}

    def add(label, root, mode='tree'):
        root = Path(root)
        assert root.exists(), str(root)
        components[label] = str(root)
        paths = [root] if root.is_file() else (root.iterdir() if mode == 'top' else root.rglob('*'))
        for p in sorted(paths):
            relative = Path(p.name) if root.is_file() else p.relative_to(root)
            if any(x in relative.parts for x in ('.git', '__pycache__', '.pytest_cache')):
                continue
            if not p.is_file():
                continue
            if mode == 'top' and p.suffix not in TEXT and p.name not in ('LICENSE', 'COPYING'):
                continue
            assert p.stat().st_size < 20 * 1024**2, ('Unexpected large source', str(p))
            data = p.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if str(p) in pins:
                assert digest == pins[str(p)], ('Pinned source changed', str(p))
            name = 'snapshot/' + label + '/' + relative.as_posix()
            assert name not in files
            files[name] = (data, dict(source=str(p), sha256=digest, bytes=len(data),
                                     existing_pin_verified=str(p) in pins))

    add('pipeline', manifest['pipeline'])
    add('backend', manifest['backend'], 'top')
    add('julia_environment', manifest['project'])
    add('kernel', CAM / 'qualification_v3', 'top')
    add('campaign', CAM / 'campaign_launch_v3', 'top')
    add('full_fault_export', CAM / 'full_fault_dispatch_v2', 'top')
    add('frozen_preparation', numerical['frozen'])
    add('scenario', numerical['scenario'])
    add('replay', OLD / 'releases/5748ac68fdc0be88')
    add('numerical_release', OLD / 'numerical_releases/6c240610c9f90e81', 'top')
    add('pc_onset', MIP / 'mip_pc_breakthrough_6case_20260916_v1', 'top')
    add('corrected_reservoir', MIP / 'corrected_hysteresis_15case_20260911_v1', 'top')
    add('corrected_reference', MIP / 'corrected_hysteresis_15case_20260911_v1/reference', 'top')
    runtime = MIP / 'corrected_hysteresis_15case_20260911_v1/runtime'
    add('reservoir_runtime', runtime, 'top')
    add('reservoir_runtime/src', runtime / 'src')
    add('bridge_reference', manifest['bridge'])
    if (CAM / 'qualification_v3/evidence').exists():
        add('qualification_evidence', CAM / 'qualification_v3/evidence', 'top')
    for name in ['comparative_three_cases_20260910_v1', 'mip_matrix_launch_20260911_v1',
                 'hybrid_low_high_wrr30_60_20260911_v1', 'hybrid_pc60_preflightfix_20260911_v1']:
        add('historical_releases/' + name, MIP / 'releases' / name)
    variant_index = 0
    for name in ['corrected_hysteresis_15case_20260911_v1', 'mip_pc_breakthrough_6case_20260916_v1']:
        for row in read(MIP / name / 'matrix.json'):
            variant_index += 1
            root = Path(row['root'])
            label = 'variants/c%02d' % variant_index
            add(label, root, 'top')
            add(label + '/source', root / 'source')
    for rel in ['full_fault_julia_s03_c022_case12_20260909_v1/bridge',
                'comparative_3case_wrr30_60_pc_conventions_20260910_v1/source',
                'comparative_3case_wrr30_60_pc_conventions_20260910_v1/reservoir_matrix_launch_20260911_v1']:
        p = MIP / rel
        if p.exists():
            add('historical/' + p.name, p, 'top')
    env = PHASE / 'reference_integration_v1/integration_v1/environments/export_py310_v1'
    for name in ['pip_freeze.txt', 'install_report.json', 'PASS.json', 'pyvenv.cfg']:
        add('python_environment/' + name, env / name)
    for label, p in [('original_inputs', BASE / 'production_freezes/collapsed_cell_union_20260722_v7'),
                     ('phase1_inputs', BASE / 'handoffs/independent_full_fault_v1_phase1_20260809_v1')]:
        for sub in ['freeze_metadata.json', 'handoff_metadata.json', 'handoff_files_sha256.csv',
                    'inventory/predict_files_sha256.csv', 'inventory/sampling_files_sha256.csv']:
            if (p / sub).exists():
                add(label + '/' + sub.replace('/', '_'), p / sub)
    archived_sources = {r['source'] for _, r in files.values()}
    for index, path in enumerate(sorted(pins)):
        p = Path(path)
        if (path not in archived_sources and BASE in p.parents
                and p.suffix in TEXT and 'site-packages' not in p.parts):
            add('pinned_evidence/p%03d' % index, p)
    assert sum(len(data) for data, _ in files.values()) < 150 * 1024**2
    record = dict(schema='mip_source_capsule_v1', components=components,
                  collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={n: r for n, (_, r) in sorted(files.items())},
                  external_pins={p: h for p, h in sorted(pins.items())
                                 if p not in {r['source'] for _, r in files.values()}},
                  originals_modified=False, production_jobs_modified=False)
    with tarfile.open(str(args.output), 'x:gz') as archive:
        for name, (data, _) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
        data = (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()
        info = tarfile.TarInfo('SOURCE_MANIFEST.json')
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps(dict(files=len(files), source_bytes=sum(len(d) for d, _ in files.values()),
                          archive=str(args.output), archive_bytes=args.output.stat().st_size,
                          archive_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
