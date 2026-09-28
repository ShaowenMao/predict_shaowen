"""Checksum-verified adapters around unchanged qualified scientific sources.

No Slurm submission, ledger mutation, source deletion or campaign discovery.
All writers require a new output directory. Run numerical stages on an
appropriately allocated compute node, never on an Engaging login node.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / 'snapshot'
MEMORY_CLASSES = {'mixed_60000': 2816, 'mixed_90000': 7040,
                  'mixed_120000': 13952, 'mixed_150000': 20096,
                  'clay_120000': 3712}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def member(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not name.startswith('snapshot/'):
        raise ValueError('Snapshot member escapes the package: ' + name)
    return path


def verify():
    manifest = read(ROOT / 'SOURCE_MANIFEST.json')
    for name, record in manifest['files'].items():
        path = member(name)
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Source checksum mismatch: ' + name)
    unexpected = [p for p in SNAP.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.relative_to(ROOT).as_posix() not in manifest['files']]
    if unexpected:
        raise ValueError('Unexpected files can change numerical source identity: ' + str(unexpected[:3]))
    return manifest


def input_file(record):
    path = Path(record['path']).resolve(strict=True)
    digest = record['sha256']
    if not re.fullmatch('[0-9a-f]{64}', digest) or sha(path) != digest:
        raise ValueError('Input checksum mismatch: ' + str(path))
    return path


def output_path(path):
    path = Path(path).resolve()
    if path == ROOT or path.is_relative_to(SNAP) or ROOT.is_relative_to(path):
        raise ValueError('Output must not replace package source')
    if path.exists():
        raise FileExistsError(path)
    return path


def fresh(path):
    path = output_path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def toml(path, top, cases=(), pins=None):
    text = ''.join(k + ' = ' + json.dumps(v) + '\n' for k, v in top.items())
    for case in cases:
        text += '\n[[cases]]\n' + ''.join(k + ' = ' + json.dumps(v) + '\n' for k, v in case.items())
    if pins is not None:
        text += '\n[pins]\n' + ''.join(json.dumps(str(k)) + ' = ' + json.dumps(v) + '\n' for k, v in pins.items())
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def julia_args(julia, script, config, prime=False):
    return [julia, '--startup-file=no', '--math-mode=ieee', '--check-bounds=yes',
            '--threads=2', '--gcthreads=1', '--cpu-target=generic',
            '--project=' + str(SNAP / 'julia_environment'),
            '--compiled-modules=' + ('yes' if prime else 'strict'),
            '--pkgimages=' + ('yes' if prime else 'existing'), str(script), str(config)]


def run(command, output, env=None):
    with output.open('x', encoding='utf-8') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                       env=env, check=True)


def prepare(specfile, out, julia):
    verify()
    spec = read(specfile)
    raw, geometry = input_file(spec['raw']), input_file(spec['geometry'])
    out = fresh(out)
    row = dict(key='core', raw=str(raw), raw_hash=sha(raw), geometry=str(geometry),
               geometry_hash=sha(geometry), out30=str(out / 'planned30'), out60=str(out / 'planned60'))
    toml(out / 'prepare.toml', dict(frozen=str(SNAP / 'frozen_preparation'),
         scenario=str(SNAP / 'scenario/scenario.jl')), [row])
    env = dict(os.environ, JULIA_DEPOT_PATH=str(out / 'depot'), JULIA_LOAD_PATH='@:@stdlib',
               JULIA_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    run(julia_args(julia, SNAP / 'campaign/prepare_batch.jl', out / 'prepare.toml', prime=True),
        out / 'prepare.log', env)
    for name in ('raw', 'geometry'):
        input_file(spec[name])
    verify()
    write(out / 'PREPARATION.json', dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
          input_spec=spec, numerical_source_modified=False, outputs={str(p.relative_to(out)): sha(p)
          for name in ('planned30', 'planned60') for p in sorted((out / name).iterdir()) if p.is_file()}))


def flow(specfile, out, julia):
    manifest = verify()
    spec = read(specfile)
    raw, geometry, audit, bridge = [input_file(spec[k]) for k in ('raw', 'geometry', 'scenario_audit', 'bridge')]
    if spec['memory_class'] not in MEMORY_CLASSES or not 1 <= int(spec['slot']) <= 522:
        raise ValueError('Unqualified memory class or invalid slot')
    known_bridge = read(SNAP / 'kernel/manifest.json')['bridge']
    expected_bridge = manifest['external_pins'].get(known_bridge)
    if expected_bridge is None:
        matches = {r['sha256'] for r in manifest['files'].values() if r['source'] == known_bridge}
        if len(matches) == 1:
            expected_bridge = matches.pop()
    if expected_bridge is None or sha(bridge) != expected_bridge:
        raise ValueError('Campaign reproduction requires the pinned Linux bridge; rebuilt bridges need separate qualification')
    out = fresh(out)
    top = dict(pipeline=str(SNAP / 'pipeline'), backend=str(SNAP / 'backend'), bridge=str(bridge),
               pipeline_hash=read(SNAP / 'kernel/manifest.json')['pipeline_hash'],
               kernel_root=str(SNAP / 'kernel'), output=str(out / 'curves'), bootstrap=False,
               memory_mib=MEMORY_CLASSES[spec['memory_class']],
               launch_lock_sha256=sha(SNAP / 'campaign/LAUNCH_LOCK.json'))
    cases = [dict(key='core', slot=int(spec['slot']), raw=str(raw), geometry=str(geometry), scenario_audit=str(audit))]
    pins = {member(n): r['sha256'] for n, r in manifest['files'].items()}
    pins[bridge] = expected_bridge
    pins.update({p: spec[k]['sha256'] for k, p in
                 (('raw', raw), ('geometry', geometry), ('scenario_audit', audit))})
    toml(out / 'prime.toml', dict(top, bootstrap=True), [], pins)
    toml(out / 'flow.toml', top, cases, pins)
    env = dict(os.environ, JULIA_DEPOT_PATH=str(out / 'depot'), JULIA_LOAD_PATH='@:@stdlib',
               JULIA_NUM_THREADS='2', JULIA_NUM_PRECOMPILE_TASKS='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    run(julia_args(julia, SNAP / 'campaign/worker.jl', out / 'prime.toml', prime=True), out / 'prime.log', env)
    run(julia_args(julia, SNAP / 'campaign/worker.jl', out / 'flow.toml'), out / 'flow.log', env)
    if not (out / 'curves/core/CORE_PASS.toml').is_file():
        raise RuntimeError('No numerical acceptance receipt; preserved outputs are not accepted science')
    for name in ('raw', 'geometry', 'scenario_audit', 'bridge'):
        input_file(spec[name])
    verify()
    write(out / 'REPRODUCTION.json', dict(source_manifest_sha256=sha(ROOT / 'SOURCE_MANIFEST.json'),
          input_spec=spec, workflow='physical_prebreakthrough_pc100_kr10', scheduler_submission=False))


def export(curve, out):
    verify()
    out = output_path(out)
    # Only path bindings are relocated; qualified conversion code is unchanged.
    common = load('common_launch', SNAP / 'campaign/common_launch.py')
    common.ROOT, common.QUAL, common.Q3 = out.parent, SNAP / 'full_fault_export', SNAP / 'kernel'
    module = load('export_sparse_capsule', SNAP / 'campaign/export_sparse.py')
    item = dict(kind='flow', curve=str(Path(curve).resolve()), key='core', old_numerical_key='standalone-reproduction')
    result = module.export(item, out)
    verify()
    return result


def onset(specfile, out):
    verify()
    spec = read(specfile)
    inputs = {k: input_file(spec[k]) for k in ('mip', 'physical', 'zero')}
    out = fresh(out)
    source = out / 'reference_inputs/case'
    source.mkdir(parents=True)
    for name, path in inputs.items():
        shutil.copyfile(path, source / (name + '.mat'))
    write(out / 'sources.json', [dict(reference=dict(name='case'), inputs={k: dict(sha256=sha(p)) for k, p in inputs.items()})])
    module = load('qualified_pc_onset', SNAP / 'pc_onset/prepare_inputs.py')
    module.ROOT = out
    module.main()
    verify()
    return read(out / 'inputs/case/pc_only_audit.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    for name in ('prepare', 'flow', 'pc-onset'):
        child = sub.add_parser(name)
        child.add_argument('spec', type=Path)
        child.add_argument('out', type=Path)
        if name != 'pc-onset':
            child.add_argument('--julia', default='julia')
    child = sub.add_parser('export')
    child.add_argument('curve', type=Path)
    child.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(dict(passed=True, files=len(verify()['files']))))
    elif args.command in ('prepare', 'flow'):
        globals()[args.command](args.spec, args.out, args.julia)
    elif args.command == 'export':
        export(args.curve, args.out)
    else:
        onset(args.spec, args.out)


if __name__ == '__main__':
    main()
