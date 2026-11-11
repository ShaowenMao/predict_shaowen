"""Maintainer-only, local snapshot import. Never submits jobs or changes sources."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    'phase1_pc40_adaptive10_2592_configuration_20261030_v2': '7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36',
    'phase1_pc40_adaptive10_runner_20261030_v4': '709f7267bd1233138a1d1e50e7999f536343f566ea01e51a352922398fdabbd8',
    'phase1_pc40_adaptive10_runner_20261030_v5': 'c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf',
    'phase1_round2_20261109_v4': '988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019',
    'phase1_round3_20261110_v1': '2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b',
    'phase1_exact_spacing_recovery_20261110_v1': '67fda590f27d0ffac6ebaac334b6ed5e62746cda7f1b97ef1b48ac4d6b48dfe3',
    'phase1_saved16_closeout_20261110_v2': 'f8b9f0254f6ede51b4ca5e5f57a86ba0a773627a8cd7a1a9219772235930b233',
    'phase1_solver_targeted_20261110_v1': '846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502',
    'phase1_solver_window8_20261110_v2': 'f7cb39e7b0934f08249a728333c972e90553566f3ae45d9553928c76ecaf97bd',
    'phase1_precision_recovery_20261111_v1': '1bf8764d77764c1813cc6bbcc59911deb52c1fa26162b8b0cabb268f8bcc4b82',
    'phase1_precision_batches_20261111_v1': 'ada40e545387809fdedd8afd1781f9c75f2f10727178611baa0fcaabf792ab80',
    'phase1_direct_primary8_20261111_v1': 'f1c938d2ba66265232da9a7fafb842674aa37a7e903dc302c1cdf80df9bbe911',
    'phase1_final_reference4_20261111_v2': '8b61b7dabc42f57c858a2d137858fece866174ede388aa75fa6479137b2c0fbe',
}
CONFIG = next(iter(PACKAGES))
RUNNER = 'phase1_pc40_adaptive10_runner_20261030_v5'
AUDIT = 'phase1_completion_audit_20261111_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe snapshot path: ' + name)
    return Path(*path.parts)


def config_source(name):
    p = PurePosixPath(name)
    return (len(p.parts) == 1 or p.parts[0] in ('profiles', 'adaptive', 'reference_tables')
            or name.startswith('sampling/code/')
            or name in ('sampling/README.md', 'sampling/CONFIG.json',
                        'sampling/phase1/scenario_catalog.csv'))


def capture(source_root, destination):
    source_root, destination = source_root.resolve(), destination.resolve()
    if destination == source_root or destination.is_relative_to(source_root):
        raise ValueError('Snapshot destination must be outside the study tree')
    if any((destination / n).exists() for n in ('reference', 'science', 'SOURCE_MANIFEST.json')):
        raise ValueError('Refusing to overwrite an existing snapshot')
    inventory, packages = {}, {}

    def copy(source, target, expected=None):
        if source.is_symlink() or not source.is_file():
            raise ValueError('Missing or linked source: ' + str(source))
        actual = sha(source)
        if expected is not None and actual != expected:
            raise ValueError('Source pin mismatch: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != actual:
            raise ValueError('Copy changed bytes: ' + str(target))
        inventory[target.relative_to(destination).as_posix()] = actual

    for name, pin in PACKAGES.items():
        source = source_root / name
        if sha(source / 'LOCK.json') != pin:
            raise ValueError('Unexpected source version: ' + name)
        lock = json.loads((source / 'LOCK.json').read_text())
        files = dict(lock.get('code', lock.get('files', {})))
        for case, digest in lock.get('plans', {}).items():
            files['plans/' + case + '.json'] = digest
        if 'science_bundle_sha256' in lock:
            files['science_bundle.zip'] = lock['science_bundle_sha256']
        chosen = {n: h for n, h in files.items() if name != CONFIG or config_source(n)}
        target = destination / 'reference' / name
        copy(source / 'LOCK.json', target / 'LOCK.json', pin)
        for n, h in sorted(chosen.items()):
            copy(source / relative(n), target / relative(n), h)
        # These are historical evidence, not a new authorization or live status.
        for n in ('HANDOFF.md', 'README.md', 'qualification/UNIT_PASS.json',
                  'qualification/FIXTURE_PASS.json'):
            if (source / n).is_file() and n not in chosen:
                copy(source / n, target / n)
        packages[name] = dict(lock_sha256=pin, omitted_locked_files={
            n: h for n, h in sorted(files.items()) if n not in chosen},
            original_cluster_directory='/orcd/data/juanes/001/shaowen/' + name)

    for n in ('audit_completion.py', 'test_audit.py', 'README.md', 'COMPLETION_AUDIT.json'):
        copy(source_root / AUDIT / n, destination / 'reference' / AUDIT / n)

    # A readable mirror of the exact scientific source bundle, not a new engine.
    lock = json.loads((source_root / RUNNER / 'LOCK.json').read_text())
    with zipfile.ZipFile(source_root / RUNNER / 'science_bundle.zip') as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(lock['science']):
            raise ValueError('Science inventory mismatch')
        for n in names:
            data = bundle.read(n)
            h = hashlib.sha256(data).hexdigest()
            if h != lock['science'][n]:
                raise ValueError('Science source mismatch: ' + n)
            target = destination / 'science' / relative(n)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            inventory[target.relative_to(destination).as_posix()] = h
    manifest = dict(schema=1, description='Read-only historical reference; not a deployment or launch authorization',
                    packages=packages, science_source=RUNNER,
                    files=dict(sorted(inventory.items())))
    with (destination / 'SOURCE_MANIFEST.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(f'Captured {len(inventory)} files; original study directories unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    capture(args.source_root, args.destination)
