"""Verify the Git reference snapshot offline; no cluster access or source imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

from capture_reference import PACKAGES, CONFIG, RUNNER, AUDIT, relative, sha


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    require(manifest['schema'] == 1, 'Unknown manifest schema')
    require(set(manifest['packages']) == set(PACKAGES), 'Snapshot package set differs')
    files = manifest['files']
    for n, expected in files.items():
        p = root / relative(n)
        require(not p.is_symlink() and p.resolve().is_relative_to(root), 'Linked/escaped file: ' + n)
        require(p.is_file() and sha(p) == expected, 'Missing/changed snapshot file: ' + n)
        if p.suffix == '.py':
            ast.parse(p.read_bytes(), filename=n)
    actual = {p.relative_to(root).as_posix() for sub in ('reference', 'science')
              for p in (root / sub).rglob('*') if p.is_file()}
    require(actual == set(files), 'Unlisted snapshot files')
    omitted = 0
    for name, pin in PACKAGES.items():
        package = root / 'reference' / name
        entry = manifest['packages'][name]
        require(entry['lock_sha256'] == pin and sha(package / 'LOCK.json') == pin,
                'Original LOCK pin differs: ' + name)
        lock = json.loads((package / 'LOCK.json').read_text())
        expected = dict(lock.get('code', lock.get('files', {})))
        expected.update({'plans/' + c + '.json': h for c, h in lock.get('plans', {}).items()})
        if 'science_bundle_sha256' in lock:
            expected['science_bundle.zip'] = lock['science_bundle_sha256']
        external = entry['omitted_locked_files']
        require(name == CONFIG or not external, 'Runtime source unexpectedly omitted: ' + name)
        require(all(n in expected and expected[n] == h for n, h in external.items()),
                'Unknown omitted file')
        omitted += len(external)
        for n, h in expected.items():
            if n in external:
                require(not (package / relative(n)).exists(), 'Omitted file unexpectedly included')
            else:
                key = (package / relative(n)).relative_to(root).as_posix()
                require(files.get(key) == h, 'LOCK coverage differs: ' + key)
        if 'science' in lock:
            with zipfile.ZipFile(package / 'science_bundle.zip') as z:
                require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate bundle member')
                require(set(z.namelist()) == set(lock['science']), 'Science bundle inventory differs')
                for n, h in lock['science'].items():
                    relative(n)
                    require(hashlib.sha256(z.read(n)).hexdigest() == h, 'Changed science member: ' + n)
    science_lock = json.loads((root / 'reference' / RUNNER / 'LOCK.json').read_text())
    require(manifest['science_source'] == RUNNER, 'Unexpected readable science mirror')
    for n, h in science_lock['science'].items():
        require(files.get('science/' + n) == h, 'Science mirror differs: ' + n)
    complete = json.loads((root / 'reference' / AUDIT / 'COMPLETION_AUDIT.json').read_text())
    require(complete['cases_complete'] == complete['cases_expected'] == 2592, 'Completion count differs')
    require(complete['configuration_sha256'] == PACKAGES[CONFIG], 'Completion configuration differs')
    require(complete['method'] == science_lock['method_sha256'], 'Completion method differs')
    require(complete['passed'] and complete['positions_complete'] == 2592 * 522
            and complete['remaining_positions'] == 0 and complete['remaining_cases'] == [],
            'Incomplete historical closeout')
    return dict(verified_files=len(files), packages=len(PACKAGES),
                external_configuration_files=omitted, external_data_verified=False,
                live_campaign_checked=False, numerical_work_run=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
