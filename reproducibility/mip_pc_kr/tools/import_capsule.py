"""Import an authenticated source capsule without replacing existing source."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def import_capsule(archive, digest, root):
    root = root.resolve()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError('Transport checksum mismatch')
    with tarfile.open(archive) as source:
        members = source.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive names')
        for item in members:
            if not item.isfile() or not (root / item.name).resolve().is_relative_to(root):
                raise ValueError('Unsafe archive member')
        data = source.extractfile('SOURCE_MANIFEST.json').read()
        manifest = json.loads(data)
        if set(names) != set(manifest['files']) | {'SOURCE_MANIFEST.json'}:
            raise ValueError('Unmanifested members')
        for name, record in manifest['files'].items():
            if not name.startswith('snapshot/'):
                raise ValueError('Non-source member')
            contents = source.extractfile(name).read()
            if len(contents) != record['bytes'] or hashlib.sha256(contents).hexdigest() != record['sha256']:
                raise ValueError('Source checksum mismatch')
            p = root / name
            if p.exists() and p.read_bytes() != contents:
                raise ValueError('Refusing to replace existing source: ' + name)
        for name in manifest['files']:
            p = root / name
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('xb') as output:
                    output.write(source.extractfile(name).read())
        p = root / 'SOURCE_MANIFEST.json.partial'
        with p.open('xb') as output:
            output.write(data)
        p.replace(root / 'SOURCE_MANIFEST.json')
    print(json.dumps(dict(imported=True, files=len(manifest['files']))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('sha256')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    import_capsule(a.archive, a.sha256, a.root)
