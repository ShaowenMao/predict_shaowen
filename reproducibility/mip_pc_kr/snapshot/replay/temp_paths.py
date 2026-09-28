"""Short, exclusive project-storage directories for MATLAB startup IPC."""
import json
import os
from pathlib import Path
import re

TEMP_ROOT = Path('/orcd/data/juanes/001/shaowen/w60tmp')
# A conservative launcher guard, not a claim about MATLAB's exact IPC limit.
MAX_TEMP_BYTES = 64


def layout(root, job):
    assert isinstance(job, str) and re.fullmatch(r'[0-9]+', job), 'Invalid job ID'
    root = Path(root)
    assert root.is_absolute() and '..' not in root.parts
    folder = root/('j'+job)
    paths = dict(folder=folder, tmp=folder/'t', preferences=folder/'p', cache=folder/'c')
    assert max(len(os.fsencode(p)) for p in paths.values()) <= MAX_TEMP_BYTES, 'Temporary path too long'
    return paths


def create(root, job, release):
    assert re.fullmatch(r'[0-9a-f]{16}', release), 'Invalid release ID'
    paths = layout(root, job)
    root = Path(root)
    assert root.resolve() == root.absolute(), 'Temporary root must not traverse symlinks'
    assert root.parent.is_dir()
    root.mkdir(mode=0o700, exist_ok=True)
    if hasattr(os, 'getuid'):
        assert root.stat().st_uid == os.getuid(), 'Temporary root has a different owner'
        assert root.stat().st_mode & 0o077 == 0, 'Temporary root must be private'
    paths['folder'].mkdir(mode=0o700, exist_ok=False)
    for name in ('tmp', 'preferences', 'cache'):
        paths[name].mkdir(mode=0o700)
    with (paths['folder']/'owner.json').open('x') as f:
        json.dump(dict(job=job, release=release, paths={k:str(v) for k,v in paths.items()}), f, indent=2)
    return paths


def validate_environment(root, job, release, env):
    paths = layout(root, job)
    owner = json.loads((paths['folder']/'owner.json').read_bytes())
    assert owner == dict(job=job, release=release, paths={k:str(v) for k,v in paths.items()})
    for p in paths.values():
        assert p.is_dir() and p.resolve() == p.absolute(), 'Unexpected temporary directory redirect'
    expected = dict(TMPDIR=paths['tmp'], TMP=paths['tmp'], TEMP=paths['tmp'],
                    TEMPDIR=paths['tmp'], MATLAB_PREFDIR=paths['preferences'], MCR_CACHE_ROOT=paths['cache'])
    for key, path in expected.items():
        assert env.get(key) == str(path), 'Unexpected temporary environment: '+key
    return paths


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    lock = json.loads((here/'source_lock.json').read_bytes())
    create(TEMP_ROOT, os.environ['SLURM_JOB_ID'], lock['release'])
