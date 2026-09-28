"""Immutable, bounded qualification artifacts; never write to source studies."""
import hashlib
import json
import os
from pathlib import Path

from bounded_io import discard_consumed_pages

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
            discard_consumed_pages(f)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == content, 'Refuse overwrite: ' + str(path)
        return
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    # Atomic no-clobber publication on the shared filesystem.
    os.link(tmp, path)
    tmp.unlink()


def pins_ok(pins):
    for path, expected in pins.items():
        assert sha(path) == expected, 'Changed pinned artifact: ' + str(path)


def verify():
    lock = read(HERE / 'LOCK.json')
    pins_ok({HERE / p: h for p, h in lock['files'].items()})
    cfg = read(HERE / 'config.json')
    assert cfg['remote'] == str(HERE)
    assert cfg['qualification_only'] and not cfg['campaign_launch']
    return cfg


def receipt(stage):
    p = HERE / 'receipts' / (stage + '.json')
    r = read(p)
    assert r['passed'] and r['lock_sha256'] == sha(HERE / 'LOCK.json')
    pins_ok(r['artifacts'])
    return r
