"""Immutable runtime transport; every staged byte is checked against old pins."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile

MAX_FILES = 5000
MAX_BYTES = 512 * 1024**2
MAX_FILE = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def validate_inventory(inventory):
    require(0 < len(inventory) <= MAX_FILES, 'Unbounded runtime inventory')
    for name, pin in inventory.items():
        p = PurePosixPath(name)
        require(p.is_absolute() and str(p) == name and '..' not in p.parts and
                '\\' not in name and '\x00' not in name, 'Unsafe runtime path')
        require(len(pin) == 64 and all(c in '0123456789abcdef' for c in pin), 'Invalid source pin')


def build(inventory, archive, scratch, progress=lambda value: None):
    """Read each original exactly once; publish only a complete checked archive."""
    validate_inventory(inventory)
    archive, scratch = Path(archive), Path(scratch)
    require(not archive.exists(), 'Existing bundle is immutable')
    require(shutil.disk_usage(scratch).free > 2*MAX_BYTES, 'Insufficient local bundle space')
    local = scratch/'runtime-building.zip'
    records, total = {}, 0
    started = time.monotonic()
    with zipfile.ZipFile(local, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for index, (name, pin) in enumerate(sorted(inventory.items())):
            path = Path(name)
            with path.open('rb') as stream:
                before = os.fstat(stream.fileno())
                require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE, 'Unsafe runtime member')
                data = stream.read(MAX_FILE+1)
                after = os.fstat(stream.fileno())
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during read')
            require(len(data) == before.st_size and hashlib.sha256(data).hexdigest() == pin,
                    'Original checksum mismatch: '+name)
            total += len(data)
            require(total <= MAX_BYTES, 'Runtime bundle exceeds reserved budget')
            member = f'runtime/{index:05d}'
            out.writestr(member, data)
            records[name] = dict(member=member, sha256=pin, size=len(data))
            if index % 100 == 0:
                progress(dict(files=index+1, total_files=len(inventory), bytes=total,
                              elapsed_seconds=time.monotonic()-started))
    require(local.stat().st_size <= MAX_BYTES, 'Transport archive too large')
    digest = sha(local)
    partial = archive.with_suffix('.partial')
    with local.open('rb') as inp, partial.open('xb') as out:
        shutil.copyfileobj(inp, out, 1024**2)
        out.flush(); os.fsync(out.fileno())
    require(sha(partial) == digest, 'Shared archive readback mismatch')
    os.link(partial, archive)
    partial.unlink()
    return dict(schema='pinned-node-local-runtime-v1', archive=str(archive), archive_sha256=digest,
                archive_bytes=archive.stat().st_size, expanded_bytes=total, files=records,
                build_seconds=time.monotonic()-started, originals_unchanged=True)


def stage(manifest, inventory, root):
    validate_inventory(inventory)
    require(manifest['schema'] == 'pinned-node-local-runtime-v1', 'Wrong bundle format')
    files = manifest['files']
    require(set(files) == set(inventory), 'Bundle inventory differs')
    require(0 <= manifest['expanded_bytes'] <= MAX_BYTES and
            0 < manifest['archive_bytes'] <= MAX_BYTES, 'Bundle budget differs')
    require(sum(v['size'] for v in files.values()) == manifest['expanded_bytes'], 'Expanded budget differs')
    members = {v['member'] for v in files.values()}
    require(members == {f'runtime/{n:05d}' for n in range(len(files))}, 'Member names differ')
    for name, value in files.items():
        require(value['sha256'] == inventory[name] and 0 <= value['size'] <= MAX_FILE, 'Member pin/size differs')
    root = Path(root)
    root.mkdir(mode=0o700, exist_ok=False)
    require(not root.is_symlink() and shutil.disk_usage(root).free > 2*MAX_BYTES, 'Unsafe/local space unavailable')
    started = time.monotonic()
    local = root/'transport.zip'
    h, count = hashlib.sha256(), 0
    with Path(manifest['archive']).open('rb') as inp, local.open('xb') as out:
        for block in iter(lambda: inp.read(1024**2), b''):
            count += len(block)
            require(count <= manifest['archive_bytes'], 'Archive grew')
            h.update(block); out.write(block)
    require(count == manifest['archive_bytes'] and h.hexdigest() == manifest['archive_sha256'], 'Transport checksum mismatch')
    mapping = {}
    with zipfile.ZipFile(local) as bundle:
        require(len(bundle.infolist()) == len(files) and set(bundle.namelist()) == members, 'Unexpected/duplicate ZIP member')
        for original, value in files.items():
            info = bundle.getinfo(value['member'])
            require(not info.is_dir() and info.file_size == value['size'] and
                    not stat.S_ISLNK(info.external_attr >> 16), 'Unsafe ZIP member')
            target = root/'tree'/original.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            data = bundle.read(info)
            require(hashlib.sha256(data).hexdigest() == inventory[original], 'Staged member checksum mismatch')
            with target.open('xb') as out:
                out.write(data)
            require(sha(target) == inventory[original], 'Local readback mismatch')
            target.chmod(0o400)
            mapping[original] = str(target)
    return dict(mapping=mapping, seconds=time.monotonic()-started, files=len(files),
                archive_sha256=manifest['archive_sha256'], expanded_bytes=manifest['expanded_bytes'])


def install_replay(replay, staged, inventory_path, inventory_sha256):
    """Move only runtime lookup; frozen replay and original-record identities stay."""
    original_read, original_atomic = replay.read, replay.atomic
    mapping = staged['mapping']
    reverse = {v: k for k, v in mapping.items()}
    original_inventory = original_read(inventory_path)
    require(sha(inventory_path) == inventory_sha256, 'Runtime inventory identity changed')
    require(set(mapping) == set(original_inventory['inventory']), 'Staged inventory incomplete')
    for field in ('MRST', 'PHYSICS'):
        original = Path(getattr(replay, field)).as_posix()
        children = [p for p in mapping if p.startswith(original+'/')]
        require(children, 'Runtime root absent: '+field)
        sample = children[0]
        new_root = mapping[sample][:-len(sample[len(original):])]
        require(Path(new_root).is_dir(), 'Staged root absent')
        setattr(replay, field, new_root if field == 'MRST' else Path(new_root))

    def read(path):
        value = original_read(path)
        if Path(path) == Path(inventory_path):
            require(sha(path) == inventory_sha256, 'Runtime inventory changed')
            value = dict(value, inventory={mapping[p]: h for p, h in value['inventory'].items()})
        return value

    def atomic(path, value, replace=False):
        if isinstance(value, dict) and 'loaded_sources' in value:
            value = dict(value, loaded_sources={reverse.get(p, p): h for p, h in value['loaded_sources'].items()},
                         runtime_staging=dict(archive_sha256=staged['archive_sha256'],
                                              inventory_sha256=inventory_sha256,
                                              local_loaded_sources=value['loaded_sources'],
                                              staged_files=staged['files'], staging_seconds=staged['seconds']))
        return original_atomic(path, value, replace)

    replay.read, replay.atomic = read, atomic
