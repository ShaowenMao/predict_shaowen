"""Exact scientific-input reuse. Never infer equality from draw IDs or averages."""
from collections import defaultdict
from pathlib import Path
import copy
import hashlib
import json
import struct


SCHEMA = 'exact_core_input_reuse_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalized_file(path, header_size, edits):
    """Hash all bytes, replacing only explicitly validated provenance fields."""
    actual = hashlib.sha256()
    normalized = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        head = stream.read(header_size)
        assert len(head) == header_size, 'Truncated input header'
        actual.update(head)
        modified = bytearray(head)
        for start, stop, value in edits:
            assert len(value) == stop - start
            modified[start:stop] = value
        normalized.update(modified)
        size += len(head)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            actual.update(block)
            normalized.update(block)
            size += len(block)
    return actual.hexdigest(), normalized.hexdigest(), size, head


def fingerprint(raw, geometry, slot, raw_hash, geometry_hash, method, *, unplanned=False):
    """The core label is not physics; geometry's raw-hash link remains enforced.

    Layouts are those of frozen FaultCorePc.read_input and read_geometry.
    No float rounding, coordinate shifting, cell sorting or property averaging.
    """
    actual, raw_normal, size, head = normalized_file(raw, 96, [(8, 16, bytes(8))])
    magic, core, n, nk, ne, ni, no, n1, n2, np_, nd, nc = struct.unpack('<12q', head)
    assert magic == 2026090904 and core == slot and n > 0
    assert nk in (1, 3, 6) and ne >= 0 and ni > 0 and no > 0
    assert n1 >= 2 and n2 >= 2 and np_ >= 2 and nd >= 0 and nc == 15
    expected = 96 + 8 * nc + 8 * n * nk + 16 * n + 16 * ne + 8 * (ni + no)
    expected += n + 32 * (n1 + n2) + 8 * (np_ + nd)
    assert size == expected and actual == raw_hash, 'Raw identity/length mismatch'
    ga, gn, gs, gh = normalized_file(
        geometry, 104, [(8, 16, bytes(8)), (72, 104, bytes.fromhex(raw_normal))])
    gm, gc, axis, nn, ee, ii, oo = struct.unpack('<7q', gh[:56])
    assert gm == 2026090906 and gc == slot and axis == 3
    assert (nn, ee, ii, oo) == (n, ne, ni, no)
    if unplanned:
        # Frozen prepare_slot.m writes a zero placeholder; the accepted replay
        # receipt binds both complete files. Frozen Julia planning sets the link.
        assert gh[72:104] == bytes(32), 'Unexpected unplanned geometry link'
    else:
        assert gh[72:104].hex() == raw_hash, 'Geometry not bound to this raw input'
    assert gs == 104 + 32 * ne + 16 * (ni + no) and ga == geometry_hash
    fields = dict(schema=SCHEMA, input_stage='unplanned' if unplanned else 'planned', raw_scientific_sha256=raw_normal,
                  geometry_scientific_sha256=gn, method_sha256=method)
    return dict(**fields, numerical_key=digest(fields), raw_bytes=size, geometry_bytes=gs)


def group_records(records, fingerprints, preferred=()):
    groups = defaultdict(list)
    for key, row in records.items():
        assert row['key'] == key
        groups[fingerprints[key]['numerical_key']].append(key)
    owners, placements, mapping = {}, {}, {}
    for numerical_key, members in sorted(groups.items()):
        members.sort()
        owner = min(members, key=lambda key: (key not in preferred, key))
        r = copy.deepcopy(records[owner])
        for key in members:
            assert records[key]['resource'] == r['resource'], 'Conflicting resource admission'
            assert fingerprints[key] == fingerprints[owner], 'Fingerprint conflict'
            assert records[key]['features'] == r['features'], 'Feature conflict'
            placement = copy.deepcopy(records[key])
            placement.update(owner=owner, numerical_key=numerical_key)
            placements[key] = placement
        r.update(numerical_key=numerical_key, placements=members)
        owners[owner] = r
        mapping[numerical_key] = dict(owner=owner, placements=members,
                                      fingerprint=fingerprints[owner])
    assert set(placements) == set(records)
    assert sum(len(g['placements']) for g in mapping.values()) == len(records)
    return owners, placements, mapping


def resolve_case(plan, case, accepted, require_complete=True):
    """Return table pointers without inheriting the representative's region label."""
    rows = sorted((r for r in plan['placements'].values() if r['case'] == case),
                  key=lambda r: r['slot'])
    if require_complete:
        assert [r['slot'] for r in rows] == list(range(1, 523)), 'Case still has scoped holds'
    result = []
    for r in rows:
        d = accepted[r['owner']]
        assert d['complete'] and d['memory']['passed'] and d['native_passed']
        assert d['numerical_key'] == r['numerical_key']
        table = d['sgof']
        assert table in d['artifacts'], 'Unbound export table'
        result.append(dict(case=case, slot=r['slot'], region=r['region'],
                           window=r['window'], slice=r['slice'], owner=r['owner'],
                           numerical_key=r['numerical_key'], sgof=table,
                           sgof_sha256=d['artifacts'][table],
                           replay_receipt=r['replay_receipt'],
                           replay_receipt_sha256=r['replay_receipt_sha256']))
    return result
