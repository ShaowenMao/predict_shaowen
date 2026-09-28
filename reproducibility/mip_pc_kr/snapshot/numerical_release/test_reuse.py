from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import hashlib
import struct
import unittest
import reuse as R


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def files(self, slot, change=None):
        header = struct.pack('<12q', 2026090904, slot, 2, 1, 1, 1, 1, 2, 2, 2, 0, 15)
        payload = bytearray(120 + 16 + 32 + 16 + 16 + 2 + 128 + 16)
        if change == 'physics':
            payload[123] = 1
        raw = header + payload
        rh = hashlib.sha256(raw).hexdigest()
        geo = bytearray(struct.pack('<7q2d', 2026090906, slot, 3, 2, 1, 1, 1, 2., 1.))
        geo.extend(bytes.fromhex(rh))
        geo.extend(bytes(64))
        if change == 'geometry':
            geo[-1] = 1
        rp = self.root / f'raw{slot}{change}.bin'
        gp = self.root / f'geo{slot}{change}.bin'
        rp.write_bytes(raw)
        gp.write_bytes(geo)
        return [rp, gp, slot, rh, hashlib.sha256(geo).hexdigest(), 'method60']

    def test_labels_do_not_change_science(self):
        a, b = self.files(1), self.files(222)
        self.assertNotEqual(a[3], b[3])
        self.assertEqual(R.fingerprint(*a), R.fingerprint(*b))

    def test_any_property_byte_changes_key(self):
        self.assertNotEqual(R.fingerprint(*self.files(1)), R.fingerprint(*self.files(2, 'physics')))

    def test_geometry_changes_key(self):
        self.assertNotEqual(R.fingerprint(*self.files(1)), R.fingerprint(*self.files(2, 'geometry')))

    def test_method_changes_key(self):
        a = self.files(1)
        b = list(a); b[-1] = 'method30'
        self.assertNotEqual(R.fingerprint(*a), R.fingerprint(*b))

    def test_bad_provenance_and_labels_rejected(self):
        for index, value in ((2, 5), (3, '0' * 64), (4, '0' * 64)):
            a = self.files(1); a[index] = value
            with self.assertRaises(AssertionError): R.fingerprint(*a)

    def test_broken_geometry_link_rejected(self):
        a = self.files(1)
        blob = bytearray(a[1].read_bytes()); blob[80] ^= 1
        a[1].write_bytes(blob); a[4] = hashlib.sha256(blob).hexdigest()
        with self.assertRaises(AssertionError): R.fingerprint(*a)

    def test_unplanned_zero_link_requires_explicit_stage(self):
        a=self.files(1); blob=bytearray(a[1].read_bytes());blob[72:104]=bytes(32)
        a[1].write_bytes(blob);a[4]=hashlib.sha256(blob).hexdigest()
        with self.assertRaises(AssertionError):R.fingerprint(*a)
        fp=R.fingerprint(*a,unplanned=True)
        self.assertEqual(fp['input_stage'],'unplanned')

    def test_unplanned_nonzero_link_not_silently_ignored(self):
        with self.assertRaises(AssertionError):R.fingerprint(*self.files(1),unplanned=True)

    def test_unplanned_labels_match(self):
        results=[]
        for slot in (1,222):
            a=self.files(slot);blob=bytearray(a[1].read_bytes());blob[72:104]=bytes(32)
            a[1].write_bytes(blob);a[4]=hashlib.sha256(blob).hexdigest()
            results.append(R.fingerprint(*a,unplanned=True))
        self.assertEqual(*results)

    def test_trailing_and_truncated_files_rejected(self):
        for tail in (b'extra', None):
            a = self.files(1); b = a[0].read_bytes()
            b = b + tail if tail else b[:-1]
            a[0].write_bytes(b); a[3] = hashlib.sha256(b).hexdigest()
            with self.assertRaises(AssertionError): R.fingerprint(*a)

    def fixture(self):
        rows = {k: dict(key=k, case='case', slot=i, region=i+8, window='W1', slice=i,
                        resource={'memory_gib':4}, features={'n':2},
                        replay_receipt=k+'.json', replay_receipt_sha256=k)
                for i, k in enumerate(('a', 'b', 'c'), 1)}
        fps = {k: R.fingerprint(*self.files(i)) for i,k in enumerate(rows, 1)}
        return rows, fps

    def test_one_owner_three_placements_and_preferred_control(self):
        rows, fps = self.fixture()
        owners, places, groups = R.group_records(rows, fps, preferred=['b'])
        self.assertEqual(list(owners), ['b'])
        self.assertEqual(len(places), 3)
        self.assertTrue(all(r['owner']=='b' for r in places.values()))
        self.assertEqual(len(groups), 1)

    def test_order_independent(self):
        rows, fps = self.fixture()
        self.assertEqual(R.group_records(rows, fps), R.group_records(dict(reversed(list(rows.items()))), fps))

    def test_resource_conflict_rejected(self):
        rows, fps = self.fixture(); rows['a']['resource']['memory_gib']=10
        with self.assertRaises(AssertionError): R.group_records(rows, fps)

    def test_no_reuse_when_science_differs(self):
        rows, fps = self.fixture(); fps['b']=R.fingerprint(*self.files(2, 'physics'))
        self.assertEqual(len(R.group_records(rows, fps)[0]), 2)

    def test_mapping_keeps_each_region(self):
        rows, fps = self.fixture(); owners, places, _ = R.group_records(rows, fps)
        done = dict(complete=True, memory={'passed':True}, native_passed=True,
                    numerical_key=owners['a']['numerical_key'], sgof='table.csv', artifacts={'table.csv':'hash'})
        result=R.resolve_case(dict(placements=places), 'case', {'a':done}, False)
        self.assertEqual([r['region'] for r in result], [9,10,11])
        self.assertEqual([r['sgof'] for r in result], ['table.csv']*3)
        with self.assertRaises(AssertionError): R.resolve_case(dict(placements=places), 'case', {'a':done})
        done['native_passed']=False
        with self.assertRaises(AssertionError): R.resolve_case(dict(placements=places), 'case', {'a':done}, False)

    def test_complete_522_mapping(self):
        rows, fps=self.fixture(); template=rows['a']; fp=fps['a']
        rows={str(i):dict(template, key=str(i),slot=i,region=i+8) for i in range(1,523)}
        owners,places,_=R.group_records(rows,{k:fp for k in rows})
        owner=next(iter(owners))
        done=dict(complete=True,memory={'passed':True},native_passed=True,
                  numerical_key=owners[owner]['numerical_key'],sgof='t',artifacts={'t':'h'})
        result=R.resolve_case(dict(placements=places),'case',{owner:done})
        self.assertEqual(len(result),522)
        self.assertEqual([r['region'] for r in result],list(range(9,531)))


if __name__ == '__main__': unittest.main()
