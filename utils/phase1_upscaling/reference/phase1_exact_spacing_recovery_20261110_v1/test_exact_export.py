import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from exact_export import load_converter, OLD, NEW


class ExactNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle = Path(__file__).parent / 'science_bundle.zip'
        with zipfile.ZipFile(bundle) as z:
            cls.source = z.read('vendor/kernel/build_tables.py')
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / 'build_tables.py'
        cls.path.write_bytes(cls.source)
        cls.sha = hashlib.sha256(cls.source).hexdigest()
        cls.convert = staticmethod(load_converter(cls.path, cls.sha))
        old = {'__name__': 'baseline_test', '__file__': str(cls.path)}
        exec(compile(cls.source, str(cls.path), 'exec'), old)
        cls.baseline = staticmethod(old['convert'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def raw(self, gap=.1):
        return np.array([[0,0,1,0,0,0,0,0],[10,.2,.5,.2,0,0,1,1],
                         [20,.7,0,.9,0,0,1,1],[30,.7+gap,0,.9,0,0,1,1]], dtype=float)

    def test_single_guard_difference(self):
        text = self.source.decode()
        self.assertEqual(text.replace(OLD, NEW).replace(NEW, OLD), text)

    def test_ordinary_tables_byte_identical(self):
        for gap in (.1, .01, 1e-6, 1e-11):
            x = self.raw(gap)
            a, da = self.baseline(x, 5)
            b, db = self.convert(x, 5)
            self.assertTrue(np.array_equal(a,b))
            self.assertEqual(da, db)

    def test_tiny_positive_nodes_preserved(self):
        for gap in (1e-13, 4e-14):
            x = self.raw(gap)
            before = x.copy()
            with self.assertRaises(AssertionError):
                self.baseline(x, 5)
            t, d = self.convert(x, 5)
            self.assertTrue(np.array_equal(t[:,0], x[:,1]))
            self.assertTrue(np.array_equal(t[1:,3], x[1:,0]))
            self.assertTrue(np.array_equal(x,before))
            self.assertTrue(np.isfinite(d['max_pc_slope_pa']))

    def test_nonmonotone_rejected(self):
        x = self.raw(-1e-13)
        with self.assertRaises(AssertionError): self.convert(x,5)

    def test_conflicting_duplicate_rejected(self):
        x = self.raw(0)
        x[-1,3] = .95
        with self.assertRaises(AssertionError): self.convert(x,5)

    def test_existing_terminal_plateau_unchanged(self):
        x = self.raw(0)
        a,da = self.baseline(x,5)
        b,db = self.convert(x,5)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(da,db)

    def test_physics_gates_still_fail(self):
        for key in ('negative_kr','kr_reversal','wrong_entry','nonfinite','positive_terminal_brine','pressure_reverse'):
            x = self.raw()
            entry = 5
            if key == 'negative_kr': x[1,3] = -.1
            if key == 'kr_reversal': x[-1,3] = .8
            if key == 'wrong_entry': entry = 15
            if key == 'nonfinite': x[2,0] = np.inf
            if key == 'positive_terminal_brine': x[2:,2] = .1
            if key == 'pressure_reverse': x[2,0] = 9
            with self.subTest(key=key), self.assertRaises(AssertionError): self.convert(x,entry)

    def test_wrong_source_rejected(self):
        with self.assertRaises(RuntimeError): load_converter(self.path, '0'*64)


if __name__ == '__main__':
    unittest.main()
