import unittest
import numpy as np
from build_tables import convert


def curve():
    return np.array([[0,0,1,0,0,0,0,0],[9,0,1,0,0,0,0,0],
        [11,.01,.9,0,0,0,0,1],[20,.3,.4,.3,0,0,1,1],
        [30,.7,0,.8,0,0,1,1],[40,.7,0,.8,0,0,1,1]],float)


class ExportTests(unittest.TestCase):
    def test_lossless_branches_and_explicit_endpoints(self):
        x=curve();original=x.copy();t,a=convert(x,10)
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(t[:,0],[0,.01,.3,.7])
        np.testing.assert_array_equal(t[:,3],[10,11,20,30])
        self.assertEqual(a['terminal_rows_removed'],1)
        self.assertEqual(a['max_absolute_krg_change'],0)
        self.assertEqual(t[-1,1],.8)

    def test_reject_interior_vertical_branch(self):
        x=curve();x[3,1]=x[2,1]
        with self.assertRaises(AssertionError):convert(x,10)

    def test_reject_changed_mobility_at_duplicate(self):
        x=curve();x[-1,3]=.9
        with self.assertRaises(AssertionError):convert(x,10)

    def test_reject_invalid_entry_bracket(self):
        for e in (0,8,12,float('nan')):
            with self.assertRaises(AssertionError):convert(curve(),e)

    def test_reject_nonmonotone_or_nearly_repeated_sg(self):
        for s in (.009,.0100000000001):
            x=curve();x[3,1]=s
            with self.assertRaises(AssertionError):convert(x,10)

    def test_bounded_cleanup(self):
        x=curve();x[2,3]=.3+1e-10
        t,a=convert(x,10)
        self.assertLess(a['max_absolute_krg_change'],1.1e-10)
        self.assertTrue(np.all(np.diff(t[:,1])>=0))
        self.assertEqual(t[-1,1],.8)

    def test_reject_large_kr_reversal(self):
        x=curve();x[2,3]=.4
        with self.assertRaises(AssertionError):convert(x,10)

    def test_roundoff_above_one(self):
        x=curve();x[2,2]=1+4e-12
        t,a=convert(x,10)
        self.assertTrue(np.all(t[:,2]<=1))
        self.assertLess(a['max_absolute_krw_change'],5e-12)
        x[2,2]=1+1e-6
        with self.assertRaises(AssertionError):convert(x,10)

    def test_never_force_positive_terminal_brine_to_zero(self):
        x=curve();x[-2:,2]=.1
        with self.assertRaises(AssertionError):convert(x,10)

    def test_reject_nan(self):
        x=curve();x[3,3]=float('nan')
        with self.assertRaises(AssertionError):convert(x,10)


if __name__=='__main__':unittest.main()
