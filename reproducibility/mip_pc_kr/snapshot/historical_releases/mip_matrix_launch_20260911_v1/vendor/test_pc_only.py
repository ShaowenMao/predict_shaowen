import unittest
import numpy as np
from build_pc_only import convert


class PcOnlyTests(unittest.TestCase):
    def setUp(self):
        self.t=np.array([[0,0,1,1], [.1,0,.8,2], [.3,.1,.5,5], [.7,.7,0,9.]])
        self.raw=np.array([[0,0,1,0,0,0,0,0], [1,0,1,0,0,0,0,0],
            [2,.1,.8,0,0,0,0,.2], [5,.3,.5,.1,0,0,1,.4], [9,.7,0,.7,0,0,1,1.]])

    def test_pc_only_preserves_coordinates_kr_and_post_branch(self):
        t,r=convert(self.t,self.raw,4.)
        np.testing.assert_array_equal(t[:,:3],self.t[:,:3])
        np.testing.assert_array_equal(t[2:],self.t[2:])
        self.assertEqual(t[0,3],4.)
        self.assertAlmostEqual(t[1,3],4.+1./3)
        self.assertTrue(r['all_terminal_values_exact'])

    def test_exact_breakthrough_sample_has_entry_plateau(self):
        t,_=convert(self.t,self.raw,5.)
        np.testing.assert_array_equal(t[:3,3],np.full(3,5.))

    def test_no_breakthrough_rejected(self):
        self.raw[:,6]=0
        with self.assertRaises(AssertionError): convert(self.t,self.raw,4.)

    def test_inconsistent_threshold_rejected(self):
        for pb in (2.,6.,float('nan')):
            with self.assertRaises(AssertionError): convert(self.t,self.raw,pb)

    def test_nonmonotone_breakthrough_flags_rejected(self):
        self.raw[-1,6]=0
        with self.assertRaises(AssertionError): convert(self.t,self.raw,4.)

    def test_input_not_mutated(self):
        t=self.t.copy(); raw=self.raw.copy()
        convert(self.t,self.raw,4.)
        np.testing.assert_array_equal(t,self.t)
        np.testing.assert_array_equal(raw,self.raw)


if __name__ == '__main__': unittest.main()
