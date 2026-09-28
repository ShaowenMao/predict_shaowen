import unittest
from dispatch_checks import memory_gib,unique_job,can_admit


class Tests(unittest.TestCase):
    def test_per_cpu(self):self.assertEqual(memory_gib('2048Mc',8),16)
    def test_per_node(self):self.assertEqual(memory_gib('18Gn',8),18)
    def test_nonfinite(self):
        with self.assertRaises(ValueError):memory_gib('nanG',1)
    def test_duplicate_live_history(self):self.assertEqual(unique_job('a',['17|a','17|a|COMPLETED']), '17')
    def test_two_jobs(self):
        with self.assertRaises(ValueError):unique_job('a',['17|a','18|a'])
    def test_missing_is_uncertain(self):self.assertIsNone(unique_job('a',['17|b']))
    def test_no_headroom(self):self.assertFalse(can_admit(500,0));self.assertFalse(can_admit(0,1983))
    def test_headroom(self):self.assertTrue(can_admit(499,1982))


if __name__=='__main__':unittest.main()
