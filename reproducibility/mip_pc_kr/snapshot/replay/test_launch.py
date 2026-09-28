import unittest
from launch import resource_for,batch_records,seconds_allowed,cutoff_epoch

class Tests(unittest.TestCase):
    def test_resources(self):
        self.assertEqual([resource_for(x) for x in (30000,30001,130000,130001,150000)],[4,5,5,6,6])
        self.assertIsNone(resource_for(150001));self.assertIsNone(resource_for(0))
    def test_boundaries(self):
        rs=[dict(case_key='a',slot=i,assignment={'window':'famp1'},memory_gib=5) for i in range(87)]
        bs=batch_records(rs)
        self.assertEqual([len(b['records']) for b in bs],[24,24,24,15])
        self.assertEqual([r['slot'] for b in bs for r in b['records']],list(range(87)))
    def test_groups(self):
        rs=[dict(case_key=c,slot=i,assignment={'window':w},memory_gib=m) for c,w,m in [('a','famp1',4),('a','famp2',6),('b','famp1',5)] for i in range(2)]
        self.assertEqual(len(batch_records(rs)),3)
    def test_maintenance(self):
        self.assertEqual(seconds_allowed(0,20000),10800)
        self.assertEqual(seconds_allowed(0,5000),3180)
        self.assertEqual(seconds_allowed(20000,20000),0)
        self.assertEqual(cutoff_epoch('2026-09-15T00:00:00'),1789444800.0)

if __name__=='__main__':unittest.main()
