import unittest
from accounting import terminal_jobs

ACTUAL='28650733|28650395_0|COMPLETED|0:0\n28650734|28650395_1|COMPLETED|0:0\n28650735|28650395_2|COMPLETED|0:0\n28650395|28650395_3|COMPLETED|0:0\n28666059|28665837_0|FAILED|1:0\n28666060|28665837_1|FAILED|1:0\n28665837|28665837_2|FAILED|1:0\n'
EXPECTED=['28665837','28666059','28666060','28650395']
class Accounting(unittest.TestCase):
    def test_actual_raw_ids(self):
        d=terminal_jobs(ACTUAL,EXPECTED)
        self.assertEqual(d['28666059'],'28665837_0')
        self.assertEqual(d['28650395'],'28650395_3')
    def test_missing_raw_task(self):
        with self.assertRaises(RuntimeError):terminal_jobs(ACTUAL.replace('28666059|28665837_0|FAILED|1:0\n',''),EXPECTED)
    def test_live_task(self):
        with self.assertRaises(RuntimeError):terminal_jobs(ACTUAL.replace('28666059|28665837_0|FAILED','28666059|28665837_0|RUNNING'),EXPECTED)
    def test_duplicate_task(self):
        with self.assertRaises(RuntimeError):terminal_jobs(ACTUAL+ACTUAL,EXPECTED)
    def test_array_parent_not_permission_for_missing_children(self):
        with self.assertRaises(RuntimeError):terminal_jobs('28665837|28665837|COMPLETED|0:0',EXPECTED)
if __name__=='__main__':unittest.main()
