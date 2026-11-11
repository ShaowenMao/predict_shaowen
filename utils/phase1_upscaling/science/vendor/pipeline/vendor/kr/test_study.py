import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import study


class Tests(unittest.TestCase):
    def test_compute_guard(self):
        with patch.dict(study.os.environ,{},clear=True):
            with self.assertRaisesRegex(AssertionError,'Compute-node'):study.core(Path('/tmp/no'),7)

    def test_receipts_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json';study.write(path,{'ok':True})
            with self.assertRaises(FileExistsError):study.write(path,{})

    def test_reference_gates(self):
        a={(0,'absolute'):dict(q=1.,accepted=True,reference_matches=True),
           (1,'gas'):dict(q=0.,accepted=True,reference_matches=True)}
        self.assertEqual(max(x['relative_difference'] for x in study.compare_pair(a,copy.deepcopy(a))),0)
        for change in ['missing','zero','difference','nan','acceptance']:
            b=copy.deepcopy(a)
            if change=='missing':b.pop((1,'gas'))
            elif change=='zero':b[1,'gas']['q']=1e-200
            elif change=='difference':b[0,'absolute']['q']=1.001
            elif change=='nan':b[0,'absolute']['q']=float('nan')
            else:b[0,'absolute']['accepted']=False
            with self.assertRaises(AssertionError):study.compare_pair(a,b)

    def test_no_mip_or_original_edits(self):
        source=(study.ROOT/'study.py').read_text()
        self.assertNotIn('prepare_states(',source)
        self.assertNotIn('cell_mip_advance(',source)
        self.assertIn("'direct'",(study.ROOT/'export_packet.m').read_text())
        kernel=(study.ROOT/'build_bridge.py').read_text()
        self.assertIn("source=numeric/'vendor'",kernel)
        self.assertNotIn('write_text',kernel)


if __name__=='__main__':unittest.main()
