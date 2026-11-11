import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import study


class Tests(unittest.TestCase):
    def test_compute_guard(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(AssertionError):study.run(Path('/unrelated'),7)

    def test_scope_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):study.prepare(Path(d)/'outside')

    def test_receipts_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';study.write(p,{'original':True})
            with self.assertRaises(FileExistsError):study.write(p,{'original':False})
            self.assertTrue(study.read(p)['original'])

    def test_copied_components(self):
        for folder,digest in [('pc',study.PC_HASH),('kr',study.KR_HASH)]:
            root=study.ROOT/'vendor'/folder;lock=study.read(root/'source_lock.json')
            self.assertEqual(lock['sha256'],digest)
            for name,sha in lock['files'].items():self.assertEqual(study.sha(root/name),sha)

    def test_runtime_has_no_reference_curve_input(self):
        code=(study.ROOT/'FaultCoreUpscaling.jl').read_text()
        self.assertNotIn('expected.csv',code)
        self.assertNotIn('qa_reference',code)
        self.assertNotIn('matlab',code.lower())
        self.assertIn('krw=qw/qa;krg=qg/qa',code)


if __name__=='__main__':unittest.main()
