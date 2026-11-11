from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import study

class Tests(unittest.TestCase):
    def test_compute_guard(self):
        with patch.dict(study.os.environ,{},clear=True):
            with self.assertRaisesRegex(AssertionError,'Compute-node'):study.core(Path('/tmp/no'),7)

    def test_exclusive_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'receipt.json';study.write(p,dict(passed=True))
            with self.assertRaises(FileExistsError):study.write(p,{})

    def test_engine_is_frozen(self):
        self.assertEqual(study.sha(study.ROOT/'vendor/StandaloneMIP.jl'),study.ENGINE_HASH)

    def test_isolated_project_is_on_load_path(self):
        shell=(study.ROOT/'validation/job.sh').read_text()
        self.assertIn('JULIA_LOAD_PATH="@:@stdlib"',shell)
        self.assertNotIn('@v#',shell)
        self.assertIn("'--project='+str(ROOT)",Path(study.__file__).read_text())

    def test_no_phase_flow_or_production_promotion(self):
        exporter=(study.ROOT/'validation/export_core.m').read_text()
        self.assertNotIn('phase_conductance(',exporter)
        self.assertNotIn('solver_worker(',Path(study.__file__).read_text())
        shell=(study.ROOT/'validation/job.sh').read_text()
        self.assertNotIn('--exclusive',shell)
        self.assertNotIn('--array',shell)
        for p in (study.ROOT/'src').glob('*.jl'):
            self.assertNotIn('run(`',p.read_text())
            self.assertNotIn('ccall',p.read_text())

    def test_immutable_engine_covers_all_events(self):
        source=(study.ROOT/'vendor/StandaloneMIP.jl').read_text()
        self.assertIn('for i in first:last;step!(m,s,m.events[i],w);end',source)
        self.assertIn('w.kw[i]>0',source)

if __name__=='__main__':unittest.main()
