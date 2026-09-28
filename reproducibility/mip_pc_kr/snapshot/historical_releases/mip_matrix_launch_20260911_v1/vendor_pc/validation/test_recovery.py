import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import core3_recovery as recovery
import study


class RecoveryTests(unittest.TestCase):
    def test_compute_guard(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(AssertionError):recovery.run(Path('/unrelated'))

    def test_output_scope_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):recovery.prepare(Path(d)/'outside')

    def test_numerical_source_guard(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'new'
            for r in [old,new]:
                (r/'src').mkdir(parents=True)
                (r/'src/core.jl').write_text('unchanged numerical fixture\n')
            files={'src/core.jl':study.sha(old/'src/core.jl')}
            identity=study.identity(files)
            study.write(old/'source_lock.json',{'files':files,'sha256':identity})
            with patch.object(recovery,'OLD_SOURCE',old),patch.object(recovery,'OLD_HASH',identity),patch.object(study,'ROOT',new):
                recovery.verify_numerics()
                (new/'src/core.jl').write_text('changed\n')
                with self.assertRaises(AssertionError):recovery.verify_numerics()


if __name__=='__main__':unittest.main()
