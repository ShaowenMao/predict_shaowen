import os
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from study import pc_nodes
from build_tables import convert


class ExtensionTests(unittest.TestCase):
    def test_pc_export_matches_existing_policy(self):
        p=np.r_[0,np.geomspace(1,1000,99)]
        sg=np.r_[np.zeros(5),np.linspace(.001,.8,90),np.repeat(.8,5)]
        raw=np.column_stack([p,sg,1-sg/.8,sg/.8,np.zeros((100,4))])
        expected,_=convert(raw,p[4]*1.01)
        actual=pc_nodes(p,sg,p[4]*1.01)
        np.testing.assert_array_equal(actual,expected[:,[0,3]])

    def test_interior_vertical_branch_is_not_silently_erased(self):
        p=np.arange(100,dtype=float);sg=np.linspace(0,.8,100);sg[40]=sg[39]
        with self.assertRaises(AssertionError):pc_nodes(p,sg,.5)

    def test_negative_saturation_or_pressure_is_rejected(self):
        p=np.arange(100,dtype=float);sg=np.linspace(0,.8,100);sg[40]=-.1
        with self.assertRaises(AssertionError):pc_nodes(p,sg,.5)

    def test_all_four_case_angle_identities(self):
        here=Path(__file__).parent
        for case,task in [('s01_c018_case07',262),('s03_c003_case08',848)]:
            for angle in (30,60):
                env=dict(os.environ,HYBRID_CASE=case,HYBRID_ANGLE=str(angle))
                code=("import study; "
                      f"assert study.CASE=={case!r} and study.TASK=={task}; "
                      f"assert study.ANGLE=={angle}; "
                      f"assert study.OUT.as_posix().endswith('/{case}/wrr{angle}'); "
                      "assert 'scratch' not in study.OUT.parts")
                subprocess.run([sys.executable,'-B','-c',code],cwd=here,env=env,check=True)

    def test_unknown_case_rejected(self):
        env=dict(os.environ,HYBRID_CASE='s03_c022_case12',HYBRID_ANGLE='30')
        p=subprocess.run([sys.executable,'-B','-c','import study'],cwd=Path(__file__).parent,
                         env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertNotEqual(p.returncode,0)

    def test_no_array_concurrency_cap(self):
        source=(Path(__file__).parent/'submit.py').read_text()
        self.assertIn("array='1-6'",source)
        self.assertNotIn("array='1-6%",source)


if __name__=='__main__':unittest.main()
