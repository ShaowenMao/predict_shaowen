import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import study
from export_audit import audit,sample,compare


def curve():return [[float(i),i/110,1-i/110,i/110,0,0,int(i>3),i/110] for i in range(100)]


class Tests(unittest.TestCase):
    def test_complete_mapping(self):
        a=[dict(geology_id='s03_c022',case_id='12',window=f'famp{w}',slice_index=str(s)) for w in range(1,7) for s in range(1,88)]
        self.assertEqual(study.coverage(a[::-1]),a)
        for b in [a[:-1],a+[a[0]],a[:-1]+[a[0]]]:
            with self.assertRaises(AssertionError):study.coverage(b)
        a[0]['case_id']='1'
        with self.assertRaises(AssertionError):study.coverage(a)

    def test_new_output_only(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'receipt';study.write(p,{'ok':1})
            with self.assertRaises(FileExistsError):study.write(p,{'ok':2})
            self.assertEqual(study.read(p),{'ok':1})

    def test_scope_protection(self):
        for p in [study.PROD,study.ROOT,Path('/tmp/a')]:
            with self.assertRaises(AssertionError):study.scope(p)

    def test_export_no_mutation(self):
        c=curve();old=copy.deepcopy(c);a=audit(c)
        self.assertEqual(c,old);self.assertTrue(a['scalar_pc_export_ready'])
        self.assertFalse(a['reservoir_export_approved'])
        self.assertIn('positive_terminal_brine_Kr_do_not_force_zero',a['flags'])

    def test_duplicate_pc_is_not_silently_removed(self):
        c=curve();c[2][1]=c[1][1]
        self.assertIn('Pc_multivalued_at_repeated_Sg',audit(c)['flags'])
        self.assertIsNone(sample(c,c[1][1],0,True))

    def test_no_extrapolation(self):
        c=curve();self.assertIsNone(sample(c,.99,3));self.assertIsNone(sample(c,-.1,3))

    def test_numerical_failure_is_not_export_flag(self):
        for col,value in [(1,1.2),(2,-.1),(0,float('nan'))]:
            c=curve();c[4][col]=value
            with self.assertRaises(AssertionError):audit(c)

    def test_comparison_identity(self):
        c=curve();d=compare(c,c,c)
        for x in d['common_sg_comparisons']:
            if x['pc_ratio'] is not None:self.assertAlmostEqual(x['pc_ratio'],1)
            if x['delta_krw'] is not None:self.assertEqual(x['delta_krw'],0)
            if x['delta_krg'] is not None:self.assertEqual(x['delta_krg'],0)

    def test_completed_resume_and_changed_identity(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);a=out/'a';a.mkdir();(a/'curve').write_bytes(b'unchanged')
            d=dict(calculation_key='a',numerical_checks_passed=True,attempt=str(a),outputs=study.hashes(a))
            study.write(out/'accepted/slot001.json',d)
            self.assertEqual(study.accepted(out,dict(slot=1,calculation_key='a')),d)
            with self.assertRaises(AssertionError):study.accepted(out,dict(slot=1,calculation_key='b'))
            (a/'curve').write_bytes(b'changed')
            with self.assertRaises(AssertionError):study.accepted(out,dict(slot=1,calculation_key='a'))


if __name__=='__main__':unittest.main()
