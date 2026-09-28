import copy
from pathlib import Path
import tempfile
import unittest
import study


class StudyTests(unittest.TestCase):
    def assignments(self):
        return [dict(geology_id='s01_c018', case_id='7', window=f'famp{w}', slice_index=str(s))
                for w in range(1, 7) for s in range(1, 88)]

    def test_matrix(self):
        v = study.variants()
        self.assertEqual(len(v), 12)
        self.assertEqual(len({x['variant'] for x in v}), 12)
        self.assertEqual(len({(x['case'], x['angle']) for x in v}), 6)
        self.assertTrue(all(not x['production_promotion'] for x in v))

    def test_coverage_and_sort(self):
        v = self.assignments()
        self.assertEqual(study.coverage(v[::-1], 's01_c018', 7), v)

    def test_duplicate_rejected(self):
        v = self.assignments()
        v[-1] = copy.copy(v[0])
        with self.assertRaises(AssertionError):
            study.coverage(v, 's01_c018', 7)

    def test_wrong_draw_case_rejected(self):
        v = self.assignments()
        v[21]['case_id'] = '8'
        with self.assertRaises(AssertionError):
            study.coverage(v, 's01_c018', 7)

    def test_missing_core_rejected(self):
        with self.assertRaises(AssertionError):
            study.coverage(self.assignments()[:-1], 's01_c018', 7)

    def test_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'value.json'
            study.write(p, dict(a=1))
            h = study.sha(p)
            study.check(tmp, {'value.json': h})
            p.write_text('changed')
            with self.assertRaises(AssertionError):
                study.check(tmp, {'value.json': h})

    def test_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'value.json'
            study.write(p, dict(a=1))
            with self.assertRaises(FileExistsError):
                study.write(p, dict(a=2))

    def test_independent_acceptance_names(self):
        names = {str(study.acceptance_path(c, a, 1)) for _, c, _ in study.CASES for a in (30, 60)}
        self.assertEqual(len(names), 6)
        self.assertTrue(all('comparative_3case_' in p for p in names))


if __name__ == '__main__':
    unittest.main()
