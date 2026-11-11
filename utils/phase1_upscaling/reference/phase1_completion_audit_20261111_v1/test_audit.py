import copy
import unittest
from audit_completion import METHOD, validate_cases


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.cases = ['case%04d' % i for i in range(2592)]
        self.markers = {c: dict(case_id=c, positions=522, method=METHOD, plan_sha256='p') for c in self.cases}
        self.seals = {c: dict(plan_sha256='p') for c in self.cases}

    def test_complete(self):
        self.assertEqual(validate_cases(self.cases, self.cases, self.markers, self.seals), 1353024)

    def test_invalid_fields(self):
        for field, value in [('case_id', 'wrong'), ('positions', 521), ('method', 'wrong'), ('plan_sha256', 'wrong')]:
            with self.subTest(field=field):
                markers = copy.deepcopy(self.markers)
                markers[self.cases[0]][field] = value
                with self.assertRaises(RuntimeError):
                    validate_cases(self.cases, self.cases, markers, self.seals)

    def test_duplicate_catalog(self):
        with self.assertRaises(RuntimeError):
            validate_cases(self.cases[:-1] + self.cases[:1], self.cases, self.markers, self.seals)

    def test_missing(self):
        self.markers.pop(self.cases[0])
        with self.assertRaises(RuntimeError):
            validate_cases(self.cases, self.cases, self.markers, self.seals)

    def test_extra(self):
        self.markers['extra'] = self.markers[self.cases[0]]
        with self.assertRaises(RuntimeError):
            validate_cases(self.cases, self.cases, self.markers, self.seals)

    def test_scope_changed(self):
        with self.assertRaises(RuntimeError):
            validate_cases(self.cases, self.cases[:-1] + ['other'], self.markers, self.seals)


if __name__ == '__main__':
    unittest.main()
