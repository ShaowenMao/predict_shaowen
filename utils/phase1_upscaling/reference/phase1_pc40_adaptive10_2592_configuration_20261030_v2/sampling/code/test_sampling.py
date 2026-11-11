import hashlib
import unittest
from unittest.mock import patch

import generate_sampling as g


class SamplingTests(unittest.TestCase):
    def test_original_golden_vector(self):
        s = g.stable_seed(1729, 's02_c022', 1, 'famp1', g.OLD_VERSION)
        self.assertEqual(s, 2037554249464577736)
        self.assertEqual(g.uniform_index(s, 1, version=g.OLD_VERSION), 1712)

    def test_identity_separation(self):
        seeds = {g.stable_seed(g.BASE_SEED, f's01_c001_pc{pc:02d}_ca40', r, w)
                 for pc in (4, 6) for r in range(1, 9) for w in g.WINDOWS}
        self.assertEqual(len(seeds), 96)
        self.assertTrue(all(0 < x < 2**63 for x in seeds))

    def test_utf8_length_prefix_and_unambiguous_fields(self):
        self.assertEqual(g.payload('ab', 3), b'2:ab|1:3|')
        self.assertEqual(g.payload('\u00e9'), b'2:\xc3\xa9|')
        self.assertNotEqual(g.payload('a', 'bc'), g.payload('ab', 'c'))

    def test_reproducible_with_replacement(self):
        s = g.stable_seed(g.BASE_SEED, 's01_c001_pc04_ca40', 1, 'famp1')
        a = [g.uniform_index(s, j, n=3) for j in range(1, 20)]
        self.assertEqual(a, [g.uniform_index(s, j, n=3) for j in range(1, 20)])
        self.assertTrue(all(1 <= v <= 3 for v in a))
        self.assertLess(len(set(a)), len(a))

    def test_rejection_boundary(self):
        class Digest:
            def __init__(self, n): self.n = n
            def digest(self): return self.n.to_bytes(8, 'big') + b'\0' * 24
        # For n=3, uint64 value zero must be rejected, not mapped modulo 3.
        with patch.object(hashlib, 'sha256', side_effect=[Digest(0), Digest(5)]) as h:
            self.assertEqual(g.uniform_index(1, 1, n=3), 3)
            self.assertEqual(h.call_count, 2)

    def test_invalid_sampling_dimensions(self):
        for n, draw in [(0, 1), (-1, 1), (2**64, 1), (3, 0)]:
            with self.assertRaises(ValueError): g.uniform_index(1, draw, n=n)


if __name__ == '__main__':
    unittest.main()
