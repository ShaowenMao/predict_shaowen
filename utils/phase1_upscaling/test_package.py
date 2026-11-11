import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import capture_reference as capture
from verify_package import verify

ROOT = Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    def test_real_snapshot_integrity(self):
        result = verify(ROOT)
        self.assertGreater(result['verified_files'], 600)
        self.assertEqual(result['packages'], 13)
        self.assertFalse(result['numerical_work_run'])
        self.assertFalse(result['external_data_verified'])

    def test_portable_safe_relative_paths(self):
        self.assertEqual(capture.relative('profiles/a.json'), Path('profiles/a.json'))
        for name in ('', '../x', '/x', 'C:/x', 'x\\y', 'x/../../y'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                capture.relative(name)

    def test_refuses_snapshot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'reference').mkdir()
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                capture.capture(root.parent / 'not_a_source', root)

    def test_refuses_destination_in_source_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(ValueError, 'outside'):
                capture.capture(root, root / 'new')

    def test_bulk_configuration_data_not_selected(self):
        self.assertFalse(capture.config_source('sampling/phase1/sampled_fields.h5'))
        self.assertFalse(capture.config_source('sampling/phase1/manifests/scenario.csv.gz'))
        self.assertFalse(capture.config_source('physics/prepared/pilot/raw.bin'))
        self.assertTrue(capture.config_source('profiles/pc04.json'))
        self.assertTrue(capture.config_source('sampling/code/generate.py'))

    def test_checksum_corruption_rejected(self):
        with patch('verify_package.sha', return_value='wrong'), self.assertRaisesRegex(ValueError, 'changed'):
            verify(ROOT)

    def test_manifest_traversal_rejected_before_file_read(self):
        original = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
        original['files'] = {'../escape': 'fake'}
        with patch('verify_package.json.loads', return_value=original), self.assertRaisesRegex(ValueError, 'Unsafe'):
            verify(ROOT)

    def test_only_configuration_bulk_files_omitted(self):
        manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
        for name, entry in manifest['packages'].items():
            if name != capture.CONFIG:
                self.assertEqual(entry['omitted_locked_files'], {})


if __name__ == '__main__':
    unittest.main()
