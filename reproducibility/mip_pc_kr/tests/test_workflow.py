import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workflow as W


def table_file(path, table):
    with h5py.File(path, 'w') as f:
        for name in ('fault/region_matrix', 'window_slice/saturation_region',
                     'window_slice/selected_sample_index', 'window_slice/exact_replay_seed',
                     'window_slice/local_perm_m2', 'window_slice/poro', 'window_slice/slice_indices'):
            f.create_dataset(name, data=np.arange(522).reshape(6, 87))
        group = f.require_group('fault/fluid_tables/SGOF')
        group.create_dataset('regions', data=np.arange(9, 531))
        refs = group.create_dataset('tables', shape=(522,), dtype=h5py.ref_dtype)
        for i in range(522):
            refs[i] = f.create_dataset('curves/c%03d' % i, data=table.T).ref
        f.create_dataset('metadata/pc_curve_treatment', data=np.array([65, 66], dtype=np.uint16))


class WorkflowTests(unittest.TestCase):
    def test_frozen_source_checksums(self):
        self.assertGreater(len(W.verify()['files']), 1000)

    def test_reject_path_escape(self):
        with self.assertRaises(ValueError):
            W.member('snapshot/../../outside.py')

    def test_checksum_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'input'
            p.write_bytes(b'original')
            record = dict(path=str(p), sha256=W.sha(p))
            self.assertEqual(W.input_file(record), p.resolve())
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                W.input_file(record)

    def test_never_reuse_output_or_package_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                W.fresh(tmp)
        with self.assertRaises(ValueError):
            W.fresh(W.SNAP / 'new')
        with self.assertRaises(ValueError):
            W.fresh(W.ROOT)

    def test_snapshot_tampering_and_addition_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = root / 'snapshot'
            snap.mkdir()
            f = snap / 'source.py'
            f.write_bytes(b'original')
            W.write(root / 'SOURCE_MANIFEST.json', dict(files={'snapshot/source.py':
                    dict(bytes=8, sha256=W.sha(f))}))
            with patch.object(W, 'ROOT', root), patch.object(W, 'SNAP', snap):
                W.verify()
                f.write_bytes(b'changed!')
                with self.assertRaises(ValueError):
                    W.verify()
                f.write_bytes(b'original')
                (snap / 'unexpected.py').write_bytes(b'extra')
                with self.assertRaises(ValueError):
                    W.verify()

    def test_physical_export_preserves_positive_pc(self):
        converter = W.load('qualified_table_test', W.SNAP / 'full_fault_export/build_tables.py')
        raw = np.array([[0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0],
                        [2, .2, .8, .2, 0, 0, 1, .2], [4, .8, 0, 1, 0, 0, 1, .8]], dtype=float)
        table, audit = converter.convert(raw, 1.5)
        np.testing.assert_array_equal(table[1:, 3], [2, 4])
        self.assertEqual(table[0, 3], 1.5)
        self.assertTrue(audit['entry_not_core_breakthrough'])
        with self.assertRaises(AssertionError):
            converter.convert(raw, 3.)

    def test_zero_onset_all_522_positions_and_no_original_changes(self):
        physical = np.array([[0, 0, 1, 100], [.1, .05, .8, 150],
                             [.2, .2, .6, 200], [.8, 1, 0, 1000]], dtype=float)
        zero = physical.copy()
        zero[:2, 3] = 200
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, z = root / 'physical.mat', root / 'zero.mat'
            table_file(p, physical)
            table_file(z, zero)
            before = {p: W.sha(p), z: W.sha(z)}
            spec = dict(mip=dict(path=str(p), sha256=before[p]),
                        physical=dict(path=str(p), sha256=before[p]),
                        zero=dict(path=str(z), sha256=before[z]))
            W.write(root / 'spec.json', spec)
            with contextlib.redirect_stdout(io.StringIO()):
                report = W.onset(root / 'spec.json', root / 'out')
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['records']), 522)
            self.assertTrue(report['all_522_kr_and_saturation_knots_bitwise_equal'])
            self.assertTrue(report['postbreakthrough_pc_nodes_bitwise_equal'])
            self.assertTrue(report['all_other_scientific_datasets_exact'])
            self.assertEqual(before, {p: W.sha(p), z: W.sha(z)})
            module = W.load('read_onset_test', W.SNAP / 'pc_onset/prepare_inputs.py')
            with h5py.File(root / 'out/inputs/case/candidate_specific.mat') as f:
                self.assertEqual(len(module.tables(f)), 522)
                for _, table in module.tables(f).values():
                    np.testing.assert_array_equal(table, zero)
            with self.assertRaises(FileExistsError):
                W.onset(root / 'spec.json', root / 'out')

    def test_export_adapter_uses_original_receipt_checks(self):
        planner = W.load('test_export_planner', W.SNAP / 'full_fault_export/planning.py')
        pressure = np.arange(100, dtype=float)
        saturation = np.maximum(pressure - 2., 0.) / 97. * .8
        ids, _ = planner.select_states(saturation.tolist())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curve = root / 'curve'
            curve.mkdir()
            # Synthetic records exercise packaging only, not physical acceptance.
            pc = np.column_stack([np.arange(1, 101), pressure, saturation,
                                  np.zeros((100, 3)), np.ones(100), saturation])
            names = 'sample,pressure_pa,sg_physical,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction,unused'
            np.savetxt(curve / 'pc100.csv', pc, delimiter=',', header=names, comments='', fmt='%.17g')
            kr = np.column_stack([np.array(ids) + 1, pressure[ids], saturation[ids],
                                  1 - saturation[ids] / .8, saturation[ids] / .8])
            np.savetxt(curve / 'kr.csv', kr, delimiter=',', header='sample,pressure_pa,sg_physical,krw,krg', comments='', fmt='%.17g')
            (curve / 'summary.toml').write_text('same_mip_microstates = true\nphase_solver_calls = 20\nintrinsic_solver_calls = 1\n[native_lifetime]\nnumeric_created = 10\nnumeric_released = 10\nsymbolic_created = 2\nsymbolic_released = 2\n')
            (curve / 'flow_calls.jsonl').write_text('{"q": 1.0}\n')
            artifacts = {p.name: W.sha(p) for p in curve.iterdir()}
            W.toml(curve / 'CORE_PASS.toml', dict(passed=True, entry_pressure_pa=2.5), pins=None)
            with (curve / 'CORE_PASS.toml').open('a') as f:
                f.write('\n[artifacts]\n' + ''.join(json.dumps(k) + ' = ' + json.dumps(v) + '\n' for k, v in artifacts.items()))
            result = W.export(curve, root / 'export')
            self.assertTrue(result['passed'])
            self.assertTrue(result['pc100_preserved'])
            self.assertTrue(result['selected_kr_preserved'])
            self.assertTrue((root / 'export/sgof.csv').is_file())
            (curve / 'kr.csv').write_text('tampered')
            with self.assertRaises(AssertionError):
                W.export(curve, root / 'rejected')
            self.assertFalse((root / 'rejected/EXPORT_PASS.json').exists())

    def test_zero_onset_rejects_different_sampling(self):
        table = np.array([[0, 0, 1, 100], [.1, .05, .8, 150], [.2, .2, .6, 200]], dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, z = root / 'p.mat', root / 'z.mat'
            table_file(p, table)
            table_file(z, table)
            with h5py.File(z, 'r+') as f:
                f['window_slice/exact_replay_seed'][0, 0] = 999
            spec = {k: dict(path=str(v), sha256=W.sha(v)) for k, v in dict(mip=p, physical=p, zero=z).items()}
            W.write(root / 'spec.json', spec)
            with self.assertRaises(AssertionError):
                W.onset(root / 'spec.json', root / 'out')
            self.assertFalse((root / 'out/inputs/case/pc_only_audit.json').exists())


if __name__ == '__main__':
    unittest.main()
