"""Fast deterministic checks; optional hash-checked Core 172 regression data."""
import json
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import data


class WorkflowTests(unittest.TestCase):
    def fixture(self):
        core = dict(n=2, phi=np.array([.2, .4]), volume=np.ones(2),
            material=np.array([1, 2]), pressures=np.array([0., 1.]))
        sg = np.array([[0., 0.], [.6, .3]])
        curve = [[0, 0, 1, 0, 0, 0, 0, 0], [1, .4, .1, .3, 0, 0, 1, 1]]
        rec = dict(curve=curve, summary=dict(event_counts=[0, 2]))
        hist = pd.DataFrame(dict(sample=[1, 2], pressure_pa=[0., 1.], sg_physical=[0., .4],
            trapped_pv_fraction=[0., 0.], immobile_pv_fraction=[0., 0.],
            gas_breakthrough=[0., 1.], invaded_pv_fraction=[0., 1.], events=[0, 2]))
        return core, rec, hist, sg

    def test_pore_volume_weighting(self):
        curve, means, errors = data.verify_arrays(*self.fixture())
        np.testing.assert_allclose(means[1], [.6, .3])
        self.assertEqual(curve[1, 1], .4)

    def test_changed_saturation_rejected(self):
        core, rec, hist, sg = self.fixture()
        sg[1, 0] += .01
        with self.assertRaises(AssertionError):
            data.verify_arrays(core, rec, hist, sg)

    def test_changed_history_rejected(self):
        core, rec, hist, sg = self.fixture()
        hist.loc[1, "trapped_pv_fraction"] = .1
        with self.assertRaises(AssertionError):
            data.verify_arrays(core, rec, hist, sg)

    def test_changed_events_rejected(self):
        core, rec, hist, sg = self.fixture()
        hist.loc[1, "events"] = 3
        with self.assertRaises(AssertionError):
            data.verify_arrays(core, rec, hist, sg)

    def test_camera_and_material_partition(self):
        core = dict(n=8, material=np.array([1, 2] * 4))
        grid, selections = data.build_grid(core, (2, 2, 2), [1., 2., 3.])
        self.assertEqual(sum(s.n_cells for s in selections.values()), 8)
        axes = data.local_axes_from_transformed_grid(grid)
        np.testing.assert_allclose(axes["x"], [.5, -np.sqrt(3)/2, 0], atol=1e-12)
        np.testing.assert_allclose(axes["z"], [0, 0, 1], atol=1e-12)
        self.assertEqual(data.outline(grid).shape, (12, 2, 3))

    def test_selection_has_seven_other_cores(self):
        selected = json.loads(Path(__file__).with_name("selection.json").read_text())
        keys = [f"{risk}_{slot:03d}" for risk, item in selected.items() for slot in item["cores"]]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(set(keys) - {"high_172"}), 7)

    @unittest.skipUnless(os.environ.get("MIPVIZ_BUNDLE") and os.environ.get("MIPVIZ_BASELINE"), "optional saved-data regression")
    def test_approved_core172_saved_data(self):
        bundle = Path(os.environ["MIPVIZ_BUNDLE"]) / "high_172"
        baseline = Path(os.environ["MIPVIZ_BASELINE"])
        entry = json.loads((bundle.parent / "manifest.json").read_text())["cores"][-1]
        self.assertEqual(entry["key"], "high_172")
        for name, digest in entry["files"].items():
            self.assertEqual(data.sha(bundle / name), digest)
        core = data.read_core(bundle / "raw.bin")
        rec = json.loads((bundle / "record.json").read_text())
        with (baseline / "saturation_states.bin").open("rb") as f:
            n, nt = np.fromfile(f, "<i8", 2)
            sg = np.fromfile(f, "<f8").reshape(nt, n)
        curve, means, errors = data.verify_arrays(core, rec, pd.read_csv(baseline / "reconstructed_history.csv"), sg)
        expected = pd.read_csv(baseline / "frame_values.csv")
        np.testing.assert_allclose(means, expected[["sand_mean_sg", "clay_mean_sg"]], rtol=1e-12, atol=1e-12)
        data.geometry(core, bundle / "geometry.bin", entry["files"]["raw.bin"])


if __name__ == "__main__":
    unittest.main()
