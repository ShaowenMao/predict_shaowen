from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module("phase1_analysis", HERE / "analyze_phase1_results.py")
SELECTION = load_module("phase2_selection", HERE / "select_phase2_geologies.py")


def write_manifest(path: Path) -> dict[str, list[str]]:
    header = [
        "design_version",
        "geology_id",
        "phase",
        "case_id",
        "case_type",
        "replicate_id",
        "window_id",
        "slice_id",
        "use_for_probabilistic_uq",
        "is_benchmark",
        "is_stress_test",
    ]
    case_ids: dict[str, list[str]] = {}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for geology_number in (1, 2):
            geology_id = f"s01_c{geology_number:03d}"
            cases = [
                (f"{geology_id}_IND_{replicate:03d}", "independent_full", replicate,
                 True, False, False)
                for replicate in range(1, 13)
            ] + [
                (f"{geology_id}_REP_MEDOID", "representative_medoid", 0,
                 False, True, False),
                (f"{geology_id}_LOW_STRESS", "low_state_stress", 0,
                 False, False, True),
                (f"{geology_id}_HIGH_STRESS", "high_state_stress", 0,
                 False, False, True),
            ]
            case_ids[geology_id] = [case[0] for case in cases]
            for case_id, case_type, replicate, use_uq, benchmark, stress in cases:
                for window in range(1, 7):
                    for slice_id in range(1, 88):
                        writer.writerow([
                            "independent_full_fault_v1",
                            geology_id,
                            "phase1",
                            case_id,
                            case_type,
                            replicate,
                            f"famp{window}",
                            slice_id,
                            str(use_uq).lower(),
                            str(benchmark).lower(),
                            str(stress).lower(),
                        ])
    return case_ids


def write_results(path: Path, case_ids: dict[str, list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["geology_id", "case_id", "qoi_name", "qoi_value"])
        for geology_number, (geology_id, ids) in enumerate(sorted(case_ids.items())):
            independent_start = 1.0 + 10.0 * geology_number
            values = [independent_start + offset for offset in range(12)]
            values += [independent_start + 5.0, independent_start, independent_start + 11.0]
            for case_id, value in zip(ids, values):
                writer.writerow([geology_id, case_id, "leakage_fraction", value])


class SamplingAnalysisTest(unittest.TestCase):
    def test_phase1_role_safe_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            results = root / "results.csv"
            case_ids = write_manifest(manifest)
            write_results(results, case_ids)
            roles = ANALYSIS.read_case_roles(manifest)
            ANALYSIS.validate_geology_design(roles)
            loaded = ANALYSIS.read_results(results, roles)
            summaries = ANALYSIS.summarize_geologies(roles, loaded)
            ANALYSIS.validate_analysis_rectangle(roles, summaries, 2)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(summaries[0]["n_independent"], 12)
            self.assertAlmostEqual(summaries[0]["independent_mean"], 6.5)
            self.assertAlmostEqual(summaries[0]["independent_sample_variance"], 13.0)
            self.assertAlmostEqual(summaries[0]["representative_medoid_bias"], -0.5)
            self.assertAlmostEqual(summaries[0]["stress_delta_high_minus_low"], 11.0)
            decomposition = ANALYSIS.variance_decomposition(summaries)
            self.assertEqual(len(decomposition), 1)
            self.assertAlmostEqual(
                decomposition[0]["level2_within_geology_variance"], 13.0
            )
            self.assertAlmostEqual(
                decomposition[0]["level1_geologic_design_variance"],
                50.0 - 13.0 / 12.0,
            )
            with self.assertRaises(ValueError):
                ANALYSIS.validate_analysis_rectangle(roles, summaries, 162)

            incomplete = summaries[:-1]
            with self.assertRaises(ValueError):
                ANALYSIS.validate_analysis_rectangle(roles, incomplete, 2)

    def test_category_stratified_maximin_is_deterministic(self) -> None:
        rows = [
            {"geology_id": f"g{index:02d}", "x": str(index), "y": str(index % 3)}
            for index in range(1, 9)
        ]
        vectors = SELECTION.normalize_features(rows, ["x", "y"])
        first = SELECTION.select_maximin(rows, vectors, 4)
        second = SELECTION.select_maximin(list(reversed(rows)), vectors, 4)
        first_ids = [row["geology_id"] for row, _ in first]
        second_ids = [row["geology_id"] for row, _ in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(set(first_ids)), 4)

        SELECTION.validate_selection_size(["low", "medium", "high"], 4, 12)
        with self.assertRaises(ValueError):
            SELECTION.validate_selection_size(["low", "medium"], 4, 12)
        with self.assertRaises(ValueError):
            SELECTION.validate_selection_size(
                ["low", "medium", "high", "other"], 3, 12
            )
        with self.assertRaises(ValueError):
            SELECTION.validate_selection_size(["low", "low", "high"], 4, 12)


if __name__ == "__main__":
    unittest.main()
