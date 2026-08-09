#!/usr/bin/env python3
"""Integration tests for the sampling-to-upscaling contract."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / ".git").exists())


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_module("sampling_adapter", HERE / "adapt_manifest_for_upscaling.py")
FREEZER = HERE / "build_replay_upscaling_freeze.py"
ASSEMBLER = load_module(
    "case_assembler",
    REPO_ROOT
    / "examples"
    / "pc_upscaling_pilot"
    / "engaging"
    / "production"
    / "assemble_geology_case_inputs.py",
)
METHOD_CONFIG = (
    REPO_ROOT
    / "examples"
    / "pc_upscaling_pilot"
    / "engaging"
    / "production"
    / "production_method_config.toml"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def canonical_rows(predict_root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    geology_id = "s01_c001"
    checkpoints: dict[str, str] = {}
    for window_number in range(1, 7):
        window = f"famp{window_number}"
        relative = f"data/scenario_01/{window}/case_001/predict_runs.mat"
        path = predict_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{window}\n".encode("utf-8"))
        checkpoints[window] = relative

    specs = [
        ("phase1", f"{geology_id}_IND_{replicate:03d}", "independent_full", replicate)
        for replicate in range(1, 13)
    ] + [
        ("phase1", f"{geology_id}_REP_MEDOID", "representative_medoid", 0),
        ("phase1", f"{geology_id}_LOW_STRESS", "low_state_stress", 0),
        ("phase1", f"{geology_id}_HIGH_STRESS", "high_state_stress", 0),
    ]
    rows: list[dict[str, object]] = []
    for phase, case_id, case_type, replicate_id in specs:
        roles = {
            "independent_full": (True, False, False),
            "representative_medoid": (False, True, False),
            "low_state_stress": (False, False, True),
            "high_state_stress": (False, False, True),
        }[case_type]
        for window_number in range(1, 7):
            window = f"famp{window_number}"
            relative = checkpoints[window]
            checkpoint = predict_root / relative
            seed_base = 100_000_000 + window_number * 1_000_000
            for slice_id in range(1, 88):
                if case_type == "independent_full":
                    source_row = (replicate_id * 137 + window_number * 31 + slice_id * 17) % 2000 + 1
                    realization_id = source_row
                    sampling_seed = replicate_id * 100 + window_number
                    source_library = "full_distribution"
                    selection_method = "uniform_with_replacement"
                else:
                    source_row = {
                        "representative_medoid": 1000,
                        "low_state_stress": 200,
                        "high_state_stress": 1800,
                    }[case_type]
                    realization_id = source_row
                    sampling_seed = 0
                    source_library = {
                        "representative_medoid": "full_distribution",
                        "low_state_stress": "low_state",
                        "high_state_stress": "high_state",
                    }[case_type]
                    selection_method = "exact_logk_medoid"
                replay_seed = seed_base + realization_id - 1
                values = [-6.0 + source_row / 1000, -5.5 + source_row / 1000, -5.0 + source_row / 1000]
                rows.append({
                    "schema_version": "full_fault_sampling_manifest_v1",
                    "design_version": "independent_full_fault_v1",
                    "geology_id": geology_id,
                    "phase": phase,
                    "case_id": case_id,
                    "case_type": case_type,
                    "replicate_id": replicate_id,
                    "window_id": window,
                    "slice_id": slice_id,
                    "source_library": source_library,
                    "source_library_row": source_row,
                    "predict_realization_id": realization_id,
                    "predict_seed": replay_seed,
                    "exact_replay_seed": replay_seed,
                    "sampling_seed": sampling_seed,
                    "sampling_seed_method": "fixture",
                    "selection_method": selection_method,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_relative_path": relative,
                    "checkpoint_hash": sha256_file(checkpoint),
                    "level2_state_path": "fixture.mat",
                    "code_commit": "a" * 40,
                    "code_dirty": "false",
                    "method_config_hash": "b" * 64,
                    "predict_code_commit": "c" * 40,
                    "predict_method_config_hash": "d" * 64,
                    "log_kxx": values[0],
                    "log_kyy": values[1],
                    "log_kzz": values[2],
                    "perm_kxx_md": 10**values[0],
                    "perm_kyy_md": 10**values[1],
                    "perm_kzz_md": 10**values[2],
                    "use_for_probabilistic_uq": str(roles[0]).lower(),
                    "is_benchmark": str(roles[1]).lower(),
                    "is_stress_test": str(roles[2]).lower(),
                })
    return rows, checkpoints


class SamplingIntegrationTest(unittest.TestCase):
    def test_case_aliases(self) -> None:
        self.assertEqual(ADAPTER.production_case_number("independent_full", 52), 52)
        self.assertEqual(ADAPTER.production_case_number("representative_medoid", 0), 101)
        self.assertEqual(ADAPTER.production_case_number("low_state_stress", 0), 102)
        self.assertEqual(ADAPTER.production_case_number("high_state_stress", 0), 103)
        with self.assertRaises(ValueError):
            ADAPTER.production_case_number("independent_full", 53)

    def test_assembled_sampling_identity(self) -> None:
        assignment = {
            "sampling_case_id": "s01_c001_IND_007",
            "phase": "phase1",
            "case_type": "independent_full",
            "replicate_id": "7",
            "use_for_probabilistic_uq": "true",
            "is_benchmark": "false",
            "is_stress_test": "false",
            "design_version": "independent_full_fault_v1",
            "adapter_schema_version": ADAPTER.SCHEMA_VERSION,
            "sampling_code_commit": "a" * 40,
            "sampling_method_config_hash": "b" * 64,
            "predict_code_commit": "c" * 40,
            "predict_method_config_hash": "d" * 64,
            "checkpoint_sha256": "e" * 64,
            "checkpoint_relative_path": "data/famp1/predict_runs.mat",
            "predict_realization_id": "11",
            "draw_seed": "1234",
        }
        identity = ASSEMBLER.sampling_identity(assignment)
        self.assertEqual(identity["SamplingCaseId"], "s01_c001_IND_007")
        self.assertEqual(identity["SamplingCaseType"], "independent_full")
        self.assertEqual(identity["SamplingReplicateId"], 7)
        self.assertEqual(identity["PredictRealizationId"], 11)
        self.assertEqual(identity["SamplingSeed"], 1234)

    def test_phase1_adapter_and_restartable_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predict_root = root / "predict"
            rows, _ = canonical_rows(predict_root)
            manifest = root / "phase1_manifest.csv"
            write_csv(manifest, list(rows[0]), rows)
            geology_catalog = root / "geology_catalog.csv"
            geology_row = {
                "geology_id": "s01_c001",
                "scenario_index": 1,
                "scenario_label": "scenario_01",
                "scenario_name": "fixture geology",
                "case_index": 1,
                "case_label": "case_001",
                "faulting_depth_m": 50,
                "sand_vcl": 0.1,
                "clay_vcl": 0.4,
            }
            write_csv(geology_catalog, list(geology_row), [geology_row])

            adapted_rows = ADAPTER.adapt_manifest(manifest, geology_catalog)
            self.assertEqual(len(adapted_rows), 15 * 6 * 87)
            aliases = {int(row["case_id"]) for row in adapted_rows}
            self.assertEqual(aliases, set(range(1, 13)) | {101, 102, 103})
            adapted = root / "adapted.csv"
            ADAPTER.write_adapted_output(
                adapted_rows, adapted, manifest, geology_catalog, overwrite=False
            )
            adapted_copy = root / "adapted_copy.csv"
            ADAPTER.write_adapted_output(
                adapted_rows, adapted_copy, manifest, geology_catalog, overwrite=False
            )
            self.assertEqual(adapted.read_bytes(), adapted_copy.read_bytes())

            freeze1 = root / "freeze1"
            freeze2 = root / "freeze2"
            command = [
                sys.executable,
                str(FREEZER),
                "--sampling-csv",
                str(adapted),
                "--predict-root",
                str(predict_root),
                "--method-config",
                str(METHOD_CONFIG),
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(freeze1),
                "--tasks-per-shard",
                "5000",
                "--allow-dirty",
                "--allow-sampling-code-mismatch",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            command[command.index(str(freeze1))] = str(freeze2)
            subprocess.run(command, check=True, capture_output=True, text=True)

            for relative in (
                "manifests/assignment_to_task.csv",
                "manifests/unique_replay_pc_tasks.csv",
                "manifests/task_shards.csv",
                "inventory/selected_predict_checkpoints_sha256.csv",
            ):
                self.assertEqual(
                    (freeze1 / relative).read_bytes(), (freeze2 / relative).read_bytes()
                )
            metadata = json.loads((freeze1 / "freeze_metadata.json").read_text())
            self.assertEqual(metadata["phases"], ["phase1"])
            self.assertEqual(metadata["geology_count"], 1)
            self.assertEqual(metadata["case_count"], 15)
            self.assertEqual(metadata["assignment_count"], 15 * 6 * 87)
            self.assertLess(metadata["unique_replay_pc_task_count"], metadata["assignment_count"])

    def test_rejects_wrong_checkpoint_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows, _ = canonical_rows(root / "predict")
            rows[0]["exact_replay_seed"] = int(rows[0]["exact_replay_seed"]) + 1
            manifest = root / "manifest.csv"
            write_csv(manifest, list(rows[0]), rows)
            geology = root / "geology.csv"
            geology_row = {
                "geology_id": "s01_c001",
                "scenario_index": 1,
                "scenario_label": "scenario_01",
                "scenario_name": "fixture",
                "case_index": 1,
                "case_label": "case_001",
                "faulting_depth_m": 50,
                "sand_vcl": 0.1,
                "clay_vcl": 0.4,
            }
            write_csv(geology, list(geology_row), [geology_row])
            with self.assertRaisesRegex(ValueError, "exact_replay_seed"):
                ADAPTER.adapt_manifest(manifest, geology)


if __name__ == "__main__":
    unittest.main(verbosity=2)
