#!/usr/bin/env python3
"""Tests for restartable phase-production status and chunk planning."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
STATUS_PATH = HERE.parent / "production" / "phase_production_status.py"
REPO_ROOT = HERE.parents[4]
PHASE_LAUNCHER = HERE.parent / "production" / "submit_independent_full_fault_phase.sh"
CHECKPOINT_WORKER = (
    REPO_ROOT
    / "examples"
    / "pc_upscaling_pilot"
    / "engaging"
    / "production"
    / "run_checkpoint_replay_pc.sh"
)
KR_WORKER = CHECKPOINT_WORKER.with_name("run_case_dynamic_kr.sh")
FINALIZE_CASE_PATH = CHECKPOINT_WORKER.with_name("finalize_case_kr.py")
VERIFY_CASE_PATH = CHECKPOINT_WORKER.with_name("verify_case_completion.py")


def load_status_module():
    spec = importlib.util.spec_from_file_location("phase_status", STATUS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(STATUS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATUS = load_status_module()
FINALIZE_CASE = load_module("finalize_case_kr", FINALIZE_CASE_PATH)
VERIFY_CASE = load_module("verify_case_completion", VERIFY_CASE_PATH)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_marker(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PhaseProductionStatusTests(unittest.TestCase):
    def test_finalizer_rejects_stale_done_marker_contract(self) -> None:
        expected = {
            "status": "complete",
            "geology_id": "s01_c001",
            "case_id": 1,
        }
        stale = dict(expected)
        stale["reservoir_ready_validation"] = {"schema_version": "1.7"}
        self.assertFalse(FINALIZE_CASE.current_done_marker_matches(stale, expected))

        current = dict(expected)
        current["reservoir_ready_validation"] = {
            "schema_version": "1.7",
            "assignment_metadata_explicit": True,
        }
        self.assertTrue(FINALIZE_CASE.current_done_marker_matches(current, expected))

    def test_legacy_case_export_remains_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qa_path = Path(temporary) / "reservoir_ready_qa_summary.csv"
            method_hash = "a" * 64
            header = [
                "GeologyId",
                "Level3CaseId",
                "WindowCount",
                "SliceCount",
                "PcCurveCount",
                "KrCurveCount",
                "PorosityCount",
                "PermeabilityCellCount",
                "PermeabilityComponentCount",
                "SwiMedoidSelectionCount",
                "MinUpscaledPorosity",
                "MaxUpscaledPorosity",
                "MinPermeabilityMD",
                "MaxSourceLogPermeabilityMismatch",
                "MaxSelectedSampleIndexMismatch",
                "MaxReplaySeedMismatch",
                "MaxPorosityIdentityMismatch",
                "MaxEndpointMismatch",
                "MaxPcMonotonicDrop",
                "MaxKrgMonotonicDrop",
                "MaxKrwMonotonicRise",
                "Passed",
                "SchemaVersion",
                "PredictCodeCommit",
                "ConfigurationHash",
                "CoordinateTransformContract",
                "AssignmentMetadataExplicit",
                "SamplingManifestHash",
                "ReplayManifestHash",
            ]
            write_csv(
                qa_path,
                header,
                [[
                    "s05_c012",
                    7,
                    6,
                    87,
                    522,
                    522,
                    522,
                    522,
                    3,
                    6,
                    0.1,
                    0.3,
                    1.0e-8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    "1.7",
                    "",
                    method_hash,
                    "fault_local_to_reservoir_grid_signed_yz_v1",
                    "false",
                    "",
                    "",
                ]],
            )
            finalized = FINALIZE_CASE.validate_reservoir_export_qa(
                qa_path, "s05_c012", 7, "b" * 40, method_hash
            )
            verified = VERIFY_CASE.validate_reservoir_qa(
                qa_path, "s05_c012", 7, 1.0e-3, "b" * 40, method_hash
            )
            self.assertFalse(finalized["assignment_metadata_explicit"])
            self.assertEqual(finalized["case_type"], "legacy_level3_case")
            self.assertFalse(verified["assignment_metadata_explicit"])
            VERIFY_CASE.validate_reservoir_marker_metadata(
                None, verified, Path(temporary) / "legacy.done.json"
            )
            with self.assertRaisesRegex(ValueError, "legacy QA conflicts"):
                VERIFY_CASE.validate_reservoir_marker_metadata(
                    {"assignment_metadata_explicit": True},
                    verified,
                    Path(temporary) / "legacy.done.json",
                )

    def test_revised_case_requires_exact_marker_provenance(self) -> None:
        manifest_hash = "1" * 64
        replay_hash = "2" * 64
        configuration_hash = "3" * 64
        report = {
            "schema_version": "1.7",
            "sampling_manifest_sha256": manifest_hash,
            "replay_manifest_sha256": replay_hash,
            "configuration_sha256": configuration_hash,
            "coordinate_transform_contract": (
                "fault_local_to_reservoir_grid_signed_yz_v1"
            ),
            "assignment_metadata_explicit": True,
        }
        marker = {
            "schema_version": "1.7",
            "sampling_manifest_sha256": manifest_hash,
            "replay_manifest_sha256": replay_hash,
            "configuration_sha256": configuration_hash,
            "coordinate_transform_contract": (
                "fault_local_to_reservoir_grid_signed_yz_v1"
            ),
            "assignment_metadata_explicit": True,
        }
        marker_path = Path("revised.done.json")
        VERIFY_CASE.validate_reservoir_marker_metadata(marker, report, marker_path)
        with self.assertRaisesRegex(ValueError, "lacks reservoir-ready"):
            VERIFY_CASE.validate_reservoir_marker_metadata(None, report, marker_path)
        marker["sampling_manifest_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "sampling_manifest_sha256"):
            VERIFY_CASE.validate_reservoir_marker_metadata(
                marker, report, marker_path
            )

    def test_runtime_and_predict_physics_code_are_separate_contracts(self) -> None:
        launcher = PHASE_LAUNCHER.read_text(encoding="utf-8")
        checkpoint_worker = CHECKPOINT_WORKER.read_text(encoding="utf-8")
        kr_worker = KR_WORKER.read_text(encoding="utf-8")

        self.assertIn('PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT:?', launcher)
        self.assertIn('git -C "${RUNTIME_REPO}" rev-parse HEAD', launcher)
        self.assertIn('git -C "${PREDICT_CODE_ROOT}" rev-parse HEAD', launcher)
        self.assertIn('"${runtime_commit}" != "${SAMPLING_COMMIT}"', launcher)
        self.assertIn('"${physics_commit}" != "${PHYSICS_COMMIT}"', launcher)

        for worker in (checkpoint_worker, kr_worker):
            self.assertIn("prepare_production_replay_batch", worker)
            self.assertIn("${PREDICT_CODE_ROOT}", worker)
        self.assertIn(
            "run('${RUNTIME_REPO}/examples/pc_upscaling_pilot/run_pc_upscaling_ip_median_examples_full87.m')",
            checkpoint_worker,
        )
        self.assertIn(
            "run('${RUNTIME_REPO}/examples/pc_upscaling_pilot/run_kr_upscaling_dyn_median_examples_full87.m')",
            kr_worker,
        )

    def test_phase_launcher_accepts_external_checkpoint_bundle(self) -> None:
        launcher = PHASE_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('EXTERNAL_CHECKPOINT_JOB_ID="${EXTERNAL_CHECKPOINT_JOB_ID:-}"', launcher)
        self.assertIn('CHECKPOINT_JOB_ID="${EXTERNAL_CHECKPOINT_JOB_ID}"', launcher)
        self.assertIn(
            '--dependency="afterany:${CHECKPOINT_JOB_ID}"',
            launcher,
        )
        self.assertIn("checkpoint_submission_elements=0", launcher)

    def test_missing_work_is_mapped_to_restartable_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(
                root / "checkpoint_manifest" / "checkpoint_groups.csv",
                ["group_index", "group_id"],
                [[index, f"g{index}"] for index in range(1, 8)],
            )
            for index in (1, 2, 4):
                write_marker(
                    root / "checkpoint_pc" / f"g{index}" / "checkpoint.done.json",
                    {"status": "complete", "group_id": f"g{index}"},
                )

            write_csv(
                root / "case_work_manifest" / "geology_work.csv",
                ["geology_work_index", "geology_id"],
                [[1, "geo1"], [2, "geo2"], [3, "geo3"]],
            )
            write_marker(
                root
                / "case_inputs"
                / "cases"
                / "geo1"
                / "geology_case_inputs.done.json",
                {"status": "complete", "geology_id": "geo1"},
            )

            write_csv(
                root / "case_work_manifest" / "case_work.csv",
                [
                    "case_work_index",
                    "geology_id",
                    "case_id",
                    "case_relative_path",
                ],
                [[index, "geo1", index, f"cases/geo1/case_{index}"] for index in range(1, 10)],
            )
            for index in (1, 9):
                write_marker(
                    root
                    / "case_results"
                    / "cases"
                    / "geo1"
                    / f"case_{index}"
                    / "case.done.json",
                    {"status": "complete", "geology_id": "geo1", "case_id": index},
                )

            checkpoint = STATUS.checkpoint_status(root)
            assembly = STATUS.assembly_status(root)
            cases = STATUS.case_status(root)
            STATUS.add_chunk_status(checkpoint, 3)
            STATUS.add_chunk_status(assembly, 2)
            STATUS.add_chunk_status(cases, 4)

            self.assertEqual(checkpoint["missing"], 4)
            self.assertEqual(checkpoint["array_spec"], "1-3")
            self.assertEqual(assembly["array_spec"], "1-2")
            self.assertEqual(cases["array_spec"], "1-2")
            self.assertEqual(cases["array_task_total"], 3)

    def test_phase_status_rejects_stale_identity_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            physics = "a" * 40
            method = "b" * 64
            write_marker(
                root / "phase_run_identity.json",
                {
                    "physics_commit": physics,
                    "production_method_config_sha256": method,
                },
            )
            write_csv(
                root / "checkpoint_manifest" / "checkpoint_groups.csv",
                ["group_index", "group_id", "checkpoint_sha256"],
                [[1, "g1", "c" * 64]],
            )
            write_marker(
                root / "checkpoint_pc" / "g1" / "checkpoint.done.json",
                {
                    "status": "complete",
                    "group_id": "g1",
                    "checkpoint_sha256": "c" * 64,
                    "physics_commit": physics,
                    "method_config_sha256": "d" * 64,
                },
            )
            write_csv(
                root / "case_work_manifest" / "geology_work.csv",
                [
                    "geology_work_index",
                    "geology_id",
                    "assignment_count",
                    "assignment_sha256",
                ],
                [[1, "geo1", 7830, "e" * 64]],
            )
            write_marker(
                root
                / "case_inputs"
                / "cases"
                / "geo1"
                / "geology_case_inputs.done.json",
                {
                    "status": "complete",
                    "geology_id": "geo1",
                    "assignment_count": 7830,
                    "assignment_sha256": "f" * 64,
                },
            )
            write_csv(
                root / "case_work_manifest" / "case_work.csv",
                ["case_work_index", "geology_id", "case_id", "case_relative_path"],
                [[1, "geo1", 1, "cases/geo1/case01"]],
            )
            write_marker(
                root
                / "case_results"
                / "cases"
                / "geo1"
                / "case01"
                / "case.done.json",
                {
                    "status": "complete",
                    "geology_id": "geo1",
                    "case_id": 1,
                    "physics_commit": physics,
                    "method_config_sha256": method,
                    "dynamic_kr_representative_count": 6,
                    "pc_assignment_count": 522,
                    "strike_collapse_used": False,
                    "amgcl_required": True,
                    "pc_representations": ["full_slice"],
                    "reservoir_ready_validation": {
                        "assignment_metadata_explicit": False,
                        "configuration_sha256": method,
                        "coordinate_transform_contract": (
                            "fault_local_to_reservoir_grid_signed_yz_v1"
                        ),
                    },
                },
            )

            self.assertEqual(STATUS.checkpoint_status(root)["missing"], 1)
            self.assertEqual(STATUS.assembly_status(root)["missing"], 1)
            self.assertEqual(STATUS.case_status(root)["missing"], 1)


if __name__ == "__main__":
    unittest.main()
