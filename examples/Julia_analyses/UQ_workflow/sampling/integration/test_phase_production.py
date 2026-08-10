#!/usr/bin/env python3
"""Tests for restartable phase-production status and chunk planning."""

from __future__ import annotations

import csv
import hashlib
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
CHECKPOINT_SUBMITTER = CHECKPOINT_WORKER.with_name(
    "submit_checkpoint_replay_pc.sh"
)
KR_WORKER = CHECKPOINT_WORKER.with_name("run_case_dynamic_kr.sh")
FINALIZE_CASE_PATH = CHECKPOINT_WORKER.with_name("finalize_case_kr.py")
VERIFY_CASE_PATH = CHECKPOINT_WORKER.with_name("verify_case_completion.py")
FINALIZE_CHECKPOINT_PATH = CHECKPOINT_WORKER.with_name(
    "finalize_checkpoint_pc.py"
)
VERIFY_CHECKPOINT_PATH = CHECKPOINT_WORKER.with_name(
    "verify_checkpoint_completion.py"
)
CONTINUATION_SUBMITTER = CHECKPOINT_WORKER.with_name(
    "submit_full_production_continuation.sh"
)


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
FINALIZE_CHECKPOINT = load_module(
    "finalize_checkpoint_pc", FINALIZE_CHECKPOINT_PATH
)
VERIFY_CHECKPOINT = load_module(
    "verify_checkpoint_completion", VERIFY_CHECKPOINT_PATH
)


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
    def test_validated_replay_tolerance_is_a_maximum_policy(self) -> None:
        errors: list[str] = []
        VERIFY_CHECKPOINT.validate_replay_tolerance(
            "strict_existing", 0.001, 0.0009, 0.005, errors
        )
        VERIFY_CHECKPOINT.validate_replay_tolerance(
            "validated_retry", 0.005, 0.0049, 0.005, errors
        )
        self.assertEqual(errors, [])

        errors = []
        VERIFY_CHECKPOINT.validate_replay_tolerance(
            "too_loose", 0.006, 0.004, 0.005, errors
        )
        VERIFY_CHECKPOINT.validate_replay_tolerance(
            "too_different", 0.005, 0.0051, 0.005, errors
        )
        self.assertEqual(len(errors), 2)

    def test_replay_identity_requires_exact_seed_index_and_architecture(self) -> None:
        selection = {
            "task_id": "rpc_test",
            "task_key_sha256": "1" * 64,
            "selected_sample_index": "17",
            "exact_replay_seed": "10042",
            "source_seed_base": "10000",
        }
        architecture_hash = "a" * 64
        replay = {
            "TaskId": "rpc_test",
            "TaskKeySha256": "1" * 64,
            "SourceRow": "1",
            "SelectedSampleIndex": "17",
            "ReplaySeed": "10042",
            "SourceAcceptedSeed": "10042",
            "AttemptIndex": "43",
            "SourceAcceptedAttemptIndex": "43",
            "ReplayMode": "direct_seed_from_selection_table",
            "DiscreteArchitectureStatus": "matched",
            "DiscreteArchitectureSha256": architecture_hash,
        }
        self.assertEqual(
            FINALIZE_CHECKPOINT.validate_replay_identity(selection, replay, 1),
            architecture_hash,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker_path = root / "checkpoint.done.json"
            write_csv(root / "selection.csv", list(selection), [list(selection.values())])
            write_csv(root / "replay_verification_by_task.csv", list(replay), [list(replay.values())])
            manifest_text = f"rpc_test,{architecture_hash}"
            marker = {
                "replay_identity_contract": (
                    "exact_checkpoint_row_seed_attempt_and_material_map_sha256_v1"
                ),
                "discrete_architecture_count": 1,
                "discrete_architecture_manifest_sha256": hashlib.sha256(
                    manifest_text.encode("ascii")
                ).hexdigest(),
            }
            errors: list[str] = []
            VERIFY_CHECKPOINT.validate_exact_replay_identity(
                marker_path, marker, errors
            )
            self.assertEqual(errors, [])

            replay["ReplaySeed"] = "10043"
            write_csv(root / "replay_verification_by_task.csv", list(replay), [list(replay.values())])
            errors = []
            VERIFY_CHECKPOINT.validate_exact_replay_identity(
                marker_path, marker, errors
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("seed mismatch", errors[0])

    def test_production_defaults_use_validated_tolerance(self) -> None:
        launcher = PHASE_LAUNCHER.read_text(encoding="utf-8")
        checkpoint_worker = CHECKPOINT_WORKER.read_text(encoding="utf-8")
        checkpoint_gate = CHECKPOINT_WORKER.with_name(
            "run_checkpoint_completion_gate.sh"
        ).read_text(encoding="utf-8")
        qualification_submitter = CHECKPOINT_WORKER.with_name(
            "submit_qualification_batch.sh"
        ).read_text(encoding="utf-8")
        for source in (
            launcher,
            checkpoint_worker,
            checkpoint_gate,
            qualification_submitter,
        ):
            self.assertIn("0.005", source)

        method_config = CHECKPOINT_WORKER.with_name(
            "production_method_config.toml"
        )
        normalized_config = method_config.read_bytes().replace(b"\r\n", b"\n")
        expected_hash = hashlib.sha256(normalized_config).hexdigest()
        acceptance_policy = CHECKPOINT_WORKER.with_name(
            "production_acceptance_policy.toml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            f'method_config_sha256 = "{expected_hash}"', acceptance_policy
        )

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
        continuation = CONTINUATION_SUBMITTER.read_text(encoding="utf-8")

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
        self.assertIn('git -C "${RUNTIME_REPO}" rev-parse HEAD', continuation)
        self.assertIn(
            'git -C "${PREDICT_CODE_ROOT}" rev-parse HEAD', continuation
        )
        self.assertIn('sha256sum "${METHOD_CONFIG}"', continuation)
        self.assertIn('PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT:-', continuation)
        self.assertIn('PREDICT_ROOT="${PREDICT_ROOT:-', continuation)
        self.assertIn('METHOD_CONFIG="${METHOD_CONFIG:-', continuation)
        self.assertIn('PREDICT_ROOT="${PREDICT_ROOT}"', continuation)
        self.assertIn(
            '"replay_tolerance_semantics": "maximum_allowed_numerical_difference"',
            continuation,
        )
        self.assertIn(
            '"completed_checkpoint_policy": "preserve_valid_existing_markers"',
            continuation,
        )
        for marker_field in (
            '"checkpoint_sha256": row["checkpoint_sha256"]',
            '"physics_commit": physics_commit',
            '"method_config_sha256": method_config_sha256',
            '"task_count": int(row["task_count"])',
        ):
            self.assertIn(marker_field, continuation)

    def test_phase_launcher_accepts_external_checkpoint_bundle(self) -> None:
        launcher = PHASE_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('EXTERNAL_CHECKPOINT_JOB_ID="${EXTERNAL_CHECKPOINT_JOB_ID:-}"', launcher)
        self.assertIn('CHECKPOINT_JOB_ID="${EXTERNAL_CHECKPOINT_JOB_ID}"', launcher)
        self.assertIn(
            '--dependency="afterany:${CHECKPOINT_JOB_ID}"',
            launcher,
        )
        self.assertIn("checkpoint_submission_elements=0", launcher)

    def test_standard_checkpoint_array_uses_proven_resource_contract(self) -> None:
        launcher = PHASE_LAUNCHER.read_text(encoding="utf-8")
        submitter = CHECKPOINT_SUBMITTER.read_text(encoding="utf-8")

        self.assertIn('CHECKPOINT_MEMORY="${CHECKPOINT_MEMORY:-18G}"', launcher)
        self.assertIn("--cpus-per-task=1", launcher)
        self.assertIn(
            '--array="${CHECKPOINT_ARRAY_SPEC}%${CHECKPOINT_MAX_CONCURRENT}"',
            launcher,
        )
        self.assertNotIn("--exclusive", launcher)
        self.assertIn(
            'CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/checkpoint}"',
            launcher,
        )
        self.assertIn(
            'CASE_TEMP_ROOT="${CASE_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/case}"',
            launcher,
        )
        self.assertIn('CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT}"', launcher)
        self.assertIn('CASE_TEMP_ROOT="${CASE_TEMP_ROOT}"', launcher)
        self.assertIn('--mem="${CHECKPOINT_MEMORY:-18G}"', submitter)
        self.assertIn('PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT:-', submitter)
        self.assertIn('METHOD_CONFIG="${METHOD_CONFIG:-', submitter)

    def test_large_transient_work_defaults_to_node_local_storage(self) -> None:
        checkpoint_worker = CHECKPOINT_WORKER.read_text(encoding="utf-8")
        kr_worker = KR_WORKER.read_text(encoding="utf-8")

        for worker in (checkpoint_worker, kr_worker):
            self.assertIn(
                'NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT:-/tmp/${USER}/predict_shaowen}"',
                worker,
            )
        self.assertIn(
            'CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/checkpoint}"',
            checkpoint_worker,
        )
        self.assertIn(
            'CASE_TEMP_ROOT="${CASE_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/case}"',
            kr_worker,
        )
        self.assertNotIn(
            'CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT:-${SCRATCH_ROOT}/tmp}"',
            checkpoint_worker,
        )
        self.assertNotIn(
            'CASE_TEMP_ROOT="${CASE_TEMP_ROOT:-${SCRATCH_ROOT}/tmp}"',
            kr_worker,
        )

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
