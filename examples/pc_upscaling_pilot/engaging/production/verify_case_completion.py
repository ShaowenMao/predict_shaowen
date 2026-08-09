#!/usr/bin/env python3
"""Validate all assembled inputs and published Pc/Kr cases in a production run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-work-csv", required=True, type=Path)
    parser.add_argument("--case-input-root", required=True, type=Path)
    parser.add_argument("--case-result-root", required=True, type=Path)
    parser.add_argument("--expected-physics-commit", required=True)
    parser.add_argument("--expected-method-hash", required=True)
    parser.add_argument(
        "--max-source-log-permeability-mismatch", type=float, default=5.0e-3
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} is not finite: {row[field]!r}")
    return value


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{label} is not a Boolean value: {value!r}")


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "minimum": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": quantile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def validate_inventory(
    marker_path: Path,
    marker: dict,
    inventory_root: Path,
    errors: list[str],
) -> None:
    inventory = marker.get("files")
    if not isinstance(inventory, dict) or not inventory:
        errors.append(f"{marker_path}: missing or empty files inventory")
        return
    for relative_path, entry in inventory.items():
        path = inventory_root / relative_path
        if not path.is_file():
            errors.append(f"{marker_path}: missing inventoried file {path}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{marker_path}: invalid inventory for {relative_path}")
            continue
        actual_bytes = path.stat().st_size
        expected_bytes = int(entry.get("bytes", -1))
        if actual_bytes != expected_bytes:
            errors.append(
                f"{marker_path}: byte count mismatch for {relative_path}: "
                f"{actual_bytes} != {expected_bytes}"
            )
        actual_hash = sha256_file(path)
        expected_hash = str(entry.get("sha256", ""))
        if actual_hash != expected_hash:
            errors.append(
                f"{marker_path}: SHA-256 mismatch for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )


def find_inventory_path(marker_path: Path, marker: dict, pattern: str) -> Path:
    matches = [
        marker_path.parent / relative_path
        for relative_path in marker.get("files", {})
        if Path(relative_path).match(pattern)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{marker_path}: expected one inventory path matching {pattern}, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_reservoir_qa(
    path: Path,
    geology_id: str,
    case_id: int,
    max_source_mismatch: float,
    expected_physics_commit: str,
    expected_method_hash: str,
) -> dict[str, float | str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one QA row, found {len(rows)}")
    row = rows[0]
    expected_integer_fields = {
        "Level3CaseId": case_id,
        "WindowCount": 6,
        "SliceCount": 87,
        "PcCurveCount": 522,
        "KrCurveCount": 522,
        "PorosityCount": 522,
        "PermeabilityCellCount": 522,
        "PermeabilityComponentCount": 3,
        "SwiMedoidSelectionCount": 6,
        "Passed": 1,
    }
    if row["GeologyId"] != geology_id:
        raise ValueError(f"{path}: geology ID does not match {geology_id}")
    for field, expected in expected_integer_fields.items():
        actual = int(float(row[field]))
        if actual != expected:
            raise ValueError(f"{path}: {field}={actual}; expected {expected}")

    source_mismatch = finite(row, "MaxSourceLogPermeabilityMismatch")
    if source_mismatch > max_source_mismatch + 1.0e-12:
        raise ValueError(
            f"{path}: source permeability mismatch {source_mismatch} exceeds "
            f"{max_source_mismatch}"
        )
    for field in (
        "MaxSelectedSampleIndexMismatch",
        "MaxReplaySeedMismatch",
        "MaxPorosityIdentityMismatch",
        "MaxEndpointMismatch",
        "MaxPcMonotonicDrop",
        "MaxKrgMonotonicDrop",
        "MaxKrwMonotonicRise",
    ):
        if abs(finite(row, field)) > 1.0e-8:
            raise ValueError(f"{path}: {field} exceeds 1e-8")
    if finite(row, "MinUpscaledPorosity") <= 0.0:
        raise ValueError(f"{path}: non-positive upscaled porosity")
    if finite(row, "MaxUpscaledPorosity") >= 1.0:
        raise ValueError(f"{path}: upscaled porosity reaches or exceeds one")
    if finite(row, "MinPermeabilityMD") <= 0.0:
        raise ValueError(f"{path}: non-positive permeability")
    assignment_metadata_explicit = parse_bool(
        row.get("AssignmentMetadataExplicit", ""), "AssignmentMetadataExplicit"
    )
    expected_text = {
        "SchemaVersion": "1.7",
        "ConfigurationHash": expected_method_hash,
        "CoordinateTransformContract": "fault_local_to_reservoir_grid_signed_yz_v1",
    }
    if assignment_metadata_explicit:
        expected_text["PredictCodeCommit"] = expected_physics_commit
    for field, expected in expected_text.items():
        if row.get(field, "") != expected:
            raise ValueError(
                f"{path}: {field}={row.get(field)!r}; expected {expected!r}"
            )
    if assignment_metadata_explicit:
        for field in ("SamplingManifestHash", "ReplayManifestHash", "ConfigurationHash"):
            if not is_sha256(row.get(field, "")):
                raise ValueError(f"{path}: {field} is not a valid SHA-256")
    return {
        "source_log_permeability_mismatch": source_mismatch,
        "schema_version": row["SchemaVersion"],
        "sampling_manifest_sha256": row.get("SamplingManifestHash", "").lower(),
        "replay_manifest_sha256": row.get("ReplayManifestHash", "").lower(),
        "configuration_sha256": row["ConfigurationHash"].lower(),
        "coordinate_transform_contract": row["CoordinateTransformContract"],
        "assignment_metadata_explicit": assignment_metadata_explicit,
    }


def validate_reservoir_marker_metadata(
    reservoir_marker: object,
    reservoir_report: dict[str, float | str],
    marker_path: Path,
) -> None:
    """Require complete marker provenance only for revised-design cases."""
    if not reservoir_report["assignment_metadata_explicit"]:
        if isinstance(reservoir_marker, dict) and reservoir_marker.get(
            "assignment_metadata_explicit"
        ) is True:
            raise ValueError(
                f"{marker_path}: legacy QA conflicts with revised-design marker metadata"
            )
        return

    if not isinstance(reservoir_marker, dict):
        raise ValueError(
            f"{marker_path}: revised case lacks reservoir-ready validation metadata"
        )
    expected = {
        "schema_version": reservoir_report["schema_version"],
        "sampling_manifest_sha256": reservoir_report["sampling_manifest_sha256"],
        "replay_manifest_sha256": reservoir_report["replay_manifest_sha256"],
        "configuration_sha256": reservoir_report["configuration_sha256"],
        "coordinate_transform_contract": reservoir_report[
            "coordinate_transform_contract"
        ],
        "assignment_metadata_explicit": True,
    }
    for field, expected_value in expected.items():
        if reservoir_marker.get(field) != expected_value:
            raise ValueError(
                f"{marker_path}: reservoir-ready {field}="
                f"{reservoir_marker.get(field)!r}; expected {expected_value!r}"
            )


def main() -> int:
    args = parse_args()
    work_rows = read_rows(args.case_work_csv.resolve())
    expected_keys = [
        (row["geology_id"], int(row["case_id"])) for row in work_rows
    ]
    errors: list[str] = []
    if len(expected_keys) != len(set(expected_keys)):
        errors.append("Case work manifest contains duplicate geology/case keys")

    history_match_errors: list[float] = []
    source_mismatches: list[float] = []
    input_markers_validated = 0
    result_markers_validated = 0

    for row in work_rows:
        geology_id = row["geology_id"]
        case_id = int(row["case_id"])
        relative_path = Path(row["case_relative_path"])

        input_marker_path = (
            args.case_input_root.resolve()
            / relative_path
            / "case_inputs.done.json"
        )
        if not input_marker_path.is_file():
            errors.append(f"Missing case-input marker: {input_marker_path}")
            continue
        try:
            input_marker = json.loads(
                input_marker_path.read_text(encoding="utf-8")
            )
            expected_input = {
                "status": "assembled",
                "geology_id": geology_id,
                "case_id": case_id,
                "assignment_count": 522,
                "slice_count": 87,
                "window_count": 6,
                "representative_replay_count": 6,
            }
            for field, expected in expected_input.items():
                if input_marker.get(field) != expected:
                    errors.append(
                        f"{input_marker_path}: {field}="
                        f"{input_marker.get(field)!r}; expected {expected!r}"
                    )
            validate_inventory(
                input_marker_path,
                input_marker,
                input_marker_path.parent / "inputs",
                errors,
            )
            input_markers_validated += 1
        except Exception as error:
            errors.append(f"{input_marker_path}: validation error: {error}")

        result_marker_path = (
            args.case_result_root.resolve() / relative_path / "case.done.json"
        )
        if not result_marker_path.is_file():
            errors.append(f"Missing case-result marker: {result_marker_path}")
            continue
        try:
            result_marker = json.loads(
                result_marker_path.read_text(encoding="utf-8")
            )
            expected_result = {
                "status": "complete",
                "geology_id": geology_id,
                "case_id": case_id,
                "physics_commit": args.expected_physics_commit,
                "method_config_sha256": args.expected_method_hash,
                "dynamic_kr_representative_count": 6,
                "pc_assignment_count": 522,
                "strike_collapse_used": False,
                "amgcl_required": True,
                "pc_representations": ["full_slice"],
            }
            for field, expected in expected_result.items():
                if result_marker.get(field) != expected:
                    errors.append(
                        f"{result_marker_path}: {field}="
                        f"{result_marker.get(field)!r}; expected {expected!r}"
                    )
            if result_marker.get("kr_validation", {}).get("summary_rows") != 6:
                errors.append(f"{result_marker_path}: invalid Kr summary coverage")
            if result_marker.get("slice_validation", {}).get(
                "endpoint_count"
            ) != 522:
                errors.append(f"{result_marker_path}: invalid slice endpoint coverage")
            validate_inventory(
                result_marker_path,
                result_marker,
                result_marker_path.parent,
                errors,
            )
            reservoir_qa_path = find_inventory_path(
                result_marker_path,
                result_marker,
                "kr/reservoir_ready/reservoir_ready_qa_summary.csv",
            )
            reservoir_report = validate_reservoir_qa(
                reservoir_qa_path,
                geology_id,
                case_id,
                args.max_source_log_permeability_mismatch,
                args.expected_physics_commit,
                args.expected_method_hash,
            )
            source_mismatches.append(
                reservoir_report["source_log_permeability_mismatch"]
            )
            validate_reservoir_marker_metadata(
                result_marker.get("reservoir_ready_validation"),
                reservoir_report,
                result_marker_path,
            )
            summary_path = find_inventory_path(
                result_marker_path,
                result_marker,
                "kr/tables/kr_curve_summary_*_dyn_swi_medoid.csv",
            )
            summary_rows = read_rows(summary_path)
            if len(summary_rows) != 6:
                raise ValueError(
                    f"{summary_path}: expected six representative Kr rows"
                )
            history_match_errors.extend(
                finite(summary_row, "HistoryMatchError")
                for summary_row in summary_rows
            )
            result_markers_validated += 1
        except Exception as error:
            errors.append(f"{result_marker_path}: validation error: {error}")

    report = {
        "status": "complete" if not errors else "failed",
        "case_work_csv": str(args.case_work_csv.resolve()),
        "expected_case_count": len(work_rows),
        "expected_geology_count": len({key[0] for key in expected_keys}),
        "input_markers_validated": input_markers_validated,
        "result_markers_validated": result_markers_validated,
        "expected_physics_commit": args.expected_physics_commit,
        "expected_method_config_sha256": args.expected_method_hash,
        "full_slice_count_per_case": 522,
        "representative_dynamic_kr_count_per_case": 6,
        "history_match_error": summarize(history_match_errors),
        "source_log_permeability_mismatch": summarize(source_mismatches),
        "error_count": len(errors),
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
