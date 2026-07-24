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
) -> dict[str, float]:
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
    return {"source_log_permeability_mismatch": source_mismatch}


def validate_branch_qa(path: Path, geology_id: str, case_id: int) -> None:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one branch QA row, found {len(rows)}")
    row = rows[0]
    if row["GeologyId"] != geology_id:
        raise ValueError(f"{path}: geology ID does not match {geology_id}")
    expected = {
        "Level3CaseId": case_id,
        "WindowCount": 6,
        "SliceCount": 87,
        "FaultCellCount": 522,
        "Passed": 1,
    }
    for field, expected_value in expected.items():
        actual = int(float(row[field]))
        if actual != expected_value:
            raise ValueError(
                f"{path}: {field}={actual}; expected {expected_value}"
            )
    for field in (
        "MaxEndpointMismatch",
        "MaxPcMonotonicDrop",
        "MaxKrgMonotonicDrop",
        "MaxKrwMonotonicRise",
        "MaxNormalizedKrShapeMismatch",
    ):
        if abs(finite(row, field)) > 1.0e-8:
            raise ValueError(f"{path}: {field} exceeds 1e-8")


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
                "pc_representations": ["full_slice", "pe_branch_medoid"],
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
            )
            source_mismatches.append(
                reservoir_report["source_log_permeability_mismatch"]
            )
            branch_qa_path = find_inventory_path(
                result_marker_path,
                result_marker,
                "kr/reservoir_ready_pe_branch_medoid/tables/pe_branch_qa_*.csv",
            )
            validate_branch_qa(branch_qa_path, geology_id, case_id)

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
