#!/usr/bin/env python3
"""Validate every checkpoint replay/Pc completion marker before case assembly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


REQUIRED_OUTPUT_FILES = (
    "selection.csv",
    "replay_verification_by_task.csv",
    "pc_summary_by_task.csv",
    "pc_curve_points_by_task.csv",
    "pc_native_curve_points_by_task.csv",
    "pc_curves_by_task.mat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-groups-csv", required=True, type=Path)
    parser.add_argument("--checkpoint-output-root", required=True, type=Path)
    parser.add_argument("--expected-physics-commit", required=True)
    parser.add_argument("--expected-method-hash", required=True)
    parser.add_argument(
        "--default-replay-tolerance-log10", type=float, default=0.005
    )
    parser.add_argument(
        "--tolerance-exception",
        action="append",
        default=[],
        metavar="GROUP_ID=TOLERANCE",
        help="Explicit group-specific replay tolerance; may be repeated.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def parse_tolerance_exceptions(values: list[str]) -> dict[str, float]:
    exceptions: dict[str, float] = {}
    for value in values:
        try:
            group_id, tolerance_text = value.rsplit("=", 1)
            tolerance = float(tolerance_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid --tolerance-exception {value!r}; "
                "expected GROUP_ID=TOLERANCE"
            ) from error
        if not group_id or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"Invalid replay tolerance exception: {value!r}")
        if group_id in exceptions:
            raise ValueError(f"Duplicate replay tolerance exception: {group_id}")
        exceptions[group_id] = tolerance
    return exceptions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def exact_int(value: str, label: str) -> int:
    numeric = float(value)
    result = int(numeric)
    if not math.isfinite(numeric) or numeric != result:
        raise ValueError(f"{label} is not an exact integer: {value!r}")
    return result


def validate_exact_replay_identity(
    marker_path: Path, marker: dict, errors: list[str]
) -> None:
    """Validate the stronger identity contract on newly published groups."""
    contract = marker.get("replay_identity_contract")
    if contract is None:
        # Existing successful groups predate the explicit architecture hash.
        # Their immutable task/checkpoint/code/config identities remain valid.
        return
    expected_contract = (
        "exact_checkpoint_row_seed_attempt_and_material_map_sha256_v1"
    )
    if contract != expected_contract:
        errors.append(
            f"{marker_path}: replay identity contract {contract!r}; "
            f"expected {expected_contract!r}"
        )
        return

    try:
        selection = csv_rows(marker_path.parent / "selection.csv")
        replay = csv_rows(marker_path.parent / "replay_verification_by_task.csv")
        if len(selection) != len(replay):
            raise ValueError("selection/replay row-count mismatch")
        replay_by_task = {row["TaskId"]: row for row in replay}
        if len(replay_by_task) != len(replay):
            raise ValueError("duplicate replay TaskId")

        architecture_hashes: list[tuple[str, str]] = []
        for source_row, selected in enumerate(selection, start=1):
            task_id = selected["task_id"]
            row = replay_by_task.get(task_id)
            if row is None:
                raise ValueError(f"missing replay task {task_id}")
            if row["TaskKeySha256"] != selected["task_key_sha256"]:
                raise ValueError(f"task-key mismatch for {task_id}")
            if exact_int(row["SourceRow"], f"{task_id} SourceRow") != source_row:
                raise ValueError(f"source-row mismatch for {task_id}")

            selected_index = exact_int(
                selected["selected_sample_index"], f"{task_id} selected index"
            )
            if exact_int(row["SelectedSampleIndex"], f"{task_id} replay index") != selected_index:
                raise ValueError(f"selected-sample mismatch for {task_id}")
            expected_seed = exact_int(
                selected["exact_replay_seed"], f"{task_id} selected seed"
            )
            replay_seed = exact_int(row["ReplaySeed"], f"{task_id} replay seed")
            source_seed = exact_int(
                row["SourceAcceptedSeed"], f"{task_id} source seed"
            )
            if replay_seed != expected_seed or source_seed != expected_seed:
                raise ValueError(f"exact seed mismatch for {task_id}")

            seed_base = exact_int(
                selected["source_seed_base"], f"{task_id} seed base"
            )
            expected_attempt = expected_seed - seed_base + 1
            replay_attempt = exact_int(
                row["AttemptIndex"], f"{task_id} replay attempt"
            )
            source_attempt = exact_int(
                row["SourceAcceptedAttemptIndex"], f"{task_id} source attempt"
            )
            if replay_attempt != expected_attempt or source_attempt != expected_attempt:
                raise ValueError(f"accepted-attempt mismatch for {task_id}")
            if not row["ReplayMode"].startswith("direct_seed_"):
                raise ValueError(f"non-direct replay for {task_id}")
            if row["DiscreteArchitectureStatus"] != "matched":
                raise ValueError(f"discrete-architecture mismatch for {task_id}")
            architecture_hash = row["DiscreteArchitectureSha256"].strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", architecture_hash) is None:
                raise ValueError(f"invalid architecture hash for {task_id}")
            architecture_hashes.append((task_id, architecture_hash))

        architecture_manifest = "\n".join(
            f"{task_id},{architecture_hash}"
            for task_id, architecture_hash in sorted(architecture_hashes)
        )
        actual_manifest_hash = hashlib.sha256(
            architecture_manifest.encode("ascii")
        ).hexdigest()
        if marker.get("discrete_architecture_count") != len(architecture_hashes):
            raise ValueError("discrete-architecture count mismatch")
        if marker.get("discrete_architecture_manifest_sha256") != actual_manifest_hash:
            raise ValueError("discrete-architecture manifest hash mismatch")
    except (KeyError, OSError, ValueError) as error:
        errors.append(f"{marker_path}: replay identity validation failed: {error}")


def validate_replay_tolerance(
    group_id: str,
    recorded_tolerance: float,
    maximum_difference: float,
    policy_maximum: float,
    errors: list[str],
) -> None:
    """Accept stricter historical markers without relaxing the policy ceiling."""
    if not math.isfinite(recorded_tolerance) or recorded_tolerance <= 0:
        errors.append(
            f"{group_id}: invalid replay tolerance {recorded_tolerance!r}"
        )
    elif recorded_tolerance > policy_maximum + 1.0e-12:
        errors.append(
            f"{group_id}: recorded replay tolerance {recorded_tolerance} exceeds "
            f"policy maximum {policy_maximum}"
        )
    if not math.isfinite(maximum_difference) or maximum_difference < 0:
        errors.append(
            f"{group_id}: invalid replay difference {maximum_difference!r}"
        )
    elif maximum_difference > recorded_tolerance + 1.0e-12:
        errors.append(
            f"{group_id}: replay difference {maximum_difference} exceeds "
            f"recorded tolerance {recorded_tolerance}"
        )
    elif maximum_difference > policy_maximum + 1.0e-12:
        errors.append(
            f"{group_id}: replay difference {maximum_difference} exceeds "
            f"policy maximum {policy_maximum}"
        )


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_file_inventory(
    marker_path: Path, marker: dict, errors: list[str]
) -> None:
    inventory = marker.get("files")
    if not isinstance(inventory, dict):
        errors.append(f"{marker_path}: missing files inventory")
        return

    for filename in REQUIRED_OUTPUT_FILES:
        entry = inventory.get(filename)
        output_path = marker_path.parent / filename
        if not isinstance(entry, dict):
            errors.append(f"{marker_path}: missing inventory entry for {filename}")
            continue
        if not output_path.is_file():
            errors.append(f"{marker_path}: missing output file {output_path}")
            continue
        expected_bytes = int(entry.get("bytes", -1))
        actual_bytes = output_path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(
                f"{marker_path}: byte count mismatch for {filename}: "
                f"{actual_bytes} != {expected_bytes}"
            )
        expected_hash = str(entry.get("sha256", ""))
        actual_hash = sha256_file(output_path)
        if actual_hash != expected_hash:
            errors.append(
                f"{marker_path}: SHA-256 mismatch for {filename}: "
                f"{actual_hash} != {expected_hash}"
            )


def main() -> int:
    args = parse_args()
    exceptions = parse_tolerance_exceptions(args.tolerance_exception)
    groups_path = args.checkpoint_groups_csv.resolve()
    output_root = args.checkpoint_output_root.resolve()
    errors: list[str] = []

    with groups_path.open(newline="", encoding="utf-8-sig") as stream:
        groups = list(csv.DictReader(stream))
    if not groups:
        raise ValueError(f"No checkpoint groups found in {groups_path}")

    group_ids = [row["group_id"] for row in groups]
    if len(group_ids) != len(set(group_ids)):
        errors.append("Checkpoint manifest contains duplicate group IDs")
    unknown_exceptions = sorted(set(exceptions) - set(group_ids))
    if unknown_exceptions:
        errors.append(
            "Tolerance exceptions reference unknown groups: "
            + ", ".join(unknown_exceptions)
        )

    replay_differences: list[float] = []
    tolerance_counts: dict[str, int] = {}
    total_tasks = 0
    validated_groups = 0

    for row in groups:
        group_id = row["group_id"]
        marker_path = output_root / group_id / "checkpoint.done.json"
        if not marker_path.is_file():
            errors.append(f"{group_id}: missing completion marker {marker_path}")
            continue
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{group_id}: unreadable marker: {error}")
            continue

        expected_task_count = int(row["task_count"])
        maximum_tolerance = exceptions.get(
            group_id, args.default_replay_tolerance_log10
        )
        actual_tolerance = float(marker.get("replay_tolerance_log10", math.nan))
        max_difference = float(
            marker.get("max_replay_abs_log10_difference", math.nan)
        )

        expected_values = {
            "status": "complete",
            "group_id": group_id,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "physics_commit": args.expected_physics_commit,
            "method_config_sha256": args.expected_method_hash,
            "task_count": expected_task_count,
            "pc_summary_row_count": expected_task_count,
            "fine_replay_maps_retained": False,
        }
        for field, expected in expected_values.items():
            actual = marker.get(field)
            if actual != expected:
                errors.append(
                    f"{group_id}: marker field {field}={actual!r}; "
                    f"expected {expected!r}"
                )

        validate_replay_tolerance(
            group_id,
            actual_tolerance,
            max_difference,
            maximum_tolerance,
            errors,
        )

        row_count_expectations = {
            "selection.csv": expected_task_count,
            "replay_verification_by_task.csv": expected_task_count,
            "pc_summary_by_task.csv": int(
                marker.get("pc_summary_row_count", -1)
            ),
            "pc_curve_points_by_task.csv": int(
                marker.get("pc_fixed_curve_row_count", -1)
            ),
            "pc_native_curve_points_by_task.csv": int(
                marker.get("pc_native_curve_row_count", -1)
            ),
        }
        for filename, expected_rows in row_count_expectations.items():
            path = marker_path.parent / filename
            if not path.is_file():
                continue
            actual_rows = csv_row_count(path)
            if expected_rows <= 0 or actual_rows != expected_rows:
                errors.append(
                    f"{group_id}: {filename} has {actual_rows} rows; "
                    f"expected {expected_rows}"
                )

        validate_file_inventory(marker_path, marker, errors)
        validate_exact_replay_identity(marker_path, marker, errors)
        if math.isfinite(max_difference):
            replay_differences.append(max_difference)
        tolerance_key = f"{actual_tolerance:.12g}"
        tolerance_counts[tolerance_key] = tolerance_counts.get(tolerance_key, 0) + 1
        total_tasks += expected_task_count
        validated_groups += 1

    report = {
        "status": "complete" if not errors else "failed",
        "checkpoint_groups_csv": str(groups_path),
        "checkpoint_output_root": str(output_root),
        "expected_group_count": len(groups),
        "validated_group_count": validated_groups,
        "total_unique_replay_pc_tasks": total_tasks,
        "expected_physics_commit": args.expected_physics_commit,
        "expected_method_config_sha256": args.expected_method_hash,
        "default_replay_tolerance_log10": args.default_replay_tolerance_log10,
        "replay_tolerance_semantics": "maximum_allowed_numerical_difference",
        "tolerance_exceptions": exceptions,
        "observed_tolerance_counts": tolerance_counts,
        "replay_difference_log10": {
            "minimum": min(replay_differences) if replay_differences else None,
            "median": (
                statistics.median(replay_differences)
                if replay_differences
                else None
            ),
            "p95": quantile(replay_differences, 0.95),
            "maximum": max(replay_differences) if replay_differences else None,
        },
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
