#!/usr/bin/env python3
"""Report restartable completion status for one full-fault production phase."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--checkpoint-chunk-size", type=int, default=1)
    parser.add_argument("--assembly-chunk-size", type=int, default=1)
    parser.add_argument("--kr-chunk-size", type=int, default=1)
    parser.add_argument(
        "--max-replay-tolerance-log10",
        type=float,
        default=0.005,
        help=(
            "Maximum accepted replay numerical-equivalence tolerance. "
            "Markers produced with a stricter tolerance remain valid."
        ),
    )
    parser.add_argument(
        "--replay-tolerance-exception",
        action="append",
        default=[],
        metavar="GROUP_ID=TOLERANCE",
        help="Explicit group-specific replay tolerance; may be repeated.",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print numeric counts and compressed missing-index ranges for Bash.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def parse_tolerance_exceptions(values: list[str]) -> dict[str, float]:
    """Parse explicit checkpoint-group numerical-equivalence exceptions."""

    exceptions: dict[str, float] = {}
    for value in values:
        try:
            group_id, tolerance_text = value.rsplit("=", 1)
            tolerance = float(tolerance_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid replay tolerance exception {value!r}; "
                "expected GROUP_ID=TOLERANCE"
            ) from error
        if not group_id or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"Invalid replay tolerance exception: {value!r}")
        if group_id in exceptions:
            raise ValueError(f"Duplicate replay tolerance exception: {group_id}")
        exceptions[group_id] = tolerance
    return exceptions


def valid_marker(path: Path, expected: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(marker.get(key) == value for key, value in expected.items())


def valid_checkpoint_marker(
    path: Path,
    expected: dict[str, object],
    maximum_tolerance: float,
) -> bool:
    """Validate checkpoint identity and maximum-tolerance semantics.

    Legacy completion markers without numerical fields remain eligible because
    their immutable checkpoint, physics, and configuration identities are still
    checked. When numerical fields are present, both must satisfy the documented
    maximum policy; a stricter recorded tolerance is intentionally accepted.
    """

    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not all(marker.get(key) == value for key, value in expected.items()):
        return False

    recorded = marker.get("replay_tolerance_log10")
    difference = marker.get("max_replay_abs_log10_difference")
    if recorded is None and difference is None:
        return True
    if recorded is None or difference is None:
        return False
    try:
        recorded_value = float(recorded)
        difference_value = float(difference)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(maximum_tolerance)
        and maximum_tolerance > 0
        and math.isfinite(recorded_value)
        and recorded_value > 0
        and recorded_value <= maximum_tolerance + 1.0e-12
        and math.isfinite(difference_value)
        and difference_value >= 0
        and difference_value <= recorded_value + 1.0e-12
    )


def read_run_identity(run_root: Path) -> dict[str, object]:
    path = run_root / "phase_run_identity.json"
    if not path.is_file():
        return {}
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return identity if isinstance(identity, dict) else {}


def valid_current_case_marker(
    path: Path,
    expected: dict[str, object],
    identity: dict[str, object],
) -> bool:
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not all(marker.get(key) == value for key, value in expected.items()):
        return False
    if not identity:
        return True
    reservoir = marker.get("reservoir_ready_validation")
    return (
        isinstance(reservoir, dict)
        and reservoir.get("assignment_metadata_explicit") is True
        and reservoir.get("configuration_sha256")
        == identity.get("production_method_config_sha256")
        and reservoir.get("coordinate_transform_contract")
        == "fault_local_to_reservoir_grid_signed_yz_v1"
    )


def compress_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ordered = sorted(set(indices))
    pieces: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        pieces.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    pieces.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(pieces)


def add_chunk_status(stage: dict[str, object], chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive")
    total = int(stage["total"])
    missing_indices = [int(value) for value in stage["missing_indices"]]
    missing_chunks = sorted(
        {(index - 1) // chunk_size + 1 for index in missing_indices}
    )
    stage["chunk_size"] = chunk_size
    stage["array_task_total"] = (total + chunk_size - 1) // chunk_size
    stage["missing_array_tasks"] = len(missing_chunks)
    stage["missing_array_task_indices"] = missing_chunks
    stage["array_spec"] = compress_indices(missing_chunks)


def checkpoint_status(
    run_root: Path,
    maximum_tolerance: float = 0.005,
    tolerance_exceptions: dict[str, float] | None = None,
) -> dict[str, object]:
    rows = read_rows(run_root / "checkpoint_manifest" / "checkpoint_groups.csv")
    identity = read_run_identity(run_root)
    missing: list[int] = []
    exceptions = tolerance_exceptions or {}
    known_group_ids = {row["group_id"] for row in rows}
    unknown_exceptions = sorted(set(exceptions) - known_group_ids)
    if unknown_exceptions:
        raise ValueError(
            "Tolerance exceptions reference unknown checkpoint groups: "
            + ", ".join(unknown_exceptions)
        )
    for row in rows:
        index = int(row["group_index"])
        marker = (
            run_root
            / "checkpoint_pc"
            / row["group_id"]
            / "checkpoint.done.json"
        )
        expected: dict[str, object] = {
            "status": "complete",
            "group_id": row["group_id"],
        }
        if row.get("checkpoint_sha256"):
            expected["checkpoint_sha256"] = row["checkpoint_sha256"]
        if identity:
            expected["physics_commit"] = identity.get("physics_commit")
            expected["method_config_sha256"] = identity.get(
                "production_method_config_sha256"
            )
        group_tolerance = exceptions.get(row["group_id"], maximum_tolerance)
        if not valid_checkpoint_marker(marker, expected, group_tolerance):
            missing.append(index)
    return {
        "total": len(rows),
        "complete": len(rows) - len(missing),
        "missing": len(missing),
        "missing_indices": missing,
        "array_spec": compress_indices(missing),
    }


def assembly_status(run_root: Path) -> dict[str, object]:
    rows = read_rows(run_root / "case_work_manifest" / "geology_work.csv")
    missing: list[int] = []
    for row in rows:
        index = int(row["geology_work_index"])
        marker = (
            run_root
            / "case_inputs"
            / "cases"
            / row["geology_id"]
            / "geology_case_inputs.done.json"
        )
        expected: dict[str, object] = {
            "status": "complete",
            "geology_id": row["geology_id"],
        }
        if row.get("assignment_count"):
            expected["assignment_count"] = int(row["assignment_count"])
        if row.get("assignment_sha256"):
            expected["assignment_sha256"] = row["assignment_sha256"]
        if not valid_marker(marker, expected):
            missing.append(index)
    return {
        "total": len(rows),
        "complete": len(rows) - len(missing),
        "missing": len(missing),
        "missing_indices": missing,
        "array_spec": compress_indices(missing),
    }


def case_status(run_root: Path) -> dict[str, object]:
    rows = read_rows(run_root / "case_work_manifest" / "case_work.csv")
    identity = read_run_identity(run_root)
    missing: list[int] = []
    for row in rows:
        index = int(row["case_work_index"])
        marker = (
            run_root
            / "case_results"
            / row["case_relative_path"]
            / "case.done.json"
        )
        expected: dict[str, object] = {
            "status": "complete",
            "geology_id": row["geology_id"],
            "case_id": int(row["case_id"]),
        }
        if identity:
            expected.update(
                {
                    "physics_commit": identity.get("physics_commit"),
                    "method_config_sha256": identity.get(
                        "production_method_config_sha256"
                    ),
                    "dynamic_kr_representative_count": 6,
                    "pc_assignment_count": 522,
                    "strike_collapse_used": False,
                    "amgcl_required": True,
                    "pc_representations": ["full_slice"],
                }
            )
        if not valid_current_case_marker(marker, expected, identity):
            missing.append(index)
    return {
        "total": len(rows),
        "complete": len(rows) - len(missing),
        "missing": len(missing),
        "missing_indices": missing,
        "array_spec": compress_indices(missing),
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    tolerance_exceptions = parse_tolerance_exceptions(
        args.replay_tolerance_exception
    )
    report = {
        "schema_version": "independent_full_fault_phase_status_v1",
        "run_root": str(run_root),
        "checkpoint": checkpoint_status(
            run_root,
            args.max_replay_tolerance_log10,
            tolerance_exceptions,
        ),
        "assembly": assembly_status(run_root),
        "dynamic_kr": case_status(run_root),
        "replay_tolerance_exceptions": tolerance_exceptions,
    }
    add_chunk_status(report["checkpoint"], args.checkpoint_chunk_size)
    add_chunk_status(report["assembly"], args.assembly_chunk_size)
    add_chunk_status(report["dynamic_kr"], args.kr_chunk_size)
    output = args.output_json
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.shell:
        for prefix, key in (
            ("CHECKPOINT", "checkpoint"),
            ("ASSEMBLY", "assembly"),
            ("KR", "dynamic_kr"),
        ):
            stage = report[key]
            print(f"{prefix}_TOTAL={stage['total']}")
            print(f"{prefix}_COMPLETE={stage['complete']}")
            print(f"{prefix}_MISSING={stage['missing']}")
            print(f"{prefix}_ARRAY_TASK_TOTAL={stage['array_task_total']}")
            print(f"{prefix}_MISSING_ARRAY_TASKS={stage['missing_array_tasks']}")
            print(f"{prefix}_ARRAY_SPEC='{stage['array_spec']}'")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
