#!/usr/bin/env python3
"""Assemble ten field-case Pc inputs from checkpoint-level compact products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path


WINDOWS = [f"famp{i}" for i in range(1, 7)]
EXPECTED_ROWS_PER_CASE = 87 * len(WINDOWS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geology-id", required=True)
    parser.add_argument("--geology-assignment-csv", type=Path)
    parser.add_argument("--sampling-csv", type=Path)
    parser.add_argument("--assignment-to-task-csv", type=Path)
    parser.add_argument("--checkpoint-output-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--case-ids",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated Level-3 case IDs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case_ids(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or any(value < 1 or value > 10 for value in values):
        raise ValueError("Case IDs must be a nonempty subset of 1:10")
    return values


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_geology_assignments(
    sampling_path: Path,
    assignment_path: Path,
    geology_id: str,
    case_ids: set[int],
) -> dict[int, list[dict[str, str]]]:
    cases: dict[int, list[dict[str, str]]] = defaultdict(list)
    with sampling_path.open(
        newline="", encoding="utf-8-sig"
    ) as sampling_stream, assignment_path.open(
        newline="", encoding="utf-8-sig"
    ) as assignment_stream:
        sampling_reader = csv.DictReader(sampling_stream)
        assignment_reader = csv.DictReader(assignment_stream)
        for row_index, (sampling, assignment) in enumerate(
            zip(sampling_reader, assignment_reader, strict=True), start=1
        ):
            if int(assignment["assignment_index"]) != row_index:
                raise ValueError(f"Assignment index mismatch at row {row_index}")
            if sampling["geology_id"] != assignment["geology_id"]:
                raise ValueError(f"Geology mismatch at row {row_index}")
            if int(sampling["case_id"]) != int(assignment["case_id"]):
                raise ValueError(f"Case mismatch at row {row_index}")
            if sampling["window"] != assignment["window"]:
                raise ValueError(f"Window mismatch at row {row_index}")
            case_id = int(sampling["case_id"])
            if sampling["geology_id"] != geology_id or case_id not in case_ids:
                continue
            combined = dict(sampling)
            combined["task_id"] = assignment["task_id"]
            combined["task_key_sha256"] = assignment["task_key_sha256"]
            combined["assignment_index"] = assignment["assignment_index"]
            cases[case_id].append(combined)
    missing = case_ids - set(cases)
    if missing:
        raise ValueError(f"Missing requested cases for {geology_id}: {sorted(missing)}")
    for case_id, rows in cases.items():
        if len(rows) != EXPECTED_ROWS_PER_CASE:
            raise ValueError(
                f"{geology_id} case {case_id:02d} has {len(rows)} rows; "
                f"expected {EXPECTED_ROWS_PER_CASE}"
            )
        rows.sort(key=lambda row: (int(row["slice_index"]), int(row["window"][4:])))
        keys = {(int(row["slice_index"]), row["window"]) for row in rows}
        expected = {(slice_id, window) for slice_id in range(1, 88) for window in WINDOWS}
        if keys != expected:
            raise ValueError(f"Case {case_id:02d} lacks exact 6x87 coverage")
    return cases


def load_combined_geology_assignments(
    combined_path: Path,
    geology_id: str,
    case_ids: set[int],
) -> dict[int, list[dict[str, str]]]:
    cases: dict[int, list[dict[str, str]]] = defaultdict(list)
    _, rows = read_rows(combined_path)
    for row in rows:
        case_id = int(row["case_id"])
        if row["geology_id"] == geology_id and case_id in case_ids:
            cases[case_id].append(row)
    missing = case_ids - set(cases)
    if missing:
        raise ValueError(f"Missing requested cases for {geology_id}: {sorted(missing)}")
    for case_id, case_rows in cases.items():
        if len(case_rows) != EXPECTED_ROWS_PER_CASE:
            raise ValueError(
                f"{geology_id} case {case_id:02d} has {len(case_rows)} rows; "
                f"expected {EXPECTED_ROWS_PER_CASE}"
            )
        case_rows.sort(
            key=lambda row: (int(row["slice_index"]), int(row["window"][4:]))
        )
        keys = {
            (int(row["slice_index"]), row["window"]) for row in case_rows
        }
        expected = {
            (slice_id, window)
            for slice_id in range(1, 88)
            for window in WINDOWS
        }
        if keys != expected:
            raise ValueError(f"Case {case_id:02d} lacks exact 6x87 coverage")
    return cases


def load_checkpoint_products(
    checkpoint_root: Path,
    geology_id: str,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    dict[str, str],
]:
    summary_by_task: dict[str, dict[str, str]] = {}
    fixed_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    native_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_marker_hashes: dict[str, str] = {}

    for window in WINDOWS:
        group_id = f"checkpoint_{geology_id}_{window}"
        group_root = checkpoint_root / group_id
        marker_path = require_file(group_root / "checkpoint.done.json")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "complete" or marker.get("group_id") != group_id:
            raise ValueError(f"Invalid checkpoint marker: {marker_path}")
        group_marker_hashes[group_id] = sha256_file(marker_path)

        _, summary_rows = read_rows(
            require_file(group_root / "pc_summary_by_task.csv")
        )
        _, fixed_rows = read_rows(
            require_file(group_root / "pc_curve_points_by_task.csv")
        )
        _, native_rows = read_rows(
            require_file(group_root / "pc_native_curve_points_by_task.csv")
        )
        for row in summary_rows:
            task_id = row["TaskId"]
            if task_id in summary_by_task:
                raise ValueError(f"Task appears in multiple checkpoints: {task_id}")
            summary_by_task[task_id] = row
        for row in fixed_rows:
            fixed_by_task[row["TaskId"]].append(row)
        for row in native_rows:
            native_by_task[row["TaskId"]].append(row)
    return summary_by_task, fixed_by_task, native_by_task, group_marker_hashes


def override_identity(
    row: dict[str, str],
    assignment: dict[str, str],
    source_row: int,
) -> dict[str, object]:
    result: dict[str, object] = dict(row)
    result.update(
        {
            "TaskId": assignment["task_id"],
            "TaskKeySha256": assignment["task_key_sha256"],
            "CurveId": source_row,
            "ReplaySourceRow": source_row,
            "GeologyId": assignment["geology_id"],
            "ScenarioName": assignment["scenario_name"],
            "CaseLabel": assignment["case_label"],
            "Level3CaseId": int(assignment["case_id"]),
            "Level3CaseName": assignment["case_name"],
            "Window": assignment["window"],
            "SliceIndex": int(assignment["slice_index"]),
            "AssignedState": assignment["assigned_state"],
            "SamplingPool": assignment["sampling_pool"],
            "SelectedSampleIndex": int(assignment["selected_sample_index"]),
            "ReplaySeed": int(assignment["exact_replay_seed"]),
        }
    )
    return result


def replay_template_row(
    assignment: dict[str, str],
    source_row: int,
) -> dict[str, object]:
    return {
        "SourceRow": source_row,
        "TaskId": assignment["task_id"],
        "TaskKeySha256": assignment["task_key_sha256"],
        "GeologyId": assignment["geology_id"],
        "ScenarioIndex": int(assignment["scenario_index"]),
        "ScenarioLabel": assignment["scenario_label"],
        "ScenarioName": assignment["scenario_name"],
        "CaseIndex": int(assignment["case_index"]),
        "CaseLabel": assignment["case_label"],
        "FaultingDepthM": float(assignment["faulting_depth_m"]),
        "SandVcl": float(assignment["sand_vcl"]),
        "ClayVcl": float(assignment["clay_vcl"]),
        "Level3CaseId": int(assignment["case_id"]),
        "Level3CaseName": assignment["case_name"],
        "CaseCategory": assignment["case_category"],
        "CaseStrength": assignment["case_strength"],
        "PatternName": assignment["pattern_name"],
        "Orientation": assignment["orientation"],
        "Window": assignment["window"],
        "SliceIndex": int(assignment["slice_index"]),
        "DrawGroupIndex": int(assignment["draw_group_index"]),
        "AssignedState": assignment["assigned_state"],
        "SamplingMode": assignment["sampling_mode"],
        "SamplingPool": assignment["sampling_pool"],
        "SelectedSampleIndex": int(assignment["selected_sample_index"]),
        "ReplaySeed": int(assignment["exact_replay_seed"]),
        "LogKxx": float(assignment["log_kxx"]),
        "LogKyy": float(assignment["log_kyy"]),
        "LogKzz": float(assignment["log_kzz"]),
        "VerificationStatus": "not_replayed_not_required",
        "MaxAbsLog10Diff": "",
        "OutputFile": "",
    }


def choose_swi_medoids(
    assignments: list[dict[str, str]],
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for window in WINDOWS:
        candidates = [
            (assignment, summary)
            for assignment, summary in zip(assignments, summaries, strict=True)
            if assignment["window"] == window
        ]
        if len(candidates) != 87:
            raise ValueError(f"Expected 87 Swi candidates for {window}")
        values = [float(summary["EffectiveSwi"]) for _, summary in candidates]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid effective Swi values for {window}")
        target = statistics.median(values)
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                abs(values[item[0]] - target),
                int(item[1][1]["ReplaySourceRow"]),
                int(item[1][0]["slice_index"]),
            ),
        )
        local_index, (assignment, summary) = ranked[0]
        result: dict[str, object] = dict(assignment)
        result.update(
            {
                "case_source_row": int(summary["ReplaySourceRow"]),
                "swi_medoid_target": target,
                "selected_effective_swi": values[local_index],
                "absolute_swi_distance": abs(values[local_index] - target),
                "candidate_count": len(candidates),
            }
        )
        selected.append(result)
    return selected


def assemble_case(
    case_id: int,
    assignments: list[dict[str, str]],
    summary_by_task: dict[str, dict[str, str]],
    fixed_by_task: dict[str, list[dict[str, str]]],
    native_by_task: dict[str, list[dict[str, str]]],
    output_root: Path,
    marker_hashes: dict[str, str],
    overwrite: bool,
) -> dict:
    geology_id = assignments[0]["geology_id"]
    case_dir = output_root / "cases" / geology_id / f"case{case_id:02d}"
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Case output exists: {case_dir}")
        shutil.rmtree(case_dir)
    input_dir = case_dir / "inputs"
    input_dir.mkdir(parents=True)

    summary_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    replay_template: list[dict[str, object]] = []

    for source_row, assignment in enumerate(assignments, start=1):
        task_id = assignment["task_id"]
        if task_id not in summary_by_task:
            raise ValueError(f"Missing checkpoint Pc summary for task {task_id}")
        if task_id not in fixed_by_task or task_id not in native_by_task:
            raise ValueError(f"Missing checkpoint Pc curve for task {task_id}")
        summary_rows.append(
            override_identity(summary_by_task[task_id], assignment, source_row)
        )
        fixed_rows.extend(
            override_identity(row, assignment, source_row)
            for row in fixed_by_task[task_id]
        )
        native_rows.extend(
            override_identity(row, assignment, source_row)
            for row in native_by_task[task_id]
        )
        replay_template.append(replay_template_row(assignment, source_row))

    representative_rows = choose_swi_medoids(assignments, summary_rows)
    token = f"{geology_id}_case{case_id:02d}"
    summary_path = input_dir / f"pc_curve_summary_{token}_ip_full87.csv"
    fixed_path = input_dir / f"pc_curve_points_{token}_ip_full87.csv"
    native_path = input_dir / f"pc_native_curve_points_{token}_ip_full87.csv"
    template_path = input_dir / f"replay_summary_template_{token}.csv"
    representative_path = input_dir / f"kr_representative_replay_{token}.csv"

    write_rows(summary_path, list(summary_rows[0].keys()), summary_rows)
    write_rows(fixed_path, list(fixed_rows[0].keys()), fixed_rows)
    write_rows(native_path, list(native_rows[0].keys()), native_rows)
    write_rows(template_path, list(replay_template[0].keys()), replay_template)
    write_rows(
        representative_path,
        list(representative_rows[0].keys()),
        representative_rows,
    )

    files = {}
    for path in (
        summary_path,
        fixed_path,
        native_path,
        template_path,
        representative_path,
    ):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "status": "assembled",
        "geology_id": geology_id,
        "case_id": case_id,
        "case_name": assignments[0]["case_name"],
        "assignment_count": len(assignments),
        "slice_count": 87,
        "window_count": 6,
        "representative_replay_count": len(representative_rows),
        "swi_medoid_slices": {
            row["window"]: int(row["slice_index"]) for row in representative_rows
        },
        "checkpoint_marker_sha256": marker_hashes,
        "files": files,
    }
    (case_dir / "case_inputs.done.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    case_ids = parse_case_ids(args.case_ids)
    checkpoint_root = args.checkpoint_output_root.resolve()
    output_root = args.output_root.resolve()

    if args.geology_assignment_csv:
        if args.sampling_csv or args.assignment_to_task_csv:
            raise ValueError(
                "Use --geology-assignment-csv or the two full-table inputs, not both"
            )
        cases = load_combined_geology_assignments(
            require_file(args.geology_assignment_csv.resolve()),
            args.geology_id,
            set(case_ids),
        )
    else:
        if not args.sampling_csv or not args.assignment_to_task_csv:
            raise ValueError(
                "--sampling-csv and --assignment-to-task-csv are required "
                "when --geology-assignment-csv is omitted"
            )
        cases = load_geology_assignments(
            require_file(args.sampling_csv.resolve()),
            require_file(args.assignment_to_task_csv.resolve()),
            args.geology_id,
            set(case_ids),
        )
    summary_by_task, fixed_by_task, native_by_task, marker_hashes = (
        load_checkpoint_products(checkpoint_root, args.geology_id)
    )

    manifests = []
    for case_id in case_ids:
        manifests.append(
            assemble_case(
                case_id,
                cases[case_id],
                summary_by_task,
                fixed_by_task,
                native_by_task,
                output_root,
                marker_hashes,
                args.overwrite,
            )
        )
    summary = {
        "status": "complete",
        "geology_id": args.geology_id,
        "case_count": len(manifests),
        "case_ids": case_ids,
        "representative_replay_count": sum(
            item["representative_replay_count"] for item in manifests
        ),
    }
    geology_dir = output_root / "cases" / args.geology_id
    (geology_dir / "geology_case_inputs.done.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
