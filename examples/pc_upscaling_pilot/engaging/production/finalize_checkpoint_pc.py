#!/usr/bin/env python3
"""Validate and publish compact Pc products for one checkpoint work unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--replay-summary-csv", required=True, type=Path)
    parser.add_argument("--pc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--physics-commit", required=True)
    parser.add_argument("--method-config-sha256", required=True)
    parser.add_argument("--replay-tolerance-log10", type=float, default=1.0e-3)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one file matching {root / pattern}, found {len(matches)}"
        )
    return matches[0]


def finite_float(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def write_rows(
    destination: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_task_identity(
    source: Path,
    destination: Path,
    task_by_source_row: dict[int, tuple[str, str]],
) -> tuple[int, set[str]]:
    output_rows: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "ReplaySourceRow" not in (reader.fieldnames or []):
            raise ValueError(f"{source} lacks ReplaySourceRow")
        fieldnames = ["TaskId", "TaskKeySha256"] + list(reader.fieldnames or [])
        for row_id, row in enumerate(reader, start=1):
            source_row = int(float(row["ReplaySourceRow"]))
            if source_row not in task_by_source_row:
                raise ValueError(
                    f"{source} row {row_id} has unknown ReplaySourceRow "
                    f"{source_row}"
                )
            task_id, task_key = task_by_source_row[source_row]
            output_rows.append(
                {"TaskId": task_id, "TaskKeySha256": task_key, **row}
            )
            seen_tasks.add(task_id)
    write_rows(destination, output_rows, fieldnames)
    return len(output_rows), seen_tasks


def validate_summary(
    summary_rows: list[dict[str, str]],
    expected_tasks: set[str],
) -> None:
    if len(summary_rows) != len(expected_tasks):
        raise ValueError(
            f"Pc summary has {len(summary_rows)} rows; "
            f"expected {len(expected_tasks)}"
        )
    seen: set[str] = set()
    for row in summary_rows:
        task_id = row["TaskId"]
        if task_id in seen:
            raise ValueError(f"Duplicate Pc summary task: {task_id}")
        seen.add(task_id)
        porosity = finite_float(row["UpscaledPorosity"], "UpscaledPorosity")
        bulk_sg_max = finite_float(row["BulkSgMax"], "BulkSgMax")
        effective_swi = finite_float(row["EffectiveSwi"], "EffectiveSwi")
        pc_min = finite_float(row["PcMinPa"], "PcMinPa")
        pc_max = finite_float(row["PcMaxPa"], "PcMaxPa")
        if not 0.0 < porosity < 1.0:
            raise ValueError(f"Invalid porosity for {task_id}: {porosity}")
        if not 0.0 <= bulk_sg_max <= 1.0:
            raise ValueError(f"Invalid BulkSgMax for {task_id}: {bulk_sg_max}")
        if not 0.0 <= effective_swi <= 1.0:
            raise ValueError(f"Invalid EffectiveSwi for {task_id}: {effective_swi}")
        if abs(effective_swi - (1.0 - bulk_sg_max)) > 1.0e-8:
            raise ValueError(f"Pc endpoint mismatch for {task_id}")
        if not 0.0 < pc_min <= pc_max:
            raise ValueError(f"Invalid Pc range for {task_id}: {pc_min}, {pc_max}")
    if seen != expected_tasks:
        raise ValueError("Pc summary task coverage does not match selection")


def main() -> int:
    args = parse_args()
    selection_path = require_file(args.selection_csv.resolve())
    replay_path = require_file(args.replay_summary_csv.resolve())
    pc_root = args.pc_root.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "checkpoint.done.json"

    if done_path.is_file():
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") == "complete"
            and marker.get("group_id") == args.group_id
            and marker.get("checkpoint_sha256") == args.checkpoint_sha256
            and marker.get("physics_commit") == args.physics_commit
            and marker.get("method_config_sha256")
            == args.method_config_sha256
        ):
            print(f"Checkpoint group already complete: {args.group_id}")
            return 0
        raise ValueError(f"Existing done marker does not match: {done_path}")
    if output_dir.exists():
        if not args.overwrite_incomplete:
            raise FileExistsError(
                f"Incomplete output exists; pass --overwrite-incomplete: {output_dir}"
            )
        shutil.rmtree(output_dir)

    selection_rows = read_rows(selection_path)
    replay_rows = read_rows(replay_path)
    if not selection_rows:
        raise ValueError("Selection is empty")
    if len(replay_rows) != len(selection_rows):
        raise ValueError(
            f"Replay summary has {len(replay_rows)} rows; "
            f"selection has {len(selection_rows)}"
        )

    task_by_source_row: dict[int, tuple[str, str]] = {}
    expected_tasks: set[str] = set()
    for row_id, row in enumerate(selection_rows, start=1):
        task_id = row["task_id"]
        task_key = row["task_key_sha256"]
        if task_id in expected_tasks:
            raise ValueError(f"Duplicate selected task: {task_id}")
        expected_tasks.add(task_id)
        task_by_source_row[row_id] = (task_id, task_key)

    replay_by_source = {
        int(float(row["SourceRow"])): row for row in replay_rows
    }
    if set(replay_by_source) != set(task_by_source_row):
        raise ValueError("Replay SourceRow coverage does not match selection")
    max_replay_diff = 0.0
    for source_row, row in replay_by_source.items():
        if row["VerificationStatus"] != "matched":
            raise ValueError(f"Replay task {source_row} did not match")
        diff = finite_float(row["MaxAbsLog10Diff"], "MaxAbsLog10Diff")
        if diff > args.replay_tolerance_log10:
            raise ValueError(
                f"Replay task {source_row} exceeds tolerance: {diff}"
            )
        max_replay_diff = max(max_replay_diff, diff)

    summary_source = find_single(
        pc_root / "tables", "pc_curve_summary_*_ip_full87.csv"
    )
    curve_source = find_single(
        pc_root / "curves", "pc_curve_points_*_ip_full87.csv"
    )
    native_source = find_single(
        pc_root / "curves", "pc_native_curve_points_*_ip_full87.csv"
    )
    mat_source = find_single(pc_root / "curves", "pc_curves_*_ip_full87.mat")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.group_id}.partial.", dir=output_dir.parent)
    )
    try:
        shutil.copy2(selection_path, partial_dir / "selection.csv")

        replay_output_rows: list[dict[str, str]] = []
        replay_fields = ["TaskId", "TaskKeySha256"] + list(
            replay_rows[0].keys()
        )
        for source_row in sorted(replay_by_source):
            task_id, task_key = task_by_source_row[source_row]
            row = dict(replay_by_source[source_row])
            # Fine-map files are temporary by design. Keep only their
            # verified provenance, not an invalid post-cleanup path.
            row["OutputFile"] = ""
            replay_output_rows.append(
                {"TaskId": task_id, "TaskKeySha256": task_key, **row}
            )
        write_rows(
            partial_dir / "replay_verification_by_task.csv",
            replay_output_rows,
            replay_fields,
        )

        summary_count, summary_tasks = add_task_identity(
            summary_source,
            partial_dir / "pc_summary_by_task.csv",
            task_by_source_row,
        )
        curve_count, curve_tasks = add_task_identity(
            curve_source,
            partial_dir / "pc_curve_points_by_task.csv",
            task_by_source_row,
        )
        native_count, native_tasks = add_task_identity(
            native_source,
            partial_dir / "pc_native_curve_points_by_task.csv",
            task_by_source_row,
        )
        summary_rows = read_rows(partial_dir / "pc_summary_by_task.csv")
        validate_summary(summary_rows, expected_tasks)
        if summary_tasks != expected_tasks:
            raise ValueError("Pc summary task coverage is incomplete")
        if curve_tasks != expected_tasks:
            raise ValueError("Fixed-grid Pc curve task coverage is incomplete")
        if native_tasks != expected_tasks:
            raise ValueError("Native Pc curve task coverage is incomplete")
        shutil.copy2(mat_source, partial_dir / "pc_curves_by_task.mat")

        files = {}
        for path in sorted(partial_dir.iterdir()):
            if path.is_file():
                files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        marker = {
            "status": "complete",
            "group_id": args.group_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": args.checkpoint_sha256,
            "physics_commit": args.physics_commit,
            "method_config_sha256": args.method_config_sha256,
            "task_count": len(expected_tasks),
            "pc_summary_row_count": summary_count,
            "pc_fixed_curve_row_count": curve_count,
            "pc_native_curve_row_count": native_count,
            "max_replay_abs_log10_difference": max_replay_diff,
            "replay_tolerance_log10": args.replay_tolerance_log10,
            "fine_replay_maps_retained": False,
            "files": files,
        }
        (partial_dir / "checkpoint.done.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(partial_dir, output_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
