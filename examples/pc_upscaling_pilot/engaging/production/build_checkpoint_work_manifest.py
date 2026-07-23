#!/usr/bin/env python3
"""Build checkpoint-centered replay/Pc work units from a frozen manifest.

Every work unit contains all unique selected PREDICT realizations belonging
to one geology-window checkpoint. This keeps each job restartable while
avoiding repeated checkpoint/metadata setup across arbitrary task shards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path


SELECTION_COLUMNS = [
    "task_id",
    "task_key_sha256",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "geology_id",
    "scenario_index",
    "scenario_label",
    "scenario_name",
    "case_index",
    "case_label",
    "faulting_depth_m",
    "sand_vcl",
    "clay_vcl",
    "case_id",
    "case_name",
    "case_category",
    "slice_index",
    "window",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "selected_sample_index",
    "source_checkpoint_file",
    "source_seed_base",
    "source_num_attempts",
    "source_num_rejected",
    "exact_replay_seed",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx",
    "perm_kyy",
    "perm_kzz",
]

GROUP_COLUMNS = [
    "group_index",
    "group_id",
    "geology_id",
    "scenario_index",
    "window",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "task_count",
    "usage_count",
    "selection_relative_path",
    "output_relative_path",
    "done_marker_relative_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", required=True, type=Path)
    parser.add_argument("--predict-root", required=True, type=Path)
    parser.add_argument("--sampling-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--case-filter-csv",
        type=Path,
        help="Optional CSV containing geology_id and case_id.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_case_filter(path: Path | None) -> set[tuple[str, int]] | None:
    if path is None:
        return None
    selected: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"geology_id", "case_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Case filter lacks columns: {sorted(missing)}")
        for row in reader:
            selected.add((row["geology_id"], int(row["case_id"])))
    if not selected:
        raise ValueError("Case filter is empty")
    return selected


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            task_key_sha256 TEXT NOT NULL,
            checkpoint_relative_path TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            geology_id TEXT NOT NULL,
            window_name TEXT NOT NULL,
            selected_sample_index INTEGER NOT NULL,
            exact_replay_seed INTEGER NOT NULL,
            source_seed_base INTEGER NOT NULL,
            expected_log_kxx TEXT NOT NULL,
            expected_log_kyy TEXT NOT NULL,
            expected_log_kzz TEXT NOT NULL,
            expected_perm_kxx_md TEXT NOT NULL,
            expected_perm_kyy_md TEXT NOT NULL,
            expected_perm_kzz_md TEXT NOT NULL,
            usage_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE first_use (
            task_id TEXT PRIMARY KEY,
            sampling_json TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE selected_tasks (
            task_id TEXT PRIMARY KEY,
            selected_usage_count INTEGER NOT NULL
        )
        """
    )
    return connection


def load_tasks(connection: sqlite3.Connection, tasks_path: Path) -> int:
    count = 0
    with tasks_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for count, row in enumerate(reader, start=1):
            connection.execute(
                """
                INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                          ?, ?, ?, ?, ?)
                """,
                (
                    row["task_id"],
                    row["task_key_sha256"],
                    row["checkpoint_relative_path"],
                    row["checkpoint_sha256"],
                    row["geology_id"],
                    row["window"],
                    int(row["selected_sample_index"]),
                    int(row["exact_replay_seed"]),
                    int(row["source_seed_base"]),
                    row["expected_log_kxx"],
                    row["expected_log_kyy"],
                    row["expected_log_kzz"],
                    row["expected_perm_kxx_md"],
                    row["expected_perm_kyy_md"],
                    row["expected_perm_kzz_md"],
                    int(row["usage_count"]),
                ),
            )
            if count % 20000 == 0:
                connection.commit()
    connection.commit()
    return count


def load_first_uses(
    connection: sqlite3.Connection,
    assignment_path: Path,
    sampling_path: Path,
    case_filter: set[tuple[str, int]] | None,
) -> tuple[int, int]:
    assignment_count = 0
    selected_assignment_count = 0
    with assignment_path.open(
        newline="", encoding="utf-8-sig"
    ) as assignment_stream, sampling_path.open(
        newline="", encoding="utf-8-sig"
    ) as sampling_stream:
        assignment_reader = csv.DictReader(assignment_stream)
        sampling_reader = csv.DictReader(sampling_stream)
        for assignment_count, (assignment, sampling) in enumerate(
            zip(assignment_reader, sampling_reader, strict=True), start=1
        ):
            if int(assignment["assignment_index"]) != assignment_count:
                raise ValueError(
                    f"Assignment index mismatch at row {assignment_count}"
                )
            if assignment["geology_id"] != sampling["geology_id"]:
                raise ValueError(
                    f"Sampling/manifest geology mismatch at row {assignment_count}"
                )
            if int(assignment["case_id"]) != int(sampling["case_id"]):
                raise ValueError(
                    f"Sampling/manifest case mismatch at row {assignment_count}"
                )
            if assignment["window"] != sampling["window"]:
                raise ValueError(
                    f"Sampling/manifest window mismatch at row {assignment_count}"
                )
            task_id = assignment["task_id"]
            connection.execute(
                "INSERT OR IGNORE INTO first_use VALUES (?, ?)",
                (task_id, json.dumps(sampling, separators=(",", ":"))),
            )
            key = (sampling["geology_id"], int(sampling["case_id"]))
            if case_filter is None or key in case_filter:
                selected_assignment_count += 1
                connection.execute(
                    """
                    INSERT INTO selected_tasks VALUES (?, 1)
                    ON CONFLICT(task_id) DO UPDATE SET
                        selected_usage_count = selected_usage_count + 1
                    """,
                    (task_id,),
                )
            if assignment_count % 20000 == 0:
                connection.commit()
    connection.commit()
    return assignment_count, selected_assignment_count


def remote_checkpoint_path(predict_root: Path, relative_path: str) -> str:
    return str((predict_root / Path(relative_path)).resolve())


def group_id(geology_id: str, window: str) -> str:
    return f"checkpoint_{geology_id}_{window.lower()}"


def export_groups(
    connection: sqlite3.Connection,
    output_root: Path,
    predict_root: Path,
) -> tuple[int, int, int]:
    selection_root = output_root / "checkpoint_selections"
    selection_root.mkdir()
    groups_path = output_root / "checkpoint_groups.csv"
    task_to_group_path = output_root / "task_to_checkpoint_group.csv"

    query = """
        SELECT t.task_id, t.task_key_sha256, t.checkpoint_relative_path,
               t.checkpoint_sha256, t.geology_id, t.window_name,
               t.selected_sample_index, t.exact_replay_seed,
               t.source_seed_base, t.expected_log_kxx,
               t.expected_log_kyy, t.expected_log_kzz,
               t.expected_perm_kxx_md, t.expected_perm_kyy_md,
               t.expected_perm_kzz_md, s.selected_usage_count,
               f.sampling_json
        FROM tasks t
        JOIN selected_tasks s ON s.task_id = t.task_id
        JOIN first_use f ON f.task_id = t.task_id
        ORDER BY t.checkpoint_relative_path, t.selected_sample_index,
                 t.task_id
    """

    group_rows: list[list] = []
    current_checkpoint = ""
    current_writer = None
    current_stream = None
    current_group_id = ""
    current_geology = ""
    current_window = ""
    current_hash = ""
    current_selection_path: Path | None = None
    current_count = 0
    current_usage = 0
    total_tasks = 0
    group_index = 0

    def close_group() -> None:
        nonlocal current_stream
        if current_stream is None or current_selection_path is None:
            return
        current_stream.close()
        group_rows.append(
            [
                group_index,
                current_group_id,
                current_geology,
                int(current_geology[1:3]),
                current_window,
                current_checkpoint,
                current_hash,
                current_count,
                current_usage,
                current_selection_path.relative_to(output_root).as_posix(),
                f"checkpoint_outputs/{current_group_id}",
                f"checkpoint_outputs/{current_group_id}/checkpoint.done.json",
            ]
        )
        current_stream = None

    with task_to_group_path.open(
        "w", newline="", encoding="utf-8"
    ) as mapping_stream:
        mapping_writer = csv.writer(mapping_stream)
        mapping_writer.writerow(
            ["task_id", "task_key_sha256", "group_id", "geology_id", "window"]
        )
        for row in connection.execute(query):
            (
                task_id,
                task_key,
                checkpoint_relative,
                checkpoint_hash,
                geology_id,
                window,
                sample_index,
                replay_seed,
                seed_base,
                log_kxx,
                log_kyy,
                log_kzz,
                perm_kxx,
                perm_kyy,
                perm_kzz,
                selected_usage_count,
                sampling_json,
            ) = row
            if checkpoint_relative != current_checkpoint:
                close_group()
                group_index += 1
                current_checkpoint = checkpoint_relative
                current_hash = checkpoint_hash
                current_geology = geology_id
                current_window = window
                current_group_id = group_id(geology_id, window)
                current_count = 0
                current_usage = 0
                current_selection_path = (
                    selection_root / f"{current_group_id}.csv"
                )
                current_stream = current_selection_path.open(
                    "w", newline="", encoding="utf-8"
                )
                current_writer = csv.DictWriter(
                    current_stream, fieldnames=SELECTION_COLUMNS
                )
                current_writer.writeheader()

            sampling = json.loads(sampling_json)
            # These rows represent unique replay/Pc tasks rather than field
            # assignments. A stable synthetic case/slice context keeps the
            # trusted MATLAB replay/Pc interfaces unchanged.
            selection = {
                "task_id": task_id,
                "task_key_sha256": task_key,
                "checkpoint_relative_path": checkpoint_relative,
                "checkpoint_sha256": checkpoint_hash,
                "geology_id": geology_id,
                "scenario_index": sampling["scenario_index"],
                "scenario_label": sampling["scenario_label"],
                "scenario_name": sampling["scenario_name"],
                "case_index": sampling["case_index"],
                "case_label": sampling["case_label"],
                "faulting_depth_m": sampling["faulting_depth_m"],
                "sand_vcl": sampling["sand_vcl"],
                "clay_vcl": sampling["clay_vcl"],
                "case_id": 1,
                "case_name": "production_unique_replay_pc",
                "case_category": "Production unique replay/Pc task",
                "slice_index": current_count + 1,
                "window": window,
                "assigned_state": sampling["assigned_state"],
                "sampling_mode": sampling["sampling_mode"],
                "sampling_pool": sampling["sampling_pool"],
                "selected_sample_index": sample_index,
                "source_checkpoint_file": remote_checkpoint_path(
                    predict_root, checkpoint_relative
                ),
                "source_seed_base": seed_base,
                "source_num_attempts": sampling["source_num_attempts"],
                "source_num_rejected": sampling["source_num_rejected"],
                "exact_replay_seed": replay_seed,
                "log_kxx": log_kxx,
                "log_kyy": log_kyy,
                "log_kzz": log_kzz,
                "perm_kxx": perm_kxx,
                "perm_kyy": perm_kyy,
                "perm_kzz": perm_kzz,
            }
            current_writer.writerow(selection)
            mapping_writer.writerow(
                [task_id, task_key, current_group_id, geology_id, window]
            )
            current_count += 1
            current_usage += selected_usage_count
            total_tasks += 1
    close_group()

    with groups_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(GROUP_COLUMNS)
        writer.writerows(group_rows)
    return len(group_rows), total_tasks, sum(row[8] for row in group_rows)


def write_inventory(output_root: Path) -> None:
    destination = output_root / "generated_files_sha256.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != destination and path.name != "build.sqlite":
                writer.writerow(
                    [
                        path.relative_to(output_root).as_posix(),
                        path.stat().st_size,
                        sha256_file(path),
                    ]
                )


def main() -> int:
    args = parse_args()
    freeze_root = args.freeze_root.resolve()
    predict_root = args.predict_root.resolve()
    sampling_path = args.sampling_csv.resolve()
    output_root = args.output_root.resolve()
    tasks_path = freeze_root / "manifests" / "unique_replay_pc_tasks.csv"
    assignment_path = freeze_root / "manifests" / "assignment_to_task.csv"
    for path in (tasks_path, assignment_path, sampling_path, predict_root):
        if not path.exists():
            raise FileNotFoundError(path)
    case_filter = read_case_filter(args.case_filter_csv)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    database_path = output_root / "build.sqlite"
    connection = initialize_database(database_path)

    task_count = load_tasks(connection, tasks_path)
    assignment_count, selected_assignment_count = load_first_uses(
        connection, assignment_path, sampling_path, case_filter
    )
    group_count, selected_task_count, selected_usage_count = export_groups(
        connection, output_root, predict_root
    )
    connection.close()
    database_path.unlink(missing_ok=True)
    Path(str(database_path) + "-wal").unlink(missing_ok=True)
    Path(str(database_path) + "-shm").unlink(missing_ok=True)

    metadata = {
        "freeze_root": str(freeze_root),
        "predict_root": str(predict_root),
        "sampling_csv": str(sampling_path),
        "case_filter_csv": (
            str(args.case_filter_csv.resolve()) if args.case_filter_csv else ""
        ),
        "frozen_unique_task_count": task_count,
        "frozen_assignment_count": assignment_count,
        "selected_assignment_count": selected_assignment_count,
        "selected_unique_task_count": selected_task_count,
        "selected_task_usage_count": selected_usage_count,
        "checkpoint_group_count": group_count,
    }
    (output_root / "checkpoint_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_inventory(output_root)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
