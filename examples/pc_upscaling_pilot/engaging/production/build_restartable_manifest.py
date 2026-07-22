#!/usr/bin/env python3
"""Build a deduplicated, restartable replay/Pc production manifest.

The input is the completed 87-slice sampling table. Every assignment is mapped
to a content-addressed task keyed by the PREDICT checkpoint, selected sample,
exact replay seed, committed code, and production method configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
REQUIRED_COLUMNS = {
    "geology_id",
    "case_id",
    "case_name",
    "case_category",
    "slice_index",
    "window",
    "draw_group_index",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "selected_sample_index",
    "source_checkpoint_file",
    "source_seed_base",
    "exact_replay_seed",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx",
    "perm_kyy",
    "perm_kzz",
}

TASK_HEADER = [
    "task_index",
    "task_id",
    "task_key_sha256",
    "shard_id",
    "code_commit",
    "method_config_sha256",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "selected_sample_index",
    "exact_replay_seed",
    "source_seed_base",
    "geology_id",
    "window",
    "expected_log_kxx",
    "expected_log_kyy",
    "expected_log_kzz",
    "expected_perm_kxx_md",
    "expected_perm_kyy_md",
    "expected_perm_kzz_md",
    "usage_count",
    "replay_output_relative_path",
    "pc_output_relative_path",
    "replay_done_marker_relative_path",
    "pc_done_marker_relative_path",
]

ASSIGNMENT_HEADER = [
    "assignment_index",
    "geology_id",
    "case_id",
    "case_name",
    "case_category",
    "slice_index",
    "window",
    "draw_group_index",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "task_id",
    "task_key_sha256",
    "selected_sample_index",
    "exact_replay_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-root", required=True, type=Path)
    parser.add_argument("--predict-root", required=True, type=Path)
    parser.add_argument("--method-config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--remote-freeze-root", default="")
    parser.add_argument("--tasks-per-shard", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def canonical_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_inventory(root: Path, destination: Path) -> tuple[int, int, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    aggregate = hashlib.sha256()
    total_bytes = 0
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in files:
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
            writer.writerow([relative, size, digest])
            aggregate.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
            total_bytes += size
    return len(files), total_bytes, aggregate.hexdigest()


def load_checkpoint_hashes(inventory_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if Path(row["relative_path"]).name == "predict_runs.mat":
                result[row["relative_path"]] = row["sha256"]
    return result


def source_relative_path(source: str, predict_root: Path) -> str:
    source_path = Path(source).resolve()
    try:
        return source_path.relative_to(predict_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Checkpoint lies outside predict root: {source_path}"
        ) from error


def task_identity(
    row: dict[str, str],
    checkpoint_relative_path: str,
    checkpoint_sha256: str,
    code_commit: str,
    method_hash: str,
) -> tuple[str, str]:
    key_fields = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_relative_path": checkpoint_relative_path,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_sample_index": int(row["selected_sample_index"]),
        "exact_replay_seed": int(row["exact_replay_seed"]),
        "code_commit": code_commit,
        "method_config_sha256": method_hash,
    }
    task_key = hashlib.sha256(canonical_json(key_fields)).hexdigest()
    return task_key, f"rpc_{task_key[:32]}"


def expected_fingerprint(row: dict[str, str]) -> str:
    values = {
        key: row[key]
        for key in (
            "source_seed_base",
            "log_kxx",
            "log_kyy",
            "log_kzz",
            "perm_kxx",
            "perm_kyy",
            "perm_kzz",
        )
    }
    return hashlib.sha256(canonical_json(values)).hexdigest()


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE tasks (
            task_key TEXT PRIMARY KEY,
            task_id TEXT UNIQUE NOT NULL,
            code_commit TEXT NOT NULL,
            method_hash TEXT NOT NULL,
            checkpoint_relative_path TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            selected_sample_index INTEGER NOT NULL,
            exact_replay_seed INTEGER NOT NULL,
            source_seed_base INTEGER NOT NULL,
            geology_id TEXT NOT NULL,
            window_name TEXT NOT NULL,
            log_kxx TEXT NOT NULL,
            log_kyy TEXT NOT NULL,
            log_kzz TEXT NOT NULL,
            perm_kxx TEXT NOT NULL,
            perm_kyy TEXT NOT NULL,
            perm_kzz TEXT NOT NULL,
            expected_fingerprint TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 1,
            mismatch_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return connection


def insert_task(
    connection: sqlite3.Connection,
    row: dict[str, str],
    task_key: str,
    task_id: str,
    checkpoint_relative_path: str,
    checkpoint_sha256: str,
    code_commit: str,
    method_hash: str,
) -> None:
    fingerprint = expected_fingerprint(row)
    connection.execute(
        """
        INSERT INTO tasks VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0
        )
        ON CONFLICT(task_key) DO UPDATE SET
            usage_count = usage_count + 1,
            mismatch_count = mismatch_count +
                CASE WHEN expected_fingerprint = excluded.expected_fingerprint
                     THEN 0 ELSE 1 END
        """,
        (
            task_key,
            task_id,
            code_commit,
            method_hash,
            checkpoint_relative_path,
            checkpoint_sha256,
            int(row["selected_sample_index"]),
            int(row["exact_replay_seed"]),
            int(row["source_seed_base"]),
            row["geology_id"],
            row["window"],
            row["log_kxx"],
            row["log_kyy"],
            row["log_kzz"],
            row["perm_kxx"],
            row["perm_kyy"],
            row["perm_kzz"],
            fingerprint,
        ),
    )


def task_row(db_row: tuple, task_index: int, tasks_per_shard: int) -> list:
    (
        task_key,
        task_id,
        code_commit,
        method_hash,
        checkpoint_relative_path,
        checkpoint_sha256,
        selected_sample_index,
        exact_replay_seed,
        source_seed_base,
        geology_id,
        window_name,
        log_kxx,
        log_kyy,
        log_kzz,
        perm_kxx,
        perm_kyy,
        perm_kzz,
        usage_count,
    ) = db_row
    shard_id = (task_index - 1) // tasks_per_shard + 1
    prefix = task_key[:2]
    return [
        task_index,
        task_id,
        task_key,
        shard_id,
        code_commit,
        method_hash,
        checkpoint_relative_path,
        checkpoint_sha256,
        selected_sample_index,
        exact_replay_seed,
        source_seed_base,
        geology_id,
        window_name,
        log_kxx,
        log_kyy,
        log_kzz,
        perm_kxx,
        perm_kyy,
        perm_kzz,
        usage_count,
        f"artifacts/replay/{prefix}/{task_id}.mat",
        f"artifacts/pc/{prefix}/{task_id}.mat",
        f"status/replay/{prefix}/{task_id}.done.json",
        f"status/pc/{prefix}/{task_id}.done.json",
    ]


def export_tasks(
    connection: sqlite3.Connection,
    manifest_root: Path,
    tasks_per_shard: int,
) -> tuple[int, int]:
    combined_path = manifest_root / "unique_replay_pc_tasks.csv"
    shards_root = manifest_root / "task_shards"
    shards_root.mkdir()
    shard_index_path = manifest_root / "task_shards.csv"

    query = """
        SELECT task_key, task_id, code_commit, method_hash,
               checkpoint_relative_path, checkpoint_sha256,
               selected_sample_index, exact_replay_seed, source_seed_base,
               geology_id, window_name, log_kxx, log_kyy, log_kzz,
               perm_kxx, perm_kyy, perm_kzz, usage_count
        FROM tasks ORDER BY task_key
    """

    shard_stream = None
    shard_writer = None
    shard_rows = 0
    shard_id = 0
    shard_summaries: list[list] = []
    task_count = 0

    with combined_path.open("w", newline="", encoding="utf-8") as combined_stream:
        combined_writer = csv.writer(combined_stream)
        combined_writer.writerow(TASK_HEADER)
        for task_count, db_row in enumerate(connection.execute(query), start=1):
            expected_shard = (task_count - 1) // tasks_per_shard + 1
            if expected_shard != shard_id:
                if shard_stream is not None:
                    shard_stream.close()
                    shard_summaries.append(
                        [shard_id, shard_rows, shard_path.name, sha256_file(shard_path)]
                    )
                shard_id = expected_shard
                shard_rows = 0
                shard_path = shards_root / f"replay_pc_tasks_{shard_id:05d}.csv"
                shard_stream = shard_path.open("w", newline="", encoding="utf-8")
                shard_writer = csv.writer(shard_stream)
                shard_writer.writerow(TASK_HEADER)

            row = task_row(db_row, task_count, tasks_per_shard)
            combined_writer.writerow(row)
            shard_writer.writerow(row)
            shard_rows += 1

    if shard_stream is not None:
        shard_stream.close()
        shard_summaries.append(
            [shard_id, shard_rows, shard_path.name, sha256_file(shard_path)]
        )

    with shard_index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shard_id", "task_count", "file_name", "sha256"])
        writer.writerows(shard_summaries)
    return task_count, len(shard_summaries)


def write_generated_inventory(root: Path) -> None:
    destination = root / "generated_files_sha256.csv"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != destination
        and path.name != "manifest_build.sqlite"
        and not path.name.startswith("manifest_build.sqlite-")
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in files:
            writer.writerow(
                [path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path)]
            )


def main() -> int:
    args = parse_args()
    if args.tasks_per_shard <= 0:
        raise ValueError("--tasks-per-shard must be positive")

    sampling_root = args.sampling_root.resolve()
    predict_root = args.predict_root.resolve()
    method_config = args.method_config.resolve()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    sampling_csv = sampling_root / "texas_field_slice_window_values.csv"

    for path in (sampling_root, predict_root, method_config, repo_root, sampling_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    inventory_root = output_root / "inventory"
    manifest_root = output_root / "manifests"
    config_root = output_root / "config"
    inventory_root.mkdir()
    manifest_root.mkdir()
    config_root.mkdir()

    code_commit = git_value(repo_root, "rev-parse", "HEAD")
    code_branch = git_value(repo_root, "branch", "--show-current")
    method_hash = sha256_file(method_config)
    shutil.copy2(method_config, config_root / method_config.name)

    print("Hashing PREDICT inputs...", flush=True)
    predict_count, predict_bytes, predict_inventory_hash = write_inventory(
        predict_root, inventory_root / "predict_files_sha256.csv"
    )
    checkpoint_hashes = load_checkpoint_hashes(
        inventory_root / "predict_files_sha256.csv"
    )
    print("Hashing sampling inputs...", flush=True)
    sampling_count, sampling_bytes, sampling_inventory_hash = write_inventory(
        sampling_root, inventory_root / "sampling_files_sha256.csv"
    )

    database_path = output_root / "manifest_build.sqlite"
    connection = initialize_database(database_path)
    assignment_path = manifest_root / "assignment_to_task.csv"
    geology_ids: set[str] = set()
    case_ids: set[tuple[str, int]] = set()
    slice_ids: set[int] = set()
    windows: set[str] = set()
    assignment_count = 0

    print("Building deduplicated task index...", flush=True)
    with sampling_csv.open(newline="", encoding="utf-8") as source_stream:
        reader = csv.DictReader(source_stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Sampling CSV lacks columns: {sorted(missing)}")
        with assignment_path.open("w", newline="", encoding="utf-8") as target_stream:
            writer = csv.writer(target_stream)
            writer.writerow(ASSIGNMENT_HEADER)
            for assignment_count, row in enumerate(reader, start=1):
                relative = source_relative_path(row["source_checkpoint_file"], predict_root)
                if relative not in checkpoint_hashes:
                    raise ValueError(f"Checkpoint absent from inventory: {relative}")
                sample_index = int(row["selected_sample_index"])
                seed_base = int(row["source_seed_base"])
                replay_seed = int(row["exact_replay_seed"])
                if replay_seed != seed_base + sample_index - 1:
                    raise ValueError(
                        f"Replay seed mismatch at assignment {assignment_count}: "
                        f"{replay_seed} != {seed_base} + {sample_index} - 1"
                    )
                task_key, task_id = task_identity(
                    row,
                    relative,
                    checkpoint_hashes[relative],
                    code_commit,
                    method_hash,
                )
                insert_task(
                    connection,
                    row,
                    task_key,
                    task_id,
                    relative,
                    checkpoint_hashes[relative],
                    code_commit,
                    method_hash,
                )
                writer.writerow(
                    [
                        assignment_count,
                        row["geology_id"],
                        row["case_id"],
                        row["case_name"],
                        row["case_category"],
                        row["slice_index"],
                        row["window"],
                        row["draw_group_index"],
                        row["assigned_state"],
                        row["sampling_mode"],
                        row["sampling_pool"],
                        task_id,
                        task_key,
                        sample_index,
                        replay_seed,
                    ]
                )
                geology_ids.add(row["geology_id"])
                case_ids.add((row["geology_id"], int(row["case_id"])))
                slice_ids.add(int(row["slice_index"]))
                windows.add(row["window"])
                if assignment_count % 10000 == 0:
                    connection.commit()
                    print(f"  assignments: {assignment_count:,}", flush=True)
    connection.commit()

    mismatch_count = connection.execute(
        "SELECT COALESCE(SUM(mismatch_count), 0) FROM tasks"
    ).fetchone()[0]
    if mismatch_count:
        raise ValueError(f"Conflicting expected values for {mismatch_count} duplicate tasks")

    print("Exporting task manifest and shards...", flush=True)
    task_count, shard_count = export_tasks(
        connection, manifest_root, args.tasks_per_shard
    )
    connection.close()
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(database_path) + suffix)
        if path.exists():
            path.unlink()

    expected_assignments = 162 * 10 * 87 * 6
    if assignment_count != expected_assignments:
        raise ValueError(
            f"Expected {expected_assignments} assignments, found {assignment_count}"
        )
    if len(geology_ids) != 162 or len(case_ids) != 1620:
        raise ValueError(
            f"Expected 162 geologies/1620 cases, found "
            f"{len(geology_ids)}/{len(case_ids)}"
        )
    if slice_ids != set(range(1, 88)):
        raise ValueError("Slice coverage is not exactly 1:87")
    if windows != {f"famp{i}" for i in range(1, 7)}:
        raise ValueError(f"Unexpected window coverage: {sorted(windows)}")

    restart_policy = {
        "schema_version": SCHEMA_VERSION,
        "completion_rule": (
            "A stage is complete only when its output and .done.json marker exist "
            "and the marker matches task_key_sha256, code_commit, method_config_sha256, "
            "checkpoint_sha256, selected_sample_index, and exact_replay_seed."
        ),
        "replay_resume": "Reuse validated replay output; otherwise rerun replay.",
        "pc_resume": "Reuse validated Pc output only after validated replay; otherwise rerun Pc.",
        "failed_task_policy": "Rerun only failed or unvalidated task IDs from the manifest.",
    }
    (manifest_root / "restart_policy.json").write_text(
        json.dumps(restart_policy, indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "code_branch": code_branch,
        "method_config_file": method_config.name,
        "method_config_sha256": method_hash,
        "remote_freeze_root": args.remote_freeze_root,
        "sampling_source_root": str(sampling_root),
        "predict_source_root": str(predict_root),
        "predict_file_count": predict_count,
        "predict_total_bytes": predict_bytes,
        "predict_inventory_sha256": predict_inventory_hash,
        "sampling_file_count": sampling_count,
        "sampling_total_bytes": sampling_bytes,
        "sampling_inventory_sha256": sampling_inventory_hash,
        "geology_count": len(geology_ids),
        "geology_case_count": len(case_ids),
        "slice_count": len(slice_ids),
        "window_count": len(windows),
        "assignment_count": assignment_count,
        "unique_replay_pc_task_count": task_count,
        "duplicate_assignment_count": assignment_count - task_count,
        "deduplication_fraction": (assignment_count - task_count) / assignment_count,
        "tasks_per_shard": args.tasks_per_shard,
        "shard_count": shard_count,
        "task_key_fields": [
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "selected_sample_index",
            "exact_replay_seed",
            "code_commit",
            "method_config_sha256",
        ],
    }
    (output_root / "freeze_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_generated_inventory(output_root)
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

