#!/usr/bin/env python3
"""Verify a generated production manifest and optionally its frozen inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--predict-root", type=Path)
    parser.add_argument("--sampling-root", type=Path)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inventory(inventory: Path, target_root: Path) -> tuple[int, int]:
    checked = 0
    checked_bytes = 0
    with inventory.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            path = target_root / Path(row["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size != int(row["bytes"]):
                raise ValueError(f"Size mismatch: {path}")
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            checked += 1
            checked_bytes += size
    return checked, checked_bytes


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as stream:
        return max(sum(1 for _ in stream) - 1, 0)


def main() -> int:
    args = parse_args()
    root = args.manifest_root.resolve()
    metadata = json.loads((root / "freeze_metadata.json").read_text(encoding="utf-8"))

    generated_count, generated_bytes = verify_inventory(
        root / "generated_files_sha256.csv", root
    )
    task_count = count_csv_rows(root / "manifests" / "unique_replay_pc_tasks.csv")
    assignment_count = count_csv_rows(root / "manifests" / "assignment_to_task.csv")
    shard_rows = 0
    shard_count = 0
    with (root / "manifests" / "task_shards.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            shard = root / "manifests" / "task_shards" / row["file_name"]
            if sha256_file(shard) != row["sha256"]:
                raise ValueError(f"Shard SHA-256 mismatch: {shard}")
            local_count = count_csv_rows(shard)
            if local_count != int(row["task_count"]):
                raise ValueError(f"Shard row-count mismatch: {shard}")
            shard_rows += local_count
            shard_count += 1

    expected_tasks = int(metadata["unique_replay_pc_task_count"])
    expected_assignments = int(metadata["assignment_count"])
    if task_count != expected_tasks or shard_rows != expected_tasks:
        raise ValueError(
            f"Task-count mismatch: combined={task_count}, shards={shard_rows}, "
            f"expected={expected_tasks}"
        )
    if assignment_count != expected_assignments:
        raise ValueError(
            f"Assignment-count mismatch: {assignment_count} != {expected_assignments}"
        )
    if shard_count != int(metadata["shard_count"]):
        raise ValueError("Shard-count mismatch")

    results = {
        "generated_files_checked": generated_count,
        "generated_bytes_checked": generated_bytes,
        "task_count": task_count,
        "assignment_count": assignment_count,
        "shard_count": shard_count,
    }
    if args.predict_root:
        count, size = verify_inventory(
            root / "inventory" / "predict_files_sha256.csv",
            args.predict_root.resolve(),
        )
        results["predict_files_checked"] = count
        results["predict_bytes_checked"] = size
    if args.sampling_root:
        count, size = verify_inventory(
            root / "inventory" / "sampling_files_sha256.csv",
            args.sampling_root.resolve(),
        )
        results["sampling_files_checked"] = count
        results["sampling_bytes_checked"] = size

    print(json.dumps(results, indent=2))
    print("Production manifest verification PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

