#!/usr/bin/env python3
"""Build deterministic, load-balanced lanes for missing checkpoint work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lane-count", required=True, type=int)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Limit selected missing groups; zero selects every missing group.",
    )
    parser.add_argument(
        "--selection",
        choices=("all", "largest"),
        default="all",
        help="Use all missing groups or stress-test the largest groups first.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def read_groups(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "group_index",
        "group_id",
        "geology_id",
        "window",
        "checkpoint_sha256",
        "task_count",
        "usage_count",
    }
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Checkpoint manifest lacks fields: {sorted(missing)}")
    return rows


def marker_is_current(
    marker_path: Path,
    row: dict[str, str],
    identity: dict[str, object],
) -> bool:
    if not marker_path.is_file():
        return False
    try:
        marker = read_json(marker_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    expected = {
        "status": "complete",
        "group_id": row["group_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "physics_commit": identity["physics_commit"],
        "method_config_sha256": identity["production_method_config_sha256"],
    }
    return all(marker.get(key) == value for key, value in expected.items())


def select_missing_groups(
    rows: list[dict[str, str]],
    run_root: Path,
    identity: dict[str, object],
    selection: str,
    max_groups: int,
) -> tuple[list[dict[str, str]], int]:
    missing_rows = [
        row
        for row in rows
        if not marker_is_current(
            run_root
            / "checkpoint_pc"
            / row["group_id"]
            / "checkpoint.done.json",
            row,
            identity,
        )
    ]
    if selection == "largest":
        ordered = sorted(
            missing_rows,
            key=lambda row: (-int(row["task_count"]), int(row["group_index"])),
        )
    else:
        ordered = sorted(missing_rows, key=lambda row: int(row["group_index"]))
    if max_groups > 0:
        ordered = ordered[:max_groups]
    return ordered, len(missing_rows)


def assign_lanes(
    rows: list[dict[str, str]], lane_count: int
) -> list[list[dict[str, str]]]:
    actual_lane_count = min(lane_count, len(rows))
    lanes: list[list[dict[str, str]]] = [[] for _ in range(actual_lane_count)]
    loads = [0] * actual_lane_count
    for row in sorted(
        rows,
        key=lambda item: (-int(item["task_count"]), int(item["group_index"])),
    ):
        lane_index = min(range(actual_lane_count), key=lambda value: (loads[value], value))
        lanes[lane_index].append(row)
        loads[lane_index] += int(row["task_count"])
    for lane in lanes:
        lane.sort(key=lambda row: int(row["group_index"]))
    return lanes


def main() -> int:
    args = parse_args()
    if args.lane_count <= 0:
        raise SystemExit("--lane-count must be positive")
    if args.max_groups < 0:
        raise SystemExit("--max-groups cannot be negative")

    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    groups_path = run_root / "checkpoint_manifest" / "checkpoint_groups.csv"
    identity_path = run_root / "phase_run_identity.json"
    if not groups_path.is_file() or not identity_path.is_file():
        raise SystemExit("The run root lacks its checkpoint manifest or run identity")

    identity = read_json(identity_path)
    rows = read_groups(groups_path)
    selected, total_missing = select_missing_groups(
        rows, run_root, identity, args.selection, args.max_groups
    )
    if not selected:
        raise SystemExit("No missing checkpoint groups were selected")
    lanes = assign_lanes(selected, args.lane_count)

    output_root.mkdir(parents=True, exist_ok=False)
    lane_path = output_root / "checkpoint_lanes.csv"
    fieldnames = [
        "lane_id",
        "lane_position",
        "group_index",
        "group_id",
        "geology_id",
        "window",
        "task_count",
        "usage_count",
    ]
    with lane_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for lane_id, lane in enumerate(lanes, start=1):
            for position, row in enumerate(lane, start=1):
                writer.writerow(
                    {
                        "lane_id": lane_id,
                        "lane_position": position,
                        "group_index": row["group_index"],
                        "group_id": row["group_id"],
                        "geology_id": row["geology_id"],
                        "window": row["window"],
                        "task_count": row["task_count"],
                        "usage_count": row["usage_count"],
                    }
                )

    lane_loads = [sum(int(row["task_count"]) for row in lane) for lane in lanes]
    lane_groups = [len(lane) for lane in lanes]
    metadata = {
        "schema_version": "checkpoint_lane_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "source_manifest": str(groups_path),
        "source_manifest_sha256": sha256_file(groups_path),
        "run_identity_sha256": sha256_file(identity_path),
        "physics_commit": identity["physics_commit"],
        "method_config_sha256": identity["production_method_config_sha256"],
        "selection": args.selection,
        "total_group_count": len(rows),
        "total_missing_group_count": total_missing,
        "selected_group_count": len(selected),
        "unselected_missing_group_count": total_missing - len(selected),
        "requested_lane_count": args.lane_count,
        "lane_count": len(lanes),
        "lane_group_counts": lane_groups,
        "lane_task_loads": lane_loads,
        "minimum_lane_task_load": min(lane_loads),
        "maximum_lane_task_load": max(lane_loads),
        "checkpoint_lanes_sha256": sha256_file(lane_path),
    }
    metadata_path = output_root / "checkpoint_lane_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
