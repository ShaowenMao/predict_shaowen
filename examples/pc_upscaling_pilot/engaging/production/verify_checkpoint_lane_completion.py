#!/usr/bin/env python3
"""Validate every checkpoint output selected by a node-bundle lane manifest."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--lane-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    identity = read_json(run_root / "phase_run_identity.json")
    with args.lane_manifest.open(newline="", encoding="utf-8-sig") as stream:
        lane_rows = list(csv.DictReader(stream))
    with (
        run_root / "checkpoint_manifest" / "checkpoint_groups.csv"
    ).open(newline="", encoding="utf-8-sig") as stream:
        groups = {
            row["group_id"]: row
            for row in csv.DictReader(stream)
        }

    failures: list[dict[str, object]] = []
    for lane_row in lane_rows:
        group_id = lane_row["group_id"]
        group = groups.get(group_id)
        marker_path = run_root / "checkpoint_pc" / group_id / "checkpoint.done.json"
        reason = ""
        marker: dict[str, object] = {}
        if group is None:
            reason = "group_missing_from_source_manifest"
        elif not marker_path.is_file():
            reason = "done_marker_missing"
        else:
            try:
                marker = read_json(marker_path)
            except (OSError, json.JSONDecodeError, ValueError):
                reason = "done_marker_invalid_json"
        if group is not None and not reason:
            expected = {
                "status": "complete",
                "group_id": group_id,
                "checkpoint_sha256": group["checkpoint_sha256"],
                "physics_commit": identity["physics_commit"],
                "method_config_sha256": identity[
                    "production_method_config_sha256"
                ],
            }
            mismatches = {
                key: {"expected": value, "observed": marker.get(key)}
                for key, value in expected.items()
                if marker.get(key) != value
            }
            if mismatches:
                reason = "done_marker_contract_mismatch"
                failures.append(
                    {
                        "lane_id": int(lane_row["lane_id"]),
                        "group_index": int(lane_row["group_index"]),
                        "group_id": group_id,
                        "reason": reason,
                        "mismatches": mismatches,
                    }
                )
                continue
        if reason:
            failures.append(
                {
                    "lane_id": int(lane_row["lane_id"]),
                    "group_index": int(lane_row["group_index"]),
                    "group_id": group_id,
                    "reason": reason,
                }
            )

    report = {
        "schema_version": "checkpoint_lane_completion_v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "lane_manifest": str(args.lane_manifest.resolve()),
        "selected_group_count": len(lane_rows),
        "completed_group_count": len(lane_rows) - len(failures),
        "failed_group_count": len(failures),
        "passed": not failures,
        "failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
