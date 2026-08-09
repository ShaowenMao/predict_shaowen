#!/usr/bin/env python3
"""Tests for deterministic node-bundled checkpoint scheduling."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
PRODUCTION = (
    REPO_ROOT / "examples" / "pc_upscaling_pilot" / "engaging" / "production"
)
BUILDER_PATH = PRODUCTION / "build_checkpoint_lane_manifest.py"
VERIFIER_PATH = PRODUCTION / "verify_checkpoint_lane_completion.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("checkpoint_lane_builder", BUILDER_PATH)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CheckpointNodeBundleTests(unittest.TestCase):
    def make_run(self, root: Path) -> list[dict[str, object]]:
        identity = {
            "physics_commit": "a" * 40,
            "production_method_config_sha256": "b" * 64,
        }
        (root / "phase_run_identity.json").write_text(
            json.dumps(identity), encoding="utf-8"
        )
        rows = [
            {
                "group_index": index,
                "group_id": f"g{index}",
                "geology_id": f"geo{index}",
                "window": "famp1",
                "checkpoint_sha256": str(index) * 64,
                "task_count": task_count,
                "usage_count": 1000 + index,
            }
            for index, task_count in enumerate((90, 80, 70, 60, 50, 40), start=1)
        ]
        write_csv(root / "checkpoint_manifest" / "checkpoint_groups.csv", rows)
        return rows

    def write_done(self, root: Path, row: dict[str, object]) -> None:
        path = root / "checkpoint_pc" / str(row["group_id"]) / "checkpoint.done.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "group_id": row["group_id"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "physics_commit": "a" * 40,
                    "method_config_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )

    def test_largest_selection_skips_current_markers_and_balances_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            rows = self.make_run(root)
            self.write_done(root, rows[0])
            stale_path = root / "checkpoint_pc" / "g2" / "checkpoint.done.json"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

            selected, missing_count = BUILDER.select_missing_groups(
                [dict((key, str(value)) for key, value in row.items()) for row in rows],
                root,
                json.loads((root / "phase_run_identity.json").read_text()),
                "largest",
                4,
            )
            self.assertEqual(missing_count, 5)
            self.assertEqual([row["group_id"] for row in selected], ["g2", "g3", "g4", "g5"])
            lanes = BUILDER.assign_lanes(selected, 2)
            loads = sorted(sum(int(row["task_count"]) for row in lane) for lane in lanes)
            self.assertEqual(loads, [130, 130])

    def test_manifest_and_completion_verifier_enforce_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            rows = self.make_run(root)
            output = Path(temporary) / "lanes"
            subprocess.run(
                [
                    "python",
                    str(BUILDER_PATH),
                    "--run-root",
                    str(root),
                    "--output-root",
                    str(output),
                    "--lane-count",
                    "2",
                    "--max-groups",
                    "2",
                    "--selection",
                    "largest",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.write_done(root, rows[0])
            self.write_done(root, rows[1])
            report_path = output / "completion.json"
            result = subprocess.run(
                [
                    "python",
                    str(VERIFIER_PATH),
                    "--run-root",
                    str(root),
                    "--lane-manifest",
                    str(output / "checkpoint_lanes.csv"),
                    "--output-json",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(report_path.read_text())["passed"])

            marker = root / "checkpoint_pc" / "g2" / "checkpoint.done.json"
            payload = json.loads(marker.read_text())
            payload["method_config_sha256"] = "c" * 64
            marker.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [
                    "python",
                    str(VERIFIER_PATH),
                    "--run-root",
                    str(root),
                    "--lane-manifest",
                    str(output / "checkpoint_lanes.csv"),
                    "--output-json",
                    str(report_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(report_path.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
