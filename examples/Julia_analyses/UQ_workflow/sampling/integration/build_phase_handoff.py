#!/usr/bin/env python3
"""Build one immutable sampling-to-upscaling handoff for a workflow phase.

The handoff contains the human-readable adapted assignments, compact
fault-local permeability MAT, exact replay/Pc task freeze, checkpoint
inventory, and hashes that tie all products to the canonical sampling
manifest. Phase 1 and Phase 2 are built separately because their geology and
case rectangles differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / ".git").exists())
ADAPTER_PATH = HERE / "adapt_manifest_for_upscaling.py"
FREEZER_PATH = HERE / "build_replay_upscaling_freeze.py"
MAT_EXPORTER_PATH = HERE / "export_fault_permeability_mat.jl"
JULIA_PROJECT = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_module("independent_sampling_adapter", ADAPTER_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--geology-catalog", required=True, type=Path)
    parser.add_argument("--predict-root", required=True, type=Path)
    parser.add_argument(
        "--method-config",
        type=Path,
        default=(
            REPO_ROOT
            / "examples"
            / "pc_upscaling_pilot"
            / "engaging"
            / "production"
            / "production_method_config.toml"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--remote-freeze-root", default="")
    parser.add_argument("--tasks-per-shard", type=int, default=5000)
    parser.add_argument("--julia-executable", default="julia")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-sampling-code-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_inventory(root: Path, destination: Path) -> dict[str, object]:
    records: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == destination:
            continue
        relative = path.relative_to(root).as_posix()
        records.append((relative, path.stat().st_size, sha256_file(path)))
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "bytes", "sha256"])
        writer.writerows(records)
    return {
        "file_count": len(records),
        "total_bytes": sum(record[1] for record in records),
        "inventory_sha256": sha256_file(destination),
    }


def run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    inputs = [
        args.manifest.resolve(),
        args.geology_catalog.resolve(),
        args.predict_root.resolve(),
        args.method_config.resolve(),
        args.repo_root.resolve(),
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    temporary = output_root.with_name(f"{output_root.name}.building.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    sampling_root = temporary / "sampling"
    sampling_root.mkdir(parents=True)

    adapted_path = sampling_root / "texas_field_slice_window_values.csv"
    adapted_rows = ADAPTER.adapt_manifest(inputs[0], inputs[1])
    phases = sorted({str(row["phase"]) for row in adapted_rows})
    if len(phases) != 1 or phases[0] not in {"phase1", "phase2"}:
        raise ValueError(
            f"A phase handoff must contain exactly one supported phase, found {phases}"
        )
    ADAPTER.write_adapted_output(
        adapted_rows,
        adapted_path,
        inputs[0],
        inputs[1],
        overwrite=False,
    )

    field_perm_path = sampling_root / "fault_permeability_independent_full_fault_v1.mat"
    run_checked(
        [
            args.julia_executable,
            f"--project={JULIA_PROJECT}",
            str(MAT_EXPORTER_PATH),
            "--sampling-csv",
            str(adapted_path),
            "--output",
            str(field_perm_path),
        ]
    )

    freeze_root = temporary / "replay_upscaling_freeze"
    freeze_command = [
        os.fspath(Path(os.sys.executable)),
        str(FREEZER_PATH),
        "--sampling-csv",
        str(adapted_path),
        "--predict-root",
        str(inputs[2]),
        "--method-config",
        str(inputs[3]),
        "--repo-root",
        str(inputs[4]),
        "--output-root",
        str(freeze_root),
        "--field-permeability-mat",
        str(field_perm_path),
        "--tasks-per-shard",
        str(args.tasks_per_shard),
    ]
    if args.remote_freeze_root:
        freeze_command.extend(["--remote-freeze-root", args.remote_freeze_root])
    if args.allow_dirty:
        freeze_command.append("--allow-dirty")
    if args.allow_sampling_code_mismatch:
        freeze_command.append("--allow-sampling-code-mismatch")
    run_checked(freeze_command)

    freeze_metadata = json.loads(
        (freeze_root / "freeze_metadata.json").read_text(encoding="utf-8")
    )
    inventory_path = temporary / "handoff_files_sha256.csv"
    inventory = write_inventory(temporary, inventory_path)
    metadata = {
        "schema_version": "independent_full_fault_phase_handoff_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_version": "independent_full_fault_v1",
        "phase": phases[0],
        "canonical_manifest": str(inputs[0]),
        "canonical_manifest_sha256": sha256_file(inputs[0]),
        "adapted_sampling_sha256": sha256_file(adapted_path),
        "field_permeability_mat_sha256": sha256_file(field_perm_path),
        "geology_catalog_sha256": sha256_file(inputs[1]),
        "production_method_config_sha256": sha256_file(inputs[3]),
        "geology_count": freeze_metadata["geology_count"],
        "case_count": freeze_metadata["case_count"],
        "assignment_count": freeze_metadata["assignment_count"],
        "unique_replay_pc_task_count": freeze_metadata[
            "unique_replay_pc_task_count"
        ],
        "development_overrides": freeze_metadata["development_overrides"],
        **inventory,
    }
    (temporary / "handoff_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output_root)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
