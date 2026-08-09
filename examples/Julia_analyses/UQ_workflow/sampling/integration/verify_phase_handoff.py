#!/usr/bin/env python3
"""Verify an immutable independent-full-fault phase handoff after transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument("--expected-geologies", type=int)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--expected-assignments", type=int)
    parser.add_argument("--allow-development", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inventory(
    root: Path,
    inventory_path: Path,
    allowed_unlisted: set[str],
) -> tuple[int, int]:
    listed: set[str] = set()
    total_bytes = 0
    with inventory_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "bytes", "sha256"}
        if required - set(reader.fieldnames or []):
            raise ValueError(f"Invalid inventory header: {inventory_path}")
        for row in reader:
            relative = row["relative_path"].replace("\\", "/")
            if relative in listed:
                raise ValueError(f"Duplicate inventory path: {relative}")
            path = (root / relative).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as error:
                raise ValueError(f"Inventory path escapes root: {relative}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size != int(row["bytes"]):
                raise ValueError(f"Inventory size mismatch: {relative}")
            if sha256_file(path) != row["sha256"].lower():
                raise ValueError(f"Inventory hash mismatch: {relative}")
            listed.add(relative)
            total_bytes += size
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in allowed_unlisted
    }
    if actual != listed:
        missing = sorted(actual - listed)
        stale = sorted(listed - actual)
        raise ValueError(
            "Inventory does not exactly match payload files: "
            f"unlisted={missing}, stale={stale}"
        )
    return len(listed), total_bytes


def expected_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    defaults = {
        "phase1": (162, 2430, 2430 * 6 * 87),
        "phase2": (12, 480, 480 * 6 * 87),
    }[args.phase]
    return (
        args.expected_geologies or defaults[0],
        args.expected_cases or defaults[1],
        args.expected_assignments or defaults[2],
    )


def main() -> int:
    args = parse_args()
    root = args.handoff_root.resolve()
    handoff_metadata_path = root / "handoff_metadata.json"
    handoff_inventory_path = root / "handoff_files_sha256.csv"
    freeze_root = root / "replay_upscaling_freeze"
    freeze_metadata_path = freeze_root / "freeze_metadata.json"
    freeze_inventory_path = freeze_root / "generated_files_sha256.csv"
    for path in (
        handoff_metadata_path,
        handoff_inventory_path,
        freeze_metadata_path,
        freeze_inventory_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    handoff = json.loads(handoff_metadata_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_metadata_path.read_text(encoding="utf-8"))
    if handoff.get("schema_version") != "independent_full_fault_phase_handoff_v1":
        raise ValueError("Unexpected handoff schema")
    if freeze.get("schema_version") != "independent_full_fault_replay_freeze_v1":
        raise ValueError("Unexpected replay-freeze schema")
    if handoff.get("design_version") != "independent_full_fault_v1":
        raise ValueError("Unexpected sampling design")
    if handoff.get("phase") != args.phase:
        raise ValueError(
            f"Handoff phase mismatch: {handoff.get('phase')!r} != {args.phase!r}"
        )
    if freeze.get("phases") != [args.phase]:
        raise ValueError(
            f"Replay-freeze phase mismatch: {freeze.get('phases')!r}"
        )

    inventory_count, inventory_bytes = verify_inventory(
        root,
        handoff_inventory_path,
        {"handoff_files_sha256.csv", "handoff_metadata.json"},
    )
    if sha256_file(handoff_inventory_path) != handoff["inventory_sha256"]:
        raise ValueError("Handoff inventory hash mismatch")
    if inventory_count != int(handoff["file_count"]):
        raise ValueError("Handoff inventory file-count mismatch")
    if inventory_bytes != int(handoff["total_bytes"]):
        raise ValueError("Handoff inventory byte-count mismatch")
    verify_inventory(
        freeze_root,
        freeze_inventory_path,
        {"generated_files_sha256.csv"},
    )

    expected_geologies, expected_cases, expected_assignments = expected_counts(args)
    actual = (
        int(handoff["geology_count"]),
        int(handoff["case_count"]),
        int(handoff["assignment_count"]),
    )
    expected = (expected_geologies, expected_cases, expected_assignments)
    if actual != expected:
        raise ValueError(f"Phase rectangle mismatch: actual={actual}, expected={expected}")
    for key in ("geology_count", "case_count", "assignment_count"):
        if int(handoff[key]) != int(freeze[key]):
            raise ValueError(f"Handoff/freeze count mismatch for {key}")

    overrides = freeze.get("development_overrides", {})
    if not args.allow_development:
        if freeze.get("sampling_code_dirty"):
            raise ValueError("Production handoff was built from a dirty worktree")
        if any(bool(value) for value in overrides.values()):
            raise ValueError("Production handoff records development overrides")
        if freeze.get("sampling_code_commit") != freeze.get(
            "manifest_sampling_code_commit"
        ):
            raise ValueError("Sampling checkout and manifest commits differ")

    sampling_root = freeze_root / "sampling"
    adapted = sampling_root / "texas_field_slice_window_values.csv"
    sidecar = adapted.with_suffix(adapted.suffix + ".metadata.json")
    field_perm = sampling_root / "fault_permeability_independent_full_fault_v1.mat"
    for path in (adapted, sidecar, field_perm):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    config_root = (freeze_root / "config").resolve()
    method_config = (
        config_root / str(freeze["production_method_config_file"])
    ).resolve()
    try:
        method_config.relative_to(config_root)
    except ValueError as error:
        raise ValueError("Production-method configuration escapes freeze root") from error
    checkpoint_inventory = (
        freeze_root / "inventory" / "selected_predict_checkpoints_sha256.csv"
    )
    for path in (method_config, checkpoint_inventory):
        if not path.is_file():
            raise FileNotFoundError(path)
    checks = {
        "adapted_sampling_sha256": sha256_file(adapted),
        "field_permeability_mat_sha256": sha256_file(field_perm),
        "canonical_manifest_sha256": sidecar_data["canonical_manifest_sha256"],
        "geology_catalog_sha256": sidecar_data["geology_catalog_sha256"],
        "production_method_config_sha256": sha256_file(method_config),
    }
    for key, actual_hash in checks.items():
        if handoff.get(key) != actual_hash:
            raise ValueError(f"Handoff hash mismatch for {key}")
    if freeze.get("sampling_input_sha256") != checks["adapted_sampling_sha256"]:
        raise ValueError("Freeze sampling hash mismatch")
    if freeze.get("field_permeability_mat_sha256") != checks[
        "field_permeability_mat_sha256"
    ]:
        raise ValueError("Freeze MAT hash mismatch")
    if sidecar_data.get("output_sha256") != checks["adapted_sampling_sha256"]:
        raise ValueError("Adapter sidecar does not match the adapted CSV")
    if freeze.get("production_method_config_sha256") != checks[
        "production_method_config_sha256"
    ]:
        raise ValueError("Freeze production-method configuration hash mismatch")
    if freeze.get("selected_checkpoint_inventory_sha256") != sha256_file(
        checkpoint_inventory
    ):
        raise ValueError("Freeze selected-checkpoint inventory hash mismatch")

    report = {
        "status": "passed",
        "phase": args.phase,
        "handoff_root": str(root),
        "geology_count": actual[0],
        "case_count": actual[1],
        "assignment_count": actual[2],
        "unique_replay_pc_task_count": int(handoff["unique_replay_pc_task_count"]),
        "sampling_code_commit": freeze["sampling_code_commit"],
        "physics_commit": freeze["physics_commit"],
        "development_mode": args.allow_development,
    }
    output = args.output_json or root.with_name(f"{root.name}.verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
