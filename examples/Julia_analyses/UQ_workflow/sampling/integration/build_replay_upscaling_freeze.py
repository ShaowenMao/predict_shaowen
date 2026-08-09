#!/usr/bin/env python3
"""Build restartable replay/Pc tasks for independent-full-fault manifests.

This is the versioned successor to the legacy ten-case manifest builder. It
accepts the adapter output, validates exact checkpoint hashes and recorded
accepted seeds, and emits the established task and assignment contracts used
by the Engaging replay/Pc/Kr pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "independent_full_fault_replay_freeze_v1"
REQUIRED_COLUMNS = {
    "adapter_schema_version",
    "design_version",
    "geology_id",
    "case_id",
    "sampling_case_id",
    "phase",
    "case_type",
    "replicate_id",
    "case_name",
    "case_category",
    "slice_index",
    "draw_group_index",
    "window",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "selected_sample_index",
    "predict_realization_id",
    "source_checkpoint_file",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "source_seed_base",
    "exact_replay_seed",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx",
    "perm_kyy",
    "perm_kzz",
    "sampling_code_commit",
    "sampling_method_config_hash",
    "predict_code_commit",
    "predict_method_config_hash",
    "use_for_probabilistic_uq",
    "is_benchmark",
    "is_stress_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-csv", required=True, type=Path)
    parser.add_argument(
        "--field-permeability-mat",
        type=Path,
        help="Optional validated compact fault-local permeability MAT to freeze.",
    )
    parser.add_argument("--predict-root", required=True, type=Path)
    parser.add_argument("--method-config", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--remote-freeze-root", default="")
    parser.add_argument("--tasks-per-shard", type=int, default=5000)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development-only override for an uncommitted repository.",
    )
    parser.add_argument(
        "--allow-sampling-code-mismatch",
        action="store_true",
        help=(
            "Development-only override when manifest sampling_code_commit "
            "does not equal the checked-out repository commit."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def load_legacy_builder(repo_root: Path):
    path = repo_root / "examples" / "pc_upscaling_pilot" / "engaging" / "production" / "build_restartable_manifest.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("legacy_restartable_manifest", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import manifest utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(predict_root: Path, relative: str) -> Path:
    root = predict_root.resolve()
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Checkpoint escapes PREDICT root: {relative}") from error
    return path


def write_checkpoint_inventory(rows: dict[str, tuple[Path, str]], path: Path) -> str:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for relative in sorted(rows):
            checkpoint, digest = rows[relative]
            size = checkpoint.stat().st_size
            writer.writerow([relative, size, digest])
    return sha256_file(path)


def validate_case_design(case_roles: dict[tuple[str, str], tuple[str, str, int]]) -> None:
    by_geology_phase: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for (geology_id, _), (phase, case_type, replicate_id) in case_roles.items():
        by_geology_phase.setdefault((geology_id, phase), []).append((case_type, replicate_id))
    for (geology_id, phase), cases in by_geology_phase.items():
        if phase == "phase1":
            independent = sorted(rep for kind, rep in cases if kind == "independent_full")
            deterministic = sorted(kind for kind, _ in cases if kind != "independent_full")
            if independent != list(range(1, 13)) or deterministic != [
                "high_state_stress", "low_state_stress", "representative_medoid"
            ]:
                raise ValueError(f"Invalid Phase 1 case design for {geology_id}")
        elif phase == "phase2":
            independent = sorted(rep for kind, rep in cases if kind == "independent_full")
            if len(cases) != 40 or independent != list(range(13, 53)):
                raise ValueError(f"Invalid Phase 2 case design for {geology_id}")
        else:
            raise ValueError(f"Unsupported phase: {phase}")


def main() -> int:
    args = parse_args()
    if args.tasks_per_shard <= 0:
        raise ValueError("--tasks-per-shard must be positive")
    sampling_csv = args.sampling_csv.resolve()
    predict_root = args.predict_root.resolve()
    method_config = args.method_config.resolve()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    for path in (sampling_csv, predict_root, method_config, repo_root):
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
    sampling_root = output_root / "sampling"
    for path in (inventory_root, manifest_root, config_root, sampling_root):
        path.mkdir()
    shutil.copy2(method_config, config_root / method_config.name)
    frozen_sampling = sampling_root / "texas_field_slice_window_values.csv"
    shutil.copy2(sampling_csv, frozen_sampling)
    sampling_metadata = sampling_csv.with_suffix(sampling_csv.suffix + ".metadata.json")
    if sampling_metadata.is_file():
        shutil.copy2(sampling_metadata, sampling_root / sampling_metadata.name)
    field_permeability_hash = ""
    if args.field_permeability_mat is not None:
        field_permeability_mat = args.field_permeability_mat.resolve()
        if not field_permeability_mat.is_file():
            raise FileNotFoundError(field_permeability_mat)
        frozen_field_permeability = sampling_root / field_permeability_mat.name
        shutil.copy2(field_permeability_mat, frozen_field_permeability)
        field_permeability_hash = sha256_file(frozen_field_permeability)

    legacy = load_legacy_builder(repo_root)
    sampling_commit = git_value(repo_root, "rev-parse", "HEAD")
    sampling_branch = git_value(repo_root, "branch", "--show-current")
    sampling_dirty = bool(git_value(repo_root, "status", "--porcelain"))
    if sampling_dirty and not args.allow_dirty:
        raise ValueError(
            "Production freeze requires a clean repository; use --allow-dirty only for development acceptance tests"
        )
    production_method_hash = sha256_file(method_config)
    sampling_hash = sha256_file(frozen_sampling)

    database_path = output_root / "manifest_build.sqlite"
    connection = legacy.initialize_database(database_path)
    assignment_path = manifest_root / "assignment_to_task.csv"
    geology_ids: set[str] = set()
    case_keys: set[tuple[str, str]] = set()
    case_roles: dict[tuple[str, str], tuple[str, str, int]] = {}
    case_coordinates: dict[tuple[str, str], set[tuple[str, int]]] = {}
    phases: set[str] = set()
    checkpoint_records: dict[str, tuple[Path, str]] = {}
    checkpoint_hash_cache: dict[str, str] = {}
    physics_commits: set[str] = set()
    predict_method_hashes: set[str] = set()
    manifest_sampling_commits: set[str] = set()
    manifest_sampling_method_hashes: set[str] = set()
    design_versions: set[str] = set()
    assignment_count = 0

    with frozen_sampling.open(newline="", encoding="utf-8-sig") as source_stream:
        reader = csv.DictReader(source_stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Adapted sampling CSV lacks columns: {sorted(missing)}")
        with assignment_path.open("w", newline="", encoding="utf-8") as target_stream:
            writer = csv.writer(target_stream)
            writer.writerow(legacy.ASSIGNMENT_HEADER)
            for assignment_count, row in enumerate(reader, start=1):
                relative = row["checkpoint_relative_path"].replace("\\", "/")
                checkpoint = checkpoint_path(predict_root, relative)
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if relative not in checkpoint_hash_cache:
                    checkpoint_hash_cache[relative] = sha256_file(checkpoint)
                actual_hash = checkpoint_hash_cache[relative]
                expected_hash = row["checkpoint_sha256"].lower()
                if actual_hash != expected_hash:
                    raise ValueError(f"Checkpoint hash mismatch: {relative}")
                checkpoint_records[relative] = (checkpoint, actual_hash)

                sample_index = int(row["selected_sample_index"])
                realization_id = int(row["predict_realization_id"])
                replay_seed = int(row["exact_replay_seed"])
                seed_base = int(row["source_seed_base"])
                if replay_seed != seed_base + realization_id - 1:
                    raise ValueError(
                        f"Accepted-seed identity mismatch at assignment {assignment_count}"
                    )
                if not 1 <= sample_index <= 2000:
                    raise ValueError(f"Source row outside 1:2000 at assignment {assignment_count}")

                physics_commit = row["predict_code_commit"]
                physics_commits.add(physics_commit)
                predict_method_hashes.add(row["predict_method_config_hash"])
                manifest_sampling_commits.add(row["sampling_code_commit"])
                manifest_sampling_method_hashes.add(
                    row["sampling_method_config_hash"]
                )
                design_versions.add(row["design_version"])
                task_key, task_id = legacy.task_identity(
                    row, relative, actual_hash, physics_commit, production_method_hash
                )
                legacy.insert_task(
                    connection,
                    row,
                    task_key,
                    task_id,
                    relative,
                    actual_hash,
                    physics_commit,
                    production_method_hash,
                )
                writer.writerow([
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
                ])

                geology_id = row["geology_id"]
                sampling_case_id = row["sampling_case_id"]
                case_key = (geology_id, sampling_case_id)
                role = (row["phase"], row["case_type"], int(row["replicate_id"]))
                phases.add(row["phase"])
                if case_key in case_roles and case_roles[case_key] != role:
                    raise ValueError(f"Inconsistent case role for {case_key}")
                case_roles[case_key] = role
                coordinates = case_coordinates.setdefault(case_key, set())
                coordinate = (row["window"], int(row["slice_index"]))
                if coordinate in coordinates:
                    raise ValueError(f"Duplicate window-slice coordinate for {case_key}: {coordinate}")
                coordinates.add(coordinate)
                geology_ids.add(geology_id)
                case_keys.add(case_key)
                if assignment_count % 10000 == 0:
                    connection.commit()
    connection.commit()

    expected_coordinates = {(f"famp{window}", slice_id) for window in range(1, 7) for slice_id in range(1, 88)}
    for case_key, coordinates in case_coordinates.items():
        if coordinates != expected_coordinates:
            raise ValueError(f"Case does not have exact 6 x 87 coverage: {case_key}")
    validate_case_design(case_roles)
    if len(phases) != 1:
        raise ValueError(
            f"A phase freeze must contain exactly one phase, found {sorted(phases)}"
        )
    if len(physics_commits) != 1 or len(predict_method_hashes) != 1:
        raise ValueError("A production freeze must use one PREDICT code and method version")
    if len(manifest_sampling_commits) != 1 or len(manifest_sampling_method_hashes) != 1:
        raise ValueError("A production freeze must use one sampling code and method version")
    if design_versions != {"independent_full_fault_v1"}:
        raise ValueError(f"Unexpected design versions: {sorted(design_versions)}")
    manifest_sampling_commit = next(iter(manifest_sampling_commits))
    if (
        manifest_sampling_commit != sampling_commit
        and not args.allow_sampling_code_mismatch
    ):
        raise ValueError(
            "Manifest sampling_code_commit does not match the checked-out repository: "
            f"{manifest_sampling_commit} != {sampling_commit}"
        )

    mismatch_count = connection.execute(
        "SELECT COALESCE(SUM(mismatch_count), 0) FROM tasks"
    ).fetchone()[0]
    if mismatch_count:
        raise ValueError(f"Conflicting expected values for {mismatch_count} duplicate tasks")
    task_count, shard_count = legacy.export_tasks(connection, manifest_root, args.tasks_per_shard)
    connection.close()
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(database_path) + suffix)
        if path.exists():
            path.unlink()

    checkpoint_inventory_hash = write_checkpoint_inventory(
        checkpoint_records, inventory_root / "selected_predict_checkpoints_sha256.csv"
    )
    restart_policy = {
        "schema_version": SCHEMA_VERSION,
        "completion_rule": (
            "A stage is complete only when output and done marker identities match "
            "task_key_sha256, physics commit, production method hash, checkpoint hash, "
            "source row, and exact checkpoint-recorded replay seed."
        ),
        "replay_resume": "Reuse only validated replay output; otherwise replay the task.",
        "pc_resume": "Reuse Pc only after validated exact replay; otherwise rerun Pc.",
        "failed_task_policy": "Rerun only failed or unvalidated task IDs.",
    }
    (manifest_root / "restart_policy.json").write_text(
        json.dumps(restart_policy, indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_version": "independent_full_fault_v1",
        "phases": sorted(phases),
        "sampling_code_commit": sampling_commit,
        "manifest_sampling_code_commit": manifest_sampling_commit,
        "sampling_method_config_hash": next(iter(manifest_sampling_method_hashes)),
        "sampling_code_branch": sampling_branch,
        "sampling_code_dirty": sampling_dirty,
        "development_overrides": {
            "allow_dirty": args.allow_dirty,
            "allow_sampling_code_mismatch": args.allow_sampling_code_mismatch,
        },
        "physics_commit": next(iter(physics_commits)),
        "predict_method_config_hash": next(iter(predict_method_hashes)),
        "production_method_config_file": method_config.name,
        "production_method_config_sha256": production_method_hash,
        "sampling_input_sha256": sampling_hash,
        "field_permeability_mat_sha256": field_permeability_hash,
        "selected_checkpoint_inventory_sha256": checkpoint_inventory_hash,
        "remote_freeze_root": args.remote_freeze_root,
        "predict_source_root": str(predict_root),
        "geology_count": len(geology_ids),
        "case_count": len(case_keys),
        "assignment_count": assignment_count,
        "unique_replay_pc_task_count": task_count,
        "duplicate_assignment_count": assignment_count - task_count,
        "deduplication_fraction": (assignment_count - task_count) / assignment_count,
        "tasks_per_shard": args.tasks_per_shard,
        "shard_count": shard_count,
        "numeric_case_aliases": {
            "independent_full": "replicate_id (1-52)",
            "representative_medoid": 101,
            "low_state_stress": 102,
            "high_state_stress": 103,
        },
        "task_key_fields": [
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "selected_sample_index",
            "exact_replay_seed",
            "physics_commit",
            "production_method_config_sha256",
        ],
    }
    (output_root / "freeze_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    legacy.write_generated_inventory(output_root)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
