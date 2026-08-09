#!/usr/bin/env python3
"""Adapt a canonical independent-full-fault manifest for replay/upscaling.

The canonical sampling manifest remains the scientific source of truth. This
adapter adds geology metadata and legacy column aliases required by the
validated MATLAB replay, Pc, and Kr programs. It does not resample, reorder,
or otherwise change any selected PREDICT realization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "independent_full_fault_upscaling_adapter_v1"
WINDOWS = [f"famp{index}" for index in range(1, 7)]

CANONICAL_REQUIRED = {
    "schema_version",
    "design_version",
    "geology_id",
    "phase",
    "case_id",
    "case_type",
    "replicate_id",
    "window_id",
    "slice_id",
    "source_library",
    "source_library_row",
    "predict_realization_id",
    "predict_seed",
    "exact_replay_seed",
    "sampling_seed",
    "selection_method",
    "checkpoint_path",
    "checkpoint_relative_path",
    "checkpoint_hash",
    "code_commit",
    "method_config_hash",
    "predict_code_commit",
    "predict_method_config_hash",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx_md",
    "perm_kyy_md",
    "perm_kzz_md",
    "use_for_probabilistic_uq",
    "is_benchmark",
    "is_stress_test",
}

GEOLOGY_REQUIRED = {
    "geology_id",
    "scenario_index",
    "scenario_label",
    "scenario_name",
    "case_index",
    "case_label",
    "faulting_depth_m",
    "sand_vcl",
    "clay_vcl",
}

OUTPUT_HEADER = [
    "adapter_schema_version",
    "design_version",
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
    "sampling_case_id",
    "phase",
    "case_type",
    "replicate_id",
    "case_name",
    "case_category",
    "case_strength",
    "pattern_name",
    "orientation",
    "group_split_id",
    "slice_index",
    "draw_group_index",
    "draw_group_slices",
    "is_shared_draw_group",
    "window",
    "similarity_group",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "selected_sample_index",
    "predict_realization_id",
    "source_pool",
    "source_pool_size",
    "draw_seed",
    "sampler_random_seed",
    "source_checkpoint_file",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "source_seed_base",
    "source_num_attempts",
    "source_num_rejected",
    "exact_replay_seed",
    "fine_scale_replay_status",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--geology-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_columns(fieldnames: Iterable[str] | None, required: set[str], label: str) -> None:
    missing = required - set(fieldnames or [])
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label} must be true or false, received {value!r}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_geology_catalog(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, GEOLOGY_REQUIRED, "Geology catalog")
        for row in reader:
            geology_id = row["geology_id"]
            if geology_id in result:
                raise ValueError(f"Duplicate geology ID in catalog: {geology_id}")
            result[geology_id] = row
    if not result:
        raise ValueError("Geology catalog is empty")
    return result


def production_case_number(case_type: str, replicate_id: int) -> int:
    """Return a stable numeric alias for MATLAB programs requiring numbers."""
    if case_type == "independent_full":
        if not 1 <= replicate_id <= 52:
            raise ValueError(f"Independent replicate is outside 1:52: {replicate_id}")
        return replicate_id
    if replicate_id != 0:
        raise ValueError(f"Deterministic case {case_type} must use replicate_id=0")
    mapping = {
        "representative_medoid": 101,
        "low_state_stress": 102,
        "high_state_stress": 103,
    }
    try:
        return mapping[case_type]
    except KeyError as error:
        raise ValueError(f"Unsupported case type: {case_type}") from error


def case_metadata(case_type: str, replicate_id: int) -> dict[str, str]:
    if case_type == "independent_full":
        return {
            "case_name": f"independent_full_{replicate_id:03d}",
            "case_category": "Probabilistic UQ",
            "case_strength": "stochastic",
            "pattern_name": "independent_full_distribution",
            "assigned_state": "independent",
            "sampling_mode": "independent",
        }
    mapping = {
        "representative_medoid": {
            "case_name": "representative_medoid",
            "case_category": "Deterministic benchmark",
            "case_strength": "deterministic",
            "pattern_name": "window_full_distribution_medoid",
            "assigned_state": "representative",
            "sampling_mode": "deterministic_benchmark",
        },
        "low_state_stress": {
            "case_name": "low_state_stress",
            "case_category": "Stress benchmark",
            "case_strength": "deterministic",
            "pattern_name": "window_low_state_medoid",
            "assigned_state": "low",
            "sampling_mode": "deterministic_stress",
        },
        "high_state_stress": {
            "case_name": "high_state_stress",
            "case_category": "Stress benchmark",
            "case_strength": "deterministic",
            "pattern_name": "window_high_state_medoid",
            "assigned_state": "high",
            "sampling_mode": "deterministic_stress",
        },
    }
    try:
        return mapping[case_type]
    except KeyError as error:
        raise ValueError(f"Unsupported case type: {case_type}") from error


def expected_roles(case_type: str) -> tuple[bool, bool, bool]:
    return {
        "independent_full": (True, False, False),
        "representative_medoid": (False, True, False),
        "low_state_stress": (False, False, True),
        "high_state_stress": (False, False, True),
    }[case_type]


def derive_seed_base(row: dict[str, str]) -> int:
    replay_seed = int(row["exact_replay_seed"])
    predict_seed = int(row["predict_seed"])
    realization_id = int(row["predict_realization_id"])
    if replay_seed != predict_seed:
        raise ValueError("exact_replay_seed must equal checkpoint-recorded predict_seed")
    seed_base = replay_seed - realization_id + 1
    if seed_base <= 0:
        raise ValueError("Derived PREDICT seed base is not positive")
    return seed_base


def adapt_row(row: dict[str, str], geology: dict[str, str]) -> dict[str, object]:
    case_type = row["case_type"]
    replicate_id = int(row["replicate_id"])
    metadata = case_metadata(case_type, replicate_id)
    expected = expected_roles(case_type)
    actual = (
        parse_bool(row["use_for_probabilistic_uq"], "use_for_probabilistic_uq"),
        parse_bool(row["is_benchmark"], "is_benchmark"),
        parse_bool(row["is_stress_test"], "is_stress_test"),
    )
    if actual != expected:
        raise ValueError(f"Case-role mismatch for {row['case_id']}: {actual} != {expected}")

    source_library = row["source_library"]
    source_pool_size = 2000 if source_library == "full_distribution" else 400
    selection_method = row["selection_method"]
    if case_type == "independent_full" and selection_method != "uniform_with_replacement":
        raise ValueError(f"Independent row has invalid selection method: {selection_method}")
    if case_type != "independent_full" and selection_method != "exact_logk_medoid":
        raise ValueError(f"Deterministic row has invalid selection method: {selection_method}")

    return {
        "adapter_schema_version": SCHEMA_VERSION,
        "design_version": row["design_version"],
        "geology_id": row["geology_id"],
        "scenario_index": geology["scenario_index"],
        "scenario_label": geology["scenario_label"],
        "scenario_name": geology["scenario_name"],
        "case_index": geology["case_index"],
        "case_label": geology["case_label"],
        "faulting_depth_m": geology["faulting_depth_m"],
        "sand_vcl": geology["sand_vcl"],
        "clay_vcl": geology["clay_vcl"],
        "case_id": production_case_number(case_type, replicate_id),
        "sampling_case_id": row["case_id"],
        "phase": row["phase"],
        "case_type": case_type,
        "replicate_id": replicate_id,
        **metadata,
        "orientation": "",
        "group_split_id": 0,
        "slice_index": int(row["slice_id"]),
        "draw_group_index": int(row["slice_id"]),
        "draw_group_slices": str(row["slice_id"]),
        "is_shared_draw_group": "false",
        "window": row["window_id"],
        "similarity_group": 0,
        "sampling_pool": source_library,
        "selected_sample_index": int(row["source_library_row"]),
        "predict_realization_id": int(row["predict_realization_id"]),
        "source_pool": source_library,
        "source_pool_size": source_pool_size,
        "draw_seed": int(row["sampling_seed"]),
        "sampler_random_seed": int(row["sampling_seed"]),
        "source_checkpoint_file": row["checkpoint_path"],
        "checkpoint_relative_path": row["checkpoint_relative_path"],
        "checkpoint_sha256": row["checkpoint_hash"].lower(),
        "source_seed_base": derive_seed_base(row),
        "source_num_attempts": "",
        "source_num_rejected": "",
        "exact_replay_seed": int(row["exact_replay_seed"]),
        "fine_scale_replay_status": "exact_seed_recorded_replay_pending",
        "log_kxx": row["log_kxx"],
        "log_kyy": row["log_kyy"],
        "log_kzz": row["log_kzz"],
        "perm_kxx": row["perm_kxx_md"],
        "perm_kyy": row["perm_kyy_md"],
        "perm_kzz": row["perm_kzz_md"],
        "sampling_code_commit": row["code_commit"],
        "sampling_method_config_hash": row["method_config_hash"],
        "predict_code_commit": row["predict_code_commit"],
        "predict_method_config_hash": row["predict_method_config_hash"],
        "use_for_probabilistic_uq": str(actual[0]).lower(),
        "is_benchmark": str(actual[1]).lower(),
        "is_stress_test": str(actual[2]).lower(),
    }


def validate_coverage(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["geology_id"]), str(row["sampling_case_id"]))].append(row)
    for key, case_rows in grouped.items():
        if len(case_rows) != 6 * 87:
            raise ValueError(f"{key} has {len(case_rows)} rows; expected 522")
        coordinates = {
            (str(row["window"]), int(row["slice_index"])) for row in case_rows
        }
        expected = {(window, slice_id) for window in WINDOWS for slice_id in range(1, 88)}
        if coordinates != expected:
            raise ValueError(f"{key} does not have exact 6 x 87 coverage")

    per_geology_phase: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for (geology_id, _), case_rows in grouped.items():
        first = case_rows[0]
        per_geology_phase[(geology_id, str(first["phase"]))][str(first["case_type"])] += 1
    for key, counts in per_geology_phase.items():
        if key[1] == "phase1" and counts != Counter({
            "independent_full": 12,
            "representative_medoid": 1,
            "low_state_stress": 1,
            "high_state_stress": 1,
        }):
            raise ValueError(f"Unexpected Phase 1 case design for {key[0]}: {dict(counts)}")
        if key[1] == "phase2" and counts != Counter({"independent_full": 40}):
            raise ValueError(f"Unexpected Phase 2 case design for {key[0]}: {dict(counts)}")


def adapt_manifest(manifest_path: Path, geology_catalog_path: Path) -> list[dict[str, object]]:
    geology_catalog = load_geology_catalog(geology_catalog_path)
    output: list[dict[str, object]] = []
    seed_bases: dict[tuple[str, str], int] = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require_columns(reader.fieldnames, CANONICAL_REQUIRED, "Canonical manifest")
        for row in reader:
            geology_id = row["geology_id"]
            if geology_id not in geology_catalog:
                raise ValueError(f"Manifest geology is absent from catalog: {geology_id}")
            adapted = adapt_row(row, geology_catalog[geology_id])
            key = (geology_id, str(adapted["window"]))
            seed_base = int(adapted["source_seed_base"])
            if key in seed_bases and seed_bases[key] != seed_base:
                raise ValueError(f"Inconsistent PREDICT seed base for {key}")
            seed_bases[key] = seed_base
            output.append(adapted)
    if not output:
        raise ValueError("Canonical manifest is empty")
    validate_coverage(output)
    output.sort(key=lambda row: (
        str(row["geology_id"]),
        str(row["phase"]),
        int(row["case_id"]),
        int(str(row["window"])[4:]),
        int(row["slice_index"]),
    ))
    return output


def write_adapted_output(
    rows: list[dict[str, object]],
    output_path: Path,
    manifest_path: Path,
    geology_catalog_path: Path,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_HEADER)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "canonical_manifest": str(manifest_path.resolve()),
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "geology_catalog": str(geology_catalog_path.resolve()),
        "geology_catalog_sha256": sha256_file(geology_catalog_path),
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "assignment_count": len(rows),
        "geology_count": len({str(row["geology_id"]) for row in rows}),
        "case_count": len({(str(row["geology_id"]), str(row["sampling_case_id"])) for row in rows}),
        "numeric_case_aliases": {
            "independent_full": "replicate_id (1-52)",
            "representative_medoid": 101,
            "low_state_stress": 102,
            "high_state_stress": 103,
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    geology_catalog = args.geology_catalog.resolve()
    output = args.output.resolve()
    for path in (manifest, geology_catalog):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = adapt_manifest(manifest, geology_catalog)
    write_adapted_output(rows, output, manifest, geology_catalog, args.overwrite)
    print(json.dumps({
        "output": str(output),
        "assignments": len(rows),
        "geologies": len({row["geology_id"] for row in rows}),
        "cases": len({(row["geology_id"], row["sampling_case_id"]) for row in rows}),
        "sha256": sha256_file(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
