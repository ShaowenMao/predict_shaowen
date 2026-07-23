#!/usr/bin/env python3
"""Verify that a frozen production bundle uses the intended PREDICT variant.

This is a semantic gate, not only a checksum gate. It prevents a validly
hashed but scientifically wrong input bundle (for example, legacy smear
placement or uncollapsed stratigraphy) from entering the production workflow.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_GEOLOGIES = 162
EXPECTED_CASES = 1620
EXPECTED_SLICES = 87
EXPECTED_WINDOWS = {f"famp{i}" for i in range(1, 7)}
EXPECTED_ASSIGNMENTS = EXPECTED_CASES * EXPECTED_SLICES * len(EXPECTED_WINDOWS)
EXPECTED_CHECKPOINTS = EXPECTED_GEOLOGIES * len(EXPECTED_WINDOWS)
EXPECTED_PREDICT_ROOT_NAME = "thickness_scenario_data_collapsed_cell_union"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", required=True, type=Path)
    parser.add_argument("--predict-root", type=Path)
    parser.add_argument("--sampling-root", type=Path)
    parser.add_argument("--expected-code-commit", default="")
    parser.add_argument("--expected-method-hash", default="")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def canonical_source_root(path_text: str) -> str:
    """Return the final directory name for Windows or POSIX provenance paths."""

    text = path_text.strip().replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def checkpoint_root_name(path_text: str) -> str:
    """Return the PREDICT dataset directory encoded in a checkpoint path."""

    normalized = path_text.strip().replace("\\", "/")
    marker = "/data/"
    if marker not in normalized:
        return ""
    return normalized.split(marker, 1)[0].rstrip("/").rsplit("/", 1)[-1]


def load_simple_toml(config_path: Path) -> dict:
    """Parse the scalar/list subset used by production_method_config.toml."""

    config: dict[str, dict[str, object]] = {"": {}}
    section = ""
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if not section:
                raise ValueError(f"Empty TOML section at line {line_number}")
            config.setdefault(section, {})
            continue
        if "=" not in line:
            raise ValueError(f"Unsupported TOML syntax at line {line_number}: {line}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        lowered = raw_value.lower()
        if lowered == "true":
            value: object = True
        elif lowered == "false":
            value = False
        else:
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Unsupported TOML value at line {line_number}: {raw_value}"
                ) from error
        config.setdefault(section, {})[key] = value
    return config


def verify_method_config(config_path: Path) -> dict:
    config = load_simple_toml(config_path)

    required = {
        ("predict", "input_variant"): "collapsed_adjacent_lithology",
        ("predict", "smear_overlap_rule"): "cell_union_psmear",
        ("replay", "require_exact_seed"): True,
        ("replay", "require_permeability_match"): True,
        ("replay", "preserve_full_3d_grid"): True,
        ("replay", "allow_strike_collapse"): False,
        ("pc", "method"): "invasion_percolation",
        ("pc", "endpoint_mode"): "native",
        ("pc", "preserve_effective_swi"): True,
        ("pc", "preserve_all_87_slices"): True,
        ("pc", "allow_strike_collapse"): False,
        ("kr", "method"): "dynamic",
        ("kr", "selection_mode"): "swi_medoid",
        ("kr", "representatives_per_case"): 6,
        ("kr", "linear_solver"): "amgcl_require",
        ("kr", "preserve_full_3d_grid"): True,
        ("kr", "allow_strike_collapse"): False,
    }
    for (section, key), expected in required.items():
        actual = config.get(section, {}).get(key)
        if actual != expected:
            raise ValueError(
                f"Method configuration mismatch: [{section}] {key}="
                f"{actual!r}, expected {expected!r}"
            )
    representations = config.get("reservoir_export", {}).get(
        "pc_representations", []
    )
    if set(representations) != {"full_slice", "pe_branch_medoid"}:
        raise ValueError(
            "Reservoir export must retain both full_slice and "
            "pe_branch_medoid Pc representations"
        )
    return config


def verify_predict_metadata(predict_root: Path) -> dict:
    metadata_path = require_file(
        predict_root / "tables" / "collapsed_cell_union_geology_run_metadata.csv"
    )
    row_count = 0
    checkpoints: set[str] = set()
    geologies: set[tuple[str, str]] = set()
    windows: set[str] = set()
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "ScenarioLabel",
            "Window",
            "CaseLabel",
            "SmearOverlapRule",
            "CollapseAdjacentLithology",
            "TargetN",
            "CheckpointFile",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"PREDICT metadata lacks columns: {sorted(missing)}")
        for row_count, row in enumerate(reader, start=1):
            if row["SmearOverlapRule"].strip().lower() != "cell_union_psmear":
                raise ValueError(
                    f"Row {row_count} does not use cell_union_psmear"
                )
            if row["CollapseAdjacentLithology"].strip().lower() not in {
                "1",
                "true",
            }:
                raise ValueError(
                    f"Row {row_count} does not use collapsed stratigraphy"
                )
            if int(float(row["TargetN"])) != 2000:
                raise ValueError(f"Row {row_count} does not contain 2000 samples")
            root_name = checkpoint_root_name(row["CheckpointFile"])
            if root_name != EXPECTED_PREDICT_ROOT_NAME:
                raise ValueError(
                    f"Row {row_count} checkpoint identifies {root_name!r}, "
                    f"expected {EXPECTED_PREDICT_ROOT_NAME!r}"
                )
            windows.add(row["Window"].strip().lower())
            geologies.add((row["ScenarioLabel"], row["CaseLabel"]))
            checkpoints.add(
                "|".join(
                    (
                        row["ScenarioLabel"],
                        row["Window"].strip().lower(),
                        row["CaseLabel"],
                    )
                )
            )

    if row_count != EXPECTED_CHECKPOINTS:
        raise ValueError(
            f"Expected {EXPECTED_CHECKPOINTS} PREDICT metadata rows, "
            f"found {row_count}"
        )
    if len(geologies) != EXPECTED_GEOLOGIES:
        raise ValueError(
            f"Expected {EXPECTED_GEOLOGIES} geologies, found {len(geologies)}"
        )
    if len(checkpoints) != EXPECTED_CHECKPOINTS:
        raise ValueError(
            f"Expected {EXPECTED_CHECKPOINTS} unique checkpoints, "
            f"found {len(checkpoints)}"
        )
    if windows != EXPECTED_WINDOWS:
        raise ValueError(f"Unexpected PREDICT windows: {sorted(windows)}")
    return {
        "metadata_file": str(metadata_path),
        "metadata_rows": row_count,
        "geology_count": len(geologies),
        "checkpoint_count": len(checkpoints),
        "windows": sorted(windows),
    }


def verify_sampling(sampling_root: Path) -> dict:
    sampling_path = require_file(
        sampling_root / "texas_field_slice_window_values.csv"
    )
    row_count = 0
    geologies: set[str] = set()
    cases: set[tuple[str, int]] = set()
    slices: set[int] = set()
    windows: set[str] = set()
    source_roots: set[str] = set()
    with sampling_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "geology_id",
            "case_id",
            "slice_index",
            "window",
            "selected_sample_index",
            "source_seed_base",
            "exact_replay_seed",
            "source_checkpoint_file",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Sampling table lacks columns: {sorted(missing)}")
        for row_count, row in enumerate(reader, start=1):
            geology_id = row["geology_id"]
            case_id = int(row["case_id"])
            slice_index = int(row["slice_index"])
            sample_index = int(row["selected_sample_index"])
            seed_base = int(row["source_seed_base"])
            replay_seed = int(row["exact_replay_seed"])
            if replay_seed != seed_base + sample_index - 1:
                raise ValueError(
                    f"Replay seed mismatch at sampling row {row_count}"
                )
            source_root = checkpoint_root_name(row["source_checkpoint_file"])
            if source_root != EXPECTED_PREDICT_ROOT_NAME:
                raise ValueError(
                    f"Sampling row {row_count} points to {source_root!r}, "
                    f"expected {EXPECTED_PREDICT_ROOT_NAME!r}"
                )
            geologies.add(geology_id)
            cases.add((geology_id, case_id))
            slices.add(slice_index)
            windows.add(row["window"].strip().lower())
            source_roots.add(source_root)

    if row_count != EXPECTED_ASSIGNMENTS:
        raise ValueError(
            f"Expected {EXPECTED_ASSIGNMENTS} assignments, found {row_count}"
        )
    if len(geologies) != EXPECTED_GEOLOGIES or len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_GEOLOGIES} geologies/{EXPECTED_CASES} cases, "
            f"found {len(geologies)}/{len(cases)}"
        )
    if slices != set(range(1, EXPECTED_SLICES + 1)):
        raise ValueError("Sampling slice coverage is not exactly 1:87")
    if windows != EXPECTED_WINDOWS:
        raise ValueError(f"Unexpected sampling windows: {sorted(windows)}")
    return {
        "sampling_file": str(sampling_path),
        "assignment_count": row_count,
        "geology_count": len(geologies),
        "case_count": len(cases),
        "slice_count": len(slices),
        "windows": sorted(windows),
        "source_roots": sorted(source_roots),
    }


def main() -> int:
    args = parse_args()
    freeze_root = args.freeze_root.resolve()
    metadata_path = require_file(freeze_root / "freeze_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    predict_root = (
        args.predict_root.resolve()
        if args.predict_root
        else (freeze_root / "inputs" / "predict").resolve()
    )
    sampling_root = (
        args.sampling_root.resolve()
        if args.sampling_root
        else (freeze_root / "inputs" / "sampling").resolve()
    )
    method_path = require_file(
        freeze_root
        / "config"
        / metadata.get("method_config_file", "production_method_config.toml")
    )

    if args.expected_code_commit:
        actual = metadata.get("code_commit", "")
        if actual != args.expected_code_commit:
            raise ValueError(
                f"Frozen code commit {actual!r} does not match "
                f"{args.expected_code_commit!r}"
            )
    method_hash = sha256_file(method_path)
    expected_method_hash = (
        args.expected_method_hash or metadata.get("method_config_sha256", "")
    )
    if method_hash != expected_method_hash:
        raise ValueError(
            f"Method hash {method_hash} does not match {expected_method_hash}"
        )

    source_root_name = canonical_source_root(
        metadata.get("predict_source_root", "")
    )
    if source_root_name != EXPECTED_PREDICT_ROOT_NAME:
        raise ValueError(
            f"Freeze metadata identifies PREDICT root {source_root_name!r}, "
            f"expected {EXPECTED_PREDICT_ROOT_NAME!r}"
        )

    verify_method_config(method_path)
    predict_report = verify_predict_metadata(predict_root)
    sampling_report = verify_sampling(sampling_root)

    report = {
        "status": "passed",
        "freeze_root": str(freeze_root),
        "code_commit": metadata.get("code_commit", ""),
        "method_config_sha256": method_hash,
        "predict_source_root_name": source_root_name,
        "predict": predict_report,
        "sampling": sampling_report,
    }
    output_path = args.output_json or freeze_root / "semantic_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
