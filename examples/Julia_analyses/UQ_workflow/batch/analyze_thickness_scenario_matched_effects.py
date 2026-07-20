"""Compute matched geologic effects from thickness-scenario PREDICT libraries.

The analysis first reduces each 2,000-realization window library to the median
of each log10 permeability component. It then differences condition pairs while
holding every non-tested geology parameter, throw window, and component fixed.

The script writes auditable distribution-, pair-, and summary-level CSV files
and recreates the dominant-effect figure used for the original PREDICT data.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("kxx", "kyy", "kzz")
COMPONENT_COLORS = {
    "kxx": "#245f9e",
    "kyy": "#ef8a0c",
    "kzz": "#24864a",
}

EFFECT_ORDER = (
    "high sand - low sand",
    "nonuniform - uniform",
    "fault depth 1000 - 50",
    "sand Vcl 0.3 - 0.1",
    "clay Vcl 0.6 - 0.4",
)


def parse_args() -> argparse.Namespace:
    """Parse input/output paths and optional figure label."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing data/ and tables/collapsed_cell_union_geology_run_metadata.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-label",
        default="collapsed layers + cell_union_psmear",
        help="Short subtitle used in the figure.",
    )
    return parser.parse_args()


def scenario_attributes(name: str) -> tuple[str, str]:
    """Return sand-level and thickness-pattern labels from ScenarioName."""
    normalized = name.strip().lower()
    sand_level = next(
        (level for level in ("low", "medium", "high") if normalized.startswith(level)),
        None,
    )
    uniformity = "nonuniform" if "nonuniform" in normalized else "uniform"
    if sand_level is None:
        raise ValueError(f"Cannot infer sand level from ScenarioName={name!r}")
    return sand_level, uniformity


def read_perms(path: Path) -> np.ndarray:
    """Read the 2-D PREDICT permeability matrix from a MATLAB v7.3 file."""
    with h5py.File(path, "r") as handle:
        if "perms" not in handle:
            raise KeyError(f"Missing 'perms' dataset in {path}")
        perms = np.asarray(handle["perms"], dtype=float)

    if perms.ndim != 2:
        raise ValueError(f"Expected a 2-D permeability matrix in {path}; got {perms.shape}")
    if perms.shape[0] == 3:
        perms = perms.T
    elif perms.shape[1] != 3:
        raise ValueError(f"Expected three permeability components in {path}; got {perms.shape}")
    if not np.all(np.isfinite(perms)) or np.any(perms <= 0.0):
        raise ValueError(f"Permeabilities must be finite and positive in {path}")
    return perms


def build_distribution_table(data_root: Path) -> list[dict[str, object]]:
    """Build one median-log10(k) record per geology-window-component library."""
    metadata_path = data_root / "tables" / "collapsed_cell_union_geology_run_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        metadata = list(csv.DictReader(stream))
    if len(metadata) != 972:
        raise ValueError(f"Expected 972 geology-window libraries; found {len(metadata)}")

    rows: list[dict[str, object]] = []
    seen_libraries: set[tuple[str, str, str]] = set()
    for index, meta in enumerate(metadata, start=1):
        scenario = meta["ScenarioLabel"]
        window = meta["Window"]
        case_label = meta["CaseLabel"]
        library_key = (scenario, window, case_label)
        if library_key in seen_libraries:
            raise ValueError(f"Duplicate library metadata: {library_key}")
        seen_libraries.add(library_key)

        mat_path = data_root / "data" / scenario / window / case_label / "predict_runs.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        perms = read_perms(mat_path)
        target_n = int(meta["TargetN"])
        if perms.shape != (target_n, 3):
            raise ValueError(f"Expected ({target_n}, 3) in {mat_path}; got {perms.shape}")

        sand_level, uniformity = scenario_attributes(meta["ScenarioName"])
        medians = np.median(np.log10(perms), axis=0)
        for component_index, component in enumerate(COMPONENTS):
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_name": meta["ScenarioName"],
                    "sand_level": sand_level,
                    "uniformity": uniformity,
                    "case_index": int(meta["CaseIndex"]),
                    "case_label": case_label,
                    "fault_depth": float(meta["FaultingDepth"]),
                    "sand_vcl": float(meta["SandVcl"]),
                    "clay_vcl": float(meta["ClayVcl"]),
                    "window": window,
                    "component": component,
                    "n_realizations": perms.shape[0],
                    "median_log10k": float(medians[component_index]),
                    "source_file": str(mat_path),
                }
            )

        if index % 100 == 0 or index == len(metadata):
            print(f"Loaded {index:3d}/{len(metadata)} PREDICT libraries", flush=True)

    if len(rows) != 2916:
        raise ValueError(f"Expected 2,916 distribution records; found {len(rows)}")
    return rows


def effect_specs() -> tuple[dict[str, object], ...]:
    """Define the five extreme-condition comparisons shown in the figure."""
    return (
        {
            "effect": "high sand - low sand",
            "left": lambda row: row["sand_level"] == "high",
            "right": lambda row: row["sand_level"] == "low",
            "match_fields": (
                "uniformity", "fault_depth", "sand_vcl", "clay_vcl", "window", "component"
            ),
            "group": lambda row: row["uniformity"],
        },
        {
            "effect": "nonuniform - uniform",
            "left": lambda row: row["uniformity"] == "nonuniform",
            "right": lambda row: row["uniformity"] == "uniform",
            "match_fields": (
                "sand_level", "fault_depth", "sand_vcl", "clay_vcl", "window", "component"
            ),
            "group": lambda row: f"{row['sand_level']} sand",
        },
        {
            "effect": "fault depth 1000 - 50",
            "left": lambda row: row["fault_depth"] == 1000.0,
            "right": lambda row: row["fault_depth"] == 50.0,
            "match_fields": ("scenario", "sand_vcl", "clay_vcl", "window", "component"),
            "group": lambda row: "all scenarios",
        },
        {
            "effect": "sand Vcl 0.3 - 0.1",
            "left": lambda row: row["sand_vcl"] == 0.3,
            "right": lambda row: row["sand_vcl"] == 0.1,
            "match_fields": ("scenario", "fault_depth", "clay_vcl", "window", "component"),
            "group": lambda row: "all scenarios",
        },
        {
            "effect": "clay Vcl 0.6 - 0.4",
            "left": lambda row: row["clay_vcl"] == 0.6,
            "right": lambda row: row["clay_vcl"] == 0.4,
            "match_fields": ("scenario", "fault_depth", "sand_vcl", "window", "component"),
            "group": lambda row: "all scenarios",
        },
    )


def matched_effects(distributions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Difference condition pairs with every non-tested field held fixed."""
    pairs: list[dict[str, object]] = []
    for spec in effect_specs():
        left_filter = spec["left"]
        right_filter = spec["right"]
        fields = spec["match_fields"]
        group_label = spec["group"]
        assert isinstance(left_filter, Callable)
        assert isinstance(right_filter, Callable)
        assert isinstance(group_label, Callable)

        right_index: dict[tuple[object, ...], dict[str, object]] = {}
        for row in distributions:
            if right_filter(row):
                key = tuple(row[field] for field in fields)
                if key in right_index:
                    raise ValueError(f"Duplicate right-side match for {spec['effect']}: {key}")
                right_index[key] = row

        used_right_keys: set[tuple[object, ...]] = set()
        for left_row in distributions:
            if not left_filter(left_row):
                continue
            key = tuple(left_row[field] for field in fields)
            right_row = right_index.get(key)
            if right_row is None:
                raise ValueError(f"Missing matched right condition for {spec['effect']}: {key}")
            used_right_keys.add(key)
            pairs.append(
                {
                    "effect": spec["effect"],
                    "effect_group": group_label(left_row),
                    "component": left_row["component"],
                    "window": left_row["window"],
                    "left_scenario": left_row["scenario"],
                    "right_scenario": right_row["scenario"],
                    "left_case_label": left_row["case_label"],
                    "right_case_label": right_row["case_label"],
                    "fault_depth_left": left_row["fault_depth"],
                    "fault_depth_right": right_row["fault_depth"],
                    "sand_vcl_left": left_row["sand_vcl"],
                    "sand_vcl_right": right_row["sand_vcl"],
                    "clay_vcl_left": left_row["clay_vcl"],
                    "clay_vcl_right": right_row["clay_vcl"],
                    "left_median_log10k": left_row["median_log10k"],
                    "right_median_log10k": right_row["median_log10k"],
                    "delta_median_log10k": (
                        float(left_row["median_log10k"]) - float(right_row["median_log10k"])
                    ),
                    "match_key": " | ".join(str(value) for value in key),
                }
            )

        if len(used_right_keys) != len(right_index):
            raise ValueError(
                f"Unmatched right-side records for {spec['effect']}: "
                f"{len(right_index) - len(used_right_keys)}"
            )

    expected = 5346
    if len(pairs) != expected:
        raise ValueError(f"Expected {expected:,} matched component pairs; found {len(pairs):,}")
    return pairs


def summarize_groups(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize pair deltas within scientifically balanced effect subgroups."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["effect"]), str(row["effect_group"]), str(row["component"]))].append(
            float(row["delta_median_log10k"])
        )

    summaries: list[dict[str, object]] = []
    for (effect, effect_group, component), values in grouped.items():
        delta = np.asarray(values, dtype=float)
        summaries.append(
            {
                "effect": effect,
                "effect_group": effect_group,
                "component": component,
                "n_matched": delta.size,
                "median_delta": float(np.median(delta)),
                "mean_delta": float(np.mean(delta)),
                "q25_delta": float(np.quantile(delta, 0.25)),
                "q75_delta": float(np.quantile(delta, 0.75)),
                "p05_delta": float(np.quantile(delta, 0.05)),
                "p95_delta": float(np.quantile(delta, 0.95)),
                "positive_pct": float(100.0 * np.mean(delta > 0.0)),
                "negative_pct": float(100.0 * np.mean(delta < 0.0)),
            }
        )

    effect_rank = {name: rank for rank, name in enumerate(EFFECT_ORDER)}
    component_rank = {name: rank for rank, name in enumerate(COMPONENTS)}
    summaries.sort(
        key=lambda row: (
            effect_rank[str(row["effect"])], component_rank[str(row["component"])], str(row["effect_group"])
        )
    )
    return summaries


def presentation_summary(group_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Pool balanced subgroup summaries exactly as in the original PPT figure."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in group_summaries:
        grouped[(str(row["effect"]), str(row["component"]))].append(row)

    rows: list[dict[str, object]] = []
    for effect in EFFECT_ORDER:
        for component in COMPONENTS:
            subgroup_rows = grouped[(effect, component)]
            if not subgroup_rows:
                raise ValueError(f"Missing presentation summary for {effect} | {component}")
            rows.append(
                {
                    "effect": effect,
                    "component": component,
                    "n_matched": sum(int(row["n_matched"]) for row in subgroup_rows),
                    "n_subgroups": len(subgroup_rows),
                    "balanced_median_delta": float(
                        np.median([float(row["median_delta"]) for row in subgroup_rows])
                    ),
                    "balanced_q25_delta": float(
                        np.median([float(row["q25_delta"]) for row in subgroup_rows])
                    ),
                    "balanced_q75_delta": float(
                        np.median([float(row["q75_delta"]) for row in subgroup_rows])
                    ),
                    "subgroups": "; ".join(str(row["effect_group"]) for row in subgroup_rows),
                }
            )
    if len(rows) != 15:
        raise ValueError(f"Expected 15 presentation rows; found {len(rows)}")
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write dictionaries to CSV with stable columns and numeric precision."""
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: f"{value:.9f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def plot_effects(rows: list[dict[str, object]], output_dir: Path, dataset_label: str) -> None:
    """Create the publication-style matched-effect plot in PNG and PDF formats."""
    labels = [f"{row['effect']} | {row['component']}" for row in rows]
    medians = np.asarray([float(row["balanced_median_delta"]) for row in rows])
    q25 = np.asarray([float(row["balanced_q25_delta"]) for row in rows])
    q75 = np.asarray([float(row["balanced_q75_delta"]) for row in rows])
    y = np.arange(len(rows))[::-1]

    figure, axis = plt.subplots(figsize=(17.0, 9.0))
    for index, row in enumerate(rows):
        component = str(row["component"])
        color = COMPONENT_COLORS[component]
        axis.errorbar(
            medians[index],
            y[index],
            xerr=[[medians[index] - q25[index]], [q75[index] - medians[index]]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8,
            elinewidth=4,
            capsize=0,
            zorder=3,
        )
        offset = 10 if medians[index] >= 0.0 else -10
        axis.annotate(
            f"{medians[index]:.2f}",
            (medians[index], y[index]),
            xytext=(offset, 0),
            textcoords="offset points",
            va="center",
            ha="left" if offset > 0 else "right",
            fontsize=11,
            color="#222222",
            zorder=4,
        )

    axis.axvline(0.0, color="#111111", linestyle=(0, (4, 3)), linewidth=2)
    axis.set_yticks(y, labels=labels, fontsize=13)
    axis.set_xlabel("Delta median log10(k)", fontsize=17)
    axis.tick_params(axis="x", labelsize=13)
    axis.grid(axis="both", color="#d8d8d8", linewidth=1.0, alpha=0.65)
    axis.set_axisbelow(True)

    left_limit = min(-5.7, float(np.min(q25)) - 0.45)
    right_limit = max(1.6, float(np.max(q75)) + 0.45)
    axis.set_xlim(left_limit, right_limit)
    axis.set_ylim(-0.7, len(rows) - 0.3)

    figure.suptitle("Matched geologic effects on PREDICT permeability", fontsize=21, weight="bold", y=0.975)
    axis.set_title(dataset_label, fontsize=14, color="#444444", pad=12)
    figure.text(
        0.5,
        0.018,
        "Points show balanced median matched effects; horizontal bars show subgroup-balanced interquartile ranges. "
        "Positive means the first condition has higher median permeability.",
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#3f4650",
    )
    figure.subplots_adjust(left=0.27, right=0.985, top=0.885, bottom=0.105)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "matched_geologic_effects_median_log10k"
    figure.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    """Run the complete matched-effect analysis and report validation counts."""
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    distributions = build_distribution_table(data_root)
    pairs = matched_effects(distributions)
    group_summaries = summarize_groups(pairs)
    presentation_rows = presentation_summary(group_summaries)

    write_csv(output_dir / "distribution_median_log10k.csv", distributions)
    write_csv(output_dir / "matched_effect_pairs_median_log10k.csv", pairs)
    write_csv(output_dir / "matched_effect_summary_by_subgroup.csv", group_summaries)
    write_csv(output_dir / "matched_effect_presentation_summary.csv", presentation_rows)
    plot_effects(presentation_rows, output_dir, args.dataset_label)

    print(f"Validated libraries: {len(distributions) // 3}")
    print(f"Distribution-component records: {len(distributions)}")
    print(f"Matched component pairs: {len(pairs)}")
    print(f"Presentation effects: {len(presentation_rows)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
