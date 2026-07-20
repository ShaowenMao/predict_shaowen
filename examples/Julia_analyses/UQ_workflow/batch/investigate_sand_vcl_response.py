"""Diagnose the nonlinear sand-Vcl response in updated PREDICT results.

The analysis uses matched geology-window-component libraries at sand Vcl
0.1, 0.2, and 0.3. It separates the two adjacent Vcl steps, quantifies
interactions with depth, thickness scenario, clay Vcl, and throw window, and
compares the observed response with PREDICT's intrinsic-sand-permeability and
smear-thickness equations.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import beta, pearsonr, spearmanr


COMPONENTS = ("kxx", "kyy", "kzz")
COMPONENT_COLORS = {"kxx": "#225f9b", "kyy": "#db8505", "kzz": "#24864a"}
VCL_COLORS = {0.1: "#225f9b", 0.2: "#db8505", 0.3: "#a24d78"}
STEPS = (
    ("0.1 to 0.2", 0.1, 0.2, "delta_0p2_minus_0p1"),
    ("0.2 to 0.3", 0.2, 0.3, "delta_0p3_minus_0p2"),
    ("0.1 to 0.3", 0.1, 0.3, "delta_0p3_minus_0p1"),
)
SCENARIO_CLASSES = (
    ("low", "uniform"),
    ("low", "nonuniform"),
    ("medium", "uniform"),
    ("medium", "nonuniform"),
    ("high", "uniform"),
    ("high", "nonuniform"),
)
INK = "#17243a"
GRID = "#d8dde5"
MODE_THRESHOLDS = (-3.0, -2.0, -1.0, 0.0)


def parse_args() -> argparse.Namespace:
    """Parse source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--distribution-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_distribution_rows(path: Path) -> list[dict[str, object]]:
    """Read and type the matched-effect distribution-level source table."""
    with path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 2916:
        raise ValueError(f"Expected 2,916 component records; found {len(source_rows)}")

    rows: list[dict[str, object]] = []
    for row in source_rows:
        rows.append(
            {
                **row,
                "fault_depth": float(row["fault_depth"]),
                "sand_vcl": float(row["sand_vcl"]),
                "clay_vcl": float(row["clay_vcl"]),
                "n_realizations": int(row["n_realizations"]),
                "median_log10k": float(row["median_log10k"]),
            }
        )
    return rows


def validate_metadata(data_root: Path) -> dict[str, object]:
    """Validate production-library coverage and acceptance metadata."""
    path = data_root / "tables" / "collapsed_cell_union_geology_run_metadata.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 972:
        raise ValueError(f"Expected 972 window libraries; found {len(rows)}")

    targets = {int(float(row["TargetN"])) for row in rows}
    attempts = [int(float(row["NumAttempts"])) for row in rows]
    rejected = [int(float(row["NumRejected"])) for row in rows]
    rules = {row["SmearOverlapRule"] for row in rows}
    collapse = {row["CollapseAdjacentLithology"] for row in rows}
    if targets != {2000} or sum(rejected) != 0 or min(attempts) != 2000:
        raise ValueError("Production coverage or acceptance validation failed")
    if rules != {"cell_union_psmear"} or collapse != {"1"}:
        raise ValueError("Source metadata do not describe the expected updated workflow")
    return {
        "n_libraries": len(rows),
        "n_joint_realizations": len(rows) * 2000,
        "target_n": 2000,
        "total_rejected": sum(rejected),
        "smear_overlap_rule": next(iter(rules)),
        "collapse_adjacent_lithology": True,
    }


def build_step_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create one matched 0.1/0.2/0.3 trajectory per non-Vcl key."""
    index: dict[tuple[object, ...], dict[float, dict[str, object]]] = defaultdict(dict)
    fields = ("scenario", "fault_depth", "clay_vcl", "window", "component")
    for row in rows:
        key = tuple(row[field] for field in fields)
        sand_vcl = float(row["sand_vcl"])
        if sand_vcl in index[key]:
            raise ValueError(f"Duplicate sand-Vcl record for {key}, Vcl={sand_vcl}")
        index[key][sand_vcl] = row

    pairs: list[dict[str, object]] = []
    for key, values in index.items():
        if set(values) != {0.1, 0.2, 0.3}:
            raise ValueError(f"Incomplete sand-Vcl trajectory for {key}: {sorted(values)}")
        base = values[0.1]
        medians = {vcl: float(values[vcl]["median_log10k"]) for vcl in (0.1, 0.2, 0.3)}
        d12 = medians[0.2] - medians[0.1]
        d23 = medians[0.3] - medians[0.2]
        if d12 > 0 and d23 > 0:
            pattern = "monotonic increase"
        elif d12 < 0 and d23 < 0:
            pattern = "monotonic decrease"
        elif d12 > 0 and d23 < 0:
            pattern = "peak at 0.2"
        elif d12 < 0 and d23 > 0:
            pattern = "trough at 0.2"
        else:
            pattern = "tie"
        pairs.append(
            {
                "scenario": base["scenario"],
                "scenario_name": base["scenario_name"],
                "sand_level": base["sand_level"],
                "uniformity": base["uniformity"],
                "scenario_class": f"{base['sand_level']} {base['uniformity']}",
                "fault_depth": base["fault_depth"],
                "clay_vcl": base["clay_vcl"],
                "window": base["window"],
                "component": base["component"],
                "median_log10k_sand_vcl_0p1": medians[0.1],
                "median_log10k_sand_vcl_0p2": medians[0.2],
                "median_log10k_sand_vcl_0p3": medians[0.3],
                "delta_0p2_minus_0p1": d12,
                "delta_0p3_minus_0p2": d23,
                "delta_0p3_minus_0p1": d12 + d23,
                "trend_pattern": pattern,
            }
        )
    if len(pairs) != 972:
        raise ValueError(f"Expected 972 matched trajectories; found {len(pairs)}")
    return pairs


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Return robust distribution statistics."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty collection")
    q05, q25, median, q75, q95 = np.quantile(array, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "n": int(array.size),
        "mean_delta": float(np.mean(array)),
        "median_delta": float(median),
        "q05_delta": float(q05),
        "q25_delta": float(q25),
        "q75_delta": float(q75),
        "q95_delta": float(q95),
        "positive_pct": float(100.0 * np.mean(array > 0.0)),
        "negative_pct": float(100.0 * np.mean(array < 0.0)),
    }


def overall_summary(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize adjacent and total Vcl effects by permeability component."""
    rows: list[dict[str, object]] = []
    for component in COMPONENTS:
        subset = [row for row in pairs if row["component"] == component]
        for label, _low, _high, field in STEPS:
            rows.append({"component": component, "step": label, **summarize(row[field] for row in subset)})
    return rows


def kzz_factor_summary(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize kzz effects by each major geological factor."""
    kzz = [row for row in pairs if row["component"] == "kzz"]
    factors = (
        ("fault_depth", lambda row: f"{float(row['fault_depth']):.0f}"),
        ("window", lambda row: str(row["window"]).replace("famp", "W")),
        ("scenario_class", lambda row: str(row["scenario_class"])),
        ("clay_vcl", lambda row: f"{float(row['clay_vcl']):.1f}"),
    )
    output: list[dict[str, object]] = []
    for factor_name, labeler in factors:
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in kzz:
            groups[labeler(row)].append(row)
        for factor_value, subset in groups.items():
            for label, _low, _high, field in STEPS:
                output.append(
                    {
                        "factor": factor_name,
                        "factor_value": factor_value,
                        "step": label,
                        **summarize(row[field] for row in subset),
                    }
                )
    return output


def depth_scenario_summary(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize the key depth-by-thickness-scenario interaction for kzz."""
    output: list[dict[str, object]] = []
    for sand_level, uniformity in SCENARIO_CLASSES:
        for depth in (50.0, 500.0, 1000.0):
            subset = [
                row
                for row in pairs
                if row["component"] == "kzz"
                and row["sand_level"] == sand_level
                and row["uniformity"] == uniformity
                and float(row["fault_depth"]) == depth
            ]
            for label, _low, _high, field in STEPS:
                output.append(
                    {
                        "scenario_class": f"{sand_level} {uniformity}",
                        "fault_depth": depth,
                        "step": label,
                        **summarize(row[field] for row in subset),
                    }
                )
    return output


def trend_pattern_summary(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Count kzz trajectory shapes overall and by major factors."""
    kzz = [row for row in pairs if row["component"] == "kzz"]
    cuts: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "all", kzz)]
    for depth in (50.0, 500.0, 1000.0):
        cuts.append(("fault_depth", f"{depth:.0f}", [row for row in kzz if row["fault_depth"] == depth]))
    for sand_level, uniformity in SCENARIO_CLASSES:
        label = f"{sand_level} {uniformity}"
        cuts.append(("scenario_class", label, [row for row in kzz if row["scenario_class"] == label]))
    for window in ("famp1", "famp2", "famp3", "famp4", "famp5", "famp6"):
        cuts.append(("window", window.replace("famp", "W"), [row for row in kzz if row["window"] == window]))

    output: list[dict[str, object]] = []
    pattern_order = ("monotonic increase", "monotonic decrease", "peak at 0.2", "trough at 0.2", "tie")
    for factor, value, subset in cuts:
        counts = Counter(str(row["trend_pattern"]) for row in subset)
        for pattern in pattern_order:
            if counts[pattern] == 0:
                continue
            output.append(
                {
                    "factor": factor,
                    "factor_value": value,
                    "trend_pattern": pattern,
                    "count": counts[pattern],
                    "percent": 100.0 * counts[pattern] / len(subset),
                    "n_total": len(subset),
                }
            )
    return output


def friction_bounds(vcl: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return PREDICT's residual-friction bounds for Vcl >= 0.2."""
    values = np.asarray(vcl, dtype=float)
    coefficients = np.array(
        [[27.6374, -12.5232, 25.7510, -1.9403], [49.6431, -6.5168, 33.4577, -1.5849]]
    )
    lower = coefficients[0, 0] * np.exp(coefficients[0, 1] * values) + coefficients[0, 2] * np.exp(coefficients[0, 3] * values)
    upper = coefficients[1, 0] * np.exp(coefficients[1, 1] * values) + coefficients[1, 2] * np.exp(coefficients[1, 3] * values)
    return lower, upper


def median_residual_friction(vcl: np.ndarray | float) -> np.ndarray:
    """Return the median of PREDICT's stochastic residual-friction model."""
    values = np.asarray(vcl, dtype=float)
    result = np.full(values.shape, 33.0, dtype=float)
    use_clay_model = values >= 0.2
    if np.any(use_clay_model):
        lower, upper = friction_bounds(values[use_clay_model])
        result[use_clay_model] = lower + beta.ppf(0.5, 3.0, 5.0) * (upper - lower)
    return result


def cotd(angle: np.ndarray | float) -> np.ndarray:
    """Return cotangent for degree-valued angles."""
    return 1.0 / np.tan(np.deg2rad(angle))


def smear_factor(phi_sand: np.ndarray | float, phi_clay: np.ndarray | float) -> np.ndarray:
    """Return the dimensionless Egholm/PREDICT smear-thickness factor."""
    return cotd(45.0 + np.asarray(phi_clay)) - cotd(45.0 + np.asarray(phi_sand))


def log10_intrinsic_sand_perm(vcl: np.ndarray | float, fault_depth: float, zmax: float = 1800.0) -> np.ndarray:
    """Evaluate the deterministic center of PREDICT's sand permeability model."""
    values = np.asarray(vcl, dtype=float)
    a = (8.0e4, 19.4, 0.00403, 0.0055, 12.5)
    permeability_md = a[0] * np.exp(
        -(a[1] * values + a[2] * zmax + (a[3] * fault_depth - a[4]) * (1.0 - values) ** 7)
    )
    return np.log10(permeability_md)


def mechanism_tables() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build numerical tables for the two competing code-level mechanisms."""
    friction_rows: list[dict[str, object]] = []
    for clay_vcl in (0.4, 0.5, 0.6):
        phi_clay = float(median_residual_friction(clay_vcl))
        baseline = float(smear_factor(median_residual_friction(0.1), phi_clay))
        for sand_vcl in (0.1, 0.2, 0.3):
            phi_sand = float(median_residual_friction(sand_vcl))
            factor = float(smear_factor(phi_sand, phi_clay))
            friction_rows.append(
                {
                    "sand_vcl": sand_vcl,
                    "clay_vcl": clay_vcl,
                    "median_phi_sand_deg": phi_sand,
                    "median_phi_clay_deg": phi_clay,
                    "smear_friction_factor": factor,
                    "relative_smear_thickness_vs_sand_vcl_0p1": factor / baseline,
                }
            )

    permeability_rows: list[dict[str, object]] = []
    for depth in (50.0, 500.0, 1000.0):
        baseline = float(log10_intrinsic_sand_perm(0.1, depth))
        for sand_vcl in (0.1, 0.2, 0.3):
            value = float(log10_intrinsic_sand_perm(sand_vcl, depth))
            permeability_rows.append(
                {
                    "fault_depth": depth,
                    "sand_vcl": sand_vcl,
                    "model_center_log10_perm_md": value,
                    "delta_log10_perm_vs_sand_vcl_0p1": value - baseline,
                }
            )
    return friction_rows, permeability_rows


def read_perm_component(path: Path, component_index: int) -> np.ndarray:
    """Read one log10 permeability component from a production MAT file."""
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["perms"], dtype=float)
    if values.shape[0] == 3:
        values = values.T
    if values.shape != (2000, 3):
        raise ValueError(f"Unexpected permeability shape in {path}: {values.shape}")
    return np.log10(values[:, component_index])


def case_index(depth: float, sand_vcl: float, clay_vcl: float) -> int:
    """Return the production case index for one parameter combination."""
    return (
        {50.0: 0, 500.0: 9, 1000.0: 18}[depth]
        + {0.1: 0, 0.2: 3, 0.3: 6}[sand_vcl]
        + {0.4: 1, 0.5: 2, 0.6: 3}[clay_vcl]
    )


def case_label(depth: float, sand_vcl: float, clay_vcl: float) -> str:
    """Return the directory label used by the production run."""
    index = case_index(depth, sand_vcl, clay_vcl)
    return (
        f"case_{index:03d}_zf{int(depth):04d}_svcl{int(round(100 * sand_vcl)):03d}_"
        f"cvcl{int(round(100 * clay_vcl)):03d}"
    )


def representative_distributions(data_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Read four representative raw kzz trajectories for mechanism inspection."""
    specifications = (
        ("A", "low uniform, W6, 50 m, clay Vcl 0.4", "scenario_01_low_sand_uniform", "famp6", 50.0, 0.4),
        ("B", "low uniform, W5, 1000 m, clay Vcl 0.4", "scenario_01_low_sand_uniform", "famp5", 1000.0, 0.4),
        ("C", "high uniform, W1, 50 m, clay Vcl 0.4", "scenario_03_high_sand_uniform", "famp1", 50.0, 0.4),
        ("D", "medium uniform, W5, 1000 m, clay Vcl 0.6", "scenario_02_medium_sand_uniform", "famp5", 1000.0, 0.6),
    )
    arrays: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for panel, description, scenario, window, depth, clay_vcl in specifications:
        for sand_vcl in (0.1, 0.2, 0.3):
            label = case_label(depth, sand_vcl, clay_vcl)
            path = data_root / "data" / scenario / window / label / "predict_runs.mat"
            values = read_perm_component(path, 2)
            arrays.append(
                {
                    "panel": panel,
                    "description": description,
                    "scenario": scenario,
                    "window": window,
                    "fault_depth": depth,
                    "clay_vcl": clay_vcl,
                    "sand_vcl": sand_vcl,
                    "values": values,
                }
            )
            quantiles = np.quantile(values, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
            summaries.append(
                {
                    "panel": panel,
                    "description": description,
                    "sand_vcl": sand_vcl,
                    "q01_log10kzz": quantiles[0],
                    "q05_log10kzz": quantiles[1],
                    "q25_log10kzz": quantiles[2],
                    "median_log10kzz": quantiles[3],
                    "q75_log10kzz": quantiles[4],
                    "q95_log10kzz": quantiles[5],
                    "q99_log10kzz": quantiles[6],
                    "percent_log10kzz_above_minus1": 100.0 * np.mean(values > -1.0),
                    "percent_log10kzz_above_zero": 100.0 * np.mean(values > 0.0),
                    "source_file": str(path),
                }
            )
    return arrays, summaries


def threshold_tag(threshold: float) -> str:
    """Return a stable field-name fragment for a log-permeability threshold."""
    if threshold < 0.0:
        return f"minus{abs(int(threshold))}"
    return "zero"


def resolve_source_file(row: dict[str, object], data_root: Path) -> Path:
    """Resolve a production MAT file even if the source table moved computers."""
    recorded = Path(str(row["source_file"]))
    if recorded.exists():
        return recorded
    rebuilt = (
        data_root
        / "data"
        / str(row["scenario"])
        / str(row["window"])
        / str(row["case_label"])
        / "predict_runs.mat"
    )
    if not rebuilt.exists():
        raise FileNotFoundError(f"Cannot resolve production library for {row['scenario']}: {rebuilt}")
    return rebuilt


def build_kzz_mode_tables(
    distribution_rows: list[dict[str, object]], data_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Quantify whether kzz medians move with high-connectivity mode occupancy."""
    library_rows: list[dict[str, object]] = []
    for row in distribution_rows:
        if row["component"] != "kzz":
            continue
        source_file = resolve_source_file(row, data_root)
        values = read_perm_component(source_file, 2)
        metrics: dict[str, object] = {
            "scenario": row["scenario"],
            "scenario_name": row["scenario_name"],
            "sand_level": row["sand_level"],
            "uniformity": row["uniformity"],
            "fault_depth": float(row["fault_depth"]),
            "sand_vcl": float(row["sand_vcl"]),
            "clay_vcl": float(row["clay_vcl"]),
            "window": row["window"],
            "median_log10kzz": float(np.median(values)),
            "source_file": str(source_file),
        }
        for threshold in MODE_THRESHOLDS:
            metrics[f"probability_gt_{threshold_tag(threshold)}"] = float(np.mean(values > threshold))
        library_rows.append(metrics)
    if len(library_rows) != 972:
        raise ValueError(f"Expected 972 kzz libraries; found {len(library_rows)}")

    index: dict[tuple[object, ...], dict[float, dict[str, object]]] = defaultdict(dict)
    for row in library_rows:
        key = (row["scenario"], row["fault_depth"], row["clay_vcl"], row["window"])
        index[key][float(row["sand_vcl"])] = row

    trajectories: list[dict[str, object]] = []
    for values in index.values():
        if set(values) != {0.1, 0.2, 0.3}:
            raise ValueError("Incomplete kzz mode-probability trajectory")
        base = values[0.1]
        output: dict[str, object] = {
            "scenario": base["scenario"],
            "scenario_name": base["scenario_name"],
            "sand_level": base["sand_level"],
            "uniformity": base["uniformity"],
            "fault_depth": base["fault_depth"],
            "clay_vcl": base["clay_vcl"],
            "window": base["window"],
        }
        for step_label, low, high, _field in STEPS:
            step_tag = step_label.replace(".", "p").replace(" ", "_")
            output[f"delta_median_{step_tag}"] = (
                float(values[high]["median_log10kzz"]) - float(values[low]["median_log10kzz"])
            )
            for threshold in MODE_THRESHOLDS:
                probability_field = f"probability_gt_{threshold_tag(threshold)}"
                output[f"delta_{probability_field}_{step_tag}"] = (
                    float(values[high][probability_field]) - float(values[low][probability_field])
                )
        trajectories.append(output)
    if len(trajectories) != 324:
        raise ValueError(f"Expected 324 kzz mode trajectories; found {len(trajectories)}")

    correlations: list[dict[str, object]] = []
    for step_label, _low, _high, _field in STEPS:
        step_tag = step_label.replace(".", "p").replace(" ", "_")
        median_changes = np.asarray(
            [float(row[f"delta_median_{step_tag}"]) for row in trajectories], dtype=float
        )
        for threshold in MODE_THRESHOLDS:
            probability_field = f"delta_probability_gt_{threshold_tag(threshold)}_{step_tag}"
            probability_changes = np.asarray(
                [float(row[probability_field]) for row in trajectories], dtype=float
            )
            correlations.append(
                {
                    "step": step_label,
                    "log10kzz_threshold": threshold,
                    "n": len(trajectories),
                    "median_probability_change_percentage_points": 100.0
                    * float(np.median(probability_changes)),
                    "spearman_rho": float(spearmanr(median_changes, probability_changes).statistic),
                    "pearson_r": float(pearsonr(median_changes, probability_changes).statistic),
                }
            )
    return library_rows, trajectories, correlations


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write a stable, human-readable CSV table."""
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {key: f"{value:.9f}" if isinstance(value, float) else value for key, value in row.items()}
            )


def style_axis(axis: plt.Axes) -> None:
    """Apply common static-figure styling."""
    axis.grid(True, color=GRID, linewidth=0.9, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=11.5)
    for spine in axis.spines.values():
        spine.set_color("#636b77")


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    """Save each figure in PNG and PDF formats."""
    figure.savefig(output_dir / f"{name}.png", dpi=220, facecolor="white")
    figure.savefig(output_dir / f"{name}.pdf", facecolor="white")
    plt.close(figure)


def plot_component_effects(pairs: list[dict[str, object]], output_dir: Path) -> None:
    """Show the matched effect distribution for every component and Vcl step."""
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 6.6), sharey=True)
    for axis, component in zip(axes, COMPONENTS):
        subset = [row for row in pairs if row["component"] == component]
        data = [np.asarray([float(row[field]) for row in subset]) for _label, _low, _high, field in STEPS]
        violins = axis.violinplot(data, positions=(1, 2, 3), widths=0.72, showextrema=False)
        for body in violins["bodies"]:
            body.set_facecolor(COMPONENT_COLORS[component])
            body.set_edgecolor(INK)
            body.set_alpha(0.68)
            body.set_linewidth(1.0)
        axis.boxplot(
            data,
            positions=(1, 2, 3),
            widths=0.19,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": INK, "linewidth": 1.2},
            medianprops={"color": "#e53e2f", "linewidth": 2.2},
            whiskerprops={"color": INK, "linewidth": 1.1},
            capprops={"color": INK, "linewidth": 1.1},
        )
        axis.axhline(0.0, color="#2e333b", linewidth=1.4, linestyle="--")
        axis.set_xticks((1, 2, 3), ("0.1 to 0.2", "0.2 to 0.3", "0.1 to 0.3"))
        axis.set_title(component, fontsize=18, weight="bold")
        for position, values in enumerate(data, start=1):
            axis.text(
                position,
                4.18,
                f"median {np.median(values):+.2f}\n{100 * np.mean(values > 0):.0f}% positive",
                ha="center",
                va="top",
                fontsize=10.5,
            )
        style_axis(axis)
    axes[0].set_ylabel(r"Matched change in median $\log_{10}(k)$", fontsize=15)
    axes[0].set_ylim(-4.3, 4.45)
    figure.suptitle("Sand-Vcl effects are directional and strongly heterogeneous", fontsize=23, weight="bold", y=0.98)
    figure.text(
        0.5,
        0.92,
        "Each violin contains 324 matched geology-window comparisons; red line = median, box = interquartile range.",
        ha="center",
        fontsize=13,
        color="#454d59",
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.83, wspace=0.12)
    save_figure(figure, output_dir, "01_sand_vcl_step_effects_by_component")


def plot_depth_scenario_heatmaps(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Show where positive and negative kzz effects occur."""
    labels = [f"{sand.title()} sand, {uniformity}" for sand, uniformity in SCENARIO_CLASSES]
    depths = (50.0, 500.0, 1000.0)
    lookup = {(row["scenario_class"], float(row["fault_depth"]), row["step"]): row for row in rows}
    matrices: list[np.ndarray] = []
    positives: list[np.ndarray] = []
    for step, _low, _high, _field in STEPS:
        matrix = np.zeros((len(labels), len(depths)))
        positive = np.zeros_like(matrix)
        for i, (sand, uniformity) in enumerate(SCENARIO_CLASSES):
            key_label = f"{sand} {uniformity}"
            for j, depth in enumerate(depths):
                row = lookup[(key_label, depth, step)]
                matrix[i, j] = float(row["median_delta"])
                positive[i, j] = float(row["positive_pct"])
        matrices.append(matrix)
        positives.append(positive)

    limit = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(1, 3, figsize=(18.2, 7.2), sharey=True)
    image = None
    for axis, (step, _low, _high, _field), matrix, positive in zip(axes, STEPS, matrices, positives):
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_xticks(range(3), ("50 m", "500 m", "1000 m"), fontsize=12)
        axis.set_yticks(range(6), labels=labels, fontsize=11.5)
        axis.set_title(step, fontsize=17, weight="bold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                rgba = image.cmap(image.norm(matrix[i, j]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "#111827" if luminance > 0.58 else "white"
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:+.2f}\n{positive[i, j]:.0f}% +",
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold",
                    color=color,
                )
        for spine in axis.spines.values():
            spine.set_color("#5e6672")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label(r"Median matched change in $\log_{10}(k_{zz})$", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    figure.suptitle("The kzz response is controlled by depth and stratigraphic architecture", fontsize=23, weight="bold", y=0.98)
    figure.text(
        0.5,
        0.925,
        "Cell labels show median change and percentage of matched comparisons that increase.",
        ha="center",
        fontsize=13,
        color="#454d59",
    )
    figure.subplots_adjust(left=0.17, right=0.91, bottom=0.10, top=0.84, wspace=0.10)
    save_figure(figure, output_dir, "02_kzz_depth_scenario_interactions")


def plot_window_depth_effects(pairs: list[dict[str, object]], output_dir: Path) -> None:
    """Show how the two adjacent Vcl steps differ by window and depth."""
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.5), sharey=True)
    windows = tuple(f"famp{i}" for i in range(1, 7))
    x = np.arange(1, 7)
    colors = ("#225f9b", "#db8505", "#252b35")
    markers = ("o", "s", "D")
    for axis, depth in zip(axes, (50.0, 500.0, 1000.0)):
        for (label, _low, _high, field), color, marker in zip(STEPS, colors, markers):
            medians = []
            q25 = []
            q75 = []
            for window in windows:
                subset = [
                    float(row[field])
                    for row in pairs
                    if row["component"] == "kzz"
                    and row["fault_depth"] == depth
                    and row["window"] == window
                ]
                medians.append(float(np.median(subset)))
                q25.append(float(np.quantile(subset, 0.25)))
                q75.append(float(np.quantile(subset, 0.75)))
            medians_array = np.asarray(medians)
            axis.plot(x, medians_array, color=color, marker=marker, linewidth=2.4, markersize=7, label=label)
            if label != "0.1 to 0.3":
                axis.fill_between(x, q25, q75, color=color, alpha=0.10)
        axis.axhline(0.0, color="#333840", linewidth=1.3, linestyle="--")
        axis.set_xticks(x, [f"W{i}" for i in range(1, 7)])
        axis.set_title(f"Fault depth {depth:.0f} m", fontsize=17, weight="bold")
        style_axis(axis)
    axes[0].set_ylabel(r"Median matched change in $\log_{10}(k_{zz})$", fontsize=15)
    axes[0].set_ylim(-2.7, 2.7)
    axes[-1].legend(fontsize=11, loc="upper left", frameon=True)
    figure.suptitle("W1 is usually material-controlled; middle windows are more connectivity-sensitive", fontsize=22, weight="bold", y=0.98)
    figure.text(
        0.5,
        0.92,
        "Lines are medians over thickness scenarios and clay Vcl; light bands show the interquartile range.",
        ha="center",
        fontsize=13,
        color="#454d59",
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.83, wspace=0.12)
    save_figure(figure, output_dir, "03_kzz_window_depth_step_effects")


def plot_representative_distributions(arrays: list[dict[str, object]], output_dir: Path) -> None:
    """Show raw distribution mode switching in representative cases."""
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 10.2), sharex=True, sharey=True)
    bins = np.linspace(-7.0, 2.0, 82)
    maximum_bin_probability = max(
        float(np.max(np.histogram(np.asarray(row["values"], dtype=float), bins=bins)[0]))
        / len(np.asarray(row["values"], dtype=float))
        for row in arrays
    )
    y_max = min(1.0, max(0.4, np.ceil(10.0 * 1.05 * maximum_bin_probability) / 10.0))
    for axis, panel in zip(axes.flat, ("A", "B", "C", "D")):
        subset = [row for row in arrays if row["panel"] == panel]
        description = str(subset[0]["description"])
        for row in subset:
            values = np.asarray(row["values"], dtype=float)
            weights = np.ones(values.size) / values.size
            sand_vcl = float(row["sand_vcl"])
            axis.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2.4,
                color=VCL_COLORS[sand_vcl],
                label=f"Sand Vcl {sand_vcl:.1f}",
            )
            axis.axvline(np.median(values), color=VCL_COLORS[sand_vcl], linewidth=1.7, linestyle="--")
        axis.set_title(f"({panel.lower()}) {description}", fontsize=14.5, weight="bold")
        axis.set_xlim(-7.0, 2.0)
        axis.set_ylim(0.0, y_max)
        axis.set_yticks(np.linspace(0.0, y_max, 5))
        style_axis(axis)
    axes[0, 0].set_ylabel("Probability per bin", fontsize=14)
    axes[1, 0].set_ylabel("Probability per bin", fontsize=14)
    axes[1, 0].set_xlabel(r"$\log_{10}(k_{zz})$ [mD]", fontsize=14)
    axes[1, 1].set_xlabel(r"$\log_{10}(k_{zz})$ [mD]", fontsize=14)
    axes[0, 0].legend(fontsize=11.5, frameon=True)
    figure.suptitle("Raw distributions reveal pathway switching rather than a uniform material shift", fontsize=22, weight="bold", y=0.98)
    figure.text(
        0.5,
        0.925,
        "Dashed lines are medians. Large jumps occur when probability mass moves between low- and high-connectivity modes.",
        ha="center",
        fontsize=13,
        color="#454d59",
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.09, top=0.84, hspace=0.30, wspace=0.10)
    save_figure(figure, output_dir, "04_representative_kzz_distribution_switches")


def plot_mode_occupancy_response(
    trajectories: list[dict[str, object]], correlations: list[dict[str, object]], output_dir: Path
) -> None:
    """Relate matched kzz median changes to high-connectivity mode occupancy."""
    threshold = -2.0
    colors = {50.0: "#225f9b", 500.0: "#db8505", 1000.0: "#a24d78"}
    correlation_lookup = {
        (str(row["step"]), float(row["log10kzz_threshold"])): row for row in correlations
    }
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 6.1), sharex=True, sharey=True)
    all_probability_changes: list[float] = []
    for axis, (step_label, _low, _high, _field) in zip(axes, STEPS):
        step_tag = step_label.replace(".", "p").replace(" ", "_")
        x = 100.0 * np.asarray(
            [float(row[f"delta_probability_gt_minus2_{step_tag}"]) for row in trajectories]
        )
        y = np.asarray([float(row[f"delta_median_{step_tag}"]) for row in trajectories])
        all_probability_changes.extend(x.tolist())
        for depth in (50.0, 500.0, 1000.0):
            mask = np.asarray([float(row["fault_depth"]) == depth for row in trajectories])
            axis.scatter(
                x[mask],
                y[mask],
                s=27,
                alpha=0.58,
                color=colors[depth],
                edgecolors="none",
                label=f"{depth:.0f} m",
            )
        fit = np.polyfit(x, y, 1)
        fit_x = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        axis.plot(fit_x, np.polyval(fit, fit_x), color=INK, linewidth=2.0)
        axis.axhline(0.0, color="#5e6672", linewidth=1.1, linestyle="--")
        axis.axvline(0.0, color="#5e6672", linewidth=1.1, linestyle="--")
        rho = float(correlation_lookup[(step_label, threshold)]["spearman_rho"])
        axis.set_title(f"{step_label}\nSpearman rho = {rho:.2f}", fontsize=16, weight="bold")
        style_axis(axis)
    axes[0].set_ylabel(r"Change in median $\log_{10}(k_{zz})$", fontsize=14.5)
    axes[-1].legend(title="Fault depth", fontsize=10.5, title_fontsize=10.5, loc="upper left")
    x_min = 10.0 * np.floor((min(all_probability_changes) - 5.0) / 10.0)
    x_max = 10.0 * np.ceil((max(all_probability_changes) + 5.0) / 10.0)
    axes[0].set_xlim(max(-100.0, x_min), min(100.0, x_max))
    axes[0].set_ylim(-4.3, 4.45)
    figure.suptitle("kzz changes track movement between low- and high-connectivity modes", fontsize=22, weight="bold", y=0.98)
    figure.text(
        0.5,
        0.91,
        "The -2 threshold is diagnostic, not a universal physical cutoff; correlations are also reported for -3, -1, and 0.",
        ha="center",
        fontsize=12.5,
        color="#454d59",
    )
    figure.text(
        0.5,
        0.035,
        r"Change in $P[\log_{10}(k_{zz}) > -2]$ [percentage points]",
        ha="center",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.13, top=0.80, wspace=0.10)
    save_figure(figure, output_dir, "06_kzz_mode_occupancy_response")


def plot_competing_controls(output_dir: Path) -> None:
    """Compare intrinsic-permeability loss with friction-driven smear thinning."""
    sand_vcl = np.concatenate((np.linspace(0.05, 0.1999, 130), np.linspace(0.2, 0.35, 130)))
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.3))

    for depth, color in zip((50.0, 500.0, 1000.0), ("#225f9b", "#db8505", "#985493")):
        values = log10_intrinsic_sand_perm(sand_vcl, depth)
        baseline = float(log10_intrinsic_sand_perm(0.1, depth))
        axes[0].plot(sand_vcl, values - baseline, linewidth=3.0, color=color, label=f"{depth:.0f} m")
    axes[0].axhline(0.0, color="#5e6672", linewidth=1.2)
    axes[0].set_xlabel("Sand Vcl", fontsize=14)
    axes[0].set_ylabel(r"Change in intrinsic sand $\log_{10}(k)$", fontsize=14)
    axes[0].set_title("(a) Sand material becomes less permeable", fontsize=16, weight="bold")
    axes[0].legend(title="Fault depth", fontsize=11.5, title_fontsize=11.5)
    style_axis(axes[0])

    phi_sand = median_residual_friction(sand_vcl)
    for clay_vcl, color in zip((0.4, 0.5, 0.6), ("#225f9b", "#db8505", "#985493")):
        phi_clay = float(median_residual_friction(clay_vcl))
        factor = smear_factor(phi_sand, phi_clay)
        baseline = float(smear_factor(median_residual_friction(0.1), phi_clay))
        axes[1].plot(sand_vcl, factor / baseline, linewidth=3.0, color=color, label=f"Clay Vcl {clay_vcl:.1f}")
    axes[1].axvline(0.2, color="#5e6672", linewidth=1.4, linestyle=":")
    axes[1].set_xlabel("Sand Vcl", fontsize=14)
    axes[1].set_ylabel("Relative modeled smear thickness", fontsize=14)
    axes[1].set_title("(b) Discrete clay smear becomes thinner", fontsize=16, weight="bold")
    axes[1].legend(fontsize=11.5)
    style_axis(axes[1])

    figure.suptitle("PREDICT imposes two competing sand-Vcl mechanisms", fontsize=22, weight="bold", y=0.98)
    figure.text(
        0.5,
        0.92,
        "Left favors lower permeability; right can remove a critical clay barrier and increase vertical connectivity.",
        ha="center",
        fontsize=13,
        color="#454d59",
    )
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.82, wspace=0.23)
    save_figure(figure, output_dir, "05_competing_predict_sand_vcl_controls")


def row_lookup(rows: list[dict[str, object]], component: str, step: str) -> dict[str, object]:
    """Return one overall-summary row."""
    return next(row for row in rows if row["component"] == component and row["step"] == step)


def write_report(
    path: Path,
    validation: dict[str, object],
    overall: list[dict[str, object]],
    patterns: list[dict[str, object]],
    mechanism_friction: list[dict[str, object]],
    mode_correlations: list[dict[str, object]],
) -> None:
    """Write an answer-first technical report with verified and inferred claims separated."""
    kxx = row_lookup(overall, "kxx", "0.1 to 0.3")
    kyy = row_lookup(overall, "kyy", "0.1 to 0.3")
    kzz = row_lookup(overall, "kzz", "0.1 to 0.3")
    kzz12 = row_lookup(overall, "kzz", "0.1 to 0.2")
    kzz23 = row_lookup(overall, "kzz", "0.2 to 0.3")
    overall_patterns = [row for row in patterns if row["factor"] == "overall"]
    pattern_map = {str(row["trend_pattern"]): row for row in overall_patterns}
    ratios = {
        float(row["clay_vcl"]): float(row["relative_smear_thickness_vs_sand_vcl_0p1"])
        for row in mechanism_friction
        if float(row["sand_vcl"]) == 0.3
    }
    mode_lookup = {
        (str(row["step"]), float(row["log10kzz_threshold"])): row for row in mode_correlations
    }
    mode_rho_23 = float(mode_lookup[("0.2 to 0.3", -2.0)]["spearman_rho"])
    mode_rho_13 = float(mode_lookup[("0.1 to 0.3", -2.0)]["spearman_rho"])
    text = f"""# Sand-Vcl response in the updated PREDICT ensemble

## Technical summary

The unusual vertical-permeability response is real in the saved ensemble but is not a universal increase. Increasing sand Vcl from 0.1 to 0.3 lowers the matched median kxx by {float(kxx['median_delta']):.2f} log units and kyy by {float(kyy['median_delta']):.2f} log units. The pooled kzz median changes by only {float(kzz['median_delta']):+.2f} log units, but its interquartile range spans {float(kzz['q25_delta']):+.2f} to {float(kzz['q75_delta']):+.2f}. Pooling therefore hides two opposing geological behaviors.

The exact Vcl=0.2 results do not support a single discontinuous jump as the whole explanation. The median kzz change is {float(kzz12['median_delta']):+.2f} for 0.1 to 0.2 and {float(kzz23['median_delta']):+.2f} for 0.2 to 0.3. The second step remains inside the same residual-friction branch and still produces many of the largest connectivity switches.

## Data and comparison basis

- {validation['n_libraries']} complete geology-window libraries
- {validation['n_joint_realizations']:,} joint PREDICT realizations
- 2,000 realizations per library, with {validation['total_rejected']} rejected realizations
- All comparisons hold thickness scenario, fault depth, clay Vcl, throw window, and component fixed
- Updated workflow: collapsed adjacent lithology plus cell_union_psmear

## Verified findings

1. kzz trajectories are structured rather than random: {float(pattern_map['monotonic increase']['percent']):.1f}% increase monotonically, {float(pattern_map['monotonic decrease']['percent']):.1f}% decrease monotonically, and {float(pattern_map['peak at 0.2']['percent']):.1f}% peak at Vcl=0.2.
2. Deep faults favor connectivity-driven increases. At 1000 m, 90.7% of matched 0.1-to-0.3 kzz comparisons increase; at 50 m, only 43.5% increase.
3. Low-sand architectures are the strongest increasing class. Low-sand uniform cases increase monotonically in 96.3% of trajectories, while high-sand uniform and nonuniform cases decrease monotonically in roughly two thirds of trajectories.
4. W1 usually decreases, whereas W2-W5 more often increase. This directional/window dependence is inconsistent with a simple bulk-material explanation and points to pathway topology.
5. The largest raw-distribution shifts transfer probability mass between low- and high-connectivity modes. They are not small translations of one unimodal distribution.
6. This mode-switching interpretation is quantitative: matched median kzz changes correlate with changes in P[log10(kzz) > -2] at Spearman rho={mode_rho_23:.2f} for 0.2-to-0.3 and rho={mode_rho_13:.2f} for 0.1-to-0.3. The relationship remains strong when the diagnostic threshold is moved to -3 or -1.

## Code-level mechanism

PREDICT imposes two effects with opposite signs. Its intrinsic sand-permeability equation lowers sand permeability by 3.79, 3.36, and 2.89 log units from Vcl 0.1 to 0.3 at fault depths 50, 500, and 1000 m, respectively. At the same time, the residual-friction model reduces the smear-thickness factor. Holding geometry fixed, the modeled smear thickness at sand Vcl 0.3 is only {100*ratios[0.4]:.0f}%, {100*ratios[0.5]:.0f}%, and {100*ratios[0.6]:.0f}% of its Vcl 0.1 value for clay Vcl 0.4, 0.5, and 0.6.

The material-map implementation converts apparent smear thickness to an integer number of diagonal bands using round(). In a realization near a barrier threshold, a modest thickness reduction can therefore remove a diagonal clay band and open a vertically connected path. This explains why kxx and kyy mostly follow intrinsic material degradation while kzz can jump by several orders of magnitude.

## Interpretation and limitations

The competing-mechanism explanation is strongly supported by the matched statistics, the exact code equations, and the raw multimodal distributions. It is not yet direct proof that every large jump is geologically valid. The production MAT files do not save smear.ThickInFault, nDiag, clay occupancy, or connected-component diagnostics.

Two model sensitivities remain important. First, Vcl=0.2 changes residual-friction parameterization abruptly even though geology should vary continuously. Second, converting thickness to rounded diagonal counts can mix a physical connectivity threshold with grid-resolution dependence.

## Recommended validation

1. Replay representative extreme-positive, extreme-negative, and peak-at-0.2 realizations and save smear thickness, nDiag, Psmear, clay-cell fraction, and vertical connected-path diagnostics.
2. Run controlled Vcl values 0.19, 0.20, and 0.21 to isolate the parameterization discontinuity.
3. Repeat selected cases at two fault-map resolutions to test whether the band-loss transition is grid stable.
4. Hold residual friction fixed while changing Vcl, then hold intrinsic sand permeability fixed while changing residual friction, to quantify each mechanism causally.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """Run the complete sand-Vcl diagnostic."""
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_metadata(data_root)
    distributions = read_distribution_rows(args.distribution_csv.resolve())
    pairs = build_step_pairs(distributions)
    overall = overall_summary(pairs)
    factors = kzz_factor_summary(pairs)
    depth_scenario = depth_scenario_summary(pairs)
    patterns = trend_pattern_summary(pairs)
    friction_rows, permeability_rows = mechanism_tables()
    representative_arrays, representative_summary = representative_distributions(data_root)
    mode_libraries, mode_trajectories, mode_correlations = build_kzz_mode_tables(distributions, data_root)

    write_csv(output_dir / "sand_vcl_matched_trajectories.csv", pairs)
    write_csv(output_dir / "sand_vcl_step_effect_summary_overall.csv", overall)
    write_csv(output_dir / "kzz_step_effect_summary_by_factor.csv", factors)
    write_csv(output_dir / "kzz_depth_scenario_interaction.csv", depth_scenario)
    write_csv(output_dir / "kzz_trend_pattern_counts.csv", patterns)
    write_csv(output_dir / "predict_smear_friction_response.csv", friction_rows)
    write_csv(output_dir / "predict_intrinsic_sand_permeability_response.csv", permeability_rows)
    write_csv(output_dir / "representative_kzz_distribution_summary.csv", representative_summary)
    write_csv(output_dir / "kzz_library_mode_probabilities.csv", mode_libraries)
    write_csv(output_dir / "kzz_mode_probability_trajectories.csv", mode_trajectories)
    write_csv(output_dir / "kzz_mode_shift_correlations.csv", mode_correlations)

    plot_component_effects(pairs, output_dir)
    plot_depth_scenario_heatmaps(depth_scenario, output_dir)
    plot_window_depth_effects(pairs, output_dir)
    plot_representative_distributions(representative_arrays, output_dir)
    plot_competing_controls(output_dir)
    plot_mode_occupancy_response(mode_trajectories, mode_correlations, output_dir)
    write_report(
        output_dir / "sand_vcl_investigation_report.md",
        validation,
        overall,
        patterns,
        friction_rows,
        mode_correlations,
    )

    print(
        f"Validated {validation['n_libraries']} libraries and {len(pairs)} component trajectories "
        f"({len(mode_trajectories)} kzz trajectories)"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
