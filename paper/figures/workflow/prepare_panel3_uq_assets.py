"""Create compact, matched uncertainty-quantification graphics for Panel 3.

The publication source figures contain several panels and were previously
cropped inside the workflow graphic.  Cropping left the two mini-plots with
different aspect ratios, typography, and visual weight.  This workflow-only
renderer reads the same audited QOI tables and redraws only the two quantities
needed by the overview:

* maximum upward CO2 migration for Cases 5--7; and
* compartment-wise CO2 containment for Case 5.

Original QOI data and publication plotting scripts are never modified.  The
source-file manifest hashes are checked before any derivative is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path(
    r"D:\codex_gom\qoi_analysis\effective_globalplateau_cases5_7_candidates_20260731"
) / "source_file_manifest.csv"

CASE_STYLES = (
    ("Case 5", "#D55E00", "-"),
    ("Case 6", "#0072B2", "--"),
    ("Case 7", "#009E73", "-."),
)

COMPARTMENTS = (
    ("storage_lm2", "#0072B2"),
    ("stratigraphy_sand", "#E69F00"),
    ("clay_and_seal", "#999999"),
    ("fault_all", "#CC79A7"),
    ("overburden_mmum_younger", "#D55E00"),
)

PLOT_TIME_MIN_YEARS = 1.0e-1
PLOT_TIME_MAX_YEARS = 1.0e3
INJECTION_END_YEARS = 50.0


def parse_args() -> argparse.Namespace:
    """Parse source-manifest and destination options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "assets" / "derived",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_source_files(manifest_path: Path) -> dict[tuple[str, str], Path]:
    """Resolve and hash-check the QOI tables used by these two graphics."""

    required_names = {
        "regional_co2_inventory_steps.tsv",
        "leakage_global_steps.tsv",
    }
    resolved: dict[tuple[str, str], Path] = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            source_path = Path(row["source_path"])
            if source_path.name not in required_names:
                continue
            key = (row["case"], source_path.name)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            observed = sha256_file(source_path)
            expected = row["sha256"].lower()
            if observed.lower() != expected:
                raise RuntimeError(
                    f"Source hash mismatch for {source_path}: "
                    f"expected {expected}, observed {observed}"
                )
            resolved[key] = source_path

    expected_keys = {
        (case_name, file_name)
        for case_name, _, _ in CASE_STYLES
        for file_name in required_names
    }
    missing = sorted(expected_keys - resolved.keys())
    if missing:
        raise RuntimeError(f"Manifest does not resolve required source files: {missing}")
    return resolved


def read_domain_all(path: Path) -> dict[str, np.ndarray]:
    """Read the domain-wide plume envelope needed for upward migration."""

    rows: list[tuple[int, float, int, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["region_id"] != "domain_all":
                continue
            rows.append(
                (
                    int(row["step"]),
                    float(row["time_years"]),
                    int(row["plume_cell_count_sg_ge_1e_4"]),
                    float(row["plume_z_min_m"]),
                )
            )
    rows.sort(key=lambda item: item[0])
    if len(rows) != 210:
        raise RuntimeError(f"Expected 210 domain_all rows in {path}; found {len(rows)}")

    times = np.asarray([row[1] for row in rows], dtype=float)
    counts = np.asarray([row[2] for row in rows], dtype=int)
    z_min = np.asarray([row[3] for row in rows], dtype=float)
    present = counts > 0
    if not np.any(present):
        raise RuntimeError(f"No thresholded plume cells found in {path}")
    injection_depth = float(z_min[np.flatnonzero(present)[0]])
    upward_reach_km = np.maximum(injection_depth - z_min, 0.0) / 1000.0
    upward_reach_km[~present] = 0.0
    return {"time_years": times, "upward_reach_km": upward_reach_km}


def read_final_injected_mass(path: Path) -> float:
    """Read final net injected CO2 mass from the global QOI table."""

    final_mass = np.nan
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            final_mass = float(row["net_domain_co2_change_kg"])
    if not np.isfinite(final_mass) or final_mass <= 0.0:
        raise RuntimeError(f"Invalid final injected mass in {path}: {final_mass}")
    return final_mass


def read_containment(path: Path, final_mass: float) -> dict[str, np.ndarray]:
    """Read Case 5 compartment inventories and normalize by injected mass."""

    source_ids = {
        "storage_lm2",
        "stratigraphy_sand",
        "stratigraphy_clay",
        "complete_top_seal_amphb",
        "fault_all",
        "overburden_mmum_younger",
    }
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    times: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            region_id = row["region_id"]
            if region_id not in source_ids:
                continue
            step = int(row["step"])
            times[step] = float(row["time_years"])
            by_step[step][region_id] = float(row["total_co2_mass_kg"])

    steps = sorted(by_step)
    if len(steps) != 210:
        raise RuntimeError(f"Expected 210 containment steps in {path}; found {len(steps)}")

    result: dict[str, np.ndarray] = {
        "time_years": np.asarray([times[step] for step in steps], dtype=float)
    }
    result["storage_lm2"] = np.asarray(
        [by_step[step]["storage_lm2"] for step in steps], dtype=float
    ) / final_mass
    result["stratigraphy_sand"] = np.asarray(
        [by_step[step]["stratigraphy_sand"] for step in steps], dtype=float
    ) / final_mass
    result["clay_and_seal"] = np.asarray(
        [
            by_step[step]["stratigraphy_clay"]
            + by_step[step]["complete_top_seal_amphb"]
            for step in steps
        ],
        dtype=float,
    ) / final_mass
    result["fault_all"] = np.asarray(
        [by_step[step]["fault_all"] for step in steps], dtype=float
    ) / final_mass
    result["overburden_mmum_younger"] = np.asarray(
        [by_step[step]["overburden_mmum_younger"] for step in steps], dtype=float
    ) / final_mass
    return result


def configure_matplotlib() -> None:
    """Apply workflow-compatible Times-like typography and quiet axes."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    )


def make_axes() -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Create a fixed-size mini-plot with identical interior geometry."""

    # Native width is 27.5 mm, matching the TeX inclusion width. Fixed axes
    # coordinates keep the two derivatives pixel-for-pixel aligned.
    fig = plt.figure(figsize=(27.5 / 25.4, 19.2 / 25.4), facecolor="none")
    ax = fig.add_axes((0.30, 0.375, 0.58, 0.585), facecolor="none")
    ax.set_xscale("log")
    ax.set_xlim(PLOT_TIME_MIN_YEARS, PLOT_TIME_MAX_YEARS)
    ax.set_xticks((0.1, 10.0, 1000.0), (r"$0.1$", r"$10$", r"$10^3$"))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel(r"Time [yr]", labelpad=1.0)
    ax.axvline(
        INJECTION_END_YEARS,
        color="0.25",
        linestyle=(0, (1.5, 1.5)),
        linewidth=0.70,
        zorder=5,
    )
    ax.grid(axis="y", color="0.83", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def save_derivative(fig: mpl.figure.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    """Save matched vector and review-raster derivatives without tight cropping."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", transparent=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, transparent=True)
    plt.close(fig)


def plot_upward_migration(
    sources: dict[tuple[str, str], Path], output_dir: Path, dpi: int
) -> None:
    """Plot the three audited maximum-upward-reach histories."""

    fig, ax = make_axes()
    for case_name, color, linestyle in CASE_STYLES:
        data = read_domain_all(
            sources[(case_name, "regional_co2_inventory_steps.tsv")]
        )
        mask = data["time_years"] >= PLOT_TIME_MIN_YEARS
        ax.step(
            data["time_years"][mask],
            data["upward_reach_km"][mask],
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=1.15,
        )
    ax.set_ylim(0.0, 2.1)
    ax.set_yticks((0.0, 1.0, 2.0))
    ax.set_ylabel("Reach [km]", fontsize=6.8, labelpad=1.2)
    save_derivative(fig, output_dir, "panel3_uq_migration", dpi)


def plot_containment(
    sources: dict[tuple[str, str], Path], output_dir: Path, dpi: int
) -> None:
    """Plot the audited Case 5 compartment-wise containment history."""

    case_name = "Case 5"
    final_mass = read_final_injected_mass(
        sources[(case_name, "leakage_global_steps.tsv")]
    )
    data = read_containment(
        sources[(case_name, "regional_co2_inventory_steps.tsv")], final_mass
    )
    mask = data["time_years"] >= PLOT_TIME_MIN_YEARS

    fig, ax = make_axes()
    ax.stackplot(
        data["time_years"][mask],
        *[data[name][mask] for name, _ in COMPARTMENTS],
        colors=[color for _, color in COMPARTMENTS],
        alpha=0.98,
        linewidth=0.25,
        edgecolor="white",
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks((0.0, 0.5, 1.0), ("0", "50", "100"))
    ax.set_ylabel("Inventory [%]", fontsize=6.8, labelpad=1.2)
    save_derivative(fig, output_dir, "panel3_uq_containment", dpi)


def main() -> None:
    """Validate sources and write the two matched Panel 3 assets."""

    args = parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    configure_matplotlib()
    sources = required_source_files(manifest)
    plot_upward_migration(sources, args.output_dir.resolve(), args.dpi)
    plot_containment(sources, args.output_dir.resolve(), args.dpi)
    print(f"Wrote matched Panel 3 UQ assets to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
