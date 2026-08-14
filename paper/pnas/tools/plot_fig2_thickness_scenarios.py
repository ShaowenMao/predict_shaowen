#!/usr/bin/env python3
"""Render Fig. 2: the six prescribed top-seal interbed architectures.

The lithology patterns are read from the authoritative scenario CSV.  The
fixed apparent-thickness vector is the footwall grid-layer definition used by
``gom_perm_varying_thickness_geology_cases_collapsed_cell_union.m``.  The
Lower Miocene 2 storage-reservoir layer in ``famp1`` is excluded, leaving the
21 Amph. B interbed layers used to calculate the reported sand proportions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


SAND_COLOR = "#D8B365"  # muted sandstone tan
CLAY_COLOR = "#8C6D5A"  # muted clay-rich earth brown
OUTLINE_COLOR = "#202020"


# Deep-to-shallow apparent thicknesses (m) for the fixed footwall layers.
# Source: getWindowOptions() in the authoritative collapsed-cell-union runner.
FOOTWALL_THICKNESS = {
    "famp1": [115.6143, 28.8949],
    "famp2": [36.9255, 35.8537, 36.8537, 36.3111],
    "famp3": [35.8537, 35.8537, 35.8537, 35.8537],
    "famp4": [35.8537, 35.8537, 35.8537, 35.9255],
    "famp5": [35.8537, 35.8537, 35.8537, 35.8537],
    "famp6": [28.2932, 33.1042, 33.1699, 33.1042],
}

SCENARIO_ORDER = [
    "scenario_01_low_sand_uniform",
    "scenario_02_medium_sand_uniform",
    "scenario_03_high_sand_uniform",
    "scenario_04_low_sand_nonuniform",
    "scenario_05_medium_sand_nonuniform",
    "scenario_06_high_sand_nonuniform",
]

COLUMN_TITLES = [
    "Low sand proportion",
    "Medium sand proportion",
    "High sand proportion",
]

ROW_TITLES = [
    "Uniform clay-interbed spacing",
    "Nonuniform clay-interbed spacing",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_output = Path(__file__).resolve().parents[1] / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=repo_root / "examples" / "thickness_scenario_designs.csv",
    )
    parser.add_argument(
        "--ratio-file",
        type=Path,
        default=repo_root
        / "examples"
        / "footwall_sand_ratio_by_thickness_scenario.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def read_scenarios(path: Path) -> dict[str, dict[str, str]]:
    scenarios: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            label = row["ScenarioLabel"]
            scenarios.setdefault(label, {})[row["Window"].lower()] = row[
                "FWPattern"
            ].upper()

    for label in SCENARIO_ORDER:
        if label not in scenarios:
            raise ValueError(f"Missing scenario {label!r} in {path}")
        missing = [f"famp{i}" for i in range(1, 7) if f"famp{i}" not in scenarios[label]]
        if missing:
            raise ValueError(f"Scenario {label!r} is missing windows {missing}")
    return scenarios


def read_sand_percent(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            values[int(row["Scenario"])] = float(row["SandPercent"])
    return values


def build_top_seal_layers(
    window_patterns: dict[str, str],
) -> tuple[list[str], list[float]]:
    """Return the 21 top-seal layers in shallow-to-deep order."""
    lithology_deep_to_shallow: list[str] = []
    thickness_deep_to_shallow: list[float] = []

    for index in range(1, 7):
        window = f"famp{index}"
        pattern = list(window_patterns[window])
        thickness = FOOTWALL_THICKNESS[window]
        if len(pattern) != len(thickness):
            raise ValueError(
                f"Pattern/thickness mismatch for {window}: "
                f"{len(pattern)} versus {len(thickness)}"
            )

        # The first famp1 layer is the LM2 storage reservoir, not Amph. B.
        start = 1 if window == "famp1" else 0
        lithology_deep_to_shallow.extend(pattern[start:])
        thickness_deep_to_shallow.extend(thickness[start:])

    if len(lithology_deep_to_shallow) != 21:
        raise AssertionError("Expected exactly 21 top-seal grid layers")

    return (
        list(reversed(lithology_deep_to_shallow)),
        list(reversed(thickness_deep_to_shallow)),
    )


def merge_adjacent_layers(
    lithology: list[str], thickness: list[float]
) -> list[tuple[str, float, float]]:
    """Merge adjacent layers of the same lithology for a clean schematic."""
    segments: list[tuple[str, float, float]] = []
    cursor = 0.0
    for material, dz in zip(lithology, thickness, strict=True):
        lower = cursor + dz
        if segments and segments[-1][0] == material:
            previous = segments[-1]
            segments[-1] = (material, previous[1], lower)
        else:
            segments.append((material, cursor, lower))
        cursor = lower
    return segments


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def render_figure(
    scenarios: dict[str, dict[str, str]],
    sand_percent: dict[int, float],
    output_dir: Path,
) -> list[Path]:
    configure_style()

    layer_sets = [build_top_seal_layers(scenarios[label]) for label in SCENARIO_ORDER]
    total_thickness = sum(layer_sets[0][1])
    if abs(total_thickness - 732.8266) > 1.0e-4:
        raise AssertionError(f"Unexpected total apparent thickness: {total_thickness}")

    for scenario_index, (lithology, thickness) in enumerate(layer_sets, start=1):
        calculated = 100.0 * sum(
            dz for material, dz in zip(lithology, thickness, strict=True) if material == "S"
        ) / total_thickness
        if abs(calculated - sand_percent[scenario_index]) > 0.011:
            raise AssertionError(
                f"Scenario {scenario_index}: calculated sand proportion "
                f"{calculated:.4f}% differs from table value "
                f"{sand_percent[scenario_index]:.4f}%"
            )

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 6.15), sharey=True)
    fig.subplots_adjust(
        left=0.145, right=0.985, bottom=0.105, top=0.825, wspace=0.075, hspace=0.12
    )

    letters = "abcdef"
    for scenario_index, ax in enumerate(axes.flat, start=1):
        lithology, thickness = layer_sets[scenario_index - 1]
        segments = merge_adjacent_layers(lithology, thickness)

        for material, upper, lower in segments:
            color = SAND_COLOR if material == "S" else CLAY_COLOR
            ax.add_patch(
                Rectangle(
                    (0.0, upper),
                    1.0,
                    lower - upper,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.65,
                )
            )

        ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                total_thickness,
                fill=False,
                edgecolor=OUTLINE_COLOR,
                linewidth=0.9,
                clip_on=False,
            )
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(total_thickness, 0.0)
        ax.set_xticks([])
        ax.grid(False)
        ax.tick_params(axis="y", direction="out", length=3.0, width=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(
            0.035,
            0.955,
            f"({letters[scenario_index - 1]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )
        ax.text(
            0.965,
            0.955,
            f"{sand_percent[scenario_index]:.2f}% sand",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )

        if scenario_index <= 3:
            ax.set_title(COLUMN_TITLES[scenario_index - 1], pad=7.0)

    tick_values = [0.0, 200.0, 400.0, 600.0, total_thickness]
    tick_labels = ["0", "200", "400", "600", "733"]
    for row in range(2):
        axes[row, 0].set_yticks(tick_values, tick_labels)
        axes[row, 0].set_ylabel(
            "Apparent thickness below top\nof interbedded interval (m)", labelpad=7.0
        )
    for row in range(2):
        for column in (1, 2):
            axes[row, column].tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle("Top-seal interbed architecture scenarios", y=0.972, fontsize=12.0)
    fig.legend(
        handles=[
            Patch(facecolor=SAND_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.6, label="Sand interbed"),
            Patch(facecolor=CLAY_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.6, label="Clay interbed"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.915),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=2.0,
    )

    fig.text(
        0.025,
        0.612,
        ROW_TITLES[0],
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.270,
        ROW_TITLES[1],
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig2_top_seal_interbed_scenarios"
    outputs = [base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    scenarios = read_scenarios(args.scenario_file)
    sand_percent = read_sand_percent(args.ratio_file)
    outputs = render_figure(scenarios, sand_percent, args.output_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
