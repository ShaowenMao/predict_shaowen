#!/usr/bin/env python3
"""Plot the five-case final qualification gate from compact CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import numpy as np


CASE_IDS = (5, 6, 8, 9, 10)
WINDOWS = tuple(f"famp{i}" for i in range(1, 7))
WINDOW_LABELS_REVERSED = tuple(f"W{i}" for i in range(6, 0, -1))
COMPONENTS = (
    ("log_kxx", r"$\log_{10}(k_{xx})$ [mD]"),
    ("log_kyy", r"$\log_{10}(k_{yy})$ [mD]"),
    ("log_kzz", r"$\log_{10}(k_{zz})$ [mD]"),
)
CASE_LABELS = {
    5: "C05\nFault-wide low\nState-wide",
    6: "C06\nFault-wide high\nState-wide",
    8: "C08\nGrouped high/low\nLocal",
    9: "C09\nGrouped low/high\nState-wide",
    10: "C10\nGrouped high/low\nState-wide",
}
STATE_COLORS = {
    "low": "#2D6CA8",
    "high": "#E58A25",
    "independent": "#B4B8BD",
}
INK = "#16243A"
GRID = "#D9DEE5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_data(data_dir: Path):
    sampling = {}
    pc = {}
    for case_id in CASE_IDS:
        sampling[case_id] = read_csv(
            data_dir / f"s06_c012_case{case_id:02d}_slice_window_values.csv"
        )
        pc[case_id] = read_csv(
            data_dir
            / f"pc_curve_summary_s06_c012_cases_{case_id:02d}_ip_full87.csv"
        )
        if len(sampling[case_id]) != 522 or len(pc[case_id]) != 522:
            raise ValueError(
                f"Case {case_id:02d} must contain 522 sampling and Pc rows"
            )
    return sampling, pc


def window_rows(rows, window: str, window_key: str):
    selected = [row for row in rows if row[window_key] == window]
    slice_key = "slice_index" if window_key == "window" else "SliceIndex"
    selected.sort(key=lambda row: int(row[slice_key]))
    if len(selected) != 87:
        raise ValueError(f"{window} must contain 87 slices")
    return selected


def sampling_field(rows, key: str) -> np.ndarray:
    values = []
    for window in reversed(WINDOWS):
        values.append(
            [float(row[key]) for row in window_rows(rows, window, "window")]
        )
    return np.asarray(values)


def pc_field(rows, key: str, scale: float = 1.0) -> np.ndarray:
    values = []
    for window in reversed(WINDOWS):
        values.append(
            [
                float(row[key]) * scale
                for row in window_rows(rows, window, "Window")
            ]
        )
    return np.asarray(values)


def state_for(rows, window: str) -> str:
    states = {row["assigned_state"] for row in window_rows(rows, window, "window")}
    if len(states) != 1:
        raise ValueError(f"{window} has mixed state labels: {states}")
    return states.pop()


def style_axis(ax):
    ax.tick_params(labelsize=12, colors=INK, direction="out")
    for spine in ax.spines.values():
        spine.set_color("#7F8790")
        spine.set_linewidth(0.8)


def save_figure(fig, output_dir: Path, stem: str):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def plot_permeability_fields(sampling, output_dir: Path):
    fig, axes = plt.subplots(
        len(CASE_IDS),
        len(COMPONENTS),
        figsize=(19.5, 12.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    image = None
    for row_index, case_id in enumerate(CASE_IDS):
        for column_index, (key, label) in enumerate(COMPONENTS):
            ax = axes[row_index, column_index]
            image = ax.imshow(
                sampling_field(sampling[case_id], key),
                cmap="cividis",
                vmin=-7,
                vmax=2,
                aspect="auto",
                interpolation="nearest",
            )
            for boundary in np.arange(0.5, 5.6, 1):
                ax.axhline(boundary, color="white", linewidth=0.6, alpha=0.70)
            ax.set_yticks(range(6), WINDOW_LABELS_REVERSED)
            ax.set_xticks(
                np.asarray([1, 15, 29, 43, 58, 72, 87]) - 1,
                ["1", "15", "29", "43", "58", "72", "87"],
            )
            if row_index == 0:
                ax.set_title(label, fontsize=18, weight="bold", color=INK, pad=8)
            if row_index == len(CASE_IDS) - 1:
                ax.set_xlabel("Along-strike slice", fontsize=13, color=INK)
            style_axis(ax)

    fig.suptitle(
        "Permeability fields for the five final-gate cases",
        fontsize=25,
        weight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.942,
        "Six throw windows × 87 slices; identical log scale in every panel",
        ha="center",
        fontsize=14,
        color="#4A5564",
    )
    fig.subplots_adjust(left=0.20, right=0.91, top=0.88, bottom=0.07, hspace=0.34)
    fig.canvas.draw()
    for row_index, case_id in enumerate(CASE_IDS):
        position = axes[row_index, 0].get_position()
        fig.text(
            0.105,
            0.5 * (position.y0 + position.y1),
            CASE_LABELS[case_id],
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color=INK,
        )
    colorbar_ax = fig.add_axes([0.93, 0.13, 0.014, 0.72])
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=[-7, -4, -1, 2])
    colorbar.set_label(r"$\log_{10}(k)$ [mD]", fontsize=14, color=INK)
    colorbar.ax.tick_params(labelsize=12, colors=INK)
    return save_figure(fig, output_dir, "01_five_case_permeability_fields")


def plot_entry_pressure_fields(pc, output_dir: Path):
    fig, axes = plt.subplots(
        len(CASE_IDS),
        1,
        figsize=(19.5, 10.8),
        sharex=True,
        sharey=True,
    )
    image = None
    for row_index, case_id in enumerate(CASE_IDS):
        ax = axes[row_index]
        pe_bar = pc_field(pc[case_id], "PercolationPcPa", scale=1.0e-5)
        image = ax.imshow(
            pe_bar,
            cmap="cividis",
            norm=LogNorm(vmin=0.03, vmax=12),
            aspect="auto",
            interpolation="nearest",
        )
        for boundary in np.arange(0.5, 5.6, 1):
            ax.axhline(boundary, color="white", linewidth=0.6, alpha=0.70)
        ax.set_yticks(range(6), WINDOW_LABELS_REVERSED)
        ax.set_xticks(
            np.asarray([1, 15, 29, 43, 58, 72, 87]) - 1,
            ["1", "15", "29", "43", "58", "72", "87"],
        )
        if row_index == len(CASE_IDS) - 1:
            ax.set_xlabel("Along-strike slice", fontsize=13, color=INK)
        style_axis(ax)

    fig.suptitle(
        "Upscaled entry capillary pressure for the five final-gate cases",
        fontsize=25,
        weight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.942,
        "Full invasion-percolation result for all 522 window-slice cells per case",
        ha="center",
        fontsize=14,
        color="#4A5564",
    )
    fig.subplots_adjust(left=0.20, right=0.91, top=0.88, bottom=0.08, hspace=0.34)
    fig.canvas.draw()
    for row_index, case_id in enumerate(CASE_IDS):
        position = axes[row_index].get_position()
        fig.text(
            0.105,
            0.5 * (position.y0 + position.y1),
            CASE_LABELS[case_id],
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color=INK,
        )
    colorbar_ax = fig.add_axes([0.93, 0.13, 0.014, 0.70])
    ticks = [0.03, 0.1, 0.3, 1, 3, 10]
    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=ticks)
    colorbar.ax.set_yticklabels([str(value) for value in ticks])
    colorbar.set_label(r"Entry pressure, $P_e$ [bar]", fontsize=14, color=INK)
    colorbar.ax.tick_params(labelsize=12, colors=INK)
    return save_figure(fig, output_dir, "02_five_case_entry_pressure_fields")


def plot_kzz_distributions(sampling, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), sharey=True)
    for window_index, window in enumerate(WINDOWS):
        ax = axes.flat[window_index]
        values = []
        colors = []
        for case_id in CASE_IDS:
            rows = window_rows(sampling[case_id], window, "window")
            values.append([float(row["log_kzz"]) for row in rows])
            colors.append(STATE_COLORS[state_for(sampling[case_id], window)])
        boxes = ax.boxplot(
            values,
            positions=np.arange(1, len(CASE_IDS) + 1),
            widths=0.62,
            whis=(5, 95),
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 2.0},
            boxprops={"edgecolor": "#2E3640", "linewidth": 1.0},
            whiskerprops={"color": "#56606B", "linewidth": 1.0},
            capprops={"color": "#56606B", "linewidth": 1.0},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.88)
        ax.set_title(f"W{window_index + 1}", fontsize=18, weight="bold", color=INK)
        ax.set_xticks(
            np.arange(1, len(CASE_IDS) + 1),
            [f"C{case_id:02d}" for case_id in CASE_IDS],
        )
        ax.set_ylim(-7, 2)
        ax.set_yticks([-7, -4, -1, 2])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Qualification case", fontsize=12, color=INK)
        if window_index % 3 == 0:
            ax.set_ylabel(r"$\log_{10}(k_{zz})$ [mD]", fontsize=13, color=INK)
        style_axis(ax)

    fig.suptitle(
        "Window-wise down-dip permeability distributions",
        fontsize=25,
        weight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.932,
        "Each box summarizes 87 slices; fill denotes the assigned state",
        ha="center",
        fontsize=14,
        color="#4A5564",
    )
    legend = [
        Patch(facecolor=STATE_COLORS["low"], edgecolor="#2E3640", label="Low"),
        Patch(facecolor=STATE_COLORS["high"], edgecolor="#2E3640", label="High"),
        Patch(
            facecolor=STATE_COLORS["independent"],
            edgecolor="#2E3640",
            label="Independent",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.08, right=0.98, hspace=0.36)
    return save_figure(fig, output_dir, "03_window_kzz_distributions_by_case")


def sample_standard_deviation(rows, key: str) -> float:
    return float(np.std([float(row[key]) for row in rows], ddof=1))


def plot_variability_ratio(sampling, pc, output_dir: Path):
    metrics = (
        ("log_kxx", r"$\log_{10}(k_{xx})$"),
        ("log_kyy", r"$\log_{10}(k_{yy})$"),
        ("log_kzz", r"$\log_{10}(k_{zz})$"),
        ("entry_pressure", r"$\log_{10}(P_e)$"),
    )
    windows = WINDOWS[1:]
    matrix = np.zeros((len(metrics), len(windows)))
    for row_index, (key, _) in enumerate(metrics):
        for column_index, window in enumerate(windows):
            if key == "entry_pressure":
                local = [
                    math.log10(float(row["PercolationPcPa"]) * 1.0e-5)
                    for row in window_rows(pc[8], window, "Window")
                ]
                wide = [
                    math.log10(float(row["PercolationPcPa"]) * 1.0e-5)
                    for row in window_rows(pc[10], window, "Window")
                ]
                numerator = float(np.std(local, ddof=1))
                denominator = float(np.std(wide, ddof=1))
            else:
                numerator = sample_standard_deviation(
                    window_rows(sampling[8], window, "window"), key
                )
                denominator = sample_standard_deviation(
                    window_rows(sampling[10], window, "window"), key
                )
            matrix[row_index, column_index] = numerator / denominator

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    image = ax.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=16,
                weight="bold",
                color="white" if value > 0.52 else INK,
            )
    ax.set_xticks(range(len(windows)), [f"W{i}" for i in range(2, 7)])
    ax.set_yticks(range(len(metrics)), [label for _, label in metrics])
    ax.tick_params(labelsize=14, colors=INK)
    ax.set_title(
        "Variability under local versus state-wide sampling",
        fontsize=23,
        weight="bold",
        color=INK,
        pad=42,
    )
    ax.text(
        0.5,
        1.035,
        "Standard-deviation ratio: Case 08 local pool / Case 10 state-wide pool",
        transform=ax.transAxes,
        ha="center",
        fontsize=13,
        color="#4A5564",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("SD ratio", fontsize=14, color=INK)
    colorbar.ax.tick_params(labelsize=12, colors=INK)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.18, right=0.92)
    return save_figure(fig, output_dir, "04_local_vs_statewide_variability")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling, pc = load_data(args.data_dir)
    files = []
    files.extend(plot_permeability_fields(sampling, args.output_dir))
    files.extend(plot_entry_pressure_fields(pc, args.output_dir))
    files.extend(plot_kzz_distributions(sampling, args.output_dir))
    files.extend(plot_variability_ratio(sampling, pc, args.output_dir))
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
