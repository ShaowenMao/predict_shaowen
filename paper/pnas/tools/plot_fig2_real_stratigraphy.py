#!/usr/bin/env python3
"""Render Fig. 2 on the real Step62 y-z stratigraphic geometry.

The Step62 active 2-D VTU supplies the exact Al/Ar triangles, layer IDs, dip,
and fault offset.  The six prescribed sand/clay patterns are mapped onto the
same 21 A-layer IDs, so geometry is held fixed and only interbed architecture
changes between panels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_fig2_thickness_scenarios import (
    CLAY_COLOR,
    COLUMN_TITLES,
    OUTLINE_COLOR,
    SAND_COLOR,
    SCENARIO_ORDER,
    build_top_seal_layers,
    configure_style,
    read_sand_percent,
    read_scenarios,
)


VIEW_Y_KM = (9.75, 15.30)
VIEW_Z_KM = (1.32, 2.02)
VERTICAL_EXAGGERATION = 4.0
FAULT_FILL_COLOR = "#F4F2ED"
FAULT_EDGE_COLOR = "#262626"
ROW_TITLES = ["Uniform", "Nonuniform"]


def default_grid_path(repo_root: Path) -> Path:
    github_root = repo_root.parent
    relative = Path(
        "setup_shaowen_resolution/grid_candidates/"
        "step_62_matched_upper_lower_transition/paraview/"
        "gom_step62_matched_upper_lower_transition_active_2d.vtu"
    )
    candidates = [
        github_root / "mrst_predict_sim_grid_integration" / relative,
        github_root / "mrst_predict_sim_grid_dev" / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-vtu",
        type=Path,
        default=default_grid_path(repo_root),
        help="Step62 active 2-D VTU containing a_layer_id and fault_unit_id.",
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    return parser.parse_args()


def load_step62_geometry(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    grid = pv.read(path)
    required = {"a_layer_id", "fault_unit_id"}
    missing = sorted(required.difference(grid.cell_data.keys()))
    if missing:
        raise ValueError(f"Step62 VTU is missing cell arrays: {missing}")
    if set(np.unique(grid.celltypes)) != {5}:
        raise ValueError("The Step62 cross section must contain triangles only")

    connectivity = np.asarray(grid.cells).reshape(-1, 4)
    if not np.all(connectivity[:, 0] == 3):
        raise ValueError("Unexpected non-triangular cell connectivity")
    triangles = connectivity[:, 1:]
    yz_km = np.asarray(grid.points)[:, 1:3] / 1000.0
    layer_ids = np.asarray(grid.cell_data["a_layer_id"], dtype=int)
    if set(np.unique(layer_ids[layer_ids > 0])) != set(range(1, 22)):
        raise ValueError("Expected Step62 A-layer IDs 1 through 21")

    fault_mask = np.asarray(grid.cell_data["fault_unit_id"], dtype=int) > 0

    edge_counts: Counter[tuple[int, int]] = Counter()
    for triangle in triangles[fault_mask]:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_counts[tuple(sorted((int(first), int(second))))] += 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    fault_boundary_segments = [yz_km[np.asarray(edge)] for edge in boundary_edges]

    return yz_km, triangles, layer_ids, fault_mask, fault_boundary_segments


def dissolve_triangles(
    points: np.ndarray, triangles: np.ndarray, cell_mask: np.ndarray
) -> list[np.ndarray]:
    """Return closed region outlines with all internal triangle edges removed.

    Drawing every grid triangle independently can create hairline gaps or a
    dotted texture in PDF viewers because adjacent vector polygons are
    antialiased separately.  Counting mesh edges and retaining only exterior
    edges produces one polygon per connected material region while preserving
    the exact Step62 boundary geometry.
    """
    selected = triangles[np.asarray(cell_mask, dtype=bool)]
    edge_counts: Counter[tuple[int, int]] = Counter()
    for triangle in selected:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_counts[tuple(sorted((int(first), int(second))))] += 1

    boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary_edges:
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)

    irregular = {vertex: len(neighbors) for vertex, neighbors in adjacency.items() if len(neighbors) != 2}
    if irregular:
        sample = list(irregular.items())[:8]
        raise ValueError(f"Material boundary is not a set of closed loops: {sample}")

    unused = set(boundary_edges)
    outlines: list[np.ndarray] = []
    while unused:
        start, current = min(unused)
        unused.remove((start, current))
        vertices = [start, current]

        while current != start:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if tuple(sorted((current, neighbor))) in unused
            ]
            if not candidates:
                raise ValueError("Encountered an open material boundary")
            following = candidates[0]
            unused.remove(tuple(sorted((current, following))))
            if following == start:
                break
            vertices.append(following)
            current = following

        outlines.append(points[np.asarray(vertices, dtype=int)])

    return outlines


def scenario_material_by_layer(
    scenarios: dict[str, dict[str, str]], label: str
) -> dict[int, str]:
    shallow_to_deep, _ = build_top_seal_layers(scenarios[label])
    deep_to_shallow = list(reversed(shallow_to_deep))
    return {layer_id: material for layer_id, material in enumerate(deep_to_shallow, 1)}


def render(
    grid_path: Path,
    scenarios: dict[str, dict[str, str]],
    sand_percent: dict[int, float],
    output_dir: Path,
) -> list[Path]:
    configure_style()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 8.6,
            "text.usetex": True,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.5,
            "svg.fonttype": "none",
        }
    )

    points, triangles, layer_ids, fault_mask, fault_segments = load_step62_geometry(
        grid_path
    )
    stratigraphy_mask = layer_ids > 0
    fault_polygons = dissolve_triangles(points, triangles, fault_mask)

    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.15), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.125,
        right=0.99,
        bottom=0.150,
        top=0.780,
        wspace=0.055,
        hspace=0.22,
    )

    letters = "abcdef"
    for scenario_index, axis in enumerate(axes.flat, start=1):
        material_by_layer = scenario_material_by_layer(
            scenarios, SCENARIO_ORDER[scenario_index - 1]
        )
        material_polygons = {
            material: dissolve_triangles(
                points,
                triangles,
                stratigraphy_mask
                & np.isin(
                    layer_ids,
                    [
                        layer_id
                        for layer_id, assigned in material_by_layer.items()
                        if assigned == material
                    ],
                ),
            )
            for material in ("C", "S")
        }

        axis.add_collection(
            PolyCollection(
                fault_polygons,
                facecolors=FAULT_FILL_COLOR,
                edgecolors="none",
                linewidths=0.0,
                antialiaseds=False,
                zorder=1.0,
            )
        )
        for material, color in (("C", CLAY_COLOR), ("S", SAND_COLOR)):
            axis.add_collection(
                PolyCollection(
                    material_polygons[material],
                    facecolors=color,
                    edgecolors="none",
                    linewidths=0.0,
                    antialiaseds=False,
                    zorder=2.0,
                )
            )
        axis.add_collection(
            LineCollection(
                fault_segments,
                colors=FAULT_EDGE_COLOR,
                linewidths=0.62,
                capstyle="round",
                joinstyle="round",
                antialiaseds=True,
                zorder=3.0,
            )
        )

        axis.set_xlim(*VIEW_Y_KM)
        axis.set_ylim(VIEW_Z_KM[1], VIEW_Z_KM[0])
        axis.set_aspect(VERTICAL_EXAGGERATION, adjustable="box")
        axis.set_facecolor("white")
        axis.grid(False)
        axis.tick_params(direction="in", length=2.8, width=0.65, pad=1.5)
        for spine in axis.spines.values():
            spine.set_color(OUTLINE_COLOR)
            spine.set_linewidth(0.75)

        axis.text(
            0.0,
            1.025,
            f"({letters[scenario_index - 1]})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=5.0,
        )
        axis.text(
            1.0,
            1.025,
            rf"{sand_percent[scenario_index]:.2f}\% sand",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="black",
            clip_on=False,
            zorder=5.0,
        )

        if scenario_index <= 3:
            axis.set_title(COLUMN_TITLES[scenario_index - 1], pad=25.0)

    for axis in axes[0, :]:
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for axis in axes[:, 1:].flat:
        axis.tick_params(axis="y", left=False, labelleft=False)

    horizontal_ticks = np.linspace(VIEW_Y_KM[0], VIEW_Y_KM[1], 4)
    horizontal_tick_labels = [f"{value:.2f}" for value in horizontal_ticks]
    for axis in axes[1, :]:
        axis.set_xticks(horizontal_ticks)
        labels = axis.set_xticklabels(horizontal_tick_labels)
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")
    vertical_ticks = np.linspace(VIEW_Z_KM[0], VIEW_Z_KM[1], 4)
    vertical_tick_labels = [f"{value:.2f}" for value in vertical_ticks]
    for axis in axes[:, 0]:
        axis.set_yticks(vertical_ticks)
        axis.set_yticklabels(vertical_tick_labels)

    figure.supylabel(r"Depth, $z$ (km)", x=0.062, fontsize=9.5)

    figure.legend(
        handles=[
            Patch(
                facecolor=SAND_COLOR,
                edgecolor=OUTLINE_COLOR,
                linewidth=0.5,
                label="Sand interbed",
            ),
            Patch(
                facecolor=CLAY_COLOR,
                edgecolor=OUTLINE_COLOR,
                linewidth=0.5,
                label="Clay interbed",
            ),
            Line2D(
                [0],
                [0],
                color=FAULT_EDGE_COLOR,
                linewidth=0.9,
                label="Main-fault edges",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.965),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )

    # Apply the fixed panel aspect before positioning shared labels. This
    # keeps row labels centered on the actual panel boxes and centers the
    # across-fault coordinate title under the middle column.
    figure.canvas.draw()
    row_centers = [
        float(np.mean([(axis.get_position().y0 + axis.get_position().y1) / 2.0 for axis in axes[row, :]]))
        for row in range(2)
    ]
    middle_position = axes[1, 1].get_position()
    middle_column_center = (middle_position.x0 + middle_position.x1) / 2.0
    figure.supxlabel(
        r"Across-fault distance, $y$ (km)",
        x=middle_column_center,
        y=0.065,
        fontsize=9.5,
    )

    figure.text(
        0.040,
        row_centers[0],
        ROW_TITLES[0],
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
    )
    figure.text(
        0.040,
        row_centers[1],
        ROW_TITLES[1],
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig2_top_seal_interbed_scenarios_real_geometry"
    outputs = [base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    figure.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[1], bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[2], bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    scenarios = read_scenarios(args.scenario_file)
    sand_percent = read_sand_percent(args.ratio_file)
    outputs = render(args.grid_vtu, scenarios, sand_percent, args.output_dir)
    print(f"GRID_VTU={args.grid_vtu.resolve()}")
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
