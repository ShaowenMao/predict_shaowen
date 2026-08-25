#!/usr/bin/env python3
"""Render directional PREDICT permeability upscaling and its ensemble.

The upper row shows the three fixed-pressure single-phase experiments on one
canonical collapsed-cell-union W3 fault-core realization. The lower row shows
the three marginal distributions from all 2,000 linked permeability
vectors in the matching production ensemble.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
PNAS_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
DATA_DIR = HERE / "data"
DEFAULT_REPLAY = DATA_DIR / "predict_w3_case14_sample590_collapsed_replay_compact.mat"
DEFAULT_ENSEMBLE = DATA_DIR / "predict_w3_case14_ensemble_2000.mat"
DEFAULT_OUTPUT = PNAS_DIR / "figures" / "fig5_predict_directional_permeability_upscaling"
CACHE_DIR = HERE / "_cache"


SAND = "#D8B365"
CLAY = "#8C6D5A"
INK = "#252525"
GRID_EDGE = "#4A4541"
# Separate the flow cue from the distribution palette.
ARROW = "#174A73"
ARROW_INSIDE = "#7F9FB2"
ARROW_INSIDE_ALPHA = 0.58
DISTRIBUTION = "#276F73"
DISPLAY_SAMPLE_INDEX = 590
NORMAL_EXAGGERATION = 42.0
PANEL_ROTATION_ABOUT_Z_DEG = -60.0
RAW_SIZE = (1900, 2050)
EXPORT_DPI = 600
# Figure S5 is exported with a tight horizontal crop and then enlarged to
# 0.92\textwidth in the SI.  A 7.3-pt source font therefore renders at about
# 11 pt on the SI page, matching Figure S4's effective typography.
SOURCE_FONT_PT = 7.3
# Preserve the connector length while reducing both surrounding vertical gaps
# by one third. Panel (b) moves farther because the connector itself shifts up.
CONNECTOR_VERTICAL_SHIFT = (0.586 - 0.560) / 3.0
PANEL_B_VERTICAL_SHIFT = CONNECTOR_VERTICAL_SHIFT + (0.5335 - 0.5035) / 3.0

AXIS_INFO = {
    "x": {
        "title": r"Fault normal ($x$)",
        "component": r"$k_{xx}^{\mathrm{eff}}$",
    },
    "y": {
        "title": r"Strike parallel ($y$)",
        "component": r"$k_{yy}^{\mathrm{eff}}$",
    },
    "z": {
        "title": r"Dip parallel ($z$)",
        "component": r"$k_{zz}^{\mathrm{eff}}$",
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "font.size": SOURCE_FONT_PT,
            "axes.titlesize": SOURCE_FONT_PT,
            "axes.labelsize": SOURCE_FONT_PT,
            "xtick.labelsize": SOURCE_FONT_PT,
            "ytick.labelsize": SOURCE_FONT_PT,
            "legend.fontsize": SOURCE_FONT_PT,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def color_u8(color: str) -> np.ndarray:
    return np.rint(255.0 * np.asarray(to_rgb(color))).astype(np.uint8)


def provenance_path(path: Path) -> str:
    """Use a portable repository-relative path whenever possible."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_grid(path: Path, normal_exaggeration: float) -> tuple[pv.StructuredGrid, dict]:
    """Load the compact replay and construct the displayed structured grid."""
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    coords = np.asarray(data["coords"], dtype=float)
    dims = np.asarray(data["cartDims"], dtype=int).ravel()
    is_smear = np.asarray(data["isSmear"], dtype=np.uint8).ravel(order="F")

    if coords.shape[1] != 3 or coords.shape[0] != int(np.prod(dims + 1)):
        raise ValueError("Unexpected replay-grid geometry")
    if is_smear.size != int(np.prod(dims)):
        raise ValueError("Unexpected material-mask size")

    points = coords.copy()
    points[:, 0] *= normal_exaggeration
    # PREDICT local z is positive down dip; display it downward.
    points[:, 2] *= -1.0

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = dims + 1
    grid.cell_data["is_clay_smear"] = is_smear
    summary = {
        "dims": dims,
        "sand_fraction": float(np.mean(is_smear == 0)),
        "clay_fraction": float(np.mean(is_smear == 1)),
        "bounds_physical": np.vstack((coords.min(axis=0), coords.max(axis=0))),
    }
    return grid, summary


def rotate_fault_core_about_z(grid: pv.StructuredGrid) -> pv.StructuredGrid:
    """Rotate the vertical core to the accepted publication camera geometry."""
    grid.rotate_z(PANEL_ROTATION_ABOUT_Z_DEG, point=grid.center, inplace=True)
    return grid


def camera_for_right_dipping_material(grid: pv.StructuredGrid):
    """View the vertical core with the internal architecture dipping right."""
    center = np.asarray(grid.center, dtype=float)
    position = center + np.array([0.72, 1.15, 0.52]) * float(grid.length)
    return [tuple(position), tuple(center), (0.0, 0.0, 1.0)]


def grid_corner(grid: pv.StructuredGrid, i: int, j: int, k: int) -> np.ndarray:
    nx, ny, _ = (int(value) for value in grid.dimensions)
    return np.asarray(grid.points[i + nx * j + nx * ny * k], dtype=float)


def boundary_face(grid: pv.StructuredGrid, axis: str, upper: bool) -> pv.PolyData:
    nx, ny, nz = (int(value) for value in grid.dimensions)
    if axis == "x":
        i = nx - 1 if upper else 0
        ijk = ((i, 0, 0), (i, ny - 1, 0), (i, ny - 1, nz - 1), (i, 0, nz - 1))
    elif axis == "y":
        j = ny - 1 if upper else 0
        ijk = ((0, j, 0), (nx - 1, j, 0), (nx - 1, j, nz - 1), (0, j, nz - 1))
    elif axis == "z":
        k = nz - 1 if upper else 0
        ijk = ((0, 0, k), (nx - 1, 0, k), (nx - 1, ny - 1, k), (0, ny - 1, k))
    else:
        raise ValueError(axis)
    points = np.vstack([grid_corner(grid, *index) for index in ijk])
    return pv.PolyData(points, np.asarray([4, 0, 1, 2, 3]))


def direction_extents(
    grid: pv.StructuredGrid, directions: dict[str, np.ndarray]
) -> dict[str, tuple[float, float, float]]:
    extents = {}
    for key, direction in directions.items():
        values = np.asarray(grid.points) @ direction
        extents[key] = (float(values.min()), float(values.max()), float(np.ptp(values)))
    return extents


def directional_flow_arrow_geometry(
    inlet_face: pv.PolyData,
    outlet_face: pv.PolyData,
    reference_length: float,
) -> dict:
    """Return short parallel arrows entering and leaving opposing faces."""
    inlet_center = np.mean(np.asarray(inlet_face.points), axis=0)
    outlet_center = np.mean(np.asarray(outlet_face.points), axis=0)
    through_vector = outlet_center - inlet_center
    through_length = float(np.linalg.norm(through_vector))
    direction = through_vector / through_length

    exterior_length = 0.085 * reference_length

    def sample_face(face: pv.PolyData) -> list[np.ndarray]:
        points = np.asarray(face.points, dtype=float)
        center = np.mean(points, axis=0)
        edge_01 = points[1] - points[0]
        edge_03 = points[3] - points[0]
        long_edge = edge_01 if np.linalg.norm(edge_01) >= np.linalg.norm(edge_03) else edge_03
        return [center + fraction * long_edge for fraction in (-0.27, 0.0, 0.27)]

    inlet_points = sample_face(inlet_face)
    outlet_points = sample_face(outlet_face)
    arrows = []
    for point in inlet_points:
        arrows.append(
            {
                "face": "inlet",
                "start": (point - direction * exterior_length).tolist(),
                "end": point.tolist(),
            }
        )
    for point in outlet_points:
        arrows.append(
            {
                "face": "outlet",
                "start": point.tolist(),
                "end": (point + direction * exterior_length).tolist(),
            }
        )

    return {
        "inlet_face_center": inlet_center.tolist(),
        "outlet_face_center": outlet_center.tolist(),
        "arrows": arrows,
        "through_block_length": through_length,
        "exterior_length_each_side": exterior_length,
    }


def render_directional_core(replay: Path, axis: str, output: Path) -> dict:
    grid, summary = load_grid(replay, NORMAL_EXAGGERATION)
    rotate_fault_core_about_z(grid)

    is_clay = np.asarray(grid.cell_data["is_clay_smear"], dtype=bool)
    rgb = np.empty((grid.n_cells, 3), dtype=np.uint8)
    rgb[~is_clay] = color_u8(SAND)
    rgb[is_clay] = color_u8(CLAY)
    grid.cell_data["material_rgb"] = rgb
    surface = grid.extract_surface(algorithm="dataset_surface")
    outline = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    inlet = boundary_face(grid, axis, upper=False)
    outlet = boundary_face(grid, axis, upper=True)
    plotter = pv.Plotter(off_screen=True, window_size=RAW_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(
        surface,
        scalars="material_rgb",
        rgb=True,
        show_scalar_bar=False,
        show_edges=True,
        edge_color=GRID_EDGE,
        line_width=0.58,
        lighting=False,
    )
    plotter.add_mesh(outline, color=INK, line_width=2.8, lighting=False)
    arrow_summary = directional_flow_arrow_geometry(
        inlet,
        outlet,
        float(grid.length),
    )
    plotter.camera_position = camera_for_right_dipping_material(grid)
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.62)
    plotter.render()
    renderer = plotter.renderer
    render_width, render_height = (int(value) for value in renderer.GetSize())
    display_arrows = []
    for arrow in arrow_summary["arrows"]:
        projected = {"face": arrow["face"]}
        for endpoint in ("start", "end"):
            point = arrow[endpoint]
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            renderer.WorldToDisplay()
            display_x, display_y, _ = renderer.GetDisplayPoint()
            projected[endpoint] = [float(display_x), float(render_height - display_y)]
        display_arrows.append(projected)
    arrow_summary["display_arrows_px"] = display_arrows
    arrow_summary["render_size_px"] = [render_width, render_height]
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(output), auto_close=True)

    return {
        **summary,
        "axis": axis,
        "sand_cells": int(np.count_nonzero(~is_clay)),
        "clay_smear_cells": int(np.count_nonzero(is_clay)),
        "total_cells": int(grid.n_cells),
        "flow_arrow": arrow_summary,
    }


def load_joint_permeability(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle["perms"], dtype=float)
    if values.shape[0] == 3:
        values = values.T
    if values.shape != (2000, 3):
        raise ValueError(f"Expected a 2000 x 3 permeability matrix, got {values.shape}")
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise ValueError("Permeability ensemble contains invalid values")
    return values


def crop_rgba_with_arrows(
    path: Path,
    display_arrows_px: list[dict],
    pad: int = 34,
) -> tuple[Image.Image, list[dict], np.ndarray]:
    """Crop one raw render while retaining every projected arrow endpoint."""
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image)
    mask = (array[:, :, 3] > 0) & np.any(array[:, :, :3] < 250, axis=2)
    ys, xs = np.nonzero(mask)
    points = np.asarray(
        [arrow[endpoint] for arrow in display_arrows_px for endpoint in ("start", "end")],
        dtype=float,
    )
    left = max(0, int(np.floor(min(float(xs.min()), float(points[:, 0].min())))) - pad)
    top = max(0, int(np.floor(min(float(ys.min()), float(points[:, 1].min())))) - pad)
    right = min(
        image.width,
        int(np.ceil(max(float(xs.max()), float(points[:, 0].max())))) + pad + 1,
    )
    bottom = min(
        image.height,
        int(np.ceil(max(float(ys.max()), float(points[:, 1].max())))) + pad + 1,
    )
    cropped = image.crop((left, top, right, bottom))
    cropped_core_mask = mask[top:bottom, left:right]
    width = float(right - left)
    height = float(bottom - top)
    axes_arrows = []
    for arrow in display_arrows_px:
        converted = {"face": arrow["face"]}
        for endpoint in ("start", "end"):
            point = arrow[endpoint]
            converted[endpoint] = (
                (float(point[0]) - left) / width,
                1.0 - (float(point[1]) - top) / height,
            )
        axes_arrows.append(converted)
    image.close()
    return cropped, axes_arrows, cropped_core_mask


def draw_face_arrows(
    axis: plt.Axes,
    arrows: list[dict],
    core_mask: np.ndarray,
    axis_key: str,
    target_length_points: float = 18.75,
) -> None:
    """Draw face-aware arrows with one physical length for every camera angle.

    This function must be called after the figure has been drawn once.  The
    initial draw lets Matplotlib apply the image-aspect adjustment before the
    arrow endpoints are converted to display coordinates.  Opacity is assigned
    to each *physical boundary-face group*, not to pixel runs that happen to
    overlap the projected core.  The visually nearer face is opaque and the
    opposing face is uniformly faded.
    """
    target_length_px = target_length_points * axis.figure.dpi / 72.0
    to_display = axis.transAxes
    to_axes = to_display.inverted()
    prepared = []
    for arrow in arrows:
        start_px = np.asarray(to_display.transform(arrow["start"]), dtype=float)
        end_px = np.asarray(to_display.transform(arrow["end"]), dtype=float)
        delta_px = end_px - start_px
        norm_px = float(np.linalg.norm(delta_px))
        if norm_px <= np.finfo(float).eps:
            continue
        equal_delta_px = (delta_px / norm_px) * target_length_px
        if arrow["face"] == "inlet":
            # Preserve the face-contact point and extend outward.
            end_equal_px = end_px
            start_equal_px = end_px - equal_delta_px
        else:
            # Preserve the face-contact point and extend outward.
            start_equal_px = start_px
            end_equal_px = start_px + equal_delta_px
        start_equal = to_axes.transform(start_equal_px)
        end_equal = to_axes.transform(end_equal_px)
        contact = end_equal if arrow["face"] == "inlet" else start_equal
        prepared.append(
            {
                "face": arrow["face"],
                "start": start_equal,
                "end": end_equal,
                "contact": np.asarray(contact, dtype=float),
            }
        )

    face_contacts = {
        face: np.mean(
            [item["contact"] for item in prepared if item["face"] == face],
            axis=0,
        )
        for face in ("inlet", "outlet")
    }
    if axis_key == "x":
        # Fault-normal panel: the right boundary face is foreground.
        opaque_face = max(face_contacts, key=lambda face: face_contacts[face][0])
    elif axis_key == "y":
        # Strike-parallel panel: the left boundary face is foreground.
        opaque_face = min(face_contacts, key=lambda face: face_contacts[face][0])
    elif axis_key == "z":
        # Dip-parallel panel: the upper boundary face is foreground.
        opaque_face = max(face_contacts, key=lambda face: face_contacts[face][1])
    else:
        raise ValueError(f"Unknown arrow-panel axis: {axis_key}")

    for item in prepared:
        foreground = item["face"] == opaque_face
        start_display = np.asarray(to_display.transform(item["start"]), dtype=float)
        end_display = np.asarray(to_display.transform(item["end"]), dtype=float)
        arrow_direction = end_display - start_display
        arrow_direction /= float(np.linalg.norm(arrow_direction))
        # Matplotlib's ``-|>`` head length is 0.4 times the mutation scale.
        # Stop the separately rendered shaft at that base so it cannot project
        # through the filled triangular head.
        head_length_px = 0.40 * 10.8 * axis.figure.dpi / 72.0
        shaft_end = to_axes.transform(end_display - arrow_direction * head_length_px)
        if foreground:
            # The near-face arrows sit in front of the core and therefore stay
            # fully opaque along their complete projected length.
            axis.plot(
                [item["start"][0], shaft_end[0]],
                [item["start"][1], shaft_end[1]],
                transform=axis.transAxes,
                color=ARROW,
                alpha=1.0,
                linewidth=1.30,
                solid_capstyle="round",
                clip_on=False,
                zorder=8,
            )
            endpoint_hidden = False
        else:
            # For the far-face arrows, fade only the portion hidden behind the
            # projected core. Any shaft or arrowhead outside the block
            # silhouette remains fully opaque.
            samples = np.linspace(0.0, 1.0, 241)
            points = (
                np.asarray(item["start"])[None, :] * (1.0 - samples[:, None])
                + np.asarray(shaft_end)[None, :] * samples[:, None]
            )
            height, width = core_mask.shape
            columns = np.clip(
                np.rint(points[:, 0] * (width - 1)).astype(int), 0, width - 1
            )
            rows = np.clip(
                np.rint((1.0 - points[:, 1]) * (height - 1)).astype(int),
                0,
                height - 1,
            )
            overlaps_core = core_mask[rows, columns]
            run_starts = np.r_[
                0, np.flatnonzero(overlaps_core[1:] != overlaps_core[:-1]) + 1
            ]
            run_ends = np.r_[run_starts[1:], overlaps_core.size]
            for run_start, run_end in zip(run_starts, run_ends, strict=True):
                segment = points[run_start:run_end]
                if segment.shape[0] == 1:
                    segment = np.vstack((segment, segment))
                hidden = bool(overlaps_core[run_start])
                axis.plot(
                    segment[:, 0],
                    segment[:, 1],
                    transform=axis.transAxes,
                    color=ARROW_INSIDE if hidden else ARROW,
                    alpha=ARROW_INSIDE_ALPHA if hidden else 1.0,
                    linewidth=1.30,
                    solid_capstyle="round",
                    clip_on=False,
                    zorder=8,
                )
            endpoint_column = int(
                np.clip(np.rint(item["end"][0] * (width - 1)), 0, width - 1)
            )
            endpoint_row = int(
                np.clip(
                    np.rint((1.0 - item["end"][1]) * (height - 1)),
                    0,
                    height - 1,
                )
            )
            endpoint_hidden = bool(core_mask[endpoint_row, endpoint_column])

        # Render every head from the complete start-to-end vector. Using the
        # same path length and mutation scale makes the head geometry identical
        # for opaque and depth-cued arrows; only its tone follows the endpoint.
        head_color = ARROW_INSIDE if endpoint_hidden else ARROW
        head_alpha = ARROW_INSIDE_ALPHA if endpoint_hidden else 1.0
        axis.add_patch(
            FancyArrowPatch(
                item["start"],
                item["end"],
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=10.8,
                linewidth=0.0,
                facecolor=head_color,
                edgecolor=head_color,
                alpha=head_alpha,
                shrinkA=0.0,
                shrinkB=0.0,
                clip_on=False,
                zorder=9,
            )
        )


def add_marginal_distributions(
    figure: plt.Figure,
    logk: np.ndarray,
    bounds: tuple[float, float] = (-6.0, 2.0),
) -> None:
    panel_scale = 0.90
    panel_width = 0.16 * panel_scale
    previous_panel_height = 0.18 * panel_scale
    # Reduce h/w by 0.1 while keeping the current width fixed.
    panel_height = previous_panel_height - 0.10 * panel_width
    column_centers = (0.320, 0.500, 0.680)
    left_positions = tuple(center - panel_width / 2.0 for center in column_centers)
    # Scale each histogram proportionally about its previous center without
    # changing any typography or the surrounding annotations.
    previous_panel_center_y = 0.272 + 0.5 * 0.18 + 0.0265
    previous_panel_top = previous_panel_center_y + 0.5 * panel_height
    # Move the complete histogram group upward until its gap to the fixed
    # panel title is half of the preceding layout's gap.
    panel_center_y = (
        previous_panel_center_y
        + 0.5 * (0.5035 - previous_panel_top)
        + PANEL_B_VERTICAL_SHIFT
    )
    bottom = panel_center_y - 0.5 * panel_height
    labels = (
        r"$\log_{10}(k_{xx}^{\mathrm{eff}}\,[\mathrm{mD}])$",
        r"$\log_{10}(k_{yy}^{\mathrm{eff}}\,[\mathrm{mD}])$",
        r"$\log_{10}(k_{zz}^{\mathrm{eff}}\,[\mathrm{mD}])$",
    )
    bins = np.linspace(bounds[0], bounds[1], 33)
    histograms = [np.histogram(logk[:, index], bins=bins)[0] for index in range(3)]
    probabilities = [counts / logk.shape[0] for counts in histograms]
    y_max = 0.30

    for index, left in enumerate(left_positions):
        axis = figure.add_axes((left, bottom, panel_width, panel_height))
        widths = np.diff(bins)
        axis.bar(
            bins[:-1],
            probabilities[index],
            width=widths,
            align="edge",
            color=DISTRIBUTION,
            edgecolor=INK,
            linewidth=0.38,
            alpha=0.82,
        )
        axis.set_xlim(bounds)
        axis.set_ylim(0.0, y_max)
        axis.set_xticks((-6, -2, 2))
        axis.set_yticks((0.0, 0.1, 0.2, 0.3))
        axis.set_xlabel(labels[index], labelpad=2.2)
        axis.tick_params(direction="in", length=3.0, width=0.65, pad=2.2)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.set_yticklabels((r"$0$", r"$0.1$", r"$0.2$", r"$0.3$"))
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0)

    figure.text(
        0.204,
        bottom + 0.5 * panel_height,
        r"Probability",
        ha="center",
        va="center",
        rotation=90,
        fontsize=SOURCE_FONT_PT,
    )

    figure.text(
        0.5,
        0.5035 + PANEL_B_VERTICAL_SHIFT,
        r"Marginal permeability distributions with $2{,}000$ realizations",
        ha="center",
        va="bottom",
        fontsize=SOURCE_FONT_PT,
    )


def compose(
    raw_images: dict[str, Path],
    render_summaries: dict[str, dict],
    permeability: np.ndarray,
    output_stem: Path,
) -> tuple[Path, Path]:
    configure_style()
    figure = plt.figure(figsize=(7.25, 5.85), facecolor="white")
    # The image axes intentionally overlap their nominal boxes: each raw
    # rendering is tall and narrow, so this reduces the visible inter-panel
    # whitespace to approximately one third of the previous gap.
    x_positions = (0.155, 0.355, 0.545)
    image_axes = []
    pending_arrows = []
    for axis_key, x in zip(("x", "y", "z"), x_positions, strict=True):
        image, face_arrows, core_mask = crop_rgba_with_arrows(
            raw_images[axis_key],
            render_summaries[axis_key]["flow_arrow"]["display_arrows_px"],
        )
        axis = figure.add_axes((x, 0.645, 0.29, 0.285))
        axis.imshow(image)
        axis.set_axis_off()
        axis._source_image = image  # keep the PIL image alive through savefig
        image_axes.append(axis)
        pending_arrows.append((axis, face_arrows, core_mask, axis_key))

    figure.text(
        0.5,
        0.955,
        r"Flow-based upscaling in three directions",
        ha="center",
        va="top",
        fontsize=SOURCE_FONT_PT,
    )
    for center, axis_key in zip((0.300, 0.500, 0.690), ("x", "y", "z"), strict=True):
        figure.text(
            center,
            0.621,
            AXIS_INFO[axis_key]["title"],
            ha="center",
            va="center",
            fontsize=SOURCE_FONT_PT,
        )
    figure.text(
        0.5,
        0.586,
        r"One joint permeability vector "
        r"$(k^{\mathrm{eff}}_{xx},\,k^{\mathrm{eff}}_{yy},\,k^{\mathrm{eff}}_{zz})$ "
        r"per realization",
        ha="center",
        va="center",
    )
    figure.add_artist(
        FancyArrowPatch(
            (0.5, 0.560 + CONNECTOR_VERTICAL_SHIFT),
            (0.5, 0.5335 + CONNECTOR_VERTICAL_SHIFT),
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=0.65,
            color=INK,
        )
    )

    figure.text(
        0.204, 0.955, r"(a)", ha="center", va="top", fontsize=SOURCE_FONT_PT
    )
    figure.text(
        0.204,
        0.5035 + PANEL_B_VERTICAL_SHIFT,
        r"(b)",
        ha="center",
        va="bottom",
        fontsize=SOURCE_FONT_PT,
    )

    logk = np.log10(permeability)
    add_marginal_distributions(figure, logk)

    # Resolve the aspect-adjusted image boxes before fixing all directional
    # arrows to the same physical length.
    figure.canvas.draw()
    for axis, face_arrows, core_mask, axis_key in pending_arrows:
        draw_face_arrows(axis, face_arrows, core_mask, axis_key)

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=EXPORT_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    figure.savefig(
        pdf_path,
        dpi=EXPORT_DPI,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)
    for axis in image_axes:
        axis._source_image.close()
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = args.replay.resolve()
    ensemble_path = args.ensemble.resolve()
    output_stem = args.output_stem.resolve()
    if not replay.is_file():
        raise FileNotFoundError(replay)
    if not ensemble_path.is_file():
        raise FileNotFoundError(ensemble_path)

    raw_images = {}
    render_summaries = {}
    for axis in ("x", "y", "z"):
        raw = CACHE_DIR / f"{output_stem.name}_raw_{axis}.png"
        raw_metadata = raw.with_suffix(".json")
        raw_images[axis] = raw
        if not args.reuse_raw or not raw.is_file():
            render_summaries[axis] = render_directional_core(replay, axis, raw)
            raw_metadata.write_text(
                json.dumps(
                    render_summaries[axis],
                    indent=2,
                    default=lambda value: np.asarray(value).tolist(),
                ),
                encoding="utf-8",
            )
        elif raw_metadata.is_file():
            render_summaries[axis] = json.loads(raw_metadata.read_text(encoding="utf-8"))
        else:
            render_summaries[axis] = render_directional_core(replay, axis, raw)
            raw_metadata.write_text(
                json.dumps(
                    render_summaries[axis],
                    indent=2,
                    default=lambda value: np.asarray(value).tolist(),
                ),
                encoding="utf-8",
            )

    permeability = load_joint_permeability(ensemble_path)
    png_path, pdf_path = compose(raw_images, render_summaries, permeability, output_stem)
    metadata = {
        "source_replay": provenance_path(replay),
        "source_ensemble": provenance_path(ensemble_path),
        "ensemble_shape": list(permeability.shape),
        "component_order": ["kxx_fault_normal", "kyy_strike_parallel", "kzz_dip_parallel"],
        "permeability_unit": "mD",
        "displayed_sample_index_matlab_1based": DISPLAY_SAMPLE_INDEX,
        "displayed_permeability_mD": permeability[DISPLAY_SAMPLE_INDEX - 1].tolist(),
        "displayed_log10_permeability_mD": np.log10(permeability[DISPLAY_SAMPLE_INDEX - 1]).tolist(),
        "normal_display_exaggeration": NORMAL_EXAGGERATION,
        "boundary_conditions": {
            "inlet_pressure_bar": 1.0,
            "outlet_pressure_bar": 0.0,
            "other_faces": "no flow",
        },
        "render_summaries": render_summaries,
    }
    metadata_path = output_stem.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=lambda value: np.asarray(value).tolist()), encoding="utf-8")
    print(f"Loaded joint permeability vectors: {permeability.shape[0]}")
    print(f"Displayed sample {DISPLAY_SAMPLE_INDEX}: {permeability[DISPLAY_SAMPLE_INDEX - 1]}")
    print(png_path)
    print(pdf_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
