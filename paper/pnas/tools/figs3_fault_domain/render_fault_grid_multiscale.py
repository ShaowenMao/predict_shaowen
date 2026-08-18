"""Render a multiscale Step62 fault-domain/grid figure for SI development.

The figure separates three scales that cannot be shown legibly in one panel:
the complete fault-domain segmentation, a central 3-D enlargement of the
actual wedge grid, and the corresponding triangular y-z cross section.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


HERE = Path(__file__).resolve().parent
V1_SCRIPT = HERE / "render_fault_domain_base.py"
V2_SCRIPT = HERE / "render_fault_domain_overview.py"

DEFAULT_OUTPUT = HERE / "step62_fault_grid_multiscale_example.png"
DEFAULT_OVERVIEW = HERE / "panel_a_fault_overview.png"
DEFAULT_ZOOM = HERE / "panel_b_central_3d_grid.png"
DEFAULT_CROSS_SECTION = HERE / "panel_c_central_yz_cross_section.png"

OVERBURDEN_COLOR = "#C6B06E"
TOP_SEAL_COLOR = "#5F382D"
STORAGE_RESERVOIR_COLOR = "#D59B47"
PREDICT_COLOR = TOP_SEAL_COLOR
GRID_EDGE_COLOR = "#D8B58C"
WINDOW_BOUNDARY_COLOR = "#FFFFFF"
STRUCTURE_COLOR = "#173B5E"
CAMERA_VECTOR = np.asarray([1.50, -1.00, 0.00], dtype=float)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("step62_fault_base", V1_SCRIPT)
OVERVIEW = load_module("step62_fault_overview", V2_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=BASE.DEFAULT_GRID_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--zoom", type=Path, default=DEFAULT_ZOOM)
    parser.add_argument(
        "--cross-section", type=Path, default=DEFAULT_CROSS_SECTION
    )
    parser.add_argument("--x-min", type=float, default=22_225.0)
    parser.add_argument("--x-max", type=float, default=22_775.0)
    parser.add_argument("--vertical-exaggeration", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 8.0,
            "text.usetex": True,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_spatial_coordinate_axes(
    image_path: Path,
    *,
    x_bounds_km: tuple[float, float],
    z_bounds_km: tuple[float, float],
    font_size: float = 20.0,
) -> None:
    """Add publication-scale physical-coordinate guides to the 3-D view.

    With the established edge-on camera, global x and y project onto the same
    screen direction.  A three-axis projected ruler would therefore be
    geometrically misleading.  The figure instead places the along-strike
    ruler below the model and adds a physical-depth ruler in a narrow left
    gutter.  Depth labels remain physical values and are not multiplied by the
    displayed vertical exaggeration.
    """
    source = Image.open(image_path).convert("RGBA")
    width, height = source.size
    # Reserve enough room for 20-pt publication labels and a clear separation
    # between each axis title and its tick labels.
    left_margin = max(560, int(round(0.130 * width)))
    top_margin = max(110, int(round(0.080 * height)))
    bottom_margin = max(520, int(round(0.370 * height)))
    canvas_width = width + left_margin
    canvas_height = top_margin + height + bottom_margin
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
    canvas.alpha_composite(source, dest=(left_margin, top_margin))

    dpi = 600.0
    figure = plt.figure(
        figsize=(canvas_width / dpi, canvas_height / dpi),
        dpi=dpi,
        facecolor="none",
    )
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(canvas, interpolation="none", origin="upper")
    axis.set_xlim(-0.5, canvas_width - 0.5)
    axis.set_ylim(canvas_height - 0.5, -0.5)
    axis.set_axis_off()

    axis_color = "#202020"

    # The along-strike direction is the long, horizontal direction in this
    # edge-on orthographic view.  Match the ruler span to the visible lower
    # model edge rather than to the full raster bounds.
    alpha = np.asarray(source.getchannel("A"))
    alpha_rows, _ = np.nonzero(alpha > 0)
    if alpha_rows.size == 0:
        raise RuntimeError(f"Rendered overview is empty: {image_path}")
    model_bottom = int(alpha_rows.max())
    band_start = max(0, model_bottom - max(12, int(round(0.018 * height))))
    bottom_columns = np.flatnonzero(
        np.any(alpha[band_start : model_bottom + 1] > 0, axis=0)
    )
    if bottom_columns.size < 2:
        raise RuntimeError("Cannot locate the lower along-strike model edge")
    x_start = left_margin + float(bottom_columns.min())
    x_stop = left_margin + float(bottom_columns.max())
    x_ruler_y = top_margin + height + 10.0
    axis.plot(
        [x_start, x_stop],
        [x_ruler_y, x_ruler_y],
        color=axis_color,
        lw=0.58,
        zorder=20,
    )
    x_ticks = np.asarray((x_bounds_km[0], 22.5, x_bounds_km[1]))
    x_positions = x_start + (
        (x_ticks - x_bounds_km[0])
        / (x_bounds_km[1] - x_bounds_km[0])
        * (x_stop - x_start)
    )
    for index, (position, value) in enumerate(zip(x_positions, x_ticks)):
        axis.plot(
            [position, position],
            [x_ruler_y - 8.0, x_ruler_y],
            color=axis_color,
            lw=0.58,
            zorder=20,
        )
        horizontal_alignment = "center"
        if index == 0:
            horizontal_alignment = "left"
        elif index == x_ticks.size - 1:
            horizontal_alignment = "right"
        axis.text(
            position,
            x_ruler_y + 20.0,
            f"{value:g}",
            fontsize=font_size,
            color="black",
            ha=horizontal_alignment,
            va="top",
            zorder=21,
        )
    axis.text(
        0.5 * (x_start + x_stop),
        x_ruler_y + 290.0,
        r"Along-strike coordinate, $x$ (km)",
        fontsize=font_size,
        color="black",
        ha="center",
        va="top",
        zorder=21,
    )

    # Physical depth increases downward.  Four ticks communicate the range
    # without crowding the narrow publication gutter.
    z_min, z_max = map(float, z_bounds_km)
    z_ruler_x = left_margin - 25.0
    z_start = top_margin + 0.030 * height
    z_stop = top_margin + 0.970 * height
    axis.plot(
        [z_ruler_x, z_ruler_x],
        [z_start, z_stop],
        color=axis_color,
        lw=0.58,
        zorder=20,
    )
    z_ticks = np.asarray((0.0, 1.0, 2.0, z_max))
    for value in z_ticks:
        position = z_start + (value - z_min) / (z_max - z_min) * (
            z_stop - z_start
        )
        axis.plot(
            [z_ruler_x - 7.0, z_ruler_x + 7.0],
            [position, position],
            color=axis_color,
            lw=0.58,
            zorder=20,
        )
        label = f"{value:g}" if value < z_max else f"{value:.2f}"
        axis.text(
            z_ruler_x - 30.0,
            position,
            label,
            fontsize=font_size,
            color="black",
            ha="right",
            va="center",
            zorder=21,
        )
    axis.text(
        88.0,
        0.5 * (z_start + z_stop),
        r"Depth, $z$ (km)",
        fontsize=font_size,
        color="black",
        rotation=90.0,
        ha="center",
        va="center",
        zorder=21,
    )

    figure.savefig(
        image_path,
        dpi=dpi,
        transparent=True,
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(figure)
    source.close()
    BASE.transparent_crop(image_path, padding=18)


def render_overview(
    args: argparse.Namespace,
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
) -> None:
    if args.overview.is_file() and not args.force:
        return
    # Keep the established multipart-workflow camera and the SI Fig. 1(c)
    # geological palette.
    OVERVIEW.OVERBURDEN_COLOR = OVERBURDEN_COLOR
    OVERVIEW.TOP_SEAL_COLOR = TOP_SEAL_COLOR
    OVERVIEW.STORAGE_RESERVOIR_COLOR = STORAGE_RESERVOIR_COLOR
    OVERVIEW.STRUCTURE_COLOR = STRUCTURE_COLOR
    OVERVIEW.render(
        SimpleNamespace(
            grid_dir=args.grid_dir,
            output=args.overview,
            width=10_000,
            height=6_250,
            vertical_exaggeration=args.vertical_exaggeration,
            camera=(1.50, -1.00, 0.00),
            camera_zoom=1.12,
        )
    )
    fault_cells = np.unique(np.concatenate(tuple(groups.values())))
    fault_nodes = np.unique(triangles[fault_cells])
    fault_yz = yz[fault_nodes] / 1000.0
    add_spatial_coordinate_axes(
        args.overview,
        x_bounds_km=(0.0, 45.0),
        z_bounds_km=(float(fault_yz[:, 1].min()), float(fault_yz[:, 1].max())),
    )


def group_interface_edges(
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
) -> list[tuple[int, int]]:
    edge_groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    for group in BASE.PREDICT_GROUPS:
        for cell_id in groups[group]:
            triangle = triangles[int(cell_id)]
            for first, second in zip(triangle, np.roll(triangle, -1)):
                edge_groups[tuple(sorted((int(first), int(second))))].add(group)
    interfaces: list[tuple[int, int]] = []
    for edge, neighbors in edge_groups.items():
        if len(neighbors) == 2:
            low, high = sorted(neighbors)
            if high == low + 1:
                interfaces.append(edge)
    return interfaces


def render_central_grid(
    args: argparse.Namespace,
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
    fault: pv.UnstructuredGrid,
) -> None:
    if args.zoom.is_file() and not args.force:
        return
    centers = fault.cell_centers().points
    keep = (
        (fault.cell_data["predict_region"] == 1)
        & (centers[:, 0] >= args.x_min)
        & (centers[:, 0] <= args.x_max)
    )
    central = fault.extract_cells(np.flatnonzero(keep)).clean()
    surface = central.extract_surface(algorithm="dataset_surface").clean()
    outline = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        feature_angle=28.0,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    full_fault_cells = np.unique(np.concatenate(tuple(groups.values())))
    exterior = OVERVIEW.boundary_edges(triangles, full_fault_cells)
    interfaces = OVERVIEW.throw_window_interface_points(
        yz, triangles, groups, exterior
    )
    x_bounds = central.bounds[0:2]
    window_lines = OVERVIEW.throw_window_lines(
        interfaces,
        float(x_bounds[0]),
        float(x_bounds[1]),
        args.vertical_exaggeration,
    )
    line_offset = 5.0 * CAMERA_VECTOR / np.linalg.norm(CAMERA_VECTOR)
    window_lines.translate(line_offset, inplace=True)

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(5_000, 3_600),
        lighting="three lights",
    )
    plotter.set_background("white", top="white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        surface,
        color=PREDICT_COLOR,
        opacity=1.0,
        smooth_shading=False,
        show_edges=True,
        edge_color=GRID_EDGE_COLOR,
        line_width=1.00,
        lighting=False,
    )
    plotter.add_mesh(
        window_lines,
        color=WINDOW_BOUNDARY_COLOR,
        line_width=3.0,
        render_lines_as_tubes=True,
        lighting=False,
    )
    plotter.add_mesh(
        outline,
        color=STRUCTURE_COLOR,
        line_width=3.0,
        render_lines_as_tubes=True,
        lighting=False,
    )

    bounds = np.asarray(central.bounds, dtype=float)
    center = np.asarray(
        [np.mean(bounds[0:2]), np.mean(bounds[2:4]), np.mean(bounds[4:6])]
    )
    extent = np.asarray(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
    )
    distance = float(np.linalg.norm(extent))
    plotter.camera_position = [
        center + distance * CAMERA_VECTOR,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.06)
    plotter.show(auto_close=False)
    try:
        plotter.screenshot(
            str(args.zoom), transparent_background=True, return_img=False
        )
    finally:
        plotter.close()
    BASE.transparent_crop(args.zoom, padding=34)


def draw_cross_section(
    axis: plt.Axes,
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
) -> None:
    predict_ids = np.concatenate(tuple(groups[g] for g in BASE.PREDICT_GROUPS))
    polygons = yz[triangles[predict_ids]] / 1000.0
    mesh = PolyCollection(
        polygons,
        facecolors=PREDICT_COLOR,
        edgecolors=GRID_EDGE_COLOR,
        linewidths=0.28,
        antialiaseds=True,
    )
    axis.add_collection(mesh)

    predict_boundary = OVERVIEW.boundary_edges(triangles, predict_ids)
    boundary_segments = [yz[list(edge)] / 1000.0 for edge in predict_boundary]
    axis.add_collection(
        LineCollection(
            boundary_segments,
            colors=STRUCTURE_COLOR,
            linewidths=1.05,
            capstyle="round",
            joinstyle="round",
        )
    )
    interface_segments = [
        yz[list(edge)] / 1000.0
        for edge in group_interface_edges(triangles, groups)
    ]
    axis.add_collection(
        LineCollection(
            interface_segments,
            colors=WINDOW_BOUNDARY_COLOR,
            linewidths=0.95,
            capstyle="round",
            joinstyle="round",
        )
    )

    all_points = polygons.reshape(-1, 2)
    y_min, z_min = all_points.min(axis=0)
    y_max, z_max = all_points.max(axis=0)
    # Reserve extra room on the right for window labels placed outside the
    # fault ribbon.
    y_pad = 0.10 * (y_max - y_min)
    z_pad = 0.05 * (z_max - z_min)
    axis.set_xlim(y_min - y_pad, y_max + y_pad)
    axis.set_ylim(z_max + z_pad, z_min - z_pad)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Across-fault coordinate, $y$ (km)", labelpad=2.0)
    axis.set_ylabel(r"Depth, $z$ (km)", labelpad=2.0)
    axis.tick_params(direction="in", length=2.5, width=0.7, pad=2.0)
    axis.set_xticks(np.linspace(y_min, y_max, 4))
    axis.set_yticks(np.linspace(z_min, z_max, 4))
    axis.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)

    for index, group in enumerate(BASE.PREDICT_GROUPS, start=1):
        group_vertices = (
            yz[triangles[groups[group]]].reshape(-1, 2) / 1000.0
        )
        group_z_min = float(group_vertices[:, 1].min())
        group_z_max = float(group_vertices[:, 1].max())
        label_z = 0.5 * (group_z_min + group_z_max)
        depth_band = np.abs(group_vertices[:, 1] - label_z) <= max(
            0.18 * (group_z_max - group_z_min), 1.0e-6
        )
        ribbon_edge_y = float(group_vertices[depth_band, 0].max())
        axis.text(
            ribbon_edge_y + 0.012,
            label_z,
            rf"$\mathrm{{W}}{index}$",
            ha="left",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="black",
            zorder=5,
        )


def compose(
    args: argparse.Namespace,
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
) -> None:
    configure_matplotlib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overview_image = np.asarray(Image.open(args.overview).convert("RGBA"))
    zoom_image = np.asarray(Image.open(args.zoom).convert("RGBA"))

    figure = plt.figure(figsize=(6.85, 4.95), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.22,
        hspace=0.30,
        height_ratios=(0.92, 1.08),
    )
    axis_a = figure.add_subplot(grid[0, :])
    axis_b = figure.add_subplot(grid[1, 0])
    axis_c = figure.add_subplot(grid[1, 1])

    for axis, image in ((axis_a, overview_image), (axis_b, zoom_image)):
        axis.imshow(image)
        axis.set_axis_off()

    axis_a.set_title(
        r"Fault domain by host interval and fault segmentation",
        pad=1.5,
    )
    axis_b.set_title(r"Central 3-D grid enlargement", pad=2.0)
    draw_cross_section(axis_c, yz, triangles, groups)

    for label, axis in zip((r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}"), (axis_a, axis_b, axis_c)):
        axis.text(
            -0.055,
            1.025,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="black",
            clip_on=False,
        )

    legend_handles = [
        Patch(facecolor=STORAGE_RESERVOIR_COLOR, edgecolor="none", label="Storage reservoir"),
        Patch(facecolor=TOP_SEAL_COLOR, edgecolor="none", label="Top seal (W1--W6)"),
        Patch(facecolor=OVERBURDEN_COLOR, edgecolor="none", label="Overburden"),
        Line2D([0], [0], color=GRID_EDGE_COLOR, lw=0.8, label="Cell edges"),
        Patch(
            facecolor=TOP_SEAL_COLOR,
            edgecolor=WINDOW_BOUNDARY_COLOR,
            linewidth=1.4,
            label="Window boundaries",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        handlelength=1.7,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    figure.savefig(
        args.output,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


def export_cross_section(
    args: argparse.Namespace,
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
) -> None:
    """Export panel (c) independently at its final publication scale."""
    configure_matplotlib()
    args.cross_section.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.40, 3.20), facecolor="white")
    draw_cross_section(axis, yz, triangles, groups)
    figure.savefig(
        args.cross_section,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    # Configure typography before the overview coordinate overlay is drawn;
    # otherwise that first panel falls back to Matplotlib's default sans font.
    configure_matplotlib()
    if not (0.0 <= args.x_min < args.x_max <= 45_000.0):
        raise ValueError("central enlargement must lie inside x=0--45 km")
    if args.vertical_exaggeration <= 0.0:
        raise ValueError("vertical exaggeration must be positive")
    for path in (args.output, args.overview, args.zoom, args.cross_section):
        path.parent.mkdir(parents=True, exist_ok=True)

    yz, triangles, groups = BASE.load_step62_fault(args.grid_dir)
    x_planes = np.r_[0.0, np.cumsum(BASE.along_strike_widths())]
    fault = BASE.build_extruded_fault(
        yz, triangles, groups, x_planes, args.vertical_exaggeration
    )
    render_overview(args, yz, triangles, groups)
    render_central_grid(args, yz, triangles, groups, fault)
    export_cross_section(args, yz, triangles, groups)
    compose(args, yz, triangles, groups)

    print(f"Full 3-D fault cells: {fault.n_cells:,}")
    print(f"Central enlargement: x={args.x_min/1000:g}--{args.x_max/1000:g} km")
    print(args.cross_section)
    print(args.output)


if __name__ == "__main__":
    main()
