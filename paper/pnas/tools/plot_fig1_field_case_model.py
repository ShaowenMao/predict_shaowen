#!/usr/bin/env python3
"""Compose the offshore-Texas field setting and Step62 model figure.

Panels (a) and (b) are cropped from the source artwork supplied with
Salo-Salgado et al. (2025).  Panel (a) intentionally places the regional map
before the regional NW-SE section.  Panel (c) reuses the established workflow
reservoir/fault rendering, includes both structural fault surfaces, and adds a
grid-free Step62 y-z geology inset using the representative medium-sand,
uniform interbed architecture.
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
from matplotlib.colors import to_rgba
from matplotlib.transforms import Bbox
from PIL import Image


OVERBURDEN_COLOR = "#C6B06E"
TOP_SEAL_COLOR = "#5F382D"
STORAGE_RESERVOIR_COLOR = "#D59B47"
UNDERBURDEN_COLOR = "#A8B3BF"
PREDICT_FAULT_COLOR = "#2C8C99"
OTHER_FAULT_COLOR = "#173B5E"
SAND_INTERBED_COLOR = "#E4C56A"
CLAY_INTERBED_COLOR = "#8A5744"
FAULT_CROSS_SECTION_COLOR = "#63BFB5"
FAULT_CROSS_SECTION_EDGE = "#1F5B63"

SOURCE_MAP_BOX = (1548, 78, 2126, 622)
SOURCE_REGIONAL_SECTION_BOX = (0, 0, 1546, 654)
SOURCE_FIELD_SECTION_BOX = (0, 666, 2128, 1629)
GRID_VIEW_Y_KM = (9.5, 15.6)
GRID_VIEW_Z_KM = (0.0, 3.35)
GEOLOGY_VERTICAL_EXAGGERATION = 1.20
RIGHT_GEOLOGY_PANEL_SCALE = 0.90
RIGHT_GEOLOGY_PANEL_Y_SHIFT = 0.015
INJECTOR_Y_KM = 12.816
INJECTOR_Z_KM = 2.012
INJECTOR_COLOR = "#D83B72"
STRIKE_ROTATION_DEG = -16.5
CROSS_FAULT_ROTATION_DEG = 21.5
X_TRIAD_ROTATION_DEG = STRIKE_ROTATION_DEG + 180.0
FAULT_CALLOUT_Z_KM = 0.85
FAULT_RIGHT_Y_AT_CALLOUT_KM = 11.907
CALLOUT_GAP_PT = 3.5
BOTTOM_CANVAS_CROP_IN = 0.45
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
        "--source-image",
        type=Path,
        default=repo_root
        / "paper"
        / "pnas"
        / "figures"
        / "source"
        / "salo_salgado_2025_field_case_source.jpeg",
    )
    parser.add_argument(
        "--model-image",
        type=Path,
        default=repo_root
        / "paper"
        / "pnas"
        / "figures"
        / "source"
        / "step62_two_faults_full_domain_unannotated.png",
    )
    parser.add_argument(
        "--grid-vtu", type=Path, default=default_grid_path(repo_root)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "paper" / "pnas" / "figures",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 7.5,
            "text.usetex": True,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.linewidth": 0.6,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return np.asarray(array[top:bottom, left:right]).copy()


def remove_embedded_panel_letters(
    regional_section: np.ndarray, field_section: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the baked-in source letters so one consistent label set is used."""
    # The regional-section letter sits within a laterally repetitive grey fill.
    # Replace its small patch with a mirrored neighboring patch of the same unit.
    regional_section[585:638, 70:118] = regional_section[
        585:638, 118:166
    ][:, ::-1]

    # The field-section letter is isolated in a white area above the lower frame.
    field_section[850:925, 120:170, :3] = 255
    if field_section.shape[2] == 4:
        field_section[850:925, 120:170, 3] = 255
    return regional_section, field_section


def load_source_panels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = np.asarray(plt.imread(path))
    regional_map = crop(source, SOURCE_MAP_BOX)
    regional_section = crop(source, SOURCE_REGIONAL_SECTION_BOX)
    field_section = crop(source, SOURCE_FIELD_SECTION_BOX)
    regional_section, field_section = remove_embedded_panel_letters(
        regional_section, field_section
    )
    return regional_map, regional_section, field_section


def compose_panel_a(
    regional_map: np.ndarray, regional_section: np.ndarray
) -> np.ndarray:
    """Place the two source frames at exactly the same height and baseline."""
    # Measured black-frame bounds in the supplied raster crops.
    map_frame = regional_map[42:527, 20:575]
    section_frame_top = 87
    section_frame_bottom = 644
    target_height = section_frame_bottom - section_frame_top
    target_width = round(map_frame.shape[1] * target_height / map_frame.shape[0])
    map_scaled = np.asarray(
        Image.fromarray(map_frame).resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
    )

    gap = 20
    canvas = np.full(
        (
            regional_section.shape[0],
            target_width + gap + regional_section.shape[1],
            regional_section.shape[2],
        ),
        255,
        dtype=regional_section.dtype,
    )
    canvas[
        section_frame_top:section_frame_bottom, :target_width
    ] = map_scaled
    canvas[:, target_width + gap :] = regional_section
    return canvas


def load_model_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(plt.imread(path))
    if image.ndim == 3 and image.shape[2] == 4:
        y_index, x_index = np.nonzero(image[:, :, 3] > 0.01)
        if x_index.size:
            padding = 10
            x0 = max(0, int(x_index.min()) - padding)
            x1 = min(image.shape[1], int(x_index.max()) + padding + 1)
            y0 = max(0, int(y_index.min()) - padding)
            y1 = min(image.shape[0], int(y_index.max()) + padding + 1)
            image = image[y0:y1, x0:x1]
    return image.copy()


def locate_model_injector(image: np.ndarray) -> tuple[float, float]:
    """Locate the renderer-registered injector marker in axes coordinates."""
    rgb = np.asarray(image[:, :, :3], dtype=float)
    target = np.asarray(mpl.colors.to_rgb(INJECTOR_COLOR), dtype=float)
    color_distance = np.linalg.norm(rgb - target[None, None, :], axis=2)
    marker = color_distance < 0.12
    if image.shape[2] == 4:
        marker &= image[:, :, 3] > 0.5
    y_index, x_index = np.nonzero(marker)
    if x_index.size == 0:
        raise ValueError(
            "The 3-D model image does not contain the registered injector marker"
        )
    x_axes = float(np.mean(x_index) / max(1, image.shape[1] - 1))
    y_axes = float(1.0 - np.mean(y_index) / max(1, image.shape[0] - 1))
    return x_axes, y_axes


def load_step62_mesh(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    grid = pv.read(path)
    required = {"a_layer_id", "fault_unit_id", "region_name"}
    missing = sorted(required.difference(grid.cell_data.keys()))
    if missing:
        raise ValueError(f"Step62 VTU is missing cell arrays: {missing}")
    if grid.n_cells != 24_886:
        raise ValueError(
            "Expected 24,886 triangles in the Step62 visualization mesh, "
            f"found {grid.n_cells:,}"
        )
    if set(np.unique(grid.celltypes)) != {5}:
        raise ValueError("The Step62 cross section must contain triangles only")

    connectivity = np.asarray(grid.cells).reshape(-1, 4)
    if not np.all(connectivity[:, 0] == 3):
        raise ValueError("Unexpected non-triangular cell connectivity")
    triangles = connectivity[:, 1:]
    yz_km = np.asarray(grid.points)[:, 1:3] / 1000.0
    layer_ids = np.asarray(grid.cell_data["a_layer_id"], dtype=int)
    fault_ids = np.asarray(grid.cell_data["fault_unit_id"], dtype=int)
    region_names = np.asarray(grid.cell_data["region_name"]).astype(str)
    return yz_km, triangles, layer_ids, fault_ids, region_names


def figure_panel_label(
    figure: mpl.figure.Figure, label: str, x: float, y: float
) -> None:
    figure.text(
        x,
        y,
        rf"\textbf{{({label})}}",
        transform=figure.transFigure,
        ha="left",
        va="top",
        fontsize=9.0,
        color="black",
        bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.9},
        zorder=20,
    )


def show_image(axis: mpl.axes.Axes, image: np.ndarray, add_frame: bool = False) -> None:
    axis.imshow(image, interpolation="lanczos")
    axis.set_xticks([])
    axis.set_yticks([])
    if add_frame:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)
            spine.set_color("#252525")
    else:
        axis.set_axis_off()


def plot_injector_symbol(
    axis: mpl.axes.Axes,
    x: float,
    y: float,
    transform: mpl.transforms.Transform,
    *,
    outer_size: float = 28,
    inner_size: float = 8,
) -> None:
    """Draw a compact injection-well bullseye at the exact well location."""
    axis.scatter(
        [x],
        [y],
        s=outer_size,
        marker="o",
        facecolor="white",
        edgecolor="#202020",
        linewidth=0.48,
        transform=transform,
        clip_on=False,
        zorder=31,
    )
    axis.scatter(
        [x],
        [y],
        s=inner_size,
        marker="o",
        facecolor=INJECTOR_COLOR,
        edgecolor="none",
        transform=transform,
        clip_on=False,
        zorder=32,
    )


def axes_arrow(
    axis: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    arrowstyle: str = "-|>",
    linewidth: float = 0.62,
    mutation_scale: float = 6.0,
    color: str = "#202020",
) -> None:
    """Draw a compact publication-weight arrow in axes coordinates."""
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=axis.transAxes,
        textcoords=axis.transAxes,
        arrowprops={
            "arrowstyle": arrowstyle,
            "color": color,
            "lw": linewidth,
            "mutation_scale": mutation_scale,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        annotation_clip=False,
        zorder=28,
    )


def offset_projected_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    axes_aspect: float,
    distance: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Offset a projected edge outward by a fixed screen-space distance.

    ``distance`` is expressed relative to the axes height.  Accounting for
    the axes aspect ratio gives the same visible edge-to-arrow gap for the two
    oblique 45-km edges and the vertical 8-km edge.
    """
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta_screen = np.asarray(
        [
            (end_array[0] - start_array[0]) * axes_aspect,
            end_array[1] - start_array[1],
        ]
    )
    length = float(np.linalg.norm(delta_screen))
    if length == 0.0:
        raise ValueError("Cannot offset a zero-length projected edge")
    outward_screen = np.asarray(
        [delta_screen[1], -delta_screen[0]]
    ) / length
    offset_axes = np.asarray(
        [
            distance * outward_screen[0] / axes_aspect,
            distance * outward_screen[1],
        ]
    )
    return tuple(start_array + offset_axes), tuple(end_array + offset_axes)


def projected_segment_angle(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    axes_aspect: float,
) -> float:
    """Return a projected edge angle in display coordinates."""
    return float(
        np.degrees(
            np.arctan2(
                end[1] - start[1],
                (end[0] - start[0]) * axes_aspect,
            )
        )
    )


def shift_text_in_display(
    artist: mpl.text.Text,
    *,
    delta_x: float = 0.0,
    delta_y: float = 0.0,
) -> None:
    """Shift a text artist by a requested number of display pixels."""
    transform = artist.get_transform()
    display_position = transform.transform(artist.get_position())
    shifted = display_position + np.asarray([delta_x, delta_y], dtype=float)
    artist.set_position(transform.inverted().transform(shifted))


def annotate_model(
    axis: mpl.axes.Axes, injector_axes: tuple[float, float]
) -> None:
    """Annotate the 3-D Step62 model without obscuring its geology."""
    direct = {
        "fontsize": 6.1,
        "color": "black",
        "ha": "center",
        "va": "center",
        "transform": axis.transAxes,
        "zorder": 29,
    }
    left_aligned = dict(direct)
    left_aligned["ha"] = "left"
    top_seal_style = dict(direct)
    top_seal_style["color"] = "white"

    # Direct geological labels follow the visible near-side layering.  Their
    # shallow rotation matches the projected x direction and avoids leaders.
    axis.text(
        0.025,
        0.550,
        "Overburden",
        rotation=STRIKE_ROTATION_DEG,
        **left_aligned,
    )
    axis.text(
        0.345,
        0.380,
        "Top seal",
        rotation=STRIKE_ROTATION_DEG,
        **top_seal_style,
    )
    axis.text(
        0.025,
        0.460,
        "Storage reservoir",
        rotation=STRIKE_ROTATION_DEG,
        **left_aligned,
    )
    axis.text(
        0.025,
        0.370,
        "Underburden",
        rotation=STRIKE_ROTATION_DEG,
        **left_aligned,
    )

    # Both structural surfaces are broad enough for direct labels.
    axis.text(
        0.435,
        0.575,
        "Main fault",
        rotation=STRIKE_ROTATION_DEG,
        **direct,
    )
    axis.text(
        0.555,
        0.700,
        "Secondary fault",
        rotation=STRIKE_ROTATION_DEG,
        **direct,
    )

    injector_x, injector_y = injector_axes
    axis.text(
        injector_x - 0.005,
        injector_y - 0.010,
        "Injector",
        fontsize=6.1,
        color="black",
        ha="center",
        va="top",
        rotation=STRIKE_ROTATION_DEG,
        transform=axis.transAxes,
        zorder=29,
    )

    # Full-domain dimensions.  All three arrows use one screen-space offset,
    # so their visible gaps from the corresponding model edges are identical.
    image_shape = np.asarray(axis.images[0].get_array()).shape
    axes_aspect = float(image_shape[1] / image_shape[0])
    projected_edges = [
        ((0.007, 0.309), (0.535, 0.012), r"45 km"),
        ((0.535, 0.012), (0.994, 0.357), r"45 km"),
        ((0.994, 0.357), (0.994, 0.687), r"8 km"),
    ]
    arrow_gap = 0.018
    label_gap = 0.047
    for edge_start, edge_end, label in projected_edges:
        arrow_start, arrow_end = offset_projected_segment(
            edge_start,
            edge_end,
            axes_aspect=axes_aspect,
            distance=arrow_gap,
        )
        label_start, label_end = offset_projected_segment(
            edge_start,
            edge_end,
            axes_aspect=axes_aspect,
            distance=label_gap,
        )
        axes_arrow(
            axis,
            arrow_start,
            arrow_end,
            arrowstyle="<|-|>",
            linewidth=0.56,
            mutation_scale=5.4,
        )
        label_position = 0.5 * (
            np.asarray(label_start) + np.asarray(label_end)
        )
        axis.text(
            float(label_position[0]),
            float(label_position[1]),
            label,
            rotation=projected_segment_angle(
                edge_start,
                edge_end,
                axes_aspect=axes_aspect,
            ),
            fontsize=5.8,
            ha="center",
            va="center",
            transform=axis.transAxes,
            clip_on=False,
            zorder=29,
        )

    # Native projected coordinate triad for the 3-D renderer: x is along
    # strike, y is across fault, and positive z points downward.  The three
    # arrows have identical lengths in display space.
    triad_origin = (0.055, 0.105)
    triad_length = 0.075
    triad_label_length = 0.089

    def triad_endpoint(angle_degrees: float, length: float) -> tuple[float, float]:
        angle_radians = np.radians(angle_degrees)
        return (
            triad_origin[0]
            + length * np.cos(angle_radians) / axes_aspect,
            triad_origin[1] + length * np.sin(angle_radians),
        )

    x_endpoint = triad_endpoint(X_TRIAD_ROTATION_DEG, triad_length)
    y_endpoint = triad_endpoint(CROSS_FAULT_ROTATION_DEG, triad_length)
    z_endpoint = triad_endpoint(-90.0, triad_length)
    axes_arrow(axis, triad_origin, x_endpoint, mutation_scale=5.8)
    axes_arrow(axis, triad_origin, y_endpoint, mutation_scale=5.8)
    axes_arrow(axis, triad_origin, z_endpoint, mutation_scale=5.8)
    axis.scatter(
        [triad_origin[0]],
        [triad_origin[1]],
        s=2.5,
        color="black",
        transform=axis.transAxes,
        clip_on=False,
        zorder=30,
    )
    x_label = triad_endpoint(X_TRIAD_ROTATION_DEG, triad_label_length)
    y_label = triad_endpoint(CROSS_FAULT_ROTATION_DEG, triad_label_length)
    z_label = triad_endpoint(-90.0, triad_label_length)
    axis.text(
        *x_label,
        r"$x$",
        fontsize=6.2,
        ha="right",
        va="center",
        transform=axis.transAxes,
        zorder=29,
    )
    axis.text(
        *y_label,
        r"$y$",
        fontsize=6.2,
        ha="left",
        va="bottom",
        transform=axis.transAxes,
        zorder=29,
    )
    axis.text(
        *z_label,
        r"$z$",
        fontsize=6.2,
        ha="center",
        va="top",
        transform=axis.transAxes,
        zorder=29,
    )


def annotate_geology_inset(axis: mpl.axes.Axes) -> None:
    """Add sparse labels with balanced callout clearances."""
    label_style = {
        "fontsize": 7.8,
        "color": "black",
        "ha": "left",
        "va": "center",
        "zorder": 12,
    }
    axis.text(10.05, 0.85, "Overburden", **label_style)
    axis.text(10.20, 1.60, "Top seal", **label_style)
    axis.text(9.82, 2.48, "Storage reservoir", **label_style)

    figure = axis.figure
    gap_pixels = CALLOUT_GAP_PT * figure.dpi / 72.0

    # Place the LaTeX horizontal arrow between the actual right-hand fault
    # boundary and its label, with the same visible gap at both ends.
    fault_arrow = axis.text(
        12.00,
        FAULT_CALLOUT_Z_KM,
        r"$\longleftarrow$",
        fontsize=9.0,
        color="black",
        ha="left",
        va="center",
        zorder=13,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    fault_display_x = axis.transData.transform(
        (FAULT_RIGHT_Y_AT_CALLOUT_KM, FAULT_CALLOUT_Z_KM)
    )[0]
    arrow_bounds = fault_arrow.get_window_extent(renderer=renderer)
    shift_text_in_display(
        fault_arrow,
        delta_x=fault_display_x + gap_pixels - arrow_bounds.x0,
    )
    figure.canvas.draw()
    arrow_bounds = fault_arrow.get_window_extent(renderer=renderer)

    fault_label = axis.text(
        12.34,
        FAULT_CALLOUT_Z_KM,
        "Main fault",
        fontsize=7.8,
        color="black",
        ha="left",
        va="center",
        zorder=13,
    )
    figure.canvas.draw()
    label_bounds = fault_label.get_window_extent(renderer=renderer)
    shift_text_in_display(
        fault_label,
        delta_x=arrow_bounds.x1 + gap_pixels - label_bounds.x0,
    )

    # The vertical LaTeX arrow uses the same point-sized gap to the injector
    # symbol above and the label below.  The marker radius includes its edge.
    injector_arrow = axis.text(
        INJECTOR_Y_KM,
        2.20,
        r"$\uparrow$",
        fontsize=9.0,
        color="black",
        ha="center",
        va="center",
        zorder=13,
    )
    figure.canvas.draw()
    injector_display_y = axis.transData.transform(
        (INJECTOR_Y_KM, INJECTOR_Z_KM)
    )[1]
    marker_radius_pixels = 1.75 * figure.dpi / 72.0
    injector_arrow_bounds = injector_arrow.get_window_extent(renderer=renderer)
    shift_text_in_display(
        injector_arrow,
        delta_y=(
            injector_display_y
            - marker_radius_pixels
            - gap_pixels
            - injector_arrow_bounds.y1
        ),
    )
    figure.canvas.draw()
    injector_arrow_bounds = injector_arrow.get_window_extent(renderer=renderer)

    injector_label = axis.text(
        INJECTOR_Y_KM,
        2.48,
        "Injector",
        fontsize=7.8,
        color="black",
        ha="center",
        va="center",
        zorder=13,
    )
    figure.canvas.draw()
    injector_label_bounds = injector_label.get_window_extent(renderer=renderer)
    shift_text_in_display(
        injector_label,
        delta_y=(
            injector_arrow_bounds.y0
            - gap_pixels
            - injector_label_bounds.y1
        ),
    )


def dissolve_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    cell_mask: np.ndarray,
) -> list[np.ndarray]:
    """Return material outlines with all internal grid edges removed."""
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

    irregular = {
        vertex: len(neighbors)
        for vertex, neighbors in adjacency.items()
        if len(neighbors) != 2
    }
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


def boundary_segments(
    points: np.ndarray,
    triangles: np.ndarray,
    cell_mask: np.ndarray,
) -> list[np.ndarray]:
    """Return only the exterior edges of a selected finite-width domain."""
    edge_counts: Counter[tuple[int, int]] = Counter()
    for triangle in triangles[np.asarray(cell_mask, dtype=bool)]:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_counts[tuple(sorted((int(first), int(second))))] += 1
    return [
        points[np.asarray(edge, dtype=int)]
        for edge, count in edge_counts.items()
        if count == 1
    ]


def plot_geology_inset(
    axis: mpl.axes.Axes,
    points: np.ndarray,
    triangles: np.ndarray,
    layer_ids: np.ndarray,
    fault_ids: np.ndarray,
    region_names: np.ndarray,
) -> None:
    masks = [
        (np.isin(region_names, ["MMUM", "Younger"]), OVERBURDEN_COLOR),
        (region_names == "AmphB", TOP_SEAL_COLOR),
        (region_names == "res_LM2", STORAGE_RESERVOIR_COLOR),
        ((layer_ids > 0) & (layer_ids % 2 == 1), CLAY_INTERBED_COLOR),
        ((layer_ids > 0) & (layer_ids % 2 == 0), SAND_INTERBED_COLOR),
    ]
    for mask, color in masks:
        outlines = dissolve_triangles(points, triangles, mask)
        axis.add_collection(
            PolyCollection(
                outlines,
                facecolors=color,
                edgecolors="none",
                linewidths=0.0,
                antialiaseds=False,
                zorder=1.0,
            )
        )

    fault_mask = fault_ids > 0
    axis.add_collection(
        PolyCollection(
            dissolve_triangles(points, triangles, fault_mask),
            facecolors=to_rgba(FAULT_CROSS_SECTION_COLOR, 0.92),
            edgecolors="none",
            linewidths=0.0,
            antialiaseds=False,
            zorder=3.0,
        )
    )
    axis.add_collection(
        LineCollection(
            boundary_segments(points, triangles, fault_mask),
            colors=FAULT_CROSS_SECTION_EDGE,
            linewidths=0.62,
            capstyle="round",
            joinstyle="round",
            antialiaseds=True,
            zorder=4.0,
        )
    )

    axis.set_xlim(*GRID_VIEW_Y_KM)
    axis.set_ylim(*GRID_VIEW_Z_KM)
    axis.invert_yaxis()
    axis.set_aspect(GEOLOGY_VERTICAL_EXAGGERATION, adjustable="box")
    axis.set_anchor("S")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_axis_off()
    plot_injector_symbol(
        axis,
        INJECTOR_Y_KM,
        INJECTOR_Z_KM,
        transform=axis.transData,
        outer_size=9,
        inner_size=2.3,
    )


def render(
    source_path: Path,
    model_path: Path,
    grid_path: Path,
    output_dir: Path,
) -> list[Path]:
    configure_style()
    regional_map, regional_section, field_section = load_source_panels(source_path)
    panel_a_image = compose_panel_a(regional_map, regional_section)
    model_image = load_model_image(model_path)
    model_injector_axes = locate_model_injector(model_image)
    points, triangles, layer_ids, fault_ids, region_names = load_step62_mesh(grid_path)

    figure = plt.figure(figsize=(6.85, 8.05), facecolor="white")
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=[2.05, 3.12, 2.30],
        left=0.018,
        right=0.992,
        bottom=0.048,
        top=0.992,
        hspace=0.035,
    )

    axis_top = figure.add_subplot(outer[0])
    show_image(axis_top, panel_a_image, add_frame=False)

    middle = outer[1].subgridspec(
        2, 1, height_ratios=[0.16, 2.96], hspace=0.0
    )
    axis_field = figure.add_subplot(middle[1])
    show_image(axis_field, field_section, add_frame=False)

    bottom = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.035)
    axis_model = figure.add_subplot(bottom[0, 0])
    axis_grid = figure.add_subplot(bottom[0, 1])
    show_image(axis_model, model_image, add_frame=False)
    plot_injector_symbol(
        axis_model,
        *model_injector_axes,
        transform=axis_model.transAxes,
    )
    annotate_model(axis_model, model_injector_axes)
    plot_geology_inset(
        axis_grid,
        points,
        triangles,
        layer_ids,
        fault_ids,
        region_names,
    )

    # Leave deliberate white space around the right-hand geology panel.  Scale
    # both dimensions equally so its geometry is unchanged, center it within
    # the right half of the row, and retain the existing bottom alignment.
    figure.canvas.draw()
    grid_box = axis_grid.get_position()
    scaled_width = grid_box.width * RIGHT_GEOLOGY_PANEL_SCALE
    scaled_height = grid_box.height * RIGHT_GEOLOGY_PANEL_SCALE
    axis_grid.set_position(
        [
            grid_box.x0 + 0.5 * (grid_box.width - scaled_width),
            grid_box.y0 + RIGHT_GEOLOGY_PANEL_Y_SHIFT,
            scaled_width,
            scaled_height,
        ]
    )
    figure.canvas.draw()
    annotate_geology_inset(axis_grid)

    label_x = outer[0].get_position(figure).x0 + 0.004
    for label, row in zip("abc", outer):
        row_box = row.get_position(figure)
        figure_panel_label(figure, label, label_x, row_box.y1 - 0.002)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig1_offshore_texas_field_case_model"
    outputs = [base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    export_bbox = Bbox.from_extents(
        0.0,
        BOTTOM_CANVAS_CROP_IN,
        figure.get_figwidth(),
        figure.get_figheight(),
    )
    figure.savefig(
        outputs[0],
        dpi=600,
        facecolor="white",
        bbox_inches=export_bbox,
        pad_inches=0.0,
    )
    figure.savefig(
        outputs[1],
        dpi=600,
        facecolor="white",
        bbox_inches=export_bbox,
        pad_inches=0.0,
    )
    figure.savefig(
        outputs[2],
        dpi=600,
        facecolor="white",
        bbox_inches=export_bbox,
        pad_inches=0.0,
    )
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = render(
        args.source_image.resolve(),
        args.model_image.resolve(),
        args.grid_vtu.resolve(),
        args.output_dir.resolve(),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
