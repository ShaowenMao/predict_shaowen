"""Compose a two-panel Step62 fault-domain segmentation figure.

Panel (a) shows the complete 3-D fault domain and its 87 along-strike
segments. Panel (b) shows the six PREDICT throw windows in the central y-z
cross section. The geometry and geological colors are inherited from the
validated Step62 rendering workflow.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch
from PIL import Image


HERE = Path(__file__).resolve().parent
WORKFLOW = HERE / "render_fault_grid_multiscale.py"
FIGURE_DIR = HERE.parents[1] / "figures"
DEFAULT_OVERVIEW = FIGURE_DIR / "source" / "figs3_fault_overview_vector_base.png"
DEFAULT_OUTPUT = FIGURE_DIR / "fig3_fault_domain_discretization.png"
DEFAULT_PDF = FIGURE_DIR / "fig3_fault_domain_discretization.pdf"


def load_workflow():
    spec = importlib.util.spec_from_file_location("fault_grid_workflow", WORKFLOW)
    if spec is None or spec.loader is None:
        raise ImportError(WORKFLOW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLOW = load_workflow()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--grid-dir", type=Path, default=FLOW.BASE.DEFAULT_GRID_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 12.0,
            "text.usetex": True,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelsize": 12.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_overview_with_vector_axes(
    axis: plt.Axes,
    overview: Image.Image,
    x_view_shift: float = 0.0,
) -> tuple[float, dict[str, object]]:
    """Draw panel (a) with vector coordinate axes around its raster model.

    The accepted 3-D geological rendering remains a high-resolution raster,
    while every ruler, tick, tick label, and axis title is drawn here as a
    Matplotlib artist so it remains vector text/linework in the PDF.
    """
    expected_size = (4239, 1418)
    if overview.size != expected_size:
        raise ValueError(
            f"Expected panel-(a) overview size {expected_size}, found {overview.size}"
        )

    canvas_width = 4785.0
    canvas_height = 1887.0
    image_x = 551.0
    image_y = 29.0
    image_width, image_height = overview.size
    axis.imshow(
        overview,
        interpolation="none",
        origin="upper",
        extent=(
            image_x,
            image_x + image_width,
            image_y + image_height,
            image_y,
        ),
    )
    # Shift the visible data window instead of placing the Matplotlib axes
    # outside the figure. This keeps the accepted panel placement while
    # avoiding an unused left gutter in tight-bounding-box exports.
    axis.set_xlim(-0.5 + x_view_shift, canvas_width - 0.5 + x_view_shift)
    axis.set_ylim(canvas_height - 0.5, -0.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    axis_color = "#202020"
    line_width = 0.58
    font_size = 12.0

    x_start = 941.0
    x_stop = 4763.0
    x_ruler_y = 1457.0
    axis.plot(
        [x_start, x_stop],
        [x_ruler_y, x_ruler_y],
        color=axis_color,
        lw=line_width,
        clip_on=False,
        zorder=20,
    )
    x_ticks = np.asarray((0.0, 22.5, 45.0))
    x_positions = np.linspace(x_start, x_stop, x_ticks.size)
    x_tick_labels = []
    for index, (position, value) in enumerate(zip(x_positions, x_ticks)):
        axis.plot(
            [position, position],
            [1449.0, x_ruler_y],
            color=axis_color,
            lw=line_width,
            clip_on=False,
            zorder=20,
        )
        horizontal_alignment = "center"
        if index == 0:
            horizontal_alignment = "left"
        elif index == x_ticks.size - 1:
            horizontal_alignment = "right"
        x_tick_labels.append(
            axis.text(
                position,
                1477.0,
                f"{value:g}",
                fontsize=font_size,
                color="black",
                ha=horizontal_alignment,
                va="top",
                clip_on=False,
                zorder=21,
            )
        )
    x_title = axis.text(
        0.5 * (x_start + x_stop),
        1747.0,
        r"Along-strike coordinate, $x$ (km)",
        fontsize=font_size,
        color="black",
        ha="center",
        va="top",
        clip_on=False,
        zorder=21,
    )

    z_ruler_x = 526.0
    z_start = 72.0
    z_stop = 1404.0
    z_max = 2.99011058
    axis.plot(
        [z_ruler_x, z_ruler_x],
        [z_start, z_stop],
        color=axis_color,
        lw=line_width,
        clip_on=False,
        zorder=20,
    )
    z_ticks = np.asarray((0.0, 1.0, 2.0, z_max))
    z_positions = z_start + z_ticks / z_max * (z_stop - z_start)
    z_tick_labels = []
    for value, position in zip(z_ticks, z_positions):
        axis.plot(
            [519.0, 533.0],
            [position, position],
            color=axis_color,
            lw=line_width,
            clip_on=False,
            zorder=20,
        )
        label = f"{value:g}" if value < z_max else f"{value:.2f}"
        z_tick_labels.append(
            axis.text(
                496.0,
                position,
                label,
                fontsize=font_size,
                color="black",
                ha="right",
                va="center",
                clip_on=False,
                zorder=21,
            )
        )
    z_title = axis.text(
        88.0,
        0.5 * (z_start + z_stop),
        r"Depth, $z$ (km)",
        fontsize=font_size,
        color="black",
        rotation=90.0,
        ha="center",
        va="center",
        clip_on=False,
        zorder=21,
    )

    label_artists = {
        "x_tick_labels": x_tick_labels,
        "z_tick_labels": z_tick_labels,
        "x_title": x_title,
        "z_title": z_title,
    }
    return (2852.0 - x_view_shift) / (canvas_width - 1.0), label_artists


def match_panel_a_axis_label_spacing(
    figure: plt.Figure,
    axis_a: plt.Axes,
    axis_b: plt.Axes,
    label_artists: dict[str, object],
) -> tuple[float, float]:
    """Match panel (a)'s title-to-tick gaps to panel (b) in display points."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    b_x_ticks = [label for label in axis_b.get_xticklabels() if label.get_visible()]
    b_z_ticks = [label for label in axis_b.get_yticklabels() if label.get_visible()]
    if not b_x_ticks or not b_z_ticks:
        raise RuntimeError("Panel (b) must have visible tick labels before spacing match")

    b_x_tick_bottom = min(label.get_window_extent(renderer).y0 for label in b_x_ticks)
    b_x_title_top = axis_b.xaxis.label.get_window_extent(renderer).y1
    target_x_gap = b_x_tick_bottom - b_x_title_top

    b_z_tick_left = min(label.get_window_extent(renderer).x0 for label in b_z_ticks)
    b_z_title_right = axis_b.yaxis.label.get_window_extent(renderer).x1
    target_z_gap = b_z_tick_left - b_z_title_right

    a_x_ticks = label_artists["x_tick_labels"]
    a_z_ticks = label_artists["z_tick_labels"]
    x_title = label_artists["x_title"]
    z_title = label_artists["z_title"]

    a_x_tick_bottom = min(label.get_window_extent(renderer).y0 for label in a_x_ticks)
    x_title_bounds = x_title.get_window_extent(renderer)
    x_shift_display = a_x_tick_bottom - target_x_gap - x_title_bounds.y1
    x_title_position = axis_a.transData.transform(x_title.get_position())
    x_title.set_position(
        axis_a.transData.inverted().transform(
            (x_title_position[0], x_title_position[1] + x_shift_display)
        )
    )

    a_z_tick_left = min(label.get_window_extent(renderer).x0 for label in a_z_ticks)
    z_title_bounds = z_title.get_window_extent(renderer)
    z_shift_display = a_z_tick_left - target_z_gap - z_title_bounds.x1
    z_title_position = axis_a.transData.transform(z_title.get_position())
    z_title.set_position(
        axis_a.transData.inverted().transform(
            (z_title_position[0] + z_shift_display, z_title_position[1])
        )
    )

    figure.canvas.draw()
    return target_x_gap * 72.0 / figure.dpi, target_z_gap * 72.0 / figure.dpi


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)

    if not args.overview.is_file():
        raise FileNotFoundError(args.overview)

    yz, triangles, groups = FLOW.BASE.load_step62_fault(args.grid_dir)
    overview = Image.open(args.overview).convert("RGBA")

    figure = plt.figure(figsize=(11.2, 3.75), facecolor="white")
    # Enlarge panel (a) about its established upper edge. This brings its
    # along-strike baseline into horizontal alignment with panel (b), while
    # using the available left margin for the corresponding width increase.
    panel_a_shift = -0.016
    panel_a_left = 0.000
    panel_a_width = 0.655
    axis_a = figure.add_axes((panel_a_left, 0.015, panel_a_width, 0.772))
    # Match the total visual height of panel (a), including its legend, while
    # aligning the lower extents of the two completed panels.
    axis_b = figure.add_axes((0.665, 0.192, 0.315, 0.713))

    panel_a_x_view_shift = (
        -panel_a_shift / panel_a_width * (4785.0 - 1.0)
    )
    strike_center_axis, panel_a_label_artists = draw_overview_with_vector_axes(
        axis_a,
        overview,
        x_view_shift=panel_a_x_view_shift,
    )
    strike_center_figure = panel_a_left + panel_a_width * strike_center_axis
    # Preserve the accepted absolute horizontal placement of the legend and
    # segment annotation after enlarging panel (a).
    alignment_nudge_figure = -0.0427307692307693
    aligned_center_figure = strike_center_figure + alignment_nudge_figure
    aligned_center_axis = strike_center_axis + alignment_nudge_figure / panel_a_width
    axis_a.text(
        aligned_center_axis,
        0.675,
        r"87 segments per throw window",
        transform=axis_a.transAxes,
        ha="center",
        va="center",
        fontsize=12.0,
        color="black",
    )

    FLOW.draw_cross_section(axis_b, yz, triangles, groups)
    axis_b.xaxis.label.set_fontsize(12.0)
    axis_b.yaxis.label.set_fontsize(12.0)
    axis_b.tick_params(labelsize=12.0, direction="in", length=3.5, width=0.8)
    for label in axis_b.texts:
        if label.get_text().startswith(r"$\mathrm{W}"):
            label.set_fontsize(12.0)
            label.set_fontweight("normal")

    figure.text(
        0.020 + panel_a_shift,
        0.955,
        r"(a)",
        ha="left",
        va="top",
        fontsize=12.0,
    )
    figure.text(0.625, 0.955, r"(b)", ha="left", va="top", fontsize=12.0)

    legend_handles = [
        Patch(
            facecolor=FLOW.OVERBURDEN_COLOR,
            edgecolor="none",
            label="Overburden fault domain",
        ),
        Patch(
            facecolor=FLOW.TOP_SEAL_COLOR,
            edgecolor="none",
            label="Top-seal fault domain",
        ),
        Patch(
            facecolor=FLOW.STORAGE_RESERVOIR_COLOR,
            edgecolor="none",
            label="Storage-reservoir fault domain",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(aligned_center_figure, 0.985),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.05,
        handleheight=0.95,
        handletextpad=0.45,
        labelspacing=0.30,
        fontsize=12.0,
    )

    x_gap_points, z_gap_points = match_panel_a_axis_label_spacing(
        figure,
        axis_a,
        axis_b,
        panel_a_label_artists,
    )

    # The accepted panel-(a) render uses a transparent coordinate canvas that
    # extends below the vector x-axis title. Crop two-thirds of that unused
    # lower margin at export without moving or rescaling either panel.
    figure.canvas.draw()
    tight_bbox = figure.get_tightbbox(figure.canvas.get_renderer())
    bottom_crop_inches = 109.0 / 600.0
    export_bbox = Bbox.from_extents(
        tight_bbox.x0,
        tight_bbox.y0 + bottom_crop_inches,
        tight_bbox.x1,
        tight_bbox.y1,
    )

    for path in (args.output, args.pdf):
        figure.savefig(
            path,
            dpi=600,
            facecolor="white",
            bbox_inches=export_bbox,
            pad_inches=0.025,
        )
    plt.close(figure)
    overview.close()

    print(args.output)
    print(args.pdf)
    print(
        "Matched panel-(a) title gaps to panel (b): "
        f"x={x_gap_points:.3f} pt, z={z_gap_points:.3f} pt"
    )


if __name__ == "__main__":
    main()
