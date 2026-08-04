r"""Render the workflow migration panels from the original simulation VTUs.

This workflow-only renderer reuses the finalized publication renderer from
``JutulDarcy.jl_shaowen`` in its default ``lithology_connected`` mode. This is
the same scientific rendering path used for the archived Case 6 and Case 7
figures. The original renderer and simulation data remain read-only.

The workflow-specific presentation changes are intentionally limited to:

* two representative cases in rows;
* gas saturation (Sg) in the left column and dissolved ratio (Rs) in the right;
* no time or colorbar annotation inside any scientific panel; and
* one short shared colorbar per column, with a centered title and endpoint
  labels only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import Colormap, Normalize


ORIGINAL_RENDERER = Path(
    r"D:\Github\JutulDarcy.jl_shaowen\scripts\visualization"
    r"\gom_rs_publication\render_gom_rs_cross_section.py"
)
RS_PRESET = Path(
    r"D:\Github\JutulDarcy.jl_shaowen\scripts\visualization"
    r"\gom_rs_publication\csp11_icefire_white_zero_paraview.json"
)
SG_PRESET = Path(
    r"D:\Github\JutulDarcy.jl_shaowen\scripts\visualization"
    r"\gom_saturation_publication"
    r"\inverted_black_body_radiation_paraview.json"
)
DEFAULT_CASE_6 = Path(
    r"D:\codex_gom\step62_effective_pc_global_plateau"
    r"\case6_s03_c012_case08_geology_v2"
    r"\gom_step62_effective_pc_global_plateau_"
    r"s03_c012_case08_geology_v2_0210.vtu"
)
DEFAULT_CASE_7 = Path(
    r"D:\codex_gom\step62_effective_pc_global_plateau"
    r"\case7_s04_c024_case03_geology_v2"
    r"\gom_step62_effective_pc_global_plateau_"
    r"s04_c024_case03_geology_v2_0210.vtu"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "assets" / "derived"

SLICE_X = 22500.0
SMOOTHING_LENGTH_M = 12.5
DISPLAY_CUTOFF = 0.015
# A slightly taller workflow canvas preserves the publication cross-section
# aspect ratio while avoiding the large side and inter-column gaps that occur
# when equal-aspect axes are forced into a shallow 2:1 canvas.
FIGURE_SIZE_INCHES = (7.2, 4.2)
VIEW_Y_LIMITS = (9775.0, 15290.5)
VIEW_Z_LIMITS = (-12.5, 3002.6)


@dataclass(frozen=True)
class ScalarSpec:
    """Rendering contract for one migration variable."""

    key: str
    array_name: str
    limits: tuple[float, float]
    title: str
    endpoint_labels: tuple[str, str]


SG_SPEC = ScalarSpec(
    key="Sg",
    array_name="Saturations_2",
    limits=(0.0, 0.6),
    title=r"$S_g$ [-]",
    endpoint_labels=("0", "0.6"),
)
RS_SPEC = ScalarSpec(
    key="Rs",
    array_name="Rs",
    limits=(0.0, 18.0),
    title=r"$R_s$ [Sm$^3$ CO$_2$ / Sm$^3$ brine]",
    endpoint_labels=("0", "18"),
)


def parse_args() -> argparse.Namespace:
    """Parse workflow-specific source and output options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-6", type=Path, default=DEFAULT_CASE_6)
    parser.add_argument("--case-7", type=Path, default=DEFAULT_CASE_7)
    parser.add_argument("--reference-renderer", type=Path, default=ORIGINAL_RENDERER)
    parser.add_argument("--rs-preset", type=Path, default=RS_PRESET)
    parser.add_argument("--sg-preset", type=Path, default=SG_PRESET)
    parser.add_argument("--slice-x", type=float, default=SLICE_X)
    parser.add_argument(
        "--smooth-length-m",
        type=float,
        default=SMOOTHING_LENGTH_M,
    )
    parser.add_argument("--png-dpi", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="panel3_migration_cross_sections_transparent",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_renderer(path: Path) -> ModuleType:
    """Load the original publication renderer as a read-only helper module."""

    spec = importlib.util.spec_from_file_location(
        "predict_original_geology_aware_renderer",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_matplotlib() -> None:
    """Use compact publication typography while leaving panels annotation-free."""

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
        }
    )


def load_section(source: Path, slice_x: float) -> pv.PolyData:
    """Read one final-time VTU and extract the complete reservoir cross-section."""

    grid = pv.read(source)
    section = grid.slice(
        normal=(1.0, 0.0, 0.0),
        origin=(slice_x, 0.0, 0.0),
    )
    if section.n_cells == 0:
        raise ValueError(f"No cells intersect x={slice_x:g} in {source}")
    return section


def render_scientific_panel(
    axes: plt.Axes,
    section: pv.PolyData,
    scalar: ScalarSpec,
    colormap: Colormap,
    reference: ModuleType,
    smoothing_length_m: float,
) -> dict[str, int | float | str]:
    """Render one complete panel using the original scientific algorithm."""

    if scalar.array_name not in section.cell_data:
        raise KeyError(f"Missing cell array {scalar.array_name!r}")
    values = np.asarray(section.cell_data[scalar.array_name], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {scalar.array_name}")

    # The archived implementation uses module globals for the active scalar.
    # Setting them before each serial panel call preserves its exact routines.
    reference.RS_ARRAY = scalar.array_name
    reference.RS_LIMITS = scalar.limits
    reference.DISPLAY_CUTOFF = DISPLAY_CUTOFF

    stratigraphy_flag = np.asarray(
        section.cell_data[reference.STRATIGRAPHY_FLAG]
    )
    fault_flag = np.asarray(section.cell_data[reference.FAULT_FLAG])
    clay_loops = reference.extract_closed_boundary_loops(
        section,
        stratigraphy_flag == 2,
    )
    fault_loops = reference.extract_closed_boundary_loops(
        section,
        fault_flag > 0,
    )
    components = reference.connected_geological_components(
        section,
        stratigraphy_smoothing_mode="lithology_connected",
    )
    y_grid = np.linspace(*reference.Y_LIMITS, reference.REGULAR_GRID_SHAPE[1])
    z_grid = np.linspace(*reference.Z_LIMITS, reference.REGULAR_GRID_SHAPE[0])
    norm = Normalize(*scalar.limits)

    axes.set_facecolor("none")
    for loop in clay_loops:
        reference.add_filled_loop(axes, loop, "#b9b7b2", zorder=1.0)

    displayed_domains = 0
    displayed_points = 0
    for component in components:
        component_mesh = component["mesh"]
        if not isinstance(component_mesh, pv.PolyData):
            raise TypeError("Geological component is not PolyData")
        if float(component["raw_rs_max"]) <= DISPLAY_CUTOFF:
            continue

        yy, zz, smoothed, _, component_displayed_points = (
            reference.interpolate_component_to_regular_grid(
                component_mesh,
                y_grid,
                z_grid,
                smoothing_length_m,
            )
        )
        if component_displayed_points <= 0:
            continue

        plume = axes.contourf(
            yy,
            zz,
            smoothed,
            levels=np.linspace(DISPLAY_CUTOFF, scalar.limits[1], 257),
            cmap=colormap,
            norm=norm,
            antialiased=False,
            zorder=2.0,
        )
        plume.set_clip_path(
            reference.add_exact_component_clip(axes, component_mesh)
        )
        displayed_domains += 1
        displayed_points += int(component_displayed_points)

    for loop in fault_loops:
        axes.plot(
            loop[:, 0],
            loop[:, 1],
            color="#303030",
            linewidth=0.65,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
            zorder=3.0,
        )

    # These are the exact publication view limits from provenance commit
    # 40cabfcb, used to produce the archived Case 6 and Case 7 figures.
    axes.set_xlim(*VIEW_Y_LIMITS)
    axes.set_ylim(VIEW_Z_LIMITS[1], VIEW_Z_LIMITS[0])
    axes.set_aspect("equal", adjustable="box")
    axes.set_axis_off()
    return {
        "scalar": scalar.key,
        "section_cells": int(section.n_cells),
        "clay_loops": len(clay_loops),
        "fault_loops": len(fault_loops),
        "geological_components": len(components),
        "displayed_components": displayed_domains,
        "displayed_regular_grid_points": displayed_points,
        "raw_min": float(np.min(values)),
        "raw_max": float(np.max(values)),
    }


def add_shared_colorbar(
    figure: plt.Figure,
    rectangle: tuple[float, float, float, float],
    scalar: ScalarSpec,
    colormap: Colormap,
) -> None:
    """Add a short shared colorbar with centered title and endpoint labels."""

    axes = figure.add_axes(rectangle)
    gradient = np.linspace(*scalar.limits, 512, dtype=float)[None, :]
    axes.imshow(
        gradient,
        extent=(*scalar.limits, 0.0, 1.0),
        aspect="auto",
        cmap=colormap,
        norm=Normalize(*scalar.limits),
        interpolation="nearest",
    )
    axes.set_xlim(*scalar.limits)
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([])
    axes.set_xticks(scalar.limits)
    axes.set_xticklabels(scalar.endpoint_labels)
    axes.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        labelbottom=True,
        direction="out",
        length=2.2,
        width=0.5,
        pad=1.0,
        labelsize=20.0,
    )
    axes.set_title(scalar.title, loc="center", pad=4.0, fontsize=22.0)
    for spine in axes.spines.values():
        spine.set_visible(False)


def main() -> None:
    """Render four full panels, shared scales, and provenance records."""

    args = parse_args()
    sources = {
        "case_6": args.case_6.expanduser().resolve(),
        "case_7": args.case_7.expanduser().resolve(),
        "reference_renderer": args.reference_renderer.expanduser().resolve(),
        "rs_preset": args.rs_preset.expanduser().resolve(),
        "sg_preset": args.sg_preset.expanduser().resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source file(s):\n" + "\n".join(missing))
    if args.smooth_length_m < 0:
        raise ValueError("smooth-length-m must be nonnegative")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.output_stem}.png"
    pdf_path = output_dir / f"{args.output_stem}.pdf"
    json_path = output_dir / f"{args.output_stem}_provenance.json"

    hashes_before = {name: sha256_file(path) for name, path in sources.items()}
    reference = load_reference_renderer(sources["reference_renderer"])
    configure_matplotlib()
    colormaps = {
        SG_SPEC.key: reference.load_csp11_colormap(sources["sg_preset"]),
        RS_SPEC.key: reference.load_csp11_colormap(sources["rs_preset"]),
    }
    sections = {
        "case_6": load_section(sources["case_6"], args.slice_x),
        "case_7": load_section(sources["case_7"], args.slice_x),
    }

    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="none")
    panel_rectangles = {
        ("case_6", SG_SPEC.key): (0.002, 0.600, 0.496, 0.380),
        ("case_6", RS_SPEC.key): (0.502, 0.600, 0.496, 0.380),
        ("case_7", SG_SPEC.key): (0.002, 0.200, 0.496, 0.380),
        ("case_7", RS_SPEC.key): (0.502, 0.200, 0.496, 0.380),
    }
    diagnostics: dict[str, dict[str, int | float | str]] = {}
    for case_key in ("case_6", "case_7"):
        for scalar in (SG_SPEC, RS_SPEC):
            axes = figure.add_axes(panel_rectangles[(case_key, scalar.key)])
            diagnostics[f"{case_key}_{scalar.key}"] = render_scientific_panel(
                axes,
                sections[case_key],
                scalar,
                colormaps[scalar.key],
                reference,
                args.smooth_length_m,
            )

    add_shared_colorbar(
        figure,
        (0.095, 0.080, 0.31, 0.014),
        SG_SPEC,
        colormaps[SG_SPEC.key],
    )
    add_shared_colorbar(
        figure,
        (0.595, 0.080, 0.31, 0.014),
        RS_SPEC,
        colormaps[RS_SPEC.key],
    )

    metadata = {
        "Title": "Panel 3 complete CO2 migration cross sections",
        "Subject": (
            "Complete Case 6 and Case 7 cross-sections with shared Sg and Rs "
            "colorbars"
        ),
        "Author": Path(__file__).name,
    }
    figure.savefig(
        png_path,
        format="png",
        dpi=args.png_dpi,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        metadata={"Title": metadata["Title"], "Description": metadata["Subject"]},
    )
    figure.savefig(
        pdf_path,
        format="pdf",
        transparent=True,
        facecolor="none",
        edgecolor="none",
        metadata=metadata,
    )
    plt.close(figure)

    hashes_after = {name: sha256_file(path) for name, path in sources.items()}
    if hashes_after != hashes_before:
        raise RuntimeError("A scientific source changed during rendering")

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "scientific_sources_modified": False,
        "reference_algorithm": str(sources["reference_renderer"]),
        "layout": {
            "rows": ["Case 6", "Case 7"],
            "columns": ["Sg", "Rs"],
            "complete_cross_section": True,
            "local_annotations": False,
            "shared_colorbar_ticks": {"Sg": [0.0, 0.6], "Rs": [0.0, 18.0]},
            "colorbar_titles_centered": True,
        },
        "rendering": {
            "slice_x_m": args.slice_x,
            "smooth_length_m": args.smooth_length_m,
            "display_cutoff": DISPLAY_CUTOFF,
            "sg_limits": list(SG_SPEC.limits),
            "rs_limits": list(RS_SPEC.limits),
            "view_y_limits_m": list(VIEW_Y_LIMITS),
            "view_z_limits_m": list(VIEW_Z_LIMITS),
        },
        "diagnostics": diagnostics,
        "inputs": {
            name: {"path": str(path), "sha256": hashes_before[name]}
            for name, path in sources.items()
        },
        "outputs": {
            "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
            "pdf": {"path": str(pdf_path), "sha256": sha256_file(pdf_path)},
        },
    }
    json_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"Created: {png_path}")
    print(f"Created: {pdf_path}")
    print(f"Created: {json_path}")
    print("Verified: complete cross-sections; scientific sources unchanged")


if __name__ == "__main__":
    main()
