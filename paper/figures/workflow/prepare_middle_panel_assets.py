"""Generate matched scientific assets for workflow Figure Panel 2.

The spatial maps use one geology (``s02_c003``) and three Level 3 case
families: independent (case 01), fault-wide (case 03), and grouped (case 07).
Each permeability, porosity, entry-pressure, and effective-Swi column comes
from the same reservoir-ready MAT file. The curve panels show all 87
along-strike curves for W4 of case 01; no medoid or other representative curve
is overlaid.

Generated PDFs are committed with the figure project, so the manuscript can
still build when the external analysis workspace is unavailable. Set
``PREDICT_WORKFLOW_PANEL2_SOURCE`` to regenerate from a relocated source-data
directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext  # noqa: E402


ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "assets" / "derived"
DEFAULT_SOURCE = Path(
    r"D:\codex_gom\UQ_workflow\representative_stratigraphy_schematic"
    r"\spatial_fault_permeability\s02_c003\source_data"
)
SOURCE = Path(os.environ.get("PREDICT_WORKFLOW_PANEL2_SOURCE", DEFAULT_SOURCE))

CASES = (1, 3, 7)
WINDOW_INDEX = 3  # W4, zero based.
MAP_WIDTH_MM = 14.8
MAP_HEIGHT_MM = 3.4
CURVE_WIDTH_MM = 25.0
CURVE_HEIGHT_MM = 18.5


def _configure_plotting() -> None:
    """Use a compact Times-like style compatible with the LaTeX figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _source_path(case_id: int) -> Path:
    return SOURCE / f"reservoir_ready_s02_c003_case{case_id:02d}.mat"


def _output_paths() -> list[Path]:
    maps = [
        DERIVED / f"panel2_{field}_case{case_id:02d}.pdf"
        for field in ("kxx", "kyy", "kzz", "phi", "pe", "swi")
        for case_id in CASES
    ]
    return maps + [
        DERIVED / "panel2_pc_w4_ensemble.pdf",
        DERIVED / "panel2_kr_w4_ensemble.pdf",
    ]


def _inputs_available() -> bool:
    return all(_source_path(case_id).is_file() for case_id in CASES)


def _outputs_current() -> bool:
    outputs = _output_paths()
    if not all(path.is_file() for path in outputs):
        return False
    newest_input = max(
        Path(__file__).stat().st_mtime,
        *(_source_path(case_id).stat().st_mtime for case_id in CASES),
    )
    return min(path.stat().st_mtime for path in outputs) >= newest_input


def _load_maps(path: Path) -> dict[str, np.ndarray]:
    """Return matched 6-by-87 full-fault fields from one reservoir-ready file."""
    with h5py.File(path, "r") as mat:
        permeability = np.asarray(
            mat["reservoirReady/effectivePermeability/mD"], dtype=float
        )
        porosity = np.asarray(mat["reservoirReady/upscaledPorosity"], dtype=float)
        pc_cells = mat["reservoirReady/pcCurves"]

        if pc_cells.shape != (87, 6):
            raise ValueError(
                f"Unexpected Pc-curve cell shape {pc_cells.shape} in {path}."
            )

        entry_pressure = np.full((6, 87), np.nan, dtype=float)
        effective_swi = np.full((6, 87), np.nan, dtype=float)
        entry_threshold = 1.0e-5 * (1.0 + 1.0e-8)
        for slice_index in range(87):
            for window_index in range(6):
                curve = _dereference_curve(
                    mat, pc_cells, slice_index, window_index
                )
                saturation = _read_vector(curve, "gasSaturation")
                pressure = _read_vector(curve, "pcBar")
                if saturation.size != pressure.size:
                    raise ValueError(
                        f"Pc coordinates differ in length for {curve.name}."
                    )
                entry_ids = np.flatnonzero(saturation > entry_threshold)
                if entry_ids.size == 0:
                    raise ValueError(
                        f"No connected entry-pressure point in {curve.name}."
                    )
                entry_value = float(pressure[entry_ids[0]])
                swi_value = float(np.asarray(curve["effectiveSwi"]).squeeze())
                bulk_sg_max = float(np.asarray(curve["bulkSgMax"]).squeeze())
                if not np.isfinite(entry_value) or entry_value <= 0:
                    raise ValueError(f"Invalid entry pressure in {curve.name}.")
                if not np.isfinite(swi_value) or not 0 <= swi_value <= 1:
                    raise ValueError(f"Invalid effective Swi in {curve.name}.")
                if abs(swi_value - (1.0 - bulk_sg_max)) > 1.0e-10:
                    raise ValueError(
                        f"Effective Swi is inconsistent with BulkSgMax in "
                        f"{curve.name}."
                    )
                entry_pressure[window_index, slice_index] = entry_value
                effective_swi[window_index, slice_index] = swi_value

    if permeability.shape != (3, 87, 6):
        raise ValueError(
            f"Unexpected permeability shape {permeability.shape} in {path}."
        )
    if porosity.shape != (87, 6):
        raise ValueError(f"Unexpected porosity shape {porosity.shape} in {path}.")

    log_permeability = np.log10(
        np.maximum(permeability.transpose(0, 2, 1), np.finfo(float).tiny)
    )
    kxx, kyy, kzz = log_permeability
    phi = porosity.T
    fields = {
        "kxx": kxx,
        "kyy": kyy,
        "kzz": kzz,
        "phi": phi,
        "pe": entry_pressure,
        "swi": effective_swi,
    }
    if not all(np.all(np.isfinite(values)) for values in fields.values()):
        raise ValueError(f"Non-finite Panel 2 map values found in {path}.")
    return fields


def _entry_pressure_cmap() -> LinearSegmentedColormap:
    """Return the established blue-to-rust entry-pressure color scale."""
    anchors = np.array(
        [
            [0.08, 0.18, 0.42],
            [0.10, 0.43, 0.67],
            [0.25, 0.68, 0.70],
            [0.88, 0.82, 0.47],
            [0.88, 0.50, 0.12],
            [0.60, 0.24, 0.08],
        ]
    )
    return LinearSegmentedColormap.from_list("entry_pressure", anchors)


def _effective_swi_cmap() -> LinearSegmentedColormap:
    """Return the established blue-to-gold effective-Swi color scale."""
    anchors = np.array(
        [
            [0.10, 0.18, 0.40],
            [0.15, 0.40, 0.58],
            [0.36, 0.61, 0.62],
            [0.72, 0.73, 0.48],
            [0.90, 0.70, 0.25],
        ]
    )
    return LinearSegmentedColormap.from_list("effective_swi", anchors)


def _save_map(
    values: np.ndarray,
    destination: Path,
    *,
    cmap: str,
    value_range: tuple[float, float],
) -> None:
    """Save one border-only heatmap at its final physical display size."""
    fig = plt.figure(
        figsize=(MAP_WIDTH_MM / 25.4, MAP_HEIGHT_MM / 25.4),
        facecolor="none",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=value_range[0],
        vmax=value_range[1],
        rasterized=True,
    )
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#222222",
            linewidth=0.45,
            clip_on=False,
        )
    )
    ax.set_axis_off()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, transparent=True, dpi=600, pad_inches=0)
    plt.close(fig)


def _dereference_curve(
    mat: h5py.File, cell_dataset: h5py.Dataset, slice_index: int, window_index: int
) -> h5py.Group:
    reference = cell_dataset[slice_index, window_index]
    if not reference:
        raise ValueError(
            f"Missing curve reference at slice {slice_index + 1}, "
            f"window {window_index + 1}."
        )
    group = mat[reference]
    if not isinstance(group, h5py.Group):
        raise TypeError("A reservoir-ready curve cell did not reference a struct group.")
    return group


def _read_vector(group: h5py.Group, field: str) -> np.ndarray:
    values = np.asarray(group[field], dtype=float).ravel()
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid curve field {group.name}/{field}.")
    return values


def _load_w4_curves(
    path: Path,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, ...]]]:
    """Load all native Pc and slice-scaled Kr curves for W4."""
    pc_curves: list[tuple[np.ndarray, np.ndarray]] = []
    kr_curves: list[tuple[np.ndarray, ...]] = []
    with h5py.File(path, "r") as mat:
        pc_cells = mat["reservoirReady/pcCurves"]
        kr_cells = mat["reservoirReady/krCurves"]
        if pc_cells.shape != (87, 6) or kr_cells.shape != (87, 6):
            raise ValueError("Expected 87-slice by 6-window Pc and Kr cell arrays.")

        for slice_index in range(87):
            pc_group = _dereference_curve(
                mat, pc_cells, slice_index, WINDOW_INDEX
            )
            kr_group = _dereference_curve(
                mat, kr_cells, slice_index, WINDOW_INDEX
            )
            pc_curves.append(
                (
                    _read_vector(pc_group, "gasSaturation"),
                    _read_vector(pc_group, "pcBar"),
                )
            )
            kr_curves.append(
                (
                    _read_vector(kr_group, "gasSaturation"),
                    _read_vector(kr_group, "krw"),
                    _read_vector(kr_group, "krg"),
                )
            )
    return pc_curves, kr_curves


def _style_curve_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.grid(True, which="major", color="#D7D7D7", linewidth=0.35)
    ax.tick_params(direction="out", pad=1.2)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.6)


def _save_pc_ensemble(
    curves: list[tuple[np.ndarray, np.ndarray]], destination: Path
) -> None:
    fig = plt.figure(
        figsize=(CURVE_WIDTH_MM / 25.4, CURVE_HEIGHT_MM / 25.4),
        facecolor="none",
    )
    ax = fig.add_axes([0.45, 0.43, 0.52, 0.53])
    for saturation, pressure in curves:
        order = np.argsort(saturation)
        ax.semilogy(
            saturation[order],
            np.maximum(pressure[order], 1.0e-2),
            color="#9AA4AE",
            linewidth=0.35,
            alpha=0.42,
        )
    _style_curve_axes(ax)
    ax.set_ylim(1.0e-2, 1.0e3)
    ax.set_yticks([1.0e-2, 1.0e0, 1.0e2])
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xlabel(r"$S_g$ [-]", labelpad=0.5)
    ax.set_ylabel(r"$P_c$ [bar]", labelpad=0.5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, transparent=True, dpi=600, pad_inches=0)
    plt.close(fig)


def _save_kr_ensemble(
    curves: list[tuple[np.ndarray, ...]], destination: Path
) -> None:
    fig = plt.figure(
        figsize=(CURVE_WIDTH_MM / 25.4, CURVE_HEIGHT_MM / 25.4),
        facecolor="none",
    )
    ax = fig.add_axes([0.45, 0.43, 0.52, 0.53])
    for saturation, krw, krg in curves:
        order = np.argsort(saturation)
        ax.plot(
            saturation[order],
            krw[order],
            color="#2B6CB0",
            linewidth=0.32,
            alpha=0.20,
        )
        ax.plot(
            saturation[order],
            krg[order],
            color="#DD6B20",
            linewidth=0.32,
            alpha=0.20,
        )
    _style_curve_axes(ax)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"$S_g$ [-]", labelpad=0.5)
    ax.set_ylabel(r"$k_r$ [-]", labelpad=0.5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, transparent=True, dpi=600, pad_inches=0)
    plt.close(fig)


def prepare() -> None:
    """Regenerate Panel 2 assets when source data are locally available."""
    outputs = _output_paths()
    if not _inputs_available():
        missing_outputs = [path for path in outputs if not path.is_file()]
        if missing_outputs:
            missing = "\n".join(str(path) for path in missing_outputs)
            raise FileNotFoundError(
                "Panel 2 sources are unavailable and generated assets are missing:\n"
                + missing
            )
        return
    if _outputs_current():
        return

    _configure_plotting()
    for case_id in CASES:
        fields = _load_maps(_source_path(case_id))
        for field in ("kxx", "kyy", "kzz"):
            _save_map(
                fields[field],
                DERIVED / f"panel2_{field}_case{case_id:02d}.pdf",
                cmap="viridis",
                value_range=(-6.0, 2.0),
            )
        _save_map(
            fields["phi"],
            DERIVED / f"panel2_phi_case{case_id:02d}.pdf",
            cmap="cividis",
            value_range=(0.16, 0.27),
        )
        _save_map(
            np.log10(fields["pe"]),
            DERIVED / f"panel2_pe_case{case_id:02d}.pdf",
            cmap=_entry_pressure_cmap(),
            value_range=(np.log10(0.03), np.log10(15.0)),
        )
        _save_map(
            fields["swi"],
            DERIVED / f"panel2_swi_case{case_id:02d}.pdf",
            cmap=_effective_swi_cmap(),
            value_range=(0.12, 0.32),
        )

    pc_curves, kr_curves = _load_w4_curves(_source_path(1))
    if len(pc_curves) != 87 or len(kr_curves) != 87:
        raise ValueError("Panel 2 must contain exactly 87 Pc and 87 Kr curves.")
    _save_pc_ensemble(pc_curves, DERIVED / "panel2_pc_w4_ensemble.pdf")
    _save_kr_ensemble(kr_curves, DERIVED / "panel2_kr_w4_ensemble.pdf")


if __name__ == "__main__":
    prepare()
