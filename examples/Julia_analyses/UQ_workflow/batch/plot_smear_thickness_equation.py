"""Visualize the smear-thickness equation and its PREDICT Vcl response.

The plotted friction angles are distribution medians from
getResidualFrictionAngle.m. They illustrate deterministic equation behavior;
individual PREDICT realizations sample stochastic friction angles around these
representative values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


BLUE = "#225f9b"
ORANGE = "#db8505"
MAGENTA = "#985493"
INK = "#17243a"
GRID = "#d8dde5"
CLAY_COLORS = {0.4: BLUE, 0.5: ORANGE, 0.6: MAGENTA}


def parse_args() -> argparse.Namespace:
    """Parse the output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def friction_bounds(vcl: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return PREDICT residual-friction bounds for Vcl >= 0.2."""
    values = np.asarray(vcl, dtype=float)
    coefficients = np.array(
        [
            [27.6374, -12.5232, 25.7510, -1.9403],
            [49.6431, -6.5168, 33.4577, -1.5849],
        ]
    )
    lower = (
        coefficients[0, 0] * np.exp(coefficients[0, 1] * values)
        + coefficients[0, 2] * np.exp(coefficients[0, 3] * values)
    )
    upper = (
        coefficients[1, 0] * np.exp(coefficients[1, 1] * values)
        + coefficients[1, 2] * np.exp(coefficients[1, 3] * values)
    )
    return lower, upper


def median_residual_friction(vcl: np.ndarray | float) -> np.ndarray:
    """Evaluate the median residual-friction angle used by PREDICT."""
    values = np.asarray(vcl, dtype=float)
    result = np.full(values.shape, 33.0, dtype=float)
    clay_model = values >= 0.2
    if np.any(clay_model):
        lower, upper = friction_bounds(values[clay_model])
        beta_median = beta.ppf(0.5, 3.0, 5.0)
        result[clay_model] = lower + beta_median * (upper - lower)
    return result


def cotd(angle_degrees: np.ndarray | float) -> np.ndarray:
    """Return cotangent for angles expressed in degrees."""
    return 1.0 / np.tan(np.deg2rad(angle_degrees))


def friction_factor(phi_sand: np.ndarray | float, phi_clay: np.ndarray | float) -> np.ndarray:
    """Return the dimensionless friction factor in the smear equation."""
    return cotd(45.0 + np.asarray(phi_clay)) - cotd(45.0 + np.asarray(phi_sand))


def style_axis(axis: plt.Axes) -> None:
    """Apply shared publication styling."""
    axis.grid(True, color=GRID, linewidth=0.9, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=12)
    for spine in axis.spines.values():
        spine.set_color("#606978")


def plot_general_controls(output_dir: Path) -> None:
    """Plot friction, source-thickness, and stretching-length controls."""
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.4))

    phi_sand = np.linspace(8.0, 40.0, 500)
    for clay_vcl, color in CLAY_COLORS.items():
        phi_clay = float(median_residual_friction(clay_vcl))
        factor = friction_factor(phi_sand, phi_clay)
        axes[0].plot(
            phi_sand,
            factor,
            color=color,
            linewidth=3.0,
            label=rf"Clay Vcl = {clay_vcl:.1f} ($\phi_c$ = {phi_clay:.1f} deg)",
        )
    phi_sand_low = float(median_residual_friction(0.1))
    phi_sand_high = float(median_residual_friction(0.3))
    axes[0].axvline(phi_sand_low, color="#272b34", linestyle="--", linewidth=2.0)
    axes[0].axvline(phi_sand_high, color="#272b34", linestyle=":", linewidth=2.5)
    axes[0].axhline(0.0, color="#626a76", linewidth=1.4)
    axes[0].text(phi_sand_low + 0.5, 0.505, "Sand Vcl 0.1\n$\\phi_s$ = 33.0 deg", fontsize=11)
    axes[0].text(phi_sand_high + 0.5, 0.505, "Sand Vcl 0.3\n$\\phi_s$ = 19.7 deg", fontsize=11)
    axes[0].set_xlim(8.0, 40.0)
    axes[0].set_ylim(-0.08, 0.60)
    axes[0].set_xlabel(r"Sand residual-friction angle, $\phi_s$ [deg]", fontsize=14)
    axes[0].set_ylabel(r"Friction factor, $\cot(\theta_c)-\cot(\theta_s)$", fontsize=14)
    axes[0].set_title("(a) Friction contrast", fontsize=16, weight="bold")
    axes[0].legend(fontsize=10.5, loc="lower right", frameon=True)
    style_axis(axes[0])

    thickness_ratio = np.linspace(0.2, 2.5, 400)
    axes[1].plot(thickness_ratio, thickness_ratio**2, color=BLUE, linewidth=3.5)
    axes[1].scatter([1.0, 2.0], [1.0, 4.0], s=90, color=ORANGE, edgecolor=INK, zorder=3)
    axes[1].annotate("baseline", (1.0, 1.0), xytext=(12, -18), textcoords="offset points", fontsize=12)
    axes[1].annotate("2x source thickness -> 4x smear thickness", (2.0, 4.0),
                     xytext=(-155, 20), textcoords="offset points", fontsize=11.5,
                     arrowprops={"arrowstyle": "->", "color": INK})
    axes[1].set_xlim(0.2, 2.5)
    axes[1].set_ylim(0.0, 6.5)
    axes[1].set_xlabel(r"Clay-source thickness ratio, $T_c/T_{c,0}$", fontsize=14)
    axes[1].set_ylabel("Smear-thickness ratio", fontsize=14)
    axes[1].set_title("(b) Quadratic source-thickness effect", fontsize=16, weight="bold")
    style_axis(axes[1])

    length_ratio = np.linspace(0.5, 3.0, 400)
    axes[2].plot(length_ratio, 1.0 / length_ratio, color=BLUE, linewidth=3.5)
    axes[2].scatter([1.0, 2.0], [1.0, 0.5], s=90, color=ORANGE, edgecolor=INK, zorder=3)
    axes[2].annotate("baseline", (1.0, 1.0), xytext=(12, 14), textcoords="offset points", fontsize=12)
    axes[2].annotate("2x stretching length -> 0.5x smear thickness", (2.0, 0.5),
                     xytext=(-145, -28), textcoords="offset points", fontsize=11.5,
                     arrowprops={"arrowstyle": "->", "color": INK})
    axes[2].set_xlim(0.5, 3.0)
    axes[2].set_ylim(0.25, 2.1)
    axes[2].set_xlabel(r"Current stretching-length ratio, $L_f/L_{f,0}$", fontsize=14)
    axes[2].set_ylabel("Smear-thickness ratio", fontsize=14)
    axes[2].set_title("(c) Inverse stretching-length effect", fontsize=16, weight="bold")
    style_axis(axes[2])

    figure.suptitle("Behavior of the PREDICT smear-thickness equation", fontsize=23, weight="bold", y=0.99)
    figure.text(
        0.5,
        0.925,
        r"$T_{\mathrm{smear}}=[\cot(\theta_c)-\cot(\theta_s)]\,T_c^2/L_f,$   "
        r"$\theta=45^{\circ}+\phi_r$",
        ha="center",
        fontsize=15,
        color="#3e4652",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.82, wspace=0.28)
    stem = output_dir / "smear_thickness_equation_general_controls"
    figure.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def plot_predict_vcl_response(output_dir: Path) -> list[dict[str, float]]:
    """Plot how PREDICT's friction model maps sand Vcl to smear thickness."""
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.4))

    below = np.linspace(0.02, 0.1999, 160)
    above = np.linspace(0.2, 0.65, 350)
    axes[0].plot(below, median_residual_friction(below), color=BLUE, linewidth=3.2)
    axes[0].plot(above, median_residual_friction(above), color=BLUE, linewidth=3.2)
    axes[0].plot([below[-1], above[0]],
                 [median_residual_friction(below[-1]), median_residual_friction(above[0])],
                 color=BLUE, linewidth=1.8, linestyle="--")
    for value, marker, color in ((0.1, "o", ORANGE), (0.3, "s", MAGENTA)):
        angle = float(median_residual_friction(value))
        axes[0].scatter(value, angle, s=110, marker=marker, color=color, edgecolor=INK, zorder=4)
        axes[0].annotate(f"Vcl {value:.1f}: {angle:.1f} deg", (value, angle),
                         xytext=(10, 9 if value == 0.1 else -23), textcoords="offset points", fontsize=11.5)
    axes[0].axvline(0.2, color="#626a76", linewidth=1.6, linestyle=":")
    axes[0].text(0.205, 34.7, "model branch changes", fontsize=10.5, color="#4b5360")
    axes[0].set_xlim(0.02, 0.65)
    axes[0].set_ylim(5.0, 39.0)
    axes[0].set_xlabel("Vcl", fontsize=14)
    axes[0].set_ylabel("Median residual-friction angle [deg]", fontsize=14)
    axes[0].set_title("(a) PREDICT friction parameterization", fontsize=16, weight="bold")
    style_axis(axes[0])

    sand_vcl = np.concatenate((np.linspace(0.02, 0.1999, 170), np.linspace(0.2, 0.39, 190)))
    rows: list[dict[str, float]] = []
    for clay_vcl, color in CLAY_COLORS.items():
        phi_clay = float(median_residual_friction(clay_vcl))
        factor = friction_factor(median_residual_friction(sand_vcl), phi_clay)
        baseline = float(friction_factor(median_residual_friction(0.1), phi_clay))
        relative = factor / baseline
        axes[1].plot(sand_vcl, relative, color=color, linewidth=3.0, label=f"Clay Vcl = {clay_vcl:.1f}")
        ratio = float(friction_factor(median_residual_friction(0.3), phi_clay) / baseline)
        rows.append(
            {
                "clay_vcl": clay_vcl,
                "median_phi_clay_deg": phi_clay,
                "median_phi_sand_vcl_0p1_deg": float(median_residual_friction(0.1)),
                "median_phi_sand_vcl_0p3_deg": float(median_residual_friction(0.3)),
                "friction_factor_sand_vcl_0p1": baseline,
                "friction_factor_sand_vcl_0p3": float(
                    friction_factor(median_residual_friction(0.3), phi_clay)
                ),
                "smear_thickness_ratio_0p3_over_0p1": ratio,
            }
        )
    axes[1].axvline(0.2, color="#626a76", linewidth=1.6, linestyle=":")
    axes[1].axvline(0.1, color="#8b929d", linewidth=1.1, linestyle="--")
    axes[1].axvline(0.3, color="#8b929d", linewidth=1.1, linestyle="--")
    axes[1].set_xlim(0.02, 0.39)
    axes[1].set_ylim(0.0, 1.12)
    axes[1].set_xlabel("Sand Vcl", fontsize=14)
    axes[1].set_ylabel("Relative smear thickness (Vcl 0.1 = 1)", fontsize=14)
    axes[1].set_title("(b) Friction-driven smear response", fontsize=16, weight="bold")
    axes[1].legend(fontsize=11.5, frameon=True)
    style_axis(axes[1])

    x = np.arange(len(rows))
    ratios = np.array([row["smear_thickness_ratio_0p3_over_0p1"] for row in rows])
    colors = [CLAY_COLORS[row["clay_vcl"]] for row in rows]
    axes[2].bar(x, ratios, width=0.62, color=colors, edgecolor=INK, linewidth=1.2)
    for xpos, ratio in zip(x, ratios):
        axes[2].text(xpos, ratio + 0.025, f"{100.0 * ratio:.0f}%", ha="center", fontsize=14, weight="bold")
    axes[2].set_xticks(x, [f"Clay Vcl\n{row['clay_vcl']:.1f}" for row in rows])
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel(r"$T_{\mathrm{smear}}(0.3)/T_{\mathrm{smear}}(0.1)$", fontsize=14)
    axes[2].set_title("(c) Sand Vcl 0.1 -> 0.3", fontsize=16, weight="bold")
    axes[2].axhline(1.0, color="#626a76", linewidth=1.6, linestyle="--")
    style_axis(axes[2])

    figure.suptitle("How sand Vcl changes modeled smear thickness in PREDICT", fontsize=23, weight="bold", y=0.99)
    figure.text(
        0.5,
        0.925,
        "Median friction angles are shown; T_clay and Lf are held fixed to isolate the friction-factor effect.",
        ha="center",
        fontsize=14,
        color="#3e4652",
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.82, wspace=0.28)
    stem = output_dir / "predict_sand_vcl_smear_thickness_response"
    figure.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return rows


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    """Write the representative parameter values plotted in the figures."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.9f}" for key, value in row.items()})


def main() -> None:
    """Generate both explanatory figures and their numerical summary."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_general_controls(output_dir)
    summary = plot_predict_vcl_response(output_dir)
    write_summary(output_dir / "predict_sand_vcl_smear_thickness_values.csv", summary)
    print(f"Wrote smear-thickness equation figures to {output_dir}")


if __name__ == "__main__":
    main()
