"""Inward ticks, expanded header spacing and camera-aligned local axes."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image
import pyvista as pv

import data as V

OUT = None
CMAP = LinearSegmentedColormap.from_list("shared_gas_saturation", V.COLORS, N=1024)
GAS, WATER = "#d85b00", "#0079af"
DPI = 240
FRAME_SIZE = (3840, 2160)
PC_RECT = [.635, .545, .33, .265]
KR_RECT = [.635, .160, .33, .265]
STATE_SECONDS = .1
START_HOLD_SECONDS, END_HOLD_SECONDS = 1, 2
VIDEO_FPS = 30
TEXT_PT = 20
PADDED_UNIT_LIMITS = (-.025, 1.025)
MATERIAL_TITLE_Y = .823
PREVIOUS_HEADER_GAP = .865 - MATERIAL_TITLE_Y
HEADER_GAP = 1.5 * PREVIOUS_HEADER_GAP
HEADER_Y = MATERIAL_TITLE_Y + HEADER_GAP



def fixed_header(fig):
    """Reserve numeric slots once so prefixes, units and decimal points stay fixed."""
    specifications = [
        ("pressure_label", r"Applied gas--brine $\Delta p =$"),
        ("pressure_value", r"$888.888$"),
        ("pressure_unit", r"$\mathrm{bar}$"),
        ("saturation_label", r"$\bar{S}_g =$"),
        ("saturation_value", r"$0.000$"),
    ]
    parts = {name: fig.text(0, HEADER_Y, text, fontsize=TEXT_PT, color="black")
        for name, text in specifications}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [parts[name].get_window_extent(renderer).width for name, _ in specifications]
    gaps = np.asarray([4, 4, 18, 4]) * DPI / 72
    total = sum(widths) + sum(gaps)
    cursor = .284 * FRAME_SIZE[0] - total / 2
    assert cursor >= .02 * FRAME_SIZE[0] and cursor + total <= .55 * FRAME_SIZE[0]
    for i, (name, _) in enumerate(specifications):
        right_aligned = name.endswith("_value")
        parts[name].set_ha("right" if right_aligned else "left")
        parts[name].set_x((cursor + widths[i] if right_aligned else cursor) / FRAME_SIZE[0])
        cursor += widths[i] + (gaps[i] if i < len(gaps) else 0)
    return parts


def add_coordinate_triad(fig, ax, grid):
    """Project fault-local axes through the same camera as the rendered core."""
    directions = V.local_axes_from_transformed_grid(grid)
    position, focus, view_up = map(np.asarray, V.camera_for_right_dipping_material(grid))
    view = focus - position
    view /= np.linalg.norm(view)
    right = np.cross(view, view_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)
    origin = np.array([.034, .245])
    figure_inches = np.array([fig.get_figwidth(), fig.get_figheight()])
    arrow_inches, label_gap_inches = .32, .09
    projected, labels = {}, []
    for name, direction in directions.items():
        screen = np.array([np.dot(direction, right), np.dot(direction, up)])
        screen /= np.linalg.norm(screen)
        projected[name] = screen.tolist()
        tip = origin + arrow_inches * screen / figure_inches
        ax.annotate("", xy=tip, xytext=origin, xycoords=fig.transFigure,
            arrowprops=dict(arrowstyle="->", color="black", lw=.9,
                mutation_scale=9, shrinkA=0, shrinkB=0), annotation_clip=False)
        label_position = tip + label_gap_inches * screen / figure_inches
        labels.append(fig.text(*label_position, "$" + name + "$", fontsize=TEXT_PT,
            color="black", ha="center", va="center"))
    np.testing.assert_allclose(np.column_stack(list(directions.values())).T
        @ np.column_stack(list(directions.values())), np.eye(3), atol=1e-10)
    assert projected["x"][0] > 0 and projected["x"][1] < 0
    assert projected["y"][0] > 0 and projected["y"][1] > 0
    assert abs(projected["z"][0]) < 1e-10 and projected["z"][1] > 0
    return dict(origin_figure=origin.tolist(), arrow_length_inches=arrow_inches,
        source="Transformed grid local axes projected with the core camera",
        projected_directions=projected), labels


def render(item, curve, sg, means, preview=False):
    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "font.size": TEXT_PT, "mathtext.fontset": "cm", "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelsize": TEXT_PT,
        "xtick.labelsize": TEXT_PT, "ytick.labelsize": TEXT_PT, "legend.fontsize": TEXT_PT,
        "axes.grid": False,
    })
    assert PC_RECT[2:] == KR_RECT[2:]
    assert to_hex(CMAP(0.0)) == V.COLORS[0]
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2100, 1640), border=False)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    surfaces = []
    for col, name in enumerate(("sand", "clay")):
        plotter.subplot(0, col)
        surface = item["selections"][name].extract_surface(algorithm="dataset_surface")
        surface.cell_data["Sg"] = sg[0, surface.cell_data["source_cell_id"]]
        plotter.add_mesh(surface, scalars="Sg", cmap=CMAP, clim=(0, 1), n_colors=1024,
            show_scalar_bar=False, lighting=False, show_edges=False)
        plotter.add_lines(V.outline(item["grid"]).reshape(-1, 3), color="#555b60", width=1.5)
        plotter.camera_position = V.camera_for_right_dipping_material(item["grid"])
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = V.camera_scale([item]) * 1.05
        surfaces.append(surface)
    plotter.show(auto_close=False, interactive=False)
    fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor="white")
    image_ax = fig.add_axes([.02, .18, .53, .68])
    picture = image_ax.imshow(plotter.screenshot(return_img=True), interpolation="none")
    image_ax.axis("off")
    triad, triad_labels = add_coordinate_triad(fig, image_ax, item["grid"])
    fig.text(.035, .907 + HEADER_GAP - PREVIOUS_HEADER_GAP,
        "Core %s / W%s, strike %s / %s" % (item["info"]["slot"], item["info"]["window"],
        item["info"]["strike_index"], item["info"]["case"].replace("_", r"\_")), fontsize=12.5, color="#596269")
    header = fixed_header(fig)
    header_reference = None
    header_max_drift = 0.0
    material_titles = [fig.text(x, MATERIAL_TITLE_Y, title, ha="center", fontsize=TEXT_PT, color="black")
        for x, title in zip((.153, .415), (r"$\mathrm{Sand}$", r"$\mathrm{Clay\ smear}$"))]
    material_labels = [fig.text(x, .191, "", ha="center", fontsize=TEXT_PT) for x in (.153, .415)]
    color_ax = fig.add_axes([.198, .137, .18, .016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
        cax=color_ax, orientation="horizontal")
    cb.set_ticks([0, 1], labels=[r"$0.0$", r"$1.0$"])
    cb.set_label(r"$S_g\,[-]$", fontsize=TEXT_PT)
    cb.outline.set_visible(False)

    pc = fig.add_axes(PC_RECT)
    pc.set(xlim=PADDED_UNIT_LIMITS, ylim=(0, max(270, float(curve[:, 0].max() / 1e5) * 1.12)), xlabel=r"$\bar{S}_g\,[-]$", ylabel=r"$P_c\,[\mathrm{bar}]$")
    pc.set_yscale("symlog", linthresh=.03)
    pc.set_yticks([0, .1, 1, 10, 100])
    pc.set_xticks(np.linspace(0, 1, 6))
    pc.xaxis.set_major_formatter(FormatStrFormatter(r"$%.2f$"))
    pc.yaxis.set_major_formatter(FormatStrFormatter(r"$%.2f$"))
    pc.grid(False, which="both", axis="both")
    pc.tick_params(axis="both", which="both", direction="in", pad=7)
    pc.plot(curve[:, 1], curve[:, 0]/1e5, color="black", alpha=.15, lw=1.8, ls="-")
    pc_line, = pc.plot([], [], color="black", lw=2.2, ls="-")
    pc_dot, = pc.plot([], [], "o", color="black", ms=7, mec="white", mew=.8)

    kr = fig.add_axes(KR_RECT)
    kr.set(xlim=PADDED_UNIT_LIMITS, ylim=PADDED_UNIT_LIMITS, xlabel=r"$\bar{S}_g\,[-]$", ylabel=r"$k_r\,[-]$")
    kr.set_xticks(np.linspace(0, 1, 6))
    kr.set_yticks([0, .25, .5, .75, 1])
    kr.xaxis.set_major_formatter(FormatStrFormatter(r"$%.2f$"))
    kr.yaxis.set_major_formatter(FormatStrFormatter(r"$%.2f$"))
    kr.grid(False, which="both", axis="both")
    kr.tick_params(axis="both", which="both", direction="in", pad=7)
    kr.plot(curve[:, 1], curve[:, 3], color=GAS, alpha=.18, lw=2, ls="-")
    kr.plot(curve[:, 1], curve[:, 2], color=WATER, alpha=.18, lw=2, ls="-")
    gas_line, = kr.plot([], [], color=GAS, lw=2.5, ls="-", label=r"$k_{rg}$")
    water_line, = kr.plot([], [], color=WATER, lw=2.5, ls="-", label=r"$k_{rw}$")
    gas_dot, = kr.plot([], [], "o", color=GAS, ms=7, mec="white", mew=.8)
    water_dot, = kr.plot([], [], "o", color=WATER, ms=7, mec="white", mew=.8)
    kr.legend(loc="center right", frameon=False, fontsize=TEXT_PT)
    fig.text(.035, .047,
        r"Increasing $\Delta p=p_g-p_w$, not elapsed time. Exact MIP states; original calculated $k_r$ values.",
        fontsize=10.3, color="#596269")
    fig.text(.035, .022,
        r"Gas enters at the bottom ($+z$ upward). Thickness exaggerated $22\times$. Exposed material surfaces; no new flow solves.",
        fontsize=10.3, color="#596269")

    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)
    samples = [0, 20, 45, 62, 80, 99] if preview else range(len(curve))
    for j in samples:
        for surface in surfaces:
            surface.cell_data["Sg"] = sg[j, surface.cell_data["source_cell_id"]]
        plotter.render()
        picture.set_data(plotter.screenshot(return_img=True))
        pressure, bulk, kw, kg = curve[j, :4]
        header["pressure_value"].set_text(rf"${pressure/1e5:.3f}$")
        header["saturation_value"].set_text(rf"${bulk:.3f}$")
        for label, mean, name in zip(material_labels, means[j], ("sand", "clay")):
            label.set_text(rf"$\bar{{S}}_{{g,\mathrm{{{name}}}}} = {mean:.3f}$")
        pc_line.set_data(curve[:j+1, 1], curve[:j+1, 0]/1e5)
        pc_dot.set_data([bulk], [pressure/1e5])
        gas_line.set_data(curve[:j+1, 1], curve[:j+1, 3])
        water_line.set_data(curve[:j+1, 1], curve[:j+1, 2])
        gas_dot.set_data([bulk], [kg])
        water_dot.set_data([bulk], [kw])
        fig.savefig(frames / f"state_{j:03d}.png", dpi=DPI, facecolor="white")
        renderer = fig.canvas.get_renderer()
        anchors = np.array([text.get_window_extent(renderer).x1 if name.endswith("_value")
            else text.get_window_extent(renderer).x0 for name, text in header.items()])
        if header_reference is None:
            header_reference = anchors
        header_max_drift = max(header_max_drift, float(np.max(np.abs(anchors - header_reference))))
        assert header_max_drift < 1e-6
        if j % 10 == 0 or preview:
            print("Rendered 4K state", j+1, flush=True)
    fig.canvas.draw()
    texts = [text for text in fig.findobj(Text) if text.get_text()]
    assert all(text.get_usetex() for text in texts)
    assert np.allclose(pc.get_window_extent().size, kr.get_window_extent().size)
    assert tuple(fig.canvas.get_width_height()) == FRAME_SIZE
    assert not pc.get_title() and not kr.get_title()
    assert not any(text.get_text().startswith("State ") for text in texts)
    assert len(kr.lines) == 6
    assert kr.get_ylim() == PADDED_UNIT_LIMITS
    assert kr.get_xlim() == pc.get_xlim() == PADDED_UNIT_LIMITS
    assert not any(line.get_visible() for ax in (pc, kr) for line in ax.get_xgridlines() + ax.get_ygridlines())
    main_texts = [text for text in texts if text.get_color() != "#596269"]
    assert all(text.get_fontsize() == TEXT_PT for text in main_texts)
    assert not any(text.get_text() == r"\textbf{Gas invasion and relative permeability}" for text in texts)
    np.testing.assert_allclose(kr.get_yticks(), [0, .25, .5, .75, 1])
    np.testing.assert_allclose(cb.get_ticks(), [0, 1])
    assert np.isclose(color_ax.get_position().width, .18)
    assert all(text.get_color() == "black" for text in material_titles)
    assert not any("k_{rg} =" in text.get_text() or "k_{rw} =" in text.get_text() for text in texts)
    tick_directions = {name: sorted({tick.get_tickdir() for axis in (ax.xaxis, ax.yaxis)
        for tick in axis.get_major_ticks() + axis.get_minor_ticks()})
        for name, ax in (("pc", pc), ("kr", kr))}
    assert tick_directions == {"pc": ["in"], "kr": ["in"]}
    assert np.isclose((HEADER_Y - MATERIAL_TITLE_Y) / PREVIOUS_HEADER_GAP, 1.5)
    triad_bounds = [label.get_window_extent(fig.canvas.get_renderer()).bounds for label in triad_labels]
    assert all(x > 0 and y > 0 and x + w < .075 * FRAME_SIZE[0]
        and y + h < .32 * FRAME_SIZE[1] for x, y, w, h in triad_bounds)
    triad["label_bounds_pixels"] = [list(bounds) for bounds in triad_bounds]
    style = dict(frame_size=list(FRAME_SIZE), native_3d_size=[2100, 1640], pc_above_kr=True,
        pc_axes_pixels=pc.get_window_extent().size.tolist(), kr_axes_pixels=kr.get_window_extent().size.tolist(),
        material_zero_saturation_colors={"sand": to_hex(CMAP(0.0)), "clay": to_hex(CMAP(0.0))},
        colorbar_count=1, color_scales="Shared original neutral-to-red saturation scale, 0 to 1",
        latex_all_text=True, font="Computer Modern", material_title_color="black",
        curve_titles=False, dynamic_kr_values=False, state_counter=False, moving_kr_vertical_guide=False,
        colorbar_width_fraction=color_ax.get_position().width,
        colorbar_tick_labels=[text.get_text() for text in color_ax.get_xticklabels()],
        kr_ylim=list(kr.get_ylim()), kr_yticks=kr.get_yticks().tolist(),
        kr_ytick_labels=[text.get_text() for text in kr.get_yticklabels()],
        header_center_x=.284, header_centered=True, fixed_header_slots=True,
        header_max_anchor_drift_pixels=header_max_drift,
        header_positions={name: list(text.get_position()) for name, text in header.items()},
        all_main_font_sizes_pt=sorted({text.get_fontsize() for text in main_texts}),
        gray_annotation_sizes_pt=sorted({text.get_fontsize() for text in texts if text.get_color() == "#596269"}),
        main_title=False, grid_lines=False, line_plot_tick_decimal_places=2,
        line_plot_tick_directions=tick_directions,
        header_to_material_row_gap=HEADER_Y - MATERIAL_TITLE_Y,
        previous_header_to_material_row_gap=PREVIOUS_HEADER_GAP,
        header_spacing_multiplier=1.5, coordinate_triad=triad,
        pc_xtick_labels=[text.get_text() for text in pc.get_xticklabels()],
        pc_ytick_labels=[text.get_text() for text in pc.get_yticklabels()],
        kr_xtick_labels=[text.get_text() for text in kr.get_xticklabels()],
        kr_xlim=list(kr.get_xlim()), pc_xlim=list(pc.get_xlim()),
        curve_axis_labels=[pc.get_xlabel(), pc.get_ylabel(), kr.get_xlabel(), kr.get_ylabel()],
        top_label_template=r"Applied gas--brine $\Delta p = xx\,\mathrm{bar}$\qquad $\bar{S}_g = xx$",
        pc_line="black solid", gas_kr_line="orange solid", brine_kr_line="blue solid",
        source_data_unchanged=True)
    (OUT / "style_checks.json").write_text(json.dumps(style, indent=2))
    plt.close(fig)
    plotter.close()
    if preview:
        sheet = Image.new("RGB", (1920, 1620), "white")
        for k, j in enumerate(samples):
            with Image.open(frames / f"state_{j:03d}.png") as img:
                sheet.paste(img.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS),
                    ((k % 2)*960, (k // 2)*540))
        sheet.save(OUT / "contact_sheet.png")


def retime_gif():
    """Edit graphic-control delays without re-quantizing any pixel or palette."""
    original = (OUT / "unpaused_encoding.gif").read_bytes()
    data = bytearray(original)
    assert data[:6] in (b"GIF87a", b"GIF89a")
    pos = 13 + (3 * 2**((data[10] & 7) + 1) if data[10] & 128 else 0)
    delay_offsets, images = [], 0

    def skip_blocks(offset):
        while data[offset]:
            offset += data[offset] + 1
        return offset + 1

    while data[pos] != 0x3B:
        if data[pos] == 0x21:
            if data[pos + 1] == 0xF9:
                assert data[pos + 2] == 4 and data[pos + 7] == 0
                delay_offsets.append(pos + 4)
            pos = skip_blocks(pos + 2)
        elif data[pos] == 0x2C:
            flags = data[pos + 9]
            pos += 10 + (3 * 2**((flags & 7) + 1) if flags & 128 else 0)
            pos = skip_blocks(pos + 1)
            images += 1
        else:
            raise ValueError(f"Unexpected GIF block at {pos}")
    assert images == len(delay_offsets) == 100
    delays = [round(STATE_SECONDS * 100)] * images
    delays[0] += START_HOLD_SECONDS * 100
    delays[-1] += END_HOLD_SECONDS * 100
    allowed = set()
    for offset, delay in zip(delay_offsets, delays):
        data[offset:offset + 2] = delay.to_bytes(2, "little")
        allowed.update((offset, offset + 1))
    assert all(a == b or i in allowed for i, (a, b) in enumerate(zip(original, data)))
    (OUT / "looping_preview.gif").write_bytes(data)


def encode():
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    inputs = [exe, "-y", "-hide_banner", "-loglevel", "warning", "-threads", "4",
        "-filter_threads", "4", "-framerate", "10",
        "-i", str(OUT / "frames/state_%03d.png")]
    subprocess.run(inputs + ["-vf",
        f"fps={VIDEO_FPS},tpad=start_duration={START_HOLD_SECONDS}:start_mode=clone:stop_duration={END_HOLD_SECONDS}:stop_mode=clone",
        "-c:v", "libx264", "-threads", "4", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(OUT / "synchronized_drainage_4k.mp4")], check=True)
    base_gif = OUT / "unpaused_encoding.gif"
    subprocess.run(inputs + ["-vf",
        "scale=1440:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
        "-loop", "0", str(base_gif)], check=True)
    retime_gif()
    print("4K movie and looping GIF exported: 13 seconds with endpoint holds", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "preview", "render", "encode", "retime"])
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--core", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    OUT = a.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "encode":
        encode()
    elif a.mode == "retime":
        retime_gif()
    else:
        checked = V.load_checked(a.bundle, a.core, OUT)
        if a.mode != "check":
            render(*checked, preview=a.mode == "preview")
