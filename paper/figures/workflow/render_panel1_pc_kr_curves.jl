using CairoMakie
using MAT
using CairoMakie.Makie.MathTeXEngine: @L_str

const DEFAULT_INPUT_FILE = raw"D:\codex_gom\UQ_workflow\representative_stratigraphy_schematic\predict_permeability_distributions\s02_c003\toc_components\source_data\reservoir_ready_s02_c003_case01.mat"
const INPUT_FILE = get(ENV, "PREDICT_WORKFLOW_PANEL1_SOURCE", DEFAULT_INPUT_FILE)
const OUTPUT_DIR = joinpath(@__DIR__, "assets")

const FIGURE_SIZE = (1400, 1760)
const PANEL_SIZE = (1400, 850)
const FONT_SIZE = 110
const LABEL_FONT_SIZE = 160
const AXIS_WIDTH = 4.0
const CURVE_WIDTH = 14.0
const GRID_COLOR = (:black, 0.12)

scalar_value(value) = value isa Number ? value : only(vec(value))

function curve_data()
    reservoir_ready = matread(INPUT_FILE)["reservoirReady"]
    pc = reservoir_ready["pcCurves"][4, 67]
    kr = reservoir_ready["krCurves"][4, 67]

    pc_source = Int(round(scalar_value(pc["replaySourceRow"])))
    kr_source = Int(round(scalar_value(kr["representativeReplaySourceRow"])))
    pc_source == kr_source || error("Pc/Kr replay-source mismatch: $(pc_source) != $(kr_source)")

    pc_sg = vec(Float64.(pc["gasSaturation"]))
    pc_bar = max.(vec(Float64.(pc["pcBar"])), 1.0e-2)
    kr_sg = vec(Float64.(kr["gasSaturation"]))
    krg = vec(Float64.(kr["krg"]))
    krw = vec(Float64.(kr["krw"]))

    return (; pc_sg, pc_bar, kr_sg, krg, krw, replay_source=pc_source)
end

function style_axis!(ax)
    ax.xticklabelsize = FONT_SIZE
    ax.yticklabelsize = FONT_SIZE
    ax.yticklabelspace = 190.0
    ax.xlabelsize = FONT_SIZE
    ax.ylabelsize = FONT_SIZE
    ax.xgridcolor = GRID_COLOR
    ax.ygridcolor = GRID_COLOR
    ax.xgridwidth = 2.0
    ax.ygridwidth = 2.0
    ax.spinewidth = AXIS_WIDTH
    ax.xtickwidth = AXIS_WIDTH
    ax.ytickwidth = AXIS_WIDTH
    ax.xticksize = 16
    ax.yticksize = 16
end

function add_pc_axis!(slot, data)
    ax = Axis(
        slot;
        xlabel=L"S_g\,[-]",
        ylabel=L"P_c\,[\mathrm{bar}]",
        limits=((0.0, 1.0), (1.0e-2, 1.0e3)),
        yscale=log10,
        xticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1.0"]),
        yticks=([1.0e-2, 1.0, 1.0e2], [L"10^{-2}", L"10^{0}", L"10^{2}"]),
    )
    style_axis!(ax)
    lines!(ax, data.pc_sg, data.pc_bar; color="#C44536", linewidth=CURVE_WIDTH)
    return ax
end

function add_kr_axis!(slot, data)
    ax = Axis(
        slot;
        xlabel=L"S_g\,[-]",
        ylabel=L"k_r\,[-]",
        limits=((0.0, 1.0), (0.0, 1.0)),
        xticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1.0"]),
        yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1.0"]),
    )
    style_axis!(ax)
    lines!(ax, data.kr_sg, data.krw; color="#2364AA", linewidth=CURVE_WIDTH)
    lines!(ax, data.kr_sg, data.krg; color="#E07A20", linewidth=CURVE_WIDTH)
    text!(ax, 0.15, 0.60; text=L"k_{rw}", color="#2364AA", fontsize=LABEL_FONT_SIZE)
    text!(ax, 0.77, 0.60; text=L"k_{rg}", color="#E07A20", fontsize=LABEL_FONT_SIZE)
    return ax
end

function save_panel(path_stem, add_axis, data)
    fig = Figure(size=PANEL_SIZE, backgroundcolor=:white, figure_padding=(65, 115, 45, 80))
    add_axis(fig[1, 1], data)
    save(path_stem * ".png", fig; px_per_unit=2)
    save(path_stem * ".pdf", fig)
end

function main()
    mkpath(OUTPUT_DIR)
    data = curve_data()

    pc_stem = joinpath(OUTPUT_DIR, "panel1_w4_upscaled_pc_curve")
    kr_stem = joinpath(OUTPUT_DIR, "panel1_w4_upscaled_kr_curves")
    save_panel(pc_stem, add_pc_axis!, data)
    save_panel(kr_stem, add_kr_axis!, data)

    fig = Figure(size=FIGURE_SIZE, backgroundcolor=:white, figure_padding=(65, 115, 50, 80))
    add_pc_axis!(fig[1, 1], data)
    add_kr_axis!(fig[2, 1], data)
    rowgap!(fig.layout, 85)

    stacked_stem = joinpath(OUTPUT_DIR, "pc_kr_curves")
    save(stacked_stem * ".png", fig; px_per_unit=2)
    save(stacked_stem * ".pdf", fig)

    println("Replay source row: ", data.replay_source)
    println("Saved: ", stacked_stem, ".pdf")
end

main()
