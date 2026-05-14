#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")))

using CairoMakie
using Printf

include(joinpath(@__DIR__, "..", "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

const STATE_LABELS = Dict(
    "low" => "Low",
    "high" => "High",
)
const STATE_ORDER = ("low", "high")
const COMPONENT_AXIS_LABELS = (
    "log10(kxx [mD])",
    "log10(kyy [mD])",
    "log10(kzz [mD])",
)
const FIXED_Y_LIMS = (-7.1, 2.1)
const FIXED_Y_TICKS = [-7.0, -4.0, -1.0, 2.0]
const FIXED_Y_TICK_LABELS = ["-7", "-4", "-1", "2"]
const PANEL_WIDTH = 430
const PANEL_ASPECT = 3 / 2
const PANEL_HEIGHT = round(Int, PANEL_WIDTH / PANEL_ASPECT)

panel_label(index::Integer) = "($(Char(Int('a') + index - 1)))"

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "state-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref")),
        "output-dir" => "",
        "fixed-count-density-reference" => "",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_help()
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(options, key) || error("Unknown option $arg")
            i < length(args) || error("Missing value for $arg")
            options[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return options
end

function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/04_review_state_libraries/plot_state_library_distributions.jl [options]")
    println()
    println("Options:")
    println("  --config <path>        Level 2 TOML config with [plotting] defaults")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where the default state component violin figure is saved")
    println("  --fixed-count-density-reference <value>")
    println("                         Override config fixed density*count reference for cross-panel violin width scaling")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "state_component_distributions") :
                  normpath(opt["output-dir"])
    fixed_count_density_reference = isempty(opt["fixed-count-density-reference"]) ?
        Float64(config["state_violin_fixed_count_density_reference"]) :
        parse(Float64, opt["fixed-count-density-reference"])
    fixed_count_density_reference > 0 || error("--fixed-count-density-reference must be positive")
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()

    states = load_states(state_root)
    fig = build_state_component_violin_grid(states; fixed_count_density_reference = fixed_count_density_reference)

    png_path = joinpath(output_root, "all_windows_state_component_violins.png")
    pdf_path = joinpath(output_root, "all_windows_state_component_violins.pdf")
    save(png_path, fig)
    save_optional_pdf(pdf_path, fig)

    println("Saved default state component violin figure:")
    println("  $png_path")
end

function save_optional_pdf(path::AbstractString, fig::Figure)
    try
        save(path, fig)
    catch err
        @warn "Skipping PDF export because the file is locked or unavailable" path exception = (err, catch_backtrace())
    end
end

function load_states(state_root::AbstractString)
    states = Dict{String, Dict{String, Any}}()
    for window in Level2IO.FIXED_WINDOWS
        state_path = joinpath(state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing Level 2 state MAT file: $state_path")
        states[window] = Level2IO.load_window_state(state_path)
    end
    return states
end

function build_state_component_violin_grid(states::Dict{String, Dict{String, Any}};
                                           fixed_count_density_reference::Real)
    windows = Level2IO.FIXED_WINDOWS
    title_font_size = 42
    header_font_size = 34
    axis_font_size = 34
    tick_font_size = 34

    fig = Figure(size = (3900, 2180),
                 figure_padding = (24, 24, 18, 18),
                 backgroundcolor = :white)
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 18)

    Label(fig[1, 1:6],
          "Fixed-scale low and high state component distributions by window",
          fontsize = title_font_size,
          font = :bold,
          halign = :center,
          tellwidth = false)

    for (widx, window) in enumerate(windows)
        state = states[window]
        chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
        silhouette = Level2Plotting.float_scalar(state["best_silhouette"])
        Label(fig[2, widx],
              @sprintf("W%d\nK = %d, sil = %.3f", widx, chosen_k, silhouette),
              fontsize = header_font_size,
              font = :bold,
              halign = :center,
              tellwidth = false)
    end

    component_rows = (3, 5, 7)
    for component_idx in 1:3
        row = component_rows[component_idx]
        for (widx, window) in enumerate(windows)
            state = states[window]
            ax = Axis(fig[row, widx],
                      xlabel = component_idx == 3 ? "State library" : "",
                      ylabel = widx == 1 ? COMPONENT_AXIS_LABELS[component_idx] : "",
                      xlabelsize = axis_font_size,
                      ylabelsize = axis_font_size,
                      xticklabelsize = tick_font_size,
                      yticklabelsize = tick_font_size,
                      xlabelpadding = 8.0,
                      ylabelpadding = 12.0,
                      xgridvisible = false,
                      ygridvisible = true,
                      ygridcolor = RGBf(0.88, 0.88, 0.88),
                      ygridwidth = 1.0,
                      topspinevisible = false,
                      rightspinevisible = false,
                      xticks = (1:2, [STATE_LABELS[label] for label in STATE_ORDER]),
                      yticks = (FIXED_Y_TICKS, FIXED_Y_TICK_LABELS),
                      aspect = AxisAspect(PANEL_ASPECT))

            add_fixed_scale_violins!(ax, state, component_idx;
                                     violin_width = 0.72,
                                     fixed_count_density_reference = fixed_count_density_reference,
                                     medoid_markersize = 24)
            xlims!(ax, 0.4, 2.6)
            ylims!(ax, FIXED_Y_LIMS...)
            add_panel_label!(ax, panel_label((component_idx - 1) * length(windows) + widx);
                             fontsize = axis_font_size)

            if component_idx != 3
                hidexdecorations!(ax, grid = false)
            end
            if widx != 1
                hideydecorations!(ax, grid = false)
            end
        end
    end

    legend_elements, legend_labels = build_legend()
    Legend(fig[8, 1:6], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 12,
           rowgap = 8,
           colgap = 26,
           tellwidth = false,
           labelsize = header_font_size)

    Label(fig[9, 1:6],
          @sprintf("All panels use fixed density-count reference = %.0f, so violin widths are comparable across windows and geologies.",
                   fixed_count_density_reference),
          fontsize = axis_font_size,
          halign = :center,
          tellwidth = false)

    rowsize!(fig.layout, 1, Fixed(70))
    rowsize!(fig.layout, 2, Fixed(88))
    for row in component_rows
        rowsize!(fig.layout, row, Fixed(PANEL_HEIGHT))
    end
    rowsize!(fig.layout, 4, Fixed(13))
    rowsize!(fig.layout, 6, Fixed(13))
    rowsize!(fig.layout, 8, Fixed(70))
    rowsize!(fig.layout, 9, Fixed(50))
    for col in 1:6
        colsize!(fig.layout, col, Fixed(PANEL_WIDTH))
    end
    resize_to_layout!(fig)

    return fig
end

function add_fixed_scale_violins!(ax,
                                  state::Dict{String, Any},
                                  component_idx::Int;
                                  violin_width::Real,
                                  fixed_count_density_reference::Real,
                                  medoid_markersize::Real)
    all_x = Float64[]
    all_y = Float64[]
    all_colors = RGBAf[]

    for (state_idx, label) in enumerate(STATE_ORDER)
        values = component_state_values(state, label, component_idx)
        base_color = Level2Plotting.STATE_COLORS[label]
        state_color = RGBAf(base_color.r, base_color.g, base_color.b, 0.72)
        append!(all_x, fill(Float64(state_idx), length(values)))
        append!(all_y, values)
        append!(all_colors, fill(state_color, length(values)))
    end

    violin!(ax, all_x, all_y;
            color = all_colors,
            width = violin_width,
            scale = :count,
            max_density = fixed_count_density_reference,
            strokecolor = :black,
            strokewidth = 1.0)

    for (state_idx, label) in enumerate(STATE_ORDER)
        medoid_value = Level2Plotting.medoid_component_values(state, label)[component_idx]
        scatter!(ax,
                 [state_idx],
                 [medoid_value];
                 marker = :diamond,
                 color = RGBf(1.0, 0.78, 0.10),
                 strokecolor = :black,
                 strokewidth = 2.2,
                 markersize = medoid_markersize)
    end
end

function component_state_values(state::Dict{String, Any},
                                label::AbstractString,
                                component_idx::Int)
    log_perms = Level2Plotting.matrix_float(state["log_perms"])
    indices = Level2Plotting.vector_int(state["$(label)_indices"])
    return vec(log_perms[indices, component_idx])
end

function build_legend()
    elements = CairoMakie.LegendElement[]
    labels = String[]
    for label in STATE_ORDER
        push!(elements,
              PolyElement(color = (Level2Plotting.STATE_COLORS[label], 0.72),
                          strokecolor = :black,
                          strokewidth = 1.0))
        push!(labels, STATE_LABELS[label])
    end
    push!(elements,
          MarkerElement(color = RGBf(1.0, 0.78, 0.10),
                        marker = :diamond,
                        markersize = 22,
                        strokecolor = :black,
                        strokewidth = 2.2))
    push!(labels, "State medoid")
    return elements, labels
end

function add_panel_label!(ax, text_label::AbstractString; fontsize::Real)
    text!(ax, 0.96, 0.96;
          space = :relative,
          align = (:right, :top),
          fontsize = fontsize,
          font = :bold,
          color = :black,
          text = text_label)
end

main(ARGS)
