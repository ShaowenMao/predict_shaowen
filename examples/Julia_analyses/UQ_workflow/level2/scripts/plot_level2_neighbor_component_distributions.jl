#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie
using Printf

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

const STATE_LABELS = Dict(
    "low" => "Low",
    "central" => "Central",
    "high" => "High",
)
const NEIGHBORHOOD_LABELS = Dict(
    "small" => "Small-neighbor",
    "large" => "Large-neighbor",
)
const NEIGHBORHOOD_DESCRIPTIONS = Dict(
    "small" => "nearest 10% within each state library",
    "large" => "nearest 35% within each state library",
)
const COMPONENT_AXIS_LABELS = (
    "log10(kxx [mD])",
    "log10(kyy [mD])",
    "log10(kzz [mD])",
)
const FIXED_Y_LIMS = (-7.1, 2.1)
const FIXED_Y_TICKS = [-7.0, -4.0, -1.0, 2.0]
const FIXED_Y_TICK_LABELS = ["-7", "-4", "-1", "2"]

panel_label(index::Integer) = "($(Char(Int('a') + index - 1)))"

function parse_args(args::Vector{String})
    options = Dict(
        "state-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref")),
        "output-dir" => "",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_neighbor_component_distributions.jl [options]")
    println()
    println("Options:")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where neighborhood component violin figures are saved")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "neighbor_component_distributions") :
                  normpath(opt["output-dir"])
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()

    states = load_states(state_root)

    for neighborhood in ("small", "large")
        combined_fig = build_all_windows_neighbor_component_grid(states, neighborhood)
        save(joinpath(output_root, "all_windows_$(neighborhood)_neighbor_component_violins_medoids.png"), combined_fig)
        save_optional_pdf(joinpath(output_root, "all_windows_$(neighborhood)_neighbor_component_violins_medoids.pdf"), combined_fig)

        for (widx, window) in enumerate(Level2IO.FIXED_WINDOWS)
            fig = build_window_neighbor_component_figure(states[window], widx, neighborhood)
            save(joinpath(output_root, "$(window)_$(neighborhood)_neighbor_component_violins_medoids.png"), fig)
            save_optional_pdf(joinpath(output_root, "$(window)_$(neighborhood)_neighbor_component_violins_medoids.pdf"), fig)
        end
    end

    println("Saved Step 2.7 neighborhood component figures to $output_root")
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

function build_all_windows_neighbor_component_grid(states::Dict{String, Dict{String, Any}},
                                                   neighborhood::AbstractString)
    windows = Level2IO.FIXED_WINDOWS
    title_font_size = 42
    header_font_size = 34
    axis_font_size = 34
    tick_font_size = 34
    panel_size = 430
    title_label = NEIGHBORHOOD_LABELS[neighborhood]

    fig = Figure(size = (3900, 2180),
                 figure_padding = (24, 24, 18, 18),
                 backgroundcolor = :white)
    rowgap!(fig.layout, 14)
    colgap!(fig.layout, 18)

    Label(fig[1, 1:6],
          "$title_label component distributions and medoids by window",
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
                      xticks = (1:3, [STATE_LABELS[label] for label in Level2Plotting.STATE_ORDER]),
                      yticks = (FIXED_Y_TICKS, FIXED_Y_TICK_LABELS),
                      aspect = AxisAspect(1))

            add_neighbor_component_violins!(ax, state, component_idx, neighborhood;
                                            violin_width = 0.72,
                                            medoid_markersize = 24)
            xlims!(ax, 0.4, 3.6)
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

    legend_elements, legend_labels = build_neighbor_distribution_legend()
    Legend(fig[8, 1:6], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 12,
           rowgap = 8,
           colgap = 26,
           tellwidth = false,
           labelsize = header_font_size)

    Label(fig[9, 1:6],
          "Violins show only $(NEIGHBORHOOD_DESCRIPTIONS[neighborhood]); gold diamonds show the state medoid.",
          fontsize = axis_font_size,
          halign = :center,
          tellwidth = false)

    rowsize!(fig.layout, 1, Fixed(70))
    rowsize!(fig.layout, 2, Fixed(88))
    for row in component_rows
        rowsize!(fig.layout, row, Fixed(panel_size))
    end
    rowsize!(fig.layout, 4, Fixed(40))
    rowsize!(fig.layout, 6, Fixed(40))
    rowsize!(fig.layout, 8, Fixed(70))
    rowsize!(fig.layout, 9, Fixed(50))
    for col in 1:6
        colsize!(fig.layout, col, Fixed(panel_size))
    end
    resize_to_layout!(fig)

    return fig
end

function build_window_neighbor_component_figure(state::Dict{String, Any},
                                                window_index::Int,
                                                neighborhood::AbstractString)
    window = String(state["window"])
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = Level2Plotting.float_scalar(state["best_silhouette"])
    title_label = NEIGHBORHOOD_LABELS[neighborhood]

    title_font_size = 34
    axis_font_size = 26
    tick_font_size = 24

    fig = Figure(size = (1900, 760),
                 figure_padding = (22, 22, 16, 16),
                 backgroundcolor = :white)
    rowgap!(fig.layout, 12)
    colgap!(fig.layout, 22)

    Label(fig[1, 1:3],
          @sprintf("W%d (%s) %s component distributions and medoids | K = %d | silhouette = %.3f",
                   window_index, window, lowercase(title_label), chosen_k, silhouette),
          fontsize = title_font_size,
          font = :bold,
          halign = :center,
          tellwidth = false)

    for component_idx in 1:3
        ax = Axis(fig[2, component_idx],
                  title = COMPONENT_AXIS_LABELS[component_idx],
                  xlabel = "State library",
                  ylabel = component_idx == 1 ? "log10 permeability" : "",
                  titlesize = axis_font_size,
                  xlabelsize = axis_font_size,
                  ylabelsize = axis_font_size,
                  xticklabelsize = tick_font_size,
                  yticklabelsize = tick_font_size,
                  xgridvisible = false,
                  ygridvisible = true,
                  ygridcolor = RGBf(0.88, 0.88, 0.88),
                  ygridwidth = 1.0,
                  topspinevisible = false,
                  rightspinevisible = false,
                  xticks = (1:3, [STATE_LABELS[label] for label in Level2Plotting.STATE_ORDER]),
                  yticks = (FIXED_Y_TICKS, FIXED_Y_TICK_LABELS),
                  aspect = AxisAspect(1))

        add_neighbor_component_violins!(ax, state, component_idx, neighborhood;
                                        violin_width = 0.72,
                                        medoid_markersize = 22)
        xlims!(ax, 0.4, 3.6)
        ylims!(ax, FIXED_Y_LIMS...)
        add_panel_label!(ax, panel_label(component_idx); fontsize = axis_font_size)
        if component_idx != 1
            hideydecorations!(ax, grid = false)
        end
    end

    legend_elements, legend_labels = build_neighbor_distribution_legend()
    Legend(fig[3, 1:3], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 10,
           rowgap = 8,
           colgap = 22,
           tellwidth = false,
           labelsize = tick_font_size)

    rowsize!(fig.layout, 1, Fixed(66))
    rowsize!(fig.layout, 2, Fixed(430))
    rowsize!(fig.layout, 3, Fixed(70))
    for col in 1:3
        colsize!(fig.layout, col, Fixed(430))
    end
    resize_to_layout!(fig)

    return fig
end

function add_neighbor_component_violins!(ax,
                                         state::Dict{String, Any},
                                         component_idx::Int,
                                         neighborhood::AbstractString;
                                         violin_width::Real,
                                         medoid_markersize::Real)
    for (state_idx, label) in enumerate(Level2Plotting.STATE_ORDER)
        values = component_neighbor_values(state, label, neighborhood, component_idx)
        xs = fill(Float64(state_idx), length(values))
        violin!(ax, xs, values;
                color = (Level2Plotting.STATE_COLORS[label], 0.72),
                width = violin_width,
                strokecolor = :black,
                strokewidth = 1.0)

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

function component_neighbor_values(state::Dict{String, Any},
                                   label::AbstractString,
                                   neighborhood::AbstractString,
                                   component_idx::Int)
    log_perms = Level2Plotting.matrix_float(state["log_perms"])
    indices = Level2Plotting.vector_int(state["$(label)_$(neighborhood)_neighbors"])
    return vec(log_perms[indices, component_idx])
end

function build_neighbor_distribution_legend()
    elements = CairoMakie.LegendElement[]
    labels = String[]
    for label in Level2Plotting.STATE_ORDER
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
