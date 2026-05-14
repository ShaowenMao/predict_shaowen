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
const POOL_LABELS = Dict(
    "local" => "Local perturbation pool",
    "state_wide" => "State-wide perturbation pool",
)
const POOL_DESCRIPTIONS = Dict(
    "local" => "medoid-centered samples within the medoid-cluster part of each state",
    "state_wide" => "all samples in each full low/high state library",
)
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/05_review_perturbation_pools/plot_perturbation_pool_distributions.jl [options]")
    println()
    println("Options:")
    println("  --config <path>        Level 2 TOML config with [plotting] defaults")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where perturbation-pool component violin figures are saved")
    println("  --fixed-count-density-reference <value>")
    println("                         Fixed density*count reference for cross-panel violin width scaling")
    println("                         Overrides the config default for both local and state-wide figures")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "perturbation_pool_distributions") :
                  normpath(opt["output-dir"])
    fixed_count_density_reference_override = isempty(opt["fixed-count-density-reference"]) ?
        NaN :
        parse(Float64, opt["fixed-count-density-reference"])
    isnan(fixed_count_density_reference_override) || fixed_count_density_reference_override > 0 ||
        error("--fixed-count-density-reference must be positive")
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()

    states = load_states(state_root)

    for pool in ("local", "state_wide")
        fixed_count_density_reference = isnan(fixed_count_density_reference_override) ?
            default_fixed_count_density_reference(pool, config) :
            fixed_count_density_reference_override
        combined_fig = build_all_windows_perturbation_pool_component_grid(
            states,
            pool;
            fixed_count_density_reference = fixed_count_density_reference,
        )
        save(joinpath(output_root, "all_windows_$(pool)_perturbation_pool_component_violins_low_high_medoids.png"), combined_fig)
        save_optional_pdf(joinpath(output_root, "all_windows_$(pool)_perturbation_pool_component_violins_low_high_medoids.pdf"), combined_fig)

        for (widx, window) in enumerate(Level2IO.FIXED_WINDOWS)
            fig = build_window_perturbation_pool_component_figure(
                states[window],
                widx,
                pool;
                fixed_count_density_reference = fixed_count_density_reference,
            )
            save(joinpath(output_root, "$(window)_$(pool)_perturbation_pool_component_violins_low_high_medoids.png"), fig)
            save_optional_pdf(joinpath(output_root, "$(window)_$(pool)_perturbation_pool_component_violins_low_high_medoids.pdf"), fig)
        end
    end

    println("Saved Step 2.7 perturbation-pool component figures to $output_root")
end

function default_fixed_count_density_reference(pool::AbstractString,
                                               config::Dict{String, Any})
    if pool == "local"
        return Float64(config["local_pool_violin_fixed_count_density_reference"])
    elseif pool == "state_wide"
        return Float64(config["state_wide_pool_violin_fixed_count_density_reference"])
    end
    error("Unknown perturbation pool: $pool")
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

function build_all_windows_perturbation_pool_component_grid(states::Dict{String, Dict{String, Any}},
                                                            pool::AbstractString;
                                                            fixed_count_density_reference::Real)
    windows = Level2IO.FIXED_WINDOWS
    title_font_size = 42
    header_font_size = 34
    axis_font_size = 34
    tick_font_size = 34
    title_label = POOL_LABELS[pool]

    fig = Figure(size = (3900, 2180),
                 figure_padding = (24, 24, 18, 18),
                 backgroundcolor = :white)
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 18)

    Label(fig[1, 1:6],
          "Fixed-scale low and high $(lowercase(title_label)) distributions by window",
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

            add_fixed_scale_pool_violins!(ax, state, component_idx, pool;
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

    legend_elements, legend_labels = build_pool_distribution_legend()
    Legend(fig[8, 1:6], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 12,
           rowgap = 8,
           colgap = 26,
           tellwidth = false,
           labelsize = header_font_size)

    Label(fig[9, 1:6],
          @sprintf("Violins show %s; gold diamonds show state medoids. Fixed density-count reference = %.0f.",
                   POOL_DESCRIPTIONS[pool],
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

function build_window_perturbation_pool_component_figure(state::Dict{String, Any},
                                                         window_index::Int,
                                                         pool::AbstractString;
                                                         fixed_count_density_reference::Real)
    window = String(state["window"])
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = Level2Plotting.float_scalar(state["best_silhouette"])
    title_label = POOL_LABELS[pool]

    title_font_size = 34
    axis_font_size = 26
    tick_font_size = 24

    fig = Figure(size = (1900, 640),
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
                  xticks = (1:2, [STATE_LABELS[label] for label in STATE_ORDER]),
                  yticks = (FIXED_Y_TICKS, FIXED_Y_TICK_LABELS),
                  aspect = AxisAspect(PANEL_ASPECT))

        add_fixed_scale_pool_violins!(ax, state, component_idx, pool;
                                      violin_width = 0.72,
                                      fixed_count_density_reference = fixed_count_density_reference,
                                      medoid_markersize = 22)
        xlims!(ax, 0.4, 2.6)
        ylims!(ax, FIXED_Y_LIMS...)
        add_panel_label!(ax, panel_label(component_idx); fontsize = axis_font_size)
        if component_idx != 1
            hideydecorations!(ax, grid = false)
        end
    end

    legend_elements, legend_labels = build_pool_distribution_legend()
    Legend(fig[3, 1:3], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 10,
           rowgap = 8,
           colgap = 22,
           tellwidth = false,
           labelsize = tick_font_size)

    rowsize!(fig.layout, 1, Fixed(66))
    rowsize!(fig.layout, 2, Fixed(PANEL_HEIGHT))
    rowsize!(fig.layout, 3, Fixed(70))
    for col in 1:3
        colsize!(fig.layout, col, Fixed(PANEL_WIDTH))
    end
    resize_to_layout!(fig)

    return fig
end

function add_fixed_scale_pool_violins!(ax,
                                       state::Dict{String, Any},
                                       component_idx::Int,
                                       pool::AbstractString;
                                       violin_width::Real,
                                       fixed_count_density_reference::Real,
                                       medoid_markersize::Real)
    all_x = Float64[]
    all_y = Float64[]
    all_colors = RGBAf[]

    for (state_idx, label) in enumerate(STATE_ORDER)
        values = component_pool_values(state, label, pool, component_idx)
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

function component_pool_values(state::Dict{String, Any},
                               label::AbstractString,
                               pool::AbstractString,
                               component_idx::Int)
    log_perms = Level2Plotting.matrix_float(state["log_perms"])
    indices = Level2Plotting.vector_int(state["$(label)_$(pool)_pool"])
    return vec(log_perms[indices, component_idx])
end

function build_pool_distribution_legend()
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
