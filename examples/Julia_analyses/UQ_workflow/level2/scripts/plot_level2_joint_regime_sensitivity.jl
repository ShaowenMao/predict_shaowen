#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_core.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Core
using .Level2Plotting

const SENSITIVITY_FRACTIONS = [0.05, 0.10, 0.15]
const BASELINE_FRACTION = 0.10
const PAIRWISE_COMPONENTS = ((1, 2), (1, 3), (2, 3))
const PAIRWISE_TITLES = ("log10(kxx) vs log10(kyy)",
                         "log10(kxx) vs log10(kzz)",
                         "log10(kyy) vs log10(kzz)")
const PAIRWISE_AXIS_LABELS = ("log10(kxx [mD])",
                              "log10(kyy [mD])",
                              "log10(kzz [mD])")
const PANEL_LABELS = ("(a)", "(b)", "(c)")
const FIXED_TICKS = [-7.0, -5.0, -3.0, -1.0, 1.0]
const FIXED_TICK_LABELS = ["-7", "-5", "-3", "-1", "1"]
const FIXED_LIMS = (-7.1, 1.1)

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "manifest" => Level2IO.default_manifest_path(),
        "output-dir" => "",
        "fractions" => join(string.(SENSITIVITY_FRACTIONS), ","),
        "baseline" => string(BASELINE_FRACTION),
        "max-points" => "1400",
        "seed" => "1729",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_joint_regime_sensitivity.jl [options]")
    println()
    println("Options:")
    println("  --config <path>       Level 2 TOML config")
    println("  --manifest <path>     Proxy manifest CSV")
    println("  --output-dir <path>   Output folder for sensitivity figures and tables")
    println("  --fractions <csv>     Min-cluster fractions to test, e.g. 0.05,0.10,0.15")
    println("  --baseline <value>    Baseline fraction for agreement comparison")
    println("  --max-points <n>      Max plotted points per scatter panel")
    println("  --seed <n>            Base random seed for downsampling")
    println("  -h, --help            Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    manifest_rows = Level2IO.read_manifest_csv(opt["manifest"], config)
    fractions = parse_fraction_list(opt["fractions"])
    baseline_fraction = parse(Float64, opt["baseline"])
    baseline_fraction in fractions ||
        error("Baseline fraction $baseline_fraction must be included in --fractions")

    output_root = isempty(opt["output-dir"]) ?
        normpath(joinpath(Level2IO.default_level2_output_root(), config["geology_id"], "figures", "joint_regimes_sensitivity")) :
        normpath(opt["output-dir"])
    table_root = joinpath(output_root, "tables")
    overview_root = joinpath(output_root, "overview")
    mkpath(table_root)
    mkpath(overview_root)

    Level2Plotting.activate_plot_theme!()
    max_points = parse(Int, opt["max-points"])
    seed = parse(Int, opt["seed"])
    proxies = [Level2IO.load_proxy_library(row) for row in manifest_rows]

    states_by_fraction = Dict{Float64, Dict{String, Dict{String, Any}}}()
    for fraction in fractions
        fraction_label = minfrac_label(fraction)
        fraction_dir = joinpath(output_root, fraction_label, "pairwise")
        mkpath(fraction_dir)
        cfg = copy(config)
        cfg["min_cluster_fraction"] = fraction

        window_states = Dict{String, Dict{String, Any}}()
        for (widx, proxy) in enumerate(proxies)
            state = Level2Core.build_window_state(proxy["log_perms"],
                                                  proxy["raw_perms"],
                                                  proxy["window"],
                                                  proxy["source_path"],
                                                  proxy["source_label"],
                                                  cfg)
            window_states[proxy["window"]] = state
            fig = build_sensitivity_pairwise_figure(state, fraction, max_points, seed + 100 * widx)
            save(joinpath(fraction_dir, "$(proxy["window"])_$(fraction_label)_joint_regimes_pairwise.png"), fig)
        end
        states_by_fraction[fraction] = window_states
    end

    summary_rows, silhouette_rows = build_summary_rows(states_by_fraction, fractions, baseline_fraction)
    Level2IO.write_csv(joinpath(table_root, "joint_regime_sensitivity_summary.csv"),
                       ["window", "min_cluster_fraction", "chosen_k", "best_silhouette",
                        "is_effectively_unimodal", "cluster_sizes", "min_required_size",
                        "same_k_as_baseline", "same_unimodality_as_baseline",
                        "same_cluster_size_count_as_baseline", "assignment_agreement_with_baseline"],
                       summary_rows)
    Level2IO.write_csv(joinpath(table_root, "joint_regime_sensitivity_silhouette_by_k.csv"),
                       ["window", "min_cluster_fraction", "k", "valid_k", "silhouette"],
                       silhouette_rows)

    overview_fig = build_sensitivity_overview_figure(states_by_fraction, fractions)
    save(joinpath(overview_root, "joint_regime_sensitivity_overview.png"), overview_fig)

    println("Saved joint regime sensitivity outputs to $output_root")
end

function parse_fraction_list(text::AbstractString)
    values = [parse(Float64, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
    isempty(values) && error("--fractions must contain at least one value")
    all(0.0 .< values .< 1.0) || error("All fractions must be between 0 and 1")
    return sort(unique(values))
end

function minfrac_label(fraction::Float64)
    return @sprintf("minfrac_%03d", round(Int, 100 * fraction))
end

function build_summary_rows(states_by_fraction, fractions, baseline_fraction::Float64)
    summary_rows = Vector{Vector{String}}()
    silhouette_rows = Vector{Vector{String}}()
    baseline_states = states_by_fraction[baseline_fraction]

    for window in Level2IO.FIXED_WINDOWS
        baseline = baseline_states[window]
        baseline_rank_assignments = Level2Plotting.cluster_rank_assignments(baseline)
        baseline_k = Level2Plotting.int_scalar(baseline["chosen_k"])
        baseline_unimodal = Level2Plotting.int_scalar(baseline["is_effectively_unimodal"])
        baseline_cluster_count = length(Level2Plotting.vector_int(baseline["cluster_sizes"]))

        for fraction in fractions
            state = states_by_fraction[fraction][window]
            chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
            is_unimodal = Level2Plotting.int_scalar(state["is_effectively_unimodal"])
            cluster_sizes = Level2Plotting.ordered_cluster_sizes(state)
            min_required_size = max(Level2Plotting.int_scalar(state["min_cluster_size"]),
                                    ceil(Int, fraction * Level2Plotting.int_scalar(state["n_samples"])))

            agreement = "NA"
            if chosen_k == baseline_k
                ranks = Level2Plotting.cluster_rank_assignments(state)
                agreement = string(round(mean(ranks .== baseline_rank_assignments), digits = 6))
            end

            push!(summary_rows, [
                window,
                string(round(fraction, digits = 4)),
                string(chosen_k),
                string(round(Level2Plotting.float_scalar(state["best_silhouette"]), digits = 6)),
                string(is_unimodal),
                join(string.(cluster_sizes), ";"),
                string(min_required_size),
                chosen_k == baseline_k ? "1" : "0",
                is_unimodal == baseline_unimodal ? "1" : "0",
                length(cluster_sizes) == baseline_cluster_count ? "1" : "0",
                agreement,
            ])

            silhouettes = Level2Plotting.vector_float(state["silhouette_by_k"])
            valid_mask = Level2Plotting.vector_int(state["valid_k_mask"])
            for k in 2:length(silhouettes)
                value = silhouettes[k]
                push!(silhouette_rows, [
                    window,
                    string(round(fraction, digits = 4)),
                    string(k),
                    string(valid_mask[k]),
                    isnan(value) ? "NaN" : string(round(value, digits = 6)),
                ])
            end
        end
    end

    return summary_rows, silhouette_rows
end

function build_sensitivity_pairwise_figure(state::Dict{String, Any},
                                           fraction::Float64,
                                           max_points::Int,
                                           seed::Int)
    window = String(state["window"])
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = round(Level2Plotting.float_scalar(state["best_silhouette"]), digits = 4)

    log_perms = Level2Plotting.matrix_float(state["log_perms"])
    cluster_ranks = Level2Plotting.cluster_rank_assignments(state)
    cluster_sizes = Level2Plotting.ordered_cluster_sizes(state)
    sample_idx = Level2Plotting.sample_indices(size(log_perms, 1), max_points, seed)
    sample_colors = [Level2Plotting.CLUSTER_COLORS[cluster_ranks[idx]] for idx in sample_idx]

    fig = Figure(size = (1860, 760))
    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 10)

    for (panel_idx, ((a, b), title)) in enumerate(zip(PAIRWISE_COMPONENTS, PAIRWISE_TITLES))
        ax = Axis(fig[1, panel_idx],
                  title = title,
                  titlealign = :left,
                  xlabel = PAIRWISE_AXIS_LABELS[a],
                  ylabel = PAIRWISE_AXIS_LABELS[b],
                  titlesize = 22,
                  xlabelsize = 22,
                  ylabelsize = 22,
                  xticklabelsize = 22,
                  yticklabelsize = 22,
                  xticklabelspace = 24.0,
                  yticklabelspace = 26.0,
                  xlabelpadding = 8.0,
                  ylabelpadding = 8.0,
                  topspinevisible = false,
                  rightspinevisible = false,
                  xticks = (FIXED_TICKS, FIXED_TICK_LABELS),
                  yticks = (FIXED_TICKS, FIXED_TICK_LABELS))

        scatter!(ax,
                 log_perms[sample_idx, a], log_perms[sample_idx, b];
                 color = sample_colors,
                 markersize = 8,
                 alpha = 0.42)
        add_cluster_medoid_circles!(ax, state, log_perms, a, b)
        xlims!(ax, FIXED_LIMS...)
        ylims!(ax, FIXED_LIMS...)
        add_panel_label!(ax, PANEL_LABELS[panel_idx])
    end
    for panel_idx in 1:3
        colsize!(fig.layout, panel_idx, Relative(1 / 3))
    end

    Label(fig[0, 2],
          @sprintf("%s joint regimes | min cluster fraction = %.2f | K = %d | silhouette = %.4f",
                   window, fraction, chosen_k, silhouette),
          fontsize = 24,
          font = :bold,
          halign = :center,
          justification = :center,
          tellwidth = false)

    legend_elements, legend_labels = build_legend_content(cluster_sizes)
    Legend(fig[2, :], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 10,
           rowgap = 8,
           colgap = 20,
           tellwidth = false,
           labelsize = 22)

    Label(fig[3, :],
          "Sensitivity figure only. Point color = regime membership. Large outlined circles = cluster medoids.",
          fontsize = 22)

    return fig
end

function add_cluster_medoid_circles!(ax, state::Dict{String, Any}, xy::Matrix{Float64}, a::Int, b::Int)
    cluster_order = Level2Plotting.vector_int(state["cluster_order"])
    cluster_medoids = Level2Plotting.vector_int(state["cluster_medoids"])
    for (rank, cluster_id) in enumerate(cluster_order)
        idx = cluster_medoids[cluster_id]
        scatter!(ax, [xy[idx, a]], [xy[idx, b]];
                 color = Level2Plotting.CLUSTER_COLORS[rank],
                 marker = :circle,
                 markersize = 22,
                 strokecolor = :black,
                 strokewidth = 2.0)
    end
end

function add_panel_label!(ax, text_label::AbstractString)
    text!(ax, 0.98, 0.98;
          space = :relative,
          align = (:right, :top),
          fontsize = 22,
          font = :bold,
          color = :black,
          text = text_label)
end

function build_legend_content(cluster_sizes)
    elements = CairoMakie.LegendElement[]
    labels = String[]
    for (rank, size_value) in enumerate(cluster_sizes)
        push!(elements,
              MarkerElement(color = Level2Plotting.CLUSTER_COLORS[rank],
                            marker = :circle,
                            markersize = 18,
                            strokecolor = :transparent))
        push!(labels, "Regime $rank (n = $size_value)")
    end
    push!(elements,
          MarkerElement(color = :white,
                        marker = :circle,
                        markersize = 20,
                        strokecolor = :black,
                        strokewidth = 2.0))
    push!(labels, "Cluster medoid")
    return elements, labels
end

function srgb_to_linear(value::Real)
    x = Float64(value)
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055)^2.4
end

function relative_luminance(color)
    r = srgb_to_linear(color.r)
    g = srgb_to_linear(color.g)
    b = srgb_to_linear(color.b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
end

function annotation_text_color(colormap, value::Real, color_min::Real, color_max::Real)
    colors = Makie.to_colormap(colormap)
    span = Float64(color_max - color_min)
    position = span == 0.0 ? 0.5 : clamp((Float64(value) - Float64(color_min)) / span, 0.0, 1.0)
    color_index = clamp(round(Int, 1 + position * (length(colors) - 1)), 1, length(colors))
    return relative_luminance(colors[color_index]) >= 0.45 ? :black : :white
end

function build_sensitivity_overview_figure(states_by_fraction, fractions)
    windows = Level2IO.FIXED_WINDOWS
    window_labels = ["W$i" for i in 1:length(windows)]
    k_matrix = zeros(Float64, length(windows), length(fractions))
    sil_matrix = zeros(Float64, length(windows), length(fractions))

    for (i, window) in enumerate(windows)
        for (j, fraction) in enumerate(fractions)
            state = states_by_fraction[fraction][window]
            k_matrix[i, j] = Level2Plotting.int_scalar(state["chosen_k"])
            sil_matrix[i, j] = Level2Plotting.float_scalar(state["best_silhouette"])
        end
    end

    axis_font_size = 20
    fig = Figure(size = (1500, 780))
    title_axis = Axis(fig[0, 1:2],
                      height = 54.0,
                      xgridvisible = false,
                      ygridvisible = false)
    hidedecorations!(title_axis)
    hidespines!(title_axis)
    text!(title_axis, 0.5, 0.5;
          space = :relative,
          align = (:center, :center),
          text = "Joint regime sensitivity to minimum cluster fraction",
          fontsize = 24,
          font = :bold,
          color = :black)

    fraction_labels = [@sprintf("%.2f", f) for f in fractions]
    k_min = 1
    k_max = max(2, ceil(Int, maximum(k_matrix)))
    k_ticks = collect(k_min:k_max)
    k_tick_labels = string.(k_ticks)
    k_colormap = cgrad(:viridis, length(k_ticks); categorical = true)

    sil_data_min = minimum(sil_matrix)
    sil_data_max = maximum(sil_matrix)
    if sil_data_min == sil_data_max
        sil_data_min -= 0.01
        sil_data_max += 0.01
    end
    sil_ticks = collect(range(sil_data_min, sil_data_max; length = 4))
    sil_tick_labels = [@sprintf("%.3f", value) for value in sil_ticks]
    sil_padding = 0.04 * (sil_data_max - sil_data_min)
    sil_color_min = sil_data_min - sil_padding
    sil_color_max = sil_data_max + sil_padding

    left_grid = GridLayout()
    right_grid = GridLayout()
    fig[1, 1] = left_grid
    fig[1, 2] = right_grid
    colsize!(fig.layout, 1, Relative(1 / 2))
    colsize!(fig.layout, 2, Relative(1 / 2))
    colgap!(fig.layout, 34)
    colgap!(left_grid, 16)
    colgap!(right_grid, 16)

    ax1 = Axis(left_grid[1, 1],
               title = "Chosen K",
               xlabel = "Minimum cluster fraction",
               ylabel = "Window",
               titlesize = 22,
               xlabelsize = axis_font_size,
               ylabelsize = axis_font_size,
               xticklabelsize = axis_font_size,
               yticklabelsize = axis_font_size,
               xticks = (1:length(fractions), fraction_labels),
               yticks = (1:length(windows), window_labels),
               yreversed = true,
               topspinevisible = false,
               rightspinevisible = false)
    hm1 = heatmap!(ax1, 1:length(fractions), 1:length(windows), permutedims(k_matrix);
                   colorrange = (k_min - 0.5, k_max + 0.5),
                   colormap = k_colormap)
    Colorbar(left_grid[1, 2], hm1;
             label = "K",
             ticks = (k_ticks, k_tick_labels),
             labelsize = axis_font_size,
             ticklabelsize = axis_font_size)
    colsize!(left_grid, 1, Relative(0.91))
    colsize!(left_grid, 2, Relative(0.09))
    for i in 1:length(windows), j in 1:length(fractions)
        label_color = annotation_text_color(k_colormap, k_matrix[i, j], k_min - 0.5, k_max + 0.5)
        text!(ax1, j, i; text = string(round(Int, k_matrix[i, j])),
              align = (:center, :center), fontsize = axis_font_size, color = label_color)
    end

    ax2 = Axis(right_grid[1, 1],
               title = "Best silhouette",
               xlabel = "Minimum cluster fraction",
               ylabel = "",
               titlesize = 22,
               xlabelsize = axis_font_size,
               xticklabelsize = axis_font_size,
               yticklabelsize = axis_font_size,
               xticks = (1:length(fractions), fraction_labels),
               yticks = (1:length(windows), window_labels),
               yreversed = true,
               topspinevisible = false,
               rightspinevisible = false)
    hm2 = heatmap!(ax2, 1:length(fractions), 1:length(windows), permutedims(sil_matrix);
                   colorrange = (sil_color_min, sil_color_max),
                   colormap = :magma)
    Colorbar(right_grid[1, 2], hm2;
             label = "silhouette",
             ticks = (sil_ticks, sil_tick_labels),
             labelsize = axis_font_size,
             ticklabelsize = axis_font_size)
    colsize!(right_grid, 1, Relative(0.91))
    colsize!(right_grid, 2, Relative(0.09))
    for i in 1:length(windows), j in 1:length(fractions)
        label_color = annotation_text_color(:magma, sil_matrix[i, j], sil_color_min, sil_color_max)
        text!(ax2, j, i; text = @sprintf("%.3f", sil_matrix[i, j]),
              align = (:center, :center), fontsize = axis_font_size, color = label_color)
    end
    return fig
end

main(ARGS)
