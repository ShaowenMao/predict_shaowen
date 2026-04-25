#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie
using Printf

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

const PAIRWISE_COMPONENTS = ((1, 2), (1, 3), (2, 3))
const PAIRWISE_TITLES = ("log10(kxx) vs log10(kyy)",
                         "log10(kxx) vs log10(kzz)",
                         "log10(kyy) vs log10(kzz)")
const PAIRWISE_AXIS_LABELS = ("log10(kxx [mD])",
                              "log10(kyy [mD])",
                              "log10(kzz [mD])")
const PANEL_LABELS = ("(a)", "(b)", "(c)")
const FIXED_TICKS = [-7.0, -4.0, -1.0, 2.0]
const FIXED_TICK_LABELS = ["-7", "-4", "-1", "2"]
const FIXED_LIMS = (-7.1, 2.1)

panel_label(index::Integer) = "($(Char(Int('a') + index - 1)))"

function parse_args(args::Vector{String})
    options = Dict(
        "state-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref")),
        "output-dir" => "",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_joint_regimes.jl [options]")
    println()
    println("Options:")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where regime-only figures are saved")
    println("  --max-points <n>       Max plotted points in scatter panels (default: 1400)")
    println("  --seed <n>             Base random seed for downsampling (default: 1729)")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "joint_regimes") :
                  normpath(opt["output-dir"])
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()

    max_points = parse(Int, opt["max-points"])
    seed = parse(Int, opt["seed"])
    states = Dict{String, Dict{String, Any}}()

    for (widx, window) in enumerate(Level2IO.FIXED_WINDOWS)
        state_path = joinpath(state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing Level 2 state MAT file: $state_path")
        state = Level2IO.load_window_state(state_path)
        states[window] = state
        pairwise_fig = build_pairwise_figure(state, max_points, seed + 100 * widx)
        pca_fig = build_pca_figure(state, max_points, seed + 100 * widx)
        resize_to_layout!(pca_fig)
        save(joinpath(output_root, "$(window)_joint_regimes_pairwise.png"), pairwise_fig)
        save_optional_pdf(joinpath(output_root, "$(window)_joint_regimes_pairwise.pdf"), pairwise_fig)
        save(joinpath(output_root, "$(window)_joint_regimes_pca.png"), pca_fig)
        save_optional_pdf(joinpath(output_root, "$(window)_joint_regimes_pca.pdf"), pca_fig)
    end

    combined_fig = build_all_windows_pairwise_grid(states, max_points, seed + 1000)
    save(joinpath(output_root, "all_windows_joint_regimes_pairwise_grid.png"), combined_fig)
    save_optional_pdf(joinpath(output_root, "all_windows_joint_regimes_pairwise_grid.pdf"), combined_fig)

    println("Saved joint regime figures to $output_root")
end

function save_optional_pdf(path::AbstractString, fig::Figure)
    try
        save(path, fig)
    catch err
        @warn "Skipping PDF export because the file is locked or unavailable" path exception = (err, catch_backtrace())
    end
end

function build_pairwise_figure(state::Dict{String, Any}, max_points::Int, seed::Int)
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
          "$window joint regimes in raw log10(k) space | K = $chosen_k | silhouette = $silhouette",
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
          "Point color = regime membership from Step 2.3. Large outlined circles = cluster medoids.",
          fontsize = 22)

    return fig
end

function build_pca_figure(state::Dict{String, Any}, max_points::Int, seed::Int)
    window = String(state["window"])
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = round(Level2Plotting.float_scalar(state["best_silhouette"]), digits = 4)

    z = Level2Plotting.matrix_float(state["local_normal_scores"])
    cluster_ranks = Level2Plotting.cluster_rank_assignments(state)
    cluster_sizes = Level2Plotting.ordered_cluster_sizes(state)
    sample_idx = Level2Plotting.sample_indices(size(z, 1), max_points, seed)
    sample_colors = [Level2Plotting.CLUSTER_COLORS[cluster_ranks[idx]] for idx in sample_idx]
    pca_coords, explained = Level2Plotting.pca_projection(z)

    fig = Figure(size = (1050, 980))
    Label(fig[0, :],
          "$window joint regimes in local normal-score PCA | K = $chosen_k | silhouette = $silhouette",
          fontsize = 24,
          font = :bold)

    ax = Axis(fig[1, 1],
              title = "Step 2.3 clustering view",
              xlabel = "PC1 ($(round(100 * explained[1], digits = 1))%)",
              ylabel = "PC2 ($(round(100 * explained[2], digits = 1))%)",
              titlesize = 22,
              xlabelsize = 20,
              ylabelsize = 20,
              xticklabelsize = 18,
              yticklabelsize = 18,
              topspinevisible = false,
              rightspinevisible = false)
    scatter!(ax,
             pca_coords[sample_idx, 1], pca_coords[sample_idx, 2];
             color = sample_colors,
             markersize = 8,
             alpha = 0.42)
    add_cluster_medoid_circles!(ax, state, pca_coords, 1, 2)

    legend_elements, legend_labels = build_legend_content(cluster_sizes)
    Legend(fig[2, 1], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 10,
           rowgap = 8,
           colgap = 20,
           tellwidth = false,
           labelsize = 18)

    Label(fig[3, 1],
          "This PCA panel is computed from the local normal-score coordinates used by the clustering step.",
          fontsize = 18)

    return fig
end

function build_all_windows_pairwise_grid(states::Dict{String, Dict{String, Any}},
                                         max_points::Int,
                                         seed::Int)
    windows = Level2IO.FIXED_WINDOWS
    title_font_size = 42
    header_font_size = 34
    axis_font_size = 34
    tick_font_size = 34
    panel_size = 430

    fig = Figure(size = (3900, 2250),
                 figure_padding = (22, 22, 18, 18),
                 backgroundcolor = :white)
    rowgap!(fig.layout, 12)
    colgap!(fig.layout, 18)

    Label(fig[1, 1:6],
          "All-window joint regimes in raw log10(k) space",
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

    for (panel_idx, (a, b)) in enumerate(PAIRWISE_COMPONENTS)
        row = 2 * panel_idx + 1

        for (widx, window) in enumerate(windows)
            state = states[window]
            log_perms = Level2Plotting.matrix_float(state["log_perms"])
            cluster_ranks = Level2Plotting.cluster_rank_assignments(state)
            sample_idx = Level2Plotting.sample_indices(size(log_perms, 1), max_points, seed + 100 * widx)
            sample_colors = [Level2Plotting.CLUSTER_COLORS[cluster_ranks[idx]] for idx in sample_idx]

            ax = Axis(fig[row, widx],
                      xlabel = "",
                      ylabel = widx == 1 ? PAIRWISE_AXIS_LABELS[b] : "",
                      xlabelsize = axis_font_size,
                      ylabelsize = axis_font_size,
                      xticklabelsize = tick_font_size,
                      yticklabelsize = tick_font_size,
                      ylabelpadding = 10.0,
                      xgridcolor = RGBf(0.88, 0.88, 0.88),
                      ygridcolor = RGBf(0.88, 0.88, 0.88),
                      xgridwidth = 1.0,
                      ygridwidth = 1.0,
                      xgridvisible = true,
                      ygridvisible = true,
                      topspinevisible = false,
                      rightspinevisible = false,
                      xticks = (FIXED_TICKS, FIXED_TICK_LABELS),
                      yticks = (FIXED_TICKS, FIXED_TICK_LABELS),
                      aspect = AxisAspect(1))
            scatter!(ax,
                     log_perms[sample_idx, a], log_perms[sample_idx, b];
                     color = sample_colors,
                     markersize = 6.5,
                     alpha = 0.78)
            add_cluster_medoid_circles!(ax, state, log_perms, a, b; markersize = 19, strokewidth = 2.6)
            xlims!(ax, FIXED_LIMS...)
            ylims!(ax, FIXED_LIMS...)
            add_panel_label!(ax, panel_label((panel_idx - 1) * length(windows) + widx); fontsize = 34)

            if widx != 1
                hideydecorations!(ax, grid = false)
            end
        end

        Label(fig[row + 1, 1:6],
              PAIRWISE_AXIS_LABELS[a],
              fontsize = axis_font_size,
              halign = :center,
              tellwidth = false)
    end

    legend_elements = [
        MarkerElement(color = Level2Plotting.CLUSTER_COLORS[1],
                      marker = :circle,
                      markersize = 22,
                      strokecolor = :transparent),
        MarkerElement(color = Level2Plotting.CLUSTER_COLORS[2],
                      marker = :circle,
                      markersize = 22,
                      strokecolor = :transparent),
        MarkerElement(color = :white,
                      marker = :circle,
                      markersize = 23,
                      strokecolor = :black,
                      strokewidth = 2.0),
    ]
    legend_labels = ["Regime 1", "Regime 2", "Cluster medoid"]
    Legend(fig[9, 1:6], legend_elements, legend_labels;
           orientation = :horizontal,
           framevisible = false,
           patchlabelgap = 10,
           rowgap = 8,
           colgap = 24,
           tellwidth = false,
           labelsize = header_font_size)

    Label(fig[10, 1:6],
          "Within each window: Regime 1 = lower-score regime; Regime 2 = higher-score regime.",
          fontsize = axis_font_size,
          halign = :center,
          tellwidth = false)

    rowsize!(fig.layout, 1, Fixed(62))
    rowsize!(fig.layout, 2, Fixed(86))
    rowsize!(fig.layout, 9, Fixed(70))
    rowsize!(fig.layout, 10, Fixed(46))
    for row in (3, 5, 7)
        rowsize!(fig.layout, row, Fixed(panel_size))
    end
    for row in (4, 6, 8)
        rowsize!(fig.layout, row, Fixed(48))
    end
    for col in 1:6
        colsize!(fig.layout, col, Fixed(panel_size))
    end
    resize_to_layout!(fig)

    return fig
end

function add_cluster_medoid_circles!(ax,
                                     xy::Matrix{Float64},
                                     medoid_indices::Vector{Int},
                                     cluster_order::Vector{Int},
                                     a::Int,
                                     b::Int;
                                     markersize::Real = 22,
                                     strokewidth::Real = 2.0)
    for (rank, cluster_id) in enumerate(cluster_order)
        idx = medoid_indices[cluster_id]
        scatter!(ax, [xy[idx, a]], [xy[idx, b]];
                 color = Level2Plotting.CLUSTER_COLORS[rank],
                 marker = :circle,
                 markersize = markersize,
                 strokecolor = :black,
                 strokewidth = strokewidth)
    end
end

function add_cluster_medoid_circles!(ax,
                                     state::Dict{String, Any},
                                     xy::Matrix{Float64},
                                     a::Int,
                                     b::Int;
                                     markersize::Real = 22,
                                     strokewidth::Real = 2.0)
    cluster_order = Level2Plotting.vector_int(state["cluster_order"])
    cluster_medoids = Level2Plotting.vector_int(state["cluster_medoids"])
    add_cluster_medoid_circles!(ax, xy, cluster_medoids, cluster_order, a, b;
                                markersize = markersize,
                                strokewidth = strokewidth)
end

function add_panel_label!(ax, text_label::AbstractString; fontsize::Real = 22)
    text!(ax, 0.98, 0.98;
          space = :relative,
          align = (:right, :top),
          fontsize = fontsize,
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

main(ARGS)
