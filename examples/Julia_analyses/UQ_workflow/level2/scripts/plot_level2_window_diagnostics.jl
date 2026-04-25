#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

const PAIRWISE_COMPONENTS = ((1, 2), (1, 3), (2, 3))
const PAIRWISE_TITLES = ("kxx vs kyy", "kxx vs kzz", "kyy vs kzz")

function parse_args(args::Vector{String})
    options = Dict(
        "state-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref")),
        "output-dir" => "",
        "max-points" => "1500",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_window_diagnostics.jl [options]")
    println()
    println("Options:")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where window diagnostic figures are saved")
    println("  --max-points <n>       Max plotted points in scatter panels (default: 1500)")
    println("  --seed <n>             Base random seed for downsampling (default: 1729)")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "window_diagnostics") :
                  normpath(opt["output-dir"])
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()

    max_points = parse(Int, opt["max-points"])
    seed = parse(Int, opt["seed"])

    for (widx, window) in enumerate(Level2IO.FIXED_WINDOWS)
        state_path = joinpath(state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing Level 2 state MAT file: $state_path")
        state = Level2IO.load_window_state(state_path)
        fig = build_window_figure(state, max_points, seed + 100 * widx)
        save(joinpath(output_root, "$(window)_level2_diagnostic.png"), fig)
    end

    println("Saved window diagnostic figures to $output_root")
end

function build_window_figure(state::Dict{String, Any}, max_points::Int, seed::Int)
    window = String(state["window"])
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = round(Level2Plotting.float_scalar(state["best_silhouette"]), digits = 4)
    fig = Figure(size = (2200, 1250))
    Label(fig[0, :],
          "$window Level 2 diagnostics | K = $chosen_k | silhouette = $silhouette",
          fontsize = 24,
          font = :bold)

    log_perms = Level2Plotting.matrix_float(state["log_perms"])
    z = Level2Plotting.matrix_float(state["local_normal_scores"])
    state_score = Level2Plotting.vector_float(state["state_score"])
    cluster_ranks = Level2Plotting.cluster_rank_assignments(state)
    sample_idx = Level2Plotting.sample_indices(size(log_perms, 1), max_points, seed)
    sample_colors = [Level2Plotting.CLUSTER_COLORS[cluster_ranks[idx]] for idx in sample_idx]

    for (panel_idx, ((a, b), title)) in enumerate(zip(PAIRWISE_COMPONENTS, PAIRWISE_TITLES))
        ax = Axis(fig[1, panel_idx],
                  title = title,
                  xlabel = Level2Plotting.COMPONENT_LABELS[a],
                  ylabel = Level2Plotting.COMPONENT_LABELS[b])
        scatter!(ax, log_perms[sample_idx, a], log_perms[sample_idx, b],
                 color = sample_colors,
                 markersize = 6,
                 alpha = 0.38)
        add_cluster_medoid_markers!(ax, state, log_perms, a, b)
    end

    pca_coords, explained = Level2Plotting.pca_projection(z)
    ax_pca = Axis(fig[1, 4],
                  title = "Local-score PCA",
                  xlabel = "PC1 ($(round(100 * explained[1], digits = 1))%)",
                  ylabel = "PC2 ($(round(100 * explained[2], digits = 1))%)")
    scatter!(ax_pca, pca_coords[sample_idx, 1], pca_coords[sample_idx, 2],
             color = sample_colors,
             markersize = 6,
             alpha = 0.38)
    add_state_projection_markers!(ax_pca, state, pca_coords)

    ax_score = Axis(fig[2, 1:2],
                    title = "State-score distribution",
                    xlabel = "state score",
                    ylabel = "count")
    for label in Level2Plotting.STATE_ORDER
        add_state_score_band!(ax_score, state, label)
    end
    hist!(ax_score, state_score, bins = 40, color = (:black, 0.30), strokewidth = 0)
    add_cluster_score_lines!(ax_score, state)
    add_state_medoid_score_lines!(ax_score, state)

    ax_components = Axis(fig[2, 3],
                         title = "State component summaries",
                         xlabel = "component",
                         ylabel = "log10 permeability")
    plot_state_component_intervals!(ax_components, state)

    ax_neighbors = Axis(fig[2, 4],
                        title = "Neighborhood distance profiles",
                        xlabel = "neighbor fraction",
                        ylabel = "distance in local-score space")
    plot_neighbor_profiles!(ax_neighbors, state)

    ax_text = Axis(fig[2, 5:6], title = "Window summary")
    hidedecorations!(ax_text)
    hidespines!(ax_text)
    summary_text = build_summary_text(state)
    text!(ax_text, 0.02, 0.98,
          text = summary_text,
          align = (:left, :top),
          space = :relative,
          fontsize = 17)

    return fig
end

function add_cluster_medoid_markers!(ax, state, log_perms, a, b)
    cluster_order = Level2Plotting.vector_int(state["cluster_order"])
    cluster_medoids = Level2Plotting.vector_int(state["cluster_medoids"])
    for (rank, cluster_id) in enumerate(cluster_order)
        idx = cluster_medoids[cluster_id]
        scatter!(ax, [log_perms[idx, a]], [log_perms[idx, b]],
                 color = Level2Plotting.CLUSTER_COLORS[rank],
                 marker = :utriangle,
                 markersize = 18,
                 strokecolor = :black,
                 strokewidth = 1.5)
    end
    global_idx = Level2Plotting.int_scalar(state["global_medoid_index"])
    scatter!(ax, [log_perms[global_idx, a]], [log_perms[global_idx, b]],
             color = :black,
             marker = :star5,
             markersize = 20)
end

function add_state_projection_markers!(ax, state, coords)
    global_idx = Level2Plotting.int_scalar(state["global_medoid_index"])
    scatter!(ax, [coords[global_idx, 1]], [coords[global_idx, 2]],
             color = :black,
             marker = :star5,
             markersize = 20)
    for label in Level2Plotting.STATE_ORDER
        idx = Level2Plotting.int_scalar(state["$(label)_medoid_index"])
        scatter!(ax, [coords[idx, 1]], [coords[idx, 2]],
                 color = Level2Plotting.STATE_COLORS[label],
                 marker = :diamond,
                 markersize = 18,
                 strokecolor = :black,
                 strokewidth = 1.2)
    end
end

function add_state_score_band!(ax, state, label)
    lo, _, hi = Level2Plotting.state_score_range(state, label)
    vspan!(ax, lo, hi, color = (Level2Plotting.STATE_COLORS[label], 0.14))
end

function add_cluster_score_lines!(ax, state)
    medians = Level2Plotting.ordered_cluster_score_medians(state)
    for (rank, value) in enumerate(medians)
        vlines!(ax, [value], color = Level2Plotting.CLUSTER_COLORS[rank], linestyle = :dash, linewidth = 2)
    end
end

function add_state_medoid_score_lines!(ax, state)
    scores = Level2Plotting.vector_float(state["state_score"])
    for label in Level2Plotting.STATE_ORDER
        idx = Level2Plotting.int_scalar(state["$(label)_medoid_index"])
        vlines!(ax, [scores[idx]], color = Level2Plotting.STATE_COLORS[label], linewidth = 3)
    end
end

function plot_state_component_intervals!(ax, state)
    xbase = 1:3
    offsets = Dict("low" => -0.22, "central" => 0.0, "high" => 0.22)
    for label in Level2Plotting.STATE_ORDER
        summary = Level2Plotting.state_component_summary(state, label)
        medoid_vals = Level2Plotting.medoid_component_values(state, label)
        xs = collect(xbase) .+ offsets[label]
        lower = summary.medians .- summary.q25
        upper = summary.q75 .- summary.medians
        errorbars!(ax, xs, summary.medians, lower, upper,
                   color = Level2Plotting.STATE_COLORS[label],
                   whiskerwidth = 8)
        scatter!(ax, xs, summary.medians,
                 color = Level2Plotting.STATE_COLORS[label],
                 markersize = 12)
        scatter!(ax, xs, medoid_vals,
                 color = Level2Plotting.STATE_COLORS[label],
                 marker = :diamond,
                 markersize = 14,
                 strokecolor = :black,
                 strokewidth = 1.0)
    end
    ax.xticks = (1:3, collect(Level2Plotting.COMPONENT_NAMES))
end

function plot_neighbor_profiles!(ax, state)
    for label in Level2Plotting.STATE_ORDER
        for (neighborhood, linestyle) in (("small", :solid), ("large", :dash))
            distances = Level2Plotting.neighbor_distance_profile(state, label, neighborhood)
            x = range(0.0, 1.0; length = length(distances))
            lines!(ax, x, distances,
                   color = Level2Plotting.STATE_COLORS[label],
                   linestyle = linestyle,
                   linewidth = neighborhood == "small" ? 3 : 2)
        end
    end
end

function build_summary_text(state)
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = round(Level2Plotting.float_scalar(state["best_silhouette"]), digits = 4)
    cluster_sizes = Level2Plotting.ordered_cluster_sizes(state)
    low_n = length(Level2Plotting.vector_int(state["low_indices"]))
    high_n = length(Level2Plotting.vector_int(state["high_indices"]))
    central_n = length(Level2Plotting.vector_int(state["central_indices"]))
    global_medoid = Level2Plotting.int_scalar(state["global_medoid_index"])
    lines = [
        "chosen K: $chosen_k",
        "best silhouette: $silhouette",
        "cluster sizes (ordered): $(join(cluster_sizes, ", "))",
        "state sizes: low=$low_n, central=$central_n, high=$high_n",
        "global medoid index: $global_medoid",
    ]
    for label in Level2Plotting.STATE_ORDER
        lo, med, hi = Level2Plotting.state_score_range(state, label)
        push!(lines, "$label score range: $(round(lo, digits = 3)) to $(round(hi, digits = 3))")
        push!(lines, "$label medoid index: $(Level2Plotting.int_scalar(state["$(label)_medoid_index"]))")
    end
    return join(lines, "\n")
end

main(ARGS)
