#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_window_grouping_3d.jl [options]")
    println()
    println("Options:")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where 3D grouping figures are saved")
    println("  --max-points <n>       Max plotted points in scatter panels (default: 1500)")
    println("  --seed <n>             Base random seed for downsampling (default: 1729)")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "window_grouping_3d") :
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
        save(joinpath(output_root, "$(window)_grouping_3d.png"), fig)
    end

    println("Saved 3D grouping figures to $output_root")
end

function build_window_figure(state::Dict{String, Any}, max_points::Int, seed::Int)
    window = String(state["window"])
    chosen_k = Level2Plotting.int_scalar(state["chosen_k"])
    silhouette = round(Level2Plotting.float_scalar(state["best_silhouette"]), digits = 4)

    log_perms = Level2Plotting.matrix_float(state["log_perms"])
    z = Level2Plotting.matrix_float(state["local_normal_scores"])
    cluster_ranks = Level2Plotting.cluster_rank_assignments(state)
    sample_idx = Level2Plotting.sample_indices(size(log_perms, 1), max_points, seed)
    sample_colors = [Level2Plotting.CLUSTER_COLORS[cluster_ranks[idx]] for idx in sample_idx]

    fig = Figure(size = (2100, 980))
    Label(fig[0, :],
          "$window 3D grouping view | K = $chosen_k | silhouette = $silhouette",
          fontsize = 24,
          font = :bold)

    ax_raw = Axis3(fig[1, 1],
                   title = "Raw log10(k) space",
                   xlabel = "log10(kxx)",
                   ylabel = "log10(kyy)",
                   zlabel = "log10(kzz)",
                   perspectiveness = 0.65,
                   elevation = 0.38,
                   azimuth = 1.05)
    scatter!(ax_raw,
             log_perms[sample_idx, 1],
             log_perms[sample_idx, 2],
             log_perms[sample_idx, 3];
             color = sample_colors,
             markersize = 8,
             alpha = 0.40)
    add_cluster_medoid_markers_3d!(ax_raw, state, log_perms)
    add_state_medoid_markers_3d!(ax_raw, state, log_perms)

    ax_z = Axis3(fig[1, 2],
                 title = "Local normal-score space",
                 xlabel = "z(kxx)",
                 ylabel = "z(kyy)",
                 zlabel = "z(kzz)",
                 perspectiveness = 0.65,
                 elevation = 0.38,
                 azimuth = 1.05)
    scatter!(ax_z,
             z[sample_idx, 1],
             z[sample_idx, 2],
             z[sample_idx, 3];
             color = sample_colors,
             markersize = 8,
             alpha = 0.40)
    add_cluster_medoid_markers_3d!(ax_z, state, z)
    add_state_medoid_markers_3d!(ax_z, state, z)

    ax_text = Axis(fig[2, 1:2], title = "Marker guide")
    hidedecorations!(ax_text)
    hidespines!(ax_text)
    text!(ax_text, 0.02, 0.98;
          space = :relative,
          align = (:left, :top),
          fontsize = 20,
          text = marker_guide_text(state))

    return fig
end

function add_cluster_medoid_markers_3d!(ax, state, xyz::Matrix{Float64})
    cluster_order = Level2Plotting.vector_int(state["cluster_order"])
    cluster_medoids = Level2Plotting.vector_int(state["cluster_medoids"])
    for (rank, cluster_id) in enumerate(cluster_order)
        idx = cluster_medoids[cluster_id]
        scatter!(ax, [xyz[idx, 1]], [xyz[idx, 2]], [xyz[idx, 3]];
                 color = Level2Plotting.CLUSTER_COLORS[rank],
                 marker = :utriangle,
                 markersize = 24,
                 strokecolor = :black,
                 strokewidth = 1.3)
    end
    global_idx = Level2Plotting.int_scalar(state["global_medoid_index"])
    scatter!(ax, [xyz[global_idx, 1]], [xyz[global_idx, 2]], [xyz[global_idx, 3]];
             color = :black,
             marker = :star5,
             markersize = 26)
end

function add_state_medoid_markers_3d!(ax, state, xyz::Matrix{Float64})
    for label in Level2Plotting.STATE_ORDER
        idx = Level2Plotting.int_scalar(state["$(label)_medoid_index"])
        scatter!(ax, [xyz[idx, 1]], [xyz[idx, 2]], [xyz[idx, 3]];
                 color = Level2Plotting.STATE_COLORS[label],
                 marker = :diamond,
                 markersize = 22,
                 strokecolor = :black,
                 strokewidth = 1.1)
    end
end

function marker_guide_text(state::Dict{String, Any})
    cluster_sizes = Level2Plotting.ordered_cluster_sizes(state)
    lines = [
        "Point colors = cluster membership from Step 2.3, ordered from lower to higher median state score.",
        "Blue/red point clouds are the detected joint regimes for K = $(Level2Plotting.int_scalar(state["chosen_k"])).",
        "Colored triangles = cluster medoids.",
        "Black star = global medoid for the whole window library.",
        "Colored diamonds = state medoids for low, central, and high libraries.",
        "Ordered cluster sizes = $(join(cluster_sizes, ", ")).",
    ]
    return join(lines, "\n")
end

main(ARGS)
