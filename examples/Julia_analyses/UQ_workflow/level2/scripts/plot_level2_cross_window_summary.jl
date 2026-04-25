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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_cross_window_summary.jl [options]")
    println()
    println("Options:")
    println("  --state-root <path>    Root folder containing built Level 2 state MAT files")
    println("  --output-dir <path>    Folder where the cross-window summary figure is saved")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "figures", "cross_window") :
                  normpath(opt["output-dir"])
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()
    states = Dict(window => Level2IO.load_window_state(joinpath(state_root, "window_states", window, "$(window)_level2_state.mat"))
                  for window in Level2IO.FIXED_WINDOWS)

    fig = build_cross_window_figure(states)
    save(joinpath(output_root, "level2_cross_window_summary.png"), fig)
    println("Saved cross-window Level 2 summary figure to $output_root")
end

function build_cross_window_figure(states::Dict{String, Dict{String, Any}})
    windows = collect(Level2IO.FIXED_WINDOWS)
    x = 1:length(windows)
    silhouettes = [Level2Plotting.float_scalar(states[w]["best_silhouette"]) for w in windows]
    max_k = maximum(length(Level2Plotting.ordered_cluster_sizes(states[w])) for w in windows)

    fig = Figure(size = (2000, 1300))
    Label(fig[0, :], "Level 2 cross-window summary", fontsize = 24, font = :bold)

    ax_sil = Axis(fig[1, 1],
                  title = "Silhouette by window",
                  xlabel = "window",
                  ylabel = "silhouette")
    barplot!(ax_sil, x, silhouettes, color = RGBf(0.196, 0.490, 0.741))
    ax_sil.xticks = (x, windows)

    ax_cluster = Axis(fig[1, 2],
                      title = "Ordered cluster sizes",
                      xlabel = "window",
                      ylabel = "samples")
    for rank in 1:max_k
        values = [rank <= length(Level2Plotting.ordered_cluster_sizes(states[w])) ?
                  Level2Plotting.ordered_cluster_sizes(states[w])[rank] : 0 for w in windows]
        barplot!(ax_cluster, x, values,
                 dodge = fill(rank, length(windows)),
                 color = Level2Plotting.CLUSTER_COLORS[rank],
                 label = "cluster rank $rank")
    end
    axislegend(ax_cluster, position = :rt)
    ax_cluster.xticks = (x, windows)

    ax_state = Axis(fig[1, 3],
                    title = "State library sizes",
                    xlabel = "window",
                    ylabel = "samples")
    for (rank, label) in enumerate(Level2Plotting.STATE_ORDER)
        values = [length(Level2Plotting.vector_int(states[w]["$(label)_indices"])) for w in windows]
        barplot!(ax_state, x, values,
                 dodge = fill(rank, length(windows)),
                 color = Level2Plotting.STATE_COLORS[label],
                 label = label)
    end
    axislegend(ax_state, position = :rt)
    ax_state.xticks = (x, windows)

    component_matrices = [hcat([Level2Plotting.state_mean_matrix(states[w])[:, comp] for w in windows]...)' for comp in 1:3]
    for comp in 1:3
        ax = Axis(fig[2, comp],
                  title = "$(Level2Plotting.COMPONENT_LABELS[comp]) mean by state",
                  xlabel = "state",
                  ylabel = "window")
        hm = heatmap!(ax, component_matrices[comp], colormap = :viridis)
        ax.xticks = (1:3, collect(Level2Plotting.STATE_ORDER))
        ax.yticks = (1:length(windows), windows)
        Colorbar(fig[2, comp + 3], hm, label = Level2Plotting.COMPONENT_LABELS[comp])
    end

    return fig
end

main(ARGS)
