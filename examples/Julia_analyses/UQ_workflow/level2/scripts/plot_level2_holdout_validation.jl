#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie
using Statistics

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

function parse_args(args::Vector{String})
    options = Dict(
        "validation-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref", "validation")),
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_level2_holdout_validation.jl [options]")
    println()
    println("Options:")
    println("  --validation-root <path>    Root folder containing validation CSV outputs")
    println("  --output-dir <path>         Folder where holdout validation figures are saved")
    println("  -h, --help                  Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    validation_root = normpath(opt["validation-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(validation_root, "figures") :
                  normpath(opt["output-dir"])
    mkpath(output_root)

    Level2Plotting.activate_plot_theme!()

    summary_table = Level2Plotting.read_simple_csv(joinpath(validation_root, "tables", "holdout_validation_summary.csv"))
    pair_table = Level2Plotting.read_simple_csv(joinpath(validation_root, "tables", "holdout_validation_pairs.csv"))

    summary_fig = build_validation_summary_figure(summary_table)
    save(joinpath(output_root, "level2_holdout_validation_summary.png"), summary_fig)

    pair_fig = build_validation_pairs_figure(pair_table)
    save(joinpath(output_root, "level2_holdout_validation_pairs.png"), pair_fig)

    println("Saved holdout validation figures to $output_root")
end

function build_validation_summary_figure(table)
    windows = Level2Plotting.csv_string_column(table, "window")
    x = 1:length(windows)
    same_k = Level2Plotting.csv_numeric_column(table, "same_k_rate")
    same_uni = Level2Plotting.csv_numeric_column(table, "same_unimodality_rate")
    mean_sil = Level2Plotting.csv_numeric_column(table, "mean_abs_silhouette_delta")
    global_d = Level2Plotting.csv_numeric_column(table, "mean_global_medoid_distance")
    low_d = Level2Plotting.csv_numeric_column(table, "mean_low_medoid_distance")
    high_d = Level2Plotting.csv_numeric_column(table, "mean_high_medoid_distance")
    central_d = Level2Plotting.csv_numeric_column(table, "mean_central_medoid_distance")

    fig = Figure(size = (1900, 900))
    Label(fig[0, :], "Holdout validation summary", fontsize = 24, font = :bold)

    ax_rates = Axis(fig[1, 1],
                    title = "Stability rates",
                    xlabel = "window",
                    ylabel = "rate")
    barplot!(ax_rates, x, same_k,
             dodge = fill(1, length(x)),
             color = RGBf(0.153, 0.392, 0.780),
             label = "same K")
    barplot!(ax_rates, x, same_uni,
             dodge = fill(2, length(x)),
             color = RGBf(0.212, 0.620, 0.365),
             label = "same unimodality")
    axislegend(ax_rates, position = :rb)
    ax_rates.xticks = (x, windows)
    ylims!(ax_rates, 0.0, 1.05)

    ax_sil = Axis(fig[1, 2],
                  title = "Mean absolute silhouette delta",
                  xlabel = "window",
                  ylabel = "mean |delta|")
    barplot!(ax_sil, x, mean_sil, color = RGBf(0.500, 0.353, 0.831))
    ax_sil.xticks = (x, windows)

    ax_dist = Axis(fig[1, 3],
                   title = "Mean medoid distances",
                   xlabel = "window",
                   ylabel = "distance")
    for (rank, (values, label, color)) in enumerate((
        (global_d, "global", :black),
        (low_d, "low", Level2Plotting.STATE_COLORS["low"]),
        (high_d, "high", Level2Plotting.STATE_COLORS["high"]),
        (central_d, "central", Level2Plotting.STATE_COLORS["central"]),
    ))
        barplot!(ax_dist, x, values,
                 dodge = fill(rank, length(x)),
                 color = color,
                 label = label)
    end
    axislegend(ax_dist, position = :rt)
    ax_dist.xticks = (x, windows)

    metric_names = [
        "same_k_rate",
        "same_unimodality_rate",
        "mean_abs_silhouette_delta",
        "mean_global_medoid_distance",
        "mean_low_medoid_distance",
        "mean_high_medoid_distance",
        "mean_central_medoid_distance",
    ]
    metric_values = hcat(
        same_k,
        same_uni,
        mean_sil,
        global_d,
        low_d,
        high_d,
        central_d,
    )
    ax_heat = Axis(fig[2, 1:3],
                   title = "Validation metric heatmap",
                   xlabel = "metric",
                   ylabel = "window")
    hm = heatmap!(ax_heat, metric_values, colormap = :magma)
    ax_heat.xticks = (1:length(metric_names), [Level2Plotting.metric_label(name) for name in metric_names])
    ax_heat.yticks = (1:length(windows), windows)
    Colorbar(fig[2, 4], hm, label = "raw value")

    return fig
end

function build_validation_pairs_figure(table)
    windows = Level2IO.FIXED_WINDOWS
    x_lookup = Dict(window => idx for (idx, window) in enumerate(windows))
    metrics = [
        "abs_silhouette_delta",
        "global_medoid_distance",
        "low_medoid_distance",
        "high_medoid_distance",
        "central_medoid_distance",
    ]

    fig = Figure(size = (2000, 1100))
    Label(fig[0, :], "Holdout validation by repeat", fontsize = 24, font = :bold)

    for (idx, metric) in enumerate(metrics)
        row = idx <= 3 ? 1 : 2
        col = idx <= 3 ? idx : idx - 3
        ax = Axis(fig[row, col],
                  title = Level2Plotting.metric_label(metric),
                  xlabel = "window",
                  ylabel = metric == "abs_silhouette_delta" ? "|delta|" : "distance")

        xs = Float64[]
        ys = Float64[]
        colors = RGBAf[]
        for row_values in table.rows
            window = row_values[1]
            metric_idx = findfirst(==(metric), table.header)
            value = parse(Float64, row_values[metric_idx])
            xbase = x_lookup[window]
            repeat_id = parse(Int, row_values[2])
            jitter = 0.09 * ((repeat_id - 3.5) / 3.5)
            push!(xs, xbase + jitter)
            push!(ys, value)
            if metric == "low_medoid_distance"
                push!(colors, RGBAf(Level2Plotting.STATE_COLORS["low"].r,
                                   Level2Plotting.STATE_COLORS["low"].g,
                                   Level2Plotting.STATE_COLORS["low"].b,
                                   0.75))
            elseif metric == "high_medoid_distance"
                push!(colors, RGBAf(Level2Plotting.STATE_COLORS["high"].r,
                                   Level2Plotting.STATE_COLORS["high"].g,
                                   Level2Plotting.STATE_COLORS["high"].b,
                                   0.75))
            elseif metric == "central_medoid_distance"
                push!(colors, RGBAf(Level2Plotting.STATE_COLORS["central"].r,
                                   Level2Plotting.STATE_COLORS["central"].g,
                                   Level2Plotting.STATE_COLORS["central"].b,
                                   0.75))
            elseif metric == "global_medoid_distance"
                push!(colors, RGBAf(0, 0, 0, 0.75))
            else
                push!(colors, RGBAf(0.500, 0.353, 0.831, 0.75))
            end
        end
        scatter!(ax, xs, ys, color = colors, markersize = 14)

        summary_table = summarize_pairs_for_metric(table, metric, windows)
        lines!(ax, 1:length(windows), summary_table, color = :black, linewidth = 2)
        scatter!(ax, 1:length(windows), summary_table, color = :black, markersize = 10)

        ax.xticks = (1:length(windows), windows)
    end

    return fig
end

function summarize_pairs_for_metric(table, metric::AbstractString, windows::Vector{String})
    metric_idx = findfirst(==(metric), table.header)
    metric_idx === nothing && error("Metric column not found: $metric")
    return [
        mean([parse(Float64, row[metric_idx]) for row in table.rows if row[1] == window])
        for window in windows
    ]
end

main(ARGS)
