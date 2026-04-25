#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie
using KernelDensity
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "manifest" => Level2IO.default_manifest_path(),
        "output-dir" => "",
        "bandwidth-factor-min" => "0.5",
        "bandwidth-factor-max" => "2.0",
        "num-bandwidths" => "5",
        "grid-size" => "801",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_marginal_kde_screening.jl [options]")
    println()
    println("Options:")
    println("  --config <path>                 Level 2 TOML config")
    println("  --manifest <path>               Proxy manifest CSV")
    println("  --output-dir <path>             Output folder for figures and summary CSV")
    println("  --bandwidth-factor-min <x>      Minimum bandwidth factor relative to h0 (default: 0.5)")
    println("  --bandwidth-factor-max <x>      Maximum bandwidth factor relative to h0 (default: 2.0)")
    println("  --num-bandwidths <n>            Number of log-spaced bandwidths (default: 5)")
    println("  --grid-size <n>                 Number of KDE grid points (default: 801)")
    println("  -h, --help                      Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    manifest_rows = Level2IO.read_manifest_csv(opt["manifest"], config)

    output_root = isempty(opt["output-dir"]) ?
        normpath(joinpath(Level2IO.default_level2_output_root(), config["geology_id"], "figures", "marginal_kde_screening")) :
        normpath(opt["output-dir"])
    figure_root = joinpath(output_root, "window_figures")
    table_root = joinpath(output_root, "tables")
    mkpath(figure_root)
    mkpath(table_root)

    factor_min = parse(Float64, opt["bandwidth-factor-min"])
    factor_max = parse(Float64, opt["bandwidth-factor-max"])
    num_bandwidths = parse(Int, opt["num-bandwidths"])
    grid_size = parse(Int, opt["grid-size"])

    Level2Plotting.activate_plot_theme!()

    summary_rows = Vector{Vector{String}}()
    for row in manifest_rows
        proxy = Level2IO.load_proxy_library(row)
        window = proxy["window"]
        values = proxy["log_perms"]
        component_results = [screen_component(vec(values[:, ic]);
                                             factor_min = factor_min,
                                             factor_max = factor_max,
                                             num_bandwidths = num_bandwidths,
                                             grid_size = grid_size) for ic in 1:3]

        fig = build_window_figure(window, component_results)
        save(joinpath(figure_root, "$(window)_marginal_kde_screening.png"), fig)
        save(joinpath(figure_root, "$(window)_marginal_kde_screening.pdf"), fig)

        append!(summary_rows, build_summary_rows(window, component_results))
    end

    Level2IO.write_csv(joinpath(table_root, "marginal_kde_screening_summary.csv"),
                       ["window", "component", "baseline_bandwidth_h0",
                        "chosen_factor", "chosen_bandwidth", "peak_count", "peak_locations_log10k"],
                       summary_rows)

    println("Saved marginal KDE screening outputs to $output_root")
end

function screen_component(values::Vector{Float64};
                          factor_min::Float64,
                          factor_max::Float64,
                          num_bandwidths::Int,
                          grid_size::Int)
    h0 = silverman_bandwidth(values)
    factors = log_spaced_factors(factor_min, factor_max, num_bandwidths)
    chosen_idx = argmin(abs.(factors .- 1.0))
    xlow = minimum(values) - 3.0 * h0
    xhigh = maximum(values) + 3.0 * h0

    kde_results = NamedTuple[]
    for factor in factors
        h = max(factor * h0, eps())
        kd = kde(values; bandwidth = h, boundary = (xlow, xhigh), npoints = grid_size)
        grid = Vector{Float64}(kd.x)
        density = Vector{Float64}(kd.density)
        peak_idx = local_maxima_indices(density)
        isempty(peak_idx) && (peak_idx = [argmax(density)])
        push!(kde_results, (
            factor = factor,
            bandwidth = h,
            grid = grid,
            density = density,
            peak_idx = peak_idx,
            peak_x = grid[peak_idx],
        ))
    end

    chosen = kde_results[chosen_idx]
    return (
        values = values,
        baseline_bandwidth = h0,
        bandwidth_factors = factors,
        chosen_index = chosen_idx,
        chosen_factor = chosen.factor,
        chosen_bandwidth = chosen.bandwidth,
        kde_results = kde_results,
        chosen_peak_idx = chosen.peak_idx,
        chosen_peak_x = chosen.peak_x,
    )
end

function build_window_figure(window::AbstractString, component_results)
    fig = Figure(size = (1850, 700))
    Label(fig[0, :], "$window marginal KDE screening in log10(k) space", fontsize = 24, font = :bold)

    xmins = Float64[]
    xmaxs = Float64[]
    ymaxs = Float64[]
    for result in component_results
        push!(xmins, minimum(first(result.kde_results).grid))
        push!(xmaxs, maximum(first(result.kde_results).grid))
        push!(ymaxs, maximum(maximum(kde_result.density) for kde_result in result.kde_results))
    end
    common_xlims = (floor(minimum(xmins)), ceil(maximum(xmaxs)))

    for (ic, result) in enumerate(component_results)
        chosen = result.kde_results[result.chosen_index]
        ladder_colors = [RGBAf(0.45, 0.45, 0.45, a) for a in range(0.45, 0.85, length = length(result.kde_results))]
        chosen_color = RGBf(0.02, 0.27, 0.58)

        ax = Axis(fig[1, ic],
                  xlabel = COMPONENT_LABELS[ic],
                  ylabel = (ic == 1 ? "Density" : ""),
                  title = @sprintf("%s: %d peak%s, chosen h = %.2f×h₀",
                                   COMPONENT_NAMES[ic],
                                   length(result.chosen_peak_idx),
                                   length(result.chosen_peak_idx) == 1 ? "" : "s",
                                   result.chosen_factor),
                  titlealign = :left,
                  titlesize = 20,
                  xlabelsize = 20,
                  ylabelsize = 20,
                  xticklabelsize = 16,
                  yticklabelsize = 16,
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false,
                  rightspinevisible = false)

        hist!(ax, result.values;
              bins = 45,
              normalization = :pdf,
              color = (RGBf(0.62, 0.64, 0.67), 0.20),
              strokecolor = (RGBf(0.62, 0.64, 0.67), 0.35),
              strokewidth = 0.6)

        for (ibw, kde_result) in enumerate(result.kde_results)
            lines!(ax, kde_result.grid, kde_result.density;
                   color = (ibw == result.chosen_index ? chosen_color : ladder_colors[ibw]),
                   linewidth = (ibw == result.chosen_index ? 3.6 : 1.5),
                   label = @sprintf("%.2f×h₀", kde_result.factor))
        end

        scatter!(ax, chosen.grid[result.chosen_peak_idx], chosen.density[result.chosen_peak_idx];
                 color = :black, marker = :circle, markersize = 9, label = "chosen peaks")

        xlims!(ax, common_xlims...)
        ylims!(ax, 0.0, 1.05 * ymaxs[ic])
        axislegend(ax, position = :lt, framevisible = true,
                   backgroundcolor = RGBAf(1, 1, 1, 0.90),
                   labelsize = 14, patchlabelgap = 8, rowgap = 5)

        peak_text = isempty(result.chosen_peak_x) ? "no peak found" :
                    join([@sprintf("%.3f", x) for x in result.chosen_peak_x], "\n")
        text!(ax, 0.03, 0.48;
              space = :relative,
              text = "peak centers:\n" * peak_text,
              align = (:left, :top),
              fontsize = 16,
              color = :black)

        panel_tag = string("(", Char('a' + ic - 1), ")")
        text!(ax, 0.98, 0.98;
              space = :relative,
              text = panel_tag,
              align = (:right, :top),
              fontsize = 18,
              font = :bold,
              color = :black)
    end

    Label(fig[2, :],
          "Gray bars: empirical distribution. Thin gray curves: bandwidth ladder. Thick blue curve: chosen baseline smoothing near h₀.",
          fontsize = 18)
    return fig
end

function build_summary_rows(window::AbstractString, component_results)
    rows = Vector{Vector{String}}()
    for (ic, result) in enumerate(component_results)
        push!(rows, [
            String(window),
            String(COMPONENT_NAMES[ic]),
            fmt(result.baseline_bandwidth),
            fmt(result.chosen_factor),
            fmt(result.chosen_bandwidth),
            string(length(result.chosen_peak_idx)),
            join([@sprintf("%.6f", x) for x in result.chosen_peak_x], "; "),
        ])
    end
    return rows
end

function log_spaced_factors(factor_min::Float64, factor_max::Float64, num_bandwidths::Int)
    factor_min > 0 || error("bandwidth-factor-min must be positive")
    factor_max > 0 || error("bandwidth-factor-max must be positive")
    num_bandwidths >= 2 || error("num-bandwidths must be at least 2")
    return collect(exp.(range(log(factor_min), log(factor_max), length = num_bandwidths)))
end

function silverman_bandwidth(values::Vector{Float64})
    n = length(values)
    s = std(values; corrected = true)
    q25, q75 = quantile(values, [0.25, 0.75])
    sigma = min(s, (q75 - q25) / 1.34)
    if !isfinite(sigma) || sigma <= 0
        sigma = max(s, 1e-3)
    end
    if !isfinite(sigma) || sigma <= 0
        sigma = 1e-3
    end
    return 0.9 * sigma * n^(-1 / 5)
end

function local_maxima_indices(density::Vector{Float64})
    idx = Int[]
    for i in 2:(length(density) - 1)
        left = density[i] - density[i - 1]
        right = density[i + 1] - density[i]
        if left > 0 && right <= 0
            push!(idx, i)
        end
    end
    return idx
end

fmt(x::Real) = @sprintf("%.6f", x)

main(ARGS)
