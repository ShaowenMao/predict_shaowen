#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using CairoMakie
using Statistics
using Printf

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Plotting

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")
const PANEL_LETTERS = ("(a)", "(b)", "(c)")
const LOGK_TICKS = [-7.0, -4.0, -1.0, 2.0]
const LOGK_TICK_LABELS = ["-7", "-4", "-1", "2"]
const LOGK_LIMITS = (-7.1, 2.1)

panel_label(index::Integer) = "($(Char(Int('a') + index - 1)))"

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "manifest" => Level2IO.default_manifest_path(),
        "output-dir" => "",
        "bins" => "36",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/plot_marginal_hist_screening.jl [options]")
    println()
    println("Options:")
    println("  --config <path>                 Level 2 TOML config")
    println("  --manifest <path>               Proxy manifest CSV")
    println("  --output-dir <path>             Output folder for figures and summary CSV")
    println("  --bins <n>                      Number of histogram bins per component (default: 36)")
    println("  -h, --help                      Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    manifest_rows = Level2IO.read_manifest_csv(opt["manifest"], config)
    nbins = parse(Int, opt["bins"])
    nbins >= 10 || error("--bins must be at least 10")

    output_root = isempty(opt["output-dir"]) ?
        normpath(joinpath(Level2IO.default_level2_output_root(), config["geology_id"], "figures", "marginal_hist_screening")) :
        normpath(opt["output-dir"])
    window_figure_root = joinpath(output_root, "window_figures")
    overview_root = joinpath(output_root, "overview")
    combined_root = joinpath(output_root, "combined")
    table_root = joinpath(output_root, "tables")
    mkpath(window_figure_root)
    mkpath(overview_root)
    mkpath(combined_root)
    mkpath(table_root)

    Level2Plotting.activate_plot_theme!()

    proxies = [Level2IO.load_proxy_library(row) for row in manifest_rows]
    bin_specs = build_global_bin_specs(proxies, nbins)
    summary_rows = Vector{Vector{String}}()

    for proxy in proxies
        window = proxy["window"]
        fig = build_window_histogram_figure(proxy, bin_specs)
        save(joinpath(window_figure_root, "$(window)_marginal_hist_screening.png"), fig)
        save(joinpath(window_figure_root, "$(window)_marginal_hist_screening.pdf"), fig)
        append!(summary_rows, build_summary_rows(proxy, bin_specs))
    end

    overview_fig = build_overview_figure(proxies, bin_specs)
    save(joinpath(overview_root, "marginal_hist_screening_overview.png"), overview_fig)
    save(joinpath(overview_root, "marginal_hist_screening_overview.pdf"), overview_fig)

    combined_fig = build_combined_publication_figure(proxies, bin_specs)
    save(joinpath(combined_root, "all_windows_marginal_histograms_grid.png"), combined_fig)
    save(joinpath(combined_root, "all_windows_marginal_histograms_grid.pdf"), combined_fig)

    Level2IO.write_csv(joinpath(table_root, "marginal_hist_screening_summary.csv"),
                       ["window", "component", "n_samples", "mean_log10k", "std_log10k",
                        "min_log10k", "q05_log10k", "q25_log10k", "median_log10k",
                        "q75_log10k", "q95_log10k", "max_log10k",
                        "global_bin_min", "global_bin_max", "bin_width"],
                       summary_rows)

    println("Saved marginal histogram screening outputs to $output_root")
end

function build_global_bin_specs(proxies, nbins::Int)
    specs = NamedTuple[]
    for ic in 1:3
        values = vcat([vec(proxy["log_perms"][:, ic]) for proxy in proxies]...)
        xmin = minimum(values)
        xmax = maximum(values)
        span = xmax - xmin
        padding = span > 0 ? 0.04 * span : 0.5
        lo = floor(xmin - padding; digits = 2)
        hi = ceil(xmax + padding; digits = 2)
        edges = collect(range(lo, hi; length = nbins + 1))
        centers = @. 0.5 * (edges[1:end-1] + edges[2:end])
        widths = diff(edges)
        push!(specs, (
            component = COMPONENT_NAMES[ic],
            label = COMPONENT_LABELS[ic],
            edges = edges,
            centers = centers,
            widths = widths,
            xlim = (first(edges), last(edges)),
            bin_width = widths[1],
        ))
    end
    return specs
end

function histogram_probabilities(values::Vector{Float64}, edges::Vector{Float64})
    nbins = length(edges) - 1
    counts = zeros(Int, nbins)
    for value in values
        bin = searchsortedlast(edges, value) - 1
        if bin < 1
            bin = 1
        elseif bin > nbins
            bin = nbins
        end
        counts[bin] += 1
    end
    probs = counts ./ max(length(values), 1)
    return counts, probs
end

function component_stats(values::Vector{Float64})
    return (
        mean = mean(values),
        std = std(values),
        min = minimum(values),
        q05 = quantile(values, 0.05),
        q25 = quantile(values, 0.25),
        median = median(values),
        q75 = quantile(values, 0.75),
        q95 = quantile(values, 0.95),
        max = maximum(values),
    )
end

function build_window_histogram_figure(proxy::Dict{String, Any}, bin_specs)
    values = proxy["log_perms"]
    window = proxy["window"]
    n = proxy["n_samples"]

    ymax = maximum(begin
        _, probs = histogram_probabilities(vec(values[:, ic]), bin_specs[ic].edges)
        maximum(probs)
    end for ic in 1:3)
    ytop = round(1.08 * ymax + 0.005; digits = 2)
    yticks = collect(range(0.0, ytop, length = 5))
    yticklabels = [@sprintf("%.2f", tick) for tick in yticks]
    fig = Figure(size = (1760, 650))
    Label(fig[0, :], "$window marginal histograms in log10(k) space", fontsize = 24, font = :bold)

    for ic in 1:3
        spec = bin_specs[ic]
        v = vec(values[:, ic])
        stats = component_stats(v)
        _, probs = histogram_probabilities(v, spec.edges)

        ax = Axis(fig[1, ic],
                  xlabel = spec.label,
                  ylabel = ic == 1 ? "Probability" : "",
                  title = COMPONENT_NAMES[ic],
                  titlealign = :left,
                  xlabelsize = 22,
                  ylabelsize = 22,
                  titlesize = 22,
                  xticklabelsize = 22,
                  yticklabelsize = 22,
                  topspinevisible = false,
                  rightspinevisible = false,
                  xticks = (LOGK_TICKS, LOGK_TICK_LABELS),
                  yticks = (yticks, yticklabels))

        barplot!(ax, spec.centers, probs;
                 width = 0.92 .* spec.widths,
                 color = (RGBf(0.35, 0.38, 0.42), 0.88),
                 strokecolor = (RGBf(0.15, 0.16, 0.18), 0.55),
                 strokewidth = 0.5)

        vlines!(ax, [stats.q25];
                color = RGBf(0.55, 0.55, 0.55),
                linestyle = :dash,
                linewidth = 1.8)
        vlines!(ax, [stats.median];
                color = RGBf(0.82, 0.24, 0.20),
                linestyle = :solid,
                linewidth = 2.4)
        vlines!(ax, [stats.q75];
                color = RGBf(0.55, 0.55, 0.55),
                linestyle = :dash,
                linewidth = 1.8)

        xlims!(ax, LOGK_LIMITS...)
        ylims!(ax, 0.0, ytop)

        text!(ax, 0.03, 0.985;
              space = :relative,
              align = (:left, :top),
              fontsize = 22,
              color = :black,
              text = @sprintf("n = %d\nmean = %.3f\nmedian = %.3f\nIQR = [%.3f, %.3f]",
                              n, stats.mean, stats.median, stats.q25, stats.q75))

        text!(ax, 0.98, 0.98;
              space = :relative,
              align = (:right, :top),
              fontsize = 22,
              font = :bold,
              color = :black,
              text = PANEL_LETTERS[ic])
    end

    Label(fig[2, :],
          "Bars show empirical probability per bin. Red line: median. Gray dashed lines: 25th and 75th percentiles.",
          fontsize = 22)
    return fig
end

function window_short_label(window)
    text = String(window)
    m = match(r"famp(\d+)", text)
    return isnothing(m) ? text : "W$(m.captures[1])"
end

function probability_tick_spec(ytop::Float64)
    yticks = collect(range(0.0, ytop, length = 5))
    yticklabels = [@sprintf("%.2f", tick) for tick in yticks]
    return yticks, yticklabels
end

function build_combined_publication_figure(proxies, bin_specs)
    n_windows = length(proxies)
    ytops = Float64[]
    for ic in 1:3
        ymax = maximum(begin
            _, probs = histogram_probabilities(vec(proxy["log_perms"][:, ic]), bin_specs[ic].edges)
            maximum(probs)
        end for proxy in proxies)
        push!(ytops, round(1.08 * ymax + 0.005; digits = 2))
    end

    title_font_size = 42
    header_font_size = 34
    axis_font_size = 34
    tick_font_size = 34
    note_font_size = 34
    panel_width = 430
    panel_height = 360

    fig = Figure(size = (n_windows * panel_width + 390, 3 * panel_height + 760),
                 backgroundcolor = :white)
    Label(fig[0, 1:n_windows],
          "All-window marginal histograms in raw log10(k) space",
          fontsize = title_font_size,
          font = :bold,
          halign = :center,
          tellwidth = false)

    for (iw, proxy) in enumerate(proxies)
        Label(fig[1, iw],
              window_short_label(proxy["window"]),
              fontsize = header_font_size,
              font = :bold,
              halign = :center,
              tellwidth = false)
    end

    for ic in 1:3
        spec = bin_specs[ic]
        yticks, yticklabels = probability_tick_spec(ytops[ic])
        axis_row = 2 * ic
        label_row = axis_row + 1

        for (iw, proxy) in enumerate(proxies)
            values = proxy["log_perms"]
            v = vec(values[:, ic])
            stats = component_stats(v)
            _, probs = histogram_probabilities(v, spec.edges)

            ax = Axis(fig[axis_row, iw],
                      width = panel_width,
                      height = panel_height,
                      ylabel = iw == 1 ? "Probability" : "",
                      xlabel = "",
                      xlabelsize = axis_font_size,
                      ylabelsize = axis_font_size,
                      xticklabelsize = tick_font_size,
                      yticklabelsize = tick_font_size,
                      topspinevisible = false,
                      rightspinevisible = false,
                      xgridvisible = true,
                      ygridvisible = true,
                      xgridcolor = RGBf(0.88, 0.88, 0.88),
                      ygridcolor = RGBf(0.88, 0.88, 0.88),
                      xgridwidth = 1.1,
                      ygridwidth = 1.1,
                      xticks = (LOGK_TICKS, LOGK_TICK_LABELS),
                      yticks = iw == 1 ? (yticks, yticklabels) : (yticks, fill("", length(yticks))))

            barplot!(ax, spec.centers, probs;
                     width = 0.92 .* spec.widths,
                     color = (RGBf(0.35, 0.38, 0.42), 0.88),
                     strokecolor = (RGBf(0.15, 0.16, 0.18), 0.45),
                     strokewidth = 0.45)

            vlines!(ax, [stats.q25, stats.q75];
                    color = RGBf(0.55, 0.55, 0.55),
                    linestyle = :dash,
                    linewidth = 2.1)
            vlines!(ax, [stats.median];
                    color = RGBf(0.82, 0.24, 0.20),
                    linestyle = :solid,
                    linewidth = 2.8)

            xlims!(ax, LOGK_LIMITS...)
            ylims!(ax, 0.0, ytops[ic])

            text!(ax, 0.965, 0.965;
                  space = :relative,
                  align = (:right, :top),
                  fontsize = 34,
                  font = :bold,
                  color = :black,
                  text = panel_label((ic - 1) * n_windows + iw))
        end

        Label(fig[label_row, 1:n_windows],
              COMPONENT_LABELS[ic],
              fontsize = axis_font_size,
              halign = :center,
              tellwidth = false)
    end

    bar_elem = PolyElement(polycolor = (RGBf(0.35, 0.38, 0.42), 0.88),
                           strokecolor = (RGBf(0.15, 0.16, 0.18), 0.45),
                           strokewidth = 0.45)
    median_elem = LineElement(color = RGBf(0.82, 0.24, 0.20),
                              linestyle = :solid,
                              linewidth = 4)
    quartile_elem = LineElement(color = RGBf(0.55, 0.55, 0.55),
                                linestyle = :dash,
                                linewidth = 3)

    Legend(fig[8, 1:n_windows],
           [bar_elem, median_elem, quartile_elem],
           ["Empirical bin probability", "Median", "25th and 75th percentiles"];
           orientation = :horizontal,
           framevisible = false,
           labelsize = note_font_size,
           tellwidth = false,
           halign = :center)

    Label(fig[9, 1:n_windows],
          "Rows use component-specific global bin edges; all panels use the same log10(k) axis ticks for direct visual comparison.",
          fontsize = note_font_size,
          halign = :center,
          tellwidth = false)

    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 18)
    return fig
end

function build_overview_figure(proxies, bin_specs)
    fig = Figure(size = (1850, 2200))
    Label(fig[0, :], "Marginal histogram overview across the six fixed windows", fontsize = 24, font = :bold)

    ymax = maximum(begin
        _, probs = histogram_probabilities(vec(proxy["log_perms"][:, ic]), bin_specs[ic].edges)
        maximum(probs)
    end for proxy in proxies for ic in 1:3)

    for ic in 1:3
        Label(fig[1, ic], COMPONENT_LABELS[ic], fontsize = 20, font = :bold)
    end

    for (iw, proxy) in enumerate(proxies)
        window = proxy["window"]
        values = proxy["log_perms"]
        for ic in 1:3
            spec = bin_specs[ic]
            v = vec(values[:, ic])
            stats = component_stats(v)
            _, probs = histogram_probabilities(v, spec.edges)
            ax = Axis(fig[iw + 1, ic],
                      xlabel = iw == length(proxies) ? spec.label : "",
                      ylabel = ic == 1 ? "$(window)\nProbability" : "",
                      xlabelsize = 16,
                      ylabelsize = 15,
                      xticklabelsize = 11,
                      yticklabelsize = 11,
                      topspinevisible = false,
                      rightspinevisible = false,
                      xticks = (LOGK_TICKS, LOGK_TICK_LABELS))

            barplot!(ax, spec.centers, probs;
                     width = 0.92 .* spec.widths,
                     color = (RGBf(0.35, 0.38, 0.42), 0.86),
                     strokecolor = (RGBf(0.15, 0.16, 0.18), 0.45),
                     strokewidth = 0.4)

            vlines!(ax, [stats.q25];
                    color = RGBf(0.55, 0.55, 0.55),
                    linestyle = :dash,
                    linewidth = 1.3)
            vlines!(ax, [stats.median];
                    color = RGBf(0.82, 0.24, 0.20),
                    linestyle = :solid,
                    linewidth = 1.8)
            vlines!(ax, [stats.q75];
                    color = RGBf(0.55, 0.55, 0.55),
                    linestyle = :dash,
                    linewidth = 1.3)

            xlims!(ax, LOGK_LIMITS...)
            ylims!(ax, 0.0, 1.08 * ymax)
        end
    end

    Label(fig[length(proxies) + 2, :],
          "All windows share the same component-specific bin edges to make shape comparisons easier.",
          fontsize = 17)
    return fig
end

function build_summary_rows(proxy::Dict{String, Any}, bin_specs)
    rows = Vector{Vector{String}}()
    values = proxy["log_perms"]
    window = proxy["window"]
    n = proxy["n_samples"]
    for ic in 1:3
        stats = component_stats(vec(values[:, ic]))
        spec = bin_specs[ic]
        push!(rows, [
            String(window),
            String(COMPONENT_NAMES[ic]),
            string(n),
            fmt(stats.mean),
            fmt(stats.std),
            fmt(stats.min),
            fmt(stats.q05),
            fmt(stats.q25),
            fmt(stats.median),
            fmt(stats.q75),
            fmt(stats.q95),
            fmt(stats.max),
            fmt(spec.xlim[1]),
            fmt(spec.xlim[2]),
            fmt(spec.bin_width),
        ])
    end
    return rows
end

fmt(value::Real) = @sprintf("%.6f", Float64(value))

main(ARGS)
