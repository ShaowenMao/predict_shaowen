#!/usr/bin/env julia

"""
Build a compact cross-window summary from completed componentwise Level 2 outputs.

This script reads the finalized componentwise outputs for multiple windows and
combines the most important metrics into two compact CSV tables:

1. component-level summary (one row per window/component)
2. window-level summary (one row per window)

The summary is intended for cross-window comparison only; it does not rerun any
analysis. It reads:
    - step-4 mode summaries
    - step-6 bootstrap mode-count and mode-existence summaries
    - reference reproducibility diagnostics
    - componentwise h-summary tables

It also computes a simple prototype component complexity proxy:

    complexity = 0.5 * mean Var(CDF indicators) + 0.5 * mean Var(mode indicators)

This proxy is only a normalized comparison aid. It is not the final allocation
score for field-scale simulations.
"""

using Printf
using Statistics

function parse_args(args::Vector{String})
    options = Dict(
        "mode-dir" => raw"D:\codex_gom\reference_componentwise_mode_screening_test_v19",
        "bootstrap-dir" => raw"D:\codex_gom\reference_componentwise_bootstrap_test_v4000_figs",
        "h-dir" => raw"D:\codex_gom\reference_componentwise_h_test_v1",
        "windows" => "famp1",
        "output-dir" => raw"D:\codex_gom\reference_componentwise_cross_window_summary",
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

    windows = String[strip(w) for w in split(options["windows"], ",") if !isempty(strip(w))]
    isempty(windows) && error("No windows specified")

    return (
        mode_dir = options["mode-dir"],
        bootstrap_dir = options["bootstrap-dir"],
        h_dir = options["h-dir"],
        output_dir = options["output-dir"],
        windows = windows,
    )
end

function print_help()
    println("Usage:")
    println("  julia predict_reference_componentwise_cross_window_summary.jl [options]")
    println()
    println("Options:")
    println("  --mode-dir <path>         Root folder with <window>/<window>_componentwise_mode_summary.csv")
    println("  --bootstrap-dir <path>    Root folder with bootstrap summaries")
    println("  --h-dir <path>            Root folder with componentwise h-summary outputs")
    println("  --windows <names>         Comma-separated list like famp1,famp2,famp3")
    println("  --output-dir <path>       Folder for the combined CSV summaries")
    println("  -h, --help                Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    component_rows = NamedTuple[]
    window_rows = NamedTuple[]

    for window in opt.windows
        push!(component_rows, collect_component_rows(window, opt)...)
    end

    window_names = unique(row.window for row in component_rows)
    for window in window_names
        push!(window_rows, build_window_row(window, component_rows))
    end

    sort!(component_rows, by = row -> (row.window, component_order(row.component)))
    sort!(window_rows, by = row -> row.window)

    component_path = joinpath(opt.output_dir, "componentwise_cross_window_component_summary.csv")
    window_path = joinpath(opt.output_dir, "componentwise_cross_window_window_summary.csv")

    write_component_summary(component_path, component_rows)
    write_window_summary(window_path, window_rows)

    println("Saved:")
    println("  - " * component_path)
    println("  - " * window_path)
end

component_order(name::AbstractString) = name == "kxx" ? 1 : name == "kyy" ? 2 : 3

function collect_component_rows(window::AbstractString, opt)
    mode_summary = read_csv_rows(joinpath(opt.mode_dir, window, "$(window)_componentwise_mode_summary.csv"))
    mode_prob_rows = read_csv_rows(joinpath(opt.bootstrap_dir, window, "$(window)_componentwise_bootstrap_mode_count_probabilities.csv"))
    existence_rows = read_csv_rows(joinpath(opt.bootstrap_dir, window, "$(window)_componentwise_bootstrap_mode_existence_summary.csv"))
    reproducibility_rows = read_csv_rows(joinpath(opt.bootstrap_dir, window, "$(window)_componentwise_reference_reproducibility.csv"))
    h_summary_rows = read_csv_rows(joinpath(opt.h_dir, window, "$(window)_componentwise_h_summary.csv"))

    out = NamedTuple[]
    for row in mode_summary
        component = row["component"]
        mode_probs = component_mode_probs(mode_prob_rows, component)
        existences = component_mode_existences(existence_rows, component)
        reproducibility = component_reproducibility(reproducibility_rows, component)
        h_stats = component_h_stats(h_summary_rows, component)

        top_count, top_prob = most_likely_mode_count(mode_probs)
        push!(out, (
            window = String(window),
            component = String(component),
            baseline_stable_mode_count = parse(Int, row["stable_mode_count"]),
            most_likely_bootstrap_mode_count = top_count,
            most_likely_bootstrap_mode_probability = top_prob,
            p_mode_1 = get(mode_probs, 1, 0.0),
            p_mode_2 = get(mode_probs, 2, 0.0),
            p_mode_3 = get(mode_probs, 3, 0.0),
            min_mode_existence_probability = isempty(existences) ? NaN : minimum(existences),
            max_mode_existence_probability = isempty(existences) ? NaN : maximum(existences),
            mode_existence_probabilities = join(fmt.(existences), ";"),
            global_mean = parse(Float64, row["global_mean"]),
            within_mode_spread_W = parse(Float64, row["within_mode_spread_W"]),
            between_mode_separation_B = parse(Float64, row["between_mode_separation_B"]),
            mode_entropy_H = parse(Float64, row["mode_entropy_H"]),
            chosen_bandwidth_factor = parse(Float64, row["chosen_bandwidth_factor"]),
            max_reference_ecdf_supnorm = reproducibility.max_ecdf_supnorm,
            max_reference_exceedance_supnorm = reproducibility.max_exceedance_supnorm,
            max_reference_abs_q05_diff = reproducibility.max_abs_q05_diff,
            max_reference_abs_q50_diff = reproducibility.max_abs_q50_diff,
            max_reference_abs_q95_diff = reproducibility.max_abs_q95_diff,
            num_cdf_features = h_stats.num_cdf_features,
            num_mode_indicator_features = h_stats.num_mode_indicator_features,
            num_mode_moment_features = h_stats.num_mode_moment_features,
            mean_cdf_indicator_variance = h_stats.mean_cdf_indicator_variance,
            max_cdf_indicator_variance = h_stats.max_cdf_indicator_variance,
            mean_mode_indicator_variance = h_stats.mean_mode_indicator_variance,
            max_mode_indicator_variance = h_stats.max_mode_indicator_variance,
            prototype_component_complexity = h_stats.prototype_component_complexity,
        ))
    end
    return out
end

function component_mode_probs(rows, component::AbstractString)
    out = Dict{Int, Float64}()
    for row in rows
        row["component"] == component || continue
        out[parse(Int, row["mode_count"])] = parse(Float64, row["probability"])
    end
    return out
end

function component_mode_existences(rows, component::AbstractString)
    out = Float64[]
    for row in rows
        row["component"] == component || continue
        push!(out, parse(Float64, row["existence_probability"]))
    end
    return out
end

function component_reproducibility(rows, component::AbstractString)
    ecdf = Float64[]
    exceed = Float64[]
    q05 = Float64[]
    q50 = Float64[]
    q95 = Float64[]
    for row in rows
        row["component"] == component || continue
        push!(ecdf, parse(Float64, row["ecdf_supnorm"]))
        push!(exceed, parse(Float64, row["exceedance_supnorm"]))
        push!(q05, parse(Float64, row["abs_q05_diff"]))
        push!(q50, parse(Float64, row["abs_q50_diff"]))
        push!(q95, parse(Float64, row["abs_q95_diff"]))
    end
    return (
        max_ecdf_supnorm = isempty(ecdf) ? NaN : maximum(ecdf),
        max_exceedance_supnorm = isempty(exceed) ? NaN : maximum(exceed),
        max_abs_q05_diff = isempty(q05) ? NaN : maximum(q05),
        max_abs_q50_diff = isempty(q50) ? NaN : maximum(q50),
        max_abs_q95_diff = isempty(q95) ? NaN : maximum(q95),
    )
end

function component_h_stats(rows, component::AbstractString)
    cdf_vars = Float64[]
    mode_vars = Float64[]
    num_mode_moment_features = 0

    for row in rows
        row["component"] == component || continue
        family = row["family"]
        feature_variance = parse(Float64, row["feature_variance"])
        if family == "cdf_indicator"
            push!(cdf_vars, feature_variance)
        elseif family == "mode_indicator"
            push!(mode_vars, feature_variance)
        elseif family == "mode_first_moment" || family == "mode_second_moment"
            num_mode_moment_features += 1
        end
    end

    mean_cdf = isempty(cdf_vars) ? NaN : mean(cdf_vars)
    mean_mode = isempty(mode_vars) ? NaN : mean(mode_vars)
    proto = if !isnan(mean_cdf) && !isnan(mean_mode)
        0.5 * mean_cdf + 0.5 * mean_mode
    elseif !isnan(mean_cdf)
        mean_cdf
    else
        mean_mode
    end

    return (
        num_cdf_features = length(cdf_vars),
        num_mode_indicator_features = length(mode_vars),
        num_mode_moment_features = num_mode_moment_features,
        mean_cdf_indicator_variance = mean_cdf,
        max_cdf_indicator_variance = isempty(cdf_vars) ? NaN : maximum(cdf_vars),
        mean_mode_indicator_variance = mean_mode,
        max_mode_indicator_variance = isempty(mode_vars) ? NaN : maximum(mode_vars),
        prototype_component_complexity = proto,
    )
end

function most_likely_mode_count(mode_probs::Dict{Int, Float64})
    isempty(mode_probs) && return (0, NaN)
    best_count = 0
    best_prob = -Inf
    for (count, prob) in mode_probs
        if prob > best_prob || (prob == best_prob && count < best_count)
            best_count = count
            best_prob = prob
        end
    end
    return (best_count, best_prob)
end

function build_window_row(window::AbstractString, component_rows)
    rows = [row for row in component_rows if row.window == window]
    complexities = [row.prototype_component_complexity for row in rows if !isnan(row.prototype_component_complexity)]
    mode_support_strings = [
        row.component * ":P2=" * fmt(row.p_mode_2) * ",min_exist=" * fmt(row.min_mode_existence_probability)
        for row in rows
    ]
    complexity_strings = [
        row.component * ":" * fmt(row.prototype_component_complexity)
        for row in rows
    ]

    robustly_bimodal = count(row -> row.p_mode_2 >= 0.9, rows)
    ambiguous = count(row -> row.most_likely_bootstrap_mode_probability < 0.9, rows)

    return (
        window = String(window),
        num_components = length(rows),
        num_components_baseline_2mode = count(row -> row.baseline_stable_mode_count == 2, rows),
        num_components_with_p2_ge_0_9 = robustly_bimodal,
        num_components_ambiguous = ambiguous,
        mean_prototype_component_complexity = isempty(complexities) ? NaN : mean(complexities),
        max_prototype_component_complexity = isempty(complexities) ? NaN : maximum(complexities),
        component_complexities = join(complexity_strings, ";"),
        component_mode_support = join(mode_support_strings, ";"),
    )
end

function read_csv_rows(filepath::AbstractString)
    isfile(filepath) || error("Missing CSV: $filepath")
    lines = readlines(filepath)
    isempty(lines) && return Dict{String, String}[]
    header = split(lines[1], ",")
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cols = split(line, ",")
        length(cols) == length(header) || error("Column mismatch in $filepath:\n$line")
        push!(rows, Dict(header[i] => cols[i] for i in eachindex(header)))
    end
    return rows
end

function write_component_summary(filepath::AbstractString, rows)
    header = [
        "window", "component",
        "baseline_stable_mode_count",
        "most_likely_bootstrap_mode_count", "most_likely_bootstrap_mode_probability",
        "p_mode_1", "p_mode_2", "p_mode_3",
        "min_mode_existence_probability", "max_mode_existence_probability", "mode_existence_probabilities",
        "global_mean", "within_mode_spread_W", "between_mode_separation_B", "mode_entropy_H",
        "chosen_bandwidth_factor",
        "max_reference_ecdf_supnorm", "max_reference_exceedance_supnorm",
        "max_reference_abs_q05_diff", "max_reference_abs_q50_diff", "max_reference_abs_q95_diff",
        "num_cdf_features", "num_mode_indicator_features", "num_mode_moment_features",
        "mean_cdf_indicator_variance", "max_cdf_indicator_variance",
        "mean_mode_indicator_variance", "max_mode_indicator_variance",
        "prototype_component_complexity",
    ]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            values = [
                row.window,
                row.component,
                string(row.baseline_stable_mode_count),
                string(row.most_likely_bootstrap_mode_count),
                fmt(row.most_likely_bootstrap_mode_probability),
                fmt(row.p_mode_1),
                fmt(row.p_mode_2),
                fmt(row.p_mode_3),
                fmt(row.min_mode_existence_probability),
                fmt(row.max_mode_existence_probability),
                row.mode_existence_probabilities,
                fmt(row.global_mean),
                fmt(row.within_mode_spread_W),
                fmt(row.between_mode_separation_B),
                fmt(row.mode_entropy_H),
                fmt(row.chosen_bandwidth_factor),
                fmt(row.max_reference_ecdf_supnorm),
                fmt(row.max_reference_exceedance_supnorm),
                fmt(row.max_reference_abs_q05_diff),
                fmt(row.max_reference_abs_q50_diff),
                fmt(row.max_reference_abs_q95_diff),
                string(row.num_cdf_features),
                string(row.num_mode_indicator_features),
                string(row.num_mode_moment_features),
                fmt(row.mean_cdf_indicator_variance),
                fmt(row.max_cdf_indicator_variance),
                fmt(row.mean_mode_indicator_variance),
                fmt(row.max_mode_indicator_variance),
                fmt(row.prototype_component_complexity),
            ]
            println(io, join(values, ","))
        end
    end
end

function write_window_summary(filepath::AbstractString, rows)
    header = [
        "window", "num_components", "num_components_baseline_2mode",
        "num_components_with_p2_ge_0_9", "num_components_ambiguous",
        "mean_prototype_component_complexity", "max_prototype_component_complexity",
        "component_complexities", "component_mode_support",
    ]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            values = [
                row.window,
                string(row.num_components),
                string(row.num_components_baseline_2mode),
                string(row.num_components_with_p2_ge_0_9),
                string(row.num_components_ambiguous),
                fmt(row.mean_prototype_component_complexity),
                fmt(row.max_prototype_component_complexity),
                row.component_complexities,
                row.component_mode_support,
            ]
            println(io, join(values, ","))
        end
    end
end

fmt(x) = (x isa Number && isfinite(x)) ? @sprintf("%.10g", x) : ""

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
