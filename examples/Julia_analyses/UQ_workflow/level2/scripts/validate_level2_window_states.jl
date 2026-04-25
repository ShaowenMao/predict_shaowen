#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using Printf
using MAT: matread

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_core.jl"))

using .Level2IO
using .Level2Core

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "state-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref")),
        "data-root" => normpath(joinpath(Level2IO.REPO_ROOT, "examples", "gom_reference_floor_full", "data")),
        "output-dir" => "",
        "holdout-repeats" => "",
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/validate_level2_window_states.jl [options]")
    println()
    println("Options:")
    println("  --config <path>           Level 2 TOML config")
    println("  --state-root <path>       Root folder containing built Level 2 states")
    println("  --data-root <path>        Root folder containing the proxy holdout MAT files")
    println("  --output-dir <path>       Output folder for validation tables and report")
    println("  --holdout-repeats <list>  Comma-separated repeat ids like 2,3,4,5")
    println("  -h, --help                Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    holdout_repeats = isempty(opt["holdout-repeats"]) ?
        config["holdout_repeats"] :
        Int[parse(Int, strip(token)) for token in split(opt["holdout-repeats"], ",") if !isempty(strip(token))]

    isempty(holdout_repeats) && error("No holdout repeats requested")

    state_root = normpath(opt["state-root"])
    data_root = normpath(opt["data-root"])
    output_root = isempty(opt["output-dir"]) ? joinpath(state_root, "validation") : normpath(opt["output-dir"])
    table_root = joinpath(output_root, "tables")
    report_root = joinpath(output_root, "reports")
    mkpath(table_root)
    mkpath(report_root)

    pair_rows = Vector{Vector{String}}()
    summary_rows = Vector{Vector{String}}()
    report_lines = String[
        "Level 2 validation report",
        "created_at = $(Level2IO.timestamp_string())",
        "state_root = $state_root",
        "data_root = $data_root",
        "holdout_repeats = $(join(string.(holdout_repeats), ", "))",
        "",
    ]

    for window in Level2IO.FIXED_WINDOWS
        state_path = joinpath(state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing built state for $window: $state_path")
        reference_state = Level2IO.load_window_state(state_path)

        window_pair_rows = Vector{Vector{String}}()
        for repeat_id in holdout_repeats
            holdout_path = joinpath(data_root, window, "small_runs", @sprintf("N2000_repeat%02d.mat", repeat_id))
            isfile(holdout_path) || error("Missing holdout MAT file: $holdout_path")

            data = matread(holdout_path)
            haskey(data, "perms") || error("Holdout MAT file does not contain perms: $holdout_path")
            raw_perms = Matrix{Float64}(data["perms"])
            holdout_state = Level2Core.build_window_state(log10.(raw_perms),
                                                          raw_perms,
                                                          window,
                                                          holdout_path,
                                                          "$(window) holdout repeat $(repeat_id)",
                                                          config)
            metrics = Level2Core.compare_window_states(reference_state, holdout_state)

            row = [
                window,
                string(repeat_id),
                metrics["same_k"],
                metrics["same_unimodality"],
                metrics["reference_k"],
                metrics["holdout_k"],
                metrics["reference_silhouette"],
                metrics["holdout_silhouette"],
                metrics["abs_silhouette_delta"],
                metrics["global_medoid_distance"],
                metrics["low_medoid_distance"],
                metrics["high_medoid_distance"],
                metrics["central_medoid_distance"],
                metrics["global_mean_distance"],
                metrics["low_mean_distance"],
                metrics["high_mean_distance"],
                metrics["central_mean_distance"],
            ]
            push!(window_pair_rows, row)
            push!(pair_rows, row)
        end

        push!(summary_rows, summarize_window_validation(window, window_pair_rows))
        push!(report_lines, "window = $window")
        push!(report_lines, "  holdout_pairs = $(length(window_pair_rows))")
        push!(report_lines, "  same_k_rate = $(summary_rows[end][2])")
        push!(report_lines, "  same_unimodality_rate = $(summary_rows[end][3])")
        push!(report_lines, "  mean_abs_silhouette_delta = $(summary_rows[end][4])")
        push!(report_lines, "  mean_global_medoid_distance = $(summary_rows[end][5])")
        push!(report_lines, "")
    end

    Level2IO.write_csv(joinpath(table_root, "holdout_validation_pairs.csv"),
                       ["window", "repeat_id", "same_k", "same_unimodality", "reference_k",
                        "holdout_k", "reference_silhouette", "holdout_silhouette",
                        "abs_silhouette_delta", "global_medoid_distance", "low_medoid_distance",
                        "high_medoid_distance", "central_medoid_distance", "global_mean_distance",
                        "low_mean_distance", "high_mean_distance", "central_mean_distance"],
                       pair_rows)
    Level2IO.write_csv(joinpath(table_root, "holdout_validation_summary.csv"),
                       ["window", "same_k_rate", "same_unimodality_rate", "mean_abs_silhouette_delta",
                        "mean_global_medoid_distance", "mean_low_medoid_distance",
                        "mean_high_medoid_distance", "mean_central_medoid_distance"],
                       summary_rows)
    Level2IO.write_text_lines(joinpath(report_root, "level2_validation_report.txt"), report_lines)

    println("Saved Level 2 validation outputs to $output_root")
end

function summarize_window_validation(window::AbstractString, rows::Vector{Vector{String}})
    same_k_rate = mean(parse.(Float64, getindex.(rows, 3)))
    same_unimodality_rate = mean(parse.(Float64, getindex.(rows, 4)))
    mean_abs_silhouette_delta = mean(parse.(Float64, getindex.(rows, 9)))
    mean_global_medoid_distance = mean(parse.(Float64, getindex.(rows, 10)))
    mean_low_medoid_distance = mean(parse.(Float64, getindex.(rows, 11)))
    mean_high_medoid_distance = mean(parse.(Float64, getindex.(rows, 12)))
    mean_central_medoid_distance = mean(parse.(Float64, getindex.(rows, 13)))

    return [
        window,
        string(round(same_k_rate, digits = 6)),
        string(round(same_unimodality_rate, digits = 6)),
        string(round(mean_abs_silhouette_delta, digits = 6)),
        string(round(mean_global_medoid_distance, digits = 6)),
        string(round(mean_low_medoid_distance, digits = 6)),
        string(round(mean_high_medoid_distance, digits = 6)),
        string(round(mean_central_medoid_distance, digits = 6)),
    ]
end

mean(values::Vector{Float64}) = sum(values) / length(values)

main(ARGS)
