#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_core.jl"))

using .Level2IO
using .Level2Core

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "manifest" => Level2IO.default_manifest_path(),
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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/build_level2_window_states.jl [options]")
    println()
    println("Note:")
    println("  This script begins Step 2.2 (local ranks and local normal scores).")
    println("  Run plot_marginal_hist_screening.jl first if you want the pre-Step-2.2")
    println("  marginal histogram review in log10(k) space.")
    println()
    println("Options:")
    println("  --config <path>       Level 2 TOML config")
    println("  --manifest <path>     Proxy manifest CSV")
    println("  --output-dir <path>   Output root for saved window-state objects")
    println("  -h, --help            Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    manifest_rows = Level2IO.read_manifest_csv(opt["manifest"], config)

    output_root = isempty(opt["output-dir"]) ?
        normpath(joinpath(Level2IO.default_level2_output_root(), config["geology_id"])) :
        normpath(opt["output-dir"])

    state_root = joinpath(output_root, "window_states")
    table_root = joinpath(output_root, "tables")
    report_root = joinpath(output_root, "reports")
    mkpath(state_root)
    mkpath(table_root)
    mkpath(report_root)

    build_rows = Vector{Vector{String}}()
    report_lines = String[
        "Level 2 build report",
        "created_at = $(Level2IO.timestamp_string())",
        "config_path = $(config["config_path"])",
        "manifest_path = $(normpath(opt["manifest"]))",
        "geology_id = $(config["geology_id"])",
        "output_root = $output_root",
        "windows = $(join(config["fixed_windows"], ", "))",
        "",
    ]

    for row in manifest_rows
        proxy = Level2IO.load_proxy_library(row)
        window = proxy["window"]
        println("Building Level 2 state for $window from $(proxy["source_path"])")

        state = Level2Core.build_window_state(proxy["log_perms"],
                                              proxy["raw_perms"],
                                              window,
                                              proxy["source_path"],
                                              proxy["source_label"],
                                              config)

        window_dir = joinpath(state_root, window)
        state_path = joinpath(window_dir, "$(window)_level2_state.mat")
        Level2IO.save_window_state(state_path, state)

        push!(build_rows, [
            window,
            proxy["source_path"],
            string(state["n_samples"]),
            string(state["chosen_k"]),
            string(round(Float64(state["best_silhouette"]), digits = 6)),
            string(Int(state["is_effectively_unimodal"])),
            string(length(vec(state["low_indices"]))),
            string(length(vec(state["high_indices"]))),
            string(length(vec(state["central_indices"]))),
            state_path,
        ])

        push!(report_lines, "window = $window")
        push!(report_lines, "  chosen_k = $(state["chosen_k"])")
        push!(report_lines, "  best_silhouette = $(round(Float64(state["best_silhouette"]), digits = 6))")
        push!(report_lines, "  unimodal = $(Bool(Int(state["is_effectively_unimodal"])))")
        push!(report_lines, "  low_n = $(length(vec(state["low_indices"])))")
        push!(report_lines, "  high_n = $(length(vec(state["high_indices"])))")
        push!(report_lines, "  central_n = $(length(vec(state["central_indices"])))")
        push!(report_lines, "")
    end

    Level2IO.write_csv(joinpath(table_root, "level2_build_summary.csv"),
                       ["window", "source_path", "n_samples", "chosen_k", "best_silhouette",
                        "is_effectively_unimodal", "low_n", "high_n", "central_n", "state_path"],
                       build_rows)
    Level2IO.write_text_lines(joinpath(report_root, "level2_build_report.txt"), report_lines)

    println("Saved Level 2 window states to $output_root")
end

main(ARGS)
