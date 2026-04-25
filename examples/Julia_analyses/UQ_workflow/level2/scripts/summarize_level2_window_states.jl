#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using Statistics

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))

using .Level2IO

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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/scripts/summarize_level2_window_states.jl [options]")
    println()
    println("Options:")
    println("  --state-root <path>    Root folder containing built window-state MAT files")
    println("  --output-dir <path>    Output folder for summary tables and report")
    println("  -h, --help             Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    state_root = normpath(opt["state-root"])
    output_root = isempty(opt["output-dir"]) ? state_root : normpath(opt["output-dir"])
    table_root = joinpath(output_root, "tables")
    point_table_root = joinpath(table_root, "window_point_tables")
    report_root = joinpath(output_root, "reports")
    mkpath(table_root)
    mkpath(point_table_root)
    mkpath(report_root)

    window_rows = Vector{Vector{String}}()
    cluster_rows = Vector{Vector{String}}()
    state_rows = Vector{Vector{String}}()
    report_lines = String[
        "Level 2 summary report",
        "created_at = $(Level2IO.timestamp_string())",
        "state_root = $state_root",
        "",
    ]

    for window in Level2IO.FIXED_WINDOWS
        state_path = joinpath(state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing window-state MAT file: $state_path")
        state = Level2IO.load_window_state(state_path)
        Level2IO.write_window_point_table(joinpath(point_table_root, "$(window)_level2_points.csv"), state)

        chosen_k = int_scalar(state["chosen_k"])
        silhouette = float_scalar(state["best_silhouette"])
        unimodal = int_scalar(state["is_effectively_unimodal"])
        low_n = length(vector_int(state["low_indices"]))
        high_n = length(vector_int(state["high_indices"]))
        central_n = length(vector_int(state["central_indices"]))

        push!(window_rows, [
            window,
            string(chosen_k),
            string(round(silhouette, digits = 6)),
            string(unimodal),
            string(low_n),
            string(high_n),
            string(central_n),
            string(int_scalar(state["global_medoid_index"])),
        ])

        cluster_sizes = vector_int(state["cluster_sizes"])
        cluster_medians = vector_float(state["cluster_score_medians"])
        cluster_medoids = vector_int(state["cluster_medoids"])
        cluster_order = vector_int(state["cluster_order"])
        for cluster_id in 1:length(cluster_sizes)
            push!(cluster_rows, [
                window,
                string(cluster_id),
                string(cluster_sizes[cluster_id]),
                string(round(cluster_medians[cluster_id], digits = 6)),
                string(cluster_medoids[cluster_id]),
                string(cluster_order[cluster_id]),
            ])
        end

        for label in ("low", "high", "central")
            scores = vector_float(state["state_score"])[vector_int(state["$(label)_indices"])]
            mean_log = vector_float(state["$(label)_mean_log_perm"])
            push!(state_rows, [
                window,
                label,
                string(length(scores)),
                string(int_scalar(state["$(label)_medoid_index"])),
                string(round(minimum(scores), digits = 6)),
                string(round(median(scores), digits = 6)),
                string(round(maximum(scores), digits = 6)),
                string(round(mean_log[1], digits = 6)),
                string(round(mean_log[2], digits = 6)),
                string(round(mean_log[3], digits = 6)),
            ])
        end

        push!(report_lines, "window = $window")
        push!(report_lines, "  chosen_k = $chosen_k")
        push!(report_lines, "  silhouette = $(round(silhouette, digits = 6))")
        push!(report_lines, "  unimodal = $(Bool(unimodal))")
        push!(report_lines, "  state_sizes = low:$low_n high:$high_n central:$central_n")
        push!(report_lines, "  point_table = $(joinpath(point_table_root, "$(window)_level2_points.csv"))")
        push!(report_lines, "")
    end

    Level2IO.write_csv(joinpath(table_root, "window_state_summary.csv"),
                       ["window", "chosen_k", "best_silhouette", "is_effectively_unimodal",
                        "low_n", "high_n", "central_n", "global_medoid_index"],
                       window_rows)
    Level2IO.write_csv(joinpath(table_root, "cluster_summary.csv"),
                       ["window", "cluster_id", "cluster_size", "cluster_median_state_score",
                        "cluster_medoid_index", "cluster_order_rank"],
                       cluster_rows)
    Level2IO.write_csv(joinpath(table_root, "state_library_summary.csv"),
                       ["window", "state_label", "library_size", "medoid_index", "score_min",
                        "score_median", "score_max", "mean_log_kxx", "mean_log_kyy", "mean_log_kzz"],
                       state_rows)
    Level2IO.write_text_lines(joinpath(report_root, "level2_summary_report.txt"), report_lines)

    println("Saved Level 2 summaries to $output_root")
end

float_scalar(value) = value isa AbstractArray ? Float64(first(vec(value))) : Float64(value)
int_scalar(value) = Int(round(float_scalar(value)))
vector_int(values) = values isa AbstractArray ? vec(Int.(values)) : [Int(values)]
vector_float(values) = values isa AbstractArray ? vec(Float64.(values)) : [Float64(values)]

main(ARGS)
