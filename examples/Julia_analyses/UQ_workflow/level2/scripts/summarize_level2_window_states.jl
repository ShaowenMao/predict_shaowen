#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using LinearAlgebra
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
    neighborhood_table_root = joinpath(table_root, "neighborhood_tables")
    report_root = joinpath(output_root, "reports")
    mkpath(table_root)
    mkpath(point_table_root)
    mkpath(neighborhood_table_root)
    mkpath(report_root)

    window_rows = Vector{Vector{String}}()
    cluster_rows = Vector{Vector{String}}()
    state_rows = Vector{Vector{String}}()
    neighborhood_summary_rows = Vector{Vector{String}}()
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
        write_neighborhood_tables!(
            neighborhood_summary_rows,
            joinpath(neighborhood_table_root, "$(window)_neighborhood_neighbors.csv"),
            state,
            window,
        )

        chosen_k = int_scalar(state["chosen_k"])
        silhouette = float_scalar(state["best_silhouette"])
        unimodal = int_scalar(state["is_effectively_unimodal"])
        low_n = length(vector_int(state["low_indices"]))
        high_n = length(vector_int(state["high_indices"]))
        central_n = length(vector_int(state["central_indices"]))
        distance_metric = get(state, "distance_metric", "local_normal")
        distance_scales = haskey(state, "distance_component_scales") ?
            join(string.(round.(vector_float(state["distance_component_scales"]), digits = 6)), ";") :
            "1.0;1.0;1.0"
        min_cluster_fraction = haskey(state, "min_cluster_fraction") ?
            string(round(float_scalar(state["min_cluster_fraction"]), digits = 4)) :
            ""
        min_cluster_size_floor = haskey(state, "min_cluster_size") ?
            int_scalar(state["min_cluster_size"]) :
            0
        effective_min_cluster_size = haskey(state, "n_samples") && !isempty(min_cluster_fraction) ?
            max(min_cluster_size_floor, ceil(Int, parse(Float64, min_cluster_fraction) * int_scalar(state["n_samples"]))) :
            min_cluster_size_floor

        push!(window_rows, [
            window,
            string(distance_metric),
            distance_scales,
            min_cluster_fraction,
            string(min_cluster_size_floor),
            string(effective_min_cluster_size),
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
            log_perms = matrix_float(state["log_perms"])
            medoid_index = int_scalar(state["$(label)_medoid_index"])
            medoid_log_perm = vec(log_perms[medoid_index, :])
            push!(state_rows, [
                window,
                label,
                string(length(scores)),
                string(medoid_index),
                string(round(medoid_log_perm[1], digits = 6)),
                string(round(medoid_log_perm[2], digits = 6)),
                string(round(medoid_log_perm[3], digits = 6)),
                string(round(minimum(scores), digits = 6)),
                string(round(median(scores), digits = 6)),
                string(round(maximum(scores), digits = 6)),
                string(round(mean_log[1], digits = 6)),
                string(round(mean_log[2], digits = 6)),
                string(round(mean_log[3], digits = 6)),
            ])
        end

        push!(report_lines, "window = $window")
        push!(report_lines, "  distance_metric = $distance_metric")
        push!(report_lines, "  distance_component_scales = $distance_scales")
        push!(report_lines, "  min_cluster_fraction = $min_cluster_fraction")
        push!(report_lines, "  effective_min_cluster_size = $effective_min_cluster_size")
        push!(report_lines, "  chosen_k = $chosen_k")
        push!(report_lines, "  silhouette = $(round(silhouette, digits = 6))")
        push!(report_lines, "  unimodal = $(Bool(unimodal))")
        push!(report_lines, "  state_sizes = low:$low_n high:$high_n central:$central_n")
        push!(report_lines, "  point_table = $(joinpath(point_table_root, "$(window)_level2_points.csv"))")
        push!(report_lines, "  neighborhood_table = $(joinpath(neighborhood_table_root, "$(window)_neighborhood_neighbors.csv"))")
        push!(report_lines, "")
    end

    Level2IO.write_csv(joinpath(table_root, "window_state_summary.csv"),
                       ["window", "distance_metric", "distance_component_scales",
                        "min_cluster_fraction", "min_cluster_size_floor", "effective_min_cluster_size",
                        "chosen_k", "best_silhouette", "is_effectively_unimodal",
                        "low_n", "high_n", "central_n", "global_medoid_index"],
                       window_rows)
    Level2IO.write_csv(joinpath(table_root, "cluster_summary.csv"),
                       ["window", "cluster_id", "cluster_size", "cluster_median_state_score",
                        "cluster_medoid_index", "cluster_order_rank"],
                       cluster_rows)
    Level2IO.write_csv(joinpath(table_root, "state_library_summary.csv"),
                       ["window", "state_label", "library_size", "medoid_index",
                        "medoid_log_kxx", "medoid_log_kyy", "medoid_log_kzz",
                        "state_score_min", "state_score_median", "state_score_max",
                        "mean_log_kxx", "mean_log_kyy", "mean_log_kzz"],
                       state_rows)
    Level2IO.write_csv(joinpath(table_root, "neighborhood_summary.csv"),
                       ["window", "state_label", "neighborhood", "neighbor_count",
                        "state_library_size", "medoid_index", "medoid_log_kxx",
                        "medoid_log_kyy", "medoid_log_kzz", "neighbor_fraction",
                        "distance_min", "distance_median", "distance_max",
                        "state_score_min", "state_score_median", "state_score_max",
                        "mean_log_kxx", "mean_log_kyy", "mean_log_kzz"],
                       neighborhood_summary_rows)
    Level2IO.write_text_lines(joinpath(report_root, "level2_summary_report.txt"), report_lines)

    println("Saved Level 2 summaries to $output_root")
end

function write_neighborhood_tables!(
    summary_rows::Vector{Vector{String}},
    output_path::AbstractString,
    state,
    window::AbstractString,
)
    log_perms = matrix_float(state["log_perms"])
    distance_features = state_distance_features(state)
    state_scores = vector_float(state["state_score"])
    detail_rows = Vector{Vector{String}}()

    for label in ("low", "central", "high")
        state_indices = vector_int(state["$(label)_indices"])
        medoid_index = int_scalar(state["$(label)_medoid_index"])
        medoid_features = vec(distance_features[medoid_index, :])
        medoid_log_perm = vec(log_perms[medoid_index, :])

        for neighborhood in ("small", "large")
            neighbor_indices = vector_int(state["$(label)_$(neighborhood)_neighbors"])
            distances = [norm(vec(distance_features[idx, :]) - medoid_features) for idx in neighbor_indices]
            order = sortperm(distances)
            sorted_indices = neighbor_indices[order]
            sorted_distances = distances[order]
            neighbor_scores = state_scores[sorted_indices]
            neighbor_logs = log_perms[sorted_indices, :]

            push!(summary_rows, [
                window,
                label,
                neighborhood,
                string(length(sorted_indices)),
                string(length(state_indices)),
                string(medoid_index),
                fmt(medoid_log_perm[1]),
                fmt(medoid_log_perm[2]),
                fmt(medoid_log_perm[3]),
                fmt(length(sorted_indices) / length(state_indices)),
                fmt(minimum(sorted_distances)),
                fmt(median(sorted_distances)),
                fmt(maximum(sorted_distances)),
                fmt(minimum(neighbor_scores)),
                fmt(median(neighbor_scores)),
                fmt(maximum(neighbor_scores)),
                fmt(mean(neighbor_logs[:, 1])),
                fmt(mean(neighbor_logs[:, 2])),
                fmt(mean(neighbor_logs[:, 3])),
            ])

            for rank in eachindex(sorted_indices)
                idx = sorted_indices[rank]
                push!(detail_rows, [
                    window,
                    label,
                    neighborhood,
                    string(rank),
                    string(idx),
                    fmt(sorted_distances[rank]),
                    fmt(state_scores[idx]),
                    fmt(log_perms[idx, 1]),
                    fmt(log_perms[idx, 2]),
                    fmt(log_perms[idx, 3]),
                    fmt(distance_features[idx, 1]),
                    fmt(distance_features[idx, 2]),
                    fmt(distance_features[idx, 3]),
                ])
            end
        end
    end

    Level2IO.write_csv(output_path,
                       ["window", "state_label", "neighborhood", "neighbor_rank",
                        "neighbor_index", "distance_to_state_medoid",
                        "state_score", "log_kxx", "log_kyy", "log_kzz",
                        "distance_feature_kxx", "distance_feature_kyy",
                        "distance_feature_kzz"],
                       detail_rows)
end

function state_distance_features(state)
    metric = String(get(state, "distance_metric", "local_normal"))
    log_perms = matrix_float(state["log_perms"])
    local_normal_scores = matrix_float(state["local_normal_scores"])
    scales = haskey(state, "distance_component_scales") ?
        vector_float(state["distance_component_scales"]) :
        ones(Float64, size(log_perms, 2))
    weights = haskey(state, "distance_weights") ?
        vector_float(state["distance_weights"]) :
        ones(Float64, size(log_perms, 2))

    values = metric == "local_normal" ? local_normal_scores : log_perms
    features = similar(values)
    for j in axes(values, 2)
        scale = scales[j] > 0 ? scales[j] : 1.0
        features[:, j] .= values[:, j] .* sqrt(weights[j]) ./ scale
    end
    return features
end

float_scalar(value) = value isa AbstractArray ? Float64(first(vec(value))) : Float64(value)
int_scalar(value) = Int(round(float_scalar(value)))
vector_int(values) = values isa AbstractArray ? vec(Int.(values)) : [Int(values)]
vector_float(values) = values isa AbstractArray ? vec(Float64.(values)) : [Float64(values)]
matrix_float(values) = Matrix{Float64}(values)
fmt(value) = string(round(Float64(value), digits = 6))

main(ARGS)
