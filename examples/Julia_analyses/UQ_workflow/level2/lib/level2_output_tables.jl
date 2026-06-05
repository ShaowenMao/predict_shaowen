#!/usr/bin/env julia

const LEVEL2_ROOT = normpath(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Statistics

include(joinpath(LEVEL2_ROOT, "lib", "level2_io.jl"))

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
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/lib/level2_output_tables.jl [options]")
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
    pool_table_root = joinpath(table_root, "perturbation_pool_tables")
    report_root = joinpath(output_root, "reports")
    mkpath(table_root)
    mkpath(point_table_root)
    mkpath(pool_table_root)
    mkpath(report_root)

    window_rows = Vector{Vector{String}}()
    cluster_rows = Vector{Vector{String}}()
    state_rows = Vector{Vector{String}}()
    state_cluster_rows = Vector{Vector{String}}()
    state_cluster_readable_rows = Vector{Vector{String}}()
    perturbation_pool_summary_rows = Vector{Vector{String}}()
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
        write_perturbation_pool_tables!(
            perturbation_pool_summary_rows,
            joinpath(pool_table_root, "$(window)_perturbation_pool_members.csv"),
            state,
            window,
        )

        chosen_k = int_scalar(state["chosen_k"])
        silhouette = float_scalar(state["best_silhouette"])
        unimodal = int_scalar(state["is_effectively_unimodal"])
        low_n = length(vector_int(state["low_indices"]))
        high_n = length(vector_int(state["high_indices"]))
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
            string(int_scalar(state["global_medoid_index"])),
        ])

        cluster_sizes = vector_int(state["cluster_sizes"])
        cluster_medians = haskey(state, "cluster_joint_rank_medians") ?
            vector_float(state["cluster_joint_rank_medians"]) :
            vector_float(state["cluster_score_medians"])
        cluster_medoids = vector_int(state["cluster_medoids"])
        cluster_order = vector_int(state["cluster_order"])
        cluster_rank_map = Dict(cluster_id => rank for (rank, cluster_id) in enumerate(cluster_order))
        for cluster_id in 1:length(cluster_sizes)
            push!(cluster_rows, [
                window,
                string(cluster_id),
                string(cluster_sizes[cluster_id]),
                string(round(cluster_medians[cluster_id], digits = 6)),
                string(cluster_medoids[cluster_id]),
                string(cluster_rank_map[cluster_id]),
            ])
        end

        for label in ("low", "high")
            state_indices = vector_int(state["$(label)_indices"])
            scores = state_joint_rank_scores(state)[state_indices]
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

            append_state_cluster_composition!(
                state_cluster_rows,
                state_cluster_readable_rows,
                window,
                label,
                state_indices,
                state,
            )
        end

        push!(report_lines, "window = $window")
        push!(report_lines, "  distance_metric = $distance_metric")
        push!(report_lines, "  distance_component_scales = $distance_scales")
        push!(report_lines, "  min_cluster_fraction = $min_cluster_fraction")
        push!(report_lines, "  effective_min_cluster_size = $effective_min_cluster_size")
        push!(report_lines, "  chosen_k = $chosen_k")
        push!(report_lines, "  silhouette = $(round(silhouette, digits = 6))")
        push!(report_lines, "  unimodal = $(Bool(unimodal))")
        push!(report_lines, "  state_sizes = low:$low_n high:$high_n")
        push!(report_lines, "  point_table = $(joinpath(point_table_root, "$(window)_level2_points.csv"))")
        push!(report_lines, "  perturbation_pool_table = $(joinpath(pool_table_root, "$(window)_perturbation_pool_members.csv"))")
        push!(report_lines, "")
    end

    Level2IO.write_csv(joinpath(table_root, "window_state_summary.csv"),
                       ["window", "distance_metric", "distance_component_scales",
                        "min_cluster_fraction", "min_cluster_size_floor", "effective_min_cluster_size",
                        "chosen_k", "best_silhouette", "is_effectively_unimodal",
                        "low_n", "high_n", "global_medoid_index"],
                       window_rows)
    Level2IO.write_csv(joinpath(table_root, "cluster_summary.csv"),
                       ["window", "original_cluster_id", "cluster_size", "cluster_median_joint_rank_score",
                        "cluster_medoid_index", "ordered_cluster_number"],
                       cluster_rows)
    Level2IO.write_csv(joinpath(table_root, "state_library_summary.csv"),
                       ["window", "state_label", "library_size", "medoid_index",
                        "medoid_log_kxx", "medoid_log_kyy", "medoid_log_kzz",
                        "joint_rank_score_min", "joint_rank_score_median", "joint_rank_score_max",
                        "mean_log_kxx", "mean_log_kyy", "mean_log_kzz"],
                       state_rows)
    Level2IO.write_csv(joinpath(table_root, "state_cluster_composition.csv"),
                       ["window", "window_label", "state_label", "ordered_cluster_number",
                        "original_cluster_id", "samples_from_cluster", "state_library_size",
                        "percent_of_state", "joint_cluster_size",
                        "percent_of_cluster_used", "cluster_median_joint_rank_score"],
                       state_cluster_rows)
    Level2IO.write_csv(joinpath(table_root, "state_cluster_composition_readable.csv"),
                       ["window", "window_label", "state_label", "state_library_size",
                        "number_of_joint_clusters_in_state", "composition_by_ordered_cluster",
                        "composition_by_original_cluster_id", "dominant_ordered_cluster",
                        "dominant_original_cluster_id", "dominant_percent_of_state",
                        "reading_note"],
                       state_cluster_readable_rows)
    Level2IO.write_csv(joinpath(table_root, "perturbation_pool_summary.csv"),
                       ["window", "state_label", "perturbation_pool", "pool_count",
                        "candidate_count", "state_library_size", "candidate_scope",
                        "medoid_index", "medoid_cluster_id", "medoid_ordered_cluster",
                        "medoid_log_kxx", "medoid_log_kyy", "medoid_log_kzz",
                        "pool_fraction_of_candidate", "pool_fraction_of_state",
                        "distance_min", "distance_median", "distance_max",
                        "joint_rank_score_min", "joint_rank_score_median", "joint_rank_score_max",
                        "mean_log_kxx", "mean_log_kyy", "mean_log_kzz"],
                       perturbation_pool_summary_rows)
    Level2IO.write_text_lines(joinpath(report_root, "level2_summary_report.txt"), report_lines)

    println("Saved Level 2 summaries to $output_root")
end

function append_state_cluster_composition!(
    detail_rows::Vector{Vector{String}},
    readable_rows::Vector{Vector{String}},
    window::AbstractString,
    label::AbstractString,
    state_indices::Vector{Int},
    state,
)
    cluster_assignments = vector_int(state["cluster_assignments"])
    cluster_sizes = vector_int(state["cluster_sizes"])
    cluster_medians = haskey(state, "cluster_joint_rank_medians") ?
        vector_float(state["cluster_joint_rank_medians"]) :
        vector_float(state["cluster_score_medians"])
    cluster_order = vector_int(state["cluster_order"])
    cluster_rank_map = Dict(cluster_id => rank for (rank, cluster_id) in enumerate(cluster_order))
    state_size = length(state_indices)
    entries = NamedTuple[]

    for cluster_id in cluster_order
        count_in_state = count(idx -> cluster_assignments[idx] == cluster_id, state_indices)
        count_in_state == 0 && continue
        order_rank = cluster_rank_map[cluster_id]
        cluster_size = cluster_sizes[cluster_id]
        median_score = cluster_medians[cluster_id]

        push!(entries, (
            cluster_id = cluster_id,
            order_rank = order_rank,
            count_in_state = count_in_state,
            cluster_size = cluster_size,
            median_score = median_score,
        ))

        push!(detail_rows, [
            window,
            window_label(window),
            label,
            string(order_rank),
            string(cluster_id),
            string(count_in_state),
            string(state_size),
            fmt(100 * count_in_state / state_size),
            string(cluster_size),
            fmt(100 * count_in_state / cluster_size),
            fmt(median_score),
        ])
    end

    isempty(entries) && return

    counts = [entry.count_in_state for entry in entries]
    dominant = entries[argmax(counts)]
    composition_by_order = join(
        [
            "Ordered cluster $(entry.order_rank): $(entry.count_in_state) samples ($(fmt(100 * entry.count_in_state / state_size))%)"
            for entry in entries
        ],
        "; ",
    )
    composition_by_cluster = join(
        [
            "Cluster $(entry.cluster_id): $(entry.count_in_state) samples"
            for entry in entries
        ],
        "; ",
    )

    push!(readable_rows, [
        window,
        window_label(window),
        label,
        string(state_size),
        string(length(entries)),
        composition_by_order,
        composition_by_cluster,
        string(dominant.order_rank),
        string(dominant.cluster_id),
        fmt(100 * dominant.count_in_state / state_size),
        state_cluster_reading_note(label, length(entries)),
    ])
end

window_label(window::AbstractString) = startswith(window, "famp") ? "W$(window[5:end])" : String(window)

function state_cluster_reading_note(label::AbstractString, n_clusters::Int)
    if n_clusters == 1
        return "This state comes from one joint cluster."
    elseif label == "low"
        return "Low state combines the lowest joint-rank cluster with the needed part of the next cluster to reach the target state size."
    elseif label == "high"
        return "High state combines the highest joint-rank cluster with the needed part of the previous cluster to reach the target state size."
    end
    return "State composition is reported by ordered joint cluster."
end

function write_perturbation_pool_tables!(
    summary_rows::Vector{Vector{String}},
    output_path::AbstractString,
    state,
    window::AbstractString,
)
    log_perms = matrix_float(state["log_perms"])
    distance_features = state_distance_features(state)
    joint_rank_scores = state_joint_rank_scores(state)
    cluster_assignments = vector_int(state["cluster_assignments"])
    cluster_order = vector_int(state["cluster_order"])
    cluster_rank_map = Dict(cluster_id => rank for (rank, cluster_id) in enumerate(cluster_order))
    detail_rows = Vector{Vector{String}}()

    for label in ("low", "high")
        state_indices = vector_int(state["$(label)_indices"])
        medoid_index = int_scalar(state["$(label)_medoid_index"])
        medoid_cluster_id = cluster_assignments[medoid_index]
        medoid_cluster_rank = cluster_rank_map[medoid_cluster_id]
        medoid_features = vec(distance_features[medoid_index, :])
        medoid_log_perm = vec(log_perms[medoid_index, :])

        for pool in ("local", "state_wide")
            pool_indices = vector_int(state["$(label)_$(pool)_pool"])
            candidate_indices = pool == "local" && haskey(state, "$(label)_local_pool_candidates") ?
                vector_int(state["$(label)_local_pool_candidates"]) :
                state_indices
            candidate_scope = pool == "local" ?
                "state_library_intersection_medoid_cluster" :
                "full_state_library"
            distances = [norm(vec(distance_features[idx, :]) - medoid_features) for idx in pool_indices]
            order = sortperm(distances)
            sorted_indices = pool_indices[order]
            sorted_distances = distances[order]
            pool_scores = joint_rank_scores[sorted_indices]
            pool_logs = log_perms[sorted_indices, :]

            push!(summary_rows, [
                window,
                label,
                pool,
                string(length(sorted_indices)),
                string(length(candidate_indices)),
                string(length(state_indices)),
                candidate_scope,
                string(medoid_index),
                string(medoid_cluster_id),
                string(medoid_cluster_rank),
                fmt(medoid_log_perm[1]),
                fmt(medoid_log_perm[2]),
                fmt(medoid_log_perm[3]),
                fmt(length(sorted_indices) / length(candidate_indices)),
                fmt(length(sorted_indices) / length(state_indices)),
                fmt(minimum(sorted_distances)),
                fmt(median(sorted_distances)),
                fmt(maximum(sorted_distances)),
                fmt(minimum(pool_scores)),
                fmt(median(pool_scores)),
                fmt(maximum(pool_scores)),
                fmt(mean(pool_logs[:, 1])),
                fmt(mean(pool_logs[:, 2])),
                fmt(mean(pool_logs[:, 3])),
            ])

            for rank in eachindex(sorted_indices)
                idx = sorted_indices[rank]
                push!(detail_rows, [
                    window,
                    label,
                    pool,
                    string(rank),
                    candidate_scope,
                    string(idx),
                    fmt(sorted_distances[rank]),
                    fmt(joint_rank_scores[idx]),
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
                       ["window", "state_label", "perturbation_pool", "pool_rank",
                        "candidate_scope", "sample_index", "distance_to_state_medoid",
                        "joint_rank_score", "log_kxx", "log_kyy", "log_kzz",
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
state_joint_rank_scores(state) = haskey(state, "joint_rank_score") ?
    vector_float(state["joint_rank_score"]) :
    vector_float(state["state_score"])
fmt(value) = string(round(Float64(value), digits = 6))


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
