"""
    Level2Core

Core algorithms for Level 2 within-window UQ analysis.

This module turns one window's joint PREDICT permeability library into a
Level 2 state object. It provides local rank transforms, physical-space
distance construction, k-medoids joint clustering, low/high state-library
selection, medoid selection, and local/state-wide perturbation pools.
"""
module Level2Core

using Statistics
using Random
using LinearAlgebra
using Dates

export FIXED_WINDOWS,
       build_window_state,
       compare_window_states

const FIXED_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]

"""
    build_window_state(log_perms, raw_perms, window, source_path, source_label, config)

Build one complete Level 2 window-state object.

The returned dictionary is the canonical MAT schema used by downstream table,
plotting, validation, and sampling scripts. Clustering is performed using the
configured distance metric, while low/high ordering is based on the local-rank
joint rank score.
"""
function build_window_state(log_perms::Matrix{Float64},
                            raw_perms::Matrix{Float64},
                            window::AbstractString,
                            source_path::AbstractString,
                            source_label::AbstractString,
                            config::Dict{String, Any})
    size(log_perms) == size(raw_perms) || error("log_perms and raw_perms must have the same size")
    size(log_perms, 2) == 3 || error("Expected exactly 3 permeability components for $window")
    size(log_perms, 1) >= 20 || error("Expected at least 20 samples for $window")

    local_ranks = compute_local_ranks(log_perms)
    joint_rank_score = compute_joint_rank_score(local_ranks, config["weights"])
    local_normal_scores = compute_local_normal_scores(local_ranks)
    distance_info = build_distance_matrix(log_perms, local_normal_scores, config)
    distance_matrix = distance_info["distance_matrix"]

    cluster_info = choose_clustering(distance_matrix, joint_rank_score, config, window)
    global_medoid_index = choose_medoid(collect(1:size(log_perms, 1)), distance_matrix)
    state_info = build_state_libraries(joint_rank_score,
                                       cluster_info["assignments"],
                                       config)

    low_stats = state_record("low", state_info["low_indices"], joint_rank_score,
                             distance_matrix, cluster_info["assignments"], config)
    high_stats = state_record("high", state_info["high_indices"], joint_rank_score,
                              distance_matrix, cluster_info["assignments"], config;
                              descending = true)
    return Dict{String, Any}(
        "schema_version" => "level2_window_state_v1",
        "created_at" => string(Dates.now()),
        "geology_id" => String(config["geology_id"]),
        "window" => String(window),
        "source_path" => String(source_path),
        "source_label" => String(source_label),
        "n_samples" => size(log_perms, 1),
        "raw_perms" => raw_perms,
        "log_perms" => log_perms,
        "local_ranks" => local_ranks,
        "local_normal_scores" => local_normal_scores,
        "joint_rank_score" => joint_rank_score,
        "joint_rank_score_order" => sortperm(joint_rank_score),
        "state_score" => joint_rank_score,
        "state_score_order" => sortperm(joint_rank_score),
        "weights" => Float64.(config["weights"]),
        "distance_metric" => distance_info["distance_metric"],
        "distance_component_scales" => distance_info["distance_component_scales"],
        "distance_weights" => distance_info["distance_weights"],
        "chosen_k" => cluster_info["chosen_k"],
        "best_silhouette" => cluster_info["best_silhouette"],
        "is_effectively_unimodal" => Int(cluster_info["is_effectively_unimodal"]),
        "silhouette_by_k" => cluster_info["silhouette_by_k"],
        "valid_k_mask" => cluster_info["valid_k_mask"],
        "cluster_assignments" => cluster_info["assignments"],
        "cluster_sizes" => cluster_info["cluster_sizes"],
        "cluster_medoids" => cluster_info["cluster_medoids"],
        "cluster_joint_rank_medians" => cluster_info["cluster_joint_rank_medians"],
        "cluster_score_medians" => cluster_info["cluster_joint_rank_medians"],
        "cluster_order" => cluster_info["cluster_order"],
        "global_medoid_index" => global_medoid_index,
        "low_indices" => low_stats["indices"],
        "high_indices" => high_stats["indices"],
        "low_ordered_indices" => low_stats["ordered_indices"],
        "high_ordered_indices" => high_stats["ordered_indices"],
        "low_medoid_index" => low_stats["medoid_index"],
        "high_medoid_index" => high_stats["medoid_index"],
        "low_medoid_cluster_id" => low_stats["medoid_cluster_id"],
        "high_medoid_cluster_id" => high_stats["medoid_cluster_id"],
        "low_local_pool_candidates" => low_stats["local_pool_candidates"],
        "high_local_pool_candidates" => high_stats["local_pool_candidates"],
        "low_local_pool" => low_stats["local_pool"],
        "low_state_wide_pool" => low_stats["state_wide_pool"],
        "high_local_pool" => high_stats["local_pool"],
        "high_state_wide_pool" => high_stats["state_wide_pool"],
        "low_mean_log_perm" => mean_vector(log_perms, low_stats["indices"]),
        "high_mean_log_perm" => mean_vector(log_perms, high_stats["indices"]),
        "state_fraction" => config["state_fraction"],
        "local_pool_fraction" => config["local_pool_fraction"],
        "local_pool_min_count" => config["local_pool_min_count"],
        "min_cluster_size" => config["min_cluster_size"],
        "min_cluster_fraction" => config["min_cluster_fraction"],
        "silhouette_threshold" => config["silhouette_threshold"],
        "max_k" => config["max_k"],
        "random_seed" => config["random_seed"],
    )
end

"""
    compare_window_states(reference_state, holdout_state)

Compute stability metrics between a reference Level 2 state and a holdout
state built from another repeat library.
"""
function compare_window_states(reference_state::Dict{String, Any},
                               holdout_state::Dict{String, Any})
    ref_log = matrix_float(reference_state["log_perms"])
    hold_log = matrix_float(holdout_state["log_perms"])

    metrics = Dict{String, Any}(
        "same_k" => int_scalar(reference_state["chosen_k"]) == int_scalar(holdout_state["chosen_k"]) ? "1" : "0",
        "same_unimodality" => int_scalar(reference_state["is_effectively_unimodal"]) ==
                              int_scalar(holdout_state["is_effectively_unimodal"]) ? "1" : "0",
        "reference_k" => string(int_scalar(reference_state["chosen_k"])),
        "holdout_k" => string(int_scalar(holdout_state["chosen_k"])),
        "reference_silhouette" => float_string(reference_state["best_silhouette"]),
        "holdout_silhouette" => float_string(holdout_state["best_silhouette"]),
        "abs_silhouette_delta" => float_string(abs(float_scalar(reference_state["best_silhouette"]) -
                                                      float_scalar(holdout_state["best_silhouette"]))),
    )

    for label in ("global", "low", "high")
        ref_index = medoid_index(reference_state, label)
        hold_index = medoid_index(holdout_state, label)
        ref_point = vec(ref_log[ref_index, :])
        hold_point = vec(hold_log[hold_index, :])
        metrics["$(label)_medoid_distance"] = float_string(norm(ref_point - hold_point))

        ref_mean = mean_vector(ref_log, state_indices(reference_state, label))
        hold_mean = mean_vector(hold_log, state_indices(holdout_state, label))
        metrics["$(label)_mean_distance"] = float_string(norm(ref_mean - hold_mean))
    end

    return metrics
end

"""
    compute_local_ranks(log_perms)

Compute component-wise empirical percentile ranks within one window.

Each column of `log_perms` is ranked independently, so the output describes
whether each realization is locally low or high for `kxx`, `kyy`, and `kzz`.
"""
function compute_local_ranks(log_perms::Matrix{Float64})
    n, p = size(log_perms)
    ranks = zeros(Float64, n, p)
    for j in 1:p
        ranks[:, j] = empirical_percentile_ranks(view(log_perms, :, j))
    end
    return ranks
end

"""
    empirical_percentile_ranks(x)

Return tie-aware empirical percentile ranks in `(0, 1)` for one vector.
"""
function empirical_percentile_ranks(x::AbstractVector{<:Real})
    n = length(x)
    order = sortperm(x)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[order[j + 1]] == x[order[i]]
            j += 1
        end
        avg_rank = (i + j) / 2
        p = avg_rank / (n + 1)
        for k in i:j
            ranks[order[k]] = p
        end
        i = j + 1
    end
    return ranks
end

"""
    compute_local_normal_scores(local_ranks)

Transform local percentile ranks to normal scores with the inverse standard
normal CDF.
"""
compute_local_normal_scores(local_ranks::Matrix{Float64}) = inv_norm_cdf.(clamp.(local_ranks, 1e-6, 1.0 - 1e-6))

"""
    inv_norm_cdf(p)

Approximate the inverse standard normal cumulative distribution function.
"""
function inv_norm_cdf(p::Float64)
    p <= 0.0 && return -Inf
    p >= 1.0 && return Inf

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow
        q = sqrt(-2.0 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    elseif p <= phigh
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    end
end

"""
    compute_joint_rank_score(local_ranks, weights)

Compute the weighted average local rank across `kxx`, `kyy`, and `kzz`.

This score is used only for low-to-high ordering and state-library
construction; it is not the clustering distance.
"""
function compute_joint_rank_score(local_ranks::Matrix{Float64}, weights::Vector{Float64})
    weight_sum = sum(weights)
    weight_sum > 0 || error("Joint-rank-score weights must sum to a positive value")
    return vec((local_ranks * weights) ./ weight_sum)
end

"""
    compute_state_score(local_ranks, weights)

Backward-compatible alias for `compute_joint_rank_score`.
"""
compute_state_score(local_ranks::Matrix{Float64}, weights::Vector{Float64}) =
    compute_joint_rank_score(local_ranks, weights)

"""
    pairwise_euclidean(z)

Compute a full pairwise Euclidean distance matrix for feature rows.
"""
function pairwise_euclidean(z::Matrix{Float64})
    n = size(z, 1)
    d = zeros(Float64, n, n)
    for i in 1:n-1
        zi = @view z[i, :]
        for j in i+1:n
            value = sqrt(sum(abs2, zi .- @view(z[j, :])))
            d[i, j] = value
            d[j, i] = value
        end
    end
    return d
end

"""
    build_distance_matrix(log_perms, local_normal_scores, config)

Build the pairwise distance matrix and distance metadata used for clustering,
medoids, and perturbation pools.

The default `log_unit` metric uses physical `log10(k)` values directly. The
`local_normal` metric is retained only as a sensitivity option.
"""
function build_distance_matrix(log_perms::Matrix{Float64},
                               local_normal_scores::Matrix{Float64},
                               config::Dict{String, Any})
    metric = String(get(config, "distance_metric", "log_unit"))
    weights = Float64.(get(config, "distance_weights", ones(size(log_perms, 2))))
    length(weights) == size(log_perms, 2) || error("distance_weights must match permeability component count")
    all(weights .> 0) || error("distance_weights must be positive")

    if metric == "local_normal"
        scales = ones(Float64, size(log_perms, 2))
        features = weighted_features(local_normal_scores, scales, weights)
    elseif metric == "log_unit"
        scales = ones(Float64, size(log_perms, 2))
        features = weighted_features(log_perms, scales, weights)
    else
        error("Unsupported distance_metric '$metric'. Use log_unit or local_normal.")
    end

    return Dict{String, Any}(
        "distance_metric" => metric,
        "distance_component_scales" => scales,
        "distance_weights" => weights,
        "distance_matrix" => pairwise_euclidean(features),
    )
end

"""
    weighted_features(values, scales, weights)

Scale and weight component columns before pairwise distance calculation.
"""
function weighted_features(values::Matrix{Float64}, scales::Vector{Float64}, weights::Vector{Float64})
    features = similar(values)
    for j in axes(values, 2)
        scale = scales[j] > 0 ? scales[j] : 1.0
        features[:, j] .= values[:, j] .* sqrt(weights[j]) ./ scale
    end
    return features
end

"""
    choose_clustering(distance_matrix, joint_rank_score, config, window)

Select the joint permeability clustering for one window.

The function tries `K = 2:max_k`, rejects clusterings with too-small clusters,
chooses the valid `K` with the highest average silhouette, and falls back to a
single-cluster state if no valid multi-cluster solution is sufficiently
separated.
"""
function choose_clustering(distance_matrix::Matrix{Float64},
                           joint_rank_score::Vector{Float64},
                           config::Dict{String, Any},
                           window::AbstractString)
    n = size(distance_matrix, 1)
    max_k = min(config["max_k"], n)
    min_size = max(config["min_cluster_size"], ceil(Int, config["min_cluster_fraction"] * n))

    silhouette_by_k = fill(NaN, max_k)
    valid_k_mask = zeros(Int, max_k)

    best = nothing
    for k in 2:max_k
        fit = fit_kmedoids(distance_matrix,
                           k,
                           config["random_seed"] + window_seed_offset(window),
                           config["num_restarts"],
                           config["max_kmedoids_iter"])
        fit === nothing && continue
        cluster_sizes = counts_from_assignments(fit["assignments"], k)
        all(cluster_sizes .>= min_size) || continue
        silhouette = average_silhouette(distance_matrix, fit["assignments"], k)
        silhouette_by_k[k] = silhouette
        valid_k_mask[k] = 1
        if best === nothing || silhouette > best["best_silhouette"]
            medians = cluster_joint_rank_medians(joint_rank_score, fit["assignments"], k)
            best = Dict{String, Any}(
                "chosen_k" => k,
                "best_silhouette" => silhouette,
                "assignments" => fit["assignments"],
                "cluster_sizes" => cluster_sizes,
                "cluster_medoids" => fit["medoids"],
                "cluster_joint_rank_medians" => medians,
                "cluster_score_medians" => medians,
                "cluster_order" => sortperm(medians),
                "is_effectively_unimodal" => silhouette < config["silhouette_threshold"],
                "silhouette_by_k" => silhouette_by_k,
                "valid_k_mask" => valid_k_mask,
            )
        end
    end

    if best === nothing || best["is_effectively_unimodal"]
        assignments = ones(Int, n)
        medoid = choose_medoid(collect(1:n), distance_matrix)
        return Dict{String, Any}(
            "chosen_k" => 1,
            "best_silhouette" => best === nothing ? 0.0 : best["best_silhouette"],
            "assignments" => assignments,
            "cluster_sizes" => [n],
            "cluster_medoids" => [medoid],
            "cluster_joint_rank_medians" => [median(joint_rank_score)],
            "cluster_score_medians" => [median(joint_rank_score)],
            "cluster_order" => [1],
            "is_effectively_unimodal" => true,
            "silhouette_by_k" => silhouette_by_k,
            "valid_k_mask" => valid_k_mask,
        )
    end

    return best
end

"""
    window_seed_offset(window)

Create a deterministic per-window seed offset for k-medoids restarts.
"""
window_seed_offset(window::AbstractString) = sum(Int(c) for c in codeunits(window))

"""
    fit_kmedoids(distance_matrix, k, seed_base, num_restarts, max_iter)

Fit k-medoids to a precomputed distance matrix using deterministic restarts.
"""
function fit_kmedoids(distance_matrix::Matrix{Float64},
                      k::Int,
                      seed_base::Integer,
                      num_restarts::Int,
                      max_iter::Int)
    n = size(distance_matrix, 1)
    best = nothing
    best_cost = Inf

    for restart in 1:num_restarts
        rng = MersenneTwister(seed_base + restart)
        medoids = init_medoids(distance_matrix, k, rng)
        assignments = zeros(Int, n)

        for _ in 1:max_iter
            new_assignments = assign_to_medoids(distance_matrix, medoids)
            new_medoids = update_medoids(distance_matrix, new_assignments, k)
            isempty(new_medoids) && break
            if new_assignments == assignments && new_medoids == medoids
                assignments = new_assignments
                medoids = new_medoids
                break
            end
            assignments = new_assignments
            medoids = new_medoids
        end

        assignments = assign_to_medoids(distance_matrix, medoids)
        cluster_sizes = counts_from_assignments(assignments, k)
        any(cluster_sizes .== 0) && continue

        cost = total_kmedoids_cost(distance_matrix, assignments, medoids)
        if cost < best_cost
            best_cost = cost
            best = Dict{String, Any}(
                "medoids" => medoids,
                "assignments" => assignments,
                "cost" => cost,
            )
        end
    end
    return best
end

"""
    init_medoids(distance_matrix, k, rng)

Initialize k-medoids using a farthest-first strategy after a random first
medoid.
"""
function init_medoids(distance_matrix::Matrix{Float64}, k::Int, rng::AbstractRNG)
    n = size(distance_matrix, 1)
    medoids = Int[rand(rng, 1:n)]
    while length(medoids) < k
        best_idx = 0
        best_distance = -Inf
        for idx in 1:n
            idx in medoids && continue
            nearest = minimum(distance_matrix[idx, medoids])
            if nearest > best_distance
                best_distance = nearest
                best_idx = idx
            end
        end
        best_idx > 0 || error("Could not initialize $k medoids")
        push!(medoids, best_idx)
    end
    return medoids
end

"""
    assign_to_medoids(distance_matrix, medoids)

Assign every sample to the nearest current medoid.
"""
function assign_to_medoids(distance_matrix::Matrix{Float64}, medoids::Vector{Int})
    n = size(distance_matrix, 1)
    assignments = zeros(Int, n)
    for i in 1:n
        distances = distance_matrix[i, medoids]
        assignments[i] = argmin(distances)
    end
    return assignments
end

"""
    update_medoids(distance_matrix, assignments, k)

Update each cluster's medoid by minimizing within-cluster total distance.
"""
function update_medoids(distance_matrix::Matrix{Float64}, assignments::Vector{Int}, k::Int)
    medoids = Int[]
    for cluster_id in 1:k
        members = findall(assignments .== cluster_id)
        isempty(members) && return Int[]
        push!(medoids, choose_medoid(members, distance_matrix))
    end
    return medoids
end

"""
    choose_medoid(indices, distance_matrix)

Choose the index with minimum total distance to all other candidate indices.
"""
function choose_medoid(indices::Vector{Int}, distance_matrix::Matrix{Float64})
    costs = [sum(distance_matrix[idx, indices]) for idx in indices]
    return indices[argmin(costs)]
end

"""
    total_kmedoids_cost(distance_matrix, assignments, medoids)

Return the total within-cluster distance to assigned medoids.
"""
total_kmedoids_cost(distance_matrix::Matrix{Float64}, assignments::Vector{Int}, medoids::Vector{Int}) =
    sum(distance_matrix[i, medoids[assignments[i]]] for i in eachindex(assignments))

"""
    average_silhouette(distance_matrix, assignments, k)

Compute the average silhouette score for a clustering on a distance matrix.
"""
function average_silhouette(distance_matrix::Matrix{Float64}, assignments::Vector{Int}, k::Int)
    k == 1 && return 0.0
    clusters = [findall(assignments .== cluster_id) for cluster_id in 1:k]
    sil = zeros(Float64, length(assignments))
    for i in eachindex(assignments)
        own_cluster = assignments[i]
        own_members = clusters[own_cluster]
        if length(own_members) == 1
            sil[i] = 0.0
            continue
        end
        a = mean(distance_matrix[i, [j for j in own_members if j != i]])
        b = Inf
        for cluster_id in 1:k
            cluster_id == own_cluster && continue
            b = min(b, mean(distance_matrix[i, clusters[cluster_id]]))
        end
        denom = max(a, b)
        sil[i] = denom == 0.0 ? 0.0 : (b - a) / denom
    end
    return mean(sil)
end

"""
    counts_from_assignments(assignments, k)

Count the number of samples assigned to each cluster id.
"""
counts_from_assignments(assignments::Vector{Int}, k::Int) = [count(==(cluster_id), assignments) for cluster_id in 1:k]

"""
    cluster_joint_rank_medians(joint_rank_score, assignments, k)

Compute each cluster's median joint rank score for low-to-high ordering.
"""
function cluster_joint_rank_medians(joint_rank_score::Vector{Float64}, assignments::Vector{Int}, k::Int)
    medians = zeros(Float64, k)
    for cluster_id in 1:k
        members = findall(assignments .== cluster_id)
        medians[cluster_id] = median(joint_rank_score[members])
    end
    return medians
end

"""
    cluster_score_medians(state_score, assignments, k)

Backward-compatible alias for `cluster_joint_rank_medians`.
"""
cluster_score_medians(state_score::Vector{Float64}, assignments::Vector{Int}, k::Int) =
    cluster_joint_rank_medians(state_score, assignments, k)

"""
    build_state_libraries(joint_rank_score, assignments, config)

Construct low and high state libraries with cluster-aware target-mass
selection.

For multi-cluster windows, extreme ordered clusters are included first and only
the needed part of a boundary cluster is added to reach the target state
fraction.
"""
function build_state_libraries(joint_rank_score::Vector{Float64},
                               assignments::Vector{Int},
                               config::Dict{String, Any})
    n = length(joint_rank_score)
    n_target = max(1, ceil(Int, config["state_fraction"] * n))

    chosen_k = maximum(assignments)
    if chosen_k == 1
        score_order = sortperm(joint_rank_score)
        low_indices = score_order[1:n_target]
        low_set = Set(low_indices)
        high_candidates = [idx for idx in reverse(score_order) if !(idx in low_set)]
        high_indices = high_candidates[1:min(n_target, length(high_candidates))]
    else
        medians = cluster_joint_rank_medians(joint_rank_score, assignments, chosen_k)
        cluster_order = sortperm(medians)
        low_indices = select_cluster_aware_state(cluster_order, assignments, joint_rank_score, n_target)
        high_indices = select_cluster_aware_state(reverse(cluster_order), assignments, joint_rank_score, n_target;
                                                 descending = true,
                                                 excluded = Set(low_indices))
    end

    return Dict{String, Any}(
        "low_indices" => unique(low_indices),
        "high_indices" => unique(high_indices),
    )
end

"""
    select_cluster_aware_state(cluster_order, assignments, joint_rank_score, n_target; descending=false, excluded=Set())

Select samples from ordered clusters until the requested target count is met.
"""
function select_cluster_aware_state(cluster_order,
                                   assignments::Vector{Int},
                                   joint_rank_score::Vector{Float64},
                                   n_target::Int;
                                   descending::Bool = false,
                                   excluded::Set{Int} = Set{Int}())
    selected = Int[]
    for cluster_id in cluster_order
        remaining = n_target - length(selected)
        remaining <= 0 && break

        members = [idx for idx in eachindex(assignments)
                   if assignments[idx] == cluster_id && !(idx in excluded)]
        isempty(members) && continue

        ordered_members = members[sortperm(joint_rank_score[members], rev = descending)]
        append!(selected, ordered_members[1:min(remaining, length(ordered_members))])
    end
    return sort(unique(selected))
end

"""
    state_record(label, indices, joint_rank_score, distance_matrix, cluster_assignments, config; descending=false)

Summarize one low/high state library with ordered indices, medoid, and
perturbation pools.
"""
function state_record(label::AbstractString,
                      indices::Vector{Int},
                      joint_rank_score::Vector{Float64},
                      distance_matrix::Matrix{Float64},
                      cluster_assignments::Vector{Int},
                      config::Dict{String, Any};
                      descending::Bool = false)
    isempty(indices) && error("State library $label is empty")
    ordered = descending ? indices[sortperm(joint_rank_score[indices], rev = true)] :
                           indices[sortperm(joint_rank_score[indices])]
    medoid_index = choose_medoid(indices, distance_matrix)
    medoid_cluster_id = cluster_assignments[medoid_index]
    local_pool_candidates = [idx for idx in indices if cluster_assignments[idx] == medoid_cluster_id]
    local_pool = local_pool_indices(medoid_index, local_pool_candidates, distance_matrix,
                                    config["local_pool_fraction"], config["local_pool_min_count"])
    state_wide_pool = sort(unique(indices))
    return Dict{String, Any}(
        "indices" => indices,
        "ordered_indices" => ordered,
        "medoid_index" => medoid_index,
        "medoid_cluster_id" => medoid_cluster_id,
        "local_pool_candidates" => local_pool_candidates,
        "local_pool" => local_pool,
        "state_wide_pool" => state_wide_pool,
    )
end

"""
    local_pool_indices(target_index, candidates, distance_matrix, fraction, min_count)

Select the nearest samples to a state medoid within the cluster-preserving
candidate set.
"""
function local_pool_indices(target_index::Int,
                            candidates::Vector{Int},
                            distance_matrix::Matrix{Float64},
                            fraction::Float64,
                            min_count::Int)
    isempty(candidates) && return Int[]
    n_fraction = ceil(Int, fraction * length(candidates))
    n_keep = min(length(candidates), max(min_count, n_fraction))
    ordered = sort(candidates; by = idx -> distance_matrix[target_index, idx])
    return ordered[1:n_keep]
end

"""
    mean_vector(values, indices)

Return the component-wise mean vector over selected rows.
"""
function mean_vector(values::Matrix{Float64}, indices::Vector{Int})
    return vec(mean(values[indices, :]; dims = 1))
end

medoid_index(state::Dict{String, Any}, label::AbstractString) =
    label == "global" ? int_scalar(state["global_medoid_index"]) :
    int_scalar(state["$(label)_medoid_index"])

state_indices(state::Dict{String, Any}, label::AbstractString) =
    label == "global" ? collect(1:size(matrix_float(state["log_perms"]), 1)) :
    vector_int(state["$(label)_indices"])

float_string(value) = string(round(float_scalar(value), digits = 6))
float_scalar(value) = value isa AbstractArray ? Float64(first(vec(value))) : Float64(value)
int_scalar(value) = Int(round(float_scalar(value)))
vector_int(values) = values isa AbstractArray ? vec(Int.(values)) : [Int(values)]
matrix_float(values) = Matrix{Float64}(values)

end
