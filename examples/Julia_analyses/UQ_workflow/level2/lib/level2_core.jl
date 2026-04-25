module Level2Core

using Statistics
using Random
using LinearAlgebra
using Dates

export FIXED_WINDOWS,
       build_window_state,
       compare_window_states

const FIXED_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]

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
    state_score = compute_state_score(local_ranks, config["weights"])
    local_normal_scores = compute_local_normal_scores(local_ranks)
    distance_matrix = pairwise_euclidean(local_normal_scores)

    cluster_info = choose_clustering(distance_matrix, state_score, config, window)
    state_info = build_state_libraries(state_score, cluster_info["assignments"], distance_matrix, config)

    low_stats = state_record("low", state_info["low_indices"], state_score, distance_matrix, config)
    high_stats = state_record("high", state_info["high_indices"], state_score, distance_matrix, config; descending = true)
    central_stats = state_record("central", state_info["central_indices"], state_score, distance_matrix, config)

    global_medoid_index = choose_medoid(collect(1:size(log_perms, 1)), distance_matrix)

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
        "state_score" => state_score,
        "state_score_order" => sortperm(state_score),
        "weights" => Float64.(config["weights"]),
        "chosen_k" => cluster_info["chosen_k"],
        "best_silhouette" => cluster_info["best_silhouette"],
        "is_effectively_unimodal" => Int(cluster_info["is_effectively_unimodal"]),
        "silhouette_by_k" => cluster_info["silhouette_by_k"],
        "valid_k_mask" => cluster_info["valid_k_mask"],
        "cluster_assignments" => cluster_info["assignments"],
        "cluster_sizes" => cluster_info["cluster_sizes"],
        "cluster_medoids" => cluster_info["cluster_medoids"],
        "cluster_score_medians" => cluster_info["cluster_score_medians"],
        "cluster_order" => cluster_info["cluster_order"],
        "global_medoid_index" => global_medoid_index,
        "low_indices" => low_stats["indices"],
        "high_indices" => high_stats["indices"],
        "central_indices" => central_stats["indices"],
        "low_ordered_indices" => low_stats["ordered_indices"],
        "high_ordered_indices" => high_stats["ordered_indices"],
        "central_ordered_indices" => central_stats["ordered_indices"],
        "low_medoid_index" => low_stats["medoid_index"],
        "high_medoid_index" => high_stats["medoid_index"],
        "central_medoid_index" => central_stats["medoid_index"],
        "low_small_neighbors" => low_stats["small_neighbors"],
        "low_large_neighbors" => low_stats["large_neighbors"],
        "high_small_neighbors" => high_stats["small_neighbors"],
        "high_large_neighbors" => high_stats["large_neighbors"],
        "central_small_neighbors" => central_stats["small_neighbors"],
        "central_large_neighbors" => central_stats["large_neighbors"],
        "low_mean_log_perm" => mean_vector(log_perms, low_stats["indices"]),
        "high_mean_log_perm" => mean_vector(log_perms, high_stats["indices"]),
        "central_mean_log_perm" => mean_vector(log_perms, central_stats["indices"]),
        "state_fraction" => config["state_fraction"],
        "small_neighbor_fraction" => config["small_neighbor_fraction"],
        "large_neighbor_fraction" => config["large_neighbor_fraction"],
        "min_cluster_size" => config["min_cluster_size"],
        "min_cluster_fraction" => config["min_cluster_fraction"],
        "silhouette_threshold" => config["silhouette_threshold"],
        "max_k" => config["max_k"],
        "random_seed" => config["random_seed"],
    )
end

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

    for label in ("global", "low", "high", "central")
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

function compute_local_ranks(log_perms::Matrix{Float64})
    n, p = size(log_perms)
    ranks = zeros(Float64, n, p)
    for j in 1:p
        ranks[:, j] = empirical_percentile_ranks(view(log_perms, :, j))
    end
    return ranks
end

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

compute_local_normal_scores(local_ranks::Matrix{Float64}) = inv_norm_cdf.(clamp.(local_ranks, 1e-6, 1.0 - 1e-6))

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

function compute_state_score(local_ranks::Matrix{Float64}, weights::Vector{Float64})
    weight_sum = sum(weights)
    weight_sum > 0 || error("State-score weights must sum to a positive value")
    return vec((local_ranks * weights) ./ weight_sum)
end

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

function choose_clustering(distance_matrix::Matrix{Float64},
                           state_score::Vector{Float64},
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
            medians = cluster_score_medians(state_score, fit["assignments"], k)
            best = Dict{String, Any}(
                "chosen_k" => k,
                "best_silhouette" => silhouette,
                "assignments" => fit["assignments"],
                "cluster_sizes" => cluster_sizes,
                "cluster_medoids" => fit["medoids"],
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
            "cluster_score_medians" => [median(state_score)],
            "cluster_order" => [1],
            "is_effectively_unimodal" => true,
            "silhouette_by_k" => silhouette_by_k,
            "valid_k_mask" => valid_k_mask,
        )
    end

    return best
end

window_seed_offset(window::AbstractString) = sum(Int(c) for c in codeunits(window))

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

function assign_to_medoids(distance_matrix::Matrix{Float64}, medoids::Vector{Int})
    n = size(distance_matrix, 1)
    assignments = zeros(Int, n)
    for i in 1:n
        distances = distance_matrix[i, medoids]
        assignments[i] = argmin(distances)
    end
    return assignments
end

function update_medoids(distance_matrix::Matrix{Float64}, assignments::Vector{Int}, k::Int)
    medoids = Int[]
    for cluster_id in 1:k
        members = findall(assignments .== cluster_id)
        isempty(members) && return Int[]
        push!(medoids, choose_medoid(members, distance_matrix))
    end
    return medoids
end

function choose_medoid(indices::Vector{Int}, distance_matrix::Matrix{Float64})
    costs = [sum(distance_matrix[idx, indices]) for idx in indices]
    return indices[argmin(costs)]
end

total_kmedoids_cost(distance_matrix::Matrix{Float64}, assignments::Vector{Int}, medoids::Vector{Int}) =
    sum(distance_matrix[i, medoids[assignments[i]]] for i in eachindex(assignments))

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

counts_from_assignments(assignments::Vector{Int}, k::Int) = [count(==(cluster_id), assignments) for cluster_id in 1:k]

function cluster_score_medians(state_score::Vector{Float64}, assignments::Vector{Int}, k::Int)
    medians = zeros(Float64, k)
    for cluster_id in 1:k
        members = findall(assignments .== cluster_id)
        medians[cluster_id] = median(state_score[members])
    end
    return medians
end

function build_state_libraries(state_score::Vector{Float64},
                               assignments::Vector{Int},
                               distance_matrix::Matrix{Float64},
                               config::Dict{String, Any})
    n = length(state_score)
    n_target = max(1, ceil(Int, config["state_fraction"] * n))

    chosen_k = maximum(assignments)
    if chosen_k == 1
        score_order = sortperm(state_score)
        low_indices = score_order[1:n_target]
        high_indices = score_order[end-n_target+1:end]
    else
        medians = cluster_score_medians(state_score, assignments, chosen_k)
        cluster_order = sortperm(medians)
        low_clusters = accumulate_clusters(cluster_order, assignments, n_target)
        high_clusters = accumulate_clusters(reverse(cluster_order), assignments, n_target)
        low_indices = sort(findall(in(low_clusters), assignments))
        high_indices = sort(findall(in(high_clusters), assignments))
    end

    blocked = Set(vcat(low_indices, high_indices))
    central_indices = choose_central_indices(state_score, n_target, blocked)
    return Dict{String, Any}(
        "low_indices" => unique(low_indices),
        "high_indices" => unique(high_indices),
        "central_indices" => unique(central_indices),
    )
end

function accumulate_clusters(cluster_order, assignments::Vector{Int}, n_target::Int)
    selected = Int[]
    total = 0
    for cluster_id in cluster_order
        push!(selected, cluster_id)
        total += count(==(cluster_id), assignments)
        total >= n_target && break
    end
    return selected
end

function choose_central_indices(state_score::Vector{Float64}, n_target::Int, blocked::Set{Int})
    med = median(state_score)
    order = sortperm(abs.(state_score .- med))
    selected = Int[]
    for idx in order
        idx in blocked && continue
        push!(selected, idx)
        length(selected) == n_target && return selected
    end
    for idx in order
        idx in selected && continue
        push!(selected, idx)
        length(selected) == n_target && break
    end
    return selected
end

function state_record(label::AbstractString,
                      indices::Vector{Int},
                      state_score::Vector{Float64},
                      distance_matrix::Matrix{Float64},
                      config::Dict{String, Any};
                      descending::Bool = false)
    isempty(indices) && error("State library $label is empty")
    ordered = descending ? indices[sortperm(state_score[indices], rev = true)] :
                           indices[sortperm(state_score[indices])]
    medoid_index = choose_medoid(indices, distance_matrix)
    small_neighbors = nearest_neighbors(medoid_index, indices, distance_matrix, config["small_neighbor_fraction"])
    large_neighbors = nearest_neighbors(medoid_index, indices, distance_matrix, config["large_neighbor_fraction"])
    return Dict{String, Any}(
        "indices" => indices,
        "ordered_indices" => ordered,
        "medoid_index" => medoid_index,
        "small_neighbors" => small_neighbors,
        "large_neighbors" => large_neighbors,
    )
end

function nearest_neighbors(target_index::Int,
                           candidates::Vector{Int},
                           distance_matrix::Matrix{Float64},
                           fraction::Float64)
    n_keep = max(1, ceil(Int, fraction * length(candidates)))
    ordered = sort(candidates; by = idx -> distance_matrix[target_index, idx])
    return ordered[1:n_keep]
end

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
