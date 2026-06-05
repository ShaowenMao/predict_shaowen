"""
    Level2Clustering

Joint permeability clustering utilities for Level 2 within-window analysis.

This module owns k-medoids fitting, silhouette scoring, medoid selection, and
cluster low-to-high ordering by median joint rank score.
"""
module Level2Clustering

using Statistics
using Random

export choose_clustering,
       window_seed_offset,
       fit_kmedoids,
       init_medoids,
       assign_to_medoids,
       update_medoids,
       choose_medoid,
       total_kmedoids_cost,
       average_silhouette,
       counts_from_assignments,
       cluster_joint_rank_medians,
       cluster_score_medians

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

end
