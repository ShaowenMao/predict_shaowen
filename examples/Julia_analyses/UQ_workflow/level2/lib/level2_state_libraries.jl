"""
    Level2StateLibraries

Low/high state-library, medoid, and perturbation-pool utilities for Level 2.

This module converts joint clusters and joint rank scores into state libraries,
then builds local and state-wide perturbation pools.
"""
module Level2StateLibraries

using Statistics
using ..Level2Clustering

export build_state_libraries,
       select_cluster_aware_state,
       state_record,
       local_pool_indices,
       mean_vector

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
        medians = Level2Clustering.cluster_joint_rank_medians(joint_rank_score, assignments, chosen_k)
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
    medoid_index = Level2Clustering.choose_medoid(indices, distance_matrix)
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

end
