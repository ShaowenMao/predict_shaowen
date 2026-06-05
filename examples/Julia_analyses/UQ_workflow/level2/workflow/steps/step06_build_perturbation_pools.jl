"""
    step06_build_perturbation_pools(state_libraries, medoids, cluster_step, config)

Construct local and state-wide perturbation pools for low/high states.

For each state, the local pool is medoid-centered and cluster-preserving: its
candidate set is the intersection of the state library and the cluster
containing the state medoid. The state-wide pool is the full low or high state
library. These pools are the final state-conditioned sampling pools used by
Level 2.
"""
function step06_build_perturbation_pools(state_libraries::Dict{String, Any},
                                         medoids::Dict{String, Any},
                                         cluster_step::Dict{String, Any},
                                         config::Dict{String, Any})
    distance_matrix = Matrix{Float64}(cluster_step["distance_info"]["distance_matrix"])
    assignments = vec(Int.(cluster_step["cluster_info"]["assignments"]))
    result = Dict{String, Any}()

    for label in ("low", "high")
        state_indices = vec(Int.(state_libraries["$(label)_indices"]))
        medoid_index = Int(medoids["$(label)_medoid_index"])
        medoid_cluster_id = assignments[medoid_index]
        candidates = [idx for idx in state_indices if assignments[idx] == medoid_cluster_id]
        local_pool = Level2StateLibraries.local_pool_indices(medoid_index,
                                                             candidates,
                                                             distance_matrix,
                                                             Float64(config["local_pool_fraction"]),
                                                             Int(config["local_pool_min_count"]))

        result["$(label)_local_pool_candidates"] = candidates
        result["$(label)_local_pool"] = local_pool
        result["$(label)_state_wide_pool"] = sort(unique(state_indices))
    end

    return result
end
