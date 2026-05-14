"""
    step04_build_low_high_state_libraries(rank_step, cluster_step, config)

Build low and high state libraries from ordered joint clusters.

Clusters are ordered by median joint rank score. The low state is filled from
the low end of the ordered clusters and the high state from the high end,
taking only the needed part of a boundary cluster to preserve the configured
target state fraction. The output contains both unordered state membership
indices and joint-rank-ordered indices for sampling diagnostics.
"""
function step04_build_low_high_state_libraries(rank_step::Dict{String, Any},
                                               cluster_step::Dict{String, Any},
                                               config::Dict{String, Any})
    joint_rank_score = vec(Float64.(rank_step["joint_rank_score"]))
    assignments = vec(Int.(cluster_step["cluster_info"]["assignments"]))
    state_info = Level2Core.build_state_libraries(joint_rank_score, assignments, config)

    low_indices = sort(unique(vec(Int.(state_info["low_indices"]))))
    high_indices = sort(unique(vec(Int.(state_info["high_indices"]))))

    return Dict{String, Any}(
        "low_indices" => low_indices,
        "high_indices" => high_indices,
        "low_ordered_indices" => low_indices[sortperm(joint_rank_score[low_indices])],
        "high_ordered_indices" => high_indices[sortperm(joint_rank_score[high_indices], rev = true)],
    )
end
