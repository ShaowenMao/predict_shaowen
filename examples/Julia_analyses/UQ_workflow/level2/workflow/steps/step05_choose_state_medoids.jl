"""
    step05_choose_state_medoids(state_libraries, cluster_step)

Choose actual PREDICT realizations as representative medoids.

The global, low-state, and high-state medoids are selected by minimizing total
distance to other samples in their respective candidate sets, using the same
pairwise distance matrix as the clustering step. The returned medoid indices
refer to rows in the original 2000-realization window library.
"""
function step05_choose_state_medoids(state_libraries::Dict{String, Any},
                                     cluster_step::Dict{String, Any})
    distance_matrix = Matrix{Float64}(cluster_step["distance_info"]["distance_matrix"])
    assignments = vec(Int.(cluster_step["cluster_info"]["assignments"]))
    n = size(distance_matrix, 1)

    low_medoid_index = Level2Clustering.choose_medoid(vec(Int.(state_libraries["low_indices"])), distance_matrix)
    high_medoid_index = Level2Clustering.choose_medoid(vec(Int.(state_libraries["high_indices"])), distance_matrix)
    global_medoid_index = Level2Clustering.choose_medoid(collect(1:n), distance_matrix)

    return Dict{String, Any}(
        "global_medoid_index" => global_medoid_index,
        "low_medoid_index" => low_medoid_index,
        "high_medoid_index" => high_medoid_index,
        "low_medoid_cluster_id" => assignments[low_medoid_index],
        "high_medoid_cluster_id" => assignments[high_medoid_index],
    )
end
