"""
    step02_detect_joint_clusters(window_data, config)

Detect joint permeability clusters for one window.

This step clusters the 2000 joint realizations in the configured physical
distance space, currently `log_unit` by default. It also computes a rank cache
because the clustering decision needs the joint rank score to order clusters
from low to high after the k-medoids fit. The returned dictionary contains:

- `distance_info`: distance metric metadata and the pairwise distance matrix.
- `cluster_info`: chosen `K`, assignments, medoids, silhouettes, and order.
- `rank_cache`: local ranks, local normal scores, and joint rank score.
"""
function step02_detect_joint_clusters(window_data::Dict{String, Any},
                                      config::Dict{String, Any})
    log_perms = Matrix{Float64}(window_data["log_perms"])
    local_ranks = Level2Core.compute_local_ranks(log_perms)
    local_normal_scores = Level2Core.compute_local_normal_scores(local_ranks)
    joint_rank_score = Level2Core.compute_joint_rank_score(local_ranks, Float64.(config["weights"]))
    distance_info = Level2Core.build_distance_matrix(log_perms, local_normal_scores, config)
    cluster_info = Level2Core.choose_clustering(distance_info["distance_matrix"],
                                                joint_rank_score,
                                                config,
                                                window_data["window"])

    return Dict{String, Any}(
        "distance_info" => distance_info,
        "cluster_info" => cluster_info,
        "rank_cache" => Dict{String, Any}(
            "local_ranks" => local_ranks,
            "local_normal_scores" => local_normal_scores,
            "joint_rank_score" => joint_rank_score,
        ),
    )
end
