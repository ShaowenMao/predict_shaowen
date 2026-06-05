@isdefined(Level2Ranks) || include(joinpath(@__DIR__, "level2_ranks.jl"))
@isdefined(Level2Distances) || include(joinpath(@__DIR__, "level2_distances.jl"))
@isdefined(Level2Clustering) || include(joinpath(@__DIR__, "level2_clustering.jl"))
@isdefined(Level2StateLibraries) || include(joinpath(@__DIR__, "level2_state_libraries.jl"))

"""
    Level2StateObject

Assembly utilities for complete Level 2 window-state objects.

This module is intentionally narrow: focused modules compute ranks, distances,
clusters, state libraries, medoids, and perturbation pools; this module only
assembles those pieces into the canonical saved Level 2 object schema.
"""
module Level2StateObject

using Dates
using ..Level2Ranks
using ..Level2Distances
using ..Level2Clustering
using ..Level2StateLibraries

export build_window_state

"""
    build_window_state(log_perms, raw_perms, window, source_path, source_label, config)

Build one complete Level 2 window-state object.

The returned dictionary is the canonical MAT schema used by downstream table,
plotting, validation, and sampling outputs. Clustering is performed using the
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

    local_ranks = Level2Ranks.compute_local_ranks(log_perms)
    joint_rank_score = Level2Ranks.compute_joint_rank_score(local_ranks, config["weights"])
    local_normal_scores = Level2Ranks.compute_local_normal_scores(local_ranks)
    distance_info = Level2Distances.build_distance_matrix(log_perms, local_normal_scores, config)
    distance_matrix = distance_info["distance_matrix"]

    cluster_info = Level2Clustering.choose_clustering(distance_matrix, joint_rank_score, config, window)
    global_medoid_index = Level2Clustering.choose_medoid(collect(1:size(log_perms, 1)), distance_matrix)
    state_info = Level2StateLibraries.build_state_libraries(joint_rank_score,
                                                            cluster_info["assignments"],
                                                            config)

    low_stats = Level2StateLibraries.state_record("low", state_info["low_indices"], joint_rank_score,
                                                  distance_matrix, cluster_info["assignments"], config)
    high_stats = Level2StateLibraries.state_record("high", state_info["high_indices"], joint_rank_score,
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
        "low_mean_log_perm" => Level2StateLibraries.mean_vector(log_perms, low_stats["indices"]),
        "high_mean_log_perm" => Level2StateLibraries.mean_vector(log_perms, high_stats["indices"]),
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

end
