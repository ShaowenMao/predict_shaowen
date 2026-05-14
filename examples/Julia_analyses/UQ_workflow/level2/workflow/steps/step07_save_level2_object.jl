using Dates
using Statistics

"""
    step07_save_level2_object(window_data, rank_step, cluster_step,
                              state_libraries, medoids, perturbation_pools,
                              config, output_root)

Package and save one complete Level 2 window-state object.

This step does not add new methodology. It gathers the outputs from Steps
01-06 into the MAT schema consumed by the table, figure, validation, and
sampling scripts. The returned tuple is `(state, state_path)`, where `state` is
the in-memory dictionary and `state_path` is the saved MAT-file path.
"""
function step07_save_level2_object(window_data::Dict{String, Any},
                                   rank_step::Dict{String, Any},
                                   cluster_step::Dict{String, Any},
                                   state_libraries::Dict{String, Any},
                                   medoids::Dict{String, Any},
                                   perturbation_pools::Dict{String, Any},
                                   config::Dict{String, Any},
                                   output_root::AbstractString)
    window = String(window_data["window"])
    log_perms = Matrix{Float64}(window_data["log_perms"])
    raw_perms = Matrix{Float64}(window_data["raw_perms"])
    joint_rank_score = vec(Float64.(rank_step["joint_rank_score"]))
    cluster_info = cluster_step["cluster_info"]
    distance_info = cluster_step["distance_info"]

    state = Dict{String, Any}(
        "schema_version" => "level2_window_state_v1",
        "created_at" => string(Dates.now()),
        "geology_id" => String(config["geology_id"]),
        "window" => window,
        "source_path" => String(window_data["source_path"]),
        "source_label" => String(window_data["source_label"]),
        "n_samples" => size(log_perms, 1),
        "raw_perms" => raw_perms,
        "log_perms" => log_perms,
        "local_ranks" => Matrix{Float64}(rank_step["local_ranks"]),
        "local_normal_scores" => Matrix{Float64}(rank_step["local_normal_scores"]),
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
        "global_medoid_index" => medoids["global_medoid_index"],
        "low_indices" => state_libraries["low_indices"],
        "high_indices" => state_libraries["high_indices"],
        "low_ordered_indices" => state_libraries["low_ordered_indices"],
        "high_ordered_indices" => state_libraries["high_ordered_indices"],
        "low_medoid_index" => medoids["low_medoid_index"],
        "high_medoid_index" => medoids["high_medoid_index"],
        "low_medoid_cluster_id" => medoids["low_medoid_cluster_id"],
        "high_medoid_cluster_id" => medoids["high_medoid_cluster_id"],
        "low_local_pool_candidates" => perturbation_pools["low_local_pool_candidates"],
        "high_local_pool_candidates" => perturbation_pools["high_local_pool_candidates"],
        "low_local_pool" => perturbation_pools["low_local_pool"],
        "low_state_wide_pool" => perturbation_pools["low_state_wide_pool"],
        "high_local_pool" => perturbation_pools["high_local_pool"],
        "high_state_wide_pool" => perturbation_pools["high_state_wide_pool"],
        "low_mean_log_perm" => mean_log_perm(log_perms, state_libraries["low_indices"]),
        "high_mean_log_perm" => mean_log_perm(log_perms, state_libraries["high_indices"]),
        "state_fraction" => config["state_fraction"],
        "local_pool_fraction" => config["local_pool_fraction"],
        "local_pool_min_count" => config["local_pool_min_count"],
        "min_cluster_size" => config["min_cluster_size"],
        "min_cluster_fraction" => config["min_cluster_fraction"],
        "silhouette_threshold" => config["silhouette_threshold"],
        "max_k" => config["max_k"],
        "random_seed" => config["random_seed"],
    )

    state_path = joinpath(output_root, "window_states", window, "$(window)_level2_state.mat")
    Level2IO.save_window_state(state_path, state)
    return state, state_path
end

"""
    mean_log_perm(log_perms, indices)

Return the component-wise mean `log10(k)` vector for a set of sample indices.
"""
function mean_log_perm(log_perms::Matrix{Float64}, indices)
    return vec(mean(log_perms[vec(Int.(indices)), :]; dims = 1))
end
