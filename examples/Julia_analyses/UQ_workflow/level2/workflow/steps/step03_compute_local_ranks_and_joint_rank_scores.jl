"""
    step03_compute_local_ranks_and_joint_rank_scores(window_data, config; cluster_step = nothing)

Compute the local rank variables used to order samples and clusters.

For each component, this step computes empirical local percentile ranks within
the current window, transforms those ranks to local normal scores, and averages
the component-wise ranks into the joint rank score. If `cluster_step` already
contains the rank cache from Step 02, that cache is reused so the modular
workflow exactly matches the clustering calculation.
"""
function step03_compute_local_ranks_and_joint_rank_scores(window_data::Dict{String, Any},
                                                          config::Dict{String, Any};
                                                          cluster_step::Union{Nothing, Dict{String, Any}} = nothing)
    if cluster_step !== nothing && haskey(cluster_step, "rank_cache")
        cache = cluster_step["rank_cache"]
        return Dict{String, Any}(
            "local_ranks" => Matrix{Float64}(cache["local_ranks"]),
            "local_normal_scores" => Matrix{Float64}(cache["local_normal_scores"]),
            "joint_rank_score" => vec(Float64.(cache["joint_rank_score"])),
        )
    end

    log_perms = Matrix{Float64}(window_data["log_perms"])
    local_ranks = Level2Core.compute_local_ranks(log_perms)
    local_normal_scores = Level2Core.compute_local_normal_scores(local_ranks)
    joint_rank_score = Level2Core.compute_joint_rank_score(local_ranks, Float64.(config["weights"]))

    return Dict{String, Any}(
        "local_ranks" => local_ranks,
        "local_normal_scores" => local_normal_scores,
        "joint_rank_score" => joint_rank_score,
    )
end
