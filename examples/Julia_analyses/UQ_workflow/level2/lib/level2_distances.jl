"""
    Level2Distances

Distance construction utilities for Level 2 within-window analysis.

This module builds the pairwise distance matrix used by joint clustering,
medoid selection, and perturbation-pool construction.
"""
module Level2Distances

export pairwise_euclidean,
       build_distance_matrix,
       weighted_features

"""
    pairwise_euclidean(z)

Compute a full pairwise Euclidean distance matrix for feature rows.
"""
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

"""
    build_distance_matrix(log_perms, local_normal_scores, config)

Build the pairwise distance matrix and distance metadata used for clustering,
medoids, and perturbation pools.

The default `log_unit` metric uses physical `log10(k)` values directly. The
`local_normal` metric is retained only as a sensitivity option.
"""
function build_distance_matrix(log_perms::Matrix{Float64},
                               local_normal_scores::Matrix{Float64},
                               config::Dict{String, Any})
    metric = String(get(config, "distance_metric", "log_unit"))
    weights = Float64.(get(config, "distance_weights", ones(size(log_perms, 2))))
    length(weights) == size(log_perms, 2) || error("distance_weights must match permeability component count")
    all(weights .> 0) || error("distance_weights must be positive")

    if metric == "local_normal"
        scales = ones(Float64, size(log_perms, 2))
        features = weighted_features(local_normal_scores, scales, weights)
    elseif metric == "log_unit"
        scales = ones(Float64, size(log_perms, 2))
        features = weighted_features(log_perms, scales, weights)
    else
        error("Unsupported distance_metric '$metric'. Use log_unit or local_normal.")
    end

    return Dict{String, Any}(
        "distance_metric" => metric,
        "distance_component_scales" => scales,
        "distance_weights" => weights,
        "distance_matrix" => pairwise_euclidean(features),
    )
end

"""
    weighted_features(values, scales, weights)

Scale and weight component columns before pairwise distance calculation.
"""
function weighted_features(values::Matrix{Float64}, scales::Vector{Float64}, weights::Vector{Float64})
    features = similar(values)
    for j in axes(values, 2)
        scale = scales[j] > 0 ? scales[j] : 1.0
        features[:, j] .= values[:, j] .* sqrt(weights[j]) ./ scale
    end
    return features
end

end
