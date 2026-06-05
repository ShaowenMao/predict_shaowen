"""
    Level2Ranks

Rank-transform utilities for Level 2 within-window analysis.

This module owns the local percentile-rank transform, local normal-score
transform, and joint rank score used to order realizations from low to high.
"""
module Level2Ranks

export compute_local_ranks,
       empirical_percentile_ranks,
       compute_local_normal_scores,
       inv_norm_cdf,
       compute_joint_rank_score,
       compute_state_score

"""
    compute_local_ranks(log_perms)

Compute component-wise empirical percentile ranks within one window.

Each column of `log_perms` is ranked independently, so the output describes
whether each realization is locally low or high for `kxx`, `kyy`, and `kzz`.
"""
function compute_local_ranks(log_perms::Matrix{Float64})
    n, p = size(log_perms)
    ranks = zeros(Float64, n, p)
    for j in 1:p
        ranks[:, j] = empirical_percentile_ranks(view(log_perms, :, j))
    end
    return ranks
end

"""
    empirical_percentile_ranks(x)

Return tie-aware empirical percentile ranks in `(0, 1)` for one vector.
"""
function empirical_percentile_ranks(x::AbstractVector{<:Real})
    n = length(x)
    order = sortperm(x)
    ranks = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[order[j + 1]] == x[order[i]]
            j += 1
        end
        avg_rank = (i + j) / 2
        p = avg_rank / (n + 1)
        for k in i:j
            ranks[order[k]] = p
        end
        i = j + 1
    end
    return ranks
end

"""
    compute_local_normal_scores(local_ranks)

Transform local percentile ranks to normal scores with the inverse standard
normal CDF.
"""
compute_local_normal_scores(local_ranks::Matrix{Float64}) = inv_norm_cdf.(clamp.(local_ranks, 1e-6, 1.0 - 1e-6))

"""
    inv_norm_cdf(p)

Approximate the inverse standard normal cumulative distribution function.
"""
function inv_norm_cdf(p::Float64)
    p <= 0.0 && return -Inf
    p >= 1.0 && return Inf

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow
        q = sqrt(-2.0 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    elseif p <= phigh
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0)
    end
end

"""
    compute_joint_rank_score(local_ranks, weights)

Compute the weighted average local rank across `kxx`, `kyy`, and `kzz`.

This score is used only for low-to-high ordering and state-library
construction; it is not the clustering distance.
"""
function compute_joint_rank_score(local_ranks::Matrix{Float64}, weights::Vector{Float64})
    weight_sum = sum(weights)
    weight_sum > 0 || error("Joint-rank-score weights must sum to a positive value")
    return vec((local_ranks * weights) ./ weight_sum)
end

"""
    compute_state_score(local_ranks, weights)

Backward-compatible alias for `compute_joint_rank_score`.
"""
compute_state_score(local_ranks::Matrix{Float64}, weights::Vector{Float64}) =
    compute_joint_rank_score(local_ranks, weights)

end
