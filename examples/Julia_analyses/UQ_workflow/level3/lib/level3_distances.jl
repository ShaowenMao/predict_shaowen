"""
    Level3Distances

Distance calculations for Level 3 window similarity analysis.

The main output is a normalized 6 by 6 distance matrix comparing each pair of
windows' empirical joint `log10(k)` distributions.
"""
module Level3Distances

using Random
using Statistics

export compute_window_distances,
       energy_distance,
       mean_cross_distance,
       mean_internal_spread,
       matrix_float

"""
    compute_window_distances(states, windows; sample_size=0, random_seed=1729)

Compute full-data pairwise window distances for one geology.

For each pair of windows, the function computes multivariate empirical energy
distance in physical 3D `log10(k)` space. It then normalizes that distance by
the average internal spread of the two windows:

```text
delta_ij = energy_ij / (0.5 * (spread_i + spread_j))
```

Set `sample_size > 0` to use deterministic row subsampling for quick
experiments. The default `sample_size = 0` uses all realizations.
"""
function compute_window_distances(states::Dict{String, Dict{String, Any}},
                                  windows::Vector{String};
                                  sample_size::Int = 0,
                                  random_seed::Int = 1729)
    libraries = Dict{String, Matrix{Float64}}()
    for (idx, window) in enumerate(windows)
        log_perms = matrix_float(states[window]["log_perms"])
        libraries[window] = maybe_subsample_rows(log_perms, sample_size, random_seed + 100 * idx)
    end

    nwin = length(windows)
    internal_spread = zeros(Float64, nwin)
    for (i, window) in enumerate(windows)
        internal_spread[i] = mean_internal_spread(libraries[window])
    end

    energy = zeros(Float64, nwin, nwin)
    normalized = zeros(Float64, nwin, nwin)
    for i in 1:nwin
        for j in i+1:nwin
            x = libraries[windows[i]]
            y = libraries[windows[j]]
            value = energy_distance(x, y; spread_x = internal_spread[i], spread_y = internal_spread[j])
            denom = 0.5 * (internal_spread[i] + internal_spread[j])
            delta = denom > 0 ? value / denom : 0.0
            energy[i, j] = value
            energy[j, i] = value
            normalized[i, j] = delta
            normalized[j, i] = delta
        end
    end

    return Dict{String, Any}(
        "windows" => windows,
        "sample_size_used" => Dict(window => size(libraries[window], 1) for window in windows),
        "internal_spread" => internal_spread,
        "energy_distance" => energy,
        "normalized_distance" => normalized,
    )
end

"""
    energy_distance(x, y; spread_x=nothing, spread_y=nothing)

Compute the empirical multivariate energy distance between two matrices whose
rows are joint `log10(k)` samples.
"""
function energy_distance(x::Matrix{Float64},
                         y::Matrix{Float64};
                         spread_x = nothing,
                         spread_y = nothing)
    sx = spread_x === nothing ? mean_internal_spread(x) : Float64(spread_x)
    sy = spread_y === nothing ? mean_internal_spread(y) : Float64(spread_y)
    dxy = mean_cross_distance(x, y)
    return max(0.0, 2.0 * dxy - sx - sy)
end

"""
    mean_cross_distance(x, y)

Return the average Euclidean distance between all rows of `x` and all rows of
`y` in 3D log-permeability space.
"""
function mean_cross_distance(x::Matrix{Float64}, y::Matrix{Float64})
    size(x, 2) == 3 || error("x must have three columns")
    size(y, 2) == 3 || error("y must have three columns")
    nx = size(x, 1)
    ny = size(y, 1)
    total = 0.0
    @inbounds for i in 1:nx
        x1 = x[i, 1]
        x2 = x[i, 2]
        x3 = x[i, 3]
        for j in 1:ny
            d1 = x1 - y[j, 1]
            d2 = x2 - y[j, 2]
            d3 = x3 - y[j, 3]
            total += sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        end
    end
    return total / (nx * ny)
end

"""
    mean_internal_spread(x)

Return the average within-window pairwise Euclidean distance.

The denominator is `n^2`, matching the empirical energy-distance convention
that includes zero self-distances.
"""
function mean_internal_spread(x::Matrix{Float64})
    size(x, 2) == 3 || error("x must have three columns")
    n = size(x, 1)
    n == 0 && error("Cannot compute spread for an empty library")
    total_upper = 0.0
    @inbounds for i in 1:n-1
        x1 = x[i, 1]
        x2 = x[i, 2]
        x3 = x[i, 3]
        for j in i+1:n
            d1 = x1 - x[j, 1]
            d2 = x2 - x[j, 2]
            d3 = x3 - x[j, 3]
            total_upper += sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        end
    end
    return 2.0 * total_upper / (n * n)
end

"""
    maybe_subsample_rows(values, sample_size, seed)

Return all rows when `sample_size <= 0`; otherwise return a deterministic
subsample without replacement.
"""
function maybe_subsample_rows(values::Matrix{Float64}, sample_size::Int, seed::Int)
    if sample_size <= 0 || size(values, 1) <= sample_size
        return values
    end
    rng = MersenneTwister(seed)
    indices = sort(randperm(rng, size(values, 1))[1:sample_size])
    return values[indices, :]
end

"""
    matrix_float(values)

Convert array-like MAT-loaded values to a `Matrix{Float64}`.
"""
matrix_float(values) = Matrix{Float64}(values)

end

