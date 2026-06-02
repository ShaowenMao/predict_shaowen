"""
    Level3Bootstrap

Bootstrap stability calculations for Level 3 window similarity analysis.

The main output is a stable-similar-pair probability matrix:

```text
C_ij = P(delta_ij <= tau_delta)
```

where `delta_ij` is the normalized multivariate energy distance between two
windows and `tau_delta` is the similarity threshold.
"""
module Level3Bootstrap

using Dates
using Random
using Statistics

using ..Level3Distances

export bootstrap_stable_pairs

"""
    bootstrap_stable_pairs(states, windows; kwargs...)

Estimate stable-similar-pair probabilities by resampling realizations within
each window and recomputing normalized energy distances.

Sampling is with replacement. Set `bootstrap_sample_size <= 0` to use each
window's full sample count.
"""
function bootstrap_stable_pairs(states::Dict{String, Dict{String, Any}},
                                windows::Vector{String};
                                bootstrap_count::Int = 100,
                                bootstrap_sample_size::Int = 2000,
                                similarity_threshold::Real = 0.25,
                                stable_pair_probability_threshold::Real = 0.80,
                                random_seed::Int = 1729,
                                show_progress::Bool = true)
    bootstrap_count > 0 || error("bootstrap_count must be positive")

    libraries = Dict{String, Matrix{Float64}}()
    for window in windows
        libraries[window] = Level3Distances.matrix_float(states[window]["log_perms"])
    end

    nwin = length(windows)
    distances = zeros(Float64, bootstrap_count, nwin, nwin)
    below_threshold_count = zeros(Int, nwin, nwin)
    sample_size_used = Dict{String, Int}()

    started_at = now()
    start_time = time()
    progress_every = max(1, bootstrap_count ÷ 10)

    for b in 1:bootstrap_count
        rng = MersenneTwister(random_seed + b)
        boot_libraries = Dict{String, Matrix{Float64}}()
        for window in windows
            source = libraries[window]
            n_source = size(source, 1)
            n_sample = bootstrap_sample_size <= 0 ? n_source : bootstrap_sample_size
            indices = rand(rng, 1:n_source, n_sample)
            boot_libraries[window] = source[indices, :]
            sample_size_used[window] = n_sample
        end

        internal_spread = zeros(Float64, nwin)
        for (i, window) in enumerate(windows)
            internal_spread[i] = Level3Distances.mean_internal_spread(boot_libraries[window])
        end

        for i in 1:nwin
            for j in i+1:nwin
                x = boot_libraries[windows[i]]
                y = boot_libraries[windows[j]]
                energy = Level3Distances.energy_distance(
                    x, y;
                    spread_x = internal_spread[i],
                    spread_y = internal_spread[j],
                )
                denom = 0.5 * (internal_spread[i] + internal_spread[j])
                delta = denom > 0 ? energy / denom : 0.0
                distances[b, i, j] = delta
                distances[b, j, i] = delta
                if delta <= similarity_threshold
                    below_threshold_count[i, j] += 1
                    below_threshold_count[j, i] += 1
                end
            end
        end

        if show_progress && (b == 1 || b == bootstrap_count || b % progress_every == 0)
            elapsed = time() - start_time
            println("  bootstrap $b / $bootstrap_count completed ($(round(elapsed, digits = 1)) s)")
        end
    end

    probability = below_threshold_count ./ bootstrap_count
    for i in 1:nwin
        probability[i, i] = 1.0
    end

    summary = summarize_bootstrap_distances(distances)
    stable_pair_matrix = falses(nwin, nwin)
    for i in 1:nwin
        for j in i+1:nwin
            is_stable = probability[i, j] >= stable_pair_probability_threshold
            stable_pair_matrix[i, j] = is_stable
            stable_pair_matrix[j, i] = is_stable
        end
    end

    return Dict{String, Any}(
        "windows" => windows,
        "bootstrap_count" => bootstrap_count,
        "bootstrap_sample_size_requested" => bootstrap_sample_size,
        "bootstrap_sample_size_used" => sample_size_used,
        "similarity_threshold" => Float64(similarity_threshold),
        "stable_pair_probability_threshold" => Float64(stable_pair_probability_threshold),
        "random_seed" => random_seed,
        "started_at" => Dates.format(started_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
        "elapsed_seconds" => time() - start_time,
        "stable_pair_probability" => probability,
        "stable_pair_matrix" => stable_pair_matrix,
        "bootstrap_distance_mean" => summary["mean"],
        "bootstrap_distance_median" => summary["median"],
        "bootstrap_distance_p10" => summary["p10"],
        "bootstrap_distance_p90" => summary["p90"],
    )
end

function summarize_bootstrap_distances(distances::Array{Float64, 3})
    nboot, nwin, _ = size(distances)
    mean_matrix = zeros(Float64, nwin, nwin)
    median_matrix = zeros(Float64, nwin, nwin)
    p10_matrix = zeros(Float64, nwin, nwin)
    p90_matrix = zeros(Float64, nwin, nwin)

    for i in 1:nwin
        for j in i+1:nwin
            values = distances[:, i, j]
            mean_matrix[i, j] = mean(values)
            mean_matrix[j, i] = mean_matrix[i, j]
            median_matrix[i, j] = median(values)
            median_matrix[j, i] = median_matrix[i, j]
            p10_matrix[i, j] = quantile(values, 0.10)
            p10_matrix[j, i] = p10_matrix[i, j]
            p90_matrix[i, j] = quantile(values, 0.90)
            p90_matrix[j, i] = p90_matrix[i, j]
        end
    end

    return Dict(
        "mean" => mean_matrix,
        "median" => median_matrix,
        "p10" => p10_matrix,
        "p90" => p90_matrix,
    )
end

end
