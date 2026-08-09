"""Exact one-dimensional empirical Wasserstein-1 distance."""
function wasserstein_1d(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    isempty(x) && error("Wasserstein input x is empty")
    isempty(y) && error("Wasserstein input y is empty")
    xs = sort(Float64.(x))
    ys = sort(Float64.(y))
    i = 1
    j = 1
    cdf_x = 0.0
    cdf_y = 0.0
    previous = min(xs[1], ys[1])
    distance = 0.0
    while i <= length(xs) || j <= length(ys)
        next_x = i <= length(xs) ? xs[i] : Inf
        next_y = j <= length(ys) ? ys[j] : Inf
        value = min(next_x, next_y)
        distance += abs(cdf_x - cdf_y) * (value - previous)
        while i <= length(xs) && xs[i] == value
            i += 1
        end
        while j <= length(ys) && ys[j] == value
            j += 1
        end
        cdf_x = (i - 1) / length(xs)
        cdf_y = (j - 1) / length(ys)
        previous = value
    end
    return distance
end

function deterministic_reduce(matrix::Matrix{Float64}, max_points::Int)
    max_points <= 0 && return matrix
    size(matrix, 1) <= max_points && return matrix
    indices = unique(round.(Int, range(1, size(matrix, 1); length = max_points)))
    return matrix[indices, :]
end

"""Multivariate energy distance using Euclidean distance in 3D logK space."""
function energy_distance_3d(x::Matrix{Float64}, y::Matrix{Float64}; max_points::Int = 0)
    size(x, 2) == 3 || error("Energy-distance input x must have three columns")
    size(y, 2) == 3 || error("Energy-distance input y must have three columns")
    x_reduced = deterministic_reduce(x, max_points)
    y_reduced = deterministic_reduce(y, max_points)
    nx = size(x_reduced, 1)
    ny = size(y_reduced, 1)

    cross_sum = 0.0
    @inbounds for i in 1:nx, j in 1:ny
        d1 = x_reduced[i, 1] - y_reduced[j, 1]
        d2 = x_reduced[i, 2] - y_reduced[j, 2]
        d3 = x_reduced[i, 3] - y_reduced[j, 3]
        cross_sum += sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    end

    within_x = 0.0
    @inbounds for i in 1:(nx - 1), j in (i + 1):nx
        d1 = x_reduced[i, 1] - x_reduced[j, 1]
        d2 = x_reduced[i, 2] - x_reduced[j, 2]
        d3 = x_reduced[i, 3] - x_reduced[j, 3]
        within_x += sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    end

    within_y = 0.0
    @inbounds for i in 1:(ny - 1), j in (i + 1):ny
        d1 = y_reduced[i, 1] - y_reduced[j, 1]
        d2 = y_reduced[i, 2] - y_reduced[j, 2]
        d3 = y_reduced[i, 3] - y_reduced[j, 3]
        within_y += sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    end

    value = 2 * cross_sum / (nx * ny) -
            2 * within_x / (nx * nx) -
            2 * within_y / (ny * ny)
    return max(0.0, value)
end

function average_tie_ranks(values::Vector{Float64})
    order = sortperm(values)
    ranks = zeros(Float64, length(values))
    position = 1
    while position <= length(order)
        endpoint = position
        while endpoint < length(order) && values[order[endpoint + 1]] == values[order[position]]
            endpoint += 1
        end
        rank_value = (position + endpoint) / 2
        for sorted_position in position:endpoint
            ranks[order[sorted_position]] = rank_value
        end
        position = endpoint + 1
    end
    return ranks
end

function pearson_correlation(x::Vector{Float64}, y::Vector{Float64})
    length(x) == length(y) || error("Correlation vectors must have equal length")
    length(x) > 1 || error("Correlation requires at least two values")
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    denominator = sqrt(sum(abs2, x_centered) * sum(abs2, y_centered))
    denominator == 0 && return 0.0
    return sum(x_centered .* y_centered) / denominator
end

spearman_correlation(x::Vector{Float64}, y::Vector{Float64}) =
    pearson_correlation(average_tie_ranks(x), average_tie_ranks(y))

function independent_rows(rows::Vector{ManifestRow})
    return [row for row in rows
            if row.case_type == INDEPENDENT_CASE && row.use_for_probabilistic_uq]
end

function sampled_logk(rows::Vector{ManifestRow}, window::AbstractString)
    selected = sort([row for row in independent_rows(rows) if row.window_id == window];
                    by = row -> (row.replicate_id, row.slice_id))
    matrix = Matrix{Float64}(undef, length(selected), 3)
    for (index, row) in enumerate(selected)
        matrix[index, :] .= (row.log_kxx, row.log_kyy, row.log_kzz)
    end
    return matrix
end

function bootstrap_source_rows(library::PredictLibrary,
                               cfg::SamplingConfig,
                               bootstrap_id::Int,
                               sample_count::Int,
                               purpose::AbstractString)
    seed = stable_case_seed(cfg.base_seed, library.geology_id, "qc", purpose,
                            bootstrap_id, library.window_id;
                            version = cfg.seed_method_version)
    return [stable_uniform_index(seed, draw_id, n_accepted(library);
                                 version = cfg.seed_method_version)
            for draw_id in 1:sample_count]
end

function pairwise_window_correlations(rows::Vector{ManifestRow},
                                      libraries::Dict{String, PredictLibrary},
                                      cfg::SamplingConfig)
    scores = Dict{String, Vector{Float64}}()
    for window in cfg.windows
        ordered = sort([row for row in independent_rows(rows) if row.window_id == window];
                       by = row -> (row.replicate_id, row.slice_id))
        scores[window] = [libraries[window].joint_rank_score[row.source_library_row]
                          for row in ordered]
    end
    correlations = Dict{Tuple{String, String}, Float64}()
    for i in 1:(length(cfg.windows) - 1), j in (i + 1):length(cfg.windows)
        first_window = cfg.windows[i]
        second_window = cfg.windows[j]
        correlations[(first_window, second_window)] =
            spearman_correlation(scores[first_window], scores[second_window])
    end
    return correlations
end

function bootstrap_independence_max(libraries::Dict{String, PredictLibrary},
                                    cfg::SamplingConfig,
                                    bootstrap_id::Int,
                                    sample_count::Int)
    scores = Dict{String, Vector{Float64}}()
    for window in cfg.windows
        library = libraries[window]
        selected = bootstrap_source_rows(library, cfg, bootstrap_id, sample_count,
                                         "independence_reference")
        scores[window] = library.joint_rank_score[selected]
    end
    max_abs = 0.0
    for i in 1:(length(cfg.windows) - 1), j in (i + 1):length(cfg.windows)
        correlation = spearman_correlation(scores[cfg.windows[i]], scores[cfg.windows[j]])
        max_abs = max(max_abs, abs(correlation))
    end
    return max_abs
end

function independence_seed_structure(rows::Vector{ManifestRow},
                                     cfg::SamplingConfig)
    selected = independent_rows(rows)
    expected_keys = Set((replicate_id, window)
                        for replicate_id in 1:cfg.phase1_n_independent
                        for window in cfg.windows)
    seeds = Dict{Tuple{Int, String}, UInt64}()
    slice_ids = Dict{Tuple{Int, String}, Set{Int}}()
    valid = true
    for row in selected
        key = (row.replicate_id, row.window_id)
        if haskey(seeds, key)
            valid &= seeds[key] == row.sampling_seed
        else
            seeds[key] = row.sampling_seed
        end
        push!(get!(slice_ids, key, Set{Int}()), row.slice_id)
        valid &= row.sampling_seed_method == cfg.seed_method_version
    end
    valid &= Set(keys(seeds)) == expected_keys
    valid &= all(get(slice_ids, key, Set{Int}()) == Set(1:cfg.n_slices)
                 for key in expected_keys)
    valid &= length(unique(values(seeds))) == length(expected_keys)
    return (
        pass = valid,
        stream_count = length(seeds),
        expected_stream_count = length(expected_keys),
        unique_seed_count = length(unique(values(seeds))),
    )
end

"""
    run_ensemble_qc(rows, libraries, cfg; bootstrap_count=cfg.qc_bootstrap_count)

Assess pooled Phase 1 marginal fidelity and verify that no unintended shared
rank structure was introduced across windows. Joint energy distance uses a
deterministic point cap configured by `qc.energy_max_points`; component-wise
Wasserstein distances use all 1,044 selected and 2,000 source values.
"""
function run_ensemble_qc(rows::Vector{ManifestRow},
                         libraries::Dict{String, PredictLibrary},
                         cfg::SamplingConfig;
                         bootstrap_count::Int = cfg.qc_bootstrap_count)
    selected = independent_rows(rows)
    expected_per_window = cfg.phase1_n_independent * cfg.n_slices
    length(selected) == expected_per_window * length(cfg.windows) ||
        error("Ensemble QC requires all Phase 1 independent assignments")

    distance_rows = Vector{Vector{Any}}()
    observed_metrics = Float64[]
    reference_by_metric = Vector{Vector{Float64}}()
    for window in cfg.windows
        library = libraries[window]
        production = sampled_logk(rows, window)
        size(production, 1) == expected_per_window || error("QC sample count mismatch for $window")
        observed = [wasserstein_1d(production[:, component], library.logK[:, component])
                    for component in 1:3]
        observed_energy = energy_distance_3d(production, library.logK;
                                             max_points = cfg.qc_energy_max_points)

        reference_w = [Float64[] for _ in 1:3]
        reference_energy = Float64[]
        for bootstrap_id in 1:bootstrap_count
            source_rows = bootstrap_source_rows(library, cfg, bootstrap_id,
                                                expected_per_window, "marginal_reference")
            synthetic = library.logK[source_rows, :]
            for component in 1:3
                push!(reference_w[component],
                      wasserstein_1d(synthetic[:, component], library.logK[:, component]))
            end
            push!(reference_energy,
                  energy_distance_3d(synthetic, library.logK;
                                     max_points = cfg.qc_energy_max_points))
        end

        for component in 1:3
            threshold = bootstrap_count > 0 ?
                quantile(reference_w[component], cfg.qc_confidence_level) : NaN
            push!(distance_rows, Any[
                window,
                ("log_kxx", "log_kyy", "log_kzz")[component],
                "wasserstein_1d",
                observed[component],
                threshold,
                bootstrap_count == 0 ? "not_evaluated" : observed[component] <= threshold,
            ])
            push!(observed_metrics, observed[component])
            push!(reference_by_metric, reference_w[component])
        end
        energy_threshold = bootstrap_count > 0 ?
            quantile(reference_energy, cfg.qc_confidence_level) : NaN
        push!(distance_rows, Any[
            window,
            "joint_logk",
            "energy_distance_3d",
            observed_energy,
            energy_threshold,
            bootstrap_count == 0 ? "not_evaluated" : observed_energy <= energy_threshold,
        ])
        push!(observed_metrics, observed_energy)
        push!(reference_by_metric, reference_energy)
    end

    individual_pass_count = bootstrap_count == 0 ? 0 :
        count(row -> row[6] === true, distance_rows)
    if bootstrap_count > 0
        # Distances have different physical scales. Normalize each metric by
        # its bootstrap median, then compare the largest observed normalized
        # distance with the bootstrap distribution of the same maximum. This
        # is a family-wise implementation check; individual 95% exceedances
        # remain useful diagnostics but are expected occasionally across 24
        # simultaneous metrics.
        metric_scale = [max(median(reference), eps(Float64))
                        for reference in reference_by_metric]
        observed_global_max = maximum(observed_metrics ./ metric_scale)
        bootstrap_global_max = [
            maximum(reference_by_metric[metric][bootstrap_id] /
                    metric_scale[metric]
                    for metric in eachindex(reference_by_metric))
            for bootstrap_id in 1:bootstrap_count
        ]
        marginal_global_threshold = quantile(
            bootstrap_global_max, cfg.qc_confidence_level)
        marginal_ensemble_pass = observed_global_max <= marginal_global_threshold
    else
        observed_global_max = NaN
        marginal_global_threshold = NaN
        marginal_ensemble_pass = "not_evaluated"
    end

    seed_structure = independence_seed_structure(rows, cfg)
    production_correlations = pairwise_window_correlations(rows, libraries, cfg)
    production_max = maximum(abs.(collect(values(production_correlations))))
    reference_max = [bootstrap_independence_max(libraries, cfg, bootstrap_id,
                                                expected_per_window)
                     for bootstrap_id in 1:bootstrap_count]
    diagnostic_correlation_threshold = bootstrap_count > 0 ?
        quantile(reference_max, cfg.qc_confidence_level) : NaN
    acceptance_correlation_threshold = bootstrap_count > 0 ?
        quantile(reference_max, cfg.qc_independence_acceptance_confidence_level) : NaN
    empirical_p_value = bootstrap_count > 0 ?
        (1 + count(value -> value >= production_max, reference_max)) /
        (bootstrap_count + 1) : NaN
    diagnostic_correlation_pass = bootstrap_count == 0 ?
        "not_evaluated" : production_max <= diagnostic_correlation_threshold
    acceptance_correlation_pass = bootstrap_count == 0 ?
        "not_evaluated" : production_max <= acceptance_correlation_threshold
    independence_implementation_pass = bootstrap_count == 0 ?
        "not_evaluated" : seed_structure.pass && acceptance_correlation_pass
    correlation_rows = [[first(pair), last(pair), value]
                        for (pair, value) in sort(collect(production_correlations);
                                                  by = item -> item[1])]

    return Dict{String, Any}(
        "distance_rows" => distance_rows,
        "individual_marginal_metric_pass_count" => individual_pass_count,
        "individual_marginal_metric_count" => length(distance_rows),
        "marginal_global_max_normalized_distance" => observed_global_max,
        "marginal_global_max_threshold" => marginal_global_threshold,
        "marginal_ensemble_pass" => marginal_ensemble_pass,
        "correlation_rows" => correlation_rows,
        "maximum_absolute_spearman" => production_max,
        "independence_seed_structure_pass" => seed_structure.pass,
        "independence_seed_stream_count" => seed_structure.stream_count,
        "independence_expected_seed_stream_count" =>
            seed_structure.expected_stream_count,
        "independence_unique_seed_count" => seed_structure.unique_seed_count,
        "independence_diagnostic_confidence_level" => cfg.qc_confidence_level,
        "independence_diagnostic_threshold" => diagnostic_correlation_threshold,
        "independence_diagnostic_pass" => diagnostic_correlation_pass,
        "independence_acceptance_confidence_level" =>
            cfg.qc_independence_acceptance_confidence_level,
        "independence_acceptance_threshold" => acceptance_correlation_threshold,
        "independence_empirical_p_value" => empirical_p_value,
        "independence_implementation_pass" => independence_implementation_pass,
        "overall_implementation_pass" => bootstrap_count == 0 ?
            "not_evaluated" :
            (marginal_ensemble_pass && independence_implementation_pass),
        "bootstrap_count" => bootstrap_count,
        "confidence_level" => cfg.qc_confidence_level,
        "energy_max_points" => cfg.qc_energy_max_points,
    )
end

"""Write ensemble-level marginal-fidelity and independence QC tables."""
function write_ensemble_qc(qc::Dict{String, Any},
                           cfg::SamplingConfig,
                           geology_id::AbstractString)
    root = joinpath(cfg.output_root, "phase1", "qc")
    distance_path = joinpath(root, "$(geology_id)_ensemble_marginal_fidelity.csv")
    correlation_path = joinpath(root, "$(geology_id)_window_rank_correlations.csv")
    summary_path = joinpath(root, "$(geology_id)_ensemble_qc_summary.toml")
    write_csv(distance_path,
              ["window_id", "component", "metric", "observed_distance",
               "bootstrap_upper_threshold", "pass"],
              qc["distance_rows"])
    write_csv(correlation_path,
              ["window_1", "window_2", "spearman_correlation"],
              qc["correlation_rows"])
    summary = Dict{String, Any}(
        "schema_version" => 2,
        "design_version" => cfg.design_version,
        "geology_id" => String(geology_id),
        "bootstrap_count" => qc["bootstrap_count"],
        "confidence_level" => qc["confidence_level"],
        "energy_max_points" => qc["energy_max_points"],
        "individual_marginal_metric_pass_count" =>
            qc["individual_marginal_metric_pass_count"],
        "individual_marginal_metric_count" =>
            qc["individual_marginal_metric_count"],
        "marginal_global_max_normalized_distance" =>
            qc["marginal_global_max_normalized_distance"],
        "marginal_global_max_threshold" =>
            qc["marginal_global_max_threshold"],
        "marginal_ensemble_pass" => qc["marginal_ensemble_pass"],
        "maximum_absolute_spearman" => qc["maximum_absolute_spearman"],
        "independence_seed_structure_pass" =>
            qc["independence_seed_structure_pass"],
        "independence_seed_stream_count" => qc["independence_seed_stream_count"],
        "independence_expected_seed_stream_count" =>
            qc["independence_expected_seed_stream_count"],
        "independence_unique_seed_count" => qc["independence_unique_seed_count"],
        "independence_diagnostic_confidence_level" =>
            qc["independence_diagnostic_confidence_level"],
        "independence_diagnostic_threshold" =>
            qc["independence_diagnostic_threshold"],
        "independence_diagnostic_pass" => qc["independence_diagnostic_pass"],
        "independence_acceptance_confidence_level" =>
            qc["independence_acceptance_confidence_level"],
        "independence_acceptance_threshold" =>
            qc["independence_acceptance_threshold"],
        "independence_empirical_p_value" => qc["independence_empirical_p_value"],
        "independence_implementation_pass" => qc["independence_implementation_pass"],
        "overall_implementation_pass" => qc["overall_implementation_pass"],
    )
    mkpath(dirname(summary_path))
    open(summary_path, "w") do io
        TOML.print(io, summary; sorted = true)
    end
    return Dict("distance_path" => distance_path,
                "correlation_path" => correlation_path,
                "summary_path" => summary_path)
end
