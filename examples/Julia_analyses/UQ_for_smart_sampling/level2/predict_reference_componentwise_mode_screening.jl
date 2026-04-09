#!/usr/bin/env julia

"""
Step-4 marginal mode screening for pooled rigorous reference ensembles.

This script follows the Level 2 workflow after the reference ensembles
have been judged sufficiently consistent to pool. For each selected window,
it pools the saved reference ensembles R1/R2/R3 and performs marginal
mode screening in log10(k) space for:

    log10(kxx), log10(kyy), log10(kzz)

The script:
1. Builds a baseline Gaussian-kernel KDE for each component.
2. Sweeps a user-defined family of bandwidth multipliers around that
   baseline bandwidth.
3. Detects candidate peaks and antimodes.
4. Screens baseline candidate modes for stability across the bandwidth
   family.
5. Defines final modal intervals from the stable baseline peaks.
6. Computes empirical mode-aware summaries from the pooled sample:
       - mode weights
       - mode means
       - within-mode variances
       - within-mode spread W
       - between-mode separation B
       - mode entropy H

The KDE is used only for mode identification. The pooled empirical sample
remains the primary source for the reported probability masses and moments.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie", "KernelDensity"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_mode_screening.jl
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_mode_screening.jl --windows famp1
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_mode_screening.jl --windows famp1 --bandwidth-factor-min 0.4 --bandwidth-factor-max 2.5 --num-bandwidths 11
"""

const REQUIRED_PACKAGES = ["MAT", "CairoMakie", "KernelDensity"]
missing_packages = [pkg for pkg in REQUIRED_PACKAGES if Base.find_package(pkg) === nothing]
if !isempty(missing_packages)
    pkg_list = join(["\"" * pkg * "\"" for pkg in missing_packages], ", ")
    error("Missing Julia packages: $(join(missing_packages, ", ")). Install them with:\n" *
          "using Pkg; Pkg.add([$pkg_list])")
end

using MAT
using CairoMakie
using KernelDensity
using Statistics
using Printf

CairoMakie.activate!()

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_mode_screening")),
        "windows" => "",
        "bandwidth-factor-min" => "0.5",
        "bandwidth-factor-max" => "2.0",
        "num-bandwidths" => "5",
        "grid-size" => "801",
        "min-mode-mass" => "0.01",
        "persistence-threshold" => "0.6",
        "min-prominence" => "0.05",
        "merge-prominence-threshold" => "0.1",
        "merge-separation-threshold" => "1.0",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_help()
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(options, key) || error("Unknown option $arg")
            i < length(args) || error("Missing value for $arg")
            options[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end

    requested_windows = isempty(options["windows"]) ? String[] :
                        String[strip(w) for w in split(options["windows"], ",") if !isempty(strip(w))]
    factor_min = parse(Float64, options["bandwidth-factor-min"])
    factor_max = parse(Float64, options["bandwidth-factor-max"])
    num_bandwidths = parse(Int, options["num-bandwidths"])

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        requested_windows = requested_windows,
        factor_min = factor_min,
        factor_max = factor_max,
        num_bandwidths = num_bandwidths,
        grid_size = parse(Int, options["grid-size"]),
        min_mode_mass = parse(Float64, options["min-mode-mass"]),
        persistence_threshold = parse(Float64, options["persistence-threshold"]),
        min_prominence = parse(Float64, options["min-prominence"]),
        merge_prominence_threshold = parse(Float64, options["merge-prominence-threshold"]),
        merge_separation_threshold = parse(Float64, options["merge-separation-threshold"]),
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_mode_screening.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>              Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>            Folder where figures and CSVs are saved")
    println("  --windows <names>              Comma-separated list like famp1,famp2")
    println("  --bandwidth-factor-min <x>     Minimum bandwidth factor relative to h0 (default: 0.5)")
    println("  --bandwidth-factor-max <x>     Maximum bandwidth factor relative to h0 (default: 2.0)")
    println("  --num-bandwidths <n>           Number of log-spaced bandwidths (default: 5)")
    println("  --grid-size <n>                Number of KDE grid points (default: 801)")
    println("  --min-mode-mass <x>            Minimum pooled mass for a stable mode (default: 0.01)")
    println("  --persistence-threshold <x>    Minimum bandwidth-support fraction (default: 0.6)")
    println("  --min-prominence <x>           Minimum relative prominence for a stable mode (default: 0.05)")
    println("  --merge-prominence-threshold <x>  Merge adjacent peaks if either prominence is below this value (default: 0.1)")
    println("  --merge-separation-threshold <x>  Merge adjacent peaks if Sw is below this value (default: 1.0)")
    println("  -h, --help                     Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No window reference folders found in $(opt.data_dir)")

    println("Processing $(length(windows)) window(s) from $(opt.data_dir)")
    for (window, reference_dir) in windows
        println("  - $window")
        process_window(window, reference_dir, opt.output_dir;
                       factor_min = opt.factor_min,
                       factor_max = opt.factor_max,
                       num_bandwidths = opt.num_bandwidths,
                       grid_size = opt.grid_size,
                       min_mode_mass = opt.min_mode_mass,
                       persistence_threshold = opt.persistence_threshold,
                       min_prominence = opt.min_prominence,
                       merge_prominence_threshold = opt.merge_prominence_threshold,
                       merge_separation_threshold = opt.merge_separation_threshold)
    end
    println("Saved outputs to $(opt.output_dir)")
end

function collect_windows(data_dir::AbstractString, requested_windows::Vector{String})
    isdir(data_dir) || error("Data directory does not exist: $data_dir")

    window_map = Dict{String, String}()
    for entry in readdir(data_dir)
        window_dir = joinpath(data_dir, entry)
        reference_dir = joinpath(window_dir, "references")
        if isdir(window_dir) && isdir(reference_dir)
            files = filter(f -> endswith(f, ".mat") && startswith(f, "reference_R"), readdir(reference_dir))
            if !isempty(files)
                window_map[entry] = reference_dir
            end
        end
    end

    if isempty(requested_windows)
        return [(window, window_map[window]) for window in sort(collect(keys(window_map)))]
    end

    missing = [w for w in requested_windows if !haskey(window_map, w)]
    isempty(missing) || error("Requested window(s) not found: $(join(missing, ", "))")
    return [(window, window_map[window]) for window in requested_windows]
end

function process_window(window::AbstractString, reference_dir::AbstractString, output_dir::AbstractString;
                        factor_min::Float64,
                        factor_max::Float64,
                        num_bandwidths::Int,
                        grid_size::Int,
                        min_mode_mass::Float64,
                        persistence_threshold::Float64,
                        min_prominence::Float64,
                        merge_prominence_threshold::Float64,
                        merge_separation_threshold::Float64)
    references = load_references(reference_dir)
    isempty(references) && error("No reference MAT files found in $reference_dir")
    pooled = pool_references(references)

    component_results = NamedTuple[]
    for ic in eachindex(COMPONENT_NAMES)
        values = pooled.y[:, ic]
        reference_values = [ref.y[:, ic] for ref in references]
        push!(component_results,
              screen_component_modes(values, reference_values;
                                     component_name = COMPONENT_NAMES[ic],
                                     factor_min = factor_min,
                                     factor_max = factor_max,
                                     num_bandwidths = num_bandwidths,
                                     grid_size = grid_size,
                                     min_mode_mass = min_mode_mass,
                                     persistence_threshold = persistence_threshold,
                                     min_prominence = min_prominence,
                                     merge_prominence_threshold = merge_prominence_threshold,
                                     merge_separation_threshold = merge_separation_threshold))
    end

    window_dir = joinpath(output_dir, window)
    mkpath(window_dir)

    save_mode_screening_figure(window, component_results,
        joinpath(window_dir, "$(window)_componentwise_mode_screening.png"))

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_mode_candidates.csv"),
                   candidate_header(),
                   build_candidate_rows(window, component_results))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_final_modes.csv"),
                   final_mode_header(references),
                   build_final_mode_rows(window, references, component_results))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_mode_summary.csv"),
                   ["window", "component", "stable_mode_count", "global_mean", "within_mode_spread_W", "between_mode_separation_B", "mode_entropy_H", "baseline_bandwidth_h0", "chosen_bandwidth_factor", "center_tolerance", "min_mode_mass", "persistence_threshold", "min_prominence", "merge_prominence_threshold", "merge_separation_threshold"],
                   build_mode_summary_rows(window, component_results))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_key_summary.csv"),
                   key_summary_header(component_results),
                   build_key_summary_rows(window, component_results))

    save_component_results_mat(joinpath(window_dir, "$(window)_componentwise_mode_screening.mat"),
                               pooled, references, component_results,
                               factor_min, factor_max, num_bandwidths,
                               grid_size, min_mode_mass, persistence_threshold,
                               min_prominence, merge_prominence_threshold,
                               merge_separation_threshold)
end

function load_references(reference_dir::AbstractString)
    files = filter(f -> endswith(f, ".mat") && startswith(f, "reference_R"), readdir(reference_dir))
    sort!(files)

    references = NamedTuple[]
    for file in files
        filepath = joinpath(reference_dir, file)
        data = matread(filepath)
        haskey(data, "perms") || error("File does not contain perms: $filepath")
        perms = Matrix{Float64}(data["perms"])
        size(perms, 2) == 3 || error("Expected perms to have 3 columns in $filepath")
        all(perms .> 0) || error("perms contains non-positive values in $filepath")
        ref_name = replace(replace(file, ".mat" => ""), "reference_" => "")
        push!(references, (name = ref_name, perms = perms, y = log10.(perms)))
    end
    return references
end

function pool_references(references)
    perms = reduce(vcat, [ref.perms for ref in references])
    y = reduce(vcat, [ref.y for ref in references])
    return (perms = perms, y = y)
end

function screen_component_modes(values::Vector{Float64}, reference_values::Vector{Vector{Float64}};
                                component_name::AbstractString,
                                factor_min::Float64,
                                factor_max::Float64,
                                num_bandwidths::Int,
                                grid_size::Int,
                                min_mode_mass::Float64,
                                persistence_threshold::Float64,
                                min_prominence::Float64,
                                merge_prominence_threshold::Float64,
                                merge_separation_threshold::Float64)
    baseline_bandwidth = silverman_bandwidth(values)
    center_tolerance = max(0.5 * baseline_bandwidth, 1e-6)
    bandwidth_factors = log_spaced_factors(factor_min, factor_max, num_bandwidths)
    xlow = minimum(values) - 3.0 * baseline_bandwidth
    xhigh = maximum(values) + 3.0 * baseline_bandwidth

    kde_results = NamedTuple[]
    for factor in bandwidth_factors
        h = max(factor * baseline_bandwidth, eps())
        kd = kde(values; bandwidth = h, boundary = (xlow, xhigh), npoints = grid_size)
        grid = Vector{Float64}(kd.x)
        density = Vector{Float64}(kd.density)
        peak_idx = local_maxima_indices(density)
        isempty(peak_idx) && (peak_idx = [argmax(density)])
        peak_idx, candidate_ranges, boundary_indices, candidate_prominences,
        candidate_widths, nearest_separation = merge_adjacent_weak_close_peaks(
            grid, density, peak_idx;
            merge_prominence_threshold = merge_prominence_threshold,
            merge_separation_threshold = merge_separation_threshold)
        min_idx = local_minima_indices(density)
        candidate_masses = [interval_mass(values, bounds.left, bounds.right) for bounds in candidate_ranges]
        candidate_means = [interval_mean(values, bounds.left, bounds.right) for bounds in candidate_ranges]
        candidate_variances = [interval_variance(values, bounds.left, bounds.right) for bounds in candidate_ranges]
        push!(kde_results, (
            factor = factor,
            bandwidth = h,
            grid = grid,
            density = density,
            peak_idx = peak_idx,
            peak_x = grid[peak_idx],
            min_idx = min_idx,
            min_x = grid[min_idx],
            candidate_ranges = candidate_ranges,
            boundary_indices = boundary_indices,
            candidate_masses = candidate_masses,
            candidate_means = candidate_means,
            candidate_variances = candidate_variances,
            candidate_prominences = candidate_prominences,
            nearest_separation = nearest_separation,
            candidate_widths = candidate_widths,
        ))
    end

    baseline_index = argmin(abs.(bandwidth_factors .- 1.0))

    clusters = cluster_peak_tracks(kde_results, center_tolerance)
    stable_mask = falses(length(clusters))
    for (i, cluster) in enumerate(clusters)
        stable_mask[i] = (cluster.persistence >= persistence_threshold) &&
                         (cluster.median_mass >= min_mode_mass) &&
                         (cluster.median_prominence >= min_prominence)
    end
    if !any(stable_mask) && !isempty(clusters)
        best_idx = argmax([cluster.median_mass for cluster in clusters])
        stable_mask[best_idx] = true
    end

    stable_clusters = clusters[stable_mask]
    stable_count = length(stable_clusters)
    chosen_bandwidth_index = choose_smoothest_admissible_bandwidth(kde_results, stable_clusters,
                                                                   min_mode_mass, min_prominence)
    chosen = kde_results[chosen_bandwidth_index]
    chosen_peak_idx, chosen_peak_x = chosen_peaks_for_stable_clusters(chosen, stable_clusters, center_tolerance)
    final_boundaries = final_boundaries_from_indices(chosen.grid, chosen.density, chosen_peak_idx)
    final_intervals = final_modal_intervals(final_boundaries)
    final_weights = [interval_mass(values, bounds.left, bounds.right) for bounds in final_intervals]
    final_means = [interval_mean(values, bounds.left, bounds.right) for bounds in final_intervals]
    final_variances = [interval_variance(values, bounds.left, bounds.right) for bounds in final_intervals]
    reference_masses = zeros(Float64, length(reference_values), stable_count)
    for (ir, ref_vals) in enumerate(reference_values)
        for im in 1:stable_count
            bounds = final_intervals[im]
            reference_masses[ir, im] = interval_mass(ref_vals, bounds.left, bounds.right)
        end
    end

    global_mean = mean(values)
    within_mode_spread = sum(final_weights .* final_variances)
    between_mode_separation = sum(final_weights .* (final_means .- global_mean).^2)
    mode_entropy = -sum((w > 0 ? w * log(w) : 0.0) for w in final_weights)

    return (
        component_name = String(component_name),
        values = copy(values),
        baseline_bandwidth = baseline_bandwidth,
        center_tolerance = center_tolerance,
        bandwidth_factors = bandwidth_factors,
        factor_min = factor_min,
        factor_max = factor_max,
        num_bandwidths = num_bandwidths,
        min_mode_mass = min_mode_mass,
        persistence_threshold = persistence_threshold,
        min_prominence = min_prominence,
        merge_prominence_threshold = merge_prominence_threshold,
        merge_separation_threshold = merge_separation_threshold,
        kde_results = kde_results,
        baseline_index = baseline_index,
        clusters = clusters,
        stable_mask = stable_mask,
        stable_clusters = stable_clusters,
        chosen_bandwidth_index = chosen_bandwidth_index,
        chosen_bandwidth_factor = chosen.factor,
        chosen_bandwidth = chosen.bandwidth,
        chosen_peak_idx = chosen_peak_idx,
        chosen_peak_x = chosen_peak_x,
        final_boundaries = final_boundaries,
        final_intervals = final_intervals,
        final_weights = final_weights,
        final_means = final_means,
        final_variances = final_variances,
        reference_masses = reference_masses,
        stable_mode_count = stable_count,
        global_mean = global_mean,
        within_mode_spread = within_mode_spread,
        between_mode_separation = between_mode_separation,
        mode_entropy = mode_entropy,
    )
end

function log_spaced_factors(factor_min::Float64, factor_max::Float64, num_bandwidths::Int)
    factor_min > 0 || error("bandwidth-factor-min must be positive")
    factor_max > 0 || error("bandwidth-factor-max must be positive")
    num_bandwidths >= 2 || error("num-bandwidths must be at least 2")
    factors = exp.(range(log(factor_min), log(factor_max), length = num_bandwidths))
    return collect(factors)
end

function silverman_bandwidth(values::Vector{Float64})
    n = length(values)
    s = std(values; corrected = true)
    q25, q75 = quantile(values, [0.25, 0.75])
    sigma = min(s, (q75 - q25) / 1.34)
    if !isfinite(sigma) || sigma <= 0
        sigma = max(s, 1e-3)
    end
    if !isfinite(sigma) || sigma <= 0
        sigma = 1e-3
    end
    return 0.9 * sigma * n^(-1/5)
end

function local_maxima_indices(density::Vector{Float64})
    idx = Int[]
    for i in 2:(length(density) - 1)
        left = density[i] - density[i - 1]
        right = density[i + 1] - density[i]
        if left > 0 && right <= 0
            push!(idx, i)
        end
    end
    return idx
end

function local_minima_indices(density::Vector{Float64})
    idx = Int[]
    for i in 2:(length(density) - 1)
        left = density[i] - density[i - 1]
        right = density[i + 1] - density[i]
        if left < 0 && right >= 0
            push!(idx, i)
        end
    end
    return idx
end

function candidate_ranges_from_density(grid::Vector{Float64}, density::Vector{Float64}, peak_idx::Vector{Int})
    boundary_indices = Int[]
    if length(peak_idx) > 1
        for i in 1:(length(peak_idx) - 1)
            lo = peak_idx[i]
            hi = peak_idx[i + 1]
            local_min_offset = argmin(density[lo:hi]) - 1
            push!(boundary_indices, lo + local_min_offset)
        end
    end

    boundaries = grid[boundary_indices]
    return final_modal_intervals(boundaries), boundary_indices
end

function candidate_peak_prominences(density::Vector{Float64}, peak_idx::Vector{Int}, boundary_indices::Vector{Int})
    n = length(peak_idx)
    prominences = zeros(Float64, n)
    if n == 1
        prominences[1] = 1.0
        return prominences
    end

    valley_heights = density[boundary_indices]
    for i in 1:n
        peak_height = density[peak_idx[i]]
        if i == 1
            ref_valley = valley_heights[1]
        elseif i == n
            ref_valley = valley_heights[end]
        else
            ref_valley = max(valley_heights[i - 1], valley_heights[i])
        end
        prominences[i] = peak_height > 0 ? max(0.0, 1.0 - ref_valley / peak_height) : 0.0
    end
    return prominences
end

function merge_adjacent_weak_close_peaks(grid::Vector{Float64}, density::Vector{Float64}, peak_idx::Vector{Int};
                                         merge_prominence_threshold::Float64,
                                         merge_separation_threshold::Float64)
    # Conservative extra guard: only merge across a shared valley if that valley
    # is sufficiently shallow relative to the smaller adjacent peak.
    merge_valley_ratio_threshold = 0.8
    current_peak_idx = sort(copy(peak_idx))
    while length(current_peak_idx) > 1
        candidate_ranges, boundary_indices = candidate_ranges_from_density(grid, density, current_peak_idx)
        candidate_prominences = candidate_peak_prominences(density, current_peak_idx, boundary_indices)
        candidate_widths = candidate_peak_widths(grid, current_peak_idx, boundary_indices)

        merge_pairs = Tuple{Int, Float64}[]
        for i in 1:(length(current_peak_idx) - 1)
            sw = peak_separation_sw(grid[current_peak_idx[i]], grid[current_peak_idx[i + 1]],
                                    candidate_widths[i], candidate_widths[i + 1])
            valley_ratio = shared_valley_ratio(density, current_peak_idx[i],
                                               current_peak_idx[i + 1],
                                               boundary_indices[i])
            if (candidate_prominences[i] < merge_prominence_threshold ||
                candidate_prominences[i + 1] < merge_prominence_threshold) &&
               valley_ratio >= merge_valley_ratio_threshold &&
               sw < merge_separation_threshold
                push!(merge_pairs, (i, sw))
            end
        end

        isempty(merge_pairs) && break
        best_pair = merge_pairs[argmin([pair[2] for pair in merge_pairs])][1]
        left_peak = current_peak_idx[best_pair]
        right_peak = current_peak_idx[best_pair + 1]
        keep_peak = density[left_peak] >= density[right_peak] ? left_peak : right_peak

        merged_peak_idx = Int[]
        if best_pair > 1
            append!(merged_peak_idx, current_peak_idx[1:(best_pair - 1)])
        end
        push!(merged_peak_idx, keep_peak)
        if best_pair + 1 < length(current_peak_idx)
            append!(merged_peak_idx, current_peak_idx[(best_pair + 2):end])
        end
        current_peak_idx = merged_peak_idx
    end

    candidate_ranges, boundary_indices = candidate_ranges_from_density(grid, density, current_peak_idx)
    candidate_prominences = candidate_peak_prominences(density, current_peak_idx, boundary_indices)
    candidate_widths = candidate_peak_widths(grid, current_peak_idx, boundary_indices)
    nearest_separation = nearest_peak_separation(grid, current_peak_idx, candidate_widths)
    return current_peak_idx, candidate_ranges, boundary_indices, candidate_prominences,
           candidate_widths, nearest_separation
end

function candidate_peak_widths(grid::Vector{Float64}, peak_idx::Vector{Int}, boundary_indices::Vector{Int})
    n = length(peak_idx)
    nearest = fill(NaN, n)
    widths = zeros(Float64, n)
    if n == 1
        widths[1] = max(grid[end] - grid[1], eps())
        return widths
    end

    for i in 1:n
        peak_x = grid[peak_idx[i]]
        if i == 1
            right_boundary = grid[boundary_indices[1]]
            widths[i] = max(2.0 * (right_boundary - peak_x), eps())
        elseif i == n
            left_boundary = grid[boundary_indices[end]]
            widths[i] = max(2.0 * (peak_x - left_boundary), eps())
        else
            left_boundary = grid[boundary_indices[i - 1]]
            right_boundary = grid[boundary_indices[i]]
            widths[i] = max(right_boundary - left_boundary, eps())
        end
    end
    return widths
end

function shared_valley_ratio(density::Vector{Float64}, left_peak_idx::Int, right_peak_idx::Int, valley_idx::Int)
    denom = max(min(density[left_peak_idx], density[right_peak_idx]), eps())
    return density[valley_idx] / denom
end

function peak_separation_sw(x1::Float64, x2::Float64, w1::Float64, w2::Float64)
    denom = max((w1 + w2) / 2.0, eps())
    return abs(x2 - x1) / denom
end

function nearest_peak_separation(grid::Vector{Float64}, peak_idx::Vector{Int}, widths::Vector{Float64})
    n = length(peak_idx)
    nearest = fill(NaN, n)
    if n <= 1
        return nearest
    end

    for i in 1:n
        sw_left = Inf
        sw_right = Inf
        if i > 1
            sw_left = peak_separation_sw(grid[peak_idx[i - 1]], grid[peak_idx[i]],
                                         widths[i - 1], widths[i])
        end
        if i < n
            sw_right = peak_separation_sw(grid[peak_idx[i]], grid[peak_idx[i + 1]],
                                          widths[i], widths[i + 1])
        end
        nearest[i] = min(sw_left, sw_right)
    end
    return nearest
end

function cluster_peak_tracks(kde_results, center_tolerance::Float64)
    clusters = NamedTuple[]
    for (ibw, result) in enumerate(kde_results)
        for ip in eachindex(result.peak_x)
            x = result.peak_x[ip]
            candidate_idx = 0
            candidate_dist = Inf
            for (ic, cluster) in enumerate(clusters)
                if any(cluster.bandwidth_indices .== ibw)
                    continue
                end
                dist = abs(cluster.center - x)
                if dist <= center_tolerance && dist < candidate_dist
                    candidate_idx = ic
                    candidate_dist = dist
                end
            end

            if candidate_idx == 0
                push!(clusters, (
                    centers = [x],
                    bandwidth_indices = [ibw],
                    peak_indices = [ip],
                    masses = [result.candidate_masses[ip]],
                    prominences = [result.candidate_prominences[ip]],
                    separations = [result.nearest_separation[ip]],
                    center = x,
                ))
            else
                old = clusters[candidate_idx]
                new_centers = vcat(old.centers, [x])
                clusters[candidate_idx] = (
                    centers = new_centers,
                    bandwidth_indices = vcat(old.bandwidth_indices, [ibw]),
                    peak_indices = vcat(old.peak_indices, [ip]),
                    masses = vcat(old.masses, [result.candidate_masses[ip]]),
                    prominences = vcat(old.prominences, [result.candidate_prominences[ip]]),
                    separations = vcat(old.separations, [result.nearest_separation[ip]]),
                    center = median(new_centers),
                )
            end
        end
    end

    sort!(clusters, by = c -> c.center)
    enriched = NamedTuple[]
    n_bandwidths = length(kde_results)
    for (i, cluster) in enumerate(clusters)
        finite_separations = [s for s in cluster.separations if isfinite(s)]
        push!(enriched, (
            id = i,
            center = cluster.center,
            centers = cluster.centers,
            bandwidth_indices = cluster.bandwidth_indices,
            peak_indices = cluster.peak_indices,
            masses = cluster.masses,
            prominences = cluster.prominences,
            separations = cluster.separations,
            persistence = length(cluster.bandwidth_indices) / n_bandwidths,
            median_mass = median(cluster.masses),
            median_prominence = median(cluster.prominences),
            median_separation = isempty(finite_separations) ? NaN : median(finite_separations),
        ))
    end
    return enriched
end

function choose_smoothest_admissible_bandwidth(kde_results, stable_clusters, min_mode_mass::Float64, min_prominence::Float64)
    isempty(stable_clusters) && return length(kde_results)

    best_index = 0
    best_coverage = -1
    for ibw in reverse(eachindex(kde_results))
        result = kde_results[ibw]
        covered = 0
        for cluster in stable_clusters
            members = findall(cluster.bandwidth_indices .== ibw)
            if isempty(members)
                continue
            end
            peak_id = cluster.peak_indices[members[1]]
            if result.candidate_masses[peak_id] >= min_mode_mass &&
               result.candidate_prominences[peak_id] >= min_prominence
                covered += 1
            end
        end
        if covered == length(stable_clusters)
            return ibw
        end
        if covered > best_coverage
            best_index = ibw
            best_coverage = covered
        end
    end

    return best_index > 0 ? best_index : argmin(abs.([result.factor for result in kde_results] .- 1.0))
end

function chosen_peaks_for_stable_clusters(chosen, stable_clusters, center_tolerance::Float64)
    peak_ids = Int[]
    peak_x = Float64[]
    for cluster in stable_clusters
        dist = abs.(chosen.peak_x .- cluster.center)
        if any(dist .<= center_tolerance)
            local_idx = argmin(dist)
        else
            local_idx = argmin(dist)
        end
        push!(peak_ids, chosen.peak_idx[local_idx])
        push!(peak_x, chosen.peak_x[local_idx])
    end

    order = sortperm(peak_x)
    return peak_ids[order], peak_x[order]
end

function final_boundaries_from_indices(grid::Vector{Float64}, density::Vector{Float64}, peak_indices::Vector{Int})
    boundaries = Float64[]
    if length(peak_indices) <= 1
        return boundaries
    end
    for i in 1:(length(peak_indices) - 1)
        lo = peak_indices[i]
        hi = peak_indices[i + 1]
        local_min_offset = argmin(density[lo:hi]) - 1
        push!(boundaries, grid[lo + local_min_offset])
    end
    return boundaries
end

function final_modal_intervals(boundaries::Vector{Float64})
    intervals = NamedTuple[]
    if isempty(boundaries)
        push!(intervals, (left = -Inf, right = Inf))
        return intervals
    end

    push!(intervals, (left = -Inf, right = boundaries[1]))
    for i in 2:length(boundaries)
        push!(intervals, (left = boundaries[i - 1], right = boundaries[i]))
    end
    push!(intervals, (left = boundaries[end], right = Inf))
    return intervals
end

function interval_mask(values::Vector{Float64}, left::Float64, right::Float64)
    if isinf(left) && left < 0 && isinf(right) && right > 0
        return trues(length(values))
    elseif isinf(left) && left < 0
        return values .<= right
    elseif isinf(right) && right > 0
        return values .> left
    else
        return (values .> left) .& (values .<= right)
    end
end

function interval_mass(values::Vector{Float64}, left::Float64, right::Float64)
    mask = interval_mask(values, left, right)
    return count(mask) / length(values)
end

function interval_mean(values::Vector{Float64}, left::Float64, right::Float64)
    mask = interval_mask(values, left, right)
    subset = values[mask]
    isempty(subset) && return NaN
    return mean(subset)
end

function interval_variance(values::Vector{Float64}, left::Float64, right::Float64)
    mask = interval_mask(values, left, right)
    subset = values[mask]
    length(subset) <= 1 && return 0.0
    return var(subset; corrected = true)
end

function nice_density_top(ymax::Float64)
    ymax <= 0 && return 1.0
    magnitude = 10.0 ^ floor(log10(ymax))
    normalized = ymax / magnitude
    nice_normalized = normalized <= 1 ? 1.0 :
                      normalized <= 2 ? 2.0 :
                      normalized <= 2.5 ? 2.5 :
                      normalized <= 5 ? 5.0 : 10.0
    return nice_normalized * magnitude
end

function fixed_density_ticks(ymax::Float64)
    ytop = nice_density_top(1.05 * ymax)
    return ytop, collect(range(0.0, ytop; length = 5))
end

function save_mode_screening_figure(window::AbstractString, component_results, output_path::AbstractString)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    common_xticks = [-7, -5, -3, -1, 1]
    common_xlims = (-7.0, 1.0)

    for (ic, result) in enumerate(component_results)
        chosen = result.kde_results[result.chosen_bandwidth_index]
        ladder_colors = [RGBAf(0.45, 0.45, 0.45, a) for a in range(0.45, 0.85, length = length(result.kde_results))]
        chosen_color = RGBf(0.02, 0.27, 0.58)
        all_ymax = maximum(maximum(kde_result.density) for kde_result in result.kde_results)
        ytop, yticks = fixed_density_ticks(all_ymax)

        ax = Axis(fig[1, ic],
                  xlabel = COMPONENT_LABELS[ic],
                  ylabel = (ic == 1 ? "Density" : ""),
                  title = @sprintf("%s: %d mode%s, chosen h = %.2f×h₀",
                                   result.component_name,
                                   result.stable_mode_count,
                                   result.stable_mode_count == 1 ? "" : "s",
                                   result.chosen_bandwidth_factor),
                  titlealign = :left,
                  titlesize = 22,
                  xlabelsize = 22,
                  ylabelsize = 22,
                  xticks = common_xticks,
                  yticks = yticks,
                  xticklabelsize = 22,
                  yticklabelsize = 22,
                  xgridcolor = RGBAf(0, 0, 0, 0.08),
                  ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false,
                  rightspinevisible = false)
        hist!(ax, result.values;
              bins = 45,
              normalization = :pdf,
              color = (RGBf(0.62, 0.64, 0.67), 0.20),
              strokecolor = (RGBf(0.62, 0.64, 0.67), 0.35),
              strokewidth = 0.6)
        for (ibw, kde_result) in enumerate(result.kde_results)
            lines!(ax, kde_result.grid, kde_result.density;
                   color = (ibw == result.chosen_bandwidth_index ? chosen_color : ladder_colors[ibw]),
                   linewidth = (ibw == result.chosen_bandwidth_index ? 3.6 : 1.5),
                   linestyle = (ibw == result.chosen_bandwidth_index ? :solid : :solid),
                   label = @sprintf("%.2f×h₀", kde_result.factor))
        end

        if !isempty(result.chosen_peak_idx)
            scatter!(ax, chosen.grid[result.chosen_peak_idx], chosen.density[result.chosen_peak_idx];
                     color = :black, marker = :circle, markersize = 9, label = "chosen peaks")
        end
        xlims!(ax, common_xlims...)
        ylims!(ax, 0.0, ytop)
        axislegend(ax, position = :lt, framevisible = true, backgroundcolor = RGBAf(1, 1, 1, 0.90),
                   labelsize = 22, patchlabelgap = 8, rowgap = 5)

        if !isempty(result.final_weights)
            summary_text = join([
                @sprintf("mode %d: π = %.3f", i, result.final_weights[i]) for i in eachindex(result.final_weights)
            ], "\n")
            text!(ax,
                  0.035, 0.49;
                  space = :relative,
                  text = summary_text,
                  align = (:left, :top),
                  fontsize = 22,
                  color = :black)
        end

        panel_tag = string("(", Char('a' + ic - 1), ")")
        text!(ax,
              0.98, 0.98;
              space = :relative,
              text = panel_tag,
              align = (:right, :top),
              fontsize = 22,
              font = :bold,
              color = :black)
    end

    Label(fig[0, :], "$window pooled marginal KDE screening in log10(k) space", fontsize = 24, font = :bold)
    Label(fig[2, :], "Gray bars: empirical distribution. Thin gray curves: bandwidth ladder. Thick blue curve: chosen smoothing used for the final modal partition.", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function candidate_header()
    return ["window", "component", "cluster_id", "center_log10k", "persistence", "median_mass", "median_prominence", "median_nearest_separation", "stable_flag"]
end

function build_candidate_rows(window::AbstractString, component_results)
    rows = Vector{Vector{String}}()
    for result in component_results
        for (i, cluster) in enumerate(result.clusters)
            push!(rows, [
                String(window),
                result.component_name,
                string(i),
                fmt(cluster.center),
                fmt(cluster.persistence),
                fmt(cluster.median_mass),
                fmt(cluster.median_prominence),
                fmt(cluster.median_separation),
                string(result.stable_mask[i]),
            ])
        end
    end
    return rows
end

function final_mode_header(references)
    header = ["window", "component", "mode_id", "center_log10k", "left_bound", "right_bound",
              "mode_weight", "mode_mean", "mode_variance", "support_fraction"]
    append!(header, ["mass_" * String(ref.name) for ref in references])
    return header
end

function build_final_mode_rows(window::AbstractString, references, component_results)
    rows = Vector{Vector{String}}()
    for result in component_results
        for mode_id in 1:result.stable_mode_count
            bounds = result.final_intervals[mode_id]
            cluster = result.stable_clusters[mode_id]
            row = [
                String(window),
                result.component_name,
                string(mode_id),
                fmt(result.chosen_peak_x[mode_id]),
                bound_string(bounds.left),
                bound_string(bounds.right),
                fmt(result.final_weights[mode_id]),
                fmt(result.final_means[mode_id]),
                fmt(result.final_variances[mode_id]),
                fmt(cluster.persistence),
            ]
            append!(row, [fmt(result.reference_masses[ir, mode_id]) for ir in 1:length(references)])
            push!(rows, row)
        end
    end
    return rows
end

function build_mode_summary_rows(window::AbstractString, component_results)
    rows = Vector{Vector{String}}()
    for result in component_results
        push!(rows, [
            String(window),
            result.component_name,
            string(result.stable_mode_count),
            fmt(result.global_mean),
            fmt(result.within_mode_spread),
            fmt(result.between_mode_separation),
            fmt(result.mode_entropy),
            fmt(result.baseline_bandwidth),
            fmt(result.chosen_bandwidth_factor),
            fmt(result.center_tolerance),
            fmt(result.min_mode_mass),
            fmt(result.persistence_threshold),
            fmt(result.min_prominence),
            fmt(result.merge_prominence_threshold),
            fmt(result.merge_separation_threshold),
        ])
    end
    return rows
end

function key_summary_header(component_results)
    max_modes = maximum(result.stable_mode_count for result in component_results)
    header = ["window", "component", "stable_mode_count", "chosen_bandwidth_factor",
              "global_mean", "within_mode_spread_W", "between_mode_separation_B", "mode_entropy_H"]
    for mode_id in 1:max_modes
        append!(header, [
            @sprintf("mode_%d_center_log10k", mode_id),
            @sprintf("mode_%d_pi", mode_id),
            @sprintf("mode_%d_left_bound", mode_id),
            @sprintf("mode_%d_right_bound", mode_id),
            @sprintf("mode_%d_mean", mode_id),
            @sprintf("mode_%d_variance", mode_id),
            @sprintf("mode_%d_support_fraction", mode_id),
            @sprintf("mode_%d_median_mass", mode_id),
            @sprintf("mode_%d_median_prominence", mode_id),
            @sprintf("mode_%d_median_nearest_separation", mode_id),
        ])
    end
    return header
end

function build_key_summary_rows(window::AbstractString, component_results)
    max_modes = maximum(result.stable_mode_count for result in component_results)
    rows = Vector{Vector{String}}()
    for result in component_results
        row = [
            String(window),
            result.component_name,
            string(result.stable_mode_count),
            fmt(result.chosen_bandwidth_factor),
            fmt(result.global_mean),
            fmt(result.within_mode_spread),
            fmt(result.between_mode_separation),
            fmt(result.mode_entropy),
        ]
        for mode_id in 1:max_modes
            if mode_id <= result.stable_mode_count
                bounds = result.final_intervals[mode_id]
                cluster = result.stable_clusters[mode_id]
                append!(row, [
                    fmt(result.chosen_peak_x[mode_id]),
                    fmt(result.final_weights[mode_id]),
                    bound_string(bounds.left),
                    bound_string(bounds.right),
                    fmt(result.final_means[mode_id]),
                    fmt(result.final_variances[mode_id]),
                    fmt(cluster.persistence),
                    fmt(cluster.median_mass),
                    fmt(cluster.median_prominence),
                    fmt(cluster.median_separation),
                ])
            else
                append!(row, fill("", 10))
            end
        end
        push!(rows, row)
    end
    return rows
end

function save_component_results_mat(filepath::AbstractString, pooled, references, component_results,
                                    factor_min::Float64,
                                    factor_max::Float64,
                                    num_bandwidths::Int,
                                    grid_size::Int,
                                    min_mode_mass::Float64,
                                    persistence_threshold::Float64,
                                    min_prominence::Float64,
                                    merge_prominence_threshold::Float64,
                                    merge_separation_threshold::Float64)
    data = Dict{String, Any}()
    data["pooledY"] = pooled.y
    data["pooledPerms"] = pooled.perms
    data["bandwidthFactorMin"] = factor_min
    data["bandwidthFactorMax"] = factor_max
    data["numBandwidths"] = num_bandwidths
    data["gridSize"] = grid_size
    data["minModeMass"] = min_mode_mass
    data["persistenceThreshold"] = persistence_threshold
    data["minProminence"] = min_prominence
    data["mergeProminenceThreshold"] = merge_prominence_threshold
    data["mergeSeparationThreshold"] = merge_separation_threshold
    data["referenceNames"] = [String(ref.name) for ref in references]

    for result in component_results
        suffix = result.component_name
        data["grid_" * suffix] = result.kde_results[result.chosen_bandwidth_index].grid
        data["bandwidthFactors_" * suffix] = result.bandwidth_factors
        data["densityMatrix_" * suffix] = hcat([kde_result.density for kde_result in result.kde_results]...)
        data["bandwidths_" * suffix] = [kde_result.bandwidth for kde_result in result.kde_results]
        data["candidateCenters_" * suffix] = [cluster.center for cluster in result.clusters]
        data["candidateSupport_" * suffix] = [cluster.persistence for cluster in result.clusters]
        data["candidateProminence_" * suffix] = [cluster.median_prominence for cluster in result.clusters]
        data["candidateSeparation_" * suffix] = [cluster.median_separation for cluster in result.clusters]
        data["candidateMass_" * suffix] = [cluster.median_mass for cluster in result.clusters]
        data["candidateStable_" * suffix] = Float64.(result.stable_mask)
        data["chosenBandwidthFactor_" * suffix] = result.chosen_bandwidth_factor
        data["chosenBandwidth_" * suffix] = result.chosen_bandwidth
        data["finalCenters_" * suffix] = result.chosen_peak_x
        data["finalBoundaries_" * suffix] = result.final_boundaries
        data["finalWeights_" * suffix] = result.final_weights
        data["finalMeans_" * suffix] = result.final_means
        data["finalVariances_" * suffix] = result.final_variances
        data["referenceMasses_" * suffix] = result.reference_masses
        data["summary_" * suffix] = [result.global_mean,
                                      result.within_mode_spread,
                                      result.between_mode_separation,
                                      result.mode_entropy]
    end

    matwrite(filepath, data)
end

function write_rows_csv(filepath::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function bound_string(x)
    if isinf(x) && x < 0
        return "-Inf"
    elseif isinf(x) && x > 0
        return "Inf"
    else
        return fmt(x)
    end
end

fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
