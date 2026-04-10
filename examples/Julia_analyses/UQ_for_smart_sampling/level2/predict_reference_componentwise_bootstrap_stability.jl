"""
Step-6 componentwise bootstrap stability for pooled rigorous references.

This script quantifies finite-sample uncertainty in the componentwise
marginal Level 2 analysis, conditional on the three rigorous reference
ensembles already generated for a fixed window.
"""

module Step4Screening
include(joinpath(@__DIR__, "predict_reference_componentwise_mode_screening.jl"))
end

using CairoMakie
using Statistics
using Printf
using Random

CairoMakie.activate!()

const COMPONENT_NAMES = Step4Screening.COMPONENT_NAMES
const COMPONENT_LABELS = Step4Screening.COMPONENT_LABELS
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const BOOT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_bootstrap_stability")),
        "windows" => "",
        "num-bootstrap" => "300",
        "seed" => "1729",
        "curve-min-quantile" => "0.005",
        "curve-max-quantile" => "0.995",
        "num-curve-points" => "201",
        "factor-min" => "0.5",
        "factor-max" => "2.0",
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

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        requested_windows = requested_windows,
        num_bootstrap = parse(Int, options["num-bootstrap"]),
        seed = parse(Int, options["seed"]),
        curve_min_quantile = parse(Float64, options["curve-min-quantile"]),
        curve_max_quantile = parse(Float64, options["curve-max-quantile"]),
        num_curve_points = parse(Int, options["num-curve-points"]),
        factor_min = parse(Float64, options["factor-min"]),
        factor_max = parse(Float64, options["factor-max"]),
        num_bandwidths = parse(Int, options["num-bandwidths"]),
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
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_bootstrap_stability.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>                Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>              Folder where bootstrap outputs are saved")
    println("  --windows <names>                Comma-separated list like famp1,famp2")
    println("  --num-bootstrap <n>              Number of stratified bootstrap replicates (default: 300)")
    println("  --seed <n>                       Random seed (default: 1729)")
    println("  --curve-min-quantile <x>         Lower curve-reporting quantile (default: 0.005)")
    println("  --curve-max-quantile <x>         Upper curve-reporting quantile (default: 0.995)")
    println("  --num-curve-points <n>           Number of reporting-grid points (default: 201)")
    println("  --factor-min <x>                 Step-4 bandwidth ladder minimum factor (default: 0.5)")
    println("  --factor-max <x>                 Step-4 bandwidth ladder maximum factor (default: 2.0)")
    println("  --num-bandwidths <n>             Step-4 bandwidth ladder size (default: 5)")
    println("  --grid-size <n>                  Step-4 KDE grid size (default: 801)")
    println("  --min-mode-mass <x>              Step-4 minimum stable-mode mass (default: 0.01)")
    println("  --persistence-threshold <x>      Step-4 persistence threshold (default: 0.6)")
    println("  --min-prominence <x>             Step-4 minimum stable-mode prominence (default: 0.05)")
    println("  --merge-prominence-threshold <x> Step-4 merge prominence threshold (default: 0.1)")
    println("  --merge-separation-threshold <x> Step-4 merge separation threshold (default: 1.0)")
    println("  -h, --help                       Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = Step4Screening.collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No window reference folders found in $(opt.data_dir)")

    println("Processing $(length(windows)) window(s) from $(opt.data_dir)")
    for (window, reference_dir) in windows
        println("  - $window")
        process_window(window, reference_dir, opt)
    end
    println("Saved outputs to $(opt.output_dir)")
end

function process_window(window::AbstractString, reference_dir::AbstractString, opt)
    references = Step4Screening.load_references(reference_dir)
    isempty(references) && error("No reference MAT files found in $reference_dir")
    pooled = Step4Screening.pool_references(references)

    baseline_results = screen_components(pooled.y, references, opt)
    curve_grids = build_curve_grids(pooled.y, opt.curve_min_quantile, opt.curve_max_quantile, opt.num_curve_points)
    baseline_curve_data = build_curve_data(pooled.y, curve_grids)

    single_reference_results = build_single_reference_results(references, opt)
    leave_one_out_results = build_leave_one_out_results(references, opt)
    pairwise_rows = build_pairwise_reference_rows(window, references, curve_grids)
    single_reference_rows = build_single_reference_rows(window, references, single_reference_results)
    leave_one_out_rows = build_leave_one_out_rows(window, references, leave_one_out_results)

    bootstrap = run_bootstrap(window, references, baseline_results, curve_grids, baseline_curve_data, opt)

    window_dir = joinpath(opt.output_dir, String(window))
    mkpath(window_dir)

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_reference_reproducibility.csv"),
                   reference_reproducibility_header(),
                   pairwise_rows)
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_single_reference_mode_summary.csv"),
                   single_reference_header(),
                   single_reference_rows)
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_leave_one_out_mode_summary.csv"),
                   leave_one_out_header(),
                   leave_one_out_rows)

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_quantile_summary.csv"),
                   bootstrap_quantile_summary_header(),
                   build_bootstrap_quantile_summary_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_curve_pointwise.csv"),
                   bootstrap_curve_pointwise_header(),
                   build_bootstrap_curve_pointwise_rows(window, curve_grids, baseline_curve_data, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_curve_summary.csv"),
                   bootstrap_curve_summary_header(),
                   build_bootstrap_curve_summary_rows(window, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_mode_count_probabilities.csv"),
                   mode_count_probability_header(),
                   build_mode_count_probability_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_mode_existence_summary.csv"),
                   mode_existence_summary_header(),
                   build_mode_existence_summary_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_anchored_summary.csv"),
                   anchored_summary_header(),
                   build_anchored_summary_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_global_summary.csv"),
                   global_summary_header(),
                   build_global_summary_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_replicates.csv"),
                   replicate_summary_header(),
                   build_replicate_summary_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_mode_replicates.csv"),
                   mode_replicate_header(),
                   build_mode_replicate_rows(window, baseline_results, bootstrap))
    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_bootstrap_metadata.csv"),
                   bootstrap_metadata_header(),
                   build_bootstrap_metadata_rows(window, references, opt))

    save_bootstrap_curve_figure(window, baseline_results, curve_grids, baseline_curve_data, bootstrap,
        joinpath(window_dir, "$(window)_componentwise_bootstrap_curve_bands.png"))
    save_bootstrap_mode_figure(window, baseline_results, bootstrap,
        joinpath(window_dir, "$(window)_componentwise_bootstrap_mode_stability.png"))
    save_bootstrap_mode_count_figure(window, baseline_results, bootstrap,
        joinpath(window_dir, "$(window)_componentwise_bootstrap_mode_count_probabilities.png"))
    save_reference_sensitivity_figure(window, references, baseline_results, single_reference_results, leave_one_out_results,
        joinpath(window_dir, "$(window)_componentwise_reference_sensitivity.png"))
    save_bootstrap_quantile_figure(window, baseline_results, bootstrap,
        joinpath(window_dir, "$(window)_componentwise_bootstrap_quantile_stability.png"))
end

function screen_components(y::Matrix{Float64}, references, opt)
    results = NamedTuple[]
    for ic in eachindex(COMPONENT_NAMES)
        values = vec(y[:, ic])
        reference_values = [vec(ref.y[:, ic]) for ref in references]
        push!(results, Step4Screening.screen_component_modes(values, reference_values;
                                                             component_name = COMPONENT_NAMES[ic],
                                                             factor_min = opt.factor_min,
                                                             factor_max = opt.factor_max,
                                                             num_bandwidths = opt.num_bandwidths,
                                                             grid_size = opt.grid_size,
                                                             min_mode_mass = opt.min_mode_mass,
                                                             persistence_threshold = opt.persistence_threshold,
                                                             min_prominence = opt.min_prominence,
                                                             merge_prominence_threshold = opt.merge_prominence_threshold,
                                                             merge_separation_threshold = opt.merge_separation_threshold))
    end
    return results
end

function build_single_reference_results(references, opt)
    results = Dict{String, Vector{NamedTuple}}()
    for ref in references
        pseudo_refs = [(name = ref.name, perms = ref.perms, y = ref.y)]
        results[String(ref.name)] = screen_components(ref.y, pseudo_refs, opt)
    end
    return results
end

function build_leave_one_out_results(references, opt)
    results = Dict{String, Vector{NamedTuple}}()
    for ref in references
        kept = [other for other in references if other.name != ref.name]
        pooled = Step4Screening.pool_references(kept)
        results[String(ref.name)] = screen_components(pooled.y, kept, opt)
    end
    return results
end

function build_curve_grids(pooled_y::Matrix{Float64}, qmin::Float64, qmax::Float64, npoints::Int)
    grids = Vector{Vector{Float64}}(undef, size(pooled_y, 2))
    for ic in 1:size(pooled_y, 2)
        values = vec(pooled_y[:, ic])
        lo, hi = quantile(values, [qmin, qmax])
        grids[ic] = collect(range(lo, hi; length = npoints))
    end
    return grids
end

function ecdf_at_thresholds(values::Vector{Float64}, thresholds::Vector{Float64})
    n = length(values)
    sorted_values = sort(values)
    counts = [searchsortedlast(sorted_values, t) for t in thresholds]
    return counts ./ n
end

function exceedance_at_thresholds(values::Vector{Float64}, thresholds::Vector{Float64})
    n = length(values)
    sorted_values = sort(values)
    counts = [n - searchsortedlast(sorted_values, t) for t in thresholds]
    return counts ./ n
end

function build_curve_data(y::Matrix{Float64}, curve_grids)
    ecdf = Vector{Vector{Float64}}(undef, size(y, 2))
    exceed = Vector{Vector{Float64}}(undef, size(y, 2))
    for ic in 1:size(y, 2)
        values = vec(y[:, ic])
        thresholds = curve_grids[ic]
        ecdf[ic] = ecdf_at_thresholds(values, thresholds)
        exceed[ic] = exceedance_at_thresholds(values, thresholds)
    end
    return (ecdf = ecdf, exceed = exceed)
end

function bootstrap_references(references, rng::AbstractRNG)
    boot_refs = NamedTuple[]
    for ref in references
        n = size(ref.perms, 1)
        idx = rand(rng, 1:n, n)
        perms = ref.perms[idx, :]
        push!(boot_refs, (name = ref.name, perms = perms, y = log10.(perms), indices = idx))
    end
    return boot_refs
end

function run_bootstrap(window::AbstractString, references, baseline_results, curve_grids, baseline_curve_data, opt)
    B = opt.num_bootstrap
    rng = MersenneTwister(opt.seed + sum(codeunits(String(window))))

    ncomp = length(COMPONENT_NAMES)
    quantile_levels = collect(BOOT_QUANTILES)
    stores = [init_component_bootstrap_store(baseline_results[ic], curve_grids[ic], quantile_levels, B)
              for ic in 1:ncomp]

    for b in 1:B
        boot_refs = bootstrap_references(references, rng)
        boot_pooled = Step4Screening.pool_references(boot_refs)
        free_results = screen_components(boot_pooled.y, boot_refs, opt)

        for ic in 1:ncomp
            baseline = baseline_results[ic]
            store = stores[ic]
            values = vec(boot_pooled.y[:, ic])
            free = free_results[ic]
            thresholds = curve_grids[ic]

            qvals = quantile(values, quantile_levels)
            store.quantiles[:, b] .= qvals
            store.ecdf[:, b] .= ecdf_at_thresholds(values, thresholds)
            store.exceed[:, b] .= exceedance_at_thresholds(values, thresholds)
            store.supnorm_ecdf[b] = maximum(abs.(store.ecdf[:, b] .- store.baseline_ecdf))
            store.supnorm_exceed[b] = maximum(abs.(store.exceed[:, b] .- store.baseline_exceed))
            store.mode_count[b] = free.stable_mode_count
            store.chosen_bandwidth_factor[b] = free.chosen_bandwidth_factor

            match = match_bootstrap_modes(baseline, free, values)
            store.missing_count[b] = match.missing_count
            store.extra_count[b] = match.extra_count
            store.split_count[b] = match.split_count
            store.merge_count[b] = match.merge_count

            for mode_id in 1:baseline.stable_mode_count
                j = match.matched_bootstrap_mode[mode_id]
                store.mode_exists[mode_id, b] = j > 0
                if j > 0
                    store.free_center[mode_id, b] = free.chosen_peak_x[j]
                    store.free_pi[mode_id, b] = free.final_weights[j]
                end
            end

            anchored = anchored_statistics(values, baseline.final_intervals)
            store.anchored_pi[:, b] .= anchored.weights
            store.anchored_mean[:, b] .= anchored.means
            store.anchored_variance[:, b] .= anchored.variances
            store.anchored_W[b] = anchored.within_mode_spread
            store.anchored_B[b] = anchored.between_mode_separation
            store.anchored_H[b] = anchored.mode_entropy
        end
    end

    return stores
end

function init_component_bootstrap_store(baseline, thresholds, quantile_levels, B::Int)
    nmodes = baseline.stable_mode_count
    return (
        baseline_ecdf = ecdf_at_thresholds(baseline.values, thresholds),
        baseline_exceed = exceedance_at_thresholds(baseline.values, thresholds),
        thresholds = thresholds,
        quantile_levels = quantile_levels,
        ecdf = zeros(length(thresholds), B),
        exceed = zeros(length(thresholds), B),
        quantiles = zeros(length(quantile_levels), B),
        supnorm_ecdf = zeros(B),
        supnorm_exceed = zeros(B),
        mode_count = zeros(Int, B),
        chosen_bandwidth_factor = zeros(B),
        missing_count = zeros(Int, B),
        extra_count = zeros(Int, B),
        split_count = zeros(Int, B),
        merge_count = zeros(Int, B),
        mode_exists = falses(nmodes, B),
        free_center = fill(NaN, nmodes, B),
        free_pi = fill(NaN, nmodes, B),
        anchored_pi = fill(NaN, nmodes, B),
        anchored_mean = fill(NaN, nmodes, B),
        anchored_variance = fill(NaN, nmodes, B),
        anchored_W = zeros(B),
        anchored_B = zeros(B),
        anchored_H = zeros(B),
    )
end

function anchored_statistics(values::Vector{Float64}, baseline_intervals)
    weights = [Step4Screening.interval_mass(values, bounds.left, bounds.right) for bounds in baseline_intervals]
    means = [Step4Screening.interval_mean(values, bounds.left, bounds.right) for bounds in baseline_intervals]
    variances = [Step4Screening.interval_variance(values, bounds.left, bounds.right) for bounds in baseline_intervals]
    global_mean = mean(values)
    W = sum(weights .* variances)
    B = sum(weights .* (means .- global_mean).^2)
    H = -sum((w > 0 ? w * log(w) : 0.0) for w in weights)
    return (weights = weights, means = means, variances = variances,
            global_mean = global_mean,
            within_mode_spread = W,
            between_mode_separation = B,
            mode_entropy = H)
end

function clip_interval(bounds, support_min::Float64, support_max::Float64)
    left = isfinite(bounds.left) ? bounds.left : support_min
    right = isfinite(bounds.right) ? bounds.right : support_max
    return (left = left, right = right)
end

function interval_overlap_length(a, b, support_min::Float64, support_max::Float64)
    aa = clip_interval(a, support_min, support_max)
    bb = clip_interval(b, support_min, support_max)
    lo = max(aa.left, bb.left)
    hi = min(aa.right, bb.right)
    return max(0.0, hi - lo)
end

function match_bootstrap_modes(baseline, free, values::Vector{Float64})
    nb = baseline.stable_mode_count
    nf = free.stable_mode_count
    matched = fill(0, nb)
    if nb == 0 || nf == 0
        return (matched_bootstrap_mode = matched,
                missing_count = nb,
                extra_count = nf,
                split_count = 0,
                merge_count = 0)
    end

    support_min = min(minimum(baseline.values), minimum(values))
    support_max = max(maximum(baseline.values), maximum(values))
    overlap = zeros(nb, nf)
    for i in 1:nb, j in 1:nf
        overlap[i, j] = interval_overlap_length(baseline.final_intervals[i], free.final_intervals[j],
                                                support_min, support_max)
    end

    split_count = count(i -> count(overlap[i, :] .> 0) > 1, 1:nb)
    merge_count = count(j -> count(overlap[:, j] .> 0) > 1, 1:nf)

    used = falses(nf)
    used_baseline = falses(nb)
    candidates = NamedTuple[]
    for i in 1:nb, j in 1:nf
        if overlap[i, j] > 0
            push!(candidates, (
                baseline_id = i,
                bootstrap_id = j,
                center_distance = abs(free.chosen_peak_x[j] - baseline.chosen_peak_x[i]),
                overlap = overlap[i, j],
            ))
        end
    end
    sort!(candidates, by = c -> (c.center_distance, -c.overlap))
    for cand in candidates
        i = cand.baseline_id
        j = cand.bootstrap_id
        if !used_baseline[i] && !used[j]
            matched[i] = j
            used_baseline[i] = true
            used[j] = true
        end
    end

    for i in 1:nb
        matched[i] > 0 && continue
        candidates = [j for j in 1:nf if !used[j]]
        isempty(candidates) && continue
        dists = [abs(free.chosen_peak_x[j] - baseline.chosen_peak_x[i]) for j in candidates]
        best_idx = argmin(dists)
        if dists[best_idx] <= baseline.center_tolerance
            best_j = candidates[best_idx]
            matched[i] = best_j
            used[best_j] = true
        end
    end

    return (
        matched_bootstrap_mode = matched,
        missing_count = count(==(0), matched),
        extra_count = count(!, used),
        split_count = split_count,
        merge_count = merge_count,
    )
end

function pairwise_reference_pairs(references)
    pairs = Tuple{Int, Int}[]
    for i in 1:length(references)-1, j in i+1:length(references)
        push!(pairs, (i, j))
    end
    return pairs
end

function build_pairwise_reference_rows(window::AbstractString, references, curve_grids)
    rows = Vector{Vector{String}}()
    pairs = pairwise_reference_pairs(references)
    qlevels = [0.05, 0.50, 0.95]
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        thresholds = curve_grids[ic]
        for (i, j) in pairs
            values_i = vec(references[i].y[:, ic])
            values_j = vec(references[j].y[:, ic])
            ecdf_i = ecdf_at_thresholds(values_i, thresholds)
            ecdf_j = ecdf_at_thresholds(values_j, thresholds)
            ex_i = exceedance_at_thresholds(values_i, thresholds)
            ex_j = exceedance_at_thresholds(values_j, thresholds)
            qi = quantile(values_i, qlevels)
            qj = quantile(values_j, qlevels)
            push!(rows, [
                String(window),
                String(comp_name),
                String(references[i].name),
                String(references[j].name),
                fmt(maximum(abs.(ecdf_i .- ecdf_j))),
                fmt(maximum(abs.(ex_i .- ex_j))),
                fmt(abs(qi[1] - qj[1])),
                fmt(abs(qi[2] - qj[2])),
                fmt(abs(qi[3] - qj[3])),
            ])
        end
    end
    return rows
end

function build_single_reference_rows(window::AbstractString, references, single_reference_results)
    rows = Vector{Vector{String}}()
    for ref in references
        results = single_reference_results[String(ref.name)]
        for result in results
            push!(rows, [
                String(window),
                result.component_name,
                String(ref.name),
                string(result.stable_mode_count),
                fmt(result.chosen_bandwidth_factor),
                join([fmt(x) for x in result.chosen_peak_x], ";"),
                join([fmt(x) for x in result.final_weights], ";"),
                fmt(result.within_mode_spread),
                fmt(result.between_mode_separation),
                fmt(result.mode_entropy),
            ])
        end
    end
    return rows
end

function build_leave_one_out_rows(window::AbstractString, references, leave_one_out_results)
    rows = Vector{Vector{String}}()
    for ref in references
        results = leave_one_out_results[String(ref.name)]
        for result in results
            push!(rows, [
                String(window),
                result.component_name,
                String(ref.name),
                string(result.stable_mode_count),
                fmt(result.chosen_bandwidth_factor),
                join([fmt(x) for x in result.chosen_peak_x], ";"),
                join([fmt(x) for x in result.final_weights], ";"),
                fmt(result.within_mode_spread),
                fmt(result.between_mode_separation),
                fmt(result.mode_entropy),
            ])
        end
    end
    return rows
end

function reference_reproducibility_header()
    return ["window", "component", "reference_a", "reference_b",
            "ecdf_supnorm", "exceedance_supnorm", "abs_q05_diff", "abs_q50_diff", "abs_q95_diff"]
end

function single_reference_header()
    return ["window", "component", "reference_name", "stable_mode_count", "chosen_bandwidth_factor",
            "mode_centers_log10k", "mode_weights_pi", "within_mode_spread_W", "between_mode_separation_B", "mode_entropy_H"]
end

function leave_one_out_header()
    return ["window", "component", "dropped_reference", "stable_mode_count", "chosen_bandwidth_factor",
            "mode_centers_log10k", "mode_weights_pi", "within_mode_spread_W", "between_mode_separation_B", "mode_entropy_H"]
end

function bootstrap_quantile_summary_header()
    return ["window", "component", "quantile_name", "quantile_level", "baseline_value",
            "bootstrap_median", "bootstrap_p10", "bootstrap_p90"]
end

function build_bootstrap_quantile_summary_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    qnames = ("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99")
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline_q = quantile(baseline_results[ic].values, collect(BOOT_QUANTILES))
        for iq in eachindex(BOOT_QUANTILES)
            vals = vec(stores[ic].quantiles[iq, :])
            push!(rows, [
                String(window),
                String(comp_name),
                String(qnames[iq]),
                fmt(BOOT_QUANTILES[iq]),
                fmt(baseline_q[iq]),
                fmt(median(vals)),
                fmt(quantile(vals, 0.10)),
                fmt(quantile(vals, 0.90)),
            ])
        end
    end
    return rows
end

function bootstrap_curve_pointwise_header()
    return ["window", "component", "curve_type", "threshold_log10k", "baseline_value",
            "bootstrap_median", "bootstrap_p10", "bootstrap_p90"]
end

function build_bootstrap_curve_pointwise_rows(window::AbstractString, curve_grids, baseline_curve_data, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        thresholds = curve_grids[ic]
        for (curve_type, baseline_vals, mat) in (("ecdf", baseline_curve_data.ecdf[ic], stores[ic].ecdf),
                                                 ("exceedance", baseline_curve_data.exceed[ic], stores[ic].exceed))
            for it in eachindex(thresholds)
                vals = vec(mat[it, :])
                push!(rows, [
                    String(window),
                    String(comp_name),
                    curve_type,
                    fmt(thresholds[it]),
                    fmt(baseline_vals[it]),
                    fmt(median(vals)),
                    fmt(quantile(vals, 0.10)),
                    fmt(quantile(vals, 0.90)),
                ])
            end
        end
    end
    return rows
end

function bootstrap_curve_summary_header()
    return ["window", "component", "curve_type", "supnorm_median", "supnorm_p10", "supnorm_p90", "simultaneous_band_q95"]
end

function build_bootstrap_curve_summary_rows(window::AbstractString, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        for (curve_type, vals) in (("ecdf", stores[ic].supnorm_ecdf), ("exceedance", stores[ic].supnorm_exceed))
            push!(rows, [
                String(window),
                String(comp_name),
                curve_type,
                fmt(median(vals)),
                fmt(quantile(vals, 0.10)),
                fmt(quantile(vals, 0.90)),
                fmt(quantile(vals, 0.95)),
            ])
        end
    end
    return rows
end

function mode_count_probability_header()
    return ["window", "component", "mode_count", "probability"]
end

function build_mode_count_probability_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        counts = stores[ic].mode_count
        for k in sort(unique(counts))
            push!(rows, [
                String(window),
                String(comp_name),
                string(k),
                fmt(mean(counts .== k)),
            ])
        end
    end
    return rows
end

function mode_existence_summary_header()
    return ["window", "component", "baseline_mode_id", "baseline_center_log10k", "baseline_pi",
            "existence_probability", "free_center_median", "free_center_p10", "free_center_p90",
            "free_pi_median", "free_pi_p10", "free_pi_p90"]
end

function build_mode_existence_summary_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline = baseline_results[ic]
        store = stores[ic]
        for mode_id in 1:baseline.stable_mode_count
            exists = vec(store.mode_exists[mode_id, :])
            centers = finite_values(vec(store.free_center[mode_id, :]))
            pis = finite_values(vec(store.free_pi[mode_id, :]))
            push!(rows, [
                String(window),
                String(comp_name),
                string(mode_id),
                fmt(baseline.chosen_peak_x[mode_id]),
                fmt(baseline.final_weights[mode_id]),
                fmt(mean(exists)),
                summary_or_blank(centers, :median),
                summary_or_blank(centers, :p10),
                summary_or_blank(centers, :p90),
                summary_or_blank(pis, :median),
                summary_or_blank(pis, :p10),
                summary_or_blank(pis, :p90),
            ])
        end
    end
    return rows
end

function anchored_summary_header()
    return ["window", "component", "baseline_mode_id", "baseline_center_log10k", "baseline_pi",
            "anchored_pi_median", "anchored_pi_p10", "anchored_pi_p90",
            "anchored_mean_median", "anchored_mean_p10", "anchored_mean_p90",
            "anchored_variance_median", "anchored_variance_p10", "anchored_variance_p90"]
end

function build_anchored_summary_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline = baseline_results[ic]
        store = stores[ic]
        for mode_id in 1:baseline.stable_mode_count
            pis = vec(store.anchored_pi[mode_id, :])
            means = vec(store.anchored_mean[mode_id, :])
            vars = vec(store.anchored_variance[mode_id, :])
            push!(rows, [
                String(window),
                String(comp_name),
                string(mode_id),
                fmt(baseline.chosen_peak_x[mode_id]),
                fmt(baseline.final_weights[mode_id]),
                fmt(median(pis)),
                fmt(quantile(pis, 0.10)),
                fmt(quantile(pis, 0.90)),
                fmt(median(means)),
                fmt(quantile(means, 0.10)),
                fmt(quantile(means, 0.90)),
                fmt(median(vars)),
                fmt(quantile(vars, 0.10)),
                fmt(quantile(vars, 0.90)),
            ])
        end
    end
    return rows
end

function global_summary_header()
    return ["window", "component", "baseline_mode_count", "baseline_chosen_bandwidth_factor",
            "chosen_bandwidth_median", "chosen_bandwidth_p10", "chosen_bandwidth_p90",
            "anchored_W_baseline", "anchored_W_median", "anchored_W_p10", "anchored_W_p90",
            "anchored_B_baseline", "anchored_B_median", "anchored_B_p10", "anchored_B_p90",
            "anchored_H_baseline", "anchored_H_median", "anchored_H_p10", "anchored_H_p90",
            "missing_modes_median", "extra_modes_median", "split_events_median", "merge_events_median"]
end

function build_global_summary_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline = baseline_results[ic]
        store = stores[ic]
        push!(rows, [
            String(window),
            String(comp_name),
            string(baseline.stable_mode_count),
            fmt(baseline.chosen_bandwidth_factor),
            fmt(median(store.chosen_bandwidth_factor)),
            fmt(quantile(store.chosen_bandwidth_factor, 0.10)),
            fmt(quantile(store.chosen_bandwidth_factor, 0.90)),
            fmt(baseline.within_mode_spread),
            fmt(median(store.anchored_W)),
            fmt(quantile(store.anchored_W, 0.10)),
            fmt(quantile(store.anchored_W, 0.90)),
            fmt(baseline.between_mode_separation),
            fmt(median(store.anchored_B)),
            fmt(quantile(store.anchored_B, 0.10)),
            fmt(quantile(store.anchored_B, 0.90)),
            fmt(baseline.mode_entropy),
            fmt(median(store.anchored_H)),
            fmt(quantile(store.anchored_H, 0.10)),
            fmt(quantile(store.anchored_H, 0.90)),
            fmt(median(store.missing_count)),
            fmt(median(store.extra_count)),
            fmt(median(store.split_count)),
            fmt(median(store.merge_count)),
        ])
    end
    return rows
end

function replicate_summary_header()
    header = ["window", "bootstrap_id", "component", "mode_count", "chosen_bandwidth_factor",
              "missing_baseline_modes", "extra_modes", "split_events", "merge_events",
              "ecdf_supnorm", "exceedance_supnorm", "anchored_W", "anchored_B", "anchored_H"]
    append!(header, ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"])
    return header
end

function build_replicate_summary_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        store = stores[ic]
        B = size(store.quantiles, 2)
        for b in 1:B
            row = [
                String(window),
                string(b),
                String(comp_name),
                string(store.mode_count[b]),
                fmt(store.chosen_bandwidth_factor[b]),
                string(store.missing_count[b]),
                string(store.extra_count[b]),
                string(store.split_count[b]),
                string(store.merge_count[b]),
                fmt(store.supnorm_ecdf[b]),
                fmt(store.supnorm_exceed[b]),
                fmt(store.anchored_W[b]),
                fmt(store.anchored_B[b]),
                fmt(store.anchored_H[b]),
            ]
            append!(row, [fmt(store.quantiles[iq, b]) for iq in 1:size(store.quantiles, 1)])
            push!(rows, row)
        end
    end
    return rows
end

function mode_replicate_header()
    return ["window", "bootstrap_id", "component", "baseline_mode_id", "exists_in_free_refit",
            "free_center_log10k", "free_pi", "anchored_pi", "anchored_mean", "anchored_variance"]
end

function build_mode_replicate_rows(window::AbstractString, baseline_results, stores)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline = baseline_results[ic]
        store = stores[ic]
        B = size(store.anchored_pi, 2)
        for mode_id in 1:baseline.stable_mode_count
            for b in 1:B
                push!(rows, [
                    String(window),
                    string(b),
                    String(comp_name),
                    string(mode_id),
                    string(store.mode_exists[mode_id, b]),
                    nan_or_blank(store.free_center[mode_id, b]),
                    nan_or_blank(store.free_pi[mode_id, b]),
                    fmt(store.anchored_pi[mode_id, b]),
                    fmt(store.anchored_mean[mode_id, b]),
                    fmt(store.anchored_variance[mode_id, b]),
                ])
            end
        end
    end
    return rows
end

function bootstrap_metadata_header()
    return ["window", "num_references", "samples_per_reference", "num_bootstrap", "seed",
            "curve_min_quantile", "curve_max_quantile", "num_curve_points",
            "factor_min", "factor_max", "num_bandwidths", "grid_size",
            "min_mode_mass", "persistence_threshold", "min_prominence",
            "merge_prominence_threshold", "merge_separation_threshold"]
end

function build_bootstrap_metadata_rows(window::AbstractString, references, opt)
    return [[String(window),
             string(length(references)),
             join([string(size(ref.y, 1)) for ref in references], ";"),
             string(opt.num_bootstrap),
             string(opt.seed),
             fmt(opt.curve_min_quantile),
             fmt(opt.curve_max_quantile),
             string(opt.num_curve_points),
             fmt(opt.factor_min),
             fmt(opt.factor_max),
             string(opt.num_bandwidths),
             string(opt.grid_size),
             fmt(opt.min_mode_mass),
             fmt(opt.persistence_threshold),
             fmt(opt.min_prominence),
             fmt(opt.merge_prominence_threshold),
             fmt(opt.merge_separation_threshold)]]
end

function finite_values(values::Vector{Float64})
    return [x for x in values if isfinite(x)]
end

function summary_or_blank(values::Vector{Float64}, which::Symbol)
    isempty(values) && return ""
    if which == :median
        return fmt(median(values))
    elseif which == :p10
        return fmt(quantile(values, 0.10))
    elseif which == :p90
        return fmt(quantile(values, 0.90))
    else
        error("Unknown summary selector: $which")
    end
end

function nan_or_blank(x::Float64)
    return isfinite(x) ? fmt(x) : ""
end

function save_bootstrap_curve_figure(window::AbstractString, baseline_results, curve_grids, baseline_curve_data, stores, output_path::AbstractString)
    fig = Figure(size = (1850, 980), fontsize = 18, backgroundcolor = :white)
    band_color = RGBAf(0.16, 0.45, 0.80, 0.18)
    median_color = RGBf(0.12, 0.35, 0.70)
    baseline_color = RGBf(0.05, 0.05, 0.05)

    for (ic, label) in enumerate(COMPONENT_LABELS)
        thresholds = curve_grids[ic]
        ax = Axis(fig[1, ic], xlabel = label, ylabel = (ic == 1 ? "Empirical CDF" : ""),
                  title = "$(COMPONENT_NAMES[ic]): bootstrap ECDF",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xgridcolor = RGBAf(0, 0, 0, 0.08), ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        lower = mapslices(x -> quantile(x, 0.10), stores[ic].ecdf; dims = 2)[:]
        upper = mapslices(x -> quantile(x, 0.90), stores[ic].ecdf; dims = 2)[:]
        med = mapslices(median, stores[ic].ecdf; dims = 2)[:]
        band!(ax, thresholds, lower, upper, color = band_color)
        lines!(ax, thresholds, med, color = median_color, linewidth = 2.6, label = "bootstrap median")
        lines!(ax, thresholds, baseline_curve_data.ecdf[ic], color = baseline_color, linewidth = 2.0, linestyle = :dash, label = "baseline")
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ic == 1
            axislegend(ax, position = :rb, framevisible = true, labelsize = 22,
                       backgroundcolor = RGBAf(1, 1, 1, 0.92), patchlabelgap = 8, rowgap = 5)
        end
    end

    for (ic, label) in enumerate(COMPONENT_LABELS)
        thresholds = curve_grids[ic]
        ax = Axis(fig[2, ic], xlabel = label, ylabel = (ic == 1 ? "Exceedance probability" : ""),
                  title = "$(COMPONENT_NAMES[ic]): bootstrap exceedance", yscale = log10,
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xgridcolor = RGBAf(0, 0, 0, 0.08), ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        lower = mapslices(x -> quantile(x, 0.10), stores[ic].exceed; dims = 2)[:]
        upper = mapslices(x -> quantile(x, 0.90), stores[ic].exceed; dims = 2)[:]
        med = mapslices(median, stores[ic].exceed; dims = 2)[:]
        band!(ax, thresholds, max.(lower, 1e-8), max.(upper, 1e-8), color = band_color)
        lines!(ax, thresholds, max.(med, 1e-8), color = median_color, linewidth = 2.6, label = "bootstrap median")
        lines!(ax, thresholds, max.(baseline_curve_data.exceed[ic], 1e-8), color = baseline_color, linewidth = 2.0, linestyle = :dash, label = "baseline")
        ylims!(ax, 1e-5, 1.0)
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('d' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ic == 1
            axislegend(ax, position = :lb, framevisible = true, labelsize = 22,
                       backgroundcolor = RGBAf(1, 1, 1, 0.92), patchlabelgap = 8, rowgap = 5)
        end
    end

    Label(fig[0, :], "$window componentwise bootstrap curve stability", fontsize = 24, font = :bold)
    Label(fig[3, :], "Blue band: bootstrap P10-P90. Blue line: bootstrap median. Dashed black line: baseline pooled estimator.", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function save_bootstrap_mode_figure(window::AbstractString, baseline_results, stores, output_path::AbstractString)
    fig = Figure(size = (1850, 980), fontsize = 18, backgroundcolor = :white)
    point_color = RGBf(0.10, 0.33, 0.65)
    range_color = RGBf(0.20, 0.55, 0.85)

    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline = baseline_results[ic]
        store = stores[ic]
        x = collect(1:baseline.stable_mode_count)
        ax = Axis(fig[1, ic], xlabel = "Baseline mode", ylabel = (ic == 1 ? "Existence probability" : ""),
                  title = "$(comp_name): structural stability",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xticks = (x, ["mode $i" for i in x]),
                  xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        probs = [mean(store.mode_exists[i, :]) for i in x]
        barplot!(ax, x, probs, color = (point_color, 0.80), strokecolor = :black, strokewidth = 0.6)
        ylims!(ax, 0.0, 1.05)
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
    end

    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline = baseline_results[ic]
        store = stores[ic]
        x = collect(1:baseline.stable_mode_count)
        ax = Axis(fig[2, ic], xlabel = "Baseline mode", ylabel = (ic == 1 ? "Anchored π" : ""),
                  title = "$(comp_name): anchored mode weights",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xticks = (x, ["mode $i" for i in x]),
                  xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        med = [median(store.anchored_pi[i, :]) for i in x]
        low = [quantile(store.anchored_pi[i, :], 0.10) for i in x]
        high = [quantile(store.anchored_pi[i, :], 0.90) for i in x]
        baseline_pi = baseline.final_weights
        rangebars!(ax, x, low, high, color = range_color, linewidth = 4)
        scatter!(ax, x, med, color = point_color, markersize = 13, label = "bootstrap median")
        scatter!(ax, x, baseline_pi, color = :black, marker = :rect, markersize = 11, label = "baseline")
        ylims!(ax, 0.0, min(1.05, maximum(vcat(high, baseline_pi)) * 1.15))
        xlims!(ax, 0.75, baseline.stable_mode_count + 0.25)
        text!(ax, 0.05, 0.98; space = :relative, text = string("(", Char('d' + ic - 1), ")"),
              align = (:left, :top), fontsize = 22, font = :bold, color = :black)
        if ic == 1
            axislegend(ax, position = :lt, framevisible = true, labelsize = 22,
                       backgroundcolor = RGBAf(1, 1, 1, 0.92), patchlabelgap = 8, rowgap = 5)
        end
    end

    Label(fig[0, :], "$window componentwise bootstrap mode stability", fontsize = 24, font = :bold)
    Label(fig[3, :], "Top row: free-refit baseline-mode existence probabilities. Bottom row: anchored π intervals (P10-P90) with bootstrap median (blue circle) and baseline value (black square).", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function save_bootstrap_mode_count_figure(window::AbstractString, baseline_results, stores, output_path::AbstractString)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    bar_color = RGBAf(0.42, 0.79, 0.75, 0.92)

    all_counts = sort(unique(vcat([collect(store.mode_count) for store in stores]...)))
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline_count = baseline_results[ic].stable_mode_count
        store = stores[ic]
        probs = [mean(store.mode_count .== count_val) for count_val in all_counts]
        x = collect(1:length(all_counts))

        ax = Axis(fig[1, ic], xlabel = "Detected mode count", ylabel = (ic == 1 ? "Bootstrap probability" : ""),
                  title = "$(comp_name): mode-count probabilities",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xticks = (x, [string(k) for k in all_counts]),
                  xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        barplot!(ax, x, probs, color = bar_color, strokecolor = :black, strokewidth = 0.7)
        ylims!(ax, 0.0, 1.05)
        for (xi, prob) in zip(x, probs)
            text!(ax, xi, min(1.02, prob + 0.035), text = @sprintf("%.3f", prob),
                  align = (:center, :bottom), fontsize = 22, color = :black)
        end
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        text!(ax, 0.04, 0.96; space = :relative, text = "baseline = $baseline_count",
              align = (:left, :top), fontsize = 22, color = :black)
    end

    Label(fig[0, :], "$window componentwise bootstrap mode-count ambiguity", fontsize = 24, font = :bold)
    Label(fig[2, :], "Blue bar marks the baseline pooled mode count. Heights are empirical bootstrap probabilities over all refits.", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function save_reference_sensitivity_figure(window::AbstractString, references, baseline_results, single_reference_results, leave_one_out_results, output_path::AbstractString)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    pooled_color = :black
    single_color = RGBf(0.10, 0.33, 0.65)
    loo_color = RGBf(0.88, 0.47, 0.13)
    caption_color = RGBf(0.35, 0.35, 0.35)

    ref_names = [String(ref.name) for ref in references]
    category_labels = vcat(["pooled"], ref_names, ["-" * name for name in ref_names])
    x = collect(1:length(category_labels))
    legend_handles = Any[]
    legend_labels = String[]

    max_modes = maximum(vcat([baseline.stable_mode_count for baseline in baseline_results],
                             [result[ic].stable_mode_count for result in values(single_reference_results) for ic in eachindex(COMPONENT_NAMES)],
                             [result[ic].stable_mode_count for result in values(leave_one_out_results) for ic in eachindex(COMPONENT_NAMES)]))

    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        ax = Axis(fig[1, ic], xlabel = "Reference subset", ylabel = (ic == 1 ? "Stable mode count" : ""),
                  title = "$(comp_name): reference sensitivity",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xticks = (x, category_labels), yticks = (collect(1:max_modes), [string(i) for i in 1:max_modes]), xticklabelrotation = 0.0,
                  xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        yvals = Float64[
            baseline_results[ic].stable_mode_count;
            [single_reference_results[name][ic].stable_mode_count for name in ref_names]...;
            [leave_one_out_results[name][ic].stable_mode_count for name in ref_names]...
        ]
        ylims!(ax, 0.45, max_modes + 0.65)
        pooled_plot = scatter!(ax, [x[1]], [yvals[1]], color = pooled_color, marker = :rect, markersize = 20)
        single_plot = scatter!(ax, x[2:1 + length(ref_names)], yvals[2:1 + length(ref_names)], color = single_color, marker = :circle, markersize = 18)
        loo_plot = scatter!(ax, x[(2 + length(ref_names)):end], yvals[(2 + length(ref_names)):end], color = loo_color, marker = :diamond, markersize = 20)
        if ic == 1
            legend_handles = [pooled_plot, single_plot, loo_plot]
            legend_labels = ["pooled baseline", "single reference", "leave-one-out pooled"]
        end
        for (xi, yi) in zip(x, yvals)
            text!(ax, xi, yi + 0.14, text = string(Int(round(yi))), align = (:center, :bottom), fontsize = 22, color = :black)
        end
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
    end

    Legend(fig[1, 1], legend_handles, legend_labels,
           orientation = :vertical, framevisible = true, labelsize = 22,
           tellheight = false, tellwidth = false, halign = :left, valign = :bottom,
           margin = (18, 18, 18, 18), padding = (10, 10, 10, 10))
    Label(fig[0, :], "$window componentwise reference sensitivity", fontsize = 24, font = :bold)
    Label(fig[2, :], "Black square: pooled baseline. Blue circles: each single reference. Orange diamonds: leave-one-out pooled fits.", fontsize = 22, color = caption_color)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function save_bootstrap_quantile_figure(window::AbstractString, baseline_results, stores, output_path::AbstractString)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    point_color = RGBf(0.10, 0.33, 0.65)
    range_color = RGBf(0.20, 0.55, 0.85)
    qlabels = ["q05", "q10", "q50", "q90", "q95"]
    qidx = [2, 3, 5, 7, 8]
    qlevels = collect(BOOT_QUANTILES)[qidx]

    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        baseline_q = quantile(baseline_results[ic].values, qlevels)
        store = stores[ic]
        med = [median(vec(store.quantiles[idx, :])) for idx in qidx]
        low = [quantile(vec(store.quantiles[idx, :]), 0.10) for idx in qidx]
        high = [quantile(vec(store.quantiles[idx, :]), 0.90) for idx in qidx]
        x = collect(1:length(qidx))

        ax = Axis(fig[1, ic], xlabel = "Quantile", ylabel = (ic == 1 ? "log10 permeability" : ""),
                  title = "$(comp_name): quantile stability",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xticks = (x, qlabels),
                  xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        rangebars!(ax, x, low, high, color = range_color, linewidth = 4)
        scatter!(ax, x, med, color = point_color, markersize = 13, label = "bootstrap median")
        scatter!(ax, x, baseline_q, color = :black, marker = :rect, markersize = 11, label = "baseline")
        xlims!(ax, 0.6, length(qidx) + 0.4)
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ic == 1
            axislegend(ax, position = :lt, framevisible = true, labelsize = 22,
                       backgroundcolor = RGBAf(1, 1, 1, 0.92), patchlabelgap = 8, rowgap = 5)
        end
    end

    Label(fig[0, :], "$window componentwise bootstrap quantile stability", fontsize = 24, font = :bold)
    Label(fig[2, :], "Blue ranges: bootstrap P10-P90. Blue circles: bootstrap median. Black squares: baseline pooled quantiles.", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function write_rows_csv(filepath::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
