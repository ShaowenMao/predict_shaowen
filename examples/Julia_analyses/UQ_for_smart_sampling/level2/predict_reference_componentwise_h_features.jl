#!/usr/bin/env julia

"""
Construct componentwise h-features for one fixed geologic input.

This script turns the completed componentwise Level 2 analysis into a concrete
set of scalar feature functions h_k for one fixed geology/window. The target is
the later Level 1 vs Level 2 contribution analysis, where h_k must be scalar
functions of one single PREDICT realization, not diagnostics of an estimated PDF.

For each selected window it:
1. Loads the pooled rigorous references R1/R2/R3.
2. Re-runs the accepted step-4 componentwise mode screening in log10(k) space.
3. Defines a threshold/CDF feature family for each component:
       h(Y_j) = 1{Y_j <= c}
   using fixed pooled quantiles and robust antimodes.
4. Defines a robust mode family for each component:
       h_occ(Y_j) = 1{Y_j in R_m}
       h_1(Y_j)   = Y_j 1{Y_j in R_m}
       h_2(Y_j)   = Y_j^2 1{Y_j in R_m}
   keeping only modes that pass a bootstrap robustness screen when bootstrap
   results are available.
5. Saves:
       - feature definitions
       - pooled feature values for every realization
       - pooled feature means and variances

This script uses the existing step-4 mode-screening logic only to define fixed
thresholds and robust mode intervals. The returned h-values themselves are
functions of single draws.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie", "KernelDensity"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_h_features.jl --windows famp1
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_h_features.jl --windows famp1 --bootstrap-dir D:\\codex_gom\\reference_componentwise_bootstrap_test_v4000_figs
"""

module Step4Screening
include(joinpath(@__DIR__, "predict_reference_componentwise_mode_screening.jl"))
end

using Statistics
using Printf

const COMPONENT_NAMES = Step4Screening.COMPONENT_NAMES
const COMPONENT_LABELS = Step4Screening.COMPONENT_LABELS
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_h_features")),
        "bootstrap-dir" => "",
        "windows" => "",
        "quantile-levels" => join(string.(DEFAULT_QUANTILES), ","),
        "robust-mode-existence-min" => "0.8",
        "robust-mode-mass-min" => "0.05",
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
    quantile_levels = Tuple(parse.(Float64, split(options["quantile-levels"], ",")))

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        bootstrap_dir = strip(options["bootstrap-dir"]),
        requested_windows = requested_windows,
        quantile_levels = quantile_levels,
        robust_mode_existence_min = parse(Float64, options["robust-mode-existence-min"]),
        robust_mode_mass_min = parse(Float64, options["robust-mode-mass-min"]),
        factor_min = parse(Float64, options["bandwidth-factor-min"]),
        factor_max = parse(Float64, options["bandwidth-factor-max"]),
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
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_h_features.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>                   Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>                 Folder where feature definitions and values are saved")
    println("  --bootstrap-dir <path>              Optional folder with bootstrap outputs for robust-mode filtering")
    println("  --windows <names>                   Comma-separated list like famp1,famp2")
    println("  --quantile-levels <list>            Comma-separated pooled quantile levels (default: 0.05,0.10,0.25,0.50,0.75,0.90,0.95)")
    println("  --robust-mode-existence-min <x>     Bootstrap existence cutoff for mode-based h (default: 0.8)")
    println("  --robust-mode-mass-min <x>          Minimum baseline mode weight for mode-based h (default: 0.05)")
    println("  --bandwidth-factor-min <x>          Step-4 bandwidth ladder minimum factor (default: 0.5)")
    println("  --bandwidth-factor-max <x>          Step-4 bandwidth ladder maximum factor (default: 2.0)")
    println("  --num-bandwidths <n>                Step-4 bandwidth ladder size (default: 5)")
    println("  --grid-size <n>                     Step-4 KDE grid size (default: 801)")
    println("  --min-mode-mass <x>                 Step-4 minimum stable-mode mass (default: 0.01)")
    println("  --persistence-threshold <x>         Step-4 persistence threshold (default: 0.6)")
    println("  --min-prominence <x>                Step-4 minimum stable-mode prominence (default: 0.05)")
    println("  --merge-prominence-threshold <x>    Step-4 merge prominence threshold (default: 0.1)")
    println("  --merge-separation-threshold <x>    Step-4 merge separation threshold (default: 1.0)")
    println("  -h, --help                          Show this help")
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

    component_results = screen_components(pooled.y, references, opt)
    bootstrap_modes = maybe_load_bootstrap_modes(window, opt.bootstrap_dir)

    feature_defs = NamedTuple[]
    feature_columns = Vector{Vector{Float64}}()

    for (ic, result) in enumerate(component_results)
        y = vec(pooled.y[:, ic])
        robust_mode_ids = select_robust_modes(result, bootstrap_modes;
                                              existence_min = opt.robust_mode_existence_min,
                                              mass_min = opt.robust_mode_mass_min)
        thresholds = build_thresholds(y, result, robust_mode_ids, opt.quantile_levels)
        append_feature_family!(feature_defs, feature_columns, window, result.component_name, y, thresholds, robust_mode_ids, result, bootstrap_modes)
    end

    sample_info = build_sample_info(references)
    window_dir = joinpath(opt.output_dir, String(window))
    mkpath(window_dir)

    write_feature_definitions(joinpath(window_dir, "$(window)_componentwise_h_definitions.csv"), feature_defs)
    write_feature_summary(joinpath(window_dir, "$(window)_componentwise_h_summary.csv"), feature_defs, feature_columns)
    write_feature_values(joinpath(window_dir, "$(window)_componentwise_h_values.csv"), pooled.y, sample_info, feature_defs, feature_columns)
    write_feature_metadata(joinpath(window_dir, "$(window)_componentwise_h_metadata.csv"), window, opt, bootstrap_modes)
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

function maybe_load_bootstrap_modes(window::AbstractString, bootstrap_dir::AbstractString)
    isempty(bootstrap_dir) && return Dict{Tuple{String, Int}, NamedTuple}()
    filepath = joinpath(bootstrap_dir, String(window), "$(window)_componentwise_bootstrap_mode_existence_summary.csv")
    isfile(filepath) || return Dict{Tuple{String, Int}, NamedTuple}()

    lines = readlines(filepath)
    isempty(lines) && return Dict{Tuple{String, Int}, NamedTuple}()
    header = split(lines[1], ",")
    idx = Dict(header[i] => i for i in eachindex(header))

    out = Dict{Tuple{String, Int}, NamedTuple}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cols = split(line, ",")
        component = cols[idx["component"]]
        mode_id = parse(Int, cols[idx["baseline_mode_id"]])
        existence_probability = parse(Float64, cols[idx["existence_probability"]])
        baseline_pi = parse(Float64, cols[idx["baseline_pi"]])
        out[(component, mode_id)] = (
            existence_probability = existence_probability,
            baseline_pi = baseline_pi,
        )
    end
    return out
end

function select_robust_modes(result, bootstrap_modes; existence_min::Float64, mass_min::Float64)
    ids = Int[]
    for mode_id in 1:result.stable_mode_count
        baseline_pi = result.final_weights[mode_id]
        key = (result.component_name, mode_id)
        exists = haskey(bootstrap_modes, key)
        existence_probability = exists ? bootstrap_modes[key].existence_probability : 1.0
        if existence_probability >= existence_min && baseline_pi >= mass_min
            push!(ids, mode_id)
        end
    end
    return ids
end

function build_thresholds(values::Vector{Float64}, result, robust_mode_ids::Vector{Int}, quantile_levels)
    thresholds = Float64[]
    append!(thresholds, quantile(values, collect(quantile_levels)))
    if length(robust_mode_ids) >= 2
        for boundary_id in 1:length(result.final_boundaries)
            left_mode = boundary_id
            right_mode = boundary_id + 1
            if (left_mode in robust_mode_ids) && (right_mode in robust_mode_ids)
                push!(thresholds, result.final_boundaries[boundary_id])
            end
        end
    end
    return dedupe_sorted(thresholds)
end

function dedupe_sorted(values::Vector{Float64}; tol::Float64 = 1e-8)
    isempty(values) && return Float64[]
    sorted = sort(values)
    out = [sorted[1]]
    for x in sorted[2:end]
        abs(x - out[end]) > tol && push!(out, x)
    end
    return out
end

function append_feature_family!(feature_defs, feature_columns, window::AbstractString, component::AbstractString,
                                values::Vector{Float64}, thresholds::Vector{Float64},
                                robust_mode_ids::Vector{Int}, result, bootstrap_modes)
    for (it, c) in enumerate(thresholds)
        feature_id = "h_" * component * "_cdf_" * lpad(string(it), 2, '0')
        column = Float64.(values .<= c)
        push!(feature_defs, (
            window = String(window),
            feature_id = feature_id,
            component = String(component),
            family = "cdf_indicator",
            role = "primary",
            threshold = c,
            left_bound = NaN,
            right_bound = NaN,
            mode_id = 0,
            bootstrap_existence_probability = NaN,
            baseline_mode_weight = NaN,
            description = "1{Y<=" * fmt(c) * "}",
        ))
        push!(feature_columns, column)
    end

    for mode_id in robust_mode_ids
        bounds = result.final_intervals[mode_id]
        mask = Step4Screening.interval_mask(values, bounds.left, bounds.right)
        occ = Float64.(mask)
        yocc = values .* occ
        y2occ = (values .^ 2) .* occ
        bootstrap_key = (component, mode_id)
        existence_probability = haskey(bootstrap_modes, bootstrap_key) ? bootstrap_modes[bootstrap_key].existence_probability : 1.0
        baseline_pi = result.final_weights[mode_id]
        left_string = bound_string(bounds.left)
        right_string = bound_string(bounds.right)

        for (suffix, family, column, desc) in (
            ("occ", "mode_indicator", occ, "1{Y in [" * left_string * "," * right_string * "]}"),
            ("y", "mode_first_moment", yocc, "Y*1{Y in [" * left_string * "," * right_string * "]}"),
            ("y2", "mode_second_moment", y2occ, "Y^2*1{Y in [" * left_string * "," * right_string * "]}"),
        )
            feature_id = "h_" * component * "_" * suffix * "_m" * lpad(string(mode_id), 2, '0')
            push!(feature_defs, (
                window = String(window),
                feature_id = feature_id,
                component = String(component),
                family = family,
                role = "primary",
                threshold = NaN,
                left_bound = bounds.left,
                right_bound = bounds.right,
                mode_id = mode_id,
                bootstrap_existence_probability = existence_probability,
                baseline_mode_weight = baseline_pi,
                description = desc,
            ))
            push!(feature_columns, column)
        end
    end
end

function build_sample_info(references)
    reference_names = String[]
    reference_sample_indices = Int[]
    for ref in references
        n = size(ref.y, 1)
        append!(reference_names, fill(String(ref.name), n))
        append!(reference_sample_indices, collect(1:n))
    end
    return (
        reference_names = reference_names,
        reference_sample_indices = reference_sample_indices,
    )
end

function write_feature_definitions(filepath::AbstractString, feature_defs)
    header = ["window", "feature_id", "component", "family", "role",
              "threshold", "left_bound", "right_bound", "mode_id",
              "bootstrap_existence_probability", "baseline_mode_weight", "description"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for def in feature_defs
            row = [
                def.window,
                def.feature_id,
                def.component,
                def.family,
                def.role,
                nan_or_blank(def.threshold),
                bound_or_blank(def.left_bound),
                bound_or_blank(def.right_bound),
                string(def.mode_id),
                nan_or_blank(def.bootstrap_existence_probability),
                nan_or_blank(def.baseline_mode_weight),
                def.description,
            ]
            println(io, join(row, ","))
        end
    end
end

function write_feature_summary(filepath::AbstractString, feature_defs, feature_columns)
    header = ["window", "feature_id", "component", "family", "mode_id",
              "feature_mean", "feature_variance",
              "threshold", "left_bound", "right_bound",
              "bootstrap_existence_probability", "baseline_mode_weight"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for (def, column) in zip(feature_defs, feature_columns)
            row = [
                def.window,
                def.feature_id,
                def.component,
                def.family,
                string(def.mode_id),
                fmt(mean(column)),
                fmt(var(column; corrected = true)),
                nan_or_blank(def.threshold),
                bound_or_blank(def.left_bound),
                bound_or_blank(def.right_bound),
                nan_or_blank(def.bootstrap_existence_probability),
                nan_or_blank(def.baseline_mode_weight),
            ]
            println(io, join(row, ","))
        end
    end
end

function write_feature_values(filepath::AbstractString, pooled_y::Matrix{Float64}, sample_info, feature_defs, feature_columns)
    header = ["sample_index", "reference_name", "reference_sample_index",
              "log10_kxx", "log10_kyy", "log10_kzz"]
    append!(header, [def.feature_id for def in feature_defs])

    n = size(pooled_y, 1)
    open(filepath, "w") do io
        println(io, join(header, ","))
        for i in 1:n
            row = [
                string(i),
                sample_info.reference_names[i],
                string(sample_info.reference_sample_indices[i]),
                fmt(pooled_y[i, 1]),
                fmt(pooled_y[i, 2]),
                fmt(pooled_y[i, 3]),
            ]
            append!(row, [fmt(column[i]) for column in feature_columns])
            println(io, join(row, ","))
        end
    end
end

function write_feature_metadata(filepath::AbstractString, window::AbstractString, opt, bootstrap_modes)
    header = ["window", "quantile_levels", "bootstrap_dir_used",
              "robust_mode_existence_min", "robust_mode_mass_min",
              "bandwidth_factor_min", "bandwidth_factor_max", "num_bandwidths",
              "grid_size", "screen_min_mode_mass", "persistence_threshold",
              "screen_min_prominence", "merge_prominence_threshold", "merge_separation_threshold",
              "bootstrap_filter_active"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            String(window),
            join(string.(opt.quantile_levels), ";"),
            isempty(opt.bootstrap_dir) ? "" : opt.bootstrap_dir,
            fmt(opt.robust_mode_existence_min),
            fmt(opt.robust_mode_mass_min),
            fmt(opt.factor_min),
            fmt(opt.factor_max),
            string(opt.num_bandwidths),
            string(opt.grid_size),
            fmt(opt.min_mode_mass),
            fmt(opt.persistence_threshold),
            fmt(opt.min_prominence),
            fmt(opt.merge_prominence_threshold),
            fmt(opt.merge_separation_threshold),
            string(!isempty(bootstrap_modes)),
        ], ","))
    end
end

function bound_string(x)
    if !isfinite(x)
        return x < 0 ? "-Inf" : "Inf"
    end
    return fmt(x)
end

function bound_or_blank(x)
    return (x isa Number && !isnan(x)) ? bound_string(x) : ""
end

function nan_or_blank(x)
    return (x isa Number && isfinite(x)) ? fmt(x) : ""
end

fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
