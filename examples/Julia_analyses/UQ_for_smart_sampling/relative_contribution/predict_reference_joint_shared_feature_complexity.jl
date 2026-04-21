#!/usr/bin/env julia

"""
Compute a rigorous shared-feature joint Level-2 variability score and combine it
with the already-rigorous shared-threshold componentwise score.

Inputs:
1. Step-3 shared joint feature dictionary CSV
2. Step-3 per-window shared joint feature summary CSV
3. Shared-threshold componentwise window complexity CSV

For each window, this script:
1. Computes family-normalized joint scores
       s^2_{i,family} = sum_k w_k Var(h_k | g_i)
   where the feature weights w_k sum to 1 within each family.
2. Combines the family scores into an overall joint score
       s^2_{i,joint} = lambda_A s^2_{i,aniso} + lambda_E s^2_{i,event}
   with family weights lambda_A + lambda_E = 1.
3. Combines the joint score with the rigorous componentwise score
       s^2_{i,total} = (1-gamma) s^2_{i,comp} + gamma s^2_{i,joint}
   where gamma is the joint mixing weight.

The resulting joint and combined scores are directly comparable across windows
because they are built from one fixed shared feature dictionary.

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/relative_contribution/predict_reference_joint_shared_feature_complexity.jl \\
        --joint-feature-definitions D:\\codex_gom\\reference_joint_feature_dictionary_famp123_v2\\joint_shared_feature_dictionary.csv \\
        --joint-feature-summary D:\\codex_gom\\reference_joint_feature_dictionary_famp123_v2\\joint_shared_feature_summary.csv \\
        --componentwise-window-summary D:\\codex_gom\\reference_componentwise_shared_threshold_complexity_famp123\\componentwise_shared_threshold_window_complexity.csv \\
        --output-dir D:\\codex_gom\\reference_joint_shared_feature_complexity_famp123_v1
"""

using Printf
using Statistics

const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const MAX_INDICATOR_VARIANCE = 0.25
const DEFAULT_WINDOWS = String[]

function parse_args(args::Vector{String})
    options = Dict(
        "joint-feature-definitions" => "",
        "joint-feature-summary" => "",
        "componentwise-window-summary" => "",
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_joint_shared_feature_complexity")),
        "windows" => "",
        "anisotropy-family-weight" => "0.5",
        "pair-event-family-weight" => "0.5",
        "joint-mixing-weight" => "0.5",
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

    isempty(options["joint-feature-definitions"]) && error("Please provide --joint-feature-definitions")
    isempty(options["joint-feature-summary"]) && error("Please provide --joint-feature-summary")
    isempty(options["componentwise-window-summary"]) && error("Please provide --componentwise-window-summary")

    requested_windows = isempty(options["windows"]) ? String[] :
                        String[strip(w) for w in split(options["windows"], ",") if !isempty(strip(w))]
    lambda_aniso = parse(Float64, options["anisotropy-family-weight"])
    lambda_pair = parse(Float64, options["pair-event-family-weight"])
    gamma = parse(Float64, options["joint-mixing-weight"])

    all(x -> 0.0 <= x <= 1.0, (lambda_aniso, lambda_pair, gamma)) ||
        error("Family weights and joint mixing weight must be in [0, 1]")
    abs(lambda_aniso + lambda_pair - 1.0) <= 1e-8 ||
        error("Anisotropy and pair-event family weights must sum to 1")

    return (
        joint_feature_definitions = normpath(options["joint-feature-definitions"]),
        joint_feature_summary = normpath(options["joint-feature-summary"]),
        componentwise_window_summary = normpath(options["componentwise-window-summary"]),
        output_dir = normpath(options["output-dir"]),
        requested_windows = requested_windows,
        lambda_aniso = lambda_aniso,
        lambda_pair = lambda_pair,
        gamma = gamma,
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/relative_contribution/predict_reference_joint_shared_feature_complexity.jl [options]")
    println()
    println("Options:")
    println("  --joint-feature-definitions <path>   Step-3 joint_shared_feature_dictionary.csv")
    println("  --joint-feature-summary <path>       Step-3 joint_shared_feature_summary.csv")
    println("  --componentwise-window-summary <path>  componentwise_shared_threshold_window_complexity.csv")
    println("  --output-dir <path>                  Folder where the score outputs are saved")
    println("  --windows <names>                    Optional comma-separated window subset")
    println("  --anisotropy-family-weight <x>       Weight on anisotropy family (default: 0.5)")
    println("  --pair-event-family-weight <x>       Weight on pair-event family (default: 0.5)")
    println("  --joint-mixing-weight <x>            Weight on joint score in combined score (default: 0.5)")
    println("  -h, --help                           Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    def_rows = read_joint_definition_rows(opt.joint_feature_definitions)
    summary_rows = read_joint_summary_rows(opt.joint_feature_summary)
    componentwise_rows = read_componentwise_window_rows(opt.componentwise_window_summary)

    all_windows = sort(unique([row.window for row in summary_rows]))
    windows = isempty(opt.requested_windows) ? all_windows : opt.requested_windows

    feature_lookup = Dict(row.feature_id => row for row in def_rows)
    family_scores, subfamily_scores = compute_joint_family_scores(summary_rows, feature_lookup, windows)
    window_scores = compute_joint_window_scores(family_scores, subfamily_scores, opt)
    combined_scores = compute_combined_scores(window_scores, componentwise_rows, opt)

    write_family_scores_csv(joinpath(opt.output_dir, "joint_shared_feature_family_scores.csv"), family_scores)
    write_subfamily_scores_csv(joinpath(opt.output_dir, "joint_shared_feature_subfamily_scores.csv"), subfamily_scores)
    write_window_scores_csv(joinpath(opt.output_dir, "joint_shared_feature_window_scores.csv"), window_scores)
    write_combined_scores_csv(joinpath(opt.output_dir, "joint_componentwise_combined_window_scores.csv"), combined_scores)
    write_metadata_csv(joinpath(opt.output_dir, "joint_shared_feature_complexity_metadata.csv"), opt, def_rows, windows)

    println("Saved outputs to $(opt.output_dir)")
end

function read_joint_definition_rows(filepath::AbstractString)
    isfile(filepath) || error("Joint feature definitions file does not exist: $filepath")
    rows = NamedTuple[]
    open(filepath, "r") do io
        readline(io) # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split(line, ",")
            length(fields) >= 12 || error("Malformed definitions row in $filepath: $line")
            push!(rows, (
                feature_id = fields[1],
                family = fields[2],
                subfamily = fields[3],
                component_a = fields[4],
                component_b = fields[5],
                component_c = fields[6],
                operator = fields[7],
                threshold_a = parse_or_nan(fields[8]),
                threshold_b = parse_or_nan(fields[9]),
                threshold_c = parse_or_nan(fields[10]),
                quantile_level = parse_or_nan(fields[11]),
                weight_within_family = parse(Float64, fields[12]),
            ))
        end
    end
    return rows
end

function read_joint_summary_rows(filepath::AbstractString)
    isfile(filepath) || error("Joint feature summary file does not exist: $filepath")
    rows = NamedTuple[]
    open(filepath, "r") do io
        readline(io) # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split(line, ",")
            length(fields) == 6 || error("Malformed summary row in $filepath: $line")
            push!(rows, (
                window = fields[1],
                feature_id = fields[2],
                family = fields[3],
                subfamily = fields[4],
                feature_mean = parse(Float64, fields[5]),
                feature_variance = parse(Float64, fields[6]),
            ))
        end
    end
    return rows
end

function read_componentwise_window_rows(filepath::AbstractString)
    isfile(filepath) || error("Componentwise window summary file does not exist: $filepath")
    rows = Dict{String, NamedTuple}()
    open(filepath, "r") do io
        readline(io) # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            fields = split(line, ",")
            length(fields) == 8 || error("Malformed componentwise row in $filepath: $line")
            rows[fields[1]] = (
                window = fields[1],
                overall_complexity_s2 = parse(Float64, fields[2]),
                overall_complexity_s = parse(Float64, fields[3]),
                overall_complexity_normalized = parse(Float64, fields[4]),
                kxx_component_complexity_s2 = parse(Float64, fields[5]),
                kyy_component_complexity_s2 = parse(Float64, fields[6]),
                kzz_component_complexity_s2 = parse(Float64, fields[7]),
                hardest_component = fields[8],
            )
        end
    end
    return rows
end

function compute_joint_family_scores(summary_rows, feature_lookup, windows)
    family_scores = NamedTuple[]
    subfamily_scores = NamedTuple[]

    by_window_family = Dict{Tuple{String, String}, Vector{NamedTuple}}()
    by_window_subfamily = Dict{Tuple{String, String, String}, Vector{NamedTuple}}()

    allowed_windows = Set(windows)
    for row in summary_rows
        row.window in allowed_windows || continue
        def = get(feature_lookup, row.feature_id, nothing)
        def === nothing && error("Missing feature definition for $(row.feature_id)")

        family_key = (row.window, row.family)
        subfamily_key = (row.window, row.family, row.subfamily)

        merged = (
            window = row.window,
            feature_id = row.feature_id,
            family = row.family,
            subfamily = row.subfamily,
            feature_mean = row.feature_mean,
            feature_variance = row.feature_variance,
            weight_within_family = def.weight_within_family,
        )

        push!(get!(by_window_family, family_key, NamedTuple[]), merged)
        push!(get!(by_window_subfamily, subfamily_key, NamedTuple[]), merged)
    end

    for key in sort(collect(keys(by_window_family)))
        rows = by_window_family[key]
        weighted_s2 = sum(r.weight_within_family * r.feature_variance for r in rows)
        push!(family_scores, (
            window = key[1],
            family = key[2],
            num_features = length(rows),
            weight_sum = sum(r.weight_within_family for r in rows),
            family_complexity_s2 = weighted_s2,
            family_complexity_s = sqrt(weighted_s2),
            family_complexity_normalized = weighted_s2 / MAX_INDICATOR_VARIANCE,
            mean_feature_variance = mean(r.feature_variance for r in rows),
            max_feature_variance = maximum(r.feature_variance for r in rows),
            dominant_feature_id = argmax_feature(rows),
        ))
    end

    for key in sort(collect(keys(by_window_subfamily)))
        rows = by_window_subfamily[key]
        weighted_s2 = sum(r.weight_within_family * r.feature_variance for r in rows)
        push!(subfamily_scores, (
            window = key[1],
            family = key[2],
            subfamily = key[3],
            num_features = length(rows),
            weight_sum = sum(r.weight_within_family for r in rows),
            subfamily_weighted_s2 = weighted_s2,
            subfamily_weighted_s = sqrt(weighted_s2),
            subfamily_weighted_normalized = weighted_s2 / MAX_INDICATOR_VARIANCE,
            mean_feature_variance = mean(r.feature_variance for r in rows),
            max_feature_variance = maximum(r.feature_variance for r in rows),
            dominant_feature_id = argmax_feature(rows),
        ))
    end

    return family_scores, subfamily_scores
end

function argmax_feature(rows)
    idx = argmax([r.feature_variance for r in rows])
    return rows[idx].feature_id
end

function compute_joint_window_scores(family_scores, subfamily_scores, opt)
    family_map = Dict((row.window, row.family) => row for row in family_scores)
    windows = sort(unique(row.window for row in family_scores))
    rows = NamedTuple[]

    for window in windows
        aniso = get(family_map, (window, "anisotropy_indicator"), nothing)
        pair = get(family_map, (window, "pair_event_indicator"), nothing)
        aniso === nothing && error("Missing anisotropy family score for $window")
        pair === nothing && error("Missing pair-event family score for $window")

        aniso_contrib = opt.lambda_aniso * aniso.family_complexity_s2
        pair_contrib = opt.lambda_pair * pair.family_complexity_s2
        joint_s2 = aniso_contrib + pair_contrib

        window_subfamilies = filter(row -> row.window == window, subfamily_scores)
        dominant_subfamily = find_dominant_subfamily(window_subfamilies, opt)

        push!(rows, (
            window = window,
            anisotropy_family_weight = opt.lambda_aniso,
            pair_event_family_weight = opt.lambda_pair,
            anisotropy_complexity_s2 = aniso.family_complexity_s2,
            pair_event_complexity_s2 = pair.family_complexity_s2,
            weighted_anisotropy_contribution_s2 = aniso_contrib,
            weighted_pair_event_contribution_s2 = pair_contrib,
            joint_complexity_s2 = joint_s2,
            joint_complexity_s = sqrt(joint_s2),
            joint_complexity_normalized = joint_s2 / MAX_INDICATOR_VARIANCE,
            dominant_joint_family = aniso_contrib >= pair_contrib ? "anisotropy_indicator" : "pair_event_indicator",
            dominant_joint_subfamily = dominant_subfamily,
        ))
    end

    return rows
end

function find_dominant_subfamily(rows, opt)
    isempty(rows) && return ""
    best_label = ""
    best_value = -Inf
    for row in rows
        family_weight = row.family == "anisotropy_indicator" ? opt.lambda_aniso : opt.lambda_pair
        value = family_weight * row.subfamily_weighted_s2
        if value > best_value
            best_value = value
            best_label = row.subfamily
        end
    end
    return best_label
end

function compute_combined_scores(window_scores, componentwise_rows, opt)
    rows = NamedTuple[]
    comp_weight = 1.0 - opt.gamma
    joint_weight = opt.gamma

    for row in window_scores
        haskey(componentwise_rows, row.window) || error("Missing componentwise score for $(row.window)")
        comp = componentwise_rows[row.window]

        weighted_comp = comp_weight * comp.overall_complexity_s2
        weighted_joint = joint_weight * row.joint_complexity_s2
        total_s2 = weighted_comp + weighted_joint

        push!(rows, (
            window = row.window,
            componentwise_weight = comp_weight,
            joint_weight = joint_weight,
            componentwise_complexity_s2 = comp.overall_complexity_s2,
            joint_complexity_s2 = row.joint_complexity_s2,
            weighted_componentwise_contribution_s2 = weighted_comp,
            weighted_joint_contribution_s2 = weighted_joint,
            combined_complexity_s2 = total_s2,
            combined_complexity_s = sqrt(total_s2),
            combined_complexity_normalized = total_s2 / MAX_INDICATOR_VARIANCE,
            dominant_domain = weighted_comp >= weighted_joint ? "componentwise" : "joint",
            hardest_component = comp.hardest_component,
            dominant_joint_family = row.dominant_joint_family,
            dominant_joint_subfamily = row.dominant_joint_subfamily,
        ))
    end

    return rows
end

function write_family_scores_csv(filepath, rows)
    header = ["window", "family", "num_features", "weight_sum", "family_complexity_s2",
              "family_complexity_s", "family_complexity_normalized", "mean_feature_variance",
              "max_feature_variance", "dominant_feature_id"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                row.family,
                string(row.num_features),
                fmt(row.weight_sum),
                fmt(row.family_complexity_s2),
                fmt(row.family_complexity_s),
                fmt(row.family_complexity_normalized),
                fmt(row.mean_feature_variance),
                fmt(row.max_feature_variance),
                row.dominant_feature_id,
            ], ","))
        end
    end
end

function write_subfamily_scores_csv(filepath, rows)
    header = ["window", "family", "subfamily", "num_features", "weight_sum", "subfamily_weighted_s2",
              "subfamily_weighted_s", "subfamily_weighted_normalized", "mean_feature_variance",
              "max_feature_variance", "dominant_feature_id"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                row.family,
                row.subfamily,
                string(row.num_features),
                fmt(row.weight_sum),
                fmt(row.subfamily_weighted_s2),
                fmt(row.subfamily_weighted_s),
                fmt(row.subfamily_weighted_normalized),
                fmt(row.mean_feature_variance),
                fmt(row.max_feature_variance),
                row.dominant_feature_id,
            ], ","))
        end
    end
end

function write_window_scores_csv(filepath, rows)
    header = ["window", "anisotropy_family_weight", "pair_event_family_weight",
              "anisotropy_complexity_s2", "pair_event_complexity_s2",
              "weighted_anisotropy_contribution_s2", "weighted_pair_event_contribution_s2",
              "joint_complexity_s2", "joint_complexity_s", "joint_complexity_normalized",
              "dominant_joint_family", "dominant_joint_subfamily"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                fmt(row.anisotropy_family_weight),
                fmt(row.pair_event_family_weight),
                fmt(row.anisotropy_complexity_s2),
                fmt(row.pair_event_complexity_s2),
                fmt(row.weighted_anisotropy_contribution_s2),
                fmt(row.weighted_pair_event_contribution_s2),
                fmt(row.joint_complexity_s2),
                fmt(row.joint_complexity_s),
                fmt(row.joint_complexity_normalized),
                row.dominant_joint_family,
                row.dominant_joint_subfamily,
            ], ","))
        end
    end
end

function write_combined_scores_csv(filepath, rows)
    header = ["window", "componentwise_weight", "joint_weight", "componentwise_complexity_s2",
              "joint_complexity_s2", "weighted_componentwise_contribution_s2",
              "weighted_joint_contribution_s2", "combined_complexity_s2", "combined_complexity_s",
              "combined_complexity_normalized", "dominant_domain", "hardest_component",
              "dominant_joint_family", "dominant_joint_subfamily"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                fmt(row.componentwise_weight),
                fmt(row.joint_weight),
                fmt(row.componentwise_complexity_s2),
                fmt(row.joint_complexity_s2),
                fmt(row.weighted_componentwise_contribution_s2),
                fmt(row.weighted_joint_contribution_s2),
                fmt(row.combined_complexity_s2),
                fmt(row.combined_complexity_s),
                fmt(row.combined_complexity_normalized),
                row.dominant_domain,
                row.hardest_component,
                row.dominant_joint_family,
                row.dominant_joint_subfamily,
            ], ","))
        end
    end
end

function write_metadata_csv(filepath, opt, def_rows, windows)
    n_aniso = count(row -> row.family == "anisotropy_indicator", def_rows)
    n_pair = count(row -> row.family == "pair_event_indicator", def_rows)
    header = ["windows", "joint_feature_definitions", "joint_feature_summary", "componentwise_window_summary",
              "anisotropy_family_weight", "pair_event_family_weight", "joint_mixing_weight",
              "num_anisotropy_features", "num_pair_event_features", "num_total_features"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            join(windows, ";"),
            opt.joint_feature_definitions,
            opt.joint_feature_summary,
            opt.componentwise_window_summary,
            fmt(opt.lambda_aniso),
            fmt(opt.lambda_pair),
            fmt(opt.gamma),
            string(n_aniso),
            string(n_pair),
            string(length(def_rows)),
        ], ","))
    end
end

parse_or_nan(x::AbstractString) = isempty(x) ? NaN : parse(Float64, x)
fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
