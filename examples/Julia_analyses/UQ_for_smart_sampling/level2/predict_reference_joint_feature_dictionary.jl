#!/usr/bin/env julia

"""
Joint step-3 shared feature dictionary for pooled rigorous references.

This script defines a first-pass shared joint feature dictionary across a set of
windows and evaluates those features on every pooled rigorous reference draw.

The current dictionary has two families:

1. Shared anisotropy indicators
       A12 = log10(kxx/kyy)
       A13 = log10(kxx/kzz)
       A23 = log10(kyy/kzz)
   with shared thresholds built from the pooled multi-window anisotropy library.

2. Shared pairwise joint event indicators
   for all three component pairs:
       (kxx, kyy), (kxx, kzz), and (kyy, kzz)
   using shared low/high thresholds from pooled multi-window componentwise
   marginals, defaulting to q10 and q90.

Outputs:
    - shared joint feature definitions
    - per-window joint feature summary (mean/variance)
    - per-sample joint feature values
    - metadata

This is the shared joint feature dictionary step only. It does not yet compute
the final joint variability/complexity score.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT"])
"""

const REQUIRED_PACKAGES = ["MAT"]
missing_packages = [pkg for pkg in REQUIRED_PACKAGES if Base.find_package(pkg) === nothing]
if !isempty(missing_packages)
    pkg_list = join(["\"" * pkg * "\"" for pkg in missing_packages], ", ")
    error("Missing Julia packages: $(join(missing_packages, ", ")). Install them with:\n" *
          "using Pkg; Pkg.add([$pkg_list])")
end

using MAT
using Statistics
using Printf

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_INDEX = Dict("kxx" => 1, "kyy" => 2, "kzz" => 3)
const ANISO_NAMES = ("A12", "A13", "A23")
const ANISO_COMPONENTS = Dict("A12" => ("kxx", "kyy"),
                              "A13" => ("kxx", "kzz"),
                              "A23" => ("kyy", "kzz"))
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_ANISO_Q = (0.10, 0.25, 0.50, 0.75, 0.90)
const DEFAULT_PAIRS = ("kxx-kyy", "kxx-kzz", "kyy-kzz")

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_joint_feature_dictionary")),
        "windows" => "",
        "anisotropy-quantile-levels" => join(string.(DEFAULT_ANISO_Q), ","),
        "pair-event-pairs" => join(DEFAULT_PAIRS, ","),
        "pair-event-low-quantile" => "0.10",
        "pair-event-high-quantile" => "0.90",
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
    isempty(requested_windows) && error("Please provide --windows, e.g. --windows famp1,famp2,famp3")

    anisotropy_quantiles = Tuple(parse.(Float64, split(options["anisotropy-quantile-levels"], ",")))
    pair_names = Tuple(String[strip(x) for x in split(options["pair-event-pairs"], ",") if !isempty(strip(x))])

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        requested_windows = requested_windows,
        anisotropy_quantiles = anisotropy_quantiles,
        pair_names = pair_names,
        pair_low_quantile = parse(Float64, options["pair-event-low-quantile"]),
        pair_high_quantile = parse(Float64, options["pair-event-high-quantile"]),
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_feature_dictionary.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>                    Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>                  Folder where the joint dictionary outputs are saved")
    println("  --windows <names>                    Comma-separated list like famp1,famp2,famp3")
    println("  --anisotropy-quantile-levels <list>  Shared anisotropy quantiles (default: 0.10,0.25,0.50,0.75,0.90)")
    println("  --pair-event-pairs <list>            Pair list like kxx-kyy,kxx-kzz,kyy-kzz")
    println("  --pair-event-low-quantile <x>        Shared low threshold quantile for joint events (default: 0.10)")
    println("  --pair-event-high-quantile <x>       Shared high threshold quantile for joint events (default: 0.90)")
    println("  -h, --help                           Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No requested windows found under $(opt.data_dir)")

    pooled_by_window = Dict{String, Matrix{Float64}}()
    component_library = Dict(name => Float64[] for name in COMPONENT_NAMES)
    anisotropy_library = Dict(name => Float64[] for name in ANISO_NAMES)

    println("Loading pooled joint data for $(length(windows)) window(s)...")
    for (window, reference_dir) in windows
        references = load_references(reference_dir)
        pooled = reduce(vcat, [ref.y for ref in references])
        pooled_by_window[window] = pooled

        for (ic, name) in enumerate(COMPONENT_NAMES)
            append!(component_library[name], vec(pooled[:, ic]))
        end

        aniso = anisotropy_matrix(pooled)
        for (ia, name) in enumerate(ANISO_NAMES)
            append!(anisotropy_library[name], vec(aniso[:, ia]))
        end
    end

    feature_defs = build_shared_feature_definitions(component_library, anisotropy_library, opt)
    feature_values = build_feature_values(opt.requested_windows, pooled_by_window, feature_defs)
    feature_summary = summarize_feature_values(feature_values)

    write_feature_definitions(joinpath(opt.output_dir, "joint_shared_feature_dictionary.csv"), feature_defs)
    write_feature_summary(joinpath(opt.output_dir, "joint_shared_feature_summary.csv"), feature_summary)
    write_feature_values(joinpath(opt.output_dir, "joint_shared_feature_values.csv"), feature_values)
    write_metadata(joinpath(opt.output_dir, "joint_shared_feature_metadata.csv"), opt, feature_defs)

    println("Saved outputs to $(opt.output_dir)")
end

function collect_windows(data_dir::AbstractString, requested_windows::Vector{String})
    isdir(data_dir) || error("Data directory does not exist: $data_dir")
    windows = Tuple{String, String}[]
    for window in requested_windows
        reference_dir = joinpath(data_dir, window, "references")
        isdir(reference_dir) || error("Reference folder does not exist: $reference_dir")
        files = filter(f -> startswith(f, "reference_R") && endswith(f, ".mat"), readdir(reference_dir))
        isempty(files) && error("No reference MAT files found in $reference_dir")
        push!(windows, (window, reference_dir))
    end
    return windows
end

function load_references(reference_dir::AbstractString)
    files = sort(filter(f -> startswith(f, "reference_R") && endswith(f, ".mat"), readdir(reference_dir)))
    references = NamedTuple[]
    for file in files
        filepath = joinpath(reference_dir, file)
        data = matread(filepath)
        haskey(data, "perms") || error("File does not contain perms: $filepath")
        perms = Matrix{Float64}(data["perms"])
        size(perms, 2) == 3 || error("Expected perms to have 3 columns in $filepath")
        all(perms .> 0) || error("perms contains non-positive values in $filepath")
        push!(references, (name = replace(replace(file, ".mat" => ""), "reference_" => ""),
                           y = log10.(perms)))
    end
    return references
end

function anisotropy_matrix(y::Matrix{Float64})
    return hcat(y[:, 1] .- y[:, 2],
                y[:, 1] .- y[:, 3],
                y[:, 2] .- y[:, 3])
end

function build_shared_feature_definitions(component_library, anisotropy_library, opt)
    defs = NamedTuple[]

    aniso_weight = 1.0 / (length(ANISO_NAMES) * length(opt.anisotropy_quantiles))
    for name in ANISO_NAMES
        thresholds = collect(quantile(anisotropy_library[name], collect(opt.anisotropy_quantiles)))
        comps = ANISO_COMPONENTS[name]
        for (i, (q, t)) in enumerate(zip(opt.anisotropy_quantiles, thresholds))
            push!(defs, (
                feature_id = @sprintf("h_%s_q%02d", name, round(Int, 100q)),
                family = "anisotropy_indicator",
                subfamily = name,
                component_a = comps[1],
                component_b = comps[2],
                component_c = "",
                operator = "<=",
                threshold_a = t,
                threshold_b = NaN,
                threshold_c = NaN,
                quantile_level = q,
                weight_within_family = aniso_weight,
                description = @sprintf("1{%s <= %.10g}", name, t),
            ))
        end
    end

    pair_event_weight = 1.0 / (4.0 * length(opt.pair_names))
    for pair_name in opt.pair_names
        comp_a, comp_b = parse_pair_name(pair_name)
        low_a = quantile(component_library[comp_a], opt.pair_low_quantile)
        high_a = quantile(component_library[comp_a], opt.pair_high_quantile)
        low_b = quantile(component_library[comp_b], opt.pair_low_quantile)
        high_b = quantile(component_library[comp_b], opt.pair_high_quantile)

        push!(defs, build_pair_event_def(pair_name, "LL", comp_a, comp_b, low_a, low_b, opt.pair_low_quantile, opt.pair_low_quantile, pair_event_weight))
        push!(defs, build_pair_event_def(pair_name, "LH", comp_a, comp_b, low_a, high_b, opt.pair_low_quantile, opt.pair_high_quantile, pair_event_weight))
        push!(defs, build_pair_event_def(pair_name, "HL", comp_a, comp_b, high_a, low_b, opt.pair_high_quantile, opt.pair_low_quantile, pair_event_weight))
        push!(defs, build_pair_event_def(pair_name, "HH", comp_a, comp_b, high_a, high_b, opt.pair_high_quantile, opt.pair_high_quantile, pair_event_weight))
    end

    return defs
end

function parse_pair_name(pair_name::AbstractString)
    pieces = split(pair_name, "-")
    length(pieces) == 2 || error("Pair must look like kxx-kzz, got: $pair_name")
    all(p -> haskey(COMPONENT_INDEX, p), pieces) || error("Unknown pair components in $pair_name")
    return pieces[1], pieces[2]
end

function build_pair_event_def(pair_name, event_code, comp_a, comp_b, thresh_a, thresh_b, q_a, q_b, weight)
    op_a = event_code[1] == 'L' ? "<=" : ">="
    op_b = event_code[2] == 'L' ? "<=" : ">="
    return (
        feature_id = "h_" * replace(pair_name, "-" => "_") * "_" * event_code,
        family = "pair_event_indicator",
        subfamily = pair_name,
        component_a = comp_a,
        component_b = comp_b,
        component_c = "",
        operator = event_code,
        threshold_a = thresh_a,
        threshold_b = thresh_b,
        threshold_c = NaN,
        quantile_level = NaN,
        weight_within_family = weight,
        description = @sprintf("1{%s %s %.10g, %s %s %.10g}", comp_a, op_a, thresh_a, comp_b, op_b, thresh_b),
    )
end

function build_feature_values(windows::Vector{String}, pooled_by_window, feature_defs)
    rows = NamedTuple[]
    for window in windows
        y = pooled_by_window[window]
        aniso = anisotropy_matrix(y)
        n = size(y, 1)
        for i in 1:n
            for def in feature_defs
                value = evaluate_feature(def, y, aniso, i)
                push!(rows, (
                    window = window,
                    sample_index = i,
                    feature_id = def.feature_id,
                    family = def.family,
                    subfamily = def.subfamily,
                    feature_value = value,
                ))
            end
        end
    end
    return rows
end

function evaluate_feature(def, y::Matrix{Float64}, aniso::Matrix{Float64}, i::Int)
    if def.family == "anisotropy_indicator"
        ia = def.subfamily == "A12" ? 1 : def.subfamily == "A13" ? 2 : 3
        return aniso[i, ia] <= def.threshold_a ? 1.0 : 0.0
    elseif def.family == "pair_event_indicator"
        a = COMPONENT_INDEX[def.component_a]
        b = COMPONENT_INDEX[def.component_b]
        va = y[i, a]
        vb = y[i, b]
        pass_a = def.operator[1] == 'L' ? (va <= def.threshold_a) : (va >= def.threshold_a)
        pass_b = def.operator[2] == 'L' ? (vb <= def.threshold_b) : (vb >= def.threshold_b)
        return (pass_a && pass_b) ? 1.0 : 0.0
    else
        error("Unknown feature family $(def.family)")
    end
end

function summarize_feature_values(feature_values)
    groups = Dict{Tuple{String, String}, Vector{Float64}}()
    families = Dict{Tuple{String, String}, Tuple{String, String}}()
    for row in feature_values
        key = (row.window, row.feature_id)
        if !haskey(groups, key)
            groups[key] = Float64[]
            families[key] = (row.family, row.subfamily)
        end
        push!(groups[key], row.feature_value)
    end

    rows = NamedTuple[]
    for key in sort(collect(keys(groups)))
        vals = groups[key]
        fam = families[key]
        push!(rows, (
            window = key[1],
            feature_id = key[2],
            family = fam[1],
            subfamily = fam[2],
            feature_mean = mean(vals),
            feature_variance = var(vals; corrected = true),
        ))
    end
    return rows
end

function write_feature_definitions(filepath, feature_defs)
    header = ["feature_id", "family", "subfamily", "component_a", "component_b", "component_c",
              "operator", "threshold_a", "threshold_b", "threshold_c", "quantile_level",
              "weight_within_family", "description"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for def in feature_defs
            println(io, join([
                def.feature_id,
                def.family,
                def.subfamily,
                def.component_a,
                def.component_b,
                def.component_c,
                def.operator,
                nan_or_blank(def.threshold_a),
                nan_or_blank(def.threshold_b),
                nan_or_blank(def.threshold_c),
                nan_or_blank(def.quantile_level),
                fmt(def.weight_within_family),
                def.description,
            ], ","))
        end
    end
end

function write_feature_summary(filepath, rows)
    header = ["window", "feature_id", "family", "subfamily", "feature_mean", "feature_variance"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                row.feature_id,
                row.family,
                row.subfamily,
                fmt(row.feature_mean),
                fmt(row.feature_variance),
            ], ","))
        end
    end
end

function write_feature_values(filepath, rows)
    header = ["window", "sample_index", "feature_id", "family", "subfamily", "feature_value"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                string(row.sample_index),
                row.feature_id,
                row.family,
                row.subfamily,
                fmt(row.feature_value),
            ], ","))
        end
    end
end

function write_metadata(filepath, opt, feature_defs)
    n_aniso = count(def -> def.family == "anisotropy_indicator", feature_defs)
    n_pair = count(def -> def.family == "pair_event_indicator", feature_defs)
    header = ["windows", "anisotropy_quantiles", "pair_names", "pair_low_quantile", "pair_high_quantile",
              "num_anisotropy_features", "num_pair_event_features", "num_total_features"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            join(opt.requested_windows, ";"),
            join(string.(opt.anisotropy_quantiles), ";"),
            join(opt.pair_names, ";"),
            fmt(opt.pair_low_quantile),
            fmt(opt.pair_high_quantile),
            string(n_aniso),
            string(n_pair),
            string(length(feature_defs)),
        ], ","))
    end
end

nan_or_blank(x) = (x isa Number && isfinite(x)) ? fmt(x) : ""
fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
