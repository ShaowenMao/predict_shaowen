#!/usr/bin/env julia

"""
Compute a rigorously comparable componentwise Level-2 complexity score using a
shared threshold/CDF feature dictionary across multiple geologic scenarios.

For each selected window, this script:
1. Loads and pools the rigorous references R1/R2/R3 in log10(k) space.
2. Builds one shared threshold grid per component from the pooled reference
   library across all selected windows, using a fixed quantile grid q.
3. Defines the shared scalar features
       h_{j,l}(Y_j) = 1{Y_j <= c_{j,l}}
   for every component j and threshold c_{j,l}.
4. Estimates
       p_{i,j,l} = E[h_{j,l} | g_i]
                 ≈ mean(1{Y_j <= c_{j,l}})
       Var(h_{j,l} | g_i) ≈ p_{i,j,l} (1 - p_{i,j,l})
5. Computes a comparable component complexity score
       s_{i,j}^2 = sum_l w_l Var(h_{j,l} | g_i)
   with default equal threshold weights summing to 1.
6. Computes an overall window score
       s_i^2 = sum_j eta_j s_{i,j}^2
   with default equal component weights summing to 1.

Because the same threshold features are used for every window, the resulting
scores are directly comparable across geologic scenarios.

Outputs:
    - shared threshold grid CSV
    - per-window, per-component, per-threshold feature table
    - per-window, per-component complexity summary
    - per-window overall complexity summary
    - metadata CSV

Required Julia packages:
    using Pkg
    Pkg.add(["MAT"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/relative_contribution/predict_reference_componentwise_shared_threshold_complexity.jl --windows famp1,famp2,famp3
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
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_Q = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_shared_threshold_complexity")),
        "windows" => "",
        "quantile-levels" => join(string.(DEFAULT_Q), ","),
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
    quantile_levels = Tuple(parse.(Float64, split(options["quantile-levels"], ",")))

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        requested_windows = requested_windows,
        quantile_levels = quantile_levels,
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/relative_contribution/predict_reference_componentwise_shared_threshold_complexity.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>          Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>        Folder where the shared-threshold complexity outputs are saved")
    println("  --windows <names>          Comma-separated list like famp1,famp2,famp3")
    println("  --quantile-levels <list>   Shared quantile grid q (default: 0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99)")
    println("  -h, --help                 Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No requested windows found under $(opt.data_dir)")

    pooled_by_window = Dict{String, Matrix{Float64}}()
    library_by_component = Dict(name => Float64[] for name in COMPONENT_NAMES)

    println("Loading pooled references for $(length(windows)) window(s)...")
    for (window, reference_dir) in windows
        references = load_references(reference_dir)
        pooled = pool_references(references)
        pooled_by_window[window] = pooled.y
        for (ic, name) in enumerate(COMPONENT_NAMES)
            append!(library_by_component[name], vec(pooled.y[:, ic]))
        end
    end

    shared_thresholds = Dict{String, Vector{Float64}}()
    for name in COMPONENT_NAMES
        shared_thresholds[name] = collect(quantile(library_by_component[name], collect(opt.quantile_levels)))
    end

    threshold_rows = build_threshold_rows(shared_thresholds, opt.quantile_levels)
    feature_rows = NamedTuple[]
    component_rows = NamedTuple[]

    threshold_weights = fill(1.0 / length(opt.quantile_levels), length(opt.quantile_levels))
    component_weights = fill(1.0 / length(COMPONENT_NAMES), length(COMPONENT_NAMES))

    for window in opt.requested_windows
        pooled_y = pooled_by_window[window]
        component_s2 = Float64[]
        for (ic, name) in enumerate(COMPONENT_NAMES)
            values = vec(pooled_y[:, ic])
            thresholds = shared_thresholds[name]
            means = [mean(values .<= c) for c in thresholds]
            variances = [p * (1 - p) for p in means]
            s2 = sum(threshold_weights .* variances)
            push!(component_s2, s2)

            append!(feature_rows, build_feature_rows(window, name, thresholds, opt.quantile_levels, means, variances, threshold_weights))
            push!(component_rows, (
                window = window,
                component = name,
                num_shared_thresholds = length(thresholds),
                component_complexity_s2 = s2,
                component_complexity_s = sqrt(s2),
                component_complexity_normalized = s2 / 0.25,
                mean_threshold_indicator_variance = mean(variances),
                max_threshold_indicator_variance = maximum(variances),
                min_threshold_indicator_variance = minimum(variances),
            ))
        end

        window_s2 = sum(component_weights .* component_s2)
        push!(component_rows, (
            window = window,
            component = "__window__",
            num_shared_thresholds = length(opt.quantile_levels),
            component_complexity_s2 = window_s2,
            component_complexity_s = sqrt(window_s2),
            component_complexity_normalized = window_s2 / 0.25,
            mean_threshold_indicator_variance = mean(component_s2),
            max_threshold_indicator_variance = maximum(component_s2),
            min_threshold_indicator_variance = minimum(component_s2),
        ))
    end

    window_rows = build_window_rows(component_rows)

    write_threshold_grid_csv(joinpath(opt.output_dir, "componentwise_shared_threshold_grid.csv"), threshold_rows)
    write_feature_table_csv(joinpath(opt.output_dir, "componentwise_shared_threshold_feature_table.csv"), feature_rows)
    write_component_summary_csv(joinpath(opt.output_dir, "componentwise_shared_threshold_component_complexity.csv"), component_rows)
    write_window_summary_csv(joinpath(opt.output_dir, "componentwise_shared_threshold_window_complexity.csv"), window_rows)
    write_metadata_csv(joinpath(opt.output_dir, "componentwise_shared_threshold_complexity_metadata.csv"), opt)

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
                           perms = perms,
                           y = log10.(perms)))
    end
    return references
end

function pool_references(references)
    perms = reduce(vcat, [ref.perms for ref in references])
    y = reduce(vcat, [ref.y for ref in references])
    return (perms = perms, y = y)
end

function build_threshold_rows(shared_thresholds, quantile_levels)
    rows = NamedTuple[]
    for name in COMPONENT_NAMES
        thresholds = shared_thresholds[name]
        for (i, (q, c)) in enumerate(zip(quantile_levels, thresholds))
            push!(rows, (
                component = name,
                threshold_id = @sprintf("%s_q%02d", name, round(Int, 100q)),
                quantile_level = q,
                threshold_log10k = c,
                threshold_weight = 1.0 / length(quantile_levels),
                order_id = i,
            ))
        end
    end
    return rows
end

function build_feature_rows(window, component, thresholds, quantile_levels, means, variances, threshold_weights)
    rows = NamedTuple[]
    for i in eachindex(thresholds)
        push!(rows, (
            window = window,
            component = component,
            threshold_id = @sprintf("%s_q%02d", component, round(Int, 100quantile_levels[i])),
            quantile_level = quantile_levels[i],
            threshold_log10k = thresholds[i],
            threshold_weight = threshold_weights[i],
            feature_mean = means[i],
            feature_variance = variances[i],
        ))
    end
    return rows
end

function build_window_rows(component_rows)
    rows = NamedTuple[]
    windows = unique(row.window for row in component_rows if row.component != "__window__")
    for window in sort(collect(windows))
        comp_rows = [row for row in component_rows if row.window == window && row.component != "__window__"]
        overall = only([row for row in component_rows if row.window == window && row.component == "__window__"])
        push!(rows, (
            window = window,
            overall_complexity_s2 = overall.component_complexity_s2,
            overall_complexity_s = overall.component_complexity_s,
            overall_complexity_normalized = overall.component_complexity_normalized,
            kxx_component_complexity_s2 = component_s2(comp_rows, "kxx"),
            kyy_component_complexity_s2 = component_s2(comp_rows, "kyy"),
            kzz_component_complexity_s2 = component_s2(comp_rows, "kzz"),
            hardest_component = argmax_component(comp_rows),
        ))
    end
    return rows
end

component_s2(rows, name) = only(row.component_complexity_s2 for row in rows if row.component == name)

function argmax_component(rows)
    best = rows[1]
    for row in rows[2:end]
        if row.component_complexity_s2 > best.component_complexity_s2
            best = row
        end
    end
    return best.component
end

function write_threshold_grid_csv(filepath, rows)
    header = ["component", "threshold_id", "quantile_level", "threshold_log10k", "threshold_weight", "order_id"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.component,
                row.threshold_id,
                fmt(row.quantile_level),
                fmt(row.threshold_log10k),
                fmt(row.threshold_weight),
                string(row.order_id),
            ], ","))
        end
    end
end

function write_feature_table_csv(filepath, rows)
    header = ["window", "component", "threshold_id", "quantile_level", "threshold_log10k", "threshold_weight", "feature_mean", "feature_variance"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                row.component,
                row.threshold_id,
                fmt(row.quantile_level),
                fmt(row.threshold_log10k),
                fmt(row.threshold_weight),
                fmt(row.feature_mean),
                fmt(row.feature_variance),
            ], ","))
        end
    end
end

function write_component_summary_csv(filepath, rows)
    header = ["window", "component", "num_shared_thresholds",
              "component_complexity_s2", "component_complexity_s", "component_complexity_normalized",
              "mean_threshold_indicator_variance", "max_threshold_indicator_variance", "min_threshold_indicator_variance"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                row.component,
                string(row.num_shared_thresholds),
                fmt(row.component_complexity_s2),
                fmt(row.component_complexity_s),
                fmt(row.component_complexity_normalized),
                fmt(row.mean_threshold_indicator_variance),
                fmt(row.max_threshold_indicator_variance),
                fmt(row.min_threshold_indicator_variance),
            ], ","))
        end
    end
end

function write_window_summary_csv(filepath, rows)
    header = ["window", "overall_complexity_s2", "overall_complexity_s", "overall_complexity_normalized",
              "kxx_component_complexity_s2", "kyy_component_complexity_s2", "kzz_component_complexity_s2",
              "hardest_component"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window,
                fmt(row.overall_complexity_s2),
                fmt(row.overall_complexity_s),
                fmt(row.overall_complexity_normalized),
                fmt(row.kxx_component_complexity_s2),
                fmt(row.kyy_component_complexity_s2),
                fmt(row.kzz_component_complexity_s2),
                row.hardest_component,
            ], ","))
        end
    end
end

function write_metadata_csv(filepath, opt)
    header = ["windows", "quantile_levels", "threshold_weight_rule", "component_weight_rule",
              "component_formula", "window_formula", "normalized_scale_reference"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            join(opt.requested_windows, ";"),
            join(string.(opt.quantile_levels), ";"),
            "equal_weights_sum_to_1",
            "equal_weights_sum_to_1",
            "s_ij^2=sum_l w_l Var(1{Y_j<=c_jl}|g_i)",
            "s_i^2=(1/3)sum_j s_ij^2",
            "divide_by_0.25",
        ], ","))
    end
end

fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
