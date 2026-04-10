#!/usr/bin/env julia

"""
Componentwise agreement check for rigorous PREDICT reference ensembles.

This script is the first Julia-side step in the rigorous Level 2 workflow.
It reads the independently generated reference ensembles R1, R2, R3 for each
selected GOM window from the `gom_perm_reference_floor_convergence` outputs
and checks whether the references agree marginally before any pooling.

For each selected window it:
1. Loads all `reference_R*.mat` files from `data/<window>/references`.
2. Transforms permeability samples to
       y = (log10(kxx), log10(kyy), log10(kzz)).
3. For each component separately, plots:
       - empirical CDFs of R1/R2/R3
       - exceedance curves of R1/R2/R3
4. Saves per-reference quantiles and across-reference quantile spreads.
5. Saves simple pairwise reference distances for each component:
       - Kolmogorov-Smirnov distance between ECDFs
       - 1D Wasserstein distance on log10(k)
       - absolute median difference

This script is intentionally componentwise only. It is meant to answer:
    "Do the independent reference ensembles agree well enough to pool them?"

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_agreement.jl
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_agreement.jl --windows famp1
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_agreement.jl --data-dir path\\to\\data --output-dir path\\to\\out
"""

const REQUIRED_PACKAGES = ["MAT", "CairoMakie"]
missing_packages = [pkg for pkg in REQUIRED_PACKAGES if Base.find_package(pkg) === nothing]
if !isempty(missing_packages)
    pkg_list = join(["\"" * pkg * "\"" for pkg in missing_packages], ", ")
    error("Missing Julia packages: $(join(missing_packages, ", ")). Install them with:\n" *
          "using Pkg; Pkg.add([$pkg_list])")
end

using MAT
using CairoMakie
using Statistics
using Printf

CairoMakie.activate!()

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")
const DEFAULT_QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_agreement")),
        "windows" => "",
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
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_agreement.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>         Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>       Folder where figures and CSVs are saved")
    println("  --windows <names>         Comma-separated list like famp1,famp2")
    println("  -h, --help                Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No window reference folders found in $(opt.data_dir)")

    println("Processing $(length(windows)) window(s) from $(opt.data_dir)")
    for (window, reference_dir) in windows
        println("  - $window")
        process_window(window, reference_dir, opt.output_dir)
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

function process_window(window::AbstractString, reference_dir::AbstractString, output_dir::AbstractString)
    references = load_references(reference_dir)
    isempty(references) && error("No reference MAT files found in $reference_dir")

    window_dir = joinpath(output_dir, window)
    mkpath(window_dir)

    quantile_rows = build_quantile_rows(window, references)
    quantile_summary_rows = build_quantile_summary_rows(window, references)
    pairwise_rows = build_pairwise_distance_rows(window, references)

    save_componentwise_agreement_figure(window, references,
        joinpath(window_dir, "$(window)_reference_componentwise_agreement.png"))

    write_rows_csv(joinpath(window_dir, "$(window)_reference_quantiles.csv"),
                   ["window", "reference", "component", "nsamples", "q05", "q10", "q50", "q90", "q95"],
                   quantile_rows)
    write_rows_csv(joinpath(window_dir, "$(window)_reference_quantile_summary.csv"),
                   ["window", "component", "quantile", "min_value", "median_value", "max_value", "range_value"],
                   quantile_summary_rows)
    write_rows_csv(joinpath(window_dir, "$(window)_reference_pairwise_distances.csv"),
                   ["window", "pair", "component", "ks_distance", "wasserstein_distance", "abs_median_difference"],
                   pairwise_rows)
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

function build_quantile_rows(window::AbstractString, references)
    rows = Vector{Vector{String}}()
    for ref in references
        for (ic, comp_name) in enumerate(COMPONENT_NAMES)
            q = quantile(view(ref.y, :, ic), collect(DEFAULT_QUANTILES))
            push!(rows, [
                String(window),
                String(ref.name),
                String(comp_name),
                string(size(ref.y, 1)),
                fmt(q[1]),
                fmt(q[2]),
                fmt(q[3]),
                fmt(q[4]),
                fmt(q[5]),
            ])
        end
    end
    return rows
end

function build_quantile_summary_rows(window::AbstractString, references)
    rows = Vector{Vector{String}}()
    q_levels = collect(DEFAULT_QUANTILES)
    q_labels = ("q05", "q10", "q50", "q90", "q95")

    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        q_matrix = reduce(vcat, [reshape(quantile(view(ref.y, :, ic), q_levels), 1, :) for ref in references])
        for iq in eachindex(q_labels)
            values = q_matrix[:, iq]
            push!(rows, [
                String(window),
                String(comp_name),
                String(q_labels[iq]),
                fmt(minimum(values)),
                fmt(median(values)),
                fmt(maximum(values)),
                fmt(maximum(values) - minimum(values)),
            ])
        end
    end

    return rows
end

function build_pairwise_distance_rows(window::AbstractString, references)
    rows = Vector{Vector{String}}()
    for i in 1:length(references)-1
        for j in i+1:length(references)
            ref_a = references[i]
            ref_b = references[j]
            pair_label = "$(ref_a.name)_vs_$(ref_b.name)"
            for (ic, comp_name) in enumerate(COMPONENT_NAMES)
                a = view(ref_a.y, :, ic)
                b = view(ref_b.y, :, ic)
                push!(rows, [
                    String(window),
                    pair_label,
                    String(comp_name),
                    fmt(ks_distance(a, b)),
                    fmt(wasserstein_1d(a, b)),
                    fmt(abs(median(a) - median(b))),
                ])
            end
        end
    end
    return rows
end

function save_componentwise_agreement_figure(window::AbstractString, references, output_path::AbstractString)
    fig = Figure(size = (1650, 950), fontsize = 18)
    colors = Makie.wong_colors()

    for (ic, label) in enumerate(COMPONENT_LABELS)
        ax = Axis(fig[1, ic],
                  xlabel = label,
                  ylabel = "Empirical CDF",
                  title = "ECDF comparison: $(COMPONENT_NAMES[ic])")
        for (ir, ref) in enumerate(references)
            x, y = ecdf_curve(view(ref.y, :, ic))
            stairs!(ax, x, y;
                    step = :post,
                    color = colors[mod1(ir, length(colors))],
                    linewidth = 2.0,
                    label = String(ref.name))
        end
        axislegend(ax, position = :rb)
    end

    for (ic, label) in enumerate(COMPONENT_LABELS)
        ax = Axis(fig[2, ic],
                  xlabel = label,
                  ylabel = "Exceedance probability",
                  title = "Exceedance comparison: $(COMPONENT_NAMES[ic])",
                  yscale = log10)
        for (ir, ref) in enumerate(references)
            x, y = exceedance_curve(view(ref.y, :, ic))
            stairs!(ax, x, y; step = :post, color = colors[mod1(ir, length(colors))], linewidth = 2.0)
        end
        ylims!(ax, 1e-4, 1.0)
    end

    Label(fig[0, :], "$window reference agreement check: componentwise ECDF and exceedance", fontsize = 22, font = :bold)
    save(output_path, fig)
end

function ecdf_curve(values)
    x = sort(collect(values))
    n = length(x)
    y = collect(1:n) ./ n
    return x, y
end

function exceedance_curve(values)
    x = sort(collect(values))
    n = length(x)
    y = 1 .- ((1:n) ./ n)
    y = max.(y, 1 / n)
    return x, y
end

function ks_distance(a, b)
    x = merge_sorted_vectors(sort(collect(a)), sort(collect(b)))
    cdf_a = ecdf_values(sort(collect(a)), x)
    cdf_b = ecdf_values(sort(collect(b)), x)
    return maximum(abs.(cdf_a .- cdf_b))
end

function ecdf_values(sorted_sample::Vector{Float64}, xgrid::Vector{Float64})
    n = length(sorted_sample)
    values = zeros(Float64, length(xgrid))
    idx = 1
    for (i, x) in enumerate(xgrid)
        while idx <= n && sorted_sample[idx] <= x
            idx += 1
        end
        values[i] = (idx - 1) / n
    end
    return values
end

function wasserstein_1d(a, b)
    x = sort(collect(a))
    y = sort(collect(b))
    support = merge_sorted_vectors(x, y)
    length(support) == 1 && return 0.0

    fx = ecdf_values(x, support)
    fy = ecdf_values(y, support)
    widths = diff(support)
    return sum(abs.(fx[1:end-1] .- fy[1:end-1]) .* widths)
end

function merge_sorted_vectors(x::Vector{Float64}, y::Vector{Float64})
    values = sort(vcat(x, y))
    isempty(values) && return values
    keep = trues(length(values))
    for i in 2:length(values)
        keep[i] = values[i] != values[i - 1]
    end
    return values[keep]
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
