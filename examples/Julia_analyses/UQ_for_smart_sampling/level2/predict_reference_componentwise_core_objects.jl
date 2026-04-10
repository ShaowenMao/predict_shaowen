#!/usr/bin/env julia

"""
Pooled componentwise nonparametric Level 2 objects for rigorous references.

This script assumes the independent reference ensembles for a given window
have already been checked for marginal agreement. It then pools the saved
reference ensembles R1/R2/R3 and builds the core componentwise
nonparametric objects in log10(k) space:

1. Empirical CDFs for log10(kxx), log10(kyy), log10(kzz)
2. Exceedance curves for the same three components
3. Quantiles and interquantile spreads from the pooled sample

The raw reference ensembles remain the primary saved source of truth. This
script creates pooled derived products for the next step of Level 2
analysis.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_core_objects.jl
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_core_objects.jl --windows famp1
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_core_objects.jl --data-dir path\\to\\data --output-dir path\\to\\out
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
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_core_objects")),
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
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_core_objects.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>         Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>       Folder where pooled figures and CSVs are saved")
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

    pooled = pool_references(references)
    window_dir = joinpath(output_dir, window)
    mkpath(window_dir)

    save_componentwise_core_figure(window, references, pooled,
        joinpath(window_dir, "$(window)_pooled_componentwise_core_objects.png"))

    write_rows_csv(joinpath(window_dir, "$(window)_pooled_componentwise_quantiles.csv"),
                   ["window", "component", "nsamples", "q05", "q10", "q50", "q90", "q95", "iqr_05_95"],
                   build_quantile_rows(window, pooled))

    write_rows_csv(joinpath(window_dir, "$(window)_pooled_componentwise_curves.csv"),
                   ["window", "component", "sample_index", "log10_value", "ecdf", "exceedance"],
                   build_curve_rows(window, pooled))

    write_rows_csv(joinpath(window_dir, "$(window)_pooled_componentwise_metadata.csv"),
                   ["window", "num_references", "samples_per_reference", "total_samples"],
                   [[String(window), string(length(references)),
                     join([string(size(ref.y, 1)) for ref in references], ";"),
                     string(size(pooled.y, 1))]])
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
    y = reduce(vcat, [ref.y for ref in references])
    perms = reduce(vcat, [ref.perms for ref in references])
    return (perms = perms, y = y)
end

function build_quantile_rows(window::AbstractString, pooled)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        values = view(pooled.y, :, ic)
        q = quantile(values, collect(DEFAULT_QUANTILES))
        push!(rows, [
            String(window),
            String(comp_name),
            string(size(pooled.y, 1)),
            fmt(q[1]),
            fmt(q[2]),
            fmt(q[3]),
            fmt(q[4]),
            fmt(q[5]),
            fmt(q[5] - q[1]),
        ])
    end
    return rows
end

function build_curve_rows(window::AbstractString, pooled)
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        x, ecdf_vals = ecdf_curve(view(pooled.y, :, ic))
        _, exceed_vals = exceedance_curve(view(pooled.y, :, ic))
        for i in eachindex(x)
            push!(rows, [
                String(window),
                String(comp_name),
                string(i),
                fmt(x[i]),
                fmt(ecdf_vals[i]),
                fmt(exceed_vals[i]),
            ])
        end
    end
    return rows
end

function save_componentwise_core_figure(window::AbstractString, references, pooled, output_path::AbstractString)
    fig = Figure(size = (1650, 950), fontsize = 18)
    colors = Makie.wong_colors()
    pooled_color = :black

    for (ic, label) in enumerate(COMPONENT_LABELS)
        ax = Axis(fig[1, ic],
                  xlabel = label,
                  ylabel = "Empirical CDF",
                  title = "Pooled ECDF: $(COMPONENT_NAMES[ic])")
        for (ir, ref) in enumerate(references)
            x_ref, y_ref = ecdf_curve(view(ref.y, :, ic))
            lines!(ax, x_ref, y_ref; color = (colors[mod1(ir, length(colors))], 0.35), linewidth = 1.4)
        end
        x_pool, y_pool = ecdf_curve(view(pooled.y, :, ic))
        stairs!(ax, x_pool, y_pool; step = :post, color = pooled_color, linewidth = 2.4, label = "Pooled")
        axislegend(ax, position = :rb)
    end

    for (ic, label) in enumerate(COMPONENT_LABELS)
        ax = Axis(fig[2, ic],
                  xlabel = label,
                  ylabel = "Exceedance probability",
                  title = "Pooled exceedance: $(COMPONENT_NAMES[ic])",
                  yscale = log10)
        for (ir, ref) in enumerate(references)
            x_ref, y_ref = exceedance_curve(view(ref.y, :, ic))
            lines!(ax, x_ref, y_ref; color = (colors[mod1(ir, length(colors))], 0.35), linewidth = 1.4)
        end
        x_pool, y_pool = exceedance_curve(view(pooled.y, :, ic))
        stairs!(ax, x_pool, y_pool; step = :post, color = pooled_color, linewidth = 2.4, label = "Pooled")
        ylims!(ax, 1e-5, 1.0)
        axislegend(ax, position = :lb)
    end

    Label(fig[0, :], "$window pooled componentwise nonparametric objects", fontsize = 22, font = :bold)
    Label(fig[3, :], "Colored thin lines = individual references; thick black line = pooled empirical estimator", fontsize = 15)
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
