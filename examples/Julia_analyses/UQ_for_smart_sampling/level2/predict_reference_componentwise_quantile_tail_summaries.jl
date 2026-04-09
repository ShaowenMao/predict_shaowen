"""
Step-5 componentwise quantile and tail summaries for pooled rigorous references.

This script follows the marginal mode-screening step and packages the
componentwise nonparametric quantile and tail summaries for a pooled
reference ensemble. For each selected window, it pools the saved
reference ensembles R1/R2/R3 and builds in log10(k) space:

1. Quantile summaries for log10(kxx), log10(kyy), log10(kzz)
2. Full empirical ECDF and exceedance curves for pooled and per-reference data
3. Tail summaries on a fixed threshold grid for pooled and per-reference data
4. A compact ECDF/exceedance figure for reporting

The underlying reference ensembles remain the source of truth. This script
creates derived componentwise summaries that can be used before the later
bootstrap step.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_quantile_tail_summaries.jl
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_quantile_tail_summaries.jl --windows famp1
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
const DEFAULT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_componentwise_quantile_tail_summaries")),
        "windows" => "",
        "threshold-min" => "-7.0",
        "threshold-max" => "1.0",
        "num-thresholds" => "65",
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
        threshold_min = parse(Float64, options["threshold-min"]),
        threshold_max = parse(Float64, options["threshold-max"]),
        num_thresholds = parse(Int, options["num-thresholds"]),
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_componentwise_quantile_tail_summaries.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>         Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>       Folder where step-5 outputs are saved")
    println("  --windows <names>         Comma-separated list like famp1,famp2")
    println("  --threshold-min <x>       Minimum threshold on log10(k) grid (default: -7)")
    println("  --threshold-max <x>       Maximum threshold on log10(k) grid (default: 1)")
    println("  --num-thresholds <n>      Number of threshold-grid points (default: 65)")
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
        process_window(window, reference_dir, opt.output_dir;
                       threshold_min = opt.threshold_min,
                       threshold_max = opt.threshold_max,
                       num_thresholds = opt.num_thresholds)
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
                        threshold_min::Float64,
                        threshold_max::Float64,
                        num_thresholds::Int)
    references = load_references(reference_dir)
    isempty(references) && error("No reference MAT files found in $reference_dir")
    pooled = pool_references(references)
    thresholds = collect(range(threshold_min, threshold_max; length = num_thresholds))

    window_dir = joinpath(output_dir, window)
    mkpath(window_dir)

    save_quantile_tail_figure(window, references, pooled,
        joinpath(window_dir, "$(window)_componentwise_quantile_tail_summaries.png"))

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_quantile_summary.csv"),
                   quantile_summary_header(),
                   build_quantile_summary_rows(window, pooled))

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_empirical_curves.csv"),
                   empirical_curve_header(),
                   build_empirical_curve_rows(window, references, pooled))

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_tail_thresholds.csv"),
                   tail_threshold_header(references),
                   build_tail_threshold_rows(window, references, pooled, thresholds))

    write_rows_csv(joinpath(window_dir, "$(window)_componentwise_quantile_tail_metadata.csv"),
                   ["window", "num_references", "samples_per_reference", "pooled_samples",
                    "threshold_min", "threshold_max", "num_thresholds"],
                   [[String(window),
                     string(length(references)),
                     join([string(size(ref.y, 1)) for ref in references], ";"),
                     string(size(pooled.y, 1)),
                     fmt(threshold_min),
                     fmt(threshold_max),
                     string(num_thresholds)]])
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

function quantile_summary_header()
    return ["window", "component", "nsamples", "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99",
            "spread_q95_q05", "spread_q90_q10", "tail_left_q01", "tail_right_q99"]
end

function build_quantile_summary_rows(window::AbstractString, pooled)
    rows = Vector{Vector{String}}()
    qlevels = collect(DEFAULT_QUANTILES)
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        values = view(pooled.y, :, ic)
        q = quantile(values, qlevels)
        push!(rows, [
            String(window),
            String(comp_name),
            string(length(values)),
            fmt(q[1]),
            fmt(q[2]),
            fmt(q[3]),
            fmt(q[4]),
            fmt(q[5]),
            fmt(q[6]),
            fmt(q[7]),
            fmt(q[8]),
            fmt(q[9]),
            fmt(q[8] - q[2]),
            fmt(q[7] - q[3]),
            fmt(q[1]),
            fmt(q[9]),
        ])
    end
    return rows
end

function empirical_curve_header()
    return ["window", "component", "ensemble", "sample_index", "log10_value", "ecdf", "exceedance"]
end

function build_empirical_curve_rows(window::AbstractString, references, pooled)
    rows = Vector{Vector{String}}()
    ensembles = vcat([(name = ref.name, y = ref.y) for ref in references], [(name = "pooled", y = pooled.y)])
    for ensemble in ensembles
        for (ic, comp_name) in enumerate(COMPONENT_NAMES)
            x, ecdf_vals = ecdf_curve(view(ensemble.y, :, ic))
            _, exceed_vals = exceedance_curve(view(ensemble.y, :, ic))
            for i in eachindex(x)
                push!(rows, [
                    String(window),
                    String(comp_name),
                    String(ensemble.name),
                    string(i),
                    fmt(x[i]),
                    fmt(ecdf_vals[i]),
                    fmt(exceed_vals[i]),
                ])
            end
        end
    end
    return rows
end

function tail_threshold_header(references)
    header = ["window", "component", "threshold_log10k", "pooled_exceedance"]
    append!(header, [String(ref.name) * "_exceedance" for ref in references])
    append!(header, ["reference_exceedance_min", "reference_exceedance_max"])
    return header
end

function build_tail_threshold_rows(window::AbstractString, references, pooled, thresholds::Vector{Float64})
    rows = Vector{Vector{String}}()
    for (ic, comp_name) in enumerate(COMPONENT_NAMES)
        pooled_values = view(pooled.y, :, ic)
        ref_values = [view(ref.y, :, ic) for ref in references]
        for threshold in thresholds
            pooled_ex = exceedance_at_threshold(pooled_values, threshold)
            ref_ex = [exceedance_at_threshold(values, threshold) for values in ref_values]
            row = [String(window), String(comp_name), fmt(threshold), fmt(pooled_ex)]
            append!(row, [fmt(x) for x in ref_ex])
            append!(row, [fmt(minimum(ref_ex)), fmt(maximum(ref_ex))])
            push!(rows, row)
        end
    end
    return rows
end

function save_quantile_tail_figure(window::AbstractString, references, pooled, output_path::AbstractString)
    fig = Figure(size = (1850, 980), fontsize = 18, backgroundcolor = :white)
    colors = Makie.wong_colors()
    pooled_color = RGBf(0.05, 0.05, 0.05)
    common_xticks = [-7, -5, -3, -1, 1]
    common_xlims = (-7.0, 1.0)

    for (ic, label) in enumerate(COMPONENT_LABELS)
        ax = Axis(fig[1, ic],
                  xlabel = label,
                  ylabel = (ic == 1 ? "Empirical CDF" : ""),
                  title = "$(COMPONENT_NAMES[ic]): ECDF",
                  xticks = common_xticks,
                  xgridcolor = RGBAf(0, 0, 0, 0.08),
                  ygridcolor = RGBAf(0, 0, 0, 0.08),
                  xlabelsize = 20,
                  ylabelsize = 20,
                  titlesize = 21,
                  xticklabelsize = 18,
                  yticklabelsize = 18,
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false,
                  rightspinevisible = false)
        for (ir, ref) in enumerate(references)
            x_ref, y_ref = ecdf_curve(view(ref.y, :, ic))
            lines!(ax, x_ref, y_ref; color = (colors[mod1(ir, length(colors))], 0.40), linewidth = 1.6)
        end
        x_pool, y_pool = ecdf_curve(view(pooled.y, :, ic))
        stairs!(ax, x_pool, y_pool; step = :post, color = pooled_color, linewidth = 2.8, label = "pooled")
        xlims!(ax, common_xlims...)
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ic == 1
            axislegend(ax, position = :rb, framevisible = true, labelsize = 16)
        end
    end

    for (ic, label) in enumerate(COMPONENT_LABELS)
        ax = Axis(fig[2, ic],
                  xlabel = label,
                  ylabel = (ic == 1 ? "Exceedance probability" : ""),
                  title = "$(COMPONENT_NAMES[ic]): tail profile",
                  xticks = common_xticks,
                  yscale = log10,
                  xgridcolor = RGBAf(0, 0, 0, 0.08),
                  ygridcolor = RGBAf(0, 0, 0, 0.08),
                  xlabelsize = 20,
                  ylabelsize = 20,
                  titlesize = 21,
                  xticklabelsize = 18,
                  yticklabelsize = 18,
                  leftspinecolor = :black,
                  bottomspinecolor = :black,
                  topspinevisible = false,
                  rightspinevisible = false)
        for (ir, ref) in enumerate(references)
            x_ref, y_ref = exceedance_curve(view(ref.y, :, ic))
            lines!(ax, x_ref, y_ref; color = (colors[mod1(ir, length(colors))], 0.40), linewidth = 1.6)
        end
        x_pool, y_pool = exceedance_curve(view(pooled.y, :, ic))
        stairs!(ax, x_pool, y_pool; step = :post, color = pooled_color, linewidth = 2.8, label = "pooled")
        xlims!(ax, common_xlims...)
        ylims!(ax, 1e-5, 1.0)
        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('d' + ic - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ic == 1
            axislegend(ax, position = :lb, framevisible = true, labelsize = 16)
        end
    end

    Label(fig[0, :], "$window componentwise quantile and tail summaries in log10(k) space", fontsize = 24, font = :bold)
    Label(fig[3, :], "Thin colored curves: independent references. Thick black curve: pooled empirical estimator.", fontsize = 18)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
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

function exceedance_at_threshold(values, threshold::Float64)
    return count(>(threshold), values) / length(values)
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
