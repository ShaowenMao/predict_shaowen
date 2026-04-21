#!/usr/bin/env julia

"""
Joint agreement check for rigorous PREDICT reference ensembles.

This script is the first joint-analysis step in the rigorous Level 2 workflow.
It reads the independently generated reference ensembles R1, R2, R3 for each
selected GOM window from the `gom_reference_floor_full` outputs and checks
whether the references agree well enough jointly before pooling them.

For each selected window it:
1. Loads all `reference_R*.mat` files from `data/<window>/references`.
2. Transforms permeability samples to
       y = (log10(kxx), log10(kyy), log10(kzz)).
3. Produces publication-style pairwise projection plots with consistent axes
   and overlaid 2D histogram contours for R1/R2/R3.
4. Computes compact joint agreement diagnostics for each reference pair:
       - 3D sliced Wasserstein distance
       - Euclidean distance between mean vectors
       - Frobenius norm of covariance-matrix difference
       - Frobenius norm of correlation-matrix difference
5. Saves per-reference joint summaries and pairwise metric tables.
6. Produces a compact metric figure for the reference-pair diagnostics.

This script is intentionally diagnostic: it is meant to answer
    "Do the independent reference ensembles agree well enough jointly to pool?"

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_agreement.jl
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_agreement.jl --windows famp1
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_agreement.jl --data-dir path\\to\\data --output-dir path\\to\\out
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
using LinearAlgebra
using Random
using Printf

CairoMakie.activate!()

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")
const PAIRWISE_COMPONENTS = ((1, 2), (1, 3), (2, 3))
const PAIRWISE_LABELS = ("kxx-kyy", "kxx-kzz", "kyy-kzz")
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_joint_agreement")),
        "windows" => "",
        "plot-max-points" => "2500",
        "hist-bins" => "45",
        "num-slices" => "96",
        "seed" => "1729",
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
        plot_max_points = parse(Int, options["plot-max-points"]),
        hist_bins = parse(Int, options["hist-bins"]),
        num_slices = parse(Int, options["num-slices"]),
        seed = parse(Int, options["seed"]),
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_agreement.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>         Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>       Folder where figures and CSVs are saved")
    println("  --windows <names>         Comma-separated list like famp1,famp2")
    println("  --plot-max-points <n>     Max plotted points per reference in projection figure (default: 2500)")
    println("  --hist-bins <n>           Number of 2D histogram bins per axis (default: 45)")
    println("  --num-slices <n>          Number of random directions for sliced Wasserstein (default: 96)")
    println("  --seed <n>                Random seed (default: 1729)")
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
        process_window(window, reference_dir, opt)
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

function process_window(window::AbstractString, reference_dir::AbstractString, opt)
    references = load_references(reference_dir)
    isempty(references) && error("No reference MAT files found in $reference_dir")

    pooled = reduce(vcat, [ref.y for ref in references])
    window_dir = joinpath(opt.output_dir, String(window))
    mkpath(window_dir)

    summary_rows = build_reference_summary_rows(window, references)
    pairwise_rows = build_pairwise_joint_rows(window, references; num_slices = opt.num_slices, seed = opt.seed)

    save_joint_projection_figure(window, references, pooled,
        joinpath(window_dir, "$(window)_reference_joint_projections.png");
        plot_max_points = opt.plot_max_points,
        hist_bins = opt.hist_bins,
        seed = opt.seed)
    save_joint_metric_figure(window, references, pairwise_rows,
        joinpath(window_dir, "$(window)_reference_joint_metrics.png"))

    write_rows_csv(joinpath(window_dir, "$(window)_reference_joint_summary.csv"),
                   reference_summary_header(),
                   summary_rows)
    write_rows_csv(joinpath(window_dir, "$(window)_reference_joint_pairwise_metrics.csv"),
                   pairwise_joint_header(),
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

function reference_summary_header()
    return ["window", "reference", "nsamples",
            "mean_kxx", "mean_kyy", "mean_kzz",
            "std_kxx", "std_kyy", "std_kzz",
            "corr_kxx_kyy", "corr_kxx_kzz", "corr_kyy_kzz"]
end

function build_reference_summary_rows(window::AbstractString, references)
    rows = Vector{Vector{String}}()
    for ref in references
        μ = vec(mean(ref.y; dims = 1))
        σ = vec(std(ref.y; dims = 1, corrected = true))
        C = cor(ref.y)
        push!(rows, [
            String(window),
            String(ref.name),
            string(size(ref.y, 1)),
            fmt(μ[1]), fmt(μ[2]), fmt(μ[3]),
            fmt(σ[1]), fmt(σ[2]), fmt(σ[3]),
            fmt(C[1, 2]), fmt(C[1, 3]), fmt(C[2, 3]),
        ])
    end
    return rows
end

function pairwise_joint_header()
    return ["window", "pair",
            "sliced_wasserstein_3d",
            "mean_shift_norm",
            "covariance_frobenius_diff",
            "correlation_frobenius_diff"]
end

function build_pairwise_joint_rows(window::AbstractString, references; num_slices::Int = 96, seed::Int = 1729)
    rng = MersenneTwister(seed + sum(codeunits(String(window))))
    directions = random_unit_directions(3, num_slices, rng)
    rows = Vector{Vector{String}}()
    for i in 1:length(references)-1
        for j in i+1:length(references)
            ref_a = references[i]
            ref_b = references[j]
            μa = vec(mean(ref_a.y; dims = 1))
            μb = vec(mean(ref_b.y; dims = 1))
            Σa = cov(ref_a.y)
            Σb = cov(ref_b.y)
            Ca = cor(ref_a.y)
            Cb = cor(ref_b.y)
            push!(rows, [
                String(window),
                "$(ref_a.name)_vs_$(ref_b.name)",
                fmt(sliced_wasserstein(ref_a.y, ref_b.y, directions)),
                fmt(norm(μa - μb)),
                fmt(norm(Σa - Σb)),
                fmt(norm(Ca - Cb)),
            ])
        end
    end
    return rows
end

function random_unit_directions(dim::Int, n::Int, rng::AbstractRNG)
    dirs = zeros(Float64, dim, n)
    for i in 1:n
        v = randn(rng, dim)
        dirs[:, i] .= v ./ norm(v)
    end
    return dirs
end

function sliced_wasserstein(a::Matrix{Float64}, b::Matrix{Float64}, directions::Matrix{Float64})
    vals = zeros(Float64, size(directions, 2))
    for i in 1:size(directions, 2)
        u = view(directions, :, i)
        pa = vec(a * u)
        pb = vec(b * u)
        vals[i] = wasserstein_1d(pa, pb)
    end
    return mean(vals)
end

function wasserstein_1d(a::Vector{Float64}, b::Vector{Float64})
    x = sort(copy(a))
    y = sort(copy(b))
    support = merge_sorted_vectors(x, y)
    length(support) <= 1 && return 0.0
    fx = ecdf_values(x, support)
    fy = ecdf_values(y, support)
    widths = diff(support)
    return sum(abs.(fx[1:end-1] .- fy[1:end-1]) .* widths)
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

function merge_sorted_vectors(x::Vector{Float64}, y::Vector{Float64})
    values = sort(vcat(x, y))
    isempty(values) && return values
    keep = trues(length(values))
    for i in 2:length(values)
        keep[i] = values[i] != values[i - 1]
    end
    return values[keep]
end

function save_joint_projection_figure(window::AbstractString, references, pooled::Matrix{Float64}, output_path::AbstractString;
                                      plot_max_points::Int = 2500,
                                      hist_bins::Int = 45,
                                      seed::Int = 1729)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    colors = [Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]]
    common_ticks = ([-7.0, -5.0, -3.0, -1.0, 1.0], ["-7", "-5", "-3", "-1", "1"])
    common_lims = (-7.0, 1.0)
    legend_handles = Any[]
    legend_labels = String[]
    rng = MersenneTwister(seed + 101)

    for (ip, (a, b)) in enumerate(PAIRWISE_COMPONENTS)
        ax = Axis(fig[1, ip],
                  xlabel = COMPONENT_LABELS[a],
                  ylabel = (ip == 1 ? COMPONENT_LABELS[b] : ""),
                  title = "Joint projection: $(PAIRWISE_LABELS[ip])",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  xticks = common_ticks, yticks = common_ticks,
                  xgridcolor = RGBAf(0, 0, 0, 0.08), ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black, bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        xlims!(ax, common_lims...)
        ylims!(ax, common_lims...)

        xedges = collect(range(common_lims[1], common_lims[2]; length = hist_bins + 1))
        yedges = collect(range(common_lims[1], common_lims[2]; length = hist_bins + 1))
        xc = 0.5 .* (xedges[1:end-1] .+ xedges[2:end])
        yc = 0.5 .* (yedges[1:end-1] .+ yedges[2:end])

        for (ir, ref) in enumerate(references)
            idx = downsample_indices(size(ref.y, 1), plot_max_points, rng)
            scatter!(ax, ref.y[idx, a], ref.y[idx, b],
                     color = (colors[ir], 0.08), markersize = 4)

            density = hist2d_density(ref.y[:, a], ref.y[:, b], xedges, yedges)
            positive = density[density .> 0]
            if !isempty(positive)
                levels = [quantile(positive, 0.70), quantile(positive, 0.90)]
                contour!(ax, xc, yc, density';
                         levels = unique(levels),
                         color = colors[ir],
                         linewidth = 2.2)
            end

            μ = vec(mean(ref.y; dims = 1))
            mean_plot = scatter!(ax, [μ[a]], [μ[b]], color = colors[ir], marker = :diamond, markersize = 16,
                                 strokecolor = :black, strokewidth = 0.8)
            if ip == 1
                push!(legend_handles, mean_plot)
                push!(legend_labels, String(ref.name))
            end
        end

        text!(ax, 0.98, 0.98; space = :relative, text = string("(", Char('a' + ip - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ip == 1
            Legend(fig[1, 1], legend_handles, legend_labels,
                   orientation = :vertical, framevisible = true, labelsize = 22,
                   tellheight = false, tellwidth = false, halign = :left, valign = :bottom,
                   margin = (18, 18, 18, 18), padding = (10, 10, 10, 10))
        end
    end

    Label(fig[0, :], "$window joint reference agreement in log10(k) space", fontsize = 24, font = :bold)
    Label(fig[2, :], "Faint points: subsampled joint samples. Contours: 2D histogram density levels. Diamonds: reference means.", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function hist2d_density(x::Vector{Float64}, y::Vector{Float64}, xedges::Vector{Float64}, yedges::Vector{Float64})
    counts = zeros(Float64, length(xedges) - 1, length(yedges) - 1)
    for i in eachindex(x)
        bx = searchsortedlast(xedges, x[i])
        by = searchsortedlast(yedges, y[i])
        if 1 <= bx < length(xedges) && 1 <= by < length(yedges)
            counts[bx, by] += 1.0
        elseif x[i] == xedges[end] && y[i] == yedges[end]
            counts[end, end] += 1.0
        end
    end
    dx = xedges[2] - xedges[1]
    dy = yedges[2] - yedges[1]
    return counts ./ (sum(counts) * dx * dy)
end

function downsample_indices(n::Int, max_points::Int, rng::AbstractRNG)
    n <= max_points && return collect(1:n)
    idx = randperm(rng, n)[1:max_points]
    sort!(idx)
    return idx
end

function save_joint_metric_figure(window::AbstractString, references, pairwise_rows, output_path::AbstractString)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    bar_color = RGBAf(0.42, 0.79, 0.75, 0.92)
    pairs = [row[2] for row in pairwise_rows]
    metrics = (
        ("3D sliced Wasserstein", 3),
        ("Mean-shift norm", 4),
        ("Covariance/correlation difference", 5),
    )

    # third panel uses covariance and correlation bars side-by-side
    ax1 = Axis(fig[1, 1], xlabel = "Reference pair", ylabel = "Distance",
               title = "Joint agreement: sliced Wasserstein",
               xlabelsize = 22, ylabelsize = 22, titlesize = 22,
               xticklabelsize = 18, yticklabelsize = 22,
               xticks = (1:length(pairs), pairs),
               xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
               leftspinecolor = :black, bottomspinecolor = :black,
               topspinevisible = false, rightspinevisible = false)
    vals1 = [parse(Float64, row[3]) for row in pairwise_rows]
    barplot!(ax1, 1:length(pairs), vals1, color = bar_color, strokecolor = :black, strokewidth = 0.7)
    for (i, v) in enumerate(vals1)
        text!(ax1, i, v, text = @sprintf("%.3f", v), align = (:center, :bottom), fontsize = 18, color = :black)
    end
    text!(ax1, 0.98, 0.98; space = :relative, text = "(a)", align = (:right, :top), fontsize = 22, font = :bold, color = :black)

    ax2 = Axis(fig[1, 2], xlabel = "Reference pair", ylabel = "Distance",
               title = "Joint agreement: mean shift",
               xlabelsize = 22, ylabelsize = 22, titlesize = 22,
               xticklabelsize = 18, yticklabelsize = 22,
               xticks = (1:length(pairs), pairs),
               xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
               leftspinecolor = :black, bottomspinecolor = :black,
               topspinevisible = false, rightspinevisible = false)
    vals2 = [parse(Float64, row[4]) for row in pairwise_rows]
    barplot!(ax2, 1:length(pairs), vals2, color = bar_color, strokecolor = :black, strokewidth = 0.7)
    for (i, v) in enumerate(vals2)
        text!(ax2, i, v, text = @sprintf("%.3f", v), align = (:center, :bottom), fontsize = 18, color = :black)
    end
    text!(ax2, 0.98, 0.98; space = :relative, text = "(b)", align = (:right, :top), fontsize = 22, font = :bold, color = :black)

    ax3 = Axis(fig[1, 3], xlabel = "Reference pair", ylabel = "Distance",
               title = "Joint agreement: dependence structure",
               xlabelsize = 22, ylabelsize = 22, titlesize = 22,
               xticklabelsize = 18, yticklabelsize = 22,
               xticks = (1:length(pairs), pairs),
               xgridvisible = false, ygridcolor = RGBAf(0, 0, 0, 0.08),
               leftspinecolor = :black, bottomspinecolor = :black,
               topspinevisible = false, rightspinevisible = false)
    vals3a = [parse(Float64, row[5]) for row in pairwise_rows]
    vals3b = [parse(Float64, row[6]) for row in pairwise_rows]
    barplot!(ax3, (1:length(pairs)) .- 0.16, vals3a, color = RGBAf(0.42, 0.79, 0.75, 0.92), width = 0.28,
             strokecolor = :black, strokewidth = 0.7, label = "covariance")
    barplot!(ax3, (1:length(pairs)) .+ 0.16, vals3b, color = RGBAf(0.10, 0.33, 0.65, 0.88), width = 0.28,
             strokecolor = :black, strokewidth = 0.7, label = "correlation")
    text!(ax3, 0.98, 0.98; space = :relative, text = "(c)", align = (:right, :top), fontsize = 22, font = :bold, color = :black)
    axislegend(ax3, position = :lt, framevisible = true, labelsize = 20,
               backgroundcolor = RGBAf(1, 1, 1, 0.92), patchlabelgap = 8, rowgap = 5)

    Label(fig[0, :], "$window joint reference agreement metrics", fontsize = 24, font = :bold)
    Label(fig[2, :], "Smaller values indicate better agreement. Third panel compares covariance and correlation matrix differences.", fontsize = 22)
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
