#!/usr/bin/env julia

"""
Joint step-2 core objects for pooled rigorous references across multiple windows.

This script builds the core pooled joint objects that sit between the initial
reference-agreement check and any later joint regime / allocation analysis.
It is designed to compare multiple windows together using the pooled rigorous
reference samples R1 ∪ R2 ∪ R3 for each window.

For each selected window it:
1. Loads and pools the rigorous references in log10(k) space:
       Y = (log10(kxx), log10(kyy), log10(kzz))
2. Computes pooled pairwise dependence summaries:
       - Pearson correlation
       - Spearman correlation
       - covariance
3. Computes anisotropy contrasts:
       A12 = Y1 - Y2
       A13 = Y1 - Y3
       A23 = Y2 - Y3
   and summarizes their means, spreads, and quantiles.
4. Builds a shared anisotropy threshold grid from the pooled multi-window
   library, to support later joint-feature selection.
5. Produces two cross-window figures:
       - pooled pairwise joint projections (rows = windows, cols = component pairs)
       - anisotropy ECDF comparison (one panel per contrast)

This step is still exploratory/diagnostic. It does not yet define the final
shared joint feature dictionary, but it provides the empirical basis for doing so.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_core_objects.jl --windows famp1,famp2,famp3
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
using Random
using Printf

CairoMakie.activate!()

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")
const PAIRWISE_COMPONENTS = ((1, 2), (1, 3), (2, 3))
const PAIRWISE_LABELS = ("kxx-kyy", "kxx-kzz", "kyy-kzz")
const ANISO_NAMES = ("A12", "A13", "A23")
const ANISO_LABELS = ("log10(kxx/kyy)", "log10(kxx/kzz)", "log10(kyy/kzz)")
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_ANISO_Q = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_reference_joint_core_objects")),
        "windows" => "",
        "plot-max-points" => "5000",
        "hist-bins" => "55",
        "seed" => "1729",
        "anisotropy-quantile-levels" => join(string.(DEFAULT_ANISO_Q), ","),
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

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        requested_windows = requested_windows,
        plot_max_points = parse(Int, options["plot-max-points"]),
        hist_bins = parse(Int, options["hist-bins"]),
        seed = parse(Int, options["seed"]),
        anisotropy_quantiles = anisotropy_quantiles,
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_joint_core_objects.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>                   Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>                 Folder where cross-window figures and CSVs are saved")
    println("  --windows <names>                   Comma-separated list like famp1,famp2,famp3")
    println("  --plot-max-points <n>               Max pooled points per panel in projection figure (default: 5000)")
    println("  --hist-bins <n>                     Number of 2D histogram bins per axis (default: 55)")
    println("  --seed <n>                          Random seed for plotting downsampling (default: 1729)")
    println("  --anisotropy-quantile-levels <list> Shared quantile grid for anisotropy thresholds (default: 0.05,0.10,0.25,0.50,0.75,0.90,0.95)")
    println("  -h, --help                          Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No requested windows found under $(opt.data_dir)")

    pooled_data = Dict{String, Matrix{Float64}}()
    aniso_library = Dict(name => Float64[] for name in ANISO_NAMES)

    println("Loading pooled joint data for $(length(windows)) window(s)...")
    for (window, reference_dir) in windows
        references = load_references(reference_dir)
        pooled = reduce(vcat, [ref.y for ref in references])
        pooled_data[window] = pooled
        aniso = anisotropy_matrix(pooled)
        for (ia, name) in enumerate(ANISO_NAMES)
            append!(aniso_library[name], vec(aniso[:, ia]))
        end
    end

    pairwise_rows = build_pairwise_dependence_rows(opt.requested_windows, pooled_data)
    anisotropy_rows = build_anisotropy_summary_rows(opt.requested_windows, pooled_data)
    threshold_rows = build_shared_anisotropy_threshold_rows(opt.anisotropy_quantiles, aniso_library)

    save_pooled_projection_figure(opt.requested_windows, pooled_data,
        joinpath(opt.output_dir, "joint_pooled_pairwise_projections.png");
        plot_max_points = opt.plot_max_points,
        hist_bins = opt.hist_bins,
        seed = opt.seed)

    save_anisotropy_ecdf_figure(opt.requested_windows, pooled_data,
        joinpath(opt.output_dir, "joint_anisotropy_ecdfs.png"))

    write_rows_csv(joinpath(opt.output_dir, "joint_pooled_pairwise_dependence.csv"),
                   pairwise_dependence_header(),
                   pairwise_rows)
    write_rows_csv(joinpath(opt.output_dir, "joint_anisotropy_summary.csv"),
                   anisotropy_summary_header(),
                   anisotropy_rows)
    write_rows_csv(joinpath(opt.output_dir, "joint_shared_anisotropy_threshold_grid.csv"),
                   anisotropy_threshold_header(),
                   threshold_rows)
    write_metadata_csv(joinpath(opt.output_dir, "joint_core_metadata.csv"), opt)

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

function pairwise_dependence_header()
    return ["window", "pair", "nsamples",
            "mean_x", "mean_y", "std_x", "std_y",
            "covariance", "pearson_correlation", "spearman_correlation"]
end

function build_pairwise_dependence_rows(windows::Vector{String}, pooled_data)
    rows = Vector{Vector{String}}()
    for window in windows
        y = pooled_data[window]
        for (ip, (a, b)) in enumerate(PAIRWISE_COMPONENTS)
            xa = vec(y[:, a])
            xb = vec(y[:, b])
            push!(rows, [
                window,
                PAIRWISE_LABELS[ip],
                string(length(xa)),
                fmt(mean(xa)),
                fmt(mean(xb)),
                fmt(std(xa; corrected = true)),
                fmt(std(xb; corrected = true)),
                fmt(cov(xa, xb)),
                fmt(cor(xa, xb)),
                fmt(spearman_corr(xa, xb)),
            ])
        end
    end
    return rows
end

function anisotropy_summary_header()
    return ["window", "contrast", "nsamples",
            "mean", "std", "q05", "q10", "q25", "q50", "q75", "q90", "q95"]
end

function build_anisotropy_summary_rows(windows::Vector{String}, pooled_data)
    rows = Vector{Vector{String}}()
    probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    for window in windows
        aniso = anisotropy_matrix(pooled_data[window])
        for (ia, name) in enumerate(ANISO_NAMES)
            vals = vec(aniso[:, ia])
            q = quantile(vals, probs)
            push!(rows, [
                window,
                name,
                string(length(vals)),
                fmt(mean(vals)),
                fmt(std(vals; corrected = true)),
                fmt(q[1]), fmt(q[2]), fmt(q[3]), fmt(q[4]), fmt(q[5]), fmt(q[6]), fmt(q[7]),
            ])
        end
    end
    return rows
end

function anisotropy_threshold_header()
    return ["contrast", "threshold_id", "quantile_level", "threshold_value", "threshold_weight", "order_id"]
end

function build_shared_anisotropy_threshold_rows(qlevels, aniso_library)
    rows = Vector{Vector{String}}()
    weight = 1.0 / length(qlevels)
    for name in ANISO_NAMES
        thresholds = quantile(aniso_library[name], collect(qlevels))
        for (i, (q, t)) in enumerate(zip(qlevels, thresholds))
            push!(rows, [
                name,
                @sprintf("%s_q%02d", name, round(Int, 100q)),
                fmt(q),
                fmt(t),
                fmt(weight),
                string(i),
            ])
        end
    end
    return rows
end

function save_pooled_projection_figure(windows::Vector{String}, pooled_data, output_path::AbstractString;
                                       plot_max_points::Int = 5000,
                                       hist_bins::Int = 55,
                                       seed::Int = 1729)
    fig = Figure(size = (1850, 1500), fontsize = 18, backgroundcolor = :white)
    common_ticks = ([-7.0, -5.0, -3.0, -1.0, 1.0], ["-7", "-5", "-3", "-1", "1"])
    common_lims = (-7.0, 1.0)
    window_colors = Dict(
        "famp1" => RGBf(0.02, 0.27, 0.58),
        "famp2" => RGBf(0.78, 0.42, 0.14),
        "famp3" => RGBf(0.12, 0.55, 0.45),
    )
    rng = MersenneTwister(seed + 311)

    for (iw, window) in enumerate(windows)
        y = pooled_data[window]
        color = get(window_colors, window, Makie.wong_colors()[mod1(iw, length(Makie.wong_colors()))])
        μ = vec(mean(y; dims = 1))

        for (ip, (a, b)) in enumerate(PAIRWISE_COMPONENTS)
            ax = Axis(fig[iw, ip],
                      xlabel = (iw == length(windows) ? COMPONENT_LABELS[a] : ""),
                      ylabel = (ip == 1 ? COMPONENT_LABELS[b] : ""),
                      title = "$(window): $(PAIRWISE_LABELS[ip])",
                      xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                      xticklabelsize = 22, yticklabelsize = 22,
                      xticks = common_ticks, yticks = common_ticks,
                      xgridcolor = RGBAf(0, 0, 0, 0.08), ygridcolor = RGBAf(0, 0, 0, 0.08),
                      leftspinecolor = :black, bottomspinecolor = :black,
                      topspinevisible = false, rightspinevisible = false)
            xlims!(ax, common_lims...)
            ylims!(ax, common_lims...)

            idx = downsample_indices(size(y, 1), plot_max_points, rng)
            scatter!(ax, y[idx, a], y[idx, b], color = (color, 0.08), markersize = 4)

            xedges = collect(range(common_lims[1], common_lims[2]; length = hist_bins + 1))
            yedges = collect(range(common_lims[1], common_lims[2]; length = hist_bins + 1))
            xc = 0.5 .* (xedges[1:end-1] .+ xedges[2:end])
            yc = 0.5 .* (yedges[1:end-1] .+ yedges[2:end])
            density = hist2d_density(y[:, a], y[:, b], xedges, yedges)
            positive = density[density .> 0]
            if !isempty(positive)
                levels = unique([quantile(positive, 0.70), quantile(positive, 0.90)])
                contour!(ax, xc, yc, density'; levels = levels, color = color, linewidth = 2.4)
            end

            scatter!(ax, [μ[a]], [μ[b]], color = :black, marker = :diamond, markersize = 15)
            ρ = cor(y[:, a], y[:, b])
            ρs = spearman_corr(vec(y[:, a]), vec(y[:, b]))
            text!(ax, 0.03, 0.96; space = :relative,
                  text = @sprintf("ρ = %.3f\nρs = %.3f", ρ, ρs),
                  align = (:left, :top), fontsize = 18, color = :black)

            panel_index = (iw - 1) * length(PAIRWISE_COMPONENTS) + ip
            text!(ax, 0.98, 0.98; space = :relative,
                  text = string("(", Char('a' + panel_index - 1), ")"),
                  align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        end
    end

    Label(fig[0, :], "Pooled joint projections across famp1, famp2, and famp3", fontsize = 24, font = :bold)
    Label(fig[length(windows) + 1, :], "Faint points: pooled joint samples. Contours: pooled 2D histogram density levels. Black diamonds: pooled means.", fontsize = 22)
    save(output_path, fig)
    root, ext = splitext(output_path)
    lowercase(ext) == ".png" && save(root * ".pdf", fig)
end

function save_anisotropy_ecdf_figure(windows::Vector{String}, pooled_data, output_path::AbstractString)
    fig = Figure(size = (1850, 640), fontsize = 18, backgroundcolor = :white)
    window_colors = Dict(
        "famp1" => RGBf(0.02, 0.27, 0.58),
        "famp2" => RGBf(0.78, 0.42, 0.14),
        "famp3" => RGBf(0.12, 0.55, 0.45),
    )

    legend_handles = Any[]
    legend_labels = String[]
    all_aniso = Dict(name => Float64[] for name in ANISO_NAMES)
    for window in windows
        aniso = anisotropy_matrix(pooled_data[window])
        for (ia, name) in enumerate(ANISO_NAMES)
            append!(all_aniso[name], vec(aniso[:, ia]))
        end
    end

    for (ia, name) in enumerate(ANISO_NAMES)
        vals_all = all_aniso[name]
        xmin = quantile(vals_all, 0.01)
        xmax = quantile(vals_all, 0.99)
        xpad = 0.06 * max(xmax - xmin, 1e-3)
        xgrid = collect(range(xmin - xpad, xmax + xpad; length = 800))

        ax = Axis(fig[1, ia],
                  xlabel = ANISO_LABELS[ia],
                  ylabel = (ia == 1 ? "ECDF" : ""),
                  title = "$(name): anisotropy comparison",
                  xlabelsize = 22, ylabelsize = 22, titlesize = 22,
                  xticklabelsize = 22, yticklabelsize = 22,
                  yticks = ([0.0, 0.25, 0.50, 0.75, 1.0], ["0.00", "0.25", "0.50", "0.75", "1.00"]),
                  xgridcolor = RGBAf(0, 0, 0, 0.08), ygridcolor = RGBAf(0, 0, 0, 0.08),
                  leftspinecolor = :black, bottomspinecolor = :black,
                  topspinevisible = false, rightspinevisible = false)
        ylims!(ax, 0.0, 1.0)

        for (iw, window) in enumerate(windows)
            color = get(window_colors, window, Makie.wong_colors()[mod1(iw, length(Makie.wong_colors()))])
            vals = sort(vec(anisotropy_matrix(pooled_data[window])[:, ia]))
            ecdf = ecdf_values(vals, xgrid)
            line = lines!(ax, xgrid, ecdf, color = color, linewidth = 3.0, label = window)
            if ia == 1
                push!(legend_handles, line)
                push!(legend_labels, window)
            end
        end

        text!(ax, 0.98, 0.98; space = :relative,
              text = string("(", Char('a' + ia - 1), ")"),
              align = (:right, :top), fontsize = 22, font = :bold, color = :black)
        if ia == 1
            axislegend(ax, position = :lt, framevisible = true, labelsize = 22,
                       backgroundcolor = RGBAf(1, 1, 1, 0.92), patchlabelgap = 8, rowgap = 5)
        end
    end

    Label(fig[0, :], "Pooled anisotropy ECDFs across famp1, famp2, and famp3", fontsize = 24, font = :bold)
    Label(fig[2, :], "Each curve is the pooled empirical CDF of a log-ratio contrast Y_a - Y_b for one window.", fontsize = 22)
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

function spearman_corr(x::Vector{Float64}, y::Vector{Float64})
    return cor(simple_ranks(x), simple_ranks(y))
end

function simple_ranks(x::Vector{Float64})
    p = sortperm(x)
    r = similar(x, Float64)
    for (rank, idx) in enumerate(p)
        r[idx] = rank
    end
    return r
end

function write_rows_csv(filepath::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function write_metadata_csv(filepath::AbstractString, opt)
    header = ["windows", "plot_max_points", "hist_bins", "seed", "anisotropy_quantile_levels"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            join(opt.requested_windows, ";"),
            string(opt.plot_max_points),
            string(opt.hist_bins),
            string(opt.seed),
            join(string.(opt.anisotropy_quantiles), ";"),
        ], ","))
    end
end

fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
