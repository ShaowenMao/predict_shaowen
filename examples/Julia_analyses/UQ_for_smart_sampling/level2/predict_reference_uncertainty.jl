#!/usr/bin/env julia

"""
Reference-solution uncertainty plots for the six GOM throw windows.

This script reads the reference `permsRef` samples stored in the MATLAB
`*_distribution_data.mat` files produced by `gom_perm_distribution_sensitivity`
and analyzes the joint random vector

    y = (log10(kxx), log10(kyy), log10(kzz))

For each selected window it generates:
1. 1D probability histograms for each component using the saved MATLAB bin edges.
2. Pairwise scatter plots in the 3 log-permeability planes.
3. A simple 3D k-means clustering view of the joint samples.

It also saves cluster assignments and cluster-center summaries to CSV.

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie", "Clustering"])

Examples:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_uncertainty.jl
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_uncertainty.jl --windows famp1
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_uncertainty.jl --windows famp1,famp2 --clusters 4
    julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_uncertainty.jl --data-dir path\\to\\data --output-dir path\\to\\out
"""

const REQUIRED_PACKAGES = ["MAT", "CairoMakie", "Clustering"]
missing_packages = [pkg for pkg in REQUIRED_PACKAGES if Base.find_package(pkg) === nothing]
if !isempty(missing_packages)
    pkg_list = join(["\"" * pkg * "\"" for pkg in missing_packages], ", ")
    error("Missing Julia packages: $(join(missing_packages, ", ")). Install them with:\n" *
          "using Pkg; Pkg.add([$pkg_list])")
end

using MAT
using CairoMakie
using Clustering
using Statistics
using LinearAlgebra
using Random
using Printf

CairoMakie.activate!()

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx [mD])", "log10(kyy [mD])", "log10(kzz [mD])")
const PAIRWISE_COMPONENTS = ((1, 2), (1, 3), (2, 3))
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_sensitivity_independent", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_sensitivity_independent", "julia_uncertainty")),
        "windows" => "",
        "clusters" => "3",
        "seed" => "1729",
        "plot-max-points" => "10000",
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
        clusters = parse(Int, options["clusters"]),
        seed = parse(Int, options["seed"]),
        plot_max_points = parse(Int, options["plot-max-points"]),
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level2/predict_reference_uncertainty.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>         Folder with *_distribution_data.mat files")
    println("  --output-dir <path>       Folder where Julia figures/CSVs are saved")
    println("  --windows <names>         Comma-separated list like famp1,famp2")
    println("  --clusters <k>            Number of k-means clusters (default: 3)")
    println("  --seed <int>              Random seed for clustering (default: 1729)")
    println("  --plot-max-points <n>     Max plotted points in scatter views (default: 10000)")
    println("  -h, --help                Show this help")
end

function main(args)
    opt = parse_args(args)

    mkpath(opt.output_dir)
    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No reference MAT files found in $(opt.data_dir)")

    println("Processing $(length(windows)) window(s) from $(opt.data_dir)")
    for (window, filepath) in windows
        println("  - $window")
        process_window(window, filepath, opt.output_dir;
                       k_clusters = opt.clusters,
                       seed = opt.seed,
                       plot_max_points = opt.plot_max_points)
    end
    println("Saved outputs to $(opt.output_dir)")
end

function collect_windows(data_dir::AbstractString, requested_windows::Vector{String})
    isdir(data_dir) || error("Data directory does not exist: $data_dir")
    files = filter(f -> endswith(f, "_distribution_data.mat"), readdir(data_dir))
    sort!(files)

    window_map = Dict{String, String}()
    for file in files
        window = replace(file, "_distribution_data.mat" => "")
        window_map[window] = joinpath(data_dir, file)
    end

    if isempty(requested_windows)
        return [(window, window_map[window]) for window in sort(collect(keys(window_map)))]
    end

    missing = [w for w in requested_windows if !haskey(window_map, w)]
    isempty(missing) || error("Requested window(s) not found: $(join(missing, ", "))")
    return [(window, window_map[window]) for window in requested_windows]
end

function process_window(window::AbstractString, filepath::AbstractString, output_dir::AbstractString;
                        k_clusters::Int = 3,
                        seed::Int = 1729,
                        plot_max_points::Int = 10000)
    ref_data = load_reference_data(filepath)
    perms_ref = ref_data.perms_ref
    bin_edges = ref_data.bin_edges
    y = log10.(perms_ref)

    cluster_result = cluster_joint_samples(y, k_clusters; seed = seed)
    plot_index = downsample_indices(size(y, 1), plot_max_points; seed = seed)

    window_dir = joinpath(output_dir, window)
    mkpath(window_dir)

    save_marginals_pairwise(window, y, bin_edges, plot_index, joinpath(window_dir, "$(window)_marginals_pairwise.png"))
    save_cluster_figure(window, y, plot_index, cluster_result, joinpath(window_dir, "$(window)_clusters_3d.png"))
    write_cluster_assignments(joinpath(window_dir, "$(window)_cluster_assignments.csv"), y, cluster_result.assignments)
    write_cluster_centers(joinpath(window_dir, "$(window)_cluster_centers.csv"), cluster_result)
end

function load_reference_data(filepath::AbstractString)
    data = matread(filepath)
    haskey(data, "permsRef") || error("File does not contain permsRef: $filepath")
    perms = Matrix{Float64}(data["permsRef"])
    size(perms, 2) == 3 || error("Expected permsRef to have 3 columns in $filepath")
    all(perms .> 0) || error("permsRef contains non-positive values in $filepath")

    if haskey(data, "binEdges")
        bin_edges = vec(Float64.(data["binEdges"]))
    else
        y = log10.(perms)
        data_min = floor(minimum(y))
        data_max = ceil(maximum(y))
        bin_edges = collect(range(data_min, data_max; length = 25))
    end

    return (perms_ref = perms, bin_edges = bin_edges)
end

function cluster_joint_samples(y::Matrix{Float64}, k::Int; seed::Int = 1729)
    k >= 1 || error("Number of clusters must be >= 1")
    n_samples = size(y, 1)
    k <= n_samples || error("Number of clusters ($k) exceeds number of samples ($n_samples)")

    mu = vec(mean(y; dims = 1))
    sigma = vec(std(y; dims = 1, corrected = true))
    sigma[sigma .== 0.0] .= 1.0

    z = (y .- reshape(mu, 1, :)) ./ reshape(sigma, 1, :)

    Random.seed!(seed)
    result = kmeans(permutedims(z), k; display = :none, maxiter = 300)

    centers_std = permutedims(result.centers)
    centers_log = centers_std .* reshape(sigma, 1, :) .+ reshape(mu, 1, :)
    sizes = [count(==(i), result.assignments) for i in 1:k]

    return (
        assignments = collect(result.assignments),
        centers_log = centers_log,
        sizes = sizes,
        means = mu,
        stds = sigma,
    )
end

function downsample_indices(n::Int, max_points::Int; seed::Int = 1729)
    n <= max_points && return collect(1:n)
    rng = MersenneTwister(seed)
    idx = randperm(rng, n)[1:max_points]
    sort!(idx)
    return idx
end

function save_marginals_pairwise(window::AbstractString, y::Matrix{Float64},
                                 bin_edges::Vector{Float64},
                                 plot_index::Vector{Int}, output_path::AbstractString)
    fig = Figure(size = (1600, 900), fontsize = 18)
    pdf_colors = [:steelblue3, :darkorange2, :seagreen4]

    for i in 1:3
        ax = Axis(fig[1, i],
                  xlabel = COMPONENT_LABELS[i],
                  ylabel = "Probability",
                  title = "Marginal histogram: $(COMPONENT_NAMES[i])")
        hist!(ax, y[:, i];
              bins = bin_edges,
              normalization = :probability,
              color = (pdf_colors[i], 0.65),
              strokecolor = :black,
              strokewidth = 1.0)
    end

    yplot = y[plot_index, :]
    for (j, (a, b)) in enumerate(PAIRWISE_COMPONENTS)
        ax = Axis(fig[2, j],
                  xlabel = COMPONENT_LABELS[a],
                  ylabel = COMPONENT_LABELS[b],
                  title = "Scatter: $(COMPONENT_NAMES[a]) vs $(COMPONENT_NAMES[b])")
        scatter!(ax, yplot[:, a], yplot[:, b],
                 color = (:black, 0.18), markersize = 4)
    end

    Label(fig[0, :], "$window reference uncertainty: histograms and pairwise scatter",
          fontsize = 22, font = :bold)
    save(output_path, fig)
end

function save_cluster_figure(window::AbstractString, y::Matrix{Float64},
                             plot_index::Vector{Int}, cluster_result, output_path::AbstractString)
    k = length(cluster_result.sizes)
    palette = cluster_palette(k)
    assignments = cluster_result.assignments
    yplot = y[plot_index, :]
    plot_assignments = assignments[plot_index]

    fig = Figure(size = (1350, 900), fontsize = 18)
    ax = Axis3(fig[1, 1],
               xlabel = COMPONENT_LABELS[1],
               ylabel = COMPONENT_LABELS[2],
               zlabel = COMPONENT_LABELS[3],
               title = "$window: simple 3D clustering")

    scatter!(ax, yplot[:, 1], yplot[:, 2], yplot[:, 3],
             color = palette[plot_assignments],
             markersize = 6, alpha = 0.35)
    scatter!(ax, cluster_result.centers_log[:, 1],
             cluster_result.centers_log[:, 2],
             cluster_result.centers_log[:, 3],
             color = palette,
             marker = :diamond,
             markersize = 26,
             strokecolor = :black,
             strokewidth = 1.5)

    elements = [MarkerElement(color = palette[i], marker = :circle, markersize = 14)
                for i in 1:k]
    labels = ["Cluster $i (n = $(cluster_result.sizes[i]))" for i in 1:k]
    Legend(fig[1, 2], elements, labels, "k-means on standardized y")

    summary_lines = [
        "Joint variable: y = (log10(kxx), log10(kyy), log10(kzz))",
        "Clustering uses standardized components before k-means.",
        @sprintf("Total samples: %d", size(y, 1)),
        @sprintf("Displayed points: %d", length(plot_index)),
    ]
    Label(fig[2, 1:2], join(summary_lines, "\n"), tellwidth = false, fontsize = 16)

    save(output_path, fig)
end

function cluster_palette(k::Int)
    base = Makie.wong_colors()
    return [base[mod1(i, length(base))] for i in 1:k]
end

function write_cluster_assignments(filepath::AbstractString, y::Matrix{Float64}, assignments::Vector{Int})
    open(filepath, "w") do io
        println(io, "sample_index,log10_kxx,log10_kyy,log10_kzz,cluster")
        for i in 1:size(y, 1)
            @printf(io, "%d,%.10f,%.10f,%.10f,%d\n",
                    i, y[i, 1], y[i, 2], y[i, 3], assignments[i])
        end
    end
end

function write_cluster_centers(filepath::AbstractString, cluster_result)
    centers = cluster_result.centers_log
    sizes = cluster_result.sizes
    open(filepath, "w") do io
        println(io, "cluster,size,center_log10_kxx,center_log10_kyy,center_log10_kzz,center_kxx_mD,center_kyy_mD,center_kzz_mD")
        for i in 1:size(centers, 1)
            @printf(io, "%d,%d,%.10f,%.10f,%.10f,%.10e,%.10e,%.10e\n",
                    i, sizes[i],
                    centers[i, 1], centers[i, 2], centers[i, 3],
                    10.0^centers[i, 1], 10.0^centers[i, 2], 10.0^centers[i, 3])
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
