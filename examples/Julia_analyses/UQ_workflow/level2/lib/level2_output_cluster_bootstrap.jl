#!/usr/bin/env julia

const LEVEL2_ROOT = normpath(joinpath(@__DIR__, ".."))

using CairoMakie
using Printf
using Random
using Statistics

include(joinpath(LEVEL2_ROOT, "lib", "level2_io.jl"))
include(joinpath(LEVEL2_ROOT, "lib", "level2_ranks.jl"))
include(joinpath(LEVEL2_ROOT, "lib", "level2_distances.jl"))
include(joinpath(LEVEL2_ROOT, "lib", "level2_clustering.jl"))
include(joinpath(LEVEL2_ROOT, "lib", "level2_plotting.jl"))

using .Level2IO
using .Level2Ranks
using .Level2Distances
using .Level2Clustering
using .Level2Plotting

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "manifest" => Level2IO.default_manifest_path(),
        "output-dir" => "",
        "bootstrap-repeats" => "50",
        "bootstrap-size" => "0",
        "seed" => "4471",
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
    return options
end

function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/lib/level2_output_cluster_bootstrap.jl [options]")
    println()
    println("Options:")
    println("  --config <path>              Level 2 TOML config")
    println("  --manifest <path>            Proxy manifest CSV")
    println("  --output-dir <path>          Output folder for bootstrap tables and figures")
    println("  --bootstrap-repeats <n>      Number of bootstrap replicates per window")
    println("  --bootstrap-size <n>         Resampled size. Use 0 for full window sample size")
    println("  --seed <n>                   Random seed for bootstrap row resampling")
    println("  -h, --help                   Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    manifest_rows = Level2IO.read_manifest_csv(opt["manifest"], config)
    bootstrap_repeats = parse(Int, opt["bootstrap-repeats"])
    bootstrap_repeats > 0 || error("--bootstrap-repeats must be positive")
    requested_bootstrap_size = parse(Int, opt["bootstrap-size"])
    requested_bootstrap_size >= 0 || error("--bootstrap-size must be non-negative")
    base_seed = parse(Int, opt["seed"])

    output_root = isempty(opt["output-dir"]) ?
        normpath(joinpath(Level2IO.default_level2_output_root(), config["geology_id"], "figures", "joint_cluster_bootstrap")) :
        normpath(opt["output-dir"])
    table_root = joinpath(output_root, "tables")
    figure_root = joinpath(output_root, "figures")
    mkpath(table_root)
    mkpath(figure_root)

    Level2Plotting.activate_plot_theme!()

    proxies = [Level2IO.load_proxy_library(row) for row in manifest_rows]
    replicate_rows = Vector{Vector{String}}()
    summary_rows = Vector{Vector{String}}()
    summary = Dict{String, Dict{String, Any}}()

    for (widx, proxy) in enumerate(proxies)
        window = proxy["window"]
        log_perms = proxy["log_perms"]
        n = size(log_perms, 1)
        bootstrap_size = requested_bootstrap_size == 0 ? n : requested_bootstrap_size
        bootstrap_size > 1 || error("Bootstrap size must be at least 2 for $window")

        println("Bootstrapping Step 2.3 joint clusters for $window ($bootstrap_repeats replicates, n = $bootstrap_size)")

        original = detect_joint_clusters(log_perms, config, window, config["random_seed"])
        k_values = Int[]
        silhouettes = Float64[]
        unimodal_values = Int[]

        for repeat_id in 1:bootstrap_repeats
            rng = MersenneTwister(base_seed + 10_000 * widx + repeat_id)
            row_idx = rand(rng, 1:n, bootstrap_size)
            boot_log = Matrix{Float64}(log_perms[row_idx, :])
            boot_seed = base_seed + 100_000 * widx + repeat_id
            result = detect_joint_clusters(boot_log, config, window, boot_seed)

            chosen_k = Int(result["chosen_k"])
            silhouette = Float64(result["best_silhouette"])
            is_unimodal = Int(result["is_effectively_unimodal"])
            cluster_sizes = ordered_cluster_sizes(result)

            push!(k_values, chosen_k)
            push!(silhouettes, silhouette)
            push!(unimodal_values, is_unimodal)
            push!(replicate_rows, [
                window,
                string(repeat_id),
                string(n),
                string(bootstrap_size),
                string(chosen_k),
                @sprintf("%.6f", silhouette),
                string(is_unimodal),
                join(string.(cluster_sizes), ";"),
            ])
        end

        window_summary = summarize_bootstrap_window(window,
                                                    original,
                                                    k_values,
                                                    silhouettes,
                                                    unimodal_values,
                                                    bootstrap_repeats,
                                                    bootstrap_size,
                                                    config["max_k"])
        summary[window] = window_summary
        push!(summary_rows, summary_row(window_summary))
    end

    Level2IO.write_csv(joinpath(table_root, "joint_cluster_bootstrap_replicates.csv"),
                       ["window", "bootstrap_id", "original_n", "bootstrap_n",
                        "chosen_k", "best_silhouette", "is_effectively_unimodal",
                        "cluster_sizes_ordered"],
                       replicate_rows)

    Level2IO.write_csv(joinpath(table_root, "joint_cluster_bootstrap_summary.csv"),
                       ["window", "original_k", "original_silhouette", "bootstrap_repeats",
                        "bootstrap_n", "modal_k", "p_same_k_as_original", "p_k2",
                        "p_effectively_unimodal", "silhouette_q05", "silhouette_median",
                        "silhouette_q95", "silhouette_min", "silhouette_max"],
                       summary_rows)

    fig = build_bootstrap_overview(summary, config["max_k"])
    save(joinpath(figure_root, "joint_cluster_bootstrap_overview.png"), fig)
    save_optional_pdf(joinpath(figure_root, "joint_cluster_bootstrap_overview.pdf"), fig)

    println("Saved Step 2.3 bootstrap outputs to $output_root")
end

function detect_joint_clusters(log_perms::Matrix{Float64},
                               config::Dict{String, Any},
                               window::AbstractString,
                               random_seed::Integer)
    cfg = copy(config)
    cfg["random_seed"] = Int(random_seed)
    local_ranks = Level2Ranks.compute_local_ranks(log_perms)
    joint_rank_score = Level2Ranks.compute_joint_rank_score(local_ranks, Float64.(cfg["weights"]))
    local_normal_scores = Level2Ranks.compute_local_normal_scores(local_ranks)
    distance_info = Level2Distances.build_distance_matrix(log_perms, local_normal_scores, cfg)
    return Level2Clustering.choose_clustering(distance_info["distance_matrix"], joint_rank_score, cfg, window)
end

function ordered_cluster_sizes(result::Dict{String, Any})
    sizes = vec(Int.(result["cluster_sizes"]))
    order = vec(Int.(result["cluster_order"]))
    return [sizes[cluster_id] for cluster_id in order]
end

function summarize_bootstrap_window(window::AbstractString,
                                    original::Dict{String, Any},
                                    k_values::Vector{Int},
                                    silhouettes::Vector{Float64},
                                    unimodal_values::Vector{Int},
                                    bootstrap_repeats::Int,
                                    bootstrap_size::Int,
                                    max_k::Int)
    original_k = Int(original["chosen_k"])
    k_counts = [count(==(k), k_values) for k in 1:max_k]
    modal_k = argmax(k_counts)
    p_by_k = [count(==(k), k_values) / bootstrap_repeats for k in 1:max_k]

    return Dict{String, Any}(
        "window" => String(window),
        "original_k" => original_k,
        "original_silhouette" => Float64(original["best_silhouette"]),
        "bootstrap_repeats" => bootstrap_repeats,
        "bootstrap_size" => bootstrap_size,
        "modal_k" => modal_k,
        "p_same_k_as_original" => count(==(original_k), k_values) / bootstrap_repeats,
        "p_k2" => 2 <= max_k ? count(==(2), k_values) / bootstrap_repeats : 0.0,
        "p_effectively_unimodal" => mean(unimodal_values),
        "silhouette_q05" => quantile(silhouettes, 0.05),
        "silhouette_median" => median(silhouettes),
        "silhouette_q95" => quantile(silhouettes, 0.95),
        "silhouette_min" => minimum(silhouettes),
        "silhouette_max" => maximum(silhouettes),
        "p_by_k" => p_by_k,
    )
end

function summary_row(summary::Dict{String, Any})
    return [
        summary["window"],
        string(summary["original_k"]),
        @sprintf("%.6f", summary["original_silhouette"]),
        string(summary["bootstrap_repeats"]),
        string(summary["bootstrap_size"]),
        string(summary["modal_k"]),
        @sprintf("%.6f", summary["p_same_k_as_original"]),
        @sprintf("%.6f", summary["p_k2"]),
        @sprintf("%.6f", summary["p_effectively_unimodal"]),
        @sprintf("%.6f", summary["silhouette_q05"]),
        @sprintf("%.6f", summary["silhouette_median"]),
        @sprintf("%.6f", summary["silhouette_q95"]),
        @sprintf("%.6f", summary["silhouette_min"]),
        @sprintf("%.6f", summary["silhouette_max"]),
    ]
end

function build_bootstrap_overview(summary::Dict{String, Dict{String, Any}}, max_k::Int)
    windows = Level2IO.FIXED_WINDOWS
    window_labels = ["W$i" for i in 1:length(windows)]
    p_matrix = zeros(Float64, length(windows), max_k)
    sil_q05 = zeros(Float64, length(windows))
    sil_med = zeros(Float64, length(windows))
    sil_q95 = zeros(Float64, length(windows))
    original_sil = zeros(Float64, length(windows))

    for (widx, window) in enumerate(windows)
        s = summary[window]
        p_matrix[widx, :] .= Float64.(s["p_by_k"])
        sil_q05[widx] = s["silhouette_q05"]
        sil_med[widx] = s["silhouette_median"]
        sil_q95[widx] = s["silhouette_q95"]
        original_sil[widx] = s["original_silhouette"]
    end

    title_font_size = 30
    axis_font_size = 24
    tick_font_size = 22

    fig = Figure(size = (1750, 820), figure_padding = 24, backgroundcolor = :white)
    Label(fig[1, 1:2],
          "Step 2.3 bootstrap stability of automatic joint-cluster detection",
          fontsize = title_font_size,
          font = :bold,
          halign = :center,
          tellwidth = false)

    ax_prob = Axis(fig[2, 1],
                   title = "Bootstrap probability of selected K",
                   xlabel = "Selected K",
                   ylabel = "Window",
                   titlesize = axis_font_size,
                   xlabelsize = axis_font_size,
                   ylabelsize = axis_font_size,
                   xticklabelsize = tick_font_size,
                   yticklabelsize = tick_font_size,
                   xticks = (1:max_k, string.(1:max_k)),
                   yticks = (1:length(windows), window_labels),
                   yreversed = true,
                   topspinevisible = false,
                   rightspinevisible = false)
    hm = heatmap!(ax_prob, 1:max_k, 1:length(windows), permutedims(p_matrix);
                  colorrange = (0.0, 1.0),
                  colormap = :viridis)
    Colorbar(fig[2, 2], hm;
             label = "Probability",
             ticks = ([0.0, 0.25, 0.50, 0.75, 1.0],
                      ["0", "0.25", "0.50", "0.75", "1"]),
             labelsize = axis_font_size,
             ticklabelsize = tick_font_size)
    for widx in 1:length(windows), k in 1:max_k
        value = p_matrix[widx, k]
        label_color = value >= 0.55 ? :white : :black
        text!(ax_prob, k, widx;
              text = @sprintf("%.2f", value),
              align = (:center, :center),
              fontsize = tick_font_size,
              color = label_color)
    end

    ax_sil = Axis(fig[2, 3],
                  title = "Bootstrap silhouette stability",
                  xlabel = "Best silhouette",
                  ylabel = "",
                  titlesize = axis_font_size,
                  xlabelsize = axis_font_size,
                  xticklabelsize = tick_font_size,
                  yticklabelsize = tick_font_size,
                  yticks = (1:length(windows), window_labels),
                  yreversed = true,
                  topspinevisible = false,
                  rightspinevisible = false)
    hideydecorations!(ax_sil, grid = false)
    for widx in 1:length(windows)
        lines!(ax_sil, [sil_q05[widx], sil_q95[widx]], [widx, widx];
               color = RGBf(0.25, 0.25, 0.25),
               linewidth = 6)
        scatter!(ax_sil, [sil_med[widx]], [widx];
                 color = RGBf(0.153, 0.392, 0.780),
                 markersize = 18,
                 strokecolor = :black,
                 strokewidth = 1.2)
        scatter!(ax_sil, [original_sil[widx]], [widx];
                 color = RGBf(0.820, 0.235, 0.196),
                 marker = :diamond,
                 markersize = 20,
                 strokecolor = :black,
                 strokewidth = 1.2)
    end
    xlims!(ax_sil, 0.20, max(0.45, maximum(sil_q95) + 0.03))

    legend_elements = [
        LineElement(color = RGBf(0.25, 0.25, 0.25), linewidth = 6),
        MarkerElement(color = RGBf(0.153, 0.392, 0.780),
                      marker = :circle,
                      markersize = 18,
                      strokecolor = :black,
                      strokewidth = 1.2),
        MarkerElement(color = RGBf(0.820, 0.235, 0.196),
                      marker = :diamond,
                      markersize = 20,
                      strokecolor = :black,
                      strokewidth = 1.2),
    ]
    Legend(fig[3, 1:3], legend_elements,
           ["5th-95th percentile", "Bootstrap median", "Full-data value"];
           orientation = :horizontal,
           framevisible = false,
           labelsize = tick_font_size,
           tellwidth = false,
           colgap = 24)

    Label(fig[4, 1:3],
          "Each bootstrap replicate resamples rows within a window and reruns the same automatic K-selection rule.",
          fontsize = tick_font_size,
          halign = :center,
          tellwidth = false)

    colsize!(fig.layout, 1, Relative(0.48))
    colsize!(fig.layout, 2, Fixed(90))
    colsize!(fig.layout, 3, Relative(0.43))
    rowsize!(fig.layout, 1, Fixed(76))
    rowsize!(fig.layout, 3, Fixed(72))
    rowsize!(fig.layout, 4, Fixed(46))
    return fig
end

function save_optional_pdf(path::AbstractString, fig::Figure)
    try
        save(path, fig)
    catch err
        @warn "Skipping PDF export because the file is locked or unavailable" path exception = (err, catch_backtrace())
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
