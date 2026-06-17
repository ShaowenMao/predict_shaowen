#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using CairoMakie
using Statistics
using Random
using Dates
using MAT

"""
    make_three_level_result_visuals.jl

Create the current retained analysis figures from the completed 162-geology
three-level sampling workflow outputs. The script reads the aggregate CSV
files under `D:/codex_gom/UQ_workflow/three_level_sampling_162/summary` by
default and exports PPT/report-ready PNG and PDF figures.

The retained figure set is intentionally small:
1. Level 2 joint cluster count distribution by window.
2. Level 3 similarity-group structure frequency.
3. Level 3 window-pair co-grouping frequency.
"""

const DEFAULT_ROOT = normpath("D:/codex_gom/UQ_workflow/three_level_sampling_162")
const WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]
const WINDOW_LABELS = ["W1", "W2", "W3", "W4", "W5", "W6"]
const COMPONENTS = ["kxx", "kyy", "kzz"]
const LOG_FIELDS = ["log_kxx", "log_kyy", "log_kzz"]

const INK = RGBf(0.04, 0.10, 0.22)
const MUTED = RGBf(0.34, 0.36, 0.42)
const GRID = RGBf(0.86, 0.87, 0.89)
const BLUE = RGBf(0.10, 0.35, 0.64)
const GOLD = RGBf(0.93, 0.58, 0.13)
const GREEN = RGBf(0.15, 0.52, 0.29)
const ROSE = RGBf(0.78, 0.22, 0.28)
const TEAL = RGBf(0.08, 0.55, 0.63)
const CHARCOAL = RGBf(0.16, 0.17, 0.20)
const LIGHT_GREY = RGBf(0.94, 0.95, 0.96)

const K_COLORS = [RGBf(0.92, 0.94, 0.97),
                  RGBf(0.61, 0.76, 0.88),
                  RGBf(0.30, 0.55, 0.78),
                  RGBf(0.97, 0.66, 0.22),
                  RGBf(0.78, 0.23, 0.18)]

const GROUP_COLORS = [RGBf(0.91, 0.93, 0.95),
                      RGBf(0.70, 0.82, 0.91),
                      RGBf(0.39, 0.63, 0.82),
                      RGBf(0.98, 0.71, 0.31),
                      RGBf(0.88, 0.41, 0.24),
                      RGBf(0.46, 0.23, 0.42)]

const STRUCTURE_COLORS = Dict(
    "6" => RGBf(0.10, 0.24, 0.48),
    "1+1+1+1+1+1" => RGBf(0.62, 0.64, 0.68),
    "2+1+1+1+1" => RGBf(0.31, 0.49, 0.72),
    "2+2+1+1" => RGBf(0.13, 0.36, 0.60),
    "2+2+2" => RGBf(0.16, 0.58, 0.63),
    "3+1+1+1" => RGBf(0.52, 0.68, 0.34),
    "3+2+1" => RGBf(0.85, 0.60, 0.17),
    "3+3" => RGBf(0.94, 0.45, 0.18),
    "4+1+1" => RGBf(0.73, 0.25, 0.28),
    "4+2" => RGBf(0.56, 0.26, 0.50),
    "5+1" => RGBf(0.30, 0.23, 0.45),
)

const RETAINED_FIGURE_BASENAMES = [
    "13_level2_cluster_count_distribution_by_window",
    "08_similarity_group_structure_frequency",
    "09_window_pair_cogrouping_frequency",
]

const DROPPED_FIGURE_BASENAMES = [
    "01_level2_cluster_complexity_atlas",
    "02_parameter_controls_cluster_complexity",
    "03_level3_window_similarity_group_atlas",
    "04_similarity_structure_distribution",
    "05_complexity_vs_grouping",
    "06_sampled_six_window_kzz_case_matrix",
    "07_sampled_case_type_component_summary",
    "10_singleton_frequency_by_window",
    "11_faultwide_coupling_support_summary",
    "12_low_high_state_separation",
]

"""
    main(args)

Entry point for result visualization generation.
"""
function main(args::Vector{String})
    opt = parse_args(args)
    root = normpath(opt["root"])
    summary_root = joinpath(root, "summary")
    output_root = isempty(opt["output-dir"]) ?
        joinpath(root, "figures", "result_analysis") :
        normpath(opt["output-dir"])
    mkpath(output_root)
    mkpath(joinpath(output_root, "tables"))
    remove_dropped_result_figures(output_root)

    level2 = read_csv_dicts(joinpath(summary_root, "level2_build_summary_all_geologies.csv"))
    groups = read_csv_dicts(joinpath(summary_root, "level3_window_similarity_group_summary_all_geologies.csv"))

    figure_paths = String[]
    push_saved!(figure_paths, save_level2_cluster_count_distribution_by_window(level2, output_root))
    push_saved!(figure_paths, save_similarity_structure_frequency(groups, output_root))
    push_saved!(figure_paths, save_window_pair_cogrouping_frequency(groups, output_root))

    write_retained_supporting_tables(level2, groups, output_root)
    write_csv(joinpath(output_root, "figure_manifest.csv"),
              ["figure_path"],
              [[path] for path in figure_paths])

    println("Generated $(length(figure_paths)) retained result-analysis figure files in $output_root")
end

"""
    parse_args(args)

Parse optional `--root` and `--output-dir` command-line arguments.
"""
function parse_args(args::Vector{String})
    opt = Dict("root" => DEFAULT_ROOT, "output-dir" => "")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage:")
            println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/batch/make_three_level_result_visuals.jl [--root <workflow-output-root>] [--output-dir <figure-dir>]")
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(opt, key) || error("Unknown option $arg")
            i < length(args) || error("Missing value for $arg")
            opt[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return opt
end

function push_saved!(paths::Vector{String}, new_paths::Vector{String})
    append!(paths, new_paths)
    return paths
end

"""
    save_level2_cluster_count_distribution_by_window(level2, output_root)

Show the distribution of selected Level 2 joint permeability cluster counts
for each throw window across all geologies.

The response variable is discrete (`K = 1..5`), so this uses a count-bubble
plot rather than a smoothed violin. Bubble area and labels show the number
of geologies assigned to each window/K bin.
"""
function save_level2_cluster_count_distribution_by_window(level2, output_root)
    count_matrix = zeros(Int, length(WINDOWS), 5)
    for row in level2
        wi = window_index(row["window"])
        k = parse(Int, row["chosen_k"])
        1 <= k <= 5 || continue
        count_matrix[wi, k] += 1
    end

    n_geologies = maximum(sum(count_matrix; dims = 2))
    max_count = maximum(count_matrix)
    mean_k = [sum(k * count_matrix[wi, k] for k in 1:5) / max(1, sum(count_matrix[wi, :]))
              for wi in 1:length(WINDOWS)]

    fig = Figure(size = (1900, 1150), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Level 2 joint cluster count distribution by window",
          fontsize = 44, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "Each bubble is one window/K bin across $(n_geologies) geologies; bubble area and labels show geology count.",
          fontsize = 27, color = MUTED, tellwidth = false)

    ax = Axis(fig[3, 1],
              xlabel = "Throw window",
              ylabel = "Number of joint permeability clusters (K)",
              xticks = (1:6, WINDOW_LABELS),
              yticks = (1:5, string.(1:5)),
              limits = ((0.45, 6.55), (0.55, 5.45)),
              xlabelsize = 32,
              ylabelsize = 32,
              xticklabelsize = 30,
              yticklabelsize = 30)
    ax.xgridcolor = GRID
    ax.ygridcolor = GRID
    ax.xgridwidth = 1.0
    ax.ygridwidth = 1.0

    for wi in 1:length(WINDOWS), k in 1:5
        count_value = count_matrix[wi, k]
        count_value == 0 && continue
        marker_size = 24 + 82 * sqrt(count_value / max_count)
        scatter!(ax, [wi], [k];
                 markersize = marker_size,
                 color = (BLUE, 0.28 + 0.58 * count_value / max_count),
                 strokecolor = INK,
                 strokewidth = 1.3)
        text!(ax, wi, k;
              text = string(count_value),
              align = (:center, :center),
              fontsize = count_value >= 10 ? 24 : 21,
              color = INK)
    end

    lines!(ax, 1:6, mean_k; color = GOLD, linewidth = 4, linestyle = :dash)
    scatter!(ax, 1:6, mean_k; marker = :diamond, markersize = 22,
             color = GOLD, strokecolor = INK, strokewidth = 1.2)
    text!(ax, 6.08, mean_k[end];
          text = "mean K",
          align = (:left, :center),
          fontsize = 24,
          color = GOLD)

    paths = save_figure(fig, output_root, "13_level2_cluster_count_distribution_by_window")
    return paths
end

"""
    save_level2_cluster_complexity_atlas(level2, output_root, scenario_order, case_order)

Figure 1: six scenario panels showing chosen cluster count `K` by case and
window.
"""
function save_level2_cluster_complexity_atlas(level2, output_root, scenario_order, case_order)
    fig = Figure(size = (2600, 1700), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1:2], "Level 2 cluster complexity atlas",
          fontsize = 46, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1:2],
          "Cell color is chosen joint permeability cluster count K for each 2000-realization window library.",
          fontsize = 28, color = MUTED, tellwidth = false)

    cmap = cgrad(K_COLORS, 5, categorical = true)
    axes = Axis[]
    for (panel, scenario_index) in enumerate(scenario_order)
        row = 3 + div(panel - 1, 2)
        col = 1 + mod(panel - 1, 2)
        scenario_rows = [r for r in level2 if r["scenario_index"] == scenario_index]
        mat = fill(NaN, 27, 6)
        for r in scenario_rows
            ci = parse(Int, r["case_index"])
            wi = window_index(r["window"])
            mat[ci, wi] = parse(Float64, r["chosen_k"])
        end
        ax = Axis(fig[row, col],
                  title = scenario_title(first(scenario_rows)),
                  xlabel = "Geology case index",
                  ylabel = col == 1 ? "Throw window" : "",
                  xticks = ([1, 9, 18, 27], ["1", "9", "18", "27"]),
                  yticks = (1:6, WINDOW_LABELS),
                  yreversed = true,
                  titlesize = 32,
                  xlabelsize = 26,
                  ylabelsize = 26,
                  xticklabelsize = 22,
                  yticklabelsize = 24)
        heatmap!(ax, 1:27, 1:6, mat; colormap = cmap, colorrange = (1, 5))
        push!(axes, ax)
    end
    Colorbar(fig[3:5, 3], limits = (1, 5), colormap = cmap,
             ticks = 1:5, label = "Chosen K", labelsize = 26,
             ticklabelsize = 24, width = 28)
    paths = save_figure(fig, output_root, "01_level2_cluster_complexity_atlas")
    return paths
end

"""
    save_cluster_complexity_controls(level2, output_root)

Figure 2: mean chosen K controlled by fault depth, sand Vcl, and clay Vcl.
"""
function save_cluster_complexity_controls(level2, output_root)
    panels = [
        ("faulting_depth_m", "Fault depth (m)", [50.0, 500.0, 1000.0]),
        ("sand_vcl", "Sand Vcl", [0.1, 0.2, 0.3]),
        ("clay_vcl", "Clay Vcl", [0.4, 0.5, 0.6]),
    ]
    fig = Figure(size = (2400, 950), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1:3], "Parameter controls on within-window cluster complexity",
          fontsize = 44, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1:3],
          "Bars show mean chosen K across all matching geology-window libraries; labels show percent of windows with K > 1.",
          fontsize = 26, color = MUTED, tellwidth = false)

    for (idx, (field, xlabel, values)) in enumerate(panels)
        means = Float64[]
        fracs = Float64[]
        labels = String[]
        for value in values
            rows = [r for r in level2 if parse(Float64, r[field]) ≈ value]
            push!(means, mean(parse.(Float64, getfield_vec(rows, "chosen_k"))))
            push!(fracs, mean([parse(Int, r["chosen_k"]) > 1 ? 1.0 : 0.0 for r in rows]))
            push!(labels, field == "faulting_depth_m" ? string(Int(value)) : string(value))
        end
        ax = Axis(fig[3, idx],
                  title = xlabel,
                  xlabel = xlabel,
                  ylabel = idx == 1 ? "Mean chosen K" : "",
                  xticks = (1:length(labels), labels),
                  limits = (nothing, (1.0, 3.05)),
                  titlesize = 34,
                  xlabelsize = 28,
                  ylabelsize = 28,
                  xticklabelsize = 26,
                  yticklabelsize = 24)
        barplot!(ax, 1:length(means), means; color = [BLUE, GOLD, GREEN],
                 strokecolor = INK, strokewidth = 1.0, width = 0.62)
        for (x, m, f) in zip(1:length(means), means, fracs)
            text!(ax, x, m + 0.08; text = "$(round(Int, 100f))%",
                  align = (:center, :bottom), fontsize = 24, color = INK)
        end
        hlines!(ax, [1.0]; color = GRID, linestyle = :dash)
    end
    paths = save_figure(fig, output_root, "02_parameter_controls_cluster_complexity")
    return paths
end

"""
    save_level3_group_count_atlas(groups, output_root, scenario_order, case_order)

Figure 3: Level 3 group-count atlas with similarity-group structures annotated.
"""
function save_level3_group_count_atlas(groups, output_root, scenario_order, case_order)
    fig = Figure(size = (2800, 1050), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Level 3 window similarity group atlas",
          fontsize = 46, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "Cell color is group count; cell text is the selected similarity-group structure for one geology.",
          fontsize = 27, color = MUTED, tellwidth = false)
    mat = fill(NaN, 27, length(scenario_order))
    struct_mat = fill("", 27, length(scenario_order))
    for r in groups
        si = findfirst(==(r["scenario_index"]), scenario_order)
        ci = parse(Int, r["case_index"])
        mat[ci, si] = parse(Float64, r["group_count"])
        struct_mat[ci, si] = r["similarity_group_structure"]
    end
    cmap = cgrad(GROUP_COLORS, 6, categorical = true)
    ax = Axis(fig[3, 1],
              xlabel = "Geology case index",
              ylabel = "Thickness scenario",
              xticks = ([1, 9, 18, 27], ["1", "9", "18", "27"]),
              yticks = (1:length(scenario_order), [short_scenario_label(scenario_order, groups, s) for s in scenario_order]),
              yreversed = true,
              xlabelsize = 30,
              ylabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 24)
    heatmap!(ax, 1:27, 1:length(scenario_order), mat; colormap = cmap, colorrange = (1, 6))
    for x in 1:27, y in 1:length(scenario_order)
        label = struct_mat[x, y]
        isempty(label) && continue
        text!(ax, x, y; text = label, align = (:center, :center),
              fontsize = 13, color = text_color_for_group_count(mat[x, y]))
    end
    Colorbar(fig[3, 2], limits = (1, 6), colormap = cmap,
             ticks = 1:6, label = "Group count", labelsize = 28,
             ticklabelsize = 24, width = 28)
    paths = save_figure(fig, output_root, "03_level3_window_similarity_group_atlas")
    return paths
end

"""
    save_similarity_structure_distribution(groups, output_root, scenario_order)

Figure 4: horizontal stacked bars showing similarity-structure composition by
thickness scenario.
"""
function save_similarity_structure_distribution(groups, output_root, scenario_order)
    structures = sort(collect(keys(STRUCTURE_COLORS)); by = structure_sort_key)
    fig = Figure(size = (2500, 1100), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Similarity-group structures vary systematically across geology",
          fontsize = 44, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "Each bar contains 27 geologies in one thickness scenario; colors show the selected window similarity-group structure.",
          fontsize = 26, color = MUTED, tellwidth = false)
    ax = Axis(fig[3, 1],
              xlabel = "Number of geologies",
              ylabel = "",
              yticks = (1:length(scenario_order), [short_scenario_label(scenario_order, groups, s) for s in scenario_order]),
              yreversed = true,
              limits = ((0, 27), (0.35, length(scenario_order) + 0.65)),
              xlabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 26)
    for (yi, scenario_index) in enumerate(scenario_order)
        rows = [r for r in groups if r["scenario_index"] == scenario_index]
        counts = Dict(structure_label => count(r -> r["similarity_group_structure"] == structure_label, rows) for structure_label in structures)
        x0 = 0.0
        for structure_label in structures
            width = counts[structure_label]
            width == 0 && continue
            rectangle!(ax, x0, yi - 0.33, width, 0.66; color = STRUCTURE_COLORS[structure_label],
                       strokecolor = :white, strokewidth = 1.4)
            if width >= 3
                text!(ax, x0 + width / 2, yi; text = string(width),
                      align = (:center, :center), fontsize = 22,
                      color = luminance(STRUCTURE_COLORS[structure_label]) < 0.45 ? :white : INK)
            end
            x0 += width
        end
    end
    Legend(fig[4, 1],
           [PolyElement(polycolor = STRUCTURE_COLORS[s]) for s in structures],
           structures;
           orientation = :horizontal, nbanks = 2, labelsize = 20,
           framevisible = false, tellheight = true)
    paths = save_figure(fig, output_root, "04_similarity_structure_distribution")
    return paths
end

"""
    save_similarity_structure_frequency(groups, output_root)

Figure 8: overall occurrence count for each possible Level 3 window
similarity-group structure across all geologies.
"""
function save_similarity_structure_frequency(groups, output_root)
    structures = sort(collect(keys(STRUCTURE_COLORS)); by = structure_sort_key)
    counts = Dict(structure => count(r -> r["similarity_group_structure"] == structure, groups) for structure in structures)
    ordered = sort(structures; by = s -> (-counts[s], structure_sort_key(s)))
    total = length(groups)

    max_count = max(1, maximum(values(counts)))
    x_limit = max_count * 1.20
    bar_color = BLUE
    zero_color = RGBf(0.90, 0.91, 0.93)

    fig = Figure(size = (1650, 1120), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Frequency of window similarity-group structures",
          fontsize = 42, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "Each bar counts how often one selected Level 3 structure appears across 162 geologies.",
          fontsize = 24, color = MUTED, tellwidth = false)

    ax = Axis(fig[3, 1],
              xlabel = "Number of geologies",
              ylabel = "Similarity-group structure",
              yticks = (1:length(ordered), ordered),
              yreversed = true,
              limits = ((0, x_limit), (0.35, length(ordered) + 0.65)),
              xlabelsize = 30,
              ylabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 26,
              xgridvisible = true,
              ygridvisible = false,
              xgridcolor = GRID,
              xgridwidth = 1.0)
    xlims!(ax, 0, x_limit)
    for (yi, structure) in enumerate(ordered)
        count_value = counts[structure]
        rectangle!(ax, 0, yi - 0.36, count_value, 0.72;
                   color = count_value == 0 ? zero_color : bar_color,
                   strokecolor = count_value == 0 ? RGBf(0.78, 0.79, 0.82) : bar_color,
                   strokewidth = 1.0)
        label_x = count_value == 0 ? 0.55 : count_value + 0.60
        pct = 100 * count_value / total
        text!(ax, label_x, yi;
              text = "$(count_value) ($(round(pct; digits = 1))%)",
              align = (:left, :center), fontsize = 24,
              color = count_value == 0 ? MUTED : INK)
    end
    hidespines!(ax, :t, :r)
    paths = save_figure(fig, output_root, "08_similarity_group_structure_frequency")
    return paths
end

"""
    save_window_pair_cogrouping_frequency(groups, output_root)

Figure 9: how often each pair of throw windows is selected into the same
Level 3 window similarity group across all geologies.
"""
function save_window_pair_cogrouping_frequency(groups, output_root)
    counts = window_pair_cogrouping_counts(groups)
    pct = 100 .* counts ./ length(groups)
    masked_pct = fill(NaN, 6, 6)
    for x in 1:6, y in 1:6
        if y > x
            masked_pct[x, y] = pct[x, y]
        end
    end

    fig = Figure(size = (1250, 1120), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Window-pair co-grouping frequency",
          fontsize = 42, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "Lower triangle only: frequency that two distinct windows share a similarity group across 162 geologies.",
          fontsize = 24, color = MUTED, tellwidth = false)
    ax = Axis(fig[3, 1],
              xlabel = "Throw window",
              ylabel = "Throw window",
              xticks = (1:6, WINDOW_LABELS),
              yticks = (1:6, WINDOW_LABELS),
              yreversed = true,
              aspect = DataAspect(),
              xlabelsize = 30,
              ylabelsize = 30,
              xticklabelsize = 28,
              yticklabelsize = 28)
    hm = heatmap!(ax, 1:6, 1:6, masked_pct;
                  colormap = cgrad([RGBf(0.96, 0.97, 0.98), RGBf(0.70, 0.82, 0.91), BLUE]),
                  colorrange = (0, 100),
                  nan_color = RGBf(1, 1, 1))
    for x in 1:6, y in 1:6
        y <= x && continue
        value = pct[x, y]
        text!(ax, x, y; text = "$(round(Int, value))%",
              align = (:center, :center), fontsize = 24, font = :bold,
              color = value >= 55 ? :white : INK)
    end
    Colorbar(fig[3, 2], hm; label = "Co-grouping frequency (%)",
             labelsize = 24, ticklabelsize = 22, ticks = [0, 25, 50, 75, 100])
    paths = save_figure(fig, output_root, "09_window_pair_cogrouping_frequency")
    return paths
end

"""
    save_singleton_frequency_by_window(groups, output_root)

Figure 10: how often each throw window is selected as a singleton similarity
group and therefore kept independent in grouped low/high cases.
"""
function save_singleton_frequency_by_window(groups, output_root)
    counts = singleton_counts_by_window(groups)
    total = length(groups)
    max_count = maximum(counts)

    fig = Figure(size = (1350, 900), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Singleton frequency by window",
          fontsize = 42, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "A singleton window has no stable similarity partner and is treated independently in grouped low/high cases.",
          fontsize = 24, color = MUTED, tellwidth = false)
    ax = Axis(fig[3, 1],
              xlabel = "Number of geologies",
              ylabel = "Throw window",
              yticks = (1:6, WINDOW_LABELS),
              yreversed = true,
              limits = ((0, max_count * 1.22), (0.35, 6.65)),
              xlabelsize = 30,
              ylabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 28,
              xgridvisible = true,
              ygridvisible = false,
              xgridcolor = GRID)
    for i in 1:6
        rectangle!(ax, 0, i - 0.36, counts[i], 0.72;
                   color = BLUE, strokecolor = BLUE, strokewidth = 1)
        pct = 100 * counts[i] / total
        text!(ax, counts[i] + max_count * 0.025, i;
              text = "$(counts[i]) ($(round(pct; digits = 1))%)",
              align = (:left, :center), fontsize = 24, color = INK)
    end
    hidespines!(ax, :t, :r)
    paths = save_figure(fig, output_root, "10_singleton_frequency_by_window")
    return paths
end

"""
    save_faultwide_coupling_support_summary(groups, output_root)

Figure 11: compact summary showing whether the selected Level 3 similarity
groups support a single fault-wide coupling assumption.
"""
function save_faultwide_coupling_support_summary(groups, output_root)
    categories = [
        "Fault-wide one group",
        "Grouped, no singleton",
        "Partial groups + singletons",
        "All windows singleton",
    ]
    counts = Dict(label => 0 for label in categories)
    for row in groups
        parsed = parse_similarity_groups(row["groups"])
        group_sizes = length.(parsed)
        if length(parsed) == 1 && group_sizes[1] == 6
            counts["Fault-wide one group"] += 1
        elseif all(==(1), group_sizes)
            counts["All windows singleton"] += 1
        elseif any(==(1), group_sizes)
            counts["Partial groups + singletons"] += 1
        else
            counts["Grouped, no singleton"] += 1
        end
    end
    labels = categories
    total = length(groups)
    max_count = maximum(values(counts))

    fig = Figure(size = (1650, 1000), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Why adaptive cross-window grouping is needed",
          fontsize = 42, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "A single fault-wide similarity group never appears; most geologies require partial grouping and independent singletons.",
          fontsize = 24, color = MUTED, tellwidth = false)
    ax = Axis(fig[3, 1],
              xlabel = "Number of geologies",
              ylabel = "",
              yticks = (1:length(labels), labels),
              yreversed = true,
              limits = ((0, max_count * 1.22), (0.35, length(labels) + 0.65)),
              xlabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 27,
              xgridvisible = true,
              ygridvisible = false,
              xgridcolor = GRID)
    for (yi, label) in enumerate(labels)
        count_value = counts[label]
        color = count_value == 0 ? LIGHT_GREY : (label == "Partial groups + singletons" ? BLUE : RGBf(0.55, 0.63, 0.72))
        rectangle!(ax, 0, yi - 0.34, count_value, 0.68;
                   color = color, strokecolor = count_value == 0 ? RGBf(0.78, 0.79, 0.82) : color,
                   strokewidth = 1.0)
        pct = 100 * count_value / total
        text!(ax, max(count_value, 0.7) + max_count * 0.025, yi;
              text = "$(count_value) ($(round(pct; digits = 1))%)",
              align = (:left, :center), fontsize = 24,
              color = count_value == 0 ? MUTED : INK)
    end
    hidespines!(ax, :t, :r)
    paths = save_figure(fig, output_root, "11_faultwide_coupling_support_summary")
    return paths
end

"""
    save_low_high_state_separation(state_separations, output_root)

Figure 12: distribution of the high-minus-low median log-permeability
separation produced by Level 2 state-library construction.
"""
function save_low_high_state_separation(state_separations, output_root)
    fields = ["delta_log_kxx", "delta_log_kyy", "delta_log_kzz"]
    labels = ["kxx", "kyy", "kzz"]
    values_by_component = Dict(label => parse.(Float64, getfield_vec(state_separations, field))
                               for (label, field) in zip(labels, fields))
    max_x = maximum(vcat(values(values_by_component)...)) * 1.08

    fig = Figure(size = (1550, 900), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1], "Low/high state separation from Level 2 reduction",
          fontsize = 42, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1],
          "Each row summarizes 972 geology-window libraries; Δ = median(high state) − median(low state) in log10(k).",
          fontsize = 24, color = MUTED, tellwidth = false)
    ax = Axis(fig[3, 1],
              xlabel = "Δ median log10(k) [high state − low state]",
              ylabel = "Permeability component",
              yticks = (1:3, labels),
              limits = ((0, max_x), (0.45, 3.55)),
              xlabelsize = 30,
              ylabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 28,
              xgridvisible = true,
              ygridvisible = false,
              xgridcolor = GRID)
    vlines!(ax, [0]; color = (:black, 0.35), linewidth = 2, linestyle = :dash)
    rng = MersenneTwister(2026)
    for (yi, label) in enumerate(labels)
        vals = values_by_component[label]
        jitter = [yi + 0.38 * (rand(rng) - 0.5) for _ in vals]
        scatter!(ax, vals, jitter; color = (BLUE, 0.12), markersize = 7,
                 strokewidth = 0)
        q05, q25, q50, q75, q95 = quantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])
        lines!(ax, [q05, q95], [yi, yi]; color = RGBf(0.48, 0.53, 0.60), linewidth = 4)
        lines!(ax, [q25, q75], [yi, yi]; color = BLUE, linewidth = 13)
        scatter!(ax, [q50], [yi]; color = GOLD, markersize = 26,
                 strokecolor = INK, strokewidth = 1.6)
        text!(ax, q95 + max_x * 0.025, yi;
              text = "median $(round(q50; digits = 2))",
              align = (:left, :center), fontsize = 23, color = INK)
    end
    hidespines!(ax, :t, :r)
    paths = save_figure(fig, output_root, "12_low_high_state_separation")
    return paths
end

"""
    save_complexity_vs_grouping(level2, groups, output_root)

Figure 5: relationship between within-window cluster complexity and Level 3
cross-window group count.
"""
function save_complexity_vs_grouping(level2, groups, output_root)
    by_geo_level2 = Dict{String, Vector{Dict{String, String}}}()
    for r in level2
        push!(get!(by_geo_level2, r["geology_id"], Dict{String, String}[]), r)
    end
    rng = MersenneTwister(2718)
    xs = Float64[]
    ys = Float64[]
    colors = RGBf[]
    for g in groups
        rows = by_geo_level2[g["geology_id"]]
        multi = count(r -> parse(Int, r["chosen_k"]) > 1, rows)
        group_count = parse(Int, g["group_count"])
        push!(xs, multi + 0.15 * (rand(rng) - 0.5))
        push!(ys, group_count + 0.15 * (rand(rng) - 0.5))
        depth = parse(Float64, g["faulting_depth_m"])
        push!(colors, depth_color(depth))
    end
    fig = Figure(size = (1700, 1250), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1:2], "Within-window complexity versus cross-window grouping",
          fontsize = 42, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1:2],
          "Each point is one geology; x counts windows with K > 1, y is the selected Level 3 group count.",
          fontsize = 26, color = MUTED, tellwidth = false)
    ax = Axis(fig[3, 1],
              xlabel = "Multi-cluster windows per geology (out of 6)",
              ylabel = "Level 3 window similarity group count",
              xticks = 0:6,
              yticks = 1:6,
              limits = ((-0.5, 6.5), (1.5, 6.5)),
              xlabelsize = 30,
              ylabelsize = 30,
              xticklabelsize = 26,
              yticklabelsize = 26)
    scatter!(ax, xs, ys; color = colors, markersize = 20,
             strokecolor = (:white, 0.8), strokewidth = 1.1)
    Legend(fig[3, 2],
           [MarkerElement(color = depth_color(d), marker = :circle, markersize = 20,
                          strokecolor = :white) for d in (50.0, 500.0, 1000.0)],
           ["50 m", "500 m", "1000 m"];
           framevisible = false, labelsize = 24,
           title = "Fault depth", titlesize = 25)
    paths = save_figure(fig, output_root, "05_complexity_vs_grouping")
    return paths
end

"""
    save_sampled_kzz_case_matrix(sampled, output_root, scenario_order)

Figure 6: sampled 6-window `log10(kzz)` values for all 10 cases per geology.
"""
function save_sampled_kzz_case_matrix(sampled, output_root, scenario_order)
    fig = Figure(size = (2500, 2200), fontsize = 28, backgroundcolor = :white)
    Label(fig[1, 1:2], "Sampled six-window permeability cases",
          fontsize = 44, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1:2],
          "Each row is one sampled multiple-window case; columns are windows; color is sampled log10(kzz).",
          fontsize = 26, color = MUTED, tellwidth = false)
    cmap = cgrad([RGBf(0.10, 0.15, 0.35), RGBf(0.22, 0.55, 0.72), RGBf(0.95, 0.84, 0.47), RGBf(0.85, 0.30, 0.20)])
    for (panel, scenario_index) in enumerate(scenario_order)
        row = 3 + div(panel - 1, 2)
        col = 1 + mod(panel - 1, 2)
        scenario_rows = [r for r in sampled if r["scenario_index"] == scenario_index]
        keys = sort(unique((parse(Int, r["case_index"]), parse(Int, r["case_id"])) for r in scenario_rows))
        mat = fill(NaN, length(keys), 6)
        key_index = Dict(key => idx for (idx, key) in enumerate(keys))
        for r in scenario_rows
            yi = key_index[(parse(Int, r["case_index"]), parse(Int, r["case_id"]))]
            wi = window_index(r["window"])
            mat[yi, wi] = parse(Float64, r["log_kzz"])
        end
        ax = Axis(fig[row, col],
                  title = short_scenario_label(scenario_order, sampled, scenario_index),
                  xlabel = "Window",
                  ylabel = col == 1 ? "Geology cases × 10 designs" : "",
                  xticks = (1:6, WINDOW_LABELS),
                  yticks = ([1, 90, 180, 270], ["1", "90", "180", "270"]),
                  yreversed = true,
                  titlesize = 30,
                  xlabelsize = 25,
                  ylabelsize = 25,
                  xticklabelsize = 24,
                  yticklabelsize = 20)
        heatmap!(ax, 1:6, 1:length(keys), permutedims(mat); colormap = cmap, colorrange = (-7, 2))
        for boundary in 10:10:length(keys)
            hlines!(ax, [boundary + 0.5]; color = (:white, 0.35), linewidth = 0.8)
        end
    end
    Colorbar(fig[3:5, 3], limits = (-7, 2), colormap = cmap,
             ticks = [-7, -4, -1, 2], label = "log10(kzz) [mD]",
             labelsize = 26, ticklabelsize = 24, width = 28)
    paths = save_figure(fig, output_root, "06_sampled_six_window_kzz_case_matrix")
    return paths
end

"""
    save_sampled_case_type_component_summary(sampled, output_root)

Figure 7: median and 10-90% interval of sampled log permeability by case class
and component.
"""
function save_sampled_case_type_component_summary(sampled, output_root)
    classes = ["Independent", "Fault-wide low", "Fault-wide high",
               "Grouped low", "Grouped high", "Grouped singleton"]
    class_rows = Dict(class => Dict(comp => Float64[] for comp in COMPONENTS) for class in classes)
    for r in sampled
        class = sampled_case_class(r)
        haskey(class_rows, class) || continue
        for (comp, field) in zip(COMPONENTS, LOG_FIELDS)
            push!(class_rows[class][comp], parse(Float64, r[field]))
        end
    end
    fig = Figure(size = (2400, 1250), fontsize = 30, backgroundcolor = :white)
    Label(fig[1, 1:3], "Sampled permeability levels by case class",
          fontsize = 44, font = :bold, color = INK, tellwidth = false)
    Label(fig[2, 1:3],
          "Dots are medians; horizontal bars are 10th-90th percentile ranges from sampled case-window assignments.",
          fontsize = 26, color = MUTED, tellwidth = false)
    colors = Dict("Independent" => CHARCOAL,
                  "Fault-wide low" => BLUE,
                  "Fault-wide high" => ROSE,
                  "Grouped low" => TEAL,
                  "Grouped high" => GOLD,
                  "Grouped singleton" => RGBf(0.50, 0.50, 0.55))
    for (idx, comp) in enumerate(COMPONENTS)
        ax = Axis(fig[3, idx],
                  title = "log10($comp)",
                  xlabel = "log10(k) [mD]",
                  ylabel = idx == 1 ? "Case class" : "",
                  yticks = (1:length(classes), classes),
                  limits = ((-7, 2), (0.5, length(classes) + 0.5)),
                  titlesize = 32,
                  xlabelsize = 27,
                  ylabelsize = 27,
                  xticklabelsize = 24,
                  yticklabelsize = 22)
        vlines!(ax, [0.0]; color = GRID, linestyle = :dash, linewidth = 1.5)
        for (yi, class) in enumerate(classes)
            values = class_rows[class][comp]
            isempty(values) && continue
            q10, q50, q90 = quantile(values, [0.10, 0.50, 0.90])
            lines!(ax, [q10, q90], [yi, yi]; color = colors[class], linewidth = 5)
            scatter!(ax, [q50], [yi]; color = colors[class], markersize = 22,
                     strokecolor = :white, strokewidth = 1.4)
        end
    end
    paths = save_figure(fig, output_root, "07_sampled_case_type_component_summary")
    return paths
end

"""
    write_retained_supporting_tables(level2, groups, output_root)

Write compact data tables that back only the retained result-analysis figures.
"""
function write_retained_supporting_tables(level2, groups, output_root)
    table_root = joinpath(output_root, "tables")
    mkpath(table_root)

    cluster_count_rows = Vector{Vector{String}}()
    for (wi, window) in enumerate(WINDOWS)
        window_rows = [r for r in level2 if r["window"] == window]
        total = length(window_rows)
        for k in 1:5
            count_value = count(r -> parse(Int, r["chosen_k"]) == k, window_rows)
            push!(cluster_count_rows, [
                WINDOW_LABELS[wi],
                window,
                string(k),
                string(count_value),
                fmt(count_value / total),
            ])
        end
    end
    write_csv(joinpath(table_root, "level2_cluster_count_distribution_by_window.csv"),
              ["window_label", "window", "chosen_k", "geology_count", "fraction_of_window_geologies"],
              cluster_count_rows)

    structures = sort(collect(keys(STRUCTURE_COLORS)); by = structure_sort_key)
    total = length(groups)
    overall_rows = Vector{Vector{String}}()
    for structure in sort(structures; by = s -> (-count(r -> r["similarity_group_structure"] == s, groups), structure_sort_key(s)))
        count_value = count(r -> r["similarity_group_structure"] == structure, groups)
        push!(overall_rows, [
            structure,
            string(count_value),
            fmt(count_value / total),
        ])
    end
    write_csv(joinpath(table_root, "similarity_structure_frequency_all_geologies.csv"),
              ["similarity_group_structure", "count", "fraction_of_geologies"],
              overall_rows)

    cogroup = window_pair_cogrouping_counts(groups)
    cogroup_rows = Vector{Vector{String}}()
    for i in 1:6, j in 1:6
        push!(cogroup_rows, [
            WINDOW_LABELS[i],
            WINDOW_LABELS[j],
            string(cogroup[i, j]),
            fmt(cogroup[i, j] / length(groups)),
        ])
    end
    write_csv(joinpath(table_root, "window_pair_cogrouping_frequency.csv"),
              ["window_i", "window_j", "cogroup_count", "cogroup_fraction"],
              cogroup_rows)
end

"""
    write_supporting_tables(level2, groups, sampled, output_root)

Write compact data tables that back the figures.
"""
function write_supporting_tables(level2, groups, sampled, state_separations, output_root)
    table_root = joinpath(output_root, "tables")
    mkpath(table_root)

    rows = Vector{Vector{String}}()
    for field in ["faulting_depth_m", "sand_vcl", "clay_vcl"]
        for value in sort(unique_values(level2, field); by = x -> parse(Float64, x))
            subset = [r for r in level2 if r[field] == value]
            mean_k = mean(parse.(Float64, getfield_vec(subset, "chosen_k")))
            frac_multi = mean([parse(Int, r["chosen_k"]) > 1 ? 1.0 : 0.0 for r in subset])
            push!(rows, [field, value, string(length(subset)), fmt(mean_k), fmt(frac_multi)])
        end
    end
    write_csv(joinpath(table_root, "cluster_complexity_parameter_summary.csv"),
              ["parameter", "value", "window_library_count", "mean_chosen_k", "fraction_multicluster"],
              rows)

    cluster_count_rows = Vector{Vector{String}}()
    for (wi, window) in enumerate(WINDOWS)
        window_rows = [r for r in level2 if r["window"] == window]
        total = length(window_rows)
        for k in 1:5
            count_value = count(r -> parse(Int, r["chosen_k"]) == k, window_rows)
            push!(cluster_count_rows, [
                WINDOW_LABELS[wi],
                window,
                string(k),
                string(count_value),
                fmt(count_value / total),
            ])
        end
    end
    write_csv(joinpath(table_root, "level2_cluster_count_distribution_by_window.csv"),
              ["window_label", "window", "chosen_k", "geology_count", "fraction_of_window_geologies"],
              cluster_count_rows)

    structures = sort(collect(keys(STRUCTURE_COLORS)); by = structure_sort_key)
    structure_rows = Vector{Vector{String}}()
    for scenario in sort(unique_values(groups, "scenario_index"); by = x -> parse(Int, x))
        scenario_rows = [r for r in groups if r["scenario_index"] == scenario]
        for structure in structures
            push!(structure_rows, [
                scenario,
                short_scenario_label(sort(unique_values(groups, "scenario_index"); by = x -> parse(Int, x)), groups, scenario),
                structure,
                string(count(r -> r["similarity_group_structure"] == structure, scenario_rows)),
            ])
        end
    end
    write_csv(joinpath(table_root, "similarity_structure_counts_by_scenario.csv"),
              ["scenario_index", "scenario_label", "similarity_group_structure", "count"],
              structure_rows)

    total = length(groups)
    overall_rows = Vector{Vector{String}}()
    for structure in sort(structures; by = s -> (-count(r -> r["similarity_group_structure"] == s, groups), structure_sort_key(s)))
        count_value = count(r -> r["similarity_group_structure"] == structure, groups)
        push!(overall_rows, [
            structure,
            string(count_value),
            fmt(count_value / total),
        ])
    end
    write_csv(joinpath(table_root, "similarity_structure_frequency_all_geologies.csv"),
              ["similarity_group_structure", "count", "fraction_of_geologies"],
              overall_rows)

    cogroup = window_pair_cogrouping_counts(groups)
    cogroup_rows = Vector{Vector{String}}()
    for i in 1:6, j in 1:6
        push!(cogroup_rows, [
            WINDOW_LABELS[i],
            WINDOW_LABELS[j],
            string(cogroup[i, j]),
            fmt(cogroup[i, j] / length(groups)),
        ])
    end
    write_csv(joinpath(table_root, "window_pair_cogrouping_frequency.csv"),
              ["window_i", "window_j", "cogroup_count", "cogroup_fraction"],
              cogroup_rows)

    singleton_counts = singleton_counts_by_window(groups)
    singleton_rows = Vector{Vector{String}}()
    for i in 1:6
        push!(singleton_rows, [
            WINDOW_LABELS[i],
            WINDOWS[i],
            string(singleton_counts[i]),
            fmt(singleton_counts[i] / length(groups)),
        ])
    end
    write_csv(joinpath(table_root, "singleton_frequency_by_window.csv"),
              ["window_label", "window", "singleton_count", "singleton_fraction"],
              singleton_rows)

    support_counts = Dict(
        "Fault-wide one group" => 0,
        "Grouped, no singleton" => 0,
        "Partial groups + singletons" => 0,
        "All windows singleton" => 0,
    )
    for row in groups
        parsed = parse_similarity_groups(row["groups"])
        group_sizes = length.(parsed)
        if length(parsed) == 1 && group_sizes[1] == 6
            support_counts["Fault-wide one group"] += 1
        elseif all(==(1), group_sizes)
            support_counts["All windows singleton"] += 1
        elseif any(==(1), group_sizes)
            support_counts["Partial groups + singletons"] += 1
        else
            support_counts["Grouped, no singleton"] += 1
        end
    end
    support_rows = [[label, string(support_counts[label]), fmt(support_counts[label] / length(groups))]
                    for label in ["Fault-wide one group", "Grouped, no singleton",
                                  "Partial groups + singletons", "All windows singleton"]]
    write_csv(joinpath(table_root, "faultwide_coupling_support_summary.csv"),
              ["category", "count", "fraction_of_geologies"],
              support_rows)

    separation_summary_rows = Vector{Vector{String}}()
    for (component, field) in zip(COMPONENTS, ["delta_log_kxx", "delta_log_kyy", "delta_log_kzz"])
        vals = parse.(Float64, getfield_vec(state_separations, field))
        q05, q25, q50, q75, q95 = quantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])
        push!(separation_summary_rows, [
            component,
            string(length(vals)),
            fmt(q05), fmt(q25), fmt(q50), fmt(q75), fmt(q95),
        ])
    end
    write_csv(joinpath(table_root, "level2_low_high_state_separation_summary.csv"),
              ["component", "n_window_libraries", "q05", "q25", "median", "q75", "q95"],
              separation_summary_rows)

    separation_rows = [[r["geology_id"], r["scenario_index"], r["case_index"], r["window"],
                        r["delta_log_kxx"], r["delta_log_kyy"], r["delta_log_kzz"]]
                       for r in state_separations]
    write_csv(joinpath(table_root, "level2_low_high_state_separation_by_window_library.csv"),
              ["geology_id", "scenario_index", "case_index", "window",
               "delta_log_kxx", "delta_log_kyy", "delta_log_kzz"],
              separation_rows)
end

"""
    compute_low_high_state_separations(level2)

Read each saved Level 2 state object and compute component-wise separation
between high- and low-state libraries. The separation is:

`median(log10(k) in high state) - median(log10(k) in low state)`.
"""
function compute_low_high_state_separations(level2)
    rows = Vector{Dict{String, String}}()
    for r in level2
        vars = matread(normpath(r["state_path"]))
        log_perms = Matrix{Float64}(vars["log_perms"])
        low_indices = vec(Int.(vars["low_indices"]))
        high_indices = vec(Int.(vars["high_indices"]))
        low_medians = [median(log_perms[low_indices, c]) for c in 1:3]
        high_medians = [median(log_perms[high_indices, c]) for c in 1:3]
        delta = high_medians .- low_medians
        push!(rows, Dict(
            "geology_id" => r["geology_id"],
            "scenario_index" => r["scenario_index"],
            "case_index" => r["case_index"],
            "window" => r["window"],
            "delta_log_kxx" => fmt(delta[1]),
            "delta_log_kyy" => fmt(delta[2]),
            "delta_log_kzz" => fmt(delta[3]),
        ))
    end
    return rows
end

"""
    parse_similarity_groups(group_string)

Convert a semicolon/plus encoded grouping such as
`famp1;famp2+famp3;famp4+famp5;famp6` into a vector of window groups.
"""
function parse_similarity_groups(group_string::AbstractString)
    return [String.(split(group_text, "+")) for group_text in split(group_string, ";") if !isempty(group_text)]
end

"""
    window_pair_cogrouping_counts(groups)

Return a 6 x 6 matrix where entry (i, j) is the number of geologies in which
windows i and j are selected into the same similarity group.
"""
function window_pair_cogrouping_counts(groups)
    counts = zeros(Int, 6, 6)
    for row in groups
        parsed = parse_similarity_groups(row["groups"])
        for group in parsed
            idx = [window_index(w) for w in group]
            for i in idx, j in idx
                counts[i, j] += 1
            end
        end
    end
    return counts
end

"""
    singleton_counts_by_window(groups)

Return how often each window appears as a singleton group across geologies.
"""
function singleton_counts_by_window(groups)
    counts = zeros(Int, 6)
    for row in groups
        parsed = parse_similarity_groups(row["groups"])
        for group in parsed
            if length(group) == 1
                counts[window_index(group[1])] += 1
            end
        end
    end
    return counts
end

function save_figure(fig::Figure, output_root::AbstractString, basename_no_ext::AbstractString)
    paths = [joinpath(output_root, "$(basename_no_ext).png"),
             joinpath(output_root, "$(basename_no_ext).pdf")]
    for path in paths
        save(path, fig)
    end
    return paths
end

"""
    remove_dropped_result_figures(output_root)

Remove older result-analysis figure files that are no longer part of the
retained visualization set. Only explicit, known basenames inside
`output_root` are removed.
"""
function remove_dropped_result_figures(output_root::AbstractString)
    for basename in DROPPED_FIGURE_BASENAMES
        for ext in (".png", ".pdf")
            path = joinpath(output_root, basename * ext)
            isfile(path) && rm(path)
        end
    end
end

function scenario_title(row::Dict{String, String})
    return "S$(row["scenario_index"]): " * titlecase(replace(row["scenario_name"], "," => " |"))
end

function short_scenario_label(scenario_order, rows, scenario_index)
    row = first(r for r in rows if r["scenario_index"] == scenario_index)
    name = replace(row["scenario_name"], "," => " |")
    return "S$(scenario_index): " * titlecase(name)
end

function window_index(window::AbstractString)
    idx = findfirst(==(String(window)), WINDOWS)
    idx === nothing && error("Unknown window: $window")
    return idx
end

function getfield_vec(rows, field)
    return [r[field] for r in rows]
end

unique_values(rows, field) = sort(collect(Set(r[field] for r in rows)))

function structure_sort_key(structure::AbstractString)
    parts = parse.(Int, split(structure, "+"))
    return (length(parts), -maximum(parts), structure)
end

function text_color_for_group_count(value)
    !isfinite(value) && return INK
    return value >= 4.0 ? :white : INK
end

function depth_color(depth::Real)
    depth == 50.0 && return BLUE
    depth == 500.0 && return GOLD
    depth == 1000.0 && return ROSE
    return CHARCOAL
end

function sampled_case_class(row::Dict{String, String})
    category = row["case_category"]
    state = row["assigned_state"]
    if category == "Independent cases"
        return "Independent"
    elseif category == "Fault-wide low/high cases"
        return state == "low" ? "Fault-wide low" : "Fault-wide high"
    elseif category == "Grouped low/high cases"
        state == "low" && return "Grouped low"
        state == "high" && return "Grouped high"
        return "Grouped singleton"
    end
    return "Other"
end

function rectangle!(ax, x, y, w, h; color, strokecolor = :white, strokewidth = 1.0)
    poly!(ax, Point2f[(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
          color = color, strokecolor = strokecolor, strokewidth = strokewidth)
end

function luminance(c::RGBf)
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
end

fmt(x::Real) = string(round(Float64(x), digits = 6))

function read_csv_dicts(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("CSV is empty: $path")
    header = parse_csv_line(lines[1])
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = parse_csv_line(line)
        length(values) == length(header) || error("Malformed CSV row in $path")
        push!(rows, Dict(key => value for (key, value) in zip(header, values)))
    end
    return rows
end

function parse_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        ch = line[i]
        if ch == '"'
            next_i = nextind(line, i)
            if in_quotes && next_i <= lastindex(line) && line[next_i] == '"'
                write(buf, '"')
                i = nextind(line, next_i)
                continue
            else
                in_quotes = !in_quotes
            end
        elseif ch == ',' && !in_quotes
            push!(fields, String(take!(buf)))
        else
            write(buf, ch)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(csv_escape.(header), ","))
        for row in rows
            length(row) == length(header) || error("Row length does not match header for $path")
            println(io, join(csv_escape.(row), ","))
        end
    end
end

function csv_escape(value::AbstractString)
    escaped = replace(value, "\"" => "\"\"")
    if occursin(',', escaped) || occursin('"', escaped) || occursin('\n', escaped)
        return "\"" * escaped * "\""
    end
    return escaped
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
