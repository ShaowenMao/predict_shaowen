"""
    Level3Plotting

Publication-oriented plotting helpers for Level 3 cross-window similarity
diagnostics. The current functions visualize Step 2 full-data distance results;
they are diagnostic only because bootstrap stability is not applied yet.
"""
module Level3Plotting

using CairoMakie

export activate_level3_plot_theme!,
       save_step2_distance_heatmap,
       save_step2_sorted_distance_pairs,
       save_step2_diagnostic_figures,
       save_bootstrap_grouping_qa_figures,
       save_stable_pair_probability_heatmap,
       save_window_similarity_group_figures,
       save_window_similarity_group_strip,
       save_multiple_window_permeability_case_figures,
       save_multiple_window_permeability_case_matrix,
       window_display_labels

"""
    activate_level3_plot_theme!()

Activate CairoMakie and set a consistent Level 3 plotting theme.
"""
function activate_level3_plot_theme!()
    CairoMakie.activate!()
    set_theme!(Theme(
        fontsize = 28,
        figure_padding = 24,
        Axis = (
            titlesize = 32,
            xlabelsize = 30,
            ylabelsize = 30,
            xticklabelsize = 26,
            yticklabelsize = 26,
            titlegap = 10,
            xgridvisible = false,
            ygridvisible = false,
            backgroundcolor = RGBf(0.985, 0.985, 0.985),
        ),
        Colorbar = (
            labelsize = 28,
            ticklabelsize = 24,
        ),
    ))
end

"""
    window_display_labels(windows)

Convert internal window names such as `famp1` to compact display labels such
as `W1`.
"""
function window_display_labels(windows::Vector{String})
    labels = String[]
    for window in windows
        m = match(r"(\d+)$", window)
        push!(labels, isnothing(m) ? window : "W$(m.captures[1])")
    end
    return labels
end

"""
    save_step2_diagnostic_figures(distance_result, output_root; kwargs...)

Save all Step 2 full-data diagnostic figures and return their file paths.
"""
function save_step2_diagnostic_figures(distance_result::Dict{String, Any},
                                       output_root::AbstractString;
                                       threshold::Real = 0.25,
                                       formats::Vector{String} = ["png", "pdf"])
    figure_root = joinpath(output_root, "figures", "step02_window_distances")
    mkpath(figure_root)

    saved_paths = String[]
    for fmt in formats
        fmt_clean = lowercase(replace(fmt, "." => ""))
        push!(saved_paths,
              save_step2_distance_heatmap(
                  joinpath(figure_root, "step02_normalized_distance_heatmap.$fmt_clean"),
                  distance_result;
                  threshold = threshold,
              ))
        push!(saved_paths,
              save_step2_sorted_distance_pairs(
                  joinpath(figure_root, "step02_sorted_window_distances.$fmt_clean"),
                  distance_result;
                  threshold = threshold,
              ))
    end
    return saved_paths
end

"""
    save_bootstrap_grouping_qa_figures(bootstrap_result, output_root; kwargs...)

Save bootstrap stability diagnostic figures and return their file paths.
"""
function save_bootstrap_grouping_qa_figures(bootstrap_result::Dict{String, Any},
                                            output_root::AbstractString;
                                            formats::Vector{String} = ["png", "pdf"])
    figure_root = joinpath(output_root, "figures", "step02_bootstrap_grouping_qa")
    mkpath(figure_root)

    saved_paths = String[]
    for fmt in formats
        fmt_clean = lowercase(replace(fmt, "." => ""))
        push!(saved_paths,
              save_stable_pair_probability_heatmap(
                  joinpath(figure_root, "step02_stable_similar_pair_probability_heatmap.$fmt_clean"),
                  bootstrap_result,
              ))
    end
    return saved_paths
end

"""
    save_window_similarity_group_figures(grouping_result, output_root; kwargs...)

Save the window similarity group figure and return its file paths.
"""
function save_window_similarity_group_figures(grouping_result::Dict{String, Any},
                                              output_root::AbstractString;
                                              formats::Vector{String} = ["png", "pdf"])
    figure_root = joinpath(output_root, "figures", "step02_window_similarity_groups")
    mkpath(figure_root)

    saved_paths = String[]
    for fmt in formats
        fmt_clean = lowercase(replace(fmt, "." => ""))
        push!(saved_paths,
              save_window_similarity_group_strip(
                  joinpath(figure_root, "step02_window_similarity_group_strip.$fmt_clean"),
                  grouping_result,
              ))
    end
    return saved_paths
end

"""
    save_window_similarity_group_strip(path, grouping_result)

Save a simple physical-order strip where each window is colored by its final
window similarity group.
"""
function save_window_similarity_group_strip(path::AbstractString,
                                           grouping_result::Dict{String, Any})
    activate_level3_plot_theme!()
    windows = String.(grouping_result["windows"])
    labels = window_display_labels(windows)
    groups = Vector{Vector{String}}(grouping_result["groups"])
    group_ids = group_id_by_window(windows, groups)
    colors = group_palette(length(groups))
    n = length(windows)

    fig = Figure(size = (1580, 540),
                 backgroundcolor = :white,
                 figure_padding = (36, 36, 34, 48))
    ax = Axis(fig[1, 1],
              title = "Window Similarity Groups",
              xlabel = "Physical window order",
              xticks = (1:n, labels),
              yticksvisible = false,
              yticklabelsvisible = false,
              ygridvisible = false,
              xgridvisible = false)
    limits!(ax, 0.35, n + 0.65, 0.45, 1.55)
    hidespines!(ax, :l, :r, :t)

    for i in 1:n
        group_id = group_ids[windows[i]]
        draw_rect!(ax, i - 0.45, i + 0.45, 0.72, 1.28;
                   color = colors[group_id],
                   strokecolor = RGBf(0.12, 0.12, 0.12),
                   strokewidth = 2.0)
        text!(ax, i, 1.05;
              text = labels[i],
              align = (:center, :center),
              fontsize = 32,
              font = :bold,
              color = :white)
        text!(ax, i, 0.83;
              text = "G$group_id",
              align = (:center, :center),
              fontsize = 23,
              color = :white)
    end

    Label(fig[2, 1],
          group_summary_text(groups),
          fontsize = 26,
          tellwidth = false)
    save(path, fig, px_per_unit = 2)
    return path
end

"""
    save_multiple_window_permeability_case_figures(case_result, output_root; kwargs...)

Save Level 3 multiple-window permeability case review figures and return their
file paths.
"""
function save_multiple_window_permeability_case_figures(case_result::Dict{String, Any},
                                                        output_root::AbstractString;
                                                        formats::Vector{String} = ["png", "pdf"])
    figure_root = joinpath(output_root, "figures", "step03_multiple_window_permeability_cases")
    mkpath(figure_root)

    saved_paths = String[]
    for fmt in formats
        fmt_clean = lowercase(replace(fmt, "." => ""))
        push!(saved_paths,
              save_multiple_window_permeability_case_matrix(
                  joinpath(figure_root, "step03_multiple_window_permeability_case_matrix.$fmt_clean"),
                  case_result,
              ))
    end
    return saved_paths
end

"""
    save_multiple_window_permeability_case_matrix(path, case_result)

Save a 10-case by 6-window matrix showing independent, low, and high
permeability-pool assignments.
"""
function save_multiple_window_permeability_case_matrix(path::AbstractString,
                                                       case_result::Dict{String, Any})
    activate_level3_plot_theme!()
    windows = String.(case_result["windows"])
    window_labels = window_display_labels(windows)
    rows = Vector{Dict{String, Any}}(case_result["case_window_table"])
    case_ids = sort(unique(Int(row["case_id"]) for row in rows))
    case_names = Dict(Int(row["case_id"]) => String(row["case_name"]) for row in rows)
    row_by_case_window = Dict((Int(row["case_id"]), String(row["window"])) => row for row in rows)

    value_by_state = Dict("independent" => 0.0, "low" => 1.0, "high" => 2.0)
    state_label = Dict("independent" => "I", "low" => "L", "high" => "H")
    matrix = [value_by_state[String(row_by_case_window[(case_id, window)]["assigned_state"])]
              for case_id in case_ids, window in windows]

    fig = Figure(size = (1640, 980),
                 backgroundcolor = :white,
                 figure_padding = (32, 72, 36, 210))
    ax = Axis(fig[1, 1],
              title = "Multiple-Window Permeability Cases",
              xticks = (1:length(windows), window_labels),
              yticks = (1:length(case_ids), [case_matrix_label(case_id, case_names[case_id]) for case_id in case_ids]),
              xlabel = "Throw window",
              ylabel = "Permeability case",
              yreversed = true,
              xgridvisible = false,
              ygridvisible = false)

    heatmap!(ax, 1:length(windows), 1:length(case_ids), matrix';
             colormap = [RGBf(0.72, 0.72, 0.72), RGBf(0.00, 0.45, 0.70), RGBf(0.90, 0.62, 0.00)],
             colorrange = (0.0, 2.0))

    for (y, case_id) in enumerate(case_ids)
        for (x, window) in enumerate(windows)
            row = row_by_case_window[(case_id, window)]
            state = String(row["assigned_state"])
            text!(ax, x, y;
                  text = state_label[state],
                  align = (:center, :center),
                  fontsize = 28,
                  font = :bold,
                  color = :white)
        end
    end

    legend_ax = Axis(fig[1, 2],
                     width = 210,
                     xticksvisible = false,
                     xticklabelsvisible = false,
                     yticksvisible = false,
                     yticklabelsvisible = false,
                     xgridvisible = false,
                     ygridvisible = false)
    limits!(legend_ax, 0, 1, 0, 1)
    hidespines!(legend_ax)
    draw_legend_patch!(legend_ax, 0.78, RGBf(0.72, 0.72, 0.72), "Independent")
    draw_legend_patch!(legend_ax, 0.57, RGBf(0.00, 0.45, 0.70), "Low state")
    draw_legend_patch!(legend_ax, 0.36, RGBf(0.90, 0.62, 0.00), "High state")

    Label(fig[2, 1:2],
          "Strong = local pools; weak = state-wide pools",
          fontsize = 24,
          tellwidth = false)
    save(path, fig, px_per_unit = 2)
    return path
end

"""
    save_stable_pair_probability_heatmap(path, bootstrap_result)

Save a lower-triangle heatmap of stable-similar-pair probabilities.
"""
function save_stable_pair_probability_heatmap(path::AbstractString,
                                              bootstrap_result::Dict{String, Any})
    activate_level3_plot_theme!()
    windows = String.(bootstrap_result["windows"])
    labels = window_display_labels(windows)
    probability = Matrix{Float64}(bootstrap_result["stable_pair_probability"])
    similarity_threshold = Float64(bootstrap_result["similarity_threshold"])
    stable_threshold = Float64(bootstrap_result["stable_pair_probability_threshold"])
    n = length(windows)

    fig = Figure(size = (1180, 1040),
                 backgroundcolor = :white,
                 figure_padding = (40, 90, 40, 40))
    ax = Axis(fig[1, 1],
              aspect = DataAspect(),
              title = "Stable Similar Pair Probability",
              xticks = (1:n, labels),
              yticks = (1:n, labels),
              xlabel = "Window",
              ylabel = "Window",
              yreversed = true)

    heatmap!(ax, 1:n, 1:n, probability;
             colormap = :viridis,
             colorrange = (0.0, 1.0))

    for i in 1:n
        for j in 1:n
            value = probability[i, j]
            label = i == j ? "1" : probability_label(value)
            text_color = value > 0.55 ? :white : :black
            text!(ax, i, j;
                  text = label,
                  align = (:center, :center),
                  fontsize = 24,
                  color = text_color)
        end
    end

    ticks = collect(range(0.0, 1.0; length = 4))
    cb_ax = Axis(fig[1, 2],
                 width = 34,
                 yaxisposition = :right,
                 xticksvisible = false,
                 xticklabelsvisible = false,
                 yticks = (ticks, [probability_label(tick) for tick in ticks]),
                 ylabel = "Stable-pair probability",
                 ylabelsize = 30,
                 yticklabelsize = 24,
                 xgridvisible = false,
                 ygridvisible = false)
    values = collect(range(0.0, 1.0; length = 256))
    heatmap!(cb_ax,
             [0.0, 1.0],
             values,
             repeat(reshape(values, 1, :), 2, 1);
             colormap = :viridis,
             colorrange = (0.0, 1.0))
    xlims!(cb_ax, 0.0, 1.0)
    ylims!(cb_ax, 0.0, 1.0)
    hidespines!(cb_ax, :t, :b, :l)

    for i in 1:n
        for j in 1:n
            j <= i && continue
            poly!(ax,
                  Point2f[(j - 0.5, i - 0.5), (j + 0.5, i - 0.5),
                           (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)];
                  color = :white,
                  strokecolor = :white)
        end
    end

    Label(fig[0, 1:2],
          "C_ij = P(normalized distance <= $(round(similarity_threshold, digits = 2))); stable if C_ij >= $(round(stable_threshold, digits = 2))",
          fontsize = 24,
          tellwidth = false)
    save(path, fig, px_per_unit = 2)
    return path
end

"""
    save_step2_distance_heatmap(path, distance_result; threshold=0.25)

Save a heatmap of the normalized full-data energy-distance matrix.
"""
function save_step2_distance_heatmap(path::AbstractString,
                                     distance_result::Dict{String, Any};
                                     threshold::Real = 0.25)
    activate_level3_plot_theme!()
    windows = String.(distance_result["windows"])
    labels = window_display_labels(windows)
    normalized = Matrix{Float64}(distance_result["normalized_distance"])
    n = length(windows)
    vmax = max(maximum(normalized), Float64(threshold))

    fig = Figure(size = (1180, 1040),
                 backgroundcolor = :white,
                 figure_padding = (40, 90, 40, 40))
    ax = Axis(fig[1, 1],
              aspect = DataAspect(),
              title = "Full-Data Window Distance",
              xticks = (1:n, labels),
              yticks = (1:n, labels),
              xlabel = "Window",
              ylabel = "Window",
              yreversed = true)

    hm = heatmap!(ax, 1:n, 1:n, normalized;
                  colormap = :viridis,
                  colorrange = (0.0, vmax))

    for i in 1:n
        for j in 1:n
            value = normalized[i, j]
            text_color = value > 0.55 * vmax ? :white : :black
            label = i == j ? "0" : distance_label(value)
            text!(ax, i, j;
                  text = label,
                  align = (:center, :center),
                  fontsize = 24,
                  color = text_color)
        end
    end

    colorbar_ticks = collect(range(0.0, vmax; length = 4))
    cb_ax = Axis(fig[1, 2],
                 width = 34,
                 yaxisposition = :right,
                 xticksvisible = false,
                 xticklabelsvisible = false,
                 yticks = (colorbar_ticks, [distance_label(tick) for tick in colorbar_ticks]),
                 ylabel = "Normalized energy distance",
                 ylabelsize = 30,
                 yticklabelsize = 24,
                 xgridvisible = false,
                 ygridvisible = false)
    colorbar_values = collect(range(0.0, vmax; length = 256))
    heatmap!(cb_ax,
             [0.0, 1.0],
             colorbar_values,
             repeat(reshape(colorbar_values, 1, :), 2, 1);
             colormap = :viridis,
             colorrange = (0.0, vmax))
    xlims!(cb_ax, 0.0, 1.0)
    ylims!(cb_ax, 0.0, vmax)
    hidespines!(cb_ax, :t, :b, :l)
    for i in 1:n
        for j in 1:n
            j <= i && continue
            poly!(ax,
                  Point2f[(j - 0.5, i - 0.5), (j + 0.5, i - 0.5),
                           (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)];
                  color = :white,
                  strokecolor = :white)
        end
    end
    save(path, fig, px_per_unit = 2)
    return path
end

"""
    save_step2_sorted_distance_pairs(path, distance_result; threshold=0.25)

Save a sorted bar plot of all pairwise normalized window distances.
"""
function save_step2_sorted_distance_pairs(path::AbstractString,
                                          distance_result::Dict{String, Any};
                                          threshold::Real = 0.25)
    activate_level3_plot_theme!()
    windows = String.(distance_result["windows"])
    labels = window_display_labels(windows)
    normalized = Matrix{Float64}(distance_result["normalized_distance"])
    n = length(windows)

    pairs = Tuple{String, Float64}[]
    for i in 1:n-1
        for j in i+1:n
            push!(pairs, ("$(labels[i])-$(labels[j])", normalized[i, j]))
        end
    end
    sort!(pairs, by = last)
    pair_labels = first.(pairs)
    distances = last.(pairs)
    colors = [d <= threshold ? RGBf(0.18, 0.48, 0.32) : RGBf(0.65, 0.65, 0.65)
              for d in distances]

    fig = Figure(size = (1540, 900), backgroundcolor = :white)
    ax = Axis(fig[1, 1],
              title = "Sorted Window-Pair Distances",
              xlabel = "Window pair",
              ylabel = "Normalized energy distance",
              xticks = (1:length(pair_labels), pair_labels),
              xticklabelrotation = pi / 4,
              ygridvisible = true,
              xgridvisible = false)

    barplot!(ax, 1:length(distances), distances;
             color = colors,
             strokewidth = 0.5,
             strokecolor = RGBf(0.15, 0.15, 0.15))
    hlines!(ax, [threshold];
            color = RGBf(0.82, 0.20, 0.18),
            linewidth = 4,
            linestyle = :dash)
    text!(ax, length(distances) - 1.5, threshold;
          text = "threshold = $(round(Float64(threshold), digits = 2))",
          align = (:right, :bottom),
          fontsize = 24,
          color = RGBf(0.55, 0.08, 0.06))

    y_max = max(maximum(distances), Float64(threshold)) * 1.14
    limits!(ax, 0.35, length(distances) + 0.65, 0, y_max)
    save(path, fig, px_per_unit = 2)
    return path
end

function distance_label(value::Real)
    value_float = Float64(value)
    value_float == 0.0 && return "0"
    return value_float < 0.01 ? string(round(value_float, digits = 3)) :
           string(round(value_float, digits = 2))
end

function probability_label(value::Real)
    return string(round(Float64(value), digits = 2))
end

function group_palette(n_groups::Integer)
    base = RGBf[
        RGBf(0.00, 0.45, 0.70),
        RGBf(0.90, 0.62, 0.00),
        RGBf(0.00, 0.62, 0.45),
        RGBf(0.80, 0.47, 0.65),
        RGBf(0.84, 0.37, 0.00),
        RGBf(0.35, 0.70, 0.90),
    ]
    return [base[mod1(i, length(base))] for i in 1:n_groups]
end

function group_id_by_window(windows::Vector{String}, groups::Vector{Vector{String}})
    ids = Dict{String, Int}()
    for (group_id, group) in enumerate(groups)
        for window in group
            ids[window] = group_id
        end
    end
    for window in windows
        haskey(ids, window) || error("Grouping result is missing window $window")
    end
    return ids
end

function group_summary_text(groups::Vector{Vector{String}})
    parts = String[]
    for (group_id, group) in enumerate(groups)
        labels = window_display_labels(group)
        push!(parts, "G$group_id: " * join(labels, "+"))
    end
    return join(parts, "     ")
end

function case_matrix_label(case_id::Int, case_name::AbstractString)
    readable = replace(String(case_name), "_" => " ")
    return "C$(lpad(case_id, 2, '0'))  $readable"
end

function draw_legend_patch!(ax, y::Real, color, label::AbstractString)
    draw_rect!(ax, 0.08, 0.25, y - 0.06, y + 0.06;
               color = color,
               strokecolor = RGBf(0.20, 0.20, 0.20),
               strokewidth = 1.0)
    text!(ax, 0.33, y;
          text = String(label),
          align = (:left, :center),
          fontsize = 23,
          color = RGBf(0.10, 0.10, 0.10))
end

function draw_rect!(ax, x0::Real, x1::Real, y0::Real, y1::Real;
                    color,
                    strokecolor = :transparent,
                    strokewidth::Real = 0.0)
    poly!(ax,
          Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
          color = color,
          strokecolor = strokecolor,
          strokewidth = strokewidth)
end

end
