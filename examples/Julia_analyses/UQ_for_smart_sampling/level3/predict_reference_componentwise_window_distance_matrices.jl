#!/usr/bin/env julia

"""
Level-3 cross-window componentwise marginal distribution distances.

This script compares the six window-specific marginal permeability distributions
window-by-window, component-by-component. It does NOT estimate cross-window
dependence. Instead, it quantifies how similar or different the one-dimensional
marginal distributions are across windows.

For each component c in {kxx, kyy, kzz}, it computes pairwise 6x6 matrices of:
    - Wasserstein-1 distance (primary metric)
    - Energy distance (secondary robustness check)
    - Two-sample KS statistic (supplementary diagnostic)

All metrics are computed directly from the pooled empirical reference samples in
log10(k) space, using the three reference ensembles R1/R2/R3 for each window.

Outputs:
    - pairwise distance table (long format)
    - 6x6 distance matrices per component and metric
    - per-component window summary table
    - optional combined normalized Wasserstein matrix across components
    - metadata

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Example:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level3/predict_reference_componentwise_window_distance_matrices.jl \\
        --windows famp1,famp2,famp3,famp4,famp5,famp6 \\
        --data-dir D:\\Github\\predict_shaowen\\examples\\gom_reference_floor_full\\data \\
        --output-dir D:\\codex_gom\\reference_level3_componentwise_window_distances_famp123456_v1
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
const EXAMPLES_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

function parse_args(args::Vector{String})
    options = Dict(
        "data-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "data")),
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_level3_componentwise_window_distances")),
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
    isempty(requested_windows) && error("Please provide --windows, e.g. --windows famp1,famp2,famp3,famp4,famp5,famp6")

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        requested_windows = requested_windows,
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level3/predict_reference_componentwise_window_distance_matrices.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>     Root folder with <window>/references/reference_R*.mat")
    println("  --output-dir <path>   Folder where the Level-3 distance outputs are saved")
    println("  --windows <names>     Comma-separated list like famp1,famp2,famp3,famp4,famp5,famp6")
    println("  -h, --help            Show this help")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    windows = collect_windows(opt.data_dir, opt.requested_windows)
    isempty(windows) && error("No requested windows found under $(opt.data_dir)")

    pooled_by_window = Dict{String, Matrix{Float64}}()

    println("Loading pooled references for $(length(windows)) window(s)...")
    for (window, reference_dir) in windows
        references = load_references(reference_dir)
        pooled_by_window[window] = reduce(vcat, [ref.y for ref in references])
    end

    pair_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    scale_rows = NamedTuple[]
    combined_wasserstein = zeros(length(opt.requested_windows), length(opt.requested_windows))
    wasserstein_matrices = Dict{String, Matrix{Float64}}()

    for (ic, component) in enumerate(COMPONENT_NAMES)
        values_by_window = Dict(window => vec(pooled_by_window[window][:, ic]) for window in opt.requested_windows)
        sorted_by_window = Dict(window => sort(values_by_window[window]) for window in opt.requested_windows)

        w1_matrix = compute_pairwise_matrix(opt.requested_windows,
                                            (wi, wj) -> wasserstein1_sorted(sorted_by_window[wi], sorted_by_window[wj]))
        energy_matrix = compute_pairwise_matrix(opt.requested_windows,
                                                (wi, wj) -> energy_distance_sorted(sorted_by_window[wi], sorted_by_window[wj]))
        ks_matrix = compute_pairwise_matrix(opt.requested_windows,
                                            (wi, wj) -> ks_statistic_sorted(sorted_by_window[wi], sorted_by_window[wj]))

        append!(pair_rows, build_pair_rows(component, opt.requested_windows, w1_matrix, energy_matrix, ks_matrix))
        append!(summary_rows, build_window_summary_rows(component, opt.requested_windows, w1_matrix, energy_matrix, ks_matrix))

        write_matrix_csv(joinpath(opt.output_dir, @sprintf("componentwise_window_wasserstein1_%s.csv", component)),
                         opt.requested_windows, w1_matrix)
        write_matrix_csv(joinpath(opt.output_dir, @sprintf("componentwise_window_energy_distance_%s.csv", component)),
                         opt.requested_windows, energy_matrix)
        write_matrix_csv(joinpath(opt.output_dir, @sprintf("componentwise_window_ks_statistic_%s.csv", component)),
                         opt.requested_windows, ks_matrix)

        wasserstein_matrices[component] = w1_matrix
        scale = median_offdiag(w1_matrix)
        scale = scale > 0 ? scale : 1.0
        push!(scale_rows, (
            component = component,
            wasserstein_scale = scale,
        ))
        combined_wasserstein .+= w1_matrix ./ scale
    end

    combined_wasserstein ./= length(COMPONENT_NAMES)

    write_pair_table_csv(joinpath(opt.output_dir, "componentwise_window_distance_pairs.csv"), pair_rows)
    write_summary_csv(joinpath(opt.output_dir, "componentwise_window_distance_summary.csv"), summary_rows)
    write_scale_csv(joinpath(opt.output_dir, "componentwise_window_wasserstein_scales.csv"), scale_rows)
    write_matrix_csv(joinpath(opt.output_dir, "componentwise_window_wasserstein1_combined_normalized.csv"),
                     opt.requested_windows, combined_wasserstein)
    write_metadata_csv(joinpath(opt.output_dir, "componentwise_window_distance_metadata.csv"), opt, pooled_by_window)
    save_wasserstein_heatmap_figure(joinpath(opt.output_dir, "componentwise_window_wasserstein1_heatmaps.png"),
                                    joinpath(opt.output_dir, "componentwise_window_wasserstein1_heatmaps.pdf"),
                                    opt.requested_windows, wasserstein_matrices)
    save_combined_wasserstein_heatmap_figure(joinpath(opt.output_dir, "componentwise_window_wasserstein1_combined_normalized_heatmap.png"),
                                             joinpath(opt.output_dir, "componentwise_window_wasserstein1_combined_normalized_heatmap.pdf"),
                                             opt.requested_windows, combined_wasserstein)

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
        push!(references, (
            name = replace(replace(file, ".mat" => ""), "reference_" => ""),
            y = log10.(perms),
        ))
    end
    return references
end

function compute_pairwise_matrix(windows::Vector{String}, metric_fn)
    n = length(windows)
    mat = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            d = metric_fn(windows[i], windows[j])
            mat[i, j] = d
            mat[j, i] = d
        end
    end
    return mat
end

function wasserstein1_sorted(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    i = 1
    j = 1
    n = length(x)
    m = length(y)
    cdf_x = 0.0
    cdf_y = 0.0
    prev = min(x[1], y[1])
    dist = 0.0

    while i <= n || j <= m
        next_x = i <= n ? x[i] : Inf
        next_y = j <= m ? y[j] : Inf
        z = min(next_x, next_y)
        if isfinite(z)
            dist += abs(cdf_x - cdf_y) * (z - prev)
            while i <= n && x[i] == z
                cdf_x += 1.0 / n
                i += 1
            end
            while j <= m && y[j] == z
                cdf_y += 1.0 / m
                j += 1
            end
            prev = z
        else
            break
        end
    end

    return dist
end

function energy_distance_sorted(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    integral = cdf_l2_integral_sorted(x, y)
    return sqrt(max(0.0, 2.0 * integral))
end

function ks_statistic_sorted(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    i = 1
    j = 1
    n = length(x)
    m = length(y)
    cdf_x = 0.0
    cdf_y = 0.0
    max_diff = 0.0

    while i <= n || j <= m
        next_x = i <= n ? x[i] : Inf
        next_y = j <= m ? y[j] : Inf
        z = min(next_x, next_y)
        if !isfinite(z)
            break
        end
        while i <= n && x[i] == z
            cdf_x += 1.0 / n
            i += 1
        end
        while j <= m && y[j] == z
            cdf_y += 1.0 / m
            j += 1
        end
        max_diff = max(max_diff, abs(cdf_x - cdf_y))
    end

    return max_diff
end

function cdf_l2_integral_sorted(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    i = 1
    j = 1
    n = length(x)
    m = length(y)
    cdf_x = 0.0
    cdf_y = 0.0
    prev = min(x[1], y[1])
    integral = 0.0

    while i <= n || j <= m
        next_x = i <= n ? x[i] : Inf
        next_y = j <= m ? y[j] : Inf
        z = min(next_x, next_y)
        if isfinite(z)
            integral += (cdf_x - cdf_y)^2 * (z - prev)
            while i <= n && x[i] == z
                cdf_x += 1.0 / n
                i += 1
            end
            while j <= m && y[j] == z
                cdf_y += 1.0 / m
                j += 1
            end
            prev = z
        else
            break
        end
    end

    return integral
end

function build_pair_rows(component, windows, w1_matrix, energy_matrix, ks_matrix)
    rows = NamedTuple[]
    n = length(windows)
    for i in 1:n
        for j in i+1:n
            push!(rows, (
                component = component,
                window_i = windows[i],
                window_j = windows[j],
                wasserstein1 = w1_matrix[i, j],
                energy_distance = energy_matrix[i, j],
                ks_statistic = ks_matrix[i, j],
            ))
        end
    end
    return rows
end

function build_window_summary_rows(component, windows, w1_matrix, energy_matrix, ks_matrix)
    rows = NamedTuple[]
    n = length(windows)
    for i in 1:n
        others = [j for j in 1:n if j != i]
        w1_vals = [w1_matrix[i, j] for j in others]
        energy_vals = [energy_matrix[i, j] for j in others]
        ks_vals = [ks_matrix[i, j] for j in others]

        w1_min_idx = others[argmin(w1_vals)]
        w1_max_idx = others[argmax(w1_vals)]
        energy_min_idx = others[argmin(energy_vals)]
        energy_max_idx = others[argmax(energy_vals)]
        ks_min_idx = others[argmin(ks_vals)]
        ks_max_idx = others[argmax(ks_vals)]

        push!(rows, (
            component = component,
            window = windows[i],
            mean_wasserstein1_to_others = mean(w1_vals),
            nearest_window_wasserstein1 = windows[w1_min_idx],
            nearest_wasserstein1 = w1_matrix[i, w1_min_idx],
            farthest_window_wasserstein1 = windows[w1_max_idx],
            farthest_wasserstein1 = w1_matrix[i, w1_max_idx],
            mean_energy_distance_to_others = mean(energy_vals),
            nearest_window_energy_distance = windows[energy_min_idx],
            nearest_energy_distance = energy_matrix[i, energy_min_idx],
            farthest_window_energy_distance = windows[energy_max_idx],
            farthest_energy_distance = energy_matrix[i, energy_max_idx],
            mean_ks_to_others = mean(ks_vals),
            nearest_window_ks = windows[ks_min_idx],
            nearest_ks = ks_matrix[i, ks_min_idx],
            farthest_window_ks = windows[ks_max_idx],
            farthest_ks = ks_matrix[i, ks_max_idx],
        ))
    end
    return rows
end

function median_offdiag(mat::AbstractMatrix{<:Real})
    vals = Float64[]
    n = size(mat, 1)
    for i in 1:n
        for j in i+1:n
            push!(vals, mat[i, j])
        end
    end
    isempty(vals) && return 0.0
    return median(vals)
end

function write_matrix_csv(filepath, windows, mat)
    open(filepath, "w") do io
        println(io, join(vcat(["window"], windows), ","))
        for (i, window) in enumerate(windows)
            row = [window]
            append!(row, [fmt(mat[i, j]) for j in 1:length(windows)])
            println(io, join(row, ","))
        end
    end
end

function write_pair_table_csv(filepath, rows)
    header = ["component", "window_i", "window_j", "wasserstein1", "energy_distance", "ks_statistic"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.component,
                row.window_i,
                row.window_j,
                fmt(row.wasserstein1),
                fmt(row.energy_distance),
                fmt(row.ks_statistic),
            ], ","))
        end
    end
end

function write_summary_csv(filepath, rows)
    header = ["component", "window",
              "mean_wasserstein1_to_others", "nearest_window_wasserstein1", "nearest_wasserstein1",
              "farthest_window_wasserstein1", "farthest_wasserstein1",
              "mean_energy_distance_to_others", "nearest_window_energy_distance", "nearest_energy_distance",
              "farthest_window_energy_distance", "farthest_energy_distance",
              "mean_ks_to_others", "nearest_window_ks", "nearest_ks", "farthest_window_ks", "farthest_ks"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.component,
                row.window,
                fmt(row.mean_wasserstein1_to_others),
                row.nearest_window_wasserstein1,
                fmt(row.nearest_wasserstein1),
                row.farthest_window_wasserstein1,
                fmt(row.farthest_wasserstein1),
                fmt(row.mean_energy_distance_to_others),
                row.nearest_window_energy_distance,
                fmt(row.nearest_energy_distance),
                row.farthest_window_energy_distance,
                fmt(row.farthest_energy_distance),
                fmt(row.mean_ks_to_others),
                row.nearest_window_ks,
                fmt(row.nearest_ks),
                row.farthest_window_ks,
                fmt(row.farthest_ks),
            ], ","))
        end
    end
end

function write_scale_csv(filepath, rows)
    header = ["component", "wasserstein_scale"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([row.component, fmt(row.wasserstein_scale)], ","))
        end
    end
end

function write_metadata_csv(filepath, opt, pooled_by_window)
    header = ["windows", "data_dir", "num_windows", "num_samples_per_window", "components", "primary_metric", "secondary_metric", "supplementary_metric"]
    sample_counts = [size(pooled_by_window[w], 1) for w in opt.requested_windows]
    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            join(opt.requested_windows, ";"),
            opt.data_dir,
            string(length(opt.requested_windows)),
            join(string.(sample_counts), ";"),
            join(COMPONENT_NAMES, ";"),
            "wasserstein1",
            "energy_distance",
            "ks_statistic",
        ], ","))
    end
end

function save_wasserstein_heatmap_figure(png_path, pdf_path, windows, wasserstein_matrices)
    labels = display_window_labels(windows)
    fig = Figure(size = (1900, 950), fontsize = 28, backgroundcolor = :white,
                 figure_padding = (40, 40, 50, 40))
    colgap!(fig.layout, 24)
    rowgap!(fig.layout, 0)
    Label(fig[1, 1:3], "Level 3 Componentwise Cross-Window\nWasserstein Distance",
          fontsize = 28,
          font = :bold,
          halign = :center,
          justification = :center,
          valign = :bottom)

    global_max = maximum(maximum(mat) for mat in values(wasserstein_matrices))
    global_max = global_max > 0 ? global_max : 1.0
    common_ticks = collect(range(0.0, global_max, length = 4))
    shared_hm = nothing

    for (idx, component) in enumerate(COMPONENT_NAMES)
        panel_label = idx == 1 ? "(a)" : idx == 2 ? "(b)" : "(c)"
        ax = Axis(fig[2, idx],
                  title = panel_label * " " * COMPONENT_LABELS[idx],
                  titlesize = 28,
                  xticks = (1:length(windows), labels),
                  yticks = (1:length(windows), labels),
                  xticklabelsize = 28,
                  yticklabelsize = 28,
                  xticklabelrotation = 0,
                  xgridvisible = false,
                  ygridvisible = false,
                  xminorgridvisible = false,
                  yminorgridvisible = false,
                  leftspinevisible = true,
                  rightspinevisible = true,
                  topspinevisible = true,
                  bottomspinevisible = true,
                  aspect = DataAspect(),
                  yreversed = true)
        mat = wasserstein_matrices[component]
        masked = lower_triangle_mask(mat)
        colorrange = (0.0, global_max)
        hm = heatmap!(ax, 1:length(windows), 1:length(windows), masked';
                      colormap = :viridis, colorrange = colorrange, nan_color = :white)
        if idx == 1
            shared_hm = hm
        end

        for i in 1:length(windows), j in 1:i
            val = mat[i, j]
            normv = colorrange[2] > 0 ? val / colorrange[2] : 0.0
            txtcolor = normv > 0.52 ? :white : :black
            text!(ax, j, i,
                  text = @sprintf("%.2f", val),
                  align = (:center, :center),
                  fontsize = 28,
                  color = txtcolor)
        end
    end

    Colorbar(fig[3, 1:3], shared_hm,
             vertical = false,
             label = "Wasserstein distance in log10(k)",
             ticks = common_ticks,
             tickformat = values -> [@sprintf("%.2f", v) for v in values],
             ticklabelsize = 28,
             labelsize = 28,
             height = 24,
             width = 1120,
             halign = :center)

    rowsize!(fig.layout, 1, Fixed(130))
    rowsize!(fig.layout, 2, Fixed(600))
    rowsize!(fig.layout, 3, Fixed(55))
    for idx in 1:3
        colsize!(fig.layout, idx, Fixed(560))
    end

    save(png_path, fig)
    save(pdf_path, fig)
end

function save_combined_wasserstein_heatmap_figure(png_path, pdf_path, windows, mat)
    labels = display_window_labels(windows)
    fig = Figure(size = (1130, 1145), fontsize = 28, backgroundcolor = :white,
                 figure_padding = (24, 24, 80, 55))
    rowgap!(fig.layout, 0)
    Label(fig[1, 1], "Level 3 Combined Cross-Window\nNormalized Wasserstein Distance",
          fontsize = 28,
          font = :bold,
          halign = :center,
          justification = :center,
          valign = :bottom)

    ax = Axis(fig[2, 1],
              title = "",
              xticks = (1:length(windows), labels),
              yticks = (1:length(windows), labels),
              xticklabelsize = 28,
              yticklabelsize = 28,
              xticklabelrotation = 0,
              xgridvisible = false,
              ygridvisible = false,
              xminorgridvisible = false,
              yminorgridvisible = false,
              leftspinevisible = true,
              rightspinevisible = true,
              topspinevisible = true,
              bottomspinevisible = true,
              aspect = DataAspect(),
              yreversed = true)
    masked = lower_triangle_mask(mat)
    colorrange = (0.0, maximum(mat))
    tick_max = colorrange[2] > 0 ? colorrange[2] : 1.0
    combined_tick_values = [0.0, tick_max / 3, 2 * tick_max / 3, tick_max]
    combined_tick_positions = tick_max > 0 ?
        [0.015 * tick_max, tick_max / 3, 2 * tick_max / 3, 0.985 * tick_max] :
        combined_tick_values
    combined_tick_labels = [@sprintf("%.2f", v) for v in combined_tick_values]
    hm = heatmap!(ax, 1:length(windows), 1:length(windows), masked';
                  colormap = :viridis, colorrange = colorrange, nan_color = :white)

    for i in 1:length(windows), j in 1:i
        val = mat[i, j]
        normv = colorrange[2] > 0 ? val / colorrange[2] : 0.0
        txtcolor = normv > 0.52 ? :white : :black
        text!(ax, j, i,
              text = @sprintf("%.2f", val),
              align = (:center, :center),
              fontsize = 28,
              color = txtcolor)
    end

    Colorbar(fig[3, 1], hm,
             vertical = false,
             label = "Normalized mean Wasserstein distance",
             ticks = (combined_tick_positions, combined_tick_labels),
             ticklabelsize = 28,
             labelsize = 28,
             height = 24,
             width = 810,
             halign = :center)

    rowsize!(fig.layout, 1, Fixed(90))
    rowsize!(fig.layout, 2, Fixed(885))
    rowsize!(fig.layout, 3, Fixed(50))

    save(png_path, fig)
    save(pdf_path, fig)
end

function display_window_labels(windows::Vector{String})
    return [window_display_label(w) for w in windows]
end

function window_display_label(window::AbstractString)
    m = match(r"famp(\d+)", lowercase(window))
    return m === nothing ? window : "W" * m.captures[1]
end

function add_centered_title!(slot, text; fontsize = 24)
    ax = Axis(slot,
              xticksvisible = false, yticksvisible = false,
              xticklabelsvisible = false, yticklabelsvisible = false,
              leftspinevisible = false, rightspinevisible = false,
              topspinevisible = false, bottomspinevisible = false,
              backgroundcolor = :white)
    hidedecorations!(ax)
    hidespines!(ax)
    limits!(ax, 0, 1, 0, 1)
    text!(ax, [0.5], [0.5],
          text = [text],
          space = :data,
          align = (:center, :center),
          fontsize = fontsize,
          font = :bold)
    return ax
end

function lower_triangle_mask(mat::AbstractMatrix{<:Real})
    masked = Matrix{Float64}(mat)
    n = size(masked, 1)
    for i in 1:n
        for j in i+1:n
            masked[i, j] = NaN
        end
    end
    return masked
end

fmt(x) = @sprintf("%.10g", x)

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
