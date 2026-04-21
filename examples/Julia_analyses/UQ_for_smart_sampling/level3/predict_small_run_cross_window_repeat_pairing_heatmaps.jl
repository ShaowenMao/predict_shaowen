#!/usr/bin/env julia

"""
Level-3 diagnostic: cross-window repeat-pair correlation heatmaps.

This script visualizes how sensitive the inferred cross-window correlation is
to the arbitrary pairing of independently generated small-run ensembles.

For each component c in {kxx, kyy, kzz}, it computes the repeat-pair matrix

    R_c(i, j) = cor(z_A[i][:, c], z_B[j][:, c])

where:
    - z_A[i][:, c] is the component-c sample vector from repeat i of window A
    - z_B[j][:, c] is the component-c sample vector from repeat j of window B

When the windows were simulated independently, the diagonal pairings
(repeat01 with repeat01, repeat02 with repeat02, ...) are not privileged by the
data-generating process. The resulting heatmaps are therefore a direct visual
check of how pairing-dependent these cross-window correlations are.

Outputs:
    - one CSV correlation matrix per component
    - one long-format CSV with every repeat-pair correlation
    - one diagonal-vs-offdiagonal summary CSV
    - one CSV of the diagonal entries for common repeat IDs
    - one three-panel heatmap figure (PNG and PDF)
    - one metadata CSV

Required Julia packages:
    using Pkg
    Pkg.add(["MAT", "CairoMakie"])

Example:
    julia examples/Julia_analyses/UQ_for_smart_sampling/level3/predict_small_run_cross_window_repeat_pairing_heatmaps.jl \
        --window-a famp1 \
        --window-b famp2 \
        --nsim 2000 \
        --data-dir D:/Github/predict_shaowen/examples/gom_reference_floor_full/data \
        --output-dir D:/Github/predict_shaowen/examples/gom_reference_floor_full/julia_level3_repeat_pair_heatmaps_famp1_vs_famp2
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
        "output-dir" => normpath(joinpath(EXAMPLES_ROOT, "gom_reference_floor_full", "julia_level3_repeat_pair_heatmaps_famp1_vs_famp2")),
        "window-a" => "famp1",
        "window-b" => "famp2",
        "nsim" => "2000",
        "use-log10" => "true",
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

    nsim = tryparse(Int, options["nsim"])
    nsim === nothing && error("--nsim must be an integer")
    nsim > 0 || error("--nsim must be positive")

    return (
        data_dir = options["data-dir"],
        output_dir = options["output-dir"],
        window_a = options["window-a"],
        window_b = options["window-b"],
        nsim = nsim,
        use_log10 = parse_bool_option(options["use-log10"], "--use-log10"),
    )
end

function print_help()
    println("Usage:")
    println("  julia examples/Julia_analyses/UQ_for_smart_sampling/level3/predict_small_run_cross_window_repeat_pairing_heatmaps.jl [options]")
    println()
    println("Options:")
    println("  --data-dir <path>     Root folder with <window>/small_runs/N<nsim>_repeatXX.mat")
    println("  --output-dir <path>   Folder where the heatmaps and CSV outputs are saved")
    println("  --window-a <name>     First window, default: famp1")
    println("  --window-b <name>     Second window, default: famp2")
    println("  --nsim <integer>      Small-run size prefix, default: 2000")
    println("  --use-log10 <bool>    Use log10(perms) before correlation, default: true")
    println("  -h, --help            Show this help")
end

function parse_bool_option(value::AbstractString, flag_name::AbstractString)
    value_lower = lowercase(strip(value))
    if value_lower in ("true", "1", "yes", "y")
        return true
    elseif value_lower in ("false", "0", "no", "n")
        return false
    end
    error("$flag_name must be one of true/false, yes/no, or 1/0")
end

function main(args)
    opt = parse_args(args)
    mkpath(opt.output_dir)

    repeats_a = load_small_runs(opt.data_dir, opt.window_a, opt.nsim, opt.use_log10)
    repeats_b = load_small_runs(opt.data_dir, opt.window_b, opt.nsim, opt.use_log10)
    row_repeats = sort(collect(keys(repeats_a)))
    col_repeats = sort(collect(keys(repeats_b)))

    println("Loaded $(length(row_repeats)) repeat(s) for $(opt.window_a) and $(length(col_repeats)) repeat(s) for $(opt.window_b).")

    corr_matrices = Dict{String, Matrix{Float64}}()
    pair_rows = NamedTuple[]

    for (ic, component) in enumerate(COMPONENT_NAMES)
        mat = compute_component_correlation_matrix(repeats_a, repeats_b, row_repeats, col_repeats, ic)
        corr_matrices[component] = mat
        write_matrix_csv(joinpath(opt.output_dir, @sprintf("%s_vs_%s_repeat_pair_correlation_%s.csv",
                                                           opt.window_a, opt.window_b, component)),
                         row_repeats, col_repeats, mat)
        append!(pair_rows, build_pair_rows(opt.window_a, opt.window_b, component, row_repeats, col_repeats, mat))
    end

    summary_rows, diagonal_rows = build_diagonal_summary_rows(corr_matrices, row_repeats, col_repeats)
    write_pair_table_csv(joinpath(opt.output_dir,
                                  @sprintf("%s_vs_%s_repeat_pair_correlations_long.csv", opt.window_a, opt.window_b)),
                         pair_rows)
    write_summary_csv(joinpath(opt.output_dir,
                               @sprintf("%s_vs_%s_repeat_pair_diagonal_summary.csv", opt.window_a, opt.window_b)),
                      summary_rows)
    write_diagonal_csv(joinpath(opt.output_dir,
                                @sprintf("%s_vs_%s_repeat_pair_diagonal_values.csv", opt.window_a, opt.window_b)),
                       diagonal_rows)
    write_metadata_csv(joinpath(opt.output_dir,
                                @sprintf("%s_vs_%s_repeat_pair_metadata.csv", opt.window_a, opt.window_b)),
                       opt, repeats_a, repeats_b, row_repeats, col_repeats)

    png_path = joinpath(opt.output_dir,
                        @sprintf("%s_vs_%s_repeat_pair_correlation_heatmaps.png", opt.window_a, opt.window_b))
    pdf_path = joinpath(opt.output_dir,
                        @sprintf("%s_vs_%s_repeat_pair_correlation_heatmaps.pdf", opt.window_a, opt.window_b))
    save_heatmap_figure(png_path, pdf_path, opt, row_repeats, col_repeats, corr_matrices)

    println("Saved outputs to $(opt.output_dir)")
end

function load_small_runs(data_dir::AbstractString, window::AbstractString, nsim::Int, use_log10::Bool)
    small_run_dir = joinpath(data_dir, window, "small_runs")
    isdir(small_run_dir) || error("Small-run directory does not exist: $small_run_dir")

    pattern = Regex("^N$(nsim)_repeat(\\d+)\\.mat\$")
    entries = Tuple{Int, String}[]
    for file in readdir(small_run_dir)
        match_obj = match(pattern, file)
        match_obj === nothing && continue
        repeat_id = parse(Int, match_obj.captures[1])
        push!(entries, (repeat_id, joinpath(small_run_dir, file)))
    end
    isempty(entries) && error("No files matching N$(nsim)_repeatXX.mat found in $small_run_dir")
    sort!(entries, by = first)

    repeats = Dict{Int, Matrix{Float64}}()
    for (repeat_id, filepath) in entries
        data = matread(filepath)
        haskey(data, "perms") || error("File does not contain perms: $filepath")
        perms = Matrix{Float64}(data["perms"])
        size(perms, 2) == 3 || error("Expected perms to have 3 columns in $filepath")
        if use_log10
            all(perms .> 0) || error("perms contains non-positive values in $filepath; cannot apply log10")
            repeats[repeat_id] = log10.(perms)
        else
            repeats[repeat_id] = perms
        end
    end

    return repeats
end

function compute_component_correlation_matrix(repeats_a, repeats_b, row_repeats, col_repeats, component_index::Int)
    mat = Matrix{Float64}(undef, length(row_repeats), length(col_repeats))
    for (i, repeat_a) in enumerate(row_repeats)
        xa = vec(repeats_a[repeat_a][:, component_index])
        for (j, repeat_b) in enumerate(col_repeats)
            yb = vec(repeats_b[repeat_b][:, component_index])
            length(xa) == length(yb) || error("Repeat $(repeat_a) and repeat $(repeat_b) have different sample counts")
            mat[i, j] = safe_cor(xa, yb)
        end
    end
    return mat
end

function safe_cor(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    std(x) > 0 || return NaN
    std(y) > 0 || return NaN
    return cor(x, y)
end

function build_pair_rows(window_a, window_b, component, row_repeats, col_repeats, mat)
    rows = NamedTuple[]
    for (i, repeat_a) in enumerate(row_repeats)
        for (j, repeat_b) in enumerate(col_repeats)
            push!(rows, (
                window_a = window_a,
                repeat_a = repeat_a,
                window_b = window_b,
                repeat_b = repeat_b,
                component = component,
                correlation = mat[i, j],
            ))
        end
    end
    return rows
end

function build_diagonal_summary_rows(corr_matrices, row_repeats, col_repeats)
    common_repeats = sort(intersect(row_repeats, col_repeats))
    row_index = Dict(repeat_id => idx for (idx, repeat_id) in enumerate(row_repeats))
    col_index = Dict(repeat_id => idx for (idx, repeat_id) in enumerate(col_repeats))

    summary_rows = NamedTuple[]
    diagonal_rows = NamedTuple[]

    for component in COMPONENT_NAMES
        mat = corr_matrices[component]
        full_vals = vec(mat)

        diagonal_vals = Float64[]
        offdiagonal_vals = Float64[]
        for repeat_i in common_repeats
            i = row_index[repeat_i]
            j = col_index[repeat_i]
            diag_val = mat[i, j]
            push!(diagonal_vals, diag_val)
            push!(diagonal_rows, (
                component = component,
                repeat_id = repeat_i,
                correlation = diag_val,
            ))
        end

        for repeat_i in common_repeats
            for repeat_j in common_repeats
                repeat_i == repeat_j && continue
                i = row_index[repeat_i]
                j = col_index[repeat_j]
                push!(offdiagonal_vals, mat[i, j])
            end
        end

        push!(summary_rows, (
            component = component,
            num_row_repeats = length(row_repeats),
            num_col_repeats = length(col_repeats),
            num_common_repeats = length(common_repeats),
            full_mean = mean(skipmissing(full_vals)),
            full_std = std(skipmissing(full_vals)),
            full_min = minimum(full_vals),
            full_max = maximum(full_vals),
            diagonal_mean = isempty(diagonal_vals) ? NaN : mean(diagonal_vals),
            diagonal_std = length(diagonal_vals) <= 1 ? NaN : std(diagonal_vals),
            diagonal_min = isempty(diagonal_vals) ? NaN : minimum(diagonal_vals),
            diagonal_max = isempty(diagonal_vals) ? NaN : maximum(diagonal_vals),
            offdiagonal_mean = isempty(offdiagonal_vals) ? NaN : mean(offdiagonal_vals),
            offdiagonal_std = length(offdiagonal_vals) <= 1 ? NaN : std(offdiagonal_vals),
            offdiagonal_min = isempty(offdiagonal_vals) ? NaN : minimum(offdiagonal_vals),
            offdiagonal_max = isempty(offdiagonal_vals) ? NaN : maximum(offdiagonal_vals),
            median_diagonal_percentile_vs_offdiagonal = isempty(diagonal_vals) || isempty(offdiagonal_vals) ? NaN :
                                                        mean(offdiagonal_vals .<= median(diagonal_vals)),
        ))
    end

    return summary_rows, diagonal_rows
end

function write_matrix_csv(filepath, row_repeats, col_repeats, mat)
    open(filepath, "w") do io
        header = ["repeat_a"]
        append!(header, [@sprintf("repeat_%02d", repeat_id) for repeat_id in col_repeats])
        println(io, join(header, ","))
        for (i, repeat_id) in enumerate(row_repeats)
            row = [@sprintf("repeat_%02d", repeat_id)]
            append!(row, [fmt(mat[i, j]) for j in 1:length(col_repeats)])
            println(io, join(row, ","))
        end
    end
end

function write_pair_table_csv(filepath, rows)
    header = ["window_a", "repeat_a", "window_b", "repeat_b", "component", "correlation"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.window_a,
                string(row.repeat_a),
                row.window_b,
                string(row.repeat_b),
                row.component,
                fmt(row.correlation),
            ], ","))
        end
    end
end

function write_summary_csv(filepath, rows)
    header = ["component", "num_row_repeats", "num_col_repeats", "num_common_repeats",
              "full_mean", "full_std", "full_min", "full_max",
              "diagonal_mean", "diagonal_std", "diagonal_min", "diagonal_max",
              "offdiagonal_mean", "offdiagonal_std", "offdiagonal_min", "offdiagonal_max",
              "median_diagonal_percentile_vs_offdiagonal"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([
                row.component,
                string(row.num_row_repeats),
                string(row.num_col_repeats),
                string(row.num_common_repeats),
                fmt(row.full_mean),
                fmt(row.full_std),
                fmt(row.full_min),
                fmt(row.full_max),
                fmt(row.diagonal_mean),
                fmt(row.diagonal_std),
                fmt(row.diagonal_min),
                fmt(row.diagonal_max),
                fmt(row.offdiagonal_mean),
                fmt(row.offdiagonal_std),
                fmt(row.offdiagonal_min),
                fmt(row.offdiagonal_max),
                fmt(row.median_diagonal_percentile_vs_offdiagonal),
            ], ","))
        end
    end
end

function write_diagonal_csv(filepath, rows)
    header = ["component", "repeat_id", "correlation"]
    open(filepath, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join([row.component, string(row.repeat_id), fmt(row.correlation)], ","))
        end
    end
end

function write_metadata_csv(filepath, opt, repeats_a, repeats_b, row_repeats, col_repeats)
    header = ["window_a", "window_b", "data_dir", "nsim", "use_log10",
              "num_repeats_a", "num_repeats_b", "common_repeat_ids", "sample_sizes_a", "sample_sizes_b"]
    sample_sizes_a = [size(repeats_a[r], 1) for r in row_repeats]
    sample_sizes_b = [size(repeats_b[r], 1) for r in col_repeats]
    common_repeats = sort(intersect(row_repeats, col_repeats))

    open(filepath, "w") do io
        println(io, join(header, ","))
        println(io, join([
            opt.window_a,
            opt.window_b,
            opt.data_dir,
            string(opt.nsim),
            string(opt.use_log10),
            string(length(row_repeats)),
            string(length(col_repeats)),
            join(string.(common_repeats), ";"),
            join(string.(sample_sizes_a), ";"),
            join(string.(sample_sizes_b), ";"),
        ], ","))
    end
end

function save_heatmap_figure(png_path, pdf_path, opt, row_repeats, col_repeats, corr_matrices)
    row_labels = display_repeat_labels(row_repeats)
    col_labels = display_repeat_labels(col_repeats)
    xticks = sparse_ticks(col_repeats, col_labels)
    yticks = sparse_ticks(row_repeats, row_labels)

    fig = Figure(size = (2200, 980), fontsize = 28, backgroundcolor = :white,
                 figure_padding = (40, 40, 50, 40))
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 8)
    Label(fig[1, 1:3],
          "Cross-Window Repeat-Pair Correlations\n$(opt.window_a) vs $(opt.window_b), N=$(opt.nsim), " *
          "$(opt.use_log10 ? "log10(k)" : "k")",
          fontsize = 30,
          font = :bold,
          halign = :center,
          justification = :center,
          valign = :bottom)

    max_abs_corr = maximum(maximum(abs.(mat)) for mat in values(corr_matrices))
    max_abs_corr = max(max_abs_corr, 1.0e-6)
    common_ticks = collect(range(-max_abs_corr, max_abs_corr, length = 5))
    shared_hm = nothing

    for (idx, component) in enumerate(COMPONENT_NAMES)
        panel_label = idx == 1 ? "(a)" : idx == 2 ? "(b)" : "(c)"
        ax = Axis(fig[2, idx],
                  title = panel_label * " " * COMPONENT_LABELS[idx],
                  titlesize = 28,
                  xticks = xticks,
                  yticks = yticks,
                  xticklabelsize = 22,
                  yticklabelsize = 22,
                  xticklabelrotation = pi / 4,
                  xlabel = "$(opt.window_b) repeats",
                  ylabel = idx == 1 ? "$(opt.window_a) repeats" : "",
                  xlabelsize = 24,
                  ylabelsize = 24,
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

        mat = corr_matrices[component]
        hm = heatmap!(ax, 1:length(col_repeats), 1:length(row_repeats), mat';
                      colormap = :balance,
                      colorrange = (-max_abs_corr, max_abs_corr))
        if idx == 1
            shared_hm = hm
        end

        if row_repeats == col_repeats
            n = length(row_repeats)
            lines!(ax, [0.5, n + 0.5], [0.5, n + 0.5], color = (:black, 0.9), linewidth = 2)
        end
    end

    cbar = Colorbar(fig[2, 4], shared_hm,
                    label = "Correlation coefficient",
                    ticks = common_ticks,
                    ticklabelsize = 22,
                    labelsize = 24,
                    height = Relative(0.85))
    cbar.width = 28

    Label(fig[3, 1:3],
          "Entry (i, j) = corr(component samples from $(opt.window_a) repeat i, $(opt.window_b) repeat j). " *
          "The matrix is generally not symmetric because rows and columns come from different windows.",
          fontsize = 20,
          halign = :center,
          justification = :center,
          valign = :top)

    save(png_path, fig)
    save(pdf_path, fig)
    return nothing
end

function display_repeat_labels(repeats)
    return [@sprintf("R%02d", repeat_id) for repeat_id in repeats]
end

function sparse_ticks(repeats, labels)
    n = length(repeats)
    desired_count = min(7, n)
    step = max(1, ceil(Int, n / desired_count))
    tick_positions = collect(1:step:n)
    if tick_positions[end] != n
        push!(tick_positions, n)
    end
    tick_positions = unique(sort(tick_positions))
    tick_labels = labels[tick_positions]
    return (tick_positions, tick_labels)
end

function fmt(x)
    if x isa Missing
        return ""
    elseif x isa AbstractFloat && isnan(x)
        return "NaN"
    end
    return @sprintf("%.6f", x)
end

main(ARGS)
