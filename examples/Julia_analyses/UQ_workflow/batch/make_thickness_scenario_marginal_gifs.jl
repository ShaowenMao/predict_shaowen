#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Printf

include(normpath(joinpath(@__DIR__, "..", "level2", "lib",
                          "level2_output_marginal_histograms.jl")))

const EXPECTED_WINDOWS = ["famp$(index)" for index in 1:6]
const EXPECTED_CASE_COUNT = 27
const EXPECTED_LIBRARY_COUNT = EXPECTED_CASE_COUNT * length(EXPECTED_WINDOWS)

"""
    parse_args(args)

Parse command-line options for the thickness-scenario GIF generator.
"""
function parse_args(args::Vector{String})
    options = Dict(
        "data-root" => normpath(joinpath(@__DIR__, "..", "..", "..",
                                         "thickness_scenario_data_collapsed_cell_union")),
        "output-root" => normpath(joinpath(@__DIR__, "..", "outputs",
                                            "thickness_scenario_marginal_gifs")),
        "scenario" => "all",
        "bins" => "36",
        "duration-ms" => "900",
        "overwrite" => "true",
    )

    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("-h", "--help")
            print_help()
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(options, key) || error("Unknown option: $arg")
            index < length(args) || error("Missing value for $arg")
            options[key] = args[index + 1]
            index += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return options
end

"""Print command-line usage."""
function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow \\")
    println("    examples/Julia_analyses/UQ_workflow/batch/make_thickness_scenario_marginal_gifs.jl [options]")
    println()
    println("Options:")
    println("  --data-root <path>       PREDICT thickness-scenario dataset")
    println("  --output-root <path>     Frame and GIF output root")
    println("  --scenario <name|all>    Generate one scenario or all scenarios")
    println("  --bins <n>               Scenario-wide bins per component (default: 36)")
    println("  --duration-ms <n>        Display duration per frame (default: 900)")
    println("  --overwrite <bool>       Replace existing frames and GIFs")
end

"""Parse a user-facing boolean option."""
function parse_bool(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("true", "1", "yes", "y") && return true
    normalized in ("false", "0", "no", "n") && return false
    error("Cannot parse boolean value: $value")
end

"""Return sorted scenario folders selected by the command-line option."""
function selected_scenarios(data_root::AbstractString, requested::AbstractString)
    scenario_root = joinpath(data_root, "data")
    isdir(scenario_root) || error("Dataset has no data folder: $scenario_root")
    scenarios = sort(filter(name -> startswith(name, "scenario_"), readdir(scenario_root)))
    if lowercase(requested) != "all"
        requested in scenarios || error("Scenario not found: $requested")
        scenarios = [String(requested)]
    end
    isempty(scenarios) && error("No thickness scenarios found in $scenario_root")
    return scenarios
end

"""Convert a scenario folder name to the title used in each GIF frame."""
function scenario_display_name(scenario::AbstractString)
    matched = match(r"^scenario_\d+_(low|medium|high)_sand_(uniform|nonuniform)$",
                    scenario)
    isnothing(matched) && error("Unexpected scenario name: $scenario")
    return "$(matched.captures[1]) sand, $(matched.captures[2])"
end

"""Parse case index and geologic parameters from a case folder name."""
function parse_case_name(case_name::AbstractString)
    matched = match(r"^case_(\d+)_zf(\d+)_svcl(\d+)_cvcl(\d+)$", case_name)
    isnothing(matched) && error("Unexpected case name: $case_name")
    return (
        index = parse(Int, matched.captures[1]),
        fault_depth = parse(Int, matched.captures[2]),
        sand_vcl = parse(Int, matched.captures[3]) / 100,
        clay_vcl = parse(Int, matched.captures[4]) / 100,
    )
end

"""Load one PREDICT `predict_runs.mat` library as a Level 2 proxy."""
function load_predict_proxy(path::AbstractString, geology_id::AbstractString)
    window = basename(dirname(dirname(path)))
    row = Dict{String, String}(
        "resolved_mat_path" => String(path),
        "window" => window,
        "geology_id" => String(geology_id),
        "sample_kind" => "predict_runs",
    )
    return Level2IO.load_proxy_library(row)
end

"""
    load_scenario_libraries(data_root, scenario)

Load and validate all 27-by-6 libraries for one thickness scenario. The
returned map is indexed first by case folder and then by window.
"""
function load_scenario_libraries(data_root::AbstractString,
                                 scenario::AbstractString)
    scenario_root = joinpath(data_root, "data", scenario)
    paths = String[]
    for (root, _, files) in walkdir(scenario_root)
        "predict_runs.mat" in files && push!(paths, joinpath(root, "predict_runs.mat"))
    end
    sort!(paths)
    length(paths) == EXPECTED_LIBRARY_COUNT ||
        error("Expected $EXPECTED_LIBRARY_COUNT libraries for $scenario, found $(length(paths))")

    case_map = Dict{String, Dict{String, Dict{String, Any}}}()
    all_proxies = Dict{String, Any}[]
    for path in paths
        case_name = basename(dirname(path))
        window = basename(dirname(dirname(path)))
        window in EXPECTED_WINDOWS || error("Unexpected window in path: $path")
        proxy = load_predict_proxy(path, scenario)
        get!(case_map, case_name, Dict{String, Dict{String, Any}}())[window] = proxy
        push!(all_proxies, proxy)
    end

    length(case_map) == EXPECTED_CASE_COUNT ||
        error("Expected $EXPECTED_CASE_COUNT cases for $scenario, found $(length(case_map))")
    for (case_name, window_map) in case_map
        sort(collect(keys(window_map))) == EXPECTED_WINDOWS ||
            error("$scenario/$case_name does not contain exactly six windows")
    end
    return case_map, all_proxies
end

"""Build the exact title shown above one case frame."""
function frame_title(scenario::AbstractString, case_name::AbstractString)
    parsed = parse_case_name(case_name)
    return @sprintf("%s | case %03d | fault depth = %d, sand Vcl = %.1f, clay Vcl = %.1f",
                    scenario_display_name(scenario), parsed.index,
                    parsed.fault_depth, parsed.sand_vcl, parsed.clay_vcl)
end

"""Fail early when the Python/Pillow GIF packaging dependency is unavailable."""
function validate_gif_tooling()
    run(`python -c "from PIL import Image"`)
end

"""
    assemble_gif(frame_root, gif_path, duration_ms)

Assemble the rendered PNG frames into a looping palette GIF. Pillow packages
and palette-quantizes the frames but does not redraw or rescale them.
"""
function assemble_gif(frame_root::AbstractString,
                      gif_path::AbstractString,
                      duration_ms::Int)
    python_code = raw"""
from pathlib import Path
import sys
from PIL import Image

frame_root = Path(sys.argv[1])
gif_path = Path(sys.argv[2])
duration_ms = int(sys.argv[3])
paths = sorted(frame_root.glob("frame_*.png"))
if len(paths) != 27:
    raise RuntimeError(f"Expected 27 frames, found {len(paths)} in {frame_root}")

frames = []
for path in paths:
    with Image.open(path) as image:
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration_ms,
    loop=0,
    optimize=True,
    disposal=2,
)

with Image.open(gif_path) as result:
    if result.n_frames != 27:
        raise RuntimeError(f"Expected 27 GIF frames, found {result.n_frames}")
    if result.size != (2970, 1900):
        raise RuntimeError(f"Unexpected GIF size: {result.size}")
"""
    mkpath(dirname(gif_path))
    run(`python -c $python_code $frame_root $gif_path $duration_ms`)
    return gif_path
end

"""Generate all case frames and one GIF for a thickness scenario."""
function generate_scenario(data_root::AbstractString,
                           output_root::AbstractString,
                           scenario::AbstractString,
                           bins::Int,
                           duration_ms::Int,
                           overwrite::Bool)
    println("\n=== $scenario ===")
    case_map, all_proxies = load_scenario_libraries(data_root, scenario)
    bin_specs = build_global_bin_specs(all_proxies, bins)
    cases = sort(collect(keys(case_map)); by = name -> parse_case_name(name).index)
    frame_root = joinpath(output_root, "frames", scenario)
    mkpath(frame_root)

    manifest_rows = Vector{Vector{String}}()
    for (position, case_name) in enumerate(cases)
        parsed = parse_case_name(case_name)
        frame_path = joinpath(frame_root, @sprintf("frame_%03d.png", parsed.index))
        title = frame_title(scenario, case_name)

        if overwrite || !isfile(frame_path)
            proxies = [case_map[case_name][window] for window in EXPECTED_WINDOWS]
            figure = build_combined_publication_figure(proxies, bin_specs)
            figure.content[1].text[] = title
            figure.scene.viewport[] = Rect2i(0, 0, 2970, 1900)
            save(frame_path, figure; px_per_unit = 1)
            position % 5 == 0 && GC.gc()
        end

        push!(manifest_rows, [
            scenario,
            string(parsed.index),
            case_name,
            title,
            frame_path,
        ])
        println(@sprintf("  frame %02d / %02d: case %03d", position,
                         length(cases), parsed.index))
        flush(stdout)
    end

    gif_name = "$(scenario)_marginal_distributions_collapsed_cell_union_level2_format.gif"
    gif_path = joinpath(output_root, gif_name)
    if overwrite || !isfile(gif_path)
        assemble_gif(frame_root, gif_path, duration_ms)
    end
    println("  GIF: $gif_path")
    return manifest_rows, gif_path
end

"""Run the GIF workflow and write a frame manifest."""
function main(args::Vector{String})
    options = parse_args(args)
    data_root = normpath(options["data-root"])
    output_root = normpath(options["output-root"])
    bins = parse(Int, options["bins"])
    duration_ms = parse(Int, options["duration-ms"])
    overwrite = parse_bool(options["overwrite"])

    bins >= 10 || error("--bins must be at least 10")
    duration_ms > 0 || error("--duration-ms must be positive")
    isdir(data_root) || error("PREDICT data root does not exist: $data_root")
    validate_gif_tooling()
    mkpath(output_root)
    Level2Plotting.activate_plot_theme!()

    all_manifest_rows = Vector{Vector{String}}()
    for scenario in selected_scenarios(data_root, options["scenario"])
        rows, gif_path = generate_scenario(data_root, output_root, scenario,
                                           bins, duration_ms, overwrite)
        for row in rows
            push!(row, gif_path)
            push!(row, data_root)
            push!(all_manifest_rows, row)
        end
    end

    Level2IO.write_csv(joinpath(output_root, "frame_manifest.csv"),
                       ["scenario", "case_index", "case_name", "title",
                        "frame_path", "gif_path", "data_root"],
                       all_manifest_rows)
    println("\nSaved GIF workflow outputs to $output_root")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
