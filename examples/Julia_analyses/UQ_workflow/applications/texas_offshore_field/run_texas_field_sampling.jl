#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

"""
    run_texas_field_sampling.jl

Application driver for extending the three-level UQ sampling workflow to the
Texas offshore field model with 87 along-strike fault slices.

The core Level 2/Level 3 workflow is not modified. This script reloads:

1. Level 2 state objects for each geology/window.
2. Deterministic Level 3 case/window pool assignments.

It then repeats those case assignments over field-slice draw groups and draws
one actual joint PREDICT realization per window from the requested pool.
"""

include(joinpath(@__DIR__, "..", "..", "level3", "lib", "level3_io.jl"))
include(joinpath(@__DIR__, "..", "..", "level2", "lib", "level2_selection.jl"))
include(joinpath(@__DIR__, "lib", "texas_field_io.jl"))
include(joinpath(@__DIR__, "lib", "texas_field_slices.jl"))
include(joinpath(@__DIR__, "lib", "texas_field_sampler.jl"))
include(joinpath(@__DIR__, "lib", "texas_field_outputs.jl"))
include(joinpath(@__DIR__, "lib", "texas_field_mat_export.jl"))

using .Level3IO
using .Level2Selection
using .TexasFieldIO
using .TexasFieldSlices
using .TexasFieldSampler
using .TexasFieldOutputs
using .TexasFieldMatExport

"""
    parse_args(args)

Parse command-line options for the Texas field sampling application.
"""
function parse_args(args::Vector{String})
    options = Dict(
        "config" => joinpath(@__DIR__, "texas_field_config.toml"),
        "output-root" => "",
        "max-geologies" => "",
        "only-geology" => "",
        "case-ids" => "",
        "list-only" => "",
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

"""
    print_help()

Print command-line usage.
"""
function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/applications/texas_offshore_field/run_texas_field_sampling.jl [options]")
    println()
    println("Options:")
    println("  --config <path>          Texas field TOML config")
    println("  --output-root <path>     Override output folder")
    println("  --max-geologies <n>      Limit number of geologies processed; 0 means all")
    println("  --only-geology <text>    Process only geology IDs containing this text")
    println("  --case-ids <csv>         Optional comma-separated Level 3 case IDs")
    println("  --list-only <true|false> Build slice groups and list workload only")
    println("  -h, --help               Show this help")
end

"""
    main(args)

Run the Texas field sampling application.
"""
function main(args::Vector{String})
    opt = parse_args(args)
    config = TexasFieldIO.read_texas_config(
        opt["config"];
        output_root_override = opt["output-root"],
        max_geologies_override = opt["max-geologies"],
        only_geology_override = opt["only-geology"],
        case_ids_override = opt["case-ids"],
        list_only_override = opt["list-only"],
    )

    assignments = TexasFieldIO.read_csv_dicts(config["assignment_table"])
    geology_ids = TexasFieldIO.select_geology_ids(assignments, config)
    draw_groups = TexasFieldSlices.build_slice_draw_groups(
        Int(config["num_slices"]),
        Vector(config["shared_slice_groups"]),
    )
    case_ids_filter = Vector{Int}(config["case_ids"])

    expected_cases = isempty(case_ids_filter) ? 10 : length(case_ids_filter)
    println("Texas offshore field sampling")
    println("  geologies: $(length(geology_ids))")
    println("  cases/geology: $expected_cases")
    println("  slices: $(config["num_slices"])")
    println("  draw groups: $(length(draw_groups))")
    println("  output: $(config["output_root"])")

    if Bool(config["list_only"])
        return nothing
    end

    output_root = String(config["output_root"])
    prepare_output_root(output_root, Bool(config["overwrite"]))
    TexasFieldOutputs.write_slice_draw_groups(
        joinpath(output_root, "texas_field_slice_draw_groups.csv"),
        TexasFieldSlices.slice_group_rows(draw_groups),
    )

    unique_path = joinpath(output_root, "texas_field_unique_draw_values.csv")
    slice_window_path = joinpath(output_root, "texas_field_slice_window_values.csv")
    matrix_path = joinpath(output_root, "texas_field_slice_case_matrix.csv")

    counts = Dict(
        "geologies" => 0,
        "cases" => 0,
        "unique_draw_window_rows" => 0,
        "slice_window_rows" => 0,
        "slice_case_rows" => 0,
    )

    open(unique_path, "w") do unique_io
        open(slice_window_path, "w") do slice_io
            open(matrix_path, "w") do matrix_io
                TexasFieldOutputs.write_csv_row(unique_io, TexasFieldOutputs.UNIQUE_DRAW_HEADER)
                TexasFieldOutputs.write_csv_row(slice_io, TexasFieldOutputs.SLICE_WINDOW_HEADER)
                TexasFieldOutputs.write_csv_row(matrix_io, TexasFieldOutputs.SLICE_CASE_MATRIX_HEADER)

                for (gi, geology_id) in enumerate(geology_ids)
                    println("  [$gi/$(length(geology_ids))] $geology_id")
                    geology_rows = [row for row in assignments if row["geology_id"] == geology_id]
                    geology_rows = TexasFieldIO.filter_case_rows(geology_rows, case_ids_filter)
                    isempty(geology_rows) && error("No assignment rows after filtering for geology $geology_id")

                    state_root = joinpath(String(config["level2_root"]), geology_id)
                    states = Level3IO.load_level2_states(state_root, String.(config["windows"]))
                    source_metadata = TexasFieldSampler.source_metadata_for_states(states)
                    rows_by_case = TexasFieldSampler.group_case_rows(geology_rows)

                    for case_id in sort(collect(keys(rows_by_case)))
                        case_rows = rows_by_case[case_id]
                        validate_case_window_rows(case_rows, String.(config["windows"]))
                        counts["cases"] += 1

                        for (draw_group_index, slices) in enumerate(draw_groups)
                            draw_seed = TexasFieldSampler.deterministic_draw_seed(
                                Int(config["random_seed"]),
                                first(case_rows),
                                case_id,
                                draw_group_index,
                            )
                            sampled_rows = TexasFieldSampler.sample_case_draw_group(
                                states,
                                case_rows;
                                random_seed = draw_seed,
                                source_metadata = source_metadata,
                            )

                            TexasFieldOutputs.write_unique_draw_rows!(
                                unique_io,
                                sampled_rows,
                                draw_group_index,
                                slices,
                                draw_seed,
                                Int(config["random_seed"]),
                            )
                            TexasFieldOutputs.write_slice_window_rows!(
                                slice_io,
                                sampled_rows,
                                draw_group_index,
                                slices,
                                draw_seed,
                                Int(config["random_seed"]),
                            )
                            TexasFieldOutputs.write_slice_case_matrix_rows!(
                                matrix_io,
                                sampled_rows,
                                draw_group_index,
                                slices,
                                draw_seed,
                                Int(config["random_seed"]),
                            )

                            counts["unique_draw_window_rows"] += length(sampled_rows)
                            counts["slice_window_rows"] += length(sampled_rows) * length(slices)
                            counts["slice_case_rows"] += length(slices)
                        end
                    end
                    counts["geologies"] += 1
                end
            end
        end
    end

    write_sampling_report(config, draw_groups, counts, output_root)
    if Bool(config["export_mat"])
        mat_path = TexasFieldMatExport.export_texas_field_mat(output_root)
        println("  MAT export: $mat_path")
        if Bool(config["validate_mat"])
            validation = TexasFieldMatExport.validate_mat_against_csv(output_root; mat_path = mat_path)
            println("  MAT validation mismatch count: $(validation["mismatch_count"])")
            validation["mismatch_count"] == 0 || error("MAT export does not match CSV outputs")
        end
    end
    println("Finished Texas field sampling.")
    println("  unique draw window rows: $(counts["unique_draw_window_rows"])")
    println("  slice window rows: $(counts["slice_window_rows"])")
    println("  slice case rows: $(counts["slice_case_rows"])")
end

"""
    prepare_output_root(output_root, overwrite)

Create the output folder and guard against accidentally appending to old files.
"""
function prepare_output_root(output_root::AbstractString, overwrite::Bool)
    mkpath(output_root)
    output_files = [
        "texas_field_unique_draw_values.csv",
        "texas_field_slice_window_values.csv",
        "texas_field_slice_case_matrix.csv",
        "texas_field_slice_draw_groups.csv",
        "texas_field_sampling_report.txt",
    ]
    for file in output_files
        path = joinpath(output_root, file)
        if isfile(path)
            overwrite || error("Output file exists and overwrite=false: $path")
            rm(path)
        end
    end
end

"""
    validate_case_window_rows(case_rows, windows)

Check that one Level 3 case has exactly one assignment row for every window.
"""
function validate_case_window_rows(case_rows::Vector{Dict{String, String}},
                                   windows::Vector{String})
    found = sort([row["window"] for row in case_rows])
    expected = sort(windows)
    found == expected || error("Case rows do not match expected windows. Found $(join(found, ", ")); expected $(join(expected, ", "))")
    return true
end

"""
    write_sampling_report(config, draw_groups, counts, output_root)

Write a compact audit report.
"""
function write_sampling_report(config::Dict{String, Any},
                               draw_groups::Vector{Vector{Int}},
                               counts::Dict{String, Int},
                               output_root::AbstractString)
    lines = String[
        "Texas offshore field sampling report",
        "created_at = $(TexasFieldIO.timestamp_string())",
        "config_path = $(config["config_path"])",
        "summary_root = $(config["summary_root"])",
        "level2_root = $(config["level2_root"])",
        "output_root = $(config["output_root"])",
        "sampler_random_seed = $(config["random_seed"])",
        "num_slices = $(config["num_slices"])",
        "draw_group_count = $(length(draw_groups))",
        "shared_draw_groups = " * join([join(group, ";") for group in draw_groups if length(group) > 1], " | "),
        "geologies_processed = $(counts["geologies"])",
        "geology_case_count = $(counts["cases"])",
        "unique_draw_window_rows = $(counts["unique_draw_window_rows"])",
        "slice_window_rows = $(counts["slice_window_rows"])",
        "slice_case_rows = $(counts["slice_case_rows"])",
        "",
        "Outputs:",
        "  texas_field_unique_draw_values.csv",
        "  texas_field_slice_window_values.csv",
        "  texas_field_slice_case_matrix.csv",
        "  texas_field_slice_draw_groups.csv",
    ]
    TexasFieldOutputs.write_report(joinpath(output_root, "texas_field_sampling_report.txt"), lines)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
