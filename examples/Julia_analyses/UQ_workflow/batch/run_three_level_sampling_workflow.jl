#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using TOML
using Dates
using Printf

"""
    run_three_level_sampling_workflow.jl

Batch driver that applies the current three-level UQ sampling workflow to all
162 geologic scenarios in `examples/thickness_scenario_data`.

The script builds a Level 1 geology catalog from the completed PREDICT outputs,
generates one Level 2 manifest/config per geology, runs the existing modular
Level 2 driver, and then runs the existing Level 3 driver to create
multiple-window permeability cases. The scientific Level 2 and Level 3 logic is
kept in the existing per-geology drivers; this file only orchestrates them.
"""

const WORKFLOW_ROOT = normpath(joinpath(@__DIR__, ".."))
const REPO_ROOT = normpath(joinpath(WORKFLOW_ROOT, "..", "..", ".."))
const FIXED_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]

"""
    parse_args(args)

Parse command-line options for the 162-geology batch workflow.
"""
function parse_args(args::Vector{String})
    options = Dict(
        "config" => joinpath(@__DIR__, "three_level_sampling_config.toml"),
        "max-geologies" => "",
        "only-geology" => "",
        "resume" => "",
        "run-level2" => "",
        "run-level3" => "",
        "list-only" => "false",
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

Print usage for the batch workflow.
"""
function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/batch/run_three_level_sampling_workflow.jl [options]")
    println()
    println("Options:")
    println("  --config <path>         Batch workflow TOML config")
    println("  --max-geologies <n>     Limit number of geologies processed; 0 means all")
    println("  --only-geology <text>   Process only geology IDs containing this text")
    println("  --resume <true|false>   Override resume behavior")
    println("  --run-level2 <true|false>")
    println("  --run-level3 <true|false>")
    println("  --list-only <true|false>  Build catalogs/manifests but do not run drivers")
    println("  -h, --help              Show this help")
end

"""
    read_batch_config(path, cli)

Read the batch TOML file, apply command-line overrides, and resolve paths.
"""
function read_batch_config(path::AbstractString, cli::Dict{String, String})
    config_path = normpath(path)
    raw = TOML.parsefile(config_path)
    paths = get(raw, "paths", Dict{String, Any}())
    run_cfg = get(raw, "run", Dict{String, Any}())

    cfg = Dict{String, Any}(
        "config_path" => config_path,
        "predict_data_root" => resolve_config_path(config_path, String(get(paths, "predict_data_root", "../../../thickness_scenario_data"))),
        "output_root" => resolve_config_path(config_path, String(get(paths, "output_root", joinpath(WORKFLOW_ROOT, "outputs", "three_level_sampling_162")))),
        "level2_driver" => resolve_config_path(config_path, String(get(paths, "level2_driver", "../level2/workflow/run_level2_workflow.jl"))),
        "level3_driver" => resolve_config_path(config_path, String(get(paths, "level3_driver", "../level3/workflow/run_level3_workflow.jl"))),
        "fixed_windows" => String.(get(get(raw, "workflow", Dict{String, Any}()), "fixed_windows", FIXED_WINDOWS)),
        "run_level2" => Bool(get(run_cfg, "run_level2", true)),
        "run_level3" => Bool(get(run_cfg, "run_level3", true)),
        "resume" => Bool(get(run_cfg, "resume", true)),
        "stop_on_error" => Bool(get(run_cfg, "stop_on_error", true)),
        "max_geologies" => Int(get(run_cfg, "max_geologies", 0)),
        "list_only" => parse_bool(get(cli, "list-only", "false")),
        "raw" => raw,
    )

    cfg["fixed_windows"] == FIXED_WINDOWS ||
        error("The batch workflow expects fixed windows $(join(FIXED_WINDOWS, ", "))")

    if !isempty(cli["max-geologies"])
        cfg["max_geologies"] = parse(Int, cli["max-geologies"])
    end
    if !isempty(cli["resume"])
        cfg["resume"] = parse_bool(cli["resume"])
    end
    if !isempty(cli["run-level2"])
        cfg["run_level2"] = parse_bool(cli["run-level2"])
    end
    if !isempty(cli["run-level3"])
        cfg["run_level3"] = parse_bool(cli["run-level3"])
    end
    cfg["only_geology"] = cli["only-geology"]

    isdir(cfg["predict_data_root"]) ||
        error("PREDICT data root does not exist: $(cfg["predict_data_root"])")
    isfile(cfg["level2_driver"]) ||
        error("Level 2 driver does not exist: $(cfg["level2_driver"])")
    isfile(cfg["level3_driver"]) ||
        error("Level 3 driver does not exist: $(cfg["level3_driver"])")
    return cfg
end

"""
    resolve_config_path(config_path, path)

Resolve `path` relative to a TOML config file location unless it is already
absolute.
"""
function resolve_config_path(config_path::AbstractString, path::AbstractString)
    isempty(path) && return ""
    return isabspath(path) ? normpath(path) : normpath(joinpath(dirname(config_path), path))
end

"""
    parse_bool(value)

Parse a TOML/CLI boolean value.
"""
function parse_bool(value)
    value isa Bool && return value
    text = lowercase(strip(String(value)))
    text in ("true", "1", "yes", "y") && return true
    text in ("false", "0", "no", "n") && return false
    error("Cannot parse boolean value: $value")
end

"""
    build_geology_catalog(cfg)

Build the 162-row Level 1 geology catalog and the 972-row window-library
catalog from the completed PREDICT simulation folder.
"""
function build_geology_catalog(cfg::Dict{String, Any})
    data_root = String(cfg["predict_data_root"])
    cases = read_csv_dicts(joinpath(data_root, "geology_case_definitions.csv"))
    scenario_rows = read_csv_dicts(joinpath(data_root, "thickness_scenario_definitions.csv"))

    scenario_info = Dict{String, Dict{String, String}}()
    for row in scenario_rows
        label = row["ScenarioLabel"]
        if !haskey(scenario_info, label)
            scenario_info[label] = Dict(
                "ScenarioIndex" => row["ScenarioIndex"],
                "ScenarioLabel" => row["ScenarioLabel"],
                "ScenarioName" => row["ScenarioName"],
            )
        end
    end

    scenarios = sort(collect(values(scenario_info)); by = row -> parse(Int, row["ScenarioIndex"]))
    sorted_cases = sort(cases; by = row -> parse(Int, row["CaseIndex"]))

    geologies = Vector{Dict{String, Any}}()
    windows = String.(cfg["fixed_windows"])
    for scenario in scenarios
        scenario_index = parse(Int, scenario["ScenarioIndex"])
        scenario_label = scenario["ScenarioLabel"]
        for case in sorted_cases
            case_index = parse(Int, case["CaseIndex"])
            case_label = case["CaseLabel"]
            geology_id = @sprintf("s%02d_c%03d", scenario_index, case_index)
            window_rows = Vector{Dict{String, Any}}()
            all_present = true
            for window in windows
                mat_path = joinpath(data_root, "data", scenario_label, window, case_label, "predict_runs.mat")
                present = isfile(mat_path)
                all_present &= present
                push!(window_rows, Dict{String, Any}(
                    "geology_id" => geology_id,
                    "window" => window,
                    "sample_kind" => "predict_2000",
                    "mat_path" => repo_relative_path(mat_path),
                    "absolute_mat_path" => normpath(mat_path),
                    "n_samples" => "2000",
                    "file_exists" => present,
                ))
            end
            push!(geologies, Dict{String, Any}(
                "geology_id" => geology_id,
                "scenario_index" => scenario_index,
                "scenario_label" => scenario_label,
                "scenario_name" => scenario["ScenarioName"],
                "case_index" => case_index,
                "case_label" => case_label,
                "faulting_depth" => case["FaultingDepth"],
                "sand_vcl" => case["SandVcl"],
                "clay_vcl" => case["ClayVcl"],
                "window_rows" => window_rows,
                "all_files_present" => all_present,
            ))
        end
    end
    return geologies
end

"""
    repo_relative_path(path)

Return a repository-relative path with forward slashes, which Level 2 resolves
against the repo root.
"""
function repo_relative_path(path::AbstractString)
    rel = relpath(normpath(path), REPO_ROOT)
    return replace(rel, "\\" => "/")
end

"""
    write_catalog_outputs(geologies, cfg)

Write the Level 1 master geology catalog and all-window library catalog.
"""
function write_catalog_outputs(geologies::Vector{Dict{String, Any}}, cfg::Dict{String, Any})
    catalog_root = joinpath(String(cfg["output_root"]), "catalog")
    mkpath(catalog_root)

    geology_rows = Vector{Vector{String}}()
    window_rows = Vector{Vector{String}}()
    for geology in geologies
        push!(geology_rows, [
            geology["geology_id"],
            string(geology["scenario_index"]),
            geology["scenario_label"],
            geology["scenario_name"],
            string(geology["case_index"]),
            geology["case_label"],
            string(geology["faulting_depth"]),
            string(geology["sand_vcl"]),
            string(geology["clay_vcl"]),
            string(geology["all_files_present"]),
        ])
        for row in geology["window_rows"]
            push!(window_rows, [
                row["geology_id"],
                row["window"],
                row["sample_kind"],
                row["mat_path"],
                row["n_samples"],
                string(row["file_exists"]),
            ])
        end
    end

    write_csv(joinpath(catalog_root, "level1_geology_catalog.csv"),
              ["geology_id", "scenario_index", "scenario_label", "scenario_name",
               "case_index", "case_label", "faulting_depth_m", "sand_vcl",
               "clay_vcl", "all_window_files_present"],
              geology_rows)
    write_csv(joinpath(catalog_root, "predict_window_library_catalog.csv"),
              ["geology_id", "window", "sample_kind", "mat_path", "n_samples", "file_exists"],
              window_rows)
end

"""
    prepare_geology_configs(geology, cfg)

Write one Level 2 manifest, one Level 2 workflow config, and one Level 3
workflow config for a geology.
"""
function prepare_geology_configs(geology::Dict{String, Any}, cfg::Dict{String, Any})
    output_root = String(cfg["output_root"])
    geology_id = String(geology["geology_id"])
    config_root = joinpath(output_root, "generated_configs")
    manifest_root = joinpath(output_root, "manifests")
    mkpath(config_root)
    mkpath(manifest_root)

    level2_root = joinpath(output_root, "level2", geology_id)
    level3_root = joinpath(output_root, "level3", geology_id)
    level2_base_path = joinpath(config_root, "level2_base", "$(geology_id)_level2_defaults.toml")
    level2_workflow_path = joinpath(config_root, "level2_workflow", "$(geology_id)_level2_workflow.toml")
    level3_workflow_path = joinpath(config_root, "level3_workflow", "$(geology_id)_level3_workflow.toml")
    manifest_path = joinpath(manifest_root, "$(geology_id)_level2_manifest.csv")

    write_level2_manifest(manifest_path, geology)
    write_level2_base_config(level2_base_path, geology, cfg)
    write_level2_workflow_config(level2_workflow_path, level2_base_path, manifest_path, level2_root, cfg)
    write_level3_workflow_config(level3_workflow_path, geology, level2_root, level3_root, cfg)

    return Dict{String, String}(
        "level2_root" => level2_root,
        "level3_root" => level3_root,
        "level2_workflow_config" => level2_workflow_path,
        "level3_workflow_config" => level3_workflow_path,
        "manifest" => manifest_path,
    )
end

"""
    write_level2_manifest(path, geology)

Write the six-row Level 2 manifest expected by the existing Level 2 driver.
"""
function write_level2_manifest(path::AbstractString, geology::Dict{String, Any})
    rows = Vector{Vector{String}}()
    for row in geology["window_rows"]
        push!(rows, [
            row["geology_id"],
            row["window"],
            row["sample_kind"],
            row["mat_path"],
            row["n_samples"],
        ])
    end
    write_csv(path, ["geology_id", "window", "sample_kind", "mat_path", "n_samples"], rows)
end

"""
    write_level2_base_config(path, geology, cfg)

Write a per-geology Level 2 baseline config. This is intentionally explicit so
the existing Level 2 driver can continue validating that the manifest geology
ID matches the config geology ID.
"""
function write_level2_base_config(path::AbstractString, geology::Dict{String, Any}, cfg::Dict{String, Any})
    raw = Dict{String, Any}(cfg["raw"])
    level2 = get(raw, "level2", Dict{String, Any}())
    plotting = get(level2, "plotting", Dict{String, Any}())
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "[workflow]")
        println(io, "geology_id = $(toml_quote(geology["geology_id"]))")
        println(io, "fixed_windows = $(toml_value(cfg["fixed_windows"]))")
        println(io)
        println(io, "[level2]")
        for key in ["state_fraction", "local_pool_fraction", "local_pool_min_count",
                    "weights", "distance_metric", "distance_weights", "max_k",
                    "silhouette_threshold", "min_cluster_fraction", "min_cluster_size",
                    "random_seed", "max_kmedoids_iter", "num_restarts"]
            haskey(level2, key) || error("Missing [level2].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(level2[key]))")
        end
        println(io)
        println(io, "[validation]")
        println(io, "holdout_repeats = []")
        println(io)
        println(io, "[plotting]")
        for key in ["state_violin_fixed_count_density_reference",
                    "local_pool_violin_fixed_count_density_reference",
                    "state_wide_pool_violin_fixed_count_density_reference"]
            haskey(plotting, key) || error("Missing [level2.plotting].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(plotting[key]))")
        end
    end
end

"""
    write_level2_workflow_config(path, base_path, manifest_path, output_root, cfg)

Write a Level 2 workflow config that points to the per-geology baseline and
manifest.
"""
function write_level2_workflow_config(path::AbstractString,
                                      base_path::AbstractString,
                                      manifest_path::AbstractString,
                                      output_root::AbstractString,
                                      cfg::Dict{String, Any})
    level2_run = get(get(Dict{String, Any}(cfg["raw"]), "level2", Dict{String, Any}()), "run", Dict{String, Any}())
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "[paths]")
        println(io, "base_config = $(toml_quote(path_for_toml(base_path)))")
        println(io, "manifest = $(toml_quote(path_for_toml(manifest_path)))")
        println(io, "output_root = $(toml_quote(path_for_toml(output_root)))")
        println(io)
        println(io, "[run]")
        for key in ["screen_input_data", "build_level2_objects", "export_tables",
                    "plot_joint_clusters", "plot_joint_clusters_3d",
                    "plot_state_libraries", "plot_perturbation_pools",
                    "run_sampling_test", "run_validation",
                    "run_joint_cluster_sensitivity", "run_joint_cluster_bootstrap"]
            println(io, "$key = $(toml_value(Bool(get(level2_run, key, false))))")
        end
    end
end

"""
    write_level3_workflow_config(path, geology, level2_root, output_root, cfg)

Write a Level 3 workflow config for one geology.
"""
function write_level3_workflow_config(path::AbstractString,
                                      geology::Dict{String, Any},
                                      level2_root::AbstractString,
                                      output_root::AbstractString,
                                      cfg::Dict{String, Any})
    raw = Dict{String, Any}(cfg["raw"])
    level3 = get(raw, "level3", Dict{String, Any}())
    level3_run = get(level3, "run", Dict{String, Any}())
    grouping = get(level3, "grouping", Dict{String, Any}())
    bootstrap = get(level3, "bootstrap", Dict{String, Any}())
    sampling = get(level3, "sampling", Dict{String, Any}())
    figures = get(level3, "figures", Dict{String, Any}())

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "[workflow]")
        println(io, "geology_id = $(toml_quote(geology["geology_id"]))")
        println(io, "fixed_windows = $(toml_value(cfg["fixed_windows"]))")
        println(io)
        println(io, "[paths]")
        println(io, "level2_state_root = $(toml_quote(path_for_toml(level2_root)))")
        println(io, "output_root = $(toml_quote(path_for_toml(output_root)))")
        println(io)
        println(io, "[run]")
        for key in ["load_level2_states", "build_window_similarity_groups",
                    "make_grouping_distance_figures", "run_bootstrap_grouping_qa",
                    "make_bootstrap_grouping_figures", "make_similarity_group_figure",
                    "build_multiple_window_permeability_cases",
                    "make_multiple_window_permeability_case_figure",
                    "sample_multiple_window_permeability_cases"]
            println(io, "$key = $(toml_value(Bool(get(level3_run, key, false))))")
        end
        println(io)
        println(io, "[level3]")
        for key in ["distance_metric", "normalization", "pairwise_sample_size", "random_seed"]
            haskey(level3, key) || error("Missing [level3].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(level3[key]))")
        end
        println(io)
        println(io, "[level3.grouping]")
        for key in ["grouping_mode", "similarity_threshold"]
            haskey(grouping, key) || error("Missing [level3.grouping].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(grouping[key]))")
        end
        println(io)
        println(io, "[level3.bootstrap]")
        for key in ["bootstrap_count", "bootstrap_sample_size", "similarity_threshold",
                    "stable_pair_probability", "random_seed", "show_progress"]
            haskey(bootstrap, key) || error("Missing [level3.bootstrap].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(bootstrap[key]))")
        end
        println(io)
        println(io, "[level3.sampling]")
        for key in ["random_seed"]
            haskey(sampling, key) || error("Missing [level3.sampling].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(sampling[key]))")
        end
        println(io)
        println(io, "[figures]")
        for key in ["similarity_threshold", "formats"]
            haskey(figures, key) || error("Missing [level3.figures].$key in $(cfg["config_path"])")
            println(io, "$key = $(toml_value(figures[key]))")
        end
    end
end

"""
    run_geology_workflow(geology, cfg, generated)

Run Level 2 and Level 3 for one geology, respecting resume settings.
"""
function run_geology_workflow(geology::Dict{String, Any},
                              cfg::Dict{String, Any},
                              generated::Dict{String, String})
    geology_id = String(geology["geology_id"])
    row = Dict{String, String}(
        "geology_id" => geology_id,
        "scenario_label" => String(geology["scenario_label"]),
        "case_label" => String(geology["case_label"]),
        "level2_status" => "not_run",
        "level3_status" => "not_run",
        "level2_root" => generated["level2_root"],
        "level3_root" => generated["level3_root"],
        "message" => "",
    )

    !Bool(geology["all_files_present"]) && error("Missing one or more MAT files for $geology_id")

    if Bool(cfg["run_level2"])
        if Bool(cfg["resume"]) && level2_done(generated["level2_root"])
            row["level2_status"] = "skipped_done"
        else
            println("  Level 2: $geology_id")
            run_julia_driver(String(cfg["level2_driver"]),
                             ["--config", generated["level2_workflow_config"],
                              "--output-root", generated["level2_root"]])
            row["level2_status"] = "completed"
        end
    end

    if Bool(cfg["run_level3"])
        if Bool(cfg["resume"]) && level3_done(generated["level3_root"])
            row["level3_status"] = "skipped_done"
        else
            println("  Level 3: $geology_id")
            run_julia_driver(String(cfg["level3_driver"]),
                             ["--config", generated["level3_workflow_config"],
                              "--output-root", generated["level3_root"],
                              "--level2-state-root", generated["level2_root"]])
            row["level3_status"] = "completed"
        end
    end

    return row
end

"""
    run_julia_driver(driver, args)

Run one existing workflow driver as a child Julia process.
"""
function run_julia_driver(driver::AbstractString, args::Vector{String})
    cmd_parts = ["julia", "--project=$(WORKFLOW_ROOT)", String(driver)]
    append!(cmd_parts, String.(args))
    cmd = Cmd(cmd_parts)
    run(cmd)
end

"""
    level2_done(root)

Return true when all six Level 2 window-state MAT files and the build summary
exist.
"""
function level2_done(root::AbstractString)
    summary = joinpath(root, "tables", "level2_build_summary.csv")
    isfile(summary) || return false
    return all(isfile(joinpath(root, "window_states", window, "$(window)_level2_state.mat"))
               for window in FIXED_WINDOWS)
end

"""
    level3_done(root)

Return true when the Level 3 grouping summary and multiple-window permeability
case tables exist.
"""
function level3_done(root::AbstractString)
    return isfile(joinpath(root, "tables", "window_similarity_group_summary.csv")) &&
           isfile(joinpath(root, "tables", "multiple_window_permeability_case_definitions.csv")) &&
           isfile(joinpath(root, "tables", "multiple_window_permeability_window_assignments.csv")) &&
           isfile(joinpath(root, "tables", "multiple_window_permeability_sampled_values.csv")) &&
           isfile(joinpath(root, "tables", "multiple_window_permeability_sampled_case_matrix.csv"))
end

"""
    filter_geologies(geologies, cfg)

Apply optional CLI filtering and max-geology limit.
"""
function filter_geologies(geologies::Vector{Dict{String, Any}}, cfg::Dict{String, Any})
    selected = geologies
    pattern = String(get(cfg, "only_geology", ""))
    if !isempty(pattern)
        selected = [g for g in selected if occursin(pattern, String(g["geology_id"])) ||
                                      occursin(pattern, String(g["scenario_label"])) ||
                                      occursin(pattern, String(g["case_label"]))]
    end
    max_geologies = Int(cfg["max_geologies"])
    if max_geologies > 0
        selected = selected[1:min(max_geologies, length(selected))]
    end
    return selected
end

"""
    write_status(path, rows)

Write the running batch status table.
"""
function write_status(path::AbstractString, rows::Vector{Dict{String, String}})
    header = ["geology_id", "scenario_label", "case_label", "level2_status",
              "level3_status", "level2_root", "level3_root", "message"]
    table_rows = [[get(row, key, "") for key in header] for row in rows]
    write_csv(path, header, table_rows)
end

"""
    read_csv_dicts(path)

Read a small CSV file into dictionaries. Supports quoted fields, including the
scenario-name commas in the thickness metadata.
"""
function read_csv_dicts(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("CSV is empty: $path")
    header = parse_csv_line(lines[1])
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = parse_csv_line(line)
        length(parts) == length(header) ||
            error("Malformed CSV row in $path: $line")
        push!(rows, Dict(key => value for (key, value) in zip(header, parts)))
    end
    return rows
end

"""
    parse_csv_line(line)

Parse one CSV line with a minimal RFC4180-compatible quoted-field parser.
"""
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

"""
    write_csv(path, header, rows)

Write a CSV file with simple escaping.
"""
function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(csv_escape.(header), ","))
        for row in rows
            length(row) == length(header) ||
                error("CSV row length does not match header for $path")
            println(io, join(csv_escape.(row), ","))
        end
    end
end

csv_escape(value) = csv_escape(String(value))
function csv_escape(value::AbstractString)
    escaped = replace(value, "\"" => "\"\"")
    if occursin(',', escaped) || occursin('"', escaped) || occursin('\n', escaped)
        return "\"" * escaped * "\""
    end
    return escaped
end

path_for_toml(path::AbstractString) = replace(normpath(path), "\\" => "/")
toml_quote(value) = "\"" * replace(String(value), "\\" => "\\\\", "\"" => "\\\"") * "\""

"""
    toml_value(value)

Format basic Julia values as TOML literals.
"""
function toml_value(value)
    value isa AbstractString && return toml_quote(value)
    value isa Bool && return value ? "true" : "false"
    value isa Integer && return string(value)
    value isa AbstractFloat && return string(value)
    value isa AbstractVector && return "[" * join(toml_value.(value), ", ") * "]"
    error("Unsupported TOML value: $value")
end

"""
    main(args)

Entry point for the full 162-geology three-level sampling workflow.
"""
function main(args::Vector{String})
    cli = parse_args(args)
    cfg = read_batch_config(cli["config"], cli)
    mkpath(String(cfg["output_root"]))

    println("Building Level 1 geology catalog from $(cfg["predict_data_root"])")
    geologies = build_geology_catalog(cfg)
    write_catalog_outputs(geologies, cfg)
    selected = filter_geologies(geologies, cfg)

    println("Catalog geologies: $(length(geologies)); selected for this run: $(length(selected))")
    println("Output root: $(cfg["output_root"])")
    if Bool(cfg["list_only"])
        println("List-only mode: generated catalog only.")
        return
    end

    status_rows = Dict{String, String}[]
    status_path = joinpath(String(cfg["output_root"]), "batch_status.csv")
    for (i, geology) in enumerate(selected)
        geology_id = String(geology["geology_id"])
        println()
        println("[$i / $(length(selected))] $geology_id | $(geology["scenario_label"]) | $(geology["case_label"])")
        generated = prepare_geology_configs(geology, cfg)
        try
            row = run_geology_workflow(geology, cfg, generated)
            row["message"] = "ok"
            push!(status_rows, row)
        catch err
            message = sprint(showerror, err)
            push!(status_rows, Dict{String, String}(
                "geology_id" => geology_id,
                "scenario_label" => String(geology["scenario_label"]),
                "case_label" => String(geology["case_label"]),
                "level2_status" => "failed",
                "level3_status" => "failed",
                "level2_root" => get(generated, "level2_root", ""),
                "level3_root" => get(generated, "level3_root", ""),
                "message" => message,
            ))
            write_status(status_path, status_rows)
            Bool(cfg["stop_on_error"]) && rethrow(err)
        end
        write_status(status_path, status_rows)
    end

    write_aggregate_outputs(geologies, cfg)

    println()
    println("Completed batch workflow. Status table: $status_path")
end

"""
    write_aggregate_outputs(geologies, cfg)

Collect completed per-geology Level 2 and Level 3 tables into four compact
cross-geology summary files.
"""
function write_aggregate_outputs(geologies::Vector{Dict{String, Any}}, cfg::Dict{String, Any})
    output_root = String(cfg["output_root"])
    summary_root = joinpath(output_root, "summary")
    mkpath(summary_root)

    meta_by_id = Dict(String(g["geology_id"]) => g for g in geologies)
    meta_header = ["geology_id", "scenario_index", "scenario_label", "scenario_name",
                   "case_index", "case_label", "faulting_depth_m", "sand_vcl", "clay_vcl"]

    aggregate_csv(
        joinpath(summary_root, "level2_build_summary_all_geologies.csv"),
        [joinpath(output_root, "level2", String(g["geology_id"]), "tables", "level2_build_summary.csv") for g in geologies],
        meta_by_id,
        meta_header,
    )
    aggregate_csv(
        joinpath(summary_root, "level3_window_similarity_group_summary_all_geologies.csv"),
        [joinpath(output_root, "level3", String(g["geology_id"]), "tables", "window_similarity_group_summary.csv") for g in geologies],
        meta_by_id,
        meta_header,
    )
    aggregate_csv(
        joinpath(summary_root, "level3_multiple_window_permeability_case_definitions_all_geologies.csv"),
        [joinpath(output_root, "level3", String(g["geology_id"]), "tables", "multiple_window_permeability_case_definitions.csv") for g in geologies],
        meta_by_id,
        meta_header,
    )
    aggregate_csv(
        joinpath(summary_root, "level3_multiple_window_permeability_window_assignments_all_geologies.csv"),
        [joinpath(output_root, "level3", String(g["geology_id"]), "tables", "multiple_window_permeability_window_assignments.csv") for g in geologies],
        meta_by_id,
        meta_header,
    )
    aggregate_csv(
        joinpath(summary_root, "level3_multiple_window_permeability_sampled_values_all_geologies.csv"),
        [joinpath(output_root, "level3", String(g["geology_id"]), "tables", "multiple_window_permeability_sampled_values.csv") for g in geologies],
        meta_by_id,
        meta_header,
    )
    aggregate_csv(
        joinpath(summary_root, "level3_multiple_window_permeability_sampled_case_matrix_all_geologies.csv"),
        [joinpath(output_root, "level3", String(g["geology_id"]), "tables", "multiple_window_permeability_sampled_case_matrix.csv") for g in geologies],
        meta_by_id,
        meta_header,
    )
end

"""
    aggregate_csv(output_path, input_paths, meta_by_id, meta_header)

Prepend Level 1 geology metadata to each completed per-geology CSV table and
write one combined summary table. Missing per-geology files are skipped so this
also works after pilot or interrupted/resumed runs.
"""
function aggregate_csv(output_path::AbstractString,
                       input_paths::Vector{String},
                       meta_by_id::Dict{String, Dict{String, Any}},
                       meta_header::Vector{String})
    rows_out = Vector{Vector{String}}()
    table_header = String[]
    for path in input_paths
        isfile(path) || continue
        rows = read_csv_dicts(path)
        isempty(rows) && continue
        if isempty(table_header)
            table_header = [key for key in keys_in_file_order(path) if !(key in meta_header)]
        end
        geology_id = infer_geology_id_from_table_or_path(rows[1], path)
        haskey(meta_by_id, geology_id) || continue
        meta = meta_by_id[geology_id]
        for row in rows
            row_geology_id = infer_geology_id_from_table_or_path(row, path)
            meta_row = meta_by_id[row_geology_id]
            push!(rows_out, [metadata_value(meta_row, key) for key in meta_header])
            append!(rows_out[end], [get(row, key, "") for key in table_header])
        end
    end

    if isempty(table_header)
        safe_write_csv(output_path, ["note"], [["No completed source tables found yet."]])
    else
        safe_write_csv(output_path, vcat(meta_header, table_header), rows_out)
    end
end

"""
    safe_write_csv(path, header, rows)

Write a CSV, falling back to a timestamped sibling file if the target is locked
by Excel or another viewer.
"""
function safe_write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    try
        write_csv(path, header, rows)
    catch err
        if err isa SystemError
            fallback = joinpath(dirname(path),
                                "$(splitext(basename(path))[1])_$(Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))$(splitext(path)[2])")
            @warn "Could not write aggregate CSV, likely because it is open elsewhere. Writing fallback file instead." path fallback
            write_csv(fallback, header, rows)
        else
            rethrow(err)
        end
    end
end

"""
    keys_in_file_order(path)

Return the CSV header order from a file.
"""
function keys_in_file_order(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return String[]
    return parse_csv_line(lines[1])
end

"""
    infer_geology_id_from_table_or_path(row, path)

Infer the geology ID from an explicit `geology_id` column when present, or from
the parent Level 2/Level 3 output folder otherwise.
"""
function infer_geology_id_from_table_or_path(row::Dict{String, String}, path::AbstractString)
    if haskey(row, "geology_id") && !isempty(row["geology_id"])
        return row["geology_id"]
    end
    parts = splitpath(normpath(path))
    for i in eachindex(parts)
        if parts[i] in ("level2", "level3") && i < length(parts)
            return parts[i + 1]
        end
    end
    error("Cannot infer geology_id from $path")
end

"""
    metadata_value(meta, key)

Return a string metadata value for the aggregate summary tables.
"""
function metadata_value(meta::Dict{String, Any}, key::String)
    if key == "faulting_depth_m"
        return string(meta["faulting_depth"])
    end
    return string(meta[key])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
