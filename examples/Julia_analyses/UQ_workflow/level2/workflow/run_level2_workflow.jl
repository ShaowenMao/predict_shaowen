#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

using TOML

include(joinpath(@__DIR__, "..", "lib", "level2_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_ranks.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_distances.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_clustering.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_state_libraries.jl"))
include(joinpath(@__DIR__, "..", "lib", "level2_state_object.jl"))

using .Level2IO
using .Level2Ranks
using .Level2Distances
using .Level2Clustering
using .Level2StateLibraries
using .Level2StateObject

"""
    LEVEL2_WORKFLOW_DOC

Command-line driver for the modular Level 2 UQ workflow.

The driver reads `level2_workflow_config.toml`, applies user-facing overrides
to the baseline Level 2 configuration, and runs the enabled workflow blocks:
input screening, modular Level 2 object construction, table export, figure
generation, and sampling smoke tests.
"""
const LEVEL2_WORKFLOW_DOC = "Documentation marker for the modular Level 2 workflow driver."

include(joinpath(@__DIR__, "steps", "step01_load_window_library.jl"))
include(joinpath(@__DIR__, "steps", "step02_detect_joint_clusters.jl"))
include(joinpath(@__DIR__, "steps", "step03_compute_local_ranks_and_joint_rank_scores.jl"))
include(joinpath(@__DIR__, "steps", "step04_build_low_high_state_libraries.jl"))
include(joinpath(@__DIR__, "steps", "step05_choose_state_medoids.jl"))
include(joinpath(@__DIR__, "steps", "step06_build_perturbation_pools.jl"))
include(joinpath(@__DIR__, "steps", "step07_save_level2_object.jl"))
include(joinpath(@__DIR__, "steps", "step08_generate_level2_outputs.jl"))

"""
    parse_args(args)

Parse command-line arguments for the modular Level 2 workflow driver.
"""
function parse_args(args::Vector{String})
    options = Dict(
        "config" => joinpath(@__DIR__, "level2_workflow_config.toml"),
        "output-root" => "",
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

Print command-line usage information for `run_level2_workflow.jl`.
"""
function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/workflow/run_level2_workflow.jl [options]")
    println()
    println("Options:")
    println("  --config <path>        Modular Level 2 workflow TOML config")
    println("  --output-root <path>   Override [paths].output_root from the workflow config")
    println("  -h, --help             Show this help")
end

"""
    read_workflow_config(path, output_root_override)

Read the modular workflow TOML file and return a normalized workflow
dictionary.

The workflow dictionary contains resolved paths, the merged Level 2 analysis
configuration, and run toggles. Values in the workflow config override the
baseline `configs/level2_defaults.toml` without modifying that baseline file.
"""
function read_workflow_config(path::AbstractString, output_root_override::AbstractString)
    config_path = normpath(path)
    raw = TOML.parsefile(config_path)
    paths = get(raw, "paths", Dict{String, Any}())
    base_config_path = resolve_workflow_path(config_path, String(get(paths, "base_config", "../../configs/level2_defaults.toml")))
    manifest_path = resolve_workflow_path(config_path, String(get(paths, "manifest", "../../configs/level2_proxy_manifest.csv")))

    level2_config = Level2IO.read_level2_config(base_config_path)
    apply_workflow_overrides!(level2_config, raw)

    output_root = isempty(output_root_override) ?
        resolve_workflow_path(config_path, String(get(paths, "output_root", joinpath(Level2IO.default_level2_output_root(), level2_config["geology_id"])))) :
        normpath(output_root_override)

    return Dict{String, Any}(
        "workflow_config_path" => config_path,
        "base_config_path" => base_config_path,
        "manifest_path" => manifest_path,
        "output_root" => output_root,
        "config" => level2_config,
        "run" => read_run_config(raw),
    )
end

"""
    resolve_workflow_path(config_path, path)

Resolve a path from the workflow config relative to the config file location.
Absolute paths are returned unchanged after normalization.
"""
function resolve_workflow_path(config_path::AbstractString, path::AbstractString)
    isempty(path) && return ""
    return isabspath(path) ? normpath(path) : normpath(joinpath(dirname(config_path), path))
end

"""
    read_run_config(raw)

Return the workflow run-toggle dictionary with defaults filled in.
"""
function read_run_config(raw::Dict{String, Any})
    defaults = Dict{String, Bool}(
        "screen_input_data" => true,
        "build_level2_objects" => true,
        "export_tables" => true,
        "plot_joint_clusters" => true,
        "plot_joint_clusters_3d" => false,
        "plot_state_libraries" => true,
        "plot_perturbation_pools" => true,
        "run_sampling_test" => true,
        "run_validation" => false,
        "run_joint_cluster_sensitivity" => false,
        "run_joint_cluster_bootstrap" => false,
    )
    for (key, value) in get(raw, "run", Dict{String, Any}())
        defaults[String(key)] = Bool(value)
    end
    return defaults
end

"""
    apply_workflow_overrides!(config, raw)

Apply user-facing workflow overrides to the loaded Level 2 configuration.

Only the `[level2]`, `[perturbation]`, and `[plotting]` sections are treated as
analysis-parameter overrides.
"""
function apply_workflow_overrides!(config::Dict{String, Any}, raw::Dict{String, Any})
    for (section_name, section) in raw
        section_name in ("level2", "perturbation", "plotting") || continue
        for (key, value) in section
            assign_config_override!(config, String(key), value)
        end
    end
end

"""
    assign_config_override!(config, key, value)

Assign one workflow override to the Level 2 config with the expected type.
"""
function assign_config_override!(config::Dict{String, Any}, key::String, value)
    if key in ("weights", "distance_weights")
        config[key] = Float64.(value)
    elseif key in ("max_k", "min_cluster_size", "random_seed", "max_kmedoids_iter",
                   "num_restarts", "local_pool_min_count")
        config[key] = Int(value)
    elseif key in ("state_fraction", "local_pool_fraction", "silhouette_threshold",
                   "min_cluster_fraction", "state_violin_fixed_count_density_reference",
                   "local_pool_violin_fixed_count_density_reference",
                   "state_wide_pool_violin_fixed_count_density_reference")
        config[key] = Float64(value)
    elseif key == "distance_metric"
        config[key] = String(value)
    else
        config[key] = value
    end
end

"""
    main(args)

Entry point for the modular Level 2 command-line workflow.
"""
function main(args::Vector{String})
    opt = parse_args(args)
    workflow = read_workflow_config(opt["config"], opt["output-root"])
    config = workflow["config"]
    output_root = workflow["output_root"]
    run_cfg = workflow["run"]

    mkpath(output_root)

    if Bool(get(run_cfg, "screen_input_data", false))
        println("Step 00: screening raw marginal histograms")
        step08_generate_input_screening(workflow)
    end

    if Bool(get(run_cfg, "build_level2_objects", true))
        println("Steps 01-07: building modular Level 2 objects")
        build_level2_objects_modular(workflow)
    end

    if Bool(get(run_cfg, "export_tables", true))
        println("Step 08: exporting Level 2 tables")
        step08_export_level2_tables(workflow)
    end

    if Bool(get(run_cfg, "plot_joint_clusters", true)) ||
       Bool(get(run_cfg, "plot_joint_clusters_3d", false)) ||
       Bool(get(run_cfg, "plot_state_libraries", true)) ||
       Bool(get(run_cfg, "plot_perturbation_pools", true)) ||
       Bool(get(run_cfg, "run_joint_cluster_sensitivity", false)) ||
       Bool(get(run_cfg, "run_joint_cluster_bootstrap", false))
        println("Step 08: generating Level 2 figures")
        step08_generate_level2_figures(workflow)
    end

    if Bool(get(run_cfg, "run_sampling_test", true))
        println("Step 08: running Level 2 sampling smoke test")
        step08_run_level2_sampling_test(workflow)
    end

    if Bool(get(run_cfg, "run_validation", false))
        println("Step 08: running Level 2 holdout validation")
        step08_run_level2_validation(workflow)
    end

    println("Completed modular Level 2 workflow at $output_root")
end

"""
    build_level2_objects_modular(workflow)

Build all six Level 2 window-state objects using the explicit Step 01-07
workflow.

This function is the modular replacement for the old single builder script. It
keeps the scientific steps separate while saving the same MAT schema and build
summary tables expected by downstream scripts.
"""
function build_level2_objects_modular(workflow::Dict{String, Any})
    config = workflow["config"]
    manifest_rows = Level2IO.read_manifest_csv(workflow["manifest_path"], config)
    output_root = workflow["output_root"]
    table_root = joinpath(output_root, "tables")
    report_root = joinpath(output_root, "reports")
    mkpath(joinpath(output_root, "window_states"))
    mkpath(table_root)
    mkpath(report_root)

    build_rows = Vector{Vector{String}}()
    report_lines = String[
        "Level 2 modular build report",
        "created_at = $(Level2IO.timestamp_string())",
        "workflow_config_path = $(workflow["workflow_config_path"])",
        "base_config_path = $(workflow["base_config_path"])",
        "manifest_path = $(workflow["manifest_path"])",
        "geology_id = $(config["geology_id"])",
        "distance_metric = $(config["distance_metric"])",
        "distance_weights = $(join(config["distance_weights"], ", "))",
        "min_cluster_fraction = $(config["min_cluster_fraction"])",
        "min_cluster_size = $(config["min_cluster_size"])",
        "output_root = $output_root",
        "windows = $(join(config["fixed_windows"], ", "))",
        "",
    ]

    for row in manifest_rows
        window_data = step01_load_window_library(row)
        window = window_data["window"]
        println("  $window: Step 01 loaded $(window_data["n_samples"]) samples from $(window_data["source_path"])")

        cluster_step = step02_detect_joint_clusters(window_data, config)
        rank_step = step03_compute_local_ranks_and_joint_rank_scores(window_data, config; cluster_step = cluster_step)
        state_libraries = step04_build_low_high_state_libraries(rank_step, cluster_step, config)
        medoids = step05_choose_state_medoids(state_libraries, cluster_step)
        perturbation_pools = step06_build_perturbation_pools(state_libraries, medoids, cluster_step, config)
        state, state_path = step07_save_level2_object(window_data,
                                                      rank_step,
                                                      cluster_step,
                                                      state_libraries,
                                                      medoids,
                                                      perturbation_pools,
                                                      config,
                                                      output_root)

        append_build_summary!(build_rows, report_lines, window, window_data, state, state_path)
    end

    Level2IO.write_csv(joinpath(table_root, "level2_build_summary.csv"),
                       ["window", "source_path", "n_samples", "distance_metric", "distance_component_scales",
                        "min_cluster_fraction", "min_cluster_size",
                        "chosen_k", "best_silhouette",
                        "is_effectively_unimodal", "low_n", "high_n", "state_path"],
                       build_rows)
    Level2IO.write_text_lines(joinpath(report_root, "level2_build_report.txt"), report_lines)
end

"""
    append_build_summary!(build_rows, report_lines, window, window_data, state, state_path)

Append one window's build metadata to the CSV rows and plain-text report lines.
"""
function append_build_summary!(build_rows::Vector{Vector{String}},
                               report_lines::Vector{String},
                               window::AbstractString,
                               window_data::Dict{String, Any},
                               state::Dict{String, Any},
                               state_path::AbstractString)
    scales = Float64.(state["distance_component_scales"])
    push!(build_rows, [
        String(window),
        String(window_data["source_path"]),
        string(state["n_samples"]),
        string(state["distance_metric"]),
        join(string.(round.(scales, digits = 6)), ";"),
        string(state["min_cluster_fraction"]),
        string(state["min_cluster_size"]),
        string(state["chosen_k"]),
        string(round(Float64(state["best_silhouette"]), digits = 6)),
        string(Int(state["is_effectively_unimodal"])),
        string(length(vec(state["low_indices"]))),
        string(length(vec(state["high_indices"]))),
        String(state_path),
    ])

    push!(report_lines, "window = $window")
    push!(report_lines, "  distance_metric = $(state["distance_metric"])")
    push!(report_lines, "  distance_component_scales = $(join(round.(scales, digits = 6), ", "))")
    push!(report_lines, "  min_cluster_fraction = $(state["min_cluster_fraction"])")
    push!(report_lines, "  min_cluster_size = $(state["min_cluster_size"])")
    push!(report_lines, "  chosen_k = $(state["chosen_k"])")
    push!(report_lines, "  best_silhouette = $(round(Float64(state["best_silhouette"]), digits = 6))")
    push!(report_lines, "  unimodal = $(Bool(Int(state["is_effectively_unimodal"])))")
    push!(report_lines, "  low_n = $(length(vec(state["low_indices"])))")
    push!(report_lines, "  high_n = $(length(vec(state["high_indices"])))")
    push!(report_lines, "")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
