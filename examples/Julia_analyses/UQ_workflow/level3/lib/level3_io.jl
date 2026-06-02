"""
    Level3IO

Input/output helpers for Level 3 cross-window similarity analysis.

Level 3 consumes the six Level 2 window-state objects for one geology and
produces distance matrices, reports, and later similarity-group objects.
"""
module Level3IO

using MAT
using TOML
using Dates

export FIXED_WINDOWS,
       WORKFLOW_ROOT,
       LEVEL3_ROOT,
       default_workflow_config_path,
       default_output_root,
       read_level3_config,
       load_level2_states,
       write_csv,
       write_text_lines,
       write_square_matrix_csv,
       timestamp_string

const FIXED_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]
const LEVEL3_ROOT = normpath(joinpath(@__DIR__, ".."))
const WORKFLOW_ROOT = normpath(joinpath(LEVEL3_ROOT, ".."))

"""
    default_workflow_config_path()

Return the default Level 3 workflow TOML configuration path.
"""
default_workflow_config_path() = normpath(joinpath(LEVEL3_ROOT, "workflow", "level3_workflow_config.toml"))

"""
    default_output_root()

Return the default Level 3 output root inside the workflow tree.
"""
default_output_root() = normpath(joinpath(LEVEL3_ROOT, "outputs"))

"""
    timestamp_string()

Return a compact local timestamp string for reports.
"""
timestamp_string() = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

"""
    read_level3_config(path; output_root_override="", level2_state_root_override="")

Read and normalize the Level 3 workflow configuration.

The optional keyword overrides make command-line testing easier without editing
the TOML file.
"""
function read_level3_config(path::AbstractString;
                            output_root_override::AbstractString = "",
                            level2_state_root_override::AbstractString = "")
    config_path = normpath(path)
    raw = TOML.parsefile(config_path)
    workflow = get(raw, "workflow", Dict{String, Any}())
    paths = get(raw, "paths", Dict{String, Any}())
    level3 = get(raw, "level3", Dict{String, Any}())
    grouping = get(level3, "grouping", Dict{String, Any}())
    bootstrap = get(level3, "bootstrap", Dict{String, Any}())
    run_cfg = get(raw, "run", Dict{String, Any}())
    figures = get(raw, "figures", Dict{String, Any}())

    windows = String.(get(workflow, "fixed_windows", FIXED_WINDOWS))
    windows == FIXED_WINDOWS || error("Level 3 currently expects fixed windows $(join(FIXED_WINDOWS, ", "))")

    level2_state_root = isempty(level2_state_root_override) ?
        resolve_config_path(config_path, String(get(paths, "level2_state_root", ""))) :
        normpath(level2_state_root_override)
    isempty(level2_state_root) && error("[paths].level2_state_root is required")

    output_root = isempty(output_root_override) ?
        resolve_config_path(config_path, String(get(paths, "output_root", default_output_root()))) :
        normpath(output_root_override)

    return Dict{String, Any}(
        "config_path" => config_path,
        "geology_id" => String(get(workflow, "geology_id", "g_ref")),
        "fixed_windows" => windows,
        "level2_state_root" => level2_state_root,
        "output_root" => output_root,
        "distance_metric" => String(get(level3, "distance_metric", "energy_log_unit")),
        "normalization" => String(get(level3, "normalization", "mean_internal_spread")),
        "pairwise_sample_size" => Int(get(level3, "pairwise_sample_size", 0)),
        "random_seed" => Int(get(level3, "random_seed", 1729)),
        "grouping_mode" => lowercase(String(get(grouping, "grouping_mode", "full_data"))),
        "grouping_similarity_threshold" => Float64(get(grouping, "similarity_threshold", 0.25)),
        "bootstrap_count" => Int(get(bootstrap, "bootstrap_count", 100)),
        "bootstrap_sample_size" => Int(get(bootstrap, "bootstrap_sample_size", 2000)),
        "bootstrap_similarity_threshold" => Float64(get(bootstrap, "similarity_threshold", 0.25)),
        "stable_pair_probability_threshold" => Float64(get(bootstrap, "stable_pair_probability", 0.80)),
        "bootstrap_random_seed" => Int(get(bootstrap, "random_seed", Int(get(level3, "random_seed", 1729)))),
        "bootstrap_show_progress" => Bool(get(bootstrap, "show_progress", true)),
        "figure_similarity_threshold" => Float64(get(figures, "similarity_threshold", 0.25)),
        "figure_formats" => String.(get(figures, "formats", ["png", "pdf"])),
        "run_load_level2_states" => Bool(get(run_cfg, "load_level2_states", true)),
        "run_build_window_similarity_groups" => Bool(get(run_cfg, "build_window_similarity_groups", true)),
        "run_make_grouping_distance_figures" => Bool(get(run_cfg, "make_grouping_distance_figures", true)),
        "run_bootstrap_grouping_qa" => Bool(get(run_cfg, "run_bootstrap_grouping_qa", false)),
        "run_make_bootstrap_grouping_figures" => Bool(get(run_cfg, "make_bootstrap_grouping_figures", false)),
        "run_make_similarity_group_figure" => Bool(get(run_cfg, "make_similarity_group_figure", true)),
        "run_build_multiple_window_permeability_cases" => Bool(get(run_cfg, "build_multiple_window_permeability_cases", true)),
        "run_make_multiple_window_permeability_case_figure" => Bool(get(run_cfg, "make_multiple_window_permeability_case_figure", true)),
    )
end

"""
    resolve_config_path(config_path, path)

Resolve a config path relative to the config file location, unless it is
already absolute.
"""
function resolve_config_path(config_path::AbstractString, path::AbstractString)
    isempty(path) && return ""
    return isabspath(path) ? normpath(path) : normpath(joinpath(dirname(config_path), path))
end

"""
    load_level2_states(level2_state_root, windows)

Load the six Level 2 window-state MAT files for one geology.

The expected layout is:

```text
<level2_state_root>/window_states/<window>/<window>_level2_state.mat
```
"""
function load_level2_states(level2_state_root::AbstractString, windows::Vector{String})
    states = Dict{String, Dict{String, Any}}()
    for window in windows
        state_path = joinpath(level2_state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing Level 2 state file for $window: $state_path")
        state = matread(state_path)
        haskey(state, "log_perms") || error("Level 2 state is missing log_perms: $state_path")
        size(Matrix{Float64}(state["log_perms"]), 2) == 3 ||
            error("Level 2 log_perms must have three columns in $state_path")
        state["level2_state_path"] = state_path
        states[window] = state
    end
    return states
end

"""
    write_csv(path, header, rows)

Write a CSV file, creating parent directories as needed.
"""
function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            length(row) == length(header) || error("CSV row length does not match header for $path")
            println(io, join(csv_escape.(row), ","))
        end
    end
    return path
end

"""
    write_text_lines(path, lines)

Write plain-text report lines.
"""
function write_text_lines(path::AbstractString, lines::Vector{String})
    mkpath(dirname(path))
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    return path
end

"""
    write_square_matrix_csv(path, labels, matrix)

Write a square matrix with row and column labels to CSV.
"""
function write_square_matrix_csv(path::AbstractString,
                                 labels::Vector{String},
                                 matrix::Matrix{Float64})
    header = ["window"; labels]
    rows = Vector{Vector{String}}()
    for (i, label) in enumerate(labels)
        push!(rows, [label; [float_string(matrix[i, j]) for j in eachindex(labels)]])
    end
    return write_csv(path, header, rows)
end

function csv_escape(value::AbstractString)
    escaped = replace(value, "\"" => "\"\"")
    if occursin(',', escaped) || occursin('"', escaped)
        return "\"" * escaped * "\""
    end
    return escaped
end

float_string(value::Real) = string(round(Float64(value), digits = 8))

end
