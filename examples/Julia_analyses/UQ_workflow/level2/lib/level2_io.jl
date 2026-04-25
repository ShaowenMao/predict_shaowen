module Level2IO

using MAT
using TOML
using Dates

export FIXED_WINDOWS,
       REPO_ROOT,
       WORKFLOW_ROOT,
       default_config_path,
       default_manifest_path,
       default_level2_output_root,
       read_level2_config,
       read_manifest_csv,
       load_proxy_library,
       save_window_state,
       load_window_state,
       write_window_point_table,
       write_csv,
       write_text_lines,
       timestamp_string

const FIXED_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]
const WORKFLOW_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REPO_ROOT = normpath(joinpath(WORKFLOW_ROOT, "..", "..", ".."))

default_config_path() = normpath(joinpath(WORKFLOW_ROOT, "configs", "level2_defaults.toml"))
default_manifest_path() = normpath(joinpath(WORKFLOW_ROOT, "configs", "level2_proxy_manifest.csv"))
default_level2_output_root() = normpath(joinpath(WORKFLOW_ROOT, "level2", "outputs"))

timestamp_string() = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

function resolve_repo_path(path::AbstractString)
    return isabspath(path) ? normpath(path) : normpath(joinpath(REPO_ROOT, path))
end

function read_level2_config(path::AbstractString)
    cfg_raw = TOML.parsefile(path)
    haskey(cfg_raw, "workflow") || error("Missing [workflow] section in $path")
    haskey(cfg_raw, "level2") || error("Missing [level2] section in $path")

    workflow = cfg_raw["workflow"]
    level2 = cfg_raw["level2"]
    validation = get(cfg_raw, "validation", Dict{String, Any}())

    windows = String.(workflow["fixed_windows"])
    windows == FIXED_WINDOWS || error("Config must use the six fixed windows $(join(FIXED_WINDOWS, ", "))")

    weights = Float64.(level2["weights"])
    length(weights) == 3 || error("Expected exactly three Level 2 weights in $path")

    return Dict{String, Any}(
        "config_path" => normpath(path),
        "repo_root" => REPO_ROOT,
        "workflow_root" => WORKFLOW_ROOT,
        "geology_id" => String(workflow["geology_id"]),
        "fixed_windows" => windows,
        "state_fraction" => Float64(level2["state_fraction"]),
        "small_neighbor_fraction" => Float64(level2["small_neighbor_fraction"]),
        "large_neighbor_fraction" => Float64(level2["large_neighbor_fraction"]),
        "weights" => weights,
        "max_k" => Int(level2["max_k"]),
        "silhouette_threshold" => Float64(level2["silhouette_threshold"]),
        "min_cluster_fraction" => Float64(level2["min_cluster_fraction"]),
        "min_cluster_size" => Int(level2["min_cluster_size"]),
        "random_seed" => Int(level2["random_seed"]),
        "max_kmedoids_iter" => Int(level2["max_kmedoids_iter"]),
        "num_restarts" => Int(level2["num_restarts"]),
        "holdout_repeats" => Int.(get(validation, "holdout_repeats", Int[])),
    )
end

function read_manifest_csv(path::AbstractString, cfg::Dict{String, Any})
    lines = readlines(path)
    isempty(lines) && error("Manifest is empty: $path")

    header = split(strip(lines[1]), ",")
    expected = ["geology_id", "window", "sample_kind", "mat_path", "n_samples"]
    header == expected || error("Manifest header must be $(join(expected, ", "))")

    rows = Dict{String, String}[]
    for line in lines[2:end]
        stripped = strip(line)
        isempty(stripped) && continue
        parts = split(stripped, ",")
        length(parts) == length(header) || error("Malformed manifest row in $path: $line")
        row = Dict{String, String}()
        for (key, value) in zip(header, parts)
            row[key] = strip(value)
        end
        row["resolved_mat_path"] = resolve_repo_path(row["mat_path"])
        push!(rows, row)
    end

    isempty(rows) && error("Manifest has no data rows: $path")
    unique(row["geology_id"] for row in rows) == [cfg["geology_id"]] ||
        error("Manifest geology ids do not match config geology id $(cfg["geology_id"])")

    windows = [row["window"] for row in rows]
    sort(windows) == sort(cfg["fixed_windows"]) ||
        error("Manifest must contain exactly one row for each fixed window")

    length(unique(windows)) == length(cfg["fixed_windows"]) ||
        error("Manifest contains duplicated windows")

    row_by_window = Dict(row["window"] => row for row in rows)
    ordered_rows = [row_by_window[window] for window in cfg["fixed_windows"]]
    for row in ordered_rows
        isfile(row["resolved_mat_path"]) || error("Proxy MAT file does not exist: $(row["resolved_mat_path"])")
    end
    return ordered_rows
end

function load_proxy_library(row::Dict{String, String})
    filepath = row["resolved_mat_path"]
    data = matread(filepath)
    haskey(data, "perms") || error("MAT file does not contain perms: $filepath")

    raw_perms = Matrix{Float64}(data["perms"])
    size(raw_perms, 2) == 3 || error("Expected a 3-column perms matrix in $filepath")
    all(raw_perms .> 0) || error("perms must be strictly positive in $filepath")

    return Dict{String, Any}(
        "window" => row["window"],
        "geology_id" => row["geology_id"],
        "sample_kind" => row["sample_kind"],
        "source_path" => filepath,
        "source_label" => "$(row["window"]) $(row["sample_kind"])",
        "raw_perms" => raw_perms,
        "log_perms" => log10.(raw_perms),
        "n_samples" => size(raw_perms, 1),
    )
end

function save_window_state(path::AbstractString, state::Dict{String, Any})
    mkpath(dirname(path))
    matwrite(path, state)
    return path
end

load_window_state(path::AbstractString) = matread(path)

function write_window_point_table(path::AbstractString, state::Dict{String, Any})
    mkpath(dirname(path))

    log_perms = Matrix{Float64}(state["log_perms"])
    local_ranks = Matrix{Float64}(state["local_ranks"])
    local_normal_scores = Matrix{Float64}(state["local_normal_scores"])
    state_score = vec(Float64.(state["state_score"]))
    cluster_assignments = vec(Int.(state["cluster_assignments"]))
    cluster_order = vec(Int.(state["cluster_order"]))

    n = size(log_perms, 1)
    size(local_ranks, 1) == n || error("local_ranks row count does not match log_perms")
    size(local_normal_scores, 1) == n || error("local_normal_scores row count does not match log_perms")

    cluster_rank_map = Dict(cluster_id => rank for (rank, cluster_id) in enumerate(cluster_order))
    low_set = Set(vec(Int.(state["low_indices"])))
    central_set = Set(vec(Int.(state["central_indices"])))
    high_set = Set(vec(Int.(state["high_indices"])))

    header = [
        "sample_index",
        "log_kxx", "log_kyy", "log_kzz",
        "local_rank_kxx", "local_rank_kyy", "local_rank_kzz",
        "local_normal_score_kxx", "local_normal_score_kyy", "local_normal_score_kzz",
        "state_score",
        "cluster_id",
        "cluster_rank",
        "is_low_state",
        "is_central_state",
        "is_high_state",
    ]

    rows = Vector{Vector{String}}(undef, n)
    for i in 1:n
        rows[i] = [
            string(i),
            float_string(log_perms[i, 1]),
            float_string(log_perms[i, 2]),
            float_string(log_perms[i, 3]),
            float_string(local_ranks[i, 1]),
            float_string(local_ranks[i, 2]),
            float_string(local_ranks[i, 3]),
            float_string(local_normal_scores[i, 1]),
            float_string(local_normal_scores[i, 2]),
            float_string(local_normal_scores[i, 3]),
            float_string(state_score[i]),
            string(cluster_assignments[i]),
            string(cluster_rank_map[cluster_assignments[i]]),
            i in low_set ? "1" : "0",
            i in central_set ? "1" : "0",
            i in high_set ? "1" : "0",
        ]
    end

    write_csv(path, header, rows)
    return path
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{String}})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            length(row) == length(header) || error("CSV row length does not match header length for $path")
            println(io, join(csv_escape.(row), ","))
        end
    end
    return path
end

function write_text_lines(path::AbstractString, lines::Vector{String})
    mkpath(dirname(path))
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    return path
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
