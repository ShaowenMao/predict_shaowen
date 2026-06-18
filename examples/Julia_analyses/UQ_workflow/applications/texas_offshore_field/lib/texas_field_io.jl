"""
    TexasFieldIO

Input/output helpers for the Texas offshore field sampling application.

This module is intentionally application-specific. It reads the completed
three-level workflow summaries and the field-application TOML config, but it
does not alter the core Level 2 or Level 3 workflow logic.
"""
module TexasFieldIO

using TOML
using Dates

export read_texas_config,
       read_csv_dicts,
       select_geology_ids,
       filter_case_rows,
       timestamp_string,
       csv_escape,
       write_csv_row

const DEFAULT_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]

"""
    read_texas_config(path; output_root_override="", max_geologies_override="", only_geology_override="", case_ids_override="")

Read and normalize the Texas field sampling configuration.
"""
function read_texas_config(path::AbstractString;
                           output_root_override::AbstractString = "",
                           max_geologies_override::AbstractString = "",
                           only_geology_override::AbstractString = "",
                           case_ids_override::AbstractString = "",
                           list_only_override::AbstractString = "")
    config_path = normpath(path)
    raw = TOML.parsefile(config_path)
    paths = get(raw, "paths", Dict{String, Any}())
    workflow = get(raw, "workflow", Dict{String, Any}())
    field = get(raw, "field", Dict{String, Any}())
    sampling = get(raw, "sampling", Dict{String, Any}())
    outputs = get(raw, "outputs", Dict{String, Any}())
    run = get(raw, "run", Dict{String, Any}())

    summary_root = resolve_config_path(config_path, String(get(paths, "summary_root", "")))
    level2_root = resolve_config_path(config_path, String(get(paths, "level2_root", "")))
    output_root = isempty(output_root_override) ?
        resolve_config_path(config_path, String(get(paths, "output_root", "texas_field_outputs"))) :
        normpath(output_root_override)
    isempty(summary_root) && error("[paths].summary_root is required")
    isempty(level2_root) && error("[paths].level2_root is required")

    case_ids_config = get(run, "case_ids", Any[])
    case_ids = isempty(case_ids_override) ?
        Int.(case_ids_config) :
        parse_case_ids(case_ids_override)

    max_geologies = isempty(max_geologies_override) ?
        Int(get(run, "max_geologies", 0)) :
        parse(Int, max_geologies_override)
    only_geology = isempty(only_geology_override) ?
        String(get(run, "only_geology", "")) :
        only_geology_override
    list_only = isempty(list_only_override) ?
        Bool(get(run, "list_only", false)) :
        parse_bool(list_only_override)

    return Dict{String, Any}(
        "config_path" => config_path,
        "summary_root" => summary_root,
        "level2_root" => level2_root,
        "output_root" => output_root,
        "assignment_table" => joinpath(summary_root, "level3_multiple_window_permeability_window_assignments_all_geologies.csv"),
        "windows" => String.(get(workflow, "fixed_windows", DEFAULT_WINDOWS)),
        "num_slices" => Int(get(field, "num_slices", 87)),
        "shared_slice_groups" => [Int.(group) for group in get(field, "shared_slice_groups", Any[])],
        "random_seed" => Int(get(sampling, "random_seed", 20260617)),
        "export_mat" => Bool(get(outputs, "export_mat", true)),
        "validate_mat" => Bool(get(outputs, "validate_mat", true)),
        "max_geologies" => max_geologies,
        "only_geology" => only_geology,
        "case_ids" => case_ids,
        "list_only" => list_only,
        "overwrite" => Bool(get(run, "overwrite", true)),
    )
end

"""
    resolve_config_path(config_path, path)

Resolve a path relative to the TOML config location unless it is absolute.
"""
function resolve_config_path(config_path::AbstractString, path::AbstractString)
    isempty(path) && return ""
    return isabspath(path) ? normpath(path) : normpath(joinpath(dirname(config_path), path))
end

"""
    parse_case_ids(text)

Parse a comma-separated case-id list such as `"1,2,7"`.
"""
function parse_case_ids(text::AbstractString)
    stripped = strip(text)
    isempty(stripped) && return Int[]
    return [parse(Int, strip(part)) for part in split(stripped, ",") if !isempty(strip(part))]
end

"""
    parse_bool(text)

Parse common command-line boolean strings.
"""
function parse_bool(text::AbstractString)
    value = lowercase(strip(text))
    value in ("true", "t", "yes", "y", "1") && return true
    value in ("false", "f", "no", "n", "0") && return false
    error("Cannot parse boolean value: $text")
end

"""
    read_csv_dicts(path)

Read a CSV file into a vector of string dictionaries.
"""
function read_csv_dicts(path::AbstractString)
    isfile(path) || error("Missing CSV file: $path")
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

"""
    parse_csv_line(line)

Parse one CSV line while respecting quoted fields.
"""
function parse_csv_line(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            next_i = nextind(line, i)
            if in_quotes && next_i <= lastindex(line) && line[next_i] == '"'
                write(buffer, '"')
                i = nextind(line, next_i)
                continue
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(buffer)))
        else
            write(buffer, c)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buffer)))
    return fields
end

"""
    select_geology_ids(rows, config)

Return ordered geology IDs after applying optional config filters.
"""
function select_geology_ids(rows::Vector{Dict{String, String}}, config::Dict{String, Any})
    ids = sort(collect(Set(row["geology_id"] for row in rows)); by = geology_sort_key)
    only = strip(String(config["only_geology"]))
    if !isempty(only)
        ids = [id for id in ids if occursin(only, id)]
    end
    max_geologies = Int(config["max_geologies"])
    if max_geologies > 0
        ids = ids[1:min(max_geologies, length(ids))]
    end
    return ids
end

"""
    geology_sort_key(geology_id)

Sort IDs like `s03_c012` by scenario then case index.
"""
function geology_sort_key(geology_id::AbstractString)
    m = match(r"^s(\d+)_c(\d+)$", geology_id)
    m === nothing && return (typemax(Int), typemax(Int), String(geology_id))
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]), String(geology_id))
end

"""
    filter_case_rows(rows, case_ids)

Keep all rows if `case_ids` is empty; otherwise keep only the requested
Level 3 case IDs.
"""
function filter_case_rows(rows::Vector{Dict{String, String}}, case_ids::Vector{Int})
    isempty(case_ids) && return rows
    wanted = Set(string.(case_ids))
    return [row for row in rows if row["case_id"] in wanted]
end

"""
    timestamp_string()

Return a compact local timestamp for reports.
"""
timestamp_string() = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

"""
    write_csv_row(io, values)

Write one CSV row with escaping.
"""
function write_csv_row(io::IO, values)
    println(io, join(csv_escape.(string.(values)), ","))
end

"""
    csv_escape(value)

Escape one CSV field.
"""
function csv_escape(value::AbstractString)
    escaped = replace(value, "\"" => "\"\"")
    if occursin(',', escaped) || occursin('"', escaped) || occursin('\n', escaped)
        return "\"" * escaped * "\""
    end
    return escaped
end

end
