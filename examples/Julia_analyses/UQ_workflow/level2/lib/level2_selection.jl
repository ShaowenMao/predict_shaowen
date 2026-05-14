"""
    Level2Selection

Selection utilities for converting Level 2 state objects into actual sampled
window realizations.

The module supports two sampling modes: state-conditioned sampling from
low/high local or state-wide pools, and independent sampling from the full
window library. Every selected result is one actual joint PREDICT realization.
"""
module Level2Selection

using Random

export SELECTION_DESIGN_HEADER,
       SELECTION_RESULT_HEADER,
       default_selection_design,
       read_selection_design_csv,
       design_rows_to_csv_rows,
       select_window_samples,
       selection_results_to_csv_rows

const SELECTION_DESIGN_HEADER = [
    "case_id",
    "window",
    "mode",
    "state_label",
    "perturbation_pool",
]

const SELECTION_RESULT_HEADER = [
    "case_id",
    "draw_order",
    "window",
    "mode",
    "state_label",
    "perturbation_pool",
    "selected_sample_index",
    "source_pool",
    "source_pool_size",
    "random_seed",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx",
    "perm_kyy",
    "perm_kzz",
]

"""
    default_selection_design(windows)

Create a small smoke-test design that samples low/high local, low/high
state-wide, and independent cases for every fixed window.
"""
function default_selection_design(windows::Vector{String})
    cases = [
        ("low_local", "state", "low", "local"),
        ("low_state_wide", "state", "low", "state_wide"),
        ("high_local", "state", "high", "local"),
        ("high_state_wide", "state", "high", "state_wide"),
        ("independent_reference", "independent", "", ""),
    ]

    rows = Dict{String, String}[]
    for (case_id, mode, state_label, perturbation_pool) in cases
        for window in windows
            push!(rows, Dict(
                "case_id" => case_id,
                "window" => window,
                "mode" => mode,
                "state_label" => state_label,
                "perturbation_pool" => perturbation_pool,
            ))
        end
    end
    return rows
end

"""
    read_selection_design_csv(path)

Read a user-provided Level 2 selection design CSV.

Required columns are `case_id`, `window`, `mode`, `state_label`, and
`perturbation_pool`.
"""
function read_selection_design_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Selection design CSV is empty: $path")
    header = split_csv_line(lines[1])

    missing = setdiff(SELECTION_DESIGN_HEADER, header)
    isempty(missing) || error("Selection design CSV is missing columns: $(join(missing, ", "))")

    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split_csv_line(line)
        length(values) == length(header) ||
            error("Malformed selection design row in $path: $line")

        row = Dict{String, String}()
        for (key, value) in zip(header, values)
            row[key] = strip(value)
        end
        push!(rows, row)
    end
    isempty(rows) && error("Selection design CSV has no data rows: $path")
    return rows
end

"""
    design_rows_to_csv_rows(rows)

Convert selection-design dictionaries to CSV rows using the standard header.
"""
function design_rows_to_csv_rows(rows::Vector{Dict{String, String}})
    return [[get(row, key, "") for key in SELECTION_DESIGN_HEADER] for row in rows]
end

"""
    select_window_samples(states, design_rows; random_seed)

Execute a Level 2 selection design against loaded window-state objects.

The returned rows include selected sample indices, log permeability values, raw
permeability values, and source-pool metadata.
"""
function select_window_samples(states::Dict{String, Dict{String, Any}},
                               design_rows::Vector{Dict{String, String}};
                               random_seed::Integer)
    rng = MersenneTwister(random_seed)
    results = Dict{String, Any}[]

    for (draw_order, request) in enumerate(design_rows)
        window = required_field(request, "window")
        haskey(states, window) || error("Selection design requested unknown window: $window")
        state = states[window]
        result = select_one_window_sample(state, request, rng, draw_order, random_seed)
        push!(results, result)
    end

    return results
end

"""
    select_one_window_sample(state, request, rng, draw_order, random_seed)

Draw one actual realization from the pool requested by a single design row.
"""
function select_one_window_sample(state::Dict{String, Any},
                                  request::Dict{String, String},
                                  rng::AbstractRNG,
                                  draw_order::Int,
                                  random_seed::Integer)
    log_perms = matrix_float(state["log_perms"])
    raw_perms = matrix_float(state["raw_perms"])
    mode = normalize_mode(required_field(request, "mode"))
    state_label = normalize_optional(get(request, "state_label", ""))
    perturbation_pool = normalize_optional(get(request, "perturbation_pool", ""))

    pool_indices, source_pool = selection_pool_indices(state, mode, state_label, perturbation_pool)
    isempty(pool_indices) && error("Selection pool is empty for request: $(request)")
    selected_sample_index = pool_indices[rand(rng, eachindex(pool_indices))]

    return Dict{String, Any}(
        "case_id" => required_field(request, "case_id"),
        "draw_order" => draw_order,
        "window" => String(state["window"]),
        "mode" => mode,
        "state_label" => state_label,
        "perturbation_pool" => perturbation_pool,
        "selected_sample_index" => selected_sample_index,
        "source_pool" => source_pool,
        "source_pool_size" => length(pool_indices),
        "random_seed" => Int(random_seed),
        "log_kxx" => log_perms[selected_sample_index, 1],
        "log_kyy" => log_perms[selected_sample_index, 2],
        "log_kzz" => log_perms[selected_sample_index, 3],
        "perm_kxx" => raw_perms[selected_sample_index, 1],
        "perm_kyy" => raw_perms[selected_sample_index, 2],
        "perm_kzz" => raw_perms[selected_sample_index, 3],
    )
end

"""
    selection_pool_indices(state, mode, state_label, perturbation_pool)

Return the sample indices and source-pool name for one sampling request.
"""
function selection_pool_indices(state::Dict{String, Any},
                                mode::AbstractString,
                                state_label::AbstractString,
                                perturbation_pool::AbstractString)
    if mode == "independent"
        n = size(matrix_float(state["log_perms"]), 1)
        return collect(1:n), "full_window_library"
    end

    mode == "state" || error("Unknown selection mode: $mode")
    state_label in ("low", "high") ||
        error("State-conditioned selection supports only low/high states, got: $state_label")
    perturbation_pool in ("local", "state_wide") ||
        error("State-conditioned selection requires local/state_wide perturbation_pool, got: $perturbation_pool")

    key = "$(state_label)_$(perturbation_pool)_pool"
    haskey(state, key) || error("Window state is missing selection pool: $key")
    return vector_int(state[key]), "$(state_label)_$(perturbation_pool)_pool"
end

"""
    selection_results_to_csv_rows(results)

Convert selection result dictionaries to rows using `SELECTION_RESULT_HEADER`.
"""
function selection_results_to_csv_rows(results::Vector{Dict{String, Any}})
    return [[format_result_value(result[key]) for key in SELECTION_RESULT_HEADER] for result in results]
end

"""
    normalize_mode(mode)

Normalize user-facing aliases to `state` or `independent`.
"""
function normalize_mode(mode::AbstractString)
    value = lowercase(strip(mode))
    if value in ("state", "state_conditioned", "conditioned")
        return "state"
    elseif value in ("independent", "ind", "full")
        return "independent"
    end
    error("Unknown selection mode: $mode")
end

"""
    normalize_optional(value)

Normalize blank-like optional CSV fields to the empty string.
"""
function normalize_optional(value::AbstractString)
    stripped = lowercase(strip(value))
    return stripped in ("", ".", "na", "none") ? "" : stripped
end

"""
    required_field(row, key)

Read a required CSV field and throw a clear error if it is missing.
"""
function required_field(row::Dict{String, String}, key::AbstractString)
    value = strip(get(row, key, ""))
    isempty(value) && error("Selection design row is missing required field: $key")
    return value
end

"""
    split_csv_line(line)

Split a CSV row while respecting quoted fields.
"""
function split_csv_line(line::AbstractString)
    values = String[]
    buffer = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if in_quotes && i < lastindex(line) && line[nextind(line, i)] == '"'
                print(buffer, '"')
                i = nextind(line, i)
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(values, String(take!(buffer)))
        else
            print(buffer, c)
        end
        i = nextind(line, i)
    end
    push!(values, String(take!(buffer)))
    return values
end

"""
    format_result_value(value)

Format one value for the selection-results CSV.
"""
function format_result_value(value)
    if value isa AbstractFloat
        return string(round(Float64(value), digits = 8))
    end
    return string(value)
end

"""
    vector_int(values)

Convert scalar or array-like values loaded from MAT files to an integer vector.
"""
vector_int(values) = values isa AbstractArray ? vec(Int.(values)) : [Int(values)]

"""
    matrix_float(values)

Convert array-like values loaded from MAT files to a `Matrix{Float64}`.
"""
matrix_float(values) = Matrix{Float64}(values)

end
