"""
    TexasFieldOutputs

Streaming CSV output helpers for Texas offshore field sampling.
"""
module TexasFieldOutputs

export UNIQUE_DRAW_HEADER,
       SLICE_WINDOW_HEADER,
       SLICE_CASE_MATRIX_HEADER,
       write_unique_draw_rows!,
       write_slice_window_rows!,
       write_slice_case_matrix_rows!,
       write_slice_draw_groups,
       write_report,
       metric_string

const WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]

const BASE_HEADER = [
    "geology_id",
    "scenario_index",
    "scenario_label",
    "scenario_name",
    "case_index",
    "case_label",
    "faulting_depth_m",
    "sand_vcl",
    "clay_vcl",
    "case_id",
    "case_name",
    "case_category",
    "case_strength",
    "pattern_name",
    "orientation",
    "group_split_id",
]

const WINDOW_HEADER = [
    "window",
    "similarity_group",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "selected_sample_index",
    "source_pool",
    "source_pool_size",
    "draw_seed",
    "sampler_random_seed",
    "source_checkpoint_file",
    "source_seed_base",
    "source_num_attempts",
    "source_num_rejected",
    "exact_replay_seed",
    "fine_scale_replay_status",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx",
    "perm_kyy",
    "perm_kzz",
]

const UNIQUE_DRAW_HEADER = [BASE_HEADER; ["draw_group_index", "draw_group_slices", "is_shared_draw_group"]; WINDOW_HEADER]
const SLICE_WINDOW_HEADER = [BASE_HEADER; ["slice_index", "draw_group_index", "draw_group_slices", "is_shared_draw_group"]; WINDOW_HEADER]

const WINDOW_FIELDS = [
    "selected_sample_index",
    "assigned_state",
    "sampling_mode",
    "sampling_pool",
    "draw_seed",
    "source_pool",
    "source_pool_size",
    "source_checkpoint_file",
    "source_seed_base",
    "source_num_attempts",
    "source_num_rejected",
    "exact_replay_seed",
    "fine_scale_replay_status",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx",
    "perm_kyy",
    "perm_kzz",
]

const SLICE_CASE_MATRIX_HEADER = [BASE_HEADER; ["slice_index", "draw_group_index", "draw_group_slices", "is_shared_draw_group"];
                                  vec(["$(window)_$(field)" for window in WINDOWS for field in WINDOW_FIELDS])]

"""
    write_unique_draw_rows!(io, selected_rows, draw_group_index, slices, draw_seed, sampler_seed)

Write one row per window before expanding shared slices.
"""
function write_unique_draw_rows!(io::IO,
                                 selected_rows::Vector{Dict{String, Any}},
                                 draw_group_index::Integer,
                                 slices::Vector{Int},
                                 draw_seed::Integer,
                                 sampler_seed::Integer)
    for row in selected_rows
        write_csv_row(io, row_values(row, [
            "draw_group_index" => draw_group_index,
            "draw_group_slices" => join(slices, ";"),
            "is_shared_draw_group" => length(slices) > 1,
            "draw_seed" => draw_seed,
            "sampler_random_seed" => sampler_seed,
        ], UNIQUE_DRAW_HEADER))
    end
end

"""
    write_slice_window_rows!(io, selected_rows, draw_group_index, slices, draw_seed, sampler_seed)

Write one row per slice/window after expanding shared slice groups.
"""
function write_slice_window_rows!(io::IO,
                                  selected_rows::Vector{Dict{String, Any}},
                                  draw_group_index::Integer,
                                  slices::Vector{Int},
                                  draw_seed::Integer,
                                  sampler_seed::Integer)
    for slice in slices, row in selected_rows
        write_csv_row(io, row_values(row, [
            "slice_index" => slice,
            "draw_group_index" => draw_group_index,
            "draw_group_slices" => join(slices, ";"),
            "is_shared_draw_group" => length(slices) > 1,
            "draw_seed" => draw_seed,
            "sampler_random_seed" => sampler_seed,
        ], SLICE_WINDOW_HEADER))
    end
end

"""
    write_slice_case_matrix_rows!(io, selected_rows, draw_group_index, slices, draw_seed, sampler_seed)

Write one wide row per slice with all six window vectors.
"""
function write_slice_case_matrix_rows!(io::IO,
                                       selected_rows::Vector{Dict{String, Any}},
                                       draw_group_index::Integer,
                                       slices::Vector{Int},
                                       draw_seed::Integer,
                                       sampler_seed::Integer)
    by_window = Dict(string(row["window"]) => row for row in selected_rows)
    first_row = first(selected_rows)
    for slice in slices
        values = Any[]
        for key in BASE_HEADER
            push!(values, first_row[key])
        end
        append!(values, [slice, draw_group_index, join(slices, ";"), length(slices) > 1])
        for window in WINDOWS
            haskey(by_window, window) || error("Missing sampled window $window")
            row = by_window[window]
            for field in WINDOW_FIELDS
                if field == "draw_seed"
                    push!(values, draw_seed)
                else
                    push!(values, get(row, field, ""))
                end
            end
        end
        write_csv_row(io, values)
    end
end

"""
    row_values(row, extra_pairs, header)

Return values following a requested CSV header.
"""
function row_values(row::Dict{String, Any}, extra_pairs, header::Vector{String})
    extra = Dict{String, Any}(string(key) => value for (key, value) in extra_pairs)
    values = Any[]
    for key in header
        if haskey(extra, key)
            push!(values, extra[key])
        elseif haskey(row, key)
            push!(values, row[key])
        else
            push!(values, "")
        end
    end
    return values
end

"""
    write_slice_draw_groups(path, rows)

Write the slice-to-draw-group table.
"""
function write_slice_draw_groups(path::AbstractString, rows::Vector{Vector{String}})
    mkpath(dirname(path))
    open(path, "w") do io
        write_csv_row(io, ["slice_index", "draw_group_index", "is_shared_draw_group", "draw_group_slices"])
        for row in rows
            write_csv_row(io, row)
        end
    end
end

"""
    write_report(path, lines)

Write a compact plain-text report.
"""
function write_report(path::AbstractString, lines::Vector{String})
    mkpath(dirname(path))
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function write_csv_row(io::IO, values)
    println(io, join(csv_escape.(format_value.(values)), ","))
end

function format_value(value)
    value isa AbstractFloat && return metric_string(value)
    return string(value)
end

metric_string(value::Real) = string(round(Float64(value), digits = 8))

function csv_escape(value::AbstractString)
    escaped = replace(value, "\"" => "\"\"")
    if occursin(',', escaped) || occursin('"', escaped) || occursin('\n', escaped)
        return "\"" * escaped * "\""
    end
    return escaped
end

end
