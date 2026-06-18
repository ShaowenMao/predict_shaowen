"""
    TexasFieldMatExport

Export Texas offshore field sampling CSV outputs to a compact MATLAB `.mat`
file for MRST workflows.

The MAT export is intentionally generated from the CSV outputs. This keeps the
human-readable audit tables and the MATLAB-ready arrays on the same data path,
which makes consistency checks straightforward.
"""
module TexasFieldMatExport

using MAT

export export_texas_field_mat,
       validate_mat_against_csv

const WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]
const COMPONENTS = ["kxx", "kyy", "kzz"]

"""
    export_texas_field_mat(output_root; mat_path="")

Read the Texas field sampling CSV outputs under `output_root` and write a
compact MATLAB file.

The main arrays have dimensions:

```text
geology x level3_case x slice x window x component
```
"""
function export_texas_field_mat(output_root::AbstractString; mat_path::AbstractString = "")
    root = normpath(output_root)
    slice_csv = joinpath(root, "texas_field_slice_window_values.csv")
    draw_group_csv = joinpath(root, "texas_field_slice_draw_groups.csv")
    isfile(slice_csv) || error("Missing slice-window CSV: $slice_csv")
    isfile(draw_group_csv) || error("Missing slice draw-group CSV: $draw_group_csv")
    out_path = isempty(mat_path) ? joinpath(root, "texas_field_sampling_compact.mat") : normpath(mat_path)

    dims = collect_dimensions(slice_csv)
    arrays = allocate_arrays(dims)
    fill_arrays_from_slice_csv!(arrays, dims, slice_csv)
    draw_map = read_slice_draw_groups(draw_group_csv)

    field_perm = Dict{String, Any}(
        "schema_version" => "texas_field_sampling_compact_v1",
        "dimension_order" => "geology x level3_case x slice x window x component",
        "geology_id" => dims.geology_ids,
        "level3_case_id" => dims.case_ids,
        "slice_index" => dims.slice_indices,
        "window" => WINDOWS,
        "component" => COMPONENTS,
        "scenario_index" => arrays.scenario_index,
        "scenario_label" => arrays.scenario_label,
        "scenario_name" => arrays.scenario_name,
        "geologic_case_index" => arrays.geologic_case_index,
        "geologic_case_label" => arrays.geologic_case_label,
        "faulting_depth_m" => arrays.faulting_depth_m,
        "sand_vcl" => arrays.sand_vcl,
        "clay_vcl" => arrays.clay_vcl,
        "level3_case_name" => arrays.level3_case_name,
        "level3_case_category" => arrays.level3_case_category,
        "level3_case_strength" => arrays.level3_case_strength,
        "pattern_name" => arrays.pattern_name,
        "orientation" => arrays.orientation,
        "group_split_id" => arrays.group_split_id,
        "assigned_state" => arrays.assigned_state,
        "sampling_mode" => arrays.sampling_mode,
        "sampling_pool" => arrays.sampling_pool,
        "slice_draw_group_index" => draw_map["slice_draw_group_index"],
        "slice_is_shared_draw_group" => draw_map["slice_is_shared_draw_group"],
        "slice_draw_group_slices" => draw_map["slice_draw_group_slices"],
        "draw_group_index" => arrays.draw_group_index,
        "draw_seed" => arrays.draw_seed,
        "sampler_random_seed" => arrays.sampler_random_seed,
        "logK" => arrays.logK,
        "perm" => arrays.perm,
        "selected_sample_index" => arrays.selected_sample_index,
        "source_pool_size" => arrays.source_pool_size,
        "source_seed_base" => arrays.source_seed_base,
        "source_num_attempts" => arrays.source_num_attempts,
        "source_num_rejected" => arrays.source_num_rejected,
        "exact_replay_seed" => arrays.exact_replay_seed,
        "fine_scale_replay_status_code" => arrays.fine_scale_replay_status_code,
        "fine_scale_replay_status_label" => ["missing", "direct_exact_replay_possible_if_PREDICT_code_unchanged", "requires_valid_attempt_replay_due_to_rejected_realizations"],
        "source_checkpoint_file" => arrays.source_checkpoint_file,
        "created_from_csv" => slice_csv,
    )

    matwrite(out_path, Dict("fieldPerm" => field_perm))
    return out_path
end

Base.@kwdef mutable struct DimensionInfo
    geology_ids::Vector{String}
    case_ids::Vector{Int}
    slice_indices::Vector{Int}
    geology_index::Dict{String, Int}
    case_index::Dict{Int, Int}
    slice_index::Dict{Int, Int}
    window_index::Dict{String, Int}
end

Base.@kwdef mutable struct FieldArrays
    logK::Array{Float64, 5}
    perm::Array{Float64, 5}
    selected_sample_index::Array{Int64, 4}
    source_pool_size::Array{Int64, 4}
    draw_group_index::Array{Int64, 3}
    draw_seed::Array{Int64, 3}
    sampler_random_seed::Int64
    source_seed_base::Matrix{Int64}
    source_num_attempts::Matrix{Int64}
    source_num_rejected::Matrix{Int64}
    exact_replay_seed::Array{Int64, 4}
    fine_scale_replay_status_code::Array{Int64, 4}
    source_checkpoint_file::Matrix{Any}
    scenario_index::Vector{Int64}
    scenario_label::Vector{Any}
    scenario_name::Vector{Any}
    geologic_case_index::Vector{Int64}
    geologic_case_label::Vector{Any}
    faulting_depth_m::Vector{Float64}
    sand_vcl::Vector{Float64}
    clay_vcl::Vector{Float64}
    level3_case_name::Matrix{Any}
    level3_case_category::Matrix{Any}
    level3_case_strength::Matrix{Any}
    pattern_name::Matrix{Any}
    orientation::Matrix{Any}
    group_split_id::Matrix{Int64}
    assigned_state::Array{Any, 3}
    sampling_mode::Array{Any, 3}
    sampling_pool::Array{Any, 3}
end

"""
    collect_dimensions(slice_csv)

First pass over the slice-window CSV to collect compact array dimensions.
"""
function collect_dimensions(slice_csv::AbstractString)
    geology_ids = String[]
    case_ids = Int[]
    slice_indices = Int[]
    geology_seen = Set{String}()
    case_seen = Set{Int}()
    slice_seen = Set{Int}()

    open(slice_csv, "r") do io
        header = parse_csv_line(readline(io))
        idx = header_index(header)
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = parse_csv_line(line)
            geology_id = values[idx["geology_id"]]
            case_id = parse(Int, values[idx["case_id"]])
            slice_index = parse(Int, values[idx["slice_index"]])

            if !(geology_id in geology_seen)
                push!(geology_ids, geology_id)
                push!(geology_seen, geology_id)
            end
            if !(case_id in case_seen)
                push!(case_ids, case_id)
                push!(case_seen, case_id)
            end
            if !(slice_index in slice_seen)
                push!(slice_indices, slice_index)
                push!(slice_seen, slice_index)
            end
        end
    end

    sort!(case_ids)
    sort!(slice_indices)
    return DimensionInfo(
        geology_ids = geology_ids,
        case_ids = case_ids,
        slice_indices = slice_indices,
        geology_index = Dict(id => i for (i, id) in enumerate(geology_ids)),
        case_index = Dict(id => i for (i, id) in enumerate(case_ids)),
        slice_index = Dict(id => i for (i, id) in enumerate(slice_indices)),
        window_index = Dict(window => i for (i, window) in enumerate(WINDOWS)),
    )
end

"""
    allocate_arrays(dims)

Allocate compact numeric and metadata arrays.
"""
function allocate_arrays(dims::DimensionInfo)
    nG = length(dims.geology_ids)
    nC = length(dims.case_ids)
    nS = length(dims.slice_indices)
    nW = length(WINDOWS)
    nK = length(COMPONENTS)

    return FieldArrays(
        logK = fill(NaN, nG, nC, nS, nW, nK),
        perm = fill(NaN, nG, nC, nS, nW, nK),
        selected_sample_index = fill(Int64(-1), nG, nC, nS, nW),
        source_pool_size = fill(Int64(-1), nG, nC, nS, nW),
        draw_group_index = fill(Int64(-1), nG, nC, nS),
        draw_seed = fill(Int64(-1), nG, nC, nS),
        sampler_random_seed = Int64(-1),
        source_seed_base = fill(Int64(-1), nG, nW),
        source_num_attempts = fill(Int64(-1), nG, nW),
        source_num_rejected = fill(Int64(-1), nG, nW),
        exact_replay_seed = fill(Int64(-1), nG, nC, nS, nW),
        fine_scale_replay_status_code = fill(Int64(0), nG, nC, nS, nW),
        source_checkpoint_file = empty_any_array(nG, nW),
        scenario_index = fill(Int64(-1), nG),
        scenario_label = empty_any_array(nG),
        scenario_name = empty_any_array(nG),
        geologic_case_index = fill(Int64(-1), nG),
        geologic_case_label = empty_any_array(nG),
        faulting_depth_m = fill(NaN, nG),
        sand_vcl = fill(NaN, nG),
        clay_vcl = fill(NaN, nG),
        level3_case_name = empty_any_array(nG, nC),
        level3_case_category = empty_any_array(nG, nC),
        level3_case_strength = empty_any_array(nG, nC),
        pattern_name = empty_any_array(nG, nC),
        orientation = empty_any_array(nG, nC),
        group_split_id = fill(Int64(-1), nG, nC),
        assigned_state = empty_any_array(nG, nC, nW),
        sampling_mode = empty_any_array(nG, nC, nW),
        sampling_pool = empty_any_array(nG, nC, nW),
    )
end

function empty_any_array(dims::Integer...)
    values = Array{Any}(undef, dims...)
    fill!(values, "")
    return values
end

"""
    fill_arrays_from_slice_csv!(arrays, dims, slice_csv)

Second pass over the slice-window CSV to fill the compact arrays.
"""
function fill_arrays_from_slice_csv!(arrays::FieldArrays, dims::DimensionInfo, slice_csv::AbstractString)
    open(slice_csv, "r") do io
        header = parse_csv_line(readline(io))
        idx = header_index(header)
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = parse_csv_line(line)
            g = dims.geology_index[values[idx["geology_id"]]]
            c = dims.case_index[parse(Int, values[idx["case_id"]])]
            s = dims.slice_index[parse(Int, values[idx["slice_index"]])]
            w = dims.window_index[values[idx["window"]]]

            arrays.logK[g, c, s, w, 1] = parse(Float64, values[idx["log_kxx"]])
            arrays.logK[g, c, s, w, 2] = parse(Float64, values[idx["log_kyy"]])
            arrays.logK[g, c, s, w, 3] = parse(Float64, values[idx["log_kzz"]])
            arrays.perm[g, c, s, w, 1] = parse(Float64, values[idx["perm_kxx"]])
            arrays.perm[g, c, s, w, 2] = parse(Float64, values[idx["perm_kyy"]])
            arrays.perm[g, c, s, w, 3] = parse(Float64, values[idx["perm_kzz"]])

            arrays.selected_sample_index[g, c, s, w] = parse(Int64, values[idx["selected_sample_index"]])
            arrays.source_pool_size[g, c, s, w] = parse(Int64, values[idx["source_pool_size"]])
            arrays.draw_group_index[g, c, s] = parse(Int64, values[idx["draw_group_index"]])
            arrays.draw_seed[g, c, s] = parse(Int64, values[idx["draw_seed"]])
            arrays.sampler_random_seed = parse(Int64, values[idx["sampler_random_seed"]])

            arrays.source_seed_base[g, w] = parse(Int64, values[idx["source_seed_base"]])
            arrays.source_num_attempts[g, w] = parse(Int64, values[idx["source_num_attempts"]])
            arrays.source_num_rejected[g, w] = parse(Int64, values[idx["source_num_rejected"]])
            arrays.exact_replay_seed[g, c, s, w] = parse_optional_int(values[idx["exact_replay_seed"]])
            arrays.fine_scale_replay_status_code[g, c, s, w] = replay_status_code(values[idx["fine_scale_replay_status"]])
            arrays.source_checkpoint_file[g, w] = values[idx["source_checkpoint_file"]]

            arrays.scenario_index[g] = parse(Int64, values[idx["scenario_index"]])
            arrays.scenario_label[g] = values[idx["scenario_label"]]
            arrays.scenario_name[g] = values[idx["scenario_name"]]
            arrays.geologic_case_index[g] = parse(Int64, values[idx["case_index"]])
            arrays.geologic_case_label[g] = values[idx["case_label"]]
            arrays.faulting_depth_m[g] = parse(Float64, values[idx["faulting_depth_m"]])
            arrays.sand_vcl[g] = parse(Float64, values[idx["sand_vcl"]])
            arrays.clay_vcl[g] = parse(Float64, values[idx["clay_vcl"]])

            arrays.level3_case_name[g, c] = values[idx["case_name"]]
            arrays.level3_case_category[g, c] = values[idx["case_category"]]
            arrays.level3_case_strength[g, c] = values[idx["case_strength"]]
            arrays.pattern_name[g, c] = values[idx["pattern_name"]]
            arrays.orientation[g, c] = values[idx["orientation"]]
            arrays.group_split_id[g, c] = parse(Int64, values[idx["group_split_id"]])
            arrays.assigned_state[g, c, w] = values[idx["assigned_state"]]
            arrays.sampling_mode[g, c, w] = values[idx["sampling_mode"]]
            arrays.sampling_pool[g, c, w] = values[idx["sampling_pool"]]
        end
    end
    return arrays
end

"""
    read_slice_draw_groups(path)

Read the slice-to-draw-group map into compact arrays.
"""
function read_slice_draw_groups(path::AbstractString)
    lines = readlines(path)
    header = parse_csv_line(lines[1])
    idx = header_index(header)
    n = length(lines) - 1
    slice_draw_group_index = fill(Int64(-1), n)
    slice_is_shared_draw_group = fill(Int64(0), n)
    slice_draw_group_slices = empty_any_array(n)
    for line in lines[2:end]
        values = parse_csv_line(line)
        slice = parse(Int, values[idx["slice_index"]])
        slice_draw_group_index[slice] = parse(Int64, values[idx["draw_group_index"]])
        slice_is_shared_draw_group[slice] = lowercase(values[idx["is_shared_draw_group"]]) == "true" ? 1 : 0
        slice_draw_group_slices[slice] = values[idx["draw_group_slices"]]
    end
    return Dict{String, Any}(
        "slice_draw_group_index" => slice_draw_group_index,
        "slice_is_shared_draw_group" => slice_is_shared_draw_group,
        "slice_draw_group_slices" => slice_draw_group_slices,
    )
end

"""
    validate_mat_against_csv(output_root; mat_path="", tolerance=1e-10)

Stream through the CSV and verify that key numeric values match the MAT arrays.
"""
function validate_mat_against_csv(output_root::AbstractString;
                                  mat_path::AbstractString = "",
                                  tolerance::Real = 1e-10)
    root = normpath(output_root)
    slice_csv = joinpath(root, "texas_field_slice_window_values.csv")
    mat_file = isempty(mat_path) ? joinpath(root, "texas_field_sampling_compact.mat") : normpath(mat_path)
    vars = matread(mat_file)
    field_perm = vars["fieldPerm"]

    geology_ids = String.(field_perm["geology_id"])
    case_ids = Int.(vec(field_perm["level3_case_id"]))
    slice_indices = Int.(vec(field_perm["slice_index"]))
    windows = String.(field_perm["window"])
    gmap = Dict(id => i for (i, id) in enumerate(geology_ids))
    cmap = Dict(id => i for (i, id) in enumerate(case_ids))
    smap = Dict(id => i for (i, id) in enumerate(slice_indices))
    wmap = Dict(id => i for (i, id) in enumerate(windows))

    logK = field_perm["logK"]
    perm = field_perm["perm"]
    selected_sample_index = field_perm["selected_sample_index"]
    exact_replay_seed = field_perm["exact_replay_seed"]
    checked = 0
    mismatches = 0

    open(slice_csv, "r") do io
        header = parse_csv_line(readline(io))
        idx = header_index(header)
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = parse_csv_line(line)
            g = gmap[values[idx["geology_id"]]]
            c = cmap[parse(Int, values[idx["case_id"]])]
            s = smap[parse(Int, values[idx["slice_index"]])]
            w = wmap[values[idx["window"]]]
            checked += 1
            mismatches += abs(logK[g, c, s, w, 1] - parse(Float64, values[idx["log_kxx"]])) > tolerance ? 1 : 0
            mismatches += abs(logK[g, c, s, w, 2] - parse(Float64, values[idx["log_kyy"]])) > tolerance ? 1 : 0
            mismatches += abs(logK[g, c, s, w, 3] - parse(Float64, values[idx["log_kzz"]])) > tolerance ? 1 : 0
            mismatches += abs(perm[g, c, s, w, 1] - parse(Float64, values[idx["perm_kxx"]])) > tolerance ? 1 : 0
            mismatches += abs(perm[g, c, s, w, 2] - parse(Float64, values[idx["perm_kyy"]])) > tolerance ? 1 : 0
            mismatches += abs(perm[g, c, s, w, 3] - parse(Float64, values[idx["perm_kzz"]])) > tolerance ? 1 : 0
            mismatches += selected_sample_index[g, c, s, w] != parse(Int, values[idx["selected_sample_index"]]) ? 1 : 0
            mismatches += exact_replay_seed[g, c, s, w] != parse_optional_int(values[idx["exact_replay_seed"]]) ? 1 : 0
        end
    end

    return Dict(
        "checked_slice_window_rows" => checked,
        "mismatch_count" => mismatches,
        "mat_path" => mat_file,
        "csv_path" => slice_csv,
    )
end

function header_index(header::Vector{String})
    return Dict(name => i for (i, name) in enumerate(header))
end

function replay_status_code(status::AbstractString)
    status == "direct_exact_replay_possible_if_PREDICT_code_unchanged" && return Int64(1)
    status == "requires_valid_attempt_replay_due_to_rejected_realizations" && return Int64(2)
    isempty(strip(status)) && return Int64(0)
    error("Unknown fine_scale_replay_status: $status")
end

function parse_optional_int(value::AbstractString)
    stripped = strip(value)
    isempty(stripped) && return Int64(-1)
    return parse(Int64, stripped)
end

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

end
