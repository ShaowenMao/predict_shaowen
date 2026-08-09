const FIELD_PERM_SCHEMA_VERSION = "independent_full_fault_field_permeability_v1"
const FIELD_PERM_DIMENSION_ORDER = "geology x sampling_case x slice x window x component"
const FIELD_PERM_WINDOWS = ["famp$(index)" for index in 1:6]
const FIELD_PERM_COMPONENTS = ["kxx", "kyy", "kzz"]

Base.@kwdef struct FieldPermDimensions
    geology_ids::Vector{String}
    case_ids::Vector{Int}
    slice_ids::Vector{Int}
    geology_index::Dict{String, Int}
    case_index::Dict{Int, Int}
    slice_index::Dict{Int, Int}
    window_index::Dict{String, Int}
end

Base.@kwdef mutable struct FieldPermArrays
    logK::Array{Float64, 5}
    perm::Array{Float64, 5}
    source_library_row::Array{Int64, 4}
    predict_realization_id::Array{Int64, 4}
    exact_replay_seed::Array{Int64, 4}
    sampling_seed::Array{Int64, 3}
    present::BitArray{4}
    sampling_case_id::Matrix{Any}
    phase::Matrix{Any}
    case_type::Matrix{Any}
    replicate_id::Matrix{Int64}
    use_for_probabilistic_uq::BitMatrix
    is_benchmark::BitMatrix
    is_stress_test::BitMatrix
    case_name::Matrix{Any}
    case_category::Matrix{Any}
    assigned_state::Array{Any, 3}
    sampling_mode::Array{Any, 3}
    sampling_pool::Array{Any, 3}
    scenario_index::Vector{Int64}
    scenario_label::Vector{Any}
    scenario_name::Vector{Any}
    geologic_case_index::Vector{Int64}
    geologic_case_label::Vector{Any}
    faulting_depth_m::Vector{Float64}
    sand_vcl::Vector{Float64}
    clay_vcl::Vector{Float64}
    checkpoint_sha256::Matrix{Any}
    checkpoint_relative_path::Matrix{Any}
    source_checkpoint_file::Matrix{Any}
end

"""Return an empty MATLAB-compatible cell array."""
function empty_mat_cell(dimensions::Integer...)
    result = Array{Any}(undef, dimensions...)
    fill!(result, "")
    return result
end

function adapted_header_index(header::Vector{String})
    return Dict(name => index for (index, name) in enumerate(header))
end

function require_adapted_columns(index::Dict{String, Int})
    required = [
        "adapter_schema_version", "design_version", "geology_id",
        "scenario_index", "scenario_label", "scenario_name", "case_index",
        "case_label", "faulting_depth_m", "sand_vcl", "clay_vcl", "case_id",
        "sampling_case_id", "phase", "case_type", "replicate_id", "case_name",
        "case_category", "slice_index", "window", "assigned_state",
        "sampling_mode", "sampling_pool", "selected_sample_index",
        "predict_realization_id", "draw_seed", "source_checkpoint_file",
        "checkpoint_relative_path", "checkpoint_sha256", "exact_replay_seed",
        "log_kxx", "log_kyy", "log_kzz", "perm_kxx", "perm_kyy",
        "perm_kzz", "sampling_code_commit", "sampling_method_config_hash",
        "predict_code_commit", "predict_method_config_hash",
        "use_for_probabilistic_uq", "is_benchmark", "is_stress_test",
    ]
    missing = filter(name -> !haskey(index, name), required)
    isempty(missing) || error("Adapted sampling CSV lacks columns: $(join(missing, ", "))")
    return nothing
end

function collect_field_perm_dimensions(path::AbstractString)
    geology_ids = String[]
    case_ids = Int[]
    slice_ids = Int[]
    geology_seen = Set{String}()
    case_seen = Set{Int}()
    slice_seen = Set{Int}()

    open(path, "r") do io
        eof(io) && error("Adapted sampling CSV is empty: $path")
        index = adapted_header_index(parse_csv_line(readline(io)))
        require_adapted_columns(index)
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = parse_csv_line(line)
            geology_id = values[index["geology_id"]]
            case_id = parse(Int, values[index["case_id"]])
            slice_id = parse(Int, values[index["slice_index"]])
            if !(geology_id in geology_seen)
                push!(geology_ids, geology_id)
                push!(geology_seen, geology_id)
            end
            if !(case_id in case_seen)
                push!(case_ids, case_id)
                push!(case_seen, case_id)
            end
            if !(slice_id in slice_seen)
                push!(slice_ids, slice_id)
                push!(slice_seen, slice_id)
            end
        end
    end

    sort!(geology_ids; by = geology_sort_key)
    sort!(case_ids)
    sort!(slice_ids)
    geology_ids == unique(geology_ids) || error("Duplicate geology labels")
    case_ids == unique(case_ids) || error("Duplicate numeric case aliases")
    slice_ids == collect(1:87) || error("Expected slice IDs 1:87; found $slice_ids")
    return FieldPermDimensions(
        geology_ids = geology_ids,
        case_ids = case_ids,
        slice_ids = slice_ids,
        geology_index = Dict(value => index for (index, value) in enumerate(geology_ids)),
        case_index = Dict(value => index for (index, value) in enumerate(case_ids)),
        slice_index = Dict(value => index for (index, value) in enumerate(slice_ids)),
        window_index = Dict(value => index for (index, value) in enumerate(FIELD_PERM_WINDOWS)),
    )
end

function allocate_field_perm_arrays(dimensions::FieldPermDimensions)
    n_geologies = length(dimensions.geology_ids)
    n_cases = length(dimensions.case_ids)
    n_slices = length(dimensions.slice_ids)
    n_windows = length(FIELD_PERM_WINDOWS)
    n_components = length(FIELD_PERM_COMPONENTS)
    return FieldPermArrays(
        logK = fill(NaN, n_geologies, n_cases, n_slices, n_windows, n_components),
        perm = fill(NaN, n_geologies, n_cases, n_slices, n_windows, n_components),
        source_library_row = fill(Int64(-1), n_geologies, n_cases, n_slices, n_windows),
        predict_realization_id = fill(Int64(-1), n_geologies, n_cases, n_slices, n_windows),
        exact_replay_seed = fill(Int64(-1), n_geologies, n_cases, n_slices, n_windows),
        sampling_seed = fill(Int64(-1), n_geologies, n_cases, n_windows),
        present = falses(n_geologies, n_cases, n_slices, n_windows),
        sampling_case_id = empty_mat_cell(n_geologies, n_cases),
        phase = empty_mat_cell(n_geologies, n_cases),
        case_type = empty_mat_cell(n_geologies, n_cases),
        replicate_id = fill(Int64(-1), n_geologies, n_cases),
        use_for_probabilistic_uq = falses(n_geologies, n_cases),
        is_benchmark = falses(n_geologies, n_cases),
        is_stress_test = falses(n_geologies, n_cases),
        case_name = empty_mat_cell(n_geologies, n_cases),
        case_category = empty_mat_cell(n_geologies, n_cases),
        assigned_state = empty_mat_cell(n_geologies, n_cases, n_windows),
        sampling_mode = empty_mat_cell(n_geologies, n_cases, n_windows),
        sampling_pool = empty_mat_cell(n_geologies, n_cases, n_windows),
        scenario_index = fill(Int64(-1), n_geologies),
        scenario_label = empty_mat_cell(n_geologies),
        scenario_name = empty_mat_cell(n_geologies),
        geologic_case_index = fill(Int64(-1), n_geologies),
        geologic_case_label = empty_mat_cell(n_geologies),
        faulting_depth_m = fill(NaN, n_geologies),
        sand_vcl = fill(NaN, n_geologies),
        clay_vcl = fill(NaN, n_geologies),
        checkpoint_sha256 = empty_mat_cell(n_geologies, n_windows),
        checkpoint_relative_path = empty_mat_cell(n_geologies, n_windows),
        source_checkpoint_file = empty_mat_cell(n_geologies, n_windows),
    )
end

function assign_consistent!(array, index, value, label::AbstractString; empty_value = "")
    current = array[index...]
    if current == empty_value
        array[index...] = value
    elseif current != value
        error("Inconsistent $label at index $(Tuple(index)): '$current' != '$value'")
    end
    return nothing
end

function assign_consistent_number!(array, index, value, label::AbstractString; missing_value = -1)
    current = array[index...]
    if current == missing_value || (current isa AbstractFloat && isnan(current))
        array[index...] = value
    elseif current != value
        error("Inconsistent $label at index $(Tuple(index)): $current != $value")
    end
    return nothing
end

function parse_adapter_bool(value::AbstractString, label::AbstractString)
    normalized = lowercase(strip(value))
    normalized == "true" && return true
    normalized == "false" && return false
    error("$label must be true or false, received '$value'")
end

function validate_case_role_flags(case_type::AbstractString,
                                  use_for_probabilistic_uq::Bool,
                                  is_benchmark::Bool,
                                  is_stress_test::Bool)
    expected = Dict(
        "independent_full" => (true, false, false),
        "representative_medoid" => (false, true, false),
        "low_state_stress" => (false, false, true),
        "high_state_stress" => (false, false, true),
    )
    haskey(expected, case_type) || error("Unsupported revised case type: $case_type")
    actual = (use_for_probabilistic_uq, is_benchmark, is_stress_test)
    actual == expected[case_type] || error(
        "Role flags for $case_type are $actual; expected $(expected[case_type])"
    )
    return nothing
end

function fill_field_perm_arrays!(arrays::FieldPermArrays,
                                 dimensions::FieldPermDimensions,
                                 path::AbstractString)
    scalar_metadata = Dict{String, Set{String}}(
        name => Set{String}() for name in [
            "adapter_schema_version", "design_version", "sampling_code_commit",
            "sampling_method_config_hash", "predict_code_commit",
            "predict_method_config_hash",
        ]
    )

    open(path, "r") do io
        index = adapted_header_index(parse_csv_line(readline(io)))
        require_adapted_columns(index)
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = parse_csv_line(line)
            g = dimensions.geology_index[values[index["geology_id"]]]
            c = dimensions.case_index[parse(Int, values[index["case_id"]])]
            s = dimensions.slice_index[parse(Int, values[index["slice_index"]])]
            window = values[index["window"]]
            haskey(dimensions.window_index, window) || error("Unknown window: $window")
            w = dimensions.window_index[window]
            arrays.present[g, c, s, w] && error(
                "Duplicate assignment for $(dimensions.geology_ids[g]), case $(dimensions.case_ids[c]), slice $(dimensions.slice_ids[s]), $window"
            )
            arrays.present[g, c, s, w] = true

            arrays.logK[g, c, s, w, :] .= parse.(Float64, values[index[name]] for name in ("log_kxx", "log_kyy", "log_kzz"))
            arrays.perm[g, c, s, w, :] .= parse.(Float64, values[index[name]] for name in ("perm_kxx", "perm_kyy", "perm_kzz"))
            arrays.source_library_row[g, c, s, w] = parse(Int64, values[index["selected_sample_index"]])
            arrays.predict_realization_id[g, c, s, w] = parse(Int64, values[index["predict_realization_id"]])
            arrays.exact_replay_seed[g, c, s, w] = parse(Int64, values[index["exact_replay_seed"]])
            seed = parse(Int64, values[index["draw_seed"]])
            assign_consistent_number!(arrays.sampling_seed, (g, c, w), seed, "sampling seed")

            assign_consistent!(arrays.sampling_case_id, (g, c), values[index["sampling_case_id"]], "sampling case ID")
            assign_consistent!(arrays.phase, (g, c), values[index["phase"]], "phase")
            assign_consistent!(arrays.case_type, (g, c), values[index["case_type"]], "case type")
            assign_consistent_number!(arrays.replicate_id, (g, c), parse(Int64, values[index["replicate_id"]]), "replicate ID")
            use_for_probabilistic_uq = parse_adapter_bool(values[index["use_for_probabilistic_uq"]], "use_for_probabilistic_uq")
            is_benchmark = parse_adapter_bool(values[index["is_benchmark"]], "is_benchmark")
            is_stress_test = parse_adapter_bool(values[index["is_stress_test"]], "is_stress_test")
            validate_case_role_flags(values[index["case_type"]],
                                     use_for_probabilistic_uq,
                                     is_benchmark,
                                     is_stress_test)
            arrays.use_for_probabilistic_uq[g, c] = use_for_probabilistic_uq
            arrays.is_benchmark[g, c] = is_benchmark
            arrays.is_stress_test[g, c] = is_stress_test
            assign_consistent!(arrays.case_name, (g, c), values[index["case_name"]], "case name")
            assign_consistent!(arrays.case_category, (g, c), values[index["case_category"]], "case category")
            assign_consistent!(arrays.assigned_state, (g, c, w), values[index["assigned_state"]], "assigned state")
            assign_consistent!(arrays.sampling_mode, (g, c, w), values[index["sampling_mode"]], "sampling mode")
            assign_consistent!(arrays.sampling_pool, (g, c, w), values[index["sampling_pool"]], "sampling pool")

            assign_consistent_number!(arrays.scenario_index, (g,), parse(Int64, values[index["scenario_index"]]), "scenario index")
            assign_consistent!(arrays.scenario_label, (g,), values[index["scenario_label"]], "scenario label")
            assign_consistent!(arrays.scenario_name, (g,), values[index["scenario_name"]], "scenario name")
            assign_consistent_number!(arrays.geologic_case_index, (g,), parse(Int64, values[index["case_index"]]), "geologic case index")
            assign_consistent!(arrays.geologic_case_label, (g,), values[index["case_label"]], "geologic case label")
            assign_consistent_number!(arrays.faulting_depth_m, (g,), parse(Float64, values[index["faulting_depth_m"]]), "faulting depth"; missing_value = NaN)
            assign_consistent_number!(arrays.sand_vcl, (g,), parse(Float64, values[index["sand_vcl"]]), "sand Vcl"; missing_value = NaN)
            assign_consistent_number!(arrays.clay_vcl, (g,), parse(Float64, values[index["clay_vcl"]]), "clay Vcl"; missing_value = NaN)
            assign_consistent!(arrays.checkpoint_sha256, (g, w), lowercase(values[index["checkpoint_sha256"]]), "checkpoint SHA-256")
            assign_consistent!(arrays.checkpoint_relative_path, (g, w), values[index["checkpoint_relative_path"]], "checkpoint relative path")
            assign_consistent!(arrays.source_checkpoint_file, (g, w), values[index["source_checkpoint_file"]], "checkpoint file")

            for (name, collected) in scalar_metadata
                push!(collected, values[index[name]])
            end
        end
    end

    all(arrays.present) || begin
        missing_count = count(!, arrays.present)
        error("Adapted sampling CSV is not rectangular; $missing_count geology/case/slice/window assignments are missing")
    end
    all(isfinite, arrays.logK) || error("Nonfinite log permeability in adapted sampling CSV")
    all(isfinite, arrays.perm) && all(arrays.perm .> 0) || error("Permeability must be positive and finite")
    maximum(abs.(arrays.logK .- log10.(arrays.perm))) <= 1.0e-10 || error(
        "Stored log permeability and permeability values are inconsistent"
    )
    all(arrays.source_library_row .> 0) || error("Source-library rows must be positive")
    all(arrays.predict_realization_id .> 0) || error("PREDICT realization IDs must be positive")
    all(arrays.exact_replay_seed .> 0) || error("Exact replay seeds must be positive")
    all(arrays.replicate_id .>= 0) || error("Replicate IDs must be nonnegative")
    for g in axes(arrays.sampling_seed, 1),
        c in axes(arrays.sampling_seed, 2),
        w in axes(arrays.sampling_seed, 3)
        seed = arrays.sampling_seed[g, c, w]
        case_type = String(arrays.case_type[g, c])
        if case_type == "independent_full"
            seed > 0 || error("Independent sampling seeds must be positive")
        else
            seed == 0 || error("Deterministic benchmark/stress sampling seeds must be zero")
        end
    end

    return Dict(name => only(values) for (name, values) in scalar_metadata)
end

function metadata_json_value(path::AbstractString, key::AbstractString)
    isfile(path) || return ""
    pattern = Regex("\\\"$(key)\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"")
    matched = match(pattern, read(path, String))
    return isnothing(matched) ? "" : matched.captures[1]
end

function fault_local_coordinate_contract()
    return Dict{String, Any}(
        "schema_version" => "fault_local_coordinate_contract_v1",
        "stored_coordinate_system" => "PREDICT fault-local orthonormal axes",
        "component_labels" => FIELD_PERM_COMPONENTS,
        "component_meanings" => ["fault_normal", "along_strike", "down_dip"],
        "reservoir_coordinate_system" => "reservoir-grid Cartesian axes",
        "rotation_required" => true,
        "downstream_transform_contract" => "fault_local_to_reservoir_grid_signed_yz_v1",
        "global_x_mapping" => "reservoir Kxx equals fault-local kyy (along strike)",
        "yz_rotation" => "theta = sign(dY/dZ) * (dip_deg - 90 deg)",
        "tensor_formula" => "K_global = R * diag(kxx_local, kyy_local, kzz_local) * transpose(R)",
        "component_formula" => "Kxx=kyy; Kyy=cos(theta)^2*kxx+sin(theta)^2*kzz; Kyz=cos(theta)*sin(theta)*(kxx-kzz); Kzz=sin(theta)^2*kxx+cos(theta)^2*kzz; Kxy=Kxz=0",
        "pre_rotated" => false,
        "geometry_dependency" => "theta must be calculated from the paired fault-node trace for each reservoir fault cell",
    )
end

"""
    export_field_permeability_mat(adapted_csv; output_path="")

Export a rectangular revised-sampling adapter CSV to a compact MATLAB
`fieldPerm` object. The CSV remains the source of truth; this function does
not resample or rotate permeability. Phase 1 and Phase 2 should be exported
separately because they contain different geology/case rectangles.
"""
function export_field_permeability_mat(adapted_csv::AbstractString;
                                       output_path::AbstractString = "")
    source = normpath(adapted_csv)
    isfile(source) || error("Adapted sampling CSV not found: $source")
    destination = isempty(output_path) ?
        joinpath(dirname(source), "fault_permeability_independent_full_fault_v1.mat") :
        normpath(output_path)
    dimensions = collect_field_perm_dimensions(source)
    arrays = allocate_field_perm_arrays(dimensions)
    scalar = fill_field_perm_arrays!(arrays, dimensions, source)
    metadata_path = source * ".metadata.json"
    isfile(metadata_path) || error(
        "Adapted sampling metadata is required for manifest provenance: $metadata_path"
    )
    canonical_manifest_hash = metadata_json_value(metadata_path, "canonical_manifest_sha256")
    geology_catalog_hash = metadata_json_value(metadata_path, "geology_catalog_sha256")
    recorded_adapter_hash = metadata_json_value(metadata_path, "output_sha256")
    adapter_hash = file_sha256(source)
    length(canonical_manifest_hash) == 64 || error("Missing canonical sampling-manifest SHA-256")
    length(geology_catalog_hash) == 64 || error("Missing geology-catalog SHA-256")
    recorded_adapter_hash == adapter_hash || error(
        "Adapted sampling CSV does not match its metadata SHA-256"
    )

    field_perm = Dict{String, Any}(
        "schema_version" => FIELD_PERM_SCHEMA_VERSION,
        "design_version" => scalar["design_version"],
        "adapter_schema_version" => scalar["adapter_schema_version"],
        "dimension_order" => FIELD_PERM_DIMENSION_ORDER,
        "geology_id" => dimensions.geology_ids,
        "level3_case_id" => dimensions.case_ids,
        "sampling_case_id" => arrays.sampling_case_id,
        "phase" => arrays.phase,
        "case_type" => arrays.case_type,
        "replicate_id" => arrays.replicate_id,
        "use_for_probabilistic_uq" => arrays.use_for_probabilistic_uq,
        "is_benchmark" => arrays.is_benchmark,
        "is_stress_test" => arrays.is_stress_test,
        "level3_case_name" => arrays.case_name,
        "level3_case_category" => arrays.case_category,
        "slice_index" => dimensions.slice_ids,
        "window" => FIELD_PERM_WINDOWS,
        "component" => FIELD_PERM_COMPONENTS,
        "scenario_index" => arrays.scenario_index,
        "scenario_label" => arrays.scenario_label,
        "scenario_name" => arrays.scenario_name,
        "geologic_case_index" => arrays.geologic_case_index,
        "geologic_case_label" => arrays.geologic_case_label,
        "faulting_depth_m" => arrays.faulting_depth_m,
        "sand_vcl" => arrays.sand_vcl,
        "clay_vcl" => arrays.clay_vcl,
        "assigned_state" => arrays.assigned_state,
        "sampling_mode" => arrays.sampling_mode,
        "sampling_pool" => arrays.sampling_pool,
        "logK" => arrays.logK,
        "perm" => arrays.perm,
        "selected_sample_index" => arrays.source_library_row,
        "predict_realization_id" => arrays.predict_realization_id,
        "exact_replay_seed" => arrays.exact_replay_seed,
        "sampling_seed" => arrays.sampling_seed,
        "checkpoint_sha256" => arrays.checkpoint_sha256,
        "checkpoint_relative_path" => arrays.checkpoint_relative_path,
        "source_checkpoint_file" => arrays.source_checkpoint_file,
        "sampling_code_commit" => scalar["sampling_code_commit"],
        "sampling_method_config_hash" => scalar["sampling_method_config_hash"],
        "predict_code_commit" => scalar["predict_code_commit"],
        "predict_method_config_hash" => scalar["predict_method_config_hash"],
        "sampling_manifest_sha256" => canonical_manifest_hash,
        "geology_catalog_sha256" => geology_catalog_hash,
        "adapter_csv_sha256" => adapter_hash,
        "adapter_metadata_sha256" => isfile(metadata_path) ? file_sha256(metadata_path) : "",
        "permeability_units" => "mD",
        "coordinate_contract" => fault_local_coordinate_contract(),
        "created_from_csv_name" => basename(source),
    )
    mkpath(dirname(destination))
    matwrite(destination, Dict("fieldPerm" => field_perm))
    validation = validate_field_permeability_mat(source, destination)
    validation["mismatch_count"] == 0 || error("MAT/CSV validation failed")
    return destination
end

"""Validate every assignment and the coordinate metadata in a MAT export."""
function validate_field_permeability_mat(adapted_csv::AbstractString,
                                         mat_path::AbstractString;
                                         tolerance::Real = 1.0e-10)
    variables = matread(mat_path)
    haskey(variables, "fieldPerm") || error("MAT file lacks fieldPerm: $mat_path")
    field_perm = variables["fieldPerm"]
    String(field_perm["schema_version"]) == FIELD_PERM_SCHEMA_VERSION || error("Unexpected fieldPerm schema")
    coordinate = field_perm["coordinate_contract"]
    Bool(coordinate["rotation_required"]) || error("Coordinate contract must require downstream rotation")
    Bool(coordinate["pre_rotated"]) && error("Fault-local permeability must not be pre-rotated")
    String(coordinate["downstream_transform_contract"]) == "fault_local_to_reservoir_grid_signed_yz_v1" || error("Unexpected coordinate transform contract")

    geology = String.(vec(field_perm["geology_id"]))
    cases = Int.(vec(field_perm["level3_case_id"]))
    slices = Int.(vec(field_perm["slice_index"]))
    windows = String.(vec(field_perm["window"]))
    geology_index = Dict(value => index for (index, value) in enumerate(geology))
    case_index = Dict(value => index for (index, value) in enumerate(cases))
    slice_index = Dict(value => index for (index, value) in enumerate(slices))
    window_index = Dict(value => index for (index, value) in enumerate(windows))
    logK = field_perm["logK"]
    perm = field_perm["perm"]
    selected = field_perm["selected_sample_index"]
    realization = field_perm["predict_realization_id"]
    replay_seed = field_perm["exact_replay_seed"]
    sampling_seed = field_perm["sampling_seed"]
    sampling_case_id = field_perm["sampling_case_id"]
    phase = field_perm["phase"]
    case_type = field_perm["case_type"]
    replicate_id = field_perm["replicate_id"]
    use_for_probabilistic_uq = field_perm["use_for_probabilistic_uq"]
    is_benchmark = field_perm["is_benchmark"]
    is_stress_test = field_perm["is_stress_test"]
    case_name = field_perm["level3_case_name"]
    case_category = field_perm["level3_case_category"]
    assigned_state = field_perm["assigned_state"]
    sampling_mode = field_perm["sampling_mode"]
    sampling_pool = field_perm["sampling_pool"]
    checkpoint_sha256 = field_perm["checkpoint_sha256"]
    checkpoint_relative_path = field_perm["checkpoint_relative_path"]
    source_checkpoint_file = field_perm["source_checkpoint_file"]
    checked = 0
    mismatches = 0

    open(adapted_csv, "r") do io
        index = adapted_header_index(parse_csv_line(readline(io)))
        for line in eachline(io)
            isempty(strip(line)) && continue
            values = parse_csv_line(line)
            g = geology_index[values[index["geology_id"]]]
            c = case_index[parse(Int, values[index["case_id"]])]
            s = slice_index[parse(Int, values[index["slice_index"]])]
            w = window_index[values[index["window"]]]
            checked += 1
            for (component, log_name, perm_name) in zip(1:3, ("log_kxx", "log_kyy", "log_kzz"), ("perm_kxx", "perm_kyy", "perm_kzz"))
                mismatches += abs(logK[g, c, s, w, component] - parse(Float64, values[index[log_name]])) > tolerance
                mismatches += abs(perm[g, c, s, w, component] - parse(Float64, values[index[perm_name]])) > tolerance
            end
            mismatches += selected[g, c, s, w] != parse(Int, values[index["selected_sample_index"]])
            mismatches += realization[g, c, s, w] != parse(Int, values[index["predict_realization_id"]])
            mismatches += replay_seed[g, c, s, w] != parse(Int, values[index["exact_replay_seed"]])
            mismatches += sampling_seed[g, c, w] != parse(Int, values[index["draw_seed"]])
            mismatches += String(sampling_case_id[g, c]) != values[index["sampling_case_id"]]
            mismatches += String(phase[g, c]) != values[index["phase"]]
            mismatches += String(case_type[g, c]) != values[index["case_type"]]
            mismatches += replicate_id[g, c] != parse(Int, values[index["replicate_id"]])
            mismatches += Bool(use_for_probabilistic_uq[g, c]) != parse_adapter_bool(values[index["use_for_probabilistic_uq"]], "use_for_probabilistic_uq")
            mismatches += Bool(is_benchmark[g, c]) != parse_adapter_bool(values[index["is_benchmark"]], "is_benchmark")
            mismatches += Bool(is_stress_test[g, c]) != parse_adapter_bool(values[index["is_stress_test"]], "is_stress_test")
            mismatches += String(case_name[g, c]) != values[index["case_name"]]
            mismatches += String(case_category[g, c]) != values[index["case_category"]]
            mismatches += String(assigned_state[g, c, w]) != values[index["assigned_state"]]
            mismatches += String(sampling_mode[g, c, w]) != values[index["sampling_mode"]]
            mismatches += String(sampling_pool[g, c, w]) != values[index["sampling_pool"]]
            mismatches += lowercase(String(checkpoint_sha256[g, w])) != lowercase(values[index["checkpoint_sha256"]])
            mismatches += String(checkpoint_relative_path[g, w]) != values[index["checkpoint_relative_path"]]
            mismatches += String(source_checkpoint_file[g, w]) != values[index["source_checkpoint_file"]]
        end
    end

    scalar_expectations = Dict(
        "design_version" => "design_version",
        "adapter_schema_version" => "adapter_schema_version",
        "sampling_code_commit" => "sampling_code_commit",
        "sampling_method_config_hash" => "sampling_method_config_hash",
        "predict_code_commit" => "predict_code_commit",
        "predict_method_config_hash" => "predict_method_config_hash",
    )
    open(adapted_csv, "r") do io
        index = adapted_header_index(parse_csv_line(readline(io)))
        first_values = nothing
        for line in eachline(io)
            isempty(strip(line)) && continue
            first_values = parse_csv_line(line)
            break
        end
        isnothing(first_values) && error("Adapted sampling CSV is empty: $adapted_csv")
        for (mat_name, csv_name) in scalar_expectations
            mismatches += String(field_perm[mat_name]) != first_values[index[csv_name]]
        end
    end
    return Dict{String, Any}(
        "checked_assignment_rows" => checked,
        "mismatch_count" => mismatches,
        "csv_path" => normpath(adapted_csv),
        "mat_path" => normpath(mat_path),
        "mat_sha256" => file_sha256(mat_path),
    )
end
