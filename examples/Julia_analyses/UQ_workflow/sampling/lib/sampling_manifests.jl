const INDEPENDENT_CASE = "independent_full"
const REPRESENTATIVE_CASE = "representative_medoid"
const LOW_STRESS_CASE = "low_state_stress"
const HIGH_STRESS_CASE = "high_state_stress"

function case_id(geology_id::AbstractString, case_type::AbstractString, replicate_id::Integer)
    case_type == INDEPENDENT_CASE && return "$(geology_id)_IND_$(lpad(replicate_id, 3, '0'))"
    case_type == REPRESENTATIVE_CASE && return "$(geology_id)_REP_MEDOID"
    case_type == LOW_STRESS_CASE && return "$(geology_id)_LOW_STRESS"
    case_type == HIGH_STRESS_CASE && return "$(geology_id)_HIGH_STRESS"
    error("Unsupported case type: $case_type")
end

"""Return the 15 default Phase 1 case specifications for one geology."""
function phase1_case_specs(cfg::SamplingConfig, geology_id::AbstractString)
    specs = CaseSpec[]
    for replicate_id in 1:cfg.phase1_n_independent
        push!(specs, CaseSpec("phase1", case_id(geology_id, INDEPENDENT_CASE, replicate_id),
                             INDEPENDENT_CASE, replicate_id, true, false, false))
    end
    cfg.phase1_include_representative &&
        push!(specs, CaseSpec("phase1", case_id(geology_id, REPRESENTATIVE_CASE, 0),
                             REPRESENTATIVE_CASE, 0, false, true, false))
    cfg.phase1_include_low_stress &&
        push!(specs, CaseSpec("phase1", case_id(geology_id, LOW_STRESS_CASE, 0),
                             LOW_STRESS_CASE, 0, false, false, true))
    cfg.phase1_include_high_stress &&
        push!(specs, CaseSpec("phase1", case_id(geology_id, HIGH_STRESS_CASE, 0),
                             HIGH_STRESS_CASE, 0, false, false, true))
    return specs
end

"""Return Phase 2 independent specifications with replicate IDs continuing Phase 1."""
function phase2_case_specs(cfg::SamplingConfig, geology_id::AbstractString)
    first_id = cfg.phase1_n_independent + 1
    last_id = cfg.phase1_n_independent + cfg.phase2_n_additional_independent
    return [CaseSpec("phase2", case_id(geology_id, INDEPENDENT_CASE, replicate_id),
                     INDEPENDENT_CASE, replicate_id, true, false, false)
            for replicate_id in first_id:last_id]
end

function deterministic_source(spec::CaseSpec, library::PredictLibrary)
    spec.case_type == REPRESENTATIVE_CASE &&
        return (library.full_medoid_row, "full_distribution")
    spec.case_type == LOW_STRESS_CASE &&
        return (library.low_medoid_row, "low_state")
    spec.case_type == HIGH_STRESS_CASE &&
        return (library.high_medoid_row, "high_state")
    error("Case $(spec.case_type) is not a deterministic benchmark")
end

"""Generate all 6 x n_slices assignments for one case."""
function generate_case_manifest(libraries::Dict{String, PredictLibrary},
                                cfg::SamplingConfig,
                                spec::CaseSpec)
    Set(keys(libraries)) == Set(cfg.windows) || error("Library windows do not match configuration")
    geology_ids = unique(library.geology_id for library in values(libraries))
    length(geology_ids) == 1 || error("Case libraries must belong to one geology")
    geology_id = only(geology_ids)
    startswith(spec.case_id, geology_id * "_") || error("Case ID/geology mismatch")

    rows = ManifestRow[]
    sizehint!(rows, length(cfg.windows) * cfg.n_slices)
    for window in cfg.windows
        library = libraries[window]
        if spec.case_type == INDEPENDENT_CASE
            sampling_seed = stable_case_seed(cfg.base_seed, geology_id, spec.phase,
                                             spec.case_type, spec.replicate_id, window;
                                             version = cfg.seed_method_version)
            for slice_id in 1:cfg.n_slices
                source_row = stable_uniform_index(sampling_seed, slice_id, n_accepted(library);
                                                  version = cfg.seed_method_version)
                push!(rows, build_manifest_row(library, cfg, spec, slice_id, source_row,
                                               sampling_seed, "full_distribution",
                                               "uniform_with_replacement"))
            end
        else
            source_row, source_library = deterministic_source(spec, library)
            for slice_id in 1:cfg.n_slices
                push!(rows, build_manifest_row(library, cfg, spec, slice_id, source_row,
                                               UInt64(0), source_library,
                                               "exact_logk_medoid"))
            end
        end
    end
    return rows
end

function build_manifest_row(library::PredictLibrary,
                            cfg::SamplingConfig,
                            spec::CaseSpec,
                            slice_id::Int,
                            source_row::Int,
                            sampling_seed::UInt64,
                            source_library::AbstractString,
                            selection_method::AbstractString)
    1 <= source_row <= n_accepted(library) || error("Selected source row is out of range")
    return ManifestRow(
        geology_id = library.geology_id,
        phase = spec.phase,
        case_id = spec.case_id,
        case_type = spec.case_type,
        replicate_id = spec.replicate_id,
        window_id = library.window_id,
        slice_id = slice_id,
        source_library = String(source_library),
        source_library_row = source_row,
        predict_realization_id = library.realization_id[source_row],
        predict_seed = library.predict_seed[source_row],
        exact_replay_seed = library.predict_seed[source_row],
        sampling_seed = sampling_seed,
        sampling_seed_method = sampling_seed == 0 ? "not_applicable" : cfg.seed_method_version,
        selection_method = String(selection_method),
        checkpoint_path = portable_path(library.checkpoint_path),
        checkpoint_relative_path = library.checkpoint_relative_path,
        checkpoint_hash = library.checkpoint_hash,
        level2_state_path = portable_path(library.level2_state_path),
        code_commit = cfg.code_commit,
        code_dirty = cfg.code_dirty,
        method_config_hash = cfg.configuration_hash,
        predict_code_commit = library.predict_code_commit,
        predict_method_config_hash = library.predict_method_config_hash,
        log_kxx = library.logK[source_row, 1],
        log_kyy = library.logK[source_row, 2],
        log_kzz = library.logK[source_row, 3],
        perm_kxx_md = library.K_mD[source_row, 1],
        perm_kyy_md = library.K_mD[source_row, 2],
        perm_kzz_md = library.K_mD[source_row, 3],
        use_for_probabilistic_uq = spec.use_for_probabilistic_uq,
        is_benchmark = spec.is_benchmark,
        is_stress_test = spec.is_stress_test,
    )
end

"""Generate and validate all Phase 1 cases for one geology."""
function generate_phase1_manifest(libraries::Dict{String, PredictLibrary}, cfg::SamplingConfig)
    geology_id = only(unique(library.geology_id for library in values(libraries)))
    rows = reduce(vcat, [generate_case_manifest(libraries, cfg, spec)
                         for spec in phase1_case_specs(cfg, geology_id)]; init = ManifestRow[])
    validate_phase1_manifest(rows, libraries, cfg)
    return canonical_manifest_rows(rows, cfg)
end

"""Generate and validate the additional Phase 2 independent cases for one geology."""
function generate_phase2_manifest(libraries::Dict{String, PredictLibrary}, cfg::SamplingConfig)
    geology_id = only(unique(library.geology_id for library in values(libraries)))
    rows = reduce(vcat, [generate_case_manifest(libraries, cfg, spec)
                         for spec in phase2_case_specs(cfg, geology_id)]; init = ManifestRow[])
    validate_phase2_manifest(rows, libraries, cfg)
    return canonical_manifest_rows(rows, cfg)
end

case_type_order(case_type::AbstractString) =
    case_type == INDEPENDENT_CASE ? 1 :
    case_type == REPRESENTATIVE_CASE ? 2 :
    case_type == LOW_STRESS_CASE ? 3 :
    case_type == HIGH_STRESS_CASE ? 4 : 99

phase_order(phase::AbstractString) = phase == "phase1" ? 1 : phase == "phase2" ? 2 : 99

function canonical_manifest_rows(rows::Vector{ManifestRow}, cfg::SamplingConfig)
    window_order = Dict(window => index for (index, window) in enumerate(cfg.windows))
    return sort(rows; by = row -> (
        geology_sort_key(row.geology_id),
        phase_order(row.phase),
        case_type_order(row.case_type),
        row.replicate_id,
        get(window_order, row.window_id, typemax(Int)),
        row.slice_id,
    ))
end

function manifest_row_values(row::ManifestRow)
    return Any[
        row.schema_version,
        row.design_version,
        row.geology_id,
        row.phase,
        row.case_id,
        row.case_type,
        row.replicate_id,
        row.window_id,
        row.slice_id,
        row.source_library,
        row.source_library_row,
        row.predict_realization_id,
        row.predict_seed,
        row.exact_replay_seed,
        UInt64(row.sampling_seed),
        row.sampling_seed_method,
        row.selection_method,
        row.checkpoint_path,
        row.checkpoint_relative_path,
        row.checkpoint_hash,
        row.level2_state_path,
        row.code_commit,
        row.code_dirty,
        row.method_config_hash,
        row.predict_code_commit,
        row.predict_method_config_hash,
        repr(row.log_kxx),
        repr(row.log_kyy),
        repr(row.log_kzz),
        repr(row.perm_kxx_md),
        repr(row.perm_kyy_md),
        repr(row.perm_kzz_md),
        row.use_for_probabilistic_uq,
        row.is_benchmark,
        row.is_stress_test,
    ]
end

"""Return canonical CSV bytes for deterministic manifest comparison and hashing."""
function canonical_manifest_bytes(rows::Vector{ManifestRow}, cfg::SamplingConfig)
    io = IOBuffer()
    println(io, join(csv_escape.(MANIFEST_HEADER), ','))
    for row in canonical_manifest_rows(rows, cfg)
        println(io, join(csv_escape.(string.(manifest_row_values(row))), ','))
    end
    return take!(io)
end
