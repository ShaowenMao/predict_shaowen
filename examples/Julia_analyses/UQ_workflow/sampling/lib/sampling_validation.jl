function group_manifest_by_case(rows::Vector{ManifestRow})
    grouped = Dict{String, Vector{ManifestRow}}()
    for row in rows
        push!(get!(grouped, row.case_id, ManifestRow[]), row)
    end
    return grouped
end

function expected_source(spec_type::AbstractString, library::PredictLibrary)
    spec_type == REPRESENTATIVE_CASE && return (library.full_medoid_row, "full_distribution")
    spec_type == LOW_STRESS_CASE && return (library.low_medoid_row, "low_state")
    spec_type == HIGH_STRESS_CASE && return (library.high_medoid_row, "high_state")
    error("No deterministic source for $spec_type")
end

"""Validate one complete 6 x n_slices case against its source libraries."""
function validate_case_manifest(rows::Vector{ManifestRow},
                                libraries::Dict{String, PredictLibrary},
                                cfg::SamplingConfig)
    isempty(rows) && error("Case manifest is empty")
    length(rows) == length(cfg.windows) * cfg.n_slices ||
        error("Case $(first(rows).case_id) must contain $(length(cfg.windows) * cfg.n_slices) rows")
    length(unique(row.case_id for row in rows)) == 1 || error("Rows contain multiple case IDs")
    length(unique(row.case_type for row in rows)) == 1 || error("Rows contain multiple case types")
    length(unique(row.phase for row in rows)) == 1 || error("Rows contain multiple phases")
    case_type = first(rows).case_type

    for window in cfg.windows
        window_rows = sort([row for row in rows if row.window_id == window]; by = row -> row.slice_id)
        length(window_rows) == cfg.n_slices || error("$window does not contain all slices")
        [row.slice_id for row in window_rows] == collect(1:cfg.n_slices) ||
            error("$window slice IDs are incomplete or duplicated")
        library = libraries[window]

        for row in window_rows
            row.schema_version == cfg.manifest_schema_version || error("Manifest schema mismatch")
            row.design_version == cfg.design_version || error("Design version mismatch")
            row.geology_id == library.geology_id || error("Manifest geology mismatch")
            1 <= row.source_library_row <= n_accepted(library) || error("Source row out of range")
            source = row.source_library_row
            row.predict_realization_id == library.realization_id[source] || error("Realization ID mismatch")
            row.predict_seed == library.predict_seed[source] || error("PREDICT seed mismatch")
            row.exact_replay_seed == library.predict_seed[source] || error("Replay seed mismatch")
            row.checkpoint_hash == library.checkpoint_hash || error("Checkpoint hash mismatch")
            row.method_config_hash == cfg.configuration_hash || error("Configuration hash mismatch")
            row.code_commit == cfg.code_commit || error("Code commit mismatch")
            row.predict_code_commit == cfg.predict_code_commit || error("PREDICT commit mismatch")
            row.predict_method_config_hash == cfg.predict_method_config_hash ||
                error("PREDICT method hash mismatch")
            row.log_kxx == library.logK[source, 1] || error("log_kxx mismatch")
            row.log_kyy == library.logK[source, 2] || error("log_kyy mismatch")
            row.log_kzz == library.logK[source, 3] || error("log_kzz mismatch")
            row.perm_kxx_md == library.K_mD[source, 1] || error("perm_kxx mismatch")
            row.perm_kyy_md == library.K_mD[source, 2] || error("perm_kyy mismatch")
            row.perm_kzz_md == library.K_mD[source, 3] || error("perm_kzz mismatch")
        end

        if case_type == INDEPENDENT_CASE
            replicate_id = first(window_rows).replicate_id
            phase = first(window_rows).phase
            seed = stable_case_seed(cfg.base_seed, library.geology_id, phase, case_type,
                                    replicate_id, window; version = cfg.seed_method_version)
            all(row.sampling_seed == seed for row in window_rows) || error("Sampling seed mismatch")
            all(row.sampling_seed_method == cfg.seed_method_version for row in window_rows) ||
                error("Sampling seed method mismatch")
            all(row.source_library == "full_distribution" for row in window_rows) ||
                error("Independent case did not use the full distribution")
            all(row.selection_method == "uniform_with_replacement" for row in window_rows) ||
                error("Independent selection method mismatch")
            expected_rows = [stable_uniform_index(seed, slice_id, n_accepted(library);
                                                  version = cfg.seed_method_version)
                             for slice_id in 1:cfg.n_slices]
            [row.source_library_row for row in window_rows] == expected_rows ||
                error("Independent selected rows are not reproducible")
            all(row.use_for_probabilistic_uq && !row.is_benchmark && !row.is_stress_test
                for row in window_rows) || error("Independent case role flags are invalid")
        else
            expected_row, expected_library = expected_source(case_type, library)
            all(row.source_library_row == expected_row for row in window_rows) ||
                error("Deterministic case is not constant along strike")
            all(row.source_library == expected_library for row in window_rows) ||
                error("Deterministic source-library label mismatch")
            all(row.sampling_seed == 0 for row in window_rows) ||
                error("Deterministic case must not claim a sampling seed")
            all(row.sampling_seed_method == "not_applicable" for row in window_rows) ||
                error("Deterministic sampling method must be not_applicable")
            all(row.selection_method == "exact_logk_medoid" for row in window_rows) ||
                error("Deterministic selection method mismatch")
            if case_type == REPRESENTATIVE_CASE
                all(!row.use_for_probabilistic_uq && row.is_benchmark && !row.is_stress_test
                    for row in window_rows) || error("Representative role flags are invalid")
            else
                all(!row.use_for_probabilistic_uq && !row.is_benchmark && row.is_stress_test
                    for row in window_rows) || error("Stress role flags are invalid")
            end
        end
    end
    return true
end

function validate_seed_uniqueness(rows::Vector{ManifestRow})
    independent = [row for row in rows if row.case_type == INDEPENDENT_CASE && row.slice_id == 1]
    keys_seen = Set{Tuple{String, String, Int, String}}()
    seeds_seen = Set{UInt64}()
    for row in independent
        key = (row.geology_id, row.phase, row.replicate_id, row.window_id)
        key in keys_seen && error("Duplicate independent seed identity: $key")
        row.sampling_seed in seeds_seen && error("Unexpected sampling-seed collision")
        push!(keys_seen, key)
        push!(seeds_seen, row.sampling_seed)
    end
    return true
end

"""Validate all expected Phase 1 cases and role counts."""
function validate_phase1_manifest(rows::Vector{ManifestRow},
                                  libraries::Dict{String, PredictLibrary},
                                  cfg::SamplingConfig)
    all(row.phase == "phase1" for row in rows) || error("Phase 1 manifest contains another phase")
    grouped = group_manifest_by_case(rows)
    expected_count = cfg.phase1_n_independent +
                     Int(cfg.phase1_include_representative) +
                     Int(cfg.phase1_include_low_stress) +
                     Int(cfg.phase1_include_high_stress)
    length(grouped) == expected_count || error("Unexpected Phase 1 case count")
    for case_rows in values(grouped)
        validate_case_manifest(case_rows, libraries, cfg)
    end
    independent_ids = sort(unique(row.replicate_id for row in rows
                                  if row.case_type == INDEPENDENT_CASE))
    independent_ids == collect(1:cfg.phase1_n_independent) ||
        error("Phase 1 independent replicate IDs are incomplete")
    count(case_rows -> first(case_rows).case_type == REPRESENTATIVE_CASE, values(grouped)) ==
        Int(cfg.phase1_include_representative) || error("Representative case count mismatch")
    count(case_rows -> first(case_rows).case_type == LOW_STRESS_CASE, values(grouped)) ==
        Int(cfg.phase1_include_low_stress) || error("Low-stress case count mismatch")
    count(case_rows -> first(case_rows).case_type == HIGH_STRESS_CASE, values(grouped)) ==
        Int(cfg.phase1_include_high_stress) || error("High-stress case count mismatch")
    validate_seed_uniqueness(rows)
    return true
end

"""Validate the additional Phase 2 independent cases and continued numbering."""
function validate_phase2_manifest(rows::Vector{ManifestRow},
                                  libraries::Dict{String, PredictLibrary},
                                  cfg::SamplingConfig)
    all(row.phase == "phase2" for row in rows) || error("Phase 2 manifest contains another phase")
    all(row.case_type == INDEPENDENT_CASE for row in rows) ||
        error("Phase 2 may contain only independent cases")
    grouped = group_manifest_by_case(rows)
    length(grouped) == cfg.phase2_n_additional_independent || error("Unexpected Phase 2 case count")
    for case_rows in values(grouped)
        validate_case_manifest(case_rows, libraries, cfg)
    end
    first_id = cfg.phase1_n_independent + 1
    last_id = cfg.phase1_n_independent + cfg.phase2_n_additional_independent
    replicate_ids = sort(unique(row.replicate_id for row in rows))
    replicate_ids == collect(first_id:last_id) || error("Phase 2 replicate IDs do not continue Phase 1")
    validate_seed_uniqueness(rows)
    return true
end
