function write_manifest_csv(path::AbstractString,
                            rows::Vector{ManifestRow},
                            cfg::SamplingConfig)
    mkpath(dirname(path))
    write(path, canonical_manifest_bytes(rows, cfg))
    return path
end

function strict_bool(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized == "true" && return true
    normalized == "false" && return false
    error("Expected true or false, received '$value'")
end

"""Read a frozen manifest CSV back into typed rows."""
function read_manifest_csv(path::AbstractString)
    raw_rows = read_csv_dicts(path)
    rows = ManifestRow[]
    for row in raw_rows
        push!(rows, ManifestRow(
            schema_version = required_csv_field(row, "schema_version"),
            design_version = required_csv_field(row, "design_version"),
            geology_id = required_csv_field(row, "geology_id"),
            phase = required_csv_field(row, "phase"),
            case_id = required_csv_field(row, "case_id"),
            case_type = required_csv_field(row, "case_type"),
            replicate_id = parse(Int, required_csv_field(row, "replicate_id")),
            window_id = required_csv_field(row, "window_id"),
            slice_id = parse(Int, required_csv_field(row, "slice_id")),
            source_library = required_csv_field(row, "source_library"),
            source_library_row = parse(Int, required_csv_field(row, "source_library_row")),
            predict_realization_id = parse(Int, required_csv_field(row, "predict_realization_id")),
            predict_seed = parse(Int, required_csv_field(row, "predict_seed")),
            exact_replay_seed = parse(Int, required_csv_field(row, "exact_replay_seed")),
            sampling_seed = parse(UInt64, required_csv_field(row, "sampling_seed")),
            sampling_seed_method = required_csv_field(row, "sampling_seed_method"),
            selection_method = required_csv_field(row, "selection_method"),
            checkpoint_path = required_csv_field(row, "checkpoint_path"),
            checkpoint_relative_path = required_csv_field(row, "checkpoint_relative_path"),
            checkpoint_hash = required_csv_field(row, "checkpoint_hash"),
            level2_state_path = required_csv_field(row, "level2_state_path"),
            code_commit = required_csv_field(row, "code_commit"),
            code_dirty = strict_bool(required_csv_field(row, "code_dirty")),
            method_config_hash = required_csv_field(row, "method_config_hash"),
            predict_code_commit = required_csv_field(row, "predict_code_commit"),
            predict_method_config_hash = required_csv_field(row, "predict_method_config_hash"),
            log_kxx = parse(Float64, required_csv_field(row, "log_kxx")),
            log_kyy = parse(Float64, required_csv_field(row, "log_kyy")),
            log_kzz = parse(Float64, required_csv_field(row, "log_kzz")),
            perm_kxx_md = parse(Float64, required_csv_field(row, "perm_kxx_md")),
            perm_kyy_md = parse(Float64, required_csv_field(row, "perm_kyy_md")),
            perm_kzz_md = parse(Float64, required_csv_field(row, "perm_kzz_md")),
            use_for_probabilistic_uq = strict_bool(required_csv_field(row, "use_for_probabilistic_uq")),
            is_benchmark = strict_bool(required_csv_field(row, "is_benchmark")),
            is_stress_test = strict_bool(required_csv_field(row, "is_stress_test")),
        ))
    end
    return rows
end

function case_catalog_rows(rows::Vector{ManifestRow})
    grouped = group_manifest_by_case(rows)
    ordered = sort(collect(values(grouped)); by = case_rows -> (
        phase_order(first(case_rows).phase),
        case_type_order(first(case_rows).case_type),
        first(case_rows).replicate_id,
    ))
    return [[
        first(case_rows).geology_id,
        first(case_rows).phase,
        first(case_rows).case_id,
        first(case_rows).case_type,
        first(case_rows).replicate_id,
        length(case_rows),
        first(case_rows).use_for_probabilistic_uq,
        first(case_rows).is_benchmark,
        first(case_rows).is_stress_test,
    ] for case_rows in ordered]
end

function library_catalog_rows(libraries::Dict{String, PredictLibrary}, cfg::SamplingConfig)
    rows = Vector{Vector{Any}}()
    for window in cfg.windows
        library = libraries[window]
        push!(rows, Any[
            library.geology_id,
            library.window_id,
            n_accepted(library),
            portable_path(library.checkpoint_path),
            library.checkpoint_relative_path,
            library.checkpoint_hash,
            portable_path(library.level2_state_path),
            library.full_medoid_row,
            library.realization_id[library.full_medoid_row],
            library.predict_seed[library.full_medoid_row],
            length(library.low_library_rows),
            library.low_medoid_row,
            library.realization_id[library.low_medoid_row],
            library.predict_seed[library.low_medoid_row],
            length(library.high_library_rows),
            library.high_medoid_row,
            library.realization_id[library.high_medoid_row],
            library.predict_seed[library.high_medoid_row],
            library.chosen_k,
            repr(library.best_silhouette),
            library.stored_full_medoid_row == library.full_medoid_row,
            library.stored_low_medoid_row == library.low_medoid_row,
            library.stored_high_medoid_row == library.high_medoid_row,
        ])
    end
    return rows
end

function effective_config_dict(cfg::SamplingConfig)
    return Dict{String, Any}(
        "schema_version" => 1,
        "design_version" => cfg.design_version,
        "manifest_schema_version" => cfg.manifest_schema_version,
        "configuration_hash" => cfg.configuration_hash,
        "code_commit" => cfg.code_commit,
        "code_dirty" => cfg.code_dirty,
        "paths" => Dict(
            "geology_catalog" => portable_path(cfg.geology_catalog_path),
            "library_catalog" => portable_path(cfg.library_catalog_path),
            "level2_root" => portable_path(cfg.level2_root),
            "predict_data_root" => portable_path(cfg.predict_data_root),
            "predict_hash_inventory" => portable_path(cfg.predict_hash_inventory_path),
            "output_root" => portable_path(cfg.output_root),
        ),
        "geometry" => Dict(
            "windows" => cfg.windows,
            "n_slices" => cfg.n_slices,
            "n_predict" => cfg.expected_n_predict,
        ),
        "phase1" => Dict(
            "n_independent" => cfg.phase1_n_independent,
            "include_representative" => cfg.phase1_include_representative,
            "include_low_stress" => cfg.phase1_include_low_stress,
            "include_high_stress" => cfg.phase1_include_high_stress,
        ),
        "phase2" => Dict(
            "n_selected_geologies" => cfg.phase2_n_selected_geologies,
            "n_additional_independent" => cfg.phase2_n_additional_independent,
        ),
        "sampling" => Dict(
            "base_seed" => Int(cfg.base_seed),
            "seed_method_version" => cfg.seed_method_version,
            "with_replacement" => cfg.sampling_with_replacement,
            "cross_window" => cfg.cross_window_assumption,
            "along_strike" => cfg.along_strike_assumption,
        ),
        "medoid" => Dict(
            "metric" => cfg.medoid_metric,
            "recompute" => cfg.recompute_medoids,
            "require_stored_match" => cfg.require_stored_medoid_match,
        ),
        "low_high" => Dict(
            "target_count" => cfg.state_target_count,
            "target_fraction" => cfg.state_target_count / cfg.expected_n_predict,
        ),
        "provenance" => Dict(
            "predict_code_commit" => cfg.predict_code_commit,
            "predict_method_config_hash" => cfg.predict_method_config_hash,
            "require_clean_code" => cfg.require_clean_code,
        ),
        "qc" => Dict(
            "bootstrap_count" => cfg.qc_bootstrap_count,
            "confidence_level" => cfg.qc_confidence_level,
            "independence_acceptance_confidence_level" =>
                cfg.qc_independence_acceptance_confidence_level,
            "energy_max_points" => cfg.qc_energy_max_points,
        ),
        "optional_arrangement_test" => Dict(
            "enabled" => cfg.optional_arrangement_test_enabled,
        ),
    )
end

"""Write one geology's manifest, catalogs, validation report, and done marker."""
function write_geology_outputs(libraries::Dict{String, PredictLibrary},
                               rows::Vector{ManifestRow},
                               cfg::SamplingConfig,
                               phase::AbstractString)
    geology_id = only(unique(row.geology_id for row in rows))
    phase in ("phase1", "phase2") || error("Unsupported phase: $phase")
    phase_root = joinpath(cfg.output_root, phase)
    manifest_path = joinpath(phase_root, "manifests", "$(geology_id)_sampling_manifest.csv")
    case_catalog_path = joinpath(phase_root, "catalogs", "$(geology_id)_case_catalog.csv")
    library_catalog_path = joinpath(cfg.output_root, "libraries", "$(geology_id)_library_catalog.csv")
    report_path = joinpath(phase_root, "qc", "$(geology_id)_manifest_validation.txt")
    done_path = joinpath(phase_root, "done", "$(geology_id).toml")

    write_manifest_csv(manifest_path, rows, cfg)
    manifest_hash = file_sha256(manifest_path)
    write_csv(case_catalog_path,
              ["geology_id", "phase", "case_id", "case_type", "replicate_id",
               "assignment_count", "use_for_probabilistic_uq", "is_benchmark", "is_stress_test"],
              case_catalog_rows(rows))
    write_csv(library_catalog_path,
              ["geology_id", "window_id", "n_accepted", "checkpoint_path",
               "checkpoint_relative_path", "checkpoint_hash", "level2_state_path",
               "full_medoid_row", "full_medoid_realization_id", "full_medoid_predict_seed",
               "low_library_count", "low_medoid_row", "low_medoid_realization_id", "low_medoid_predict_seed",
               "high_library_count", "high_medoid_row", "high_medoid_realization_id", "high_medoid_predict_seed",
               "chosen_k", "best_silhouette", "stored_full_medoid_match",
               "stored_low_medoid_match", "stored_high_medoid_match"],
              library_catalog_rows(libraries, cfg))

    mkpath(dirname(report_path))
    open(report_path, "w") do io
        println(io, "Full-fault sampling manifest validation")
        println(io, "geology_id = $geology_id")
        println(io, "phase = $phase")
        println(io, "design_version = $(cfg.design_version)")
        println(io, "manifest_schema_version = $(cfg.manifest_schema_version)")
        println(io, "case_count = $(length(group_manifest_by_case(rows)))")
        println(io, "assignment_count = $(length(rows))")
        println(io, "manifest_sha256 = $manifest_hash")
        println(io, "configuration_sha256 = $(cfg.configuration_hash)")
        println(io, "code_commit = $(cfg.code_commit)")
        println(io, "code_dirty = $(cfg.code_dirty)")
        println(io, "validation = PASS")
    end

    marker = Dict{String, Any}(
        "schema_version" => 1,
        "design_version" => cfg.design_version,
        "geology_id" => geology_id,
        "phase" => String(phase),
        "manifest_relative_path" => portable_path(relpath(manifest_path, cfg.output_root)),
        "manifest_sha256" => manifest_hash,
        "configuration_sha256" => cfg.configuration_hash,
        "code_commit" => cfg.code_commit,
        "code_dirty" => cfg.code_dirty,
        "case_count" => length(group_manifest_by_case(rows)),
        "assignment_count" => length(rows),
        "created_at" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    )
    mkpath(dirname(done_path))
    open(done_path, "w") do io
        TOML.print(io, marker; sorted = true)
    end
    return Dict(
        "manifest_path" => manifest_path,
        "manifest_hash" => manifest_hash,
        "case_catalog_path" => case_catalog_path,
        "library_catalog_path" => library_catalog_path,
        "validation_report_path" => report_path,
        "done_path" => done_path,
    )
end

"""Combine deterministic per-geology manifests without loading them into memory."""
function combine_phase_manifests(cfg::SamplingConfig,
                                 phase::AbstractString,
                                 geology_ids::Vector{String})
    combined_path = joinpath(cfg.output_root, phase, "$(phase)_sampling_manifest_all_geologies.csv")
    mkpath(dirname(combined_path))
    open(combined_path, "w") do output
        wrote_header = false
        for geology_id in sort(geology_ids; by = geology_sort_key)
            path = joinpath(cfg.output_root, phase, "manifests", "$(geology_id)_sampling_manifest.csv")
            isfile(path) || error("Cannot combine missing geology manifest: $path")
            open(path, "r") do input
                for (line_number, line) in enumerate(eachline(input))
                    if line_number == 1
                        if !wrote_header
                            println(output, line)
                            wrote_header = true
                        elseif line != join(csv_escape.(MANIFEST_HEADER), ',')
                            error("Manifest header mismatch in $path")
                        end
                    else
                        println(output, line)
                    end
                end
            end
        end
    end
    return Dict("path" => combined_path, "sha256" => file_sha256(combined_path))
end

function write_effective_config(cfg::SamplingConfig)
    path = joinpath(cfg.output_root, "config", "effective_sampling_config.toml")
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, effective_config_dict(cfg); sorted = true)
    end
    return path
end
