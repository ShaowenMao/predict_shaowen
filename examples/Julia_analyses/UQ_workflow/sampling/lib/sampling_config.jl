const DEFAULT_WINDOWS = ["famp1", "famp2", "famp3", "famp4", "famp5", "famp6"]

function resolve_config_path(config_path::AbstractString, path::AbstractString)
    isempty(strip(path)) && return ""
    return isabspath(path) ? normpath(path) : normpath(joinpath(dirname(config_path), path))
end

"""
    read_sampling_config(path; output_root_override="")

Read, normalize, and validate the `independent_full_fault_v1` configuration.
The configuration hash is computed from the effective TOML values after any
explicit output-root override is applied.
"""
function read_sampling_config(path::AbstractString;
                              output_root_override::AbstractString = "")
    config_path = normpath(abspath(path))
    isfile(config_path) || error("Sampling configuration does not exist: $config_path")
    raw = TOML.parsefile(config_path)

    design = get(raw, "design", Dict{String, Any}())
    paths = get(raw, "paths", Dict{String, Any}())
    geometry = get(raw, "geometry", Dict{String, Any}())
    phase1 = get(raw, "phase1", Dict{String, Any}())
    phase2 = get(raw, "phase2", Dict{String, Any}())
    sampling = get(raw, "sampling", Dict{String, Any}())
    medoid = get(raw, "medoid", Dict{String, Any}())
    low_high = get(raw, "low_high", Dict{String, Any}())
    provenance = get(raw, "provenance", Dict{String, Any}())
    qc = get(raw, "qc", Dict{String, Any}())
    optional_arrangement = get(raw, "optional_arrangement_test", Dict{String, Any}())

    design_version = String(get(design, "version", DESIGN_VERSION))
    design_version == DESIGN_VERSION ||
        error("Unsupported sampling design '$design_version'; expected '$DESIGN_VERSION'")

    repo_root = resolve_config_path(config_path, String(get(paths, "repo_root", "../../../..")))
    geology_catalog_path = resolve_config_path(config_path, String(get(paths, "geology_catalog", "")))
    library_catalog_path = resolve_config_path(config_path, String(get(paths, "library_catalog", "")))
    level2_root = resolve_config_path(config_path, String(get(paths, "level2_root", "")))
    predict_data_root = resolve_config_path(config_path, String(get(paths, "predict_data_root", "")))
    predict_hash_inventory_path = resolve_config_path(
        config_path, String(get(paths, "predict_hash_inventory", "")))
    output_root = isempty(output_root_override) ?
        resolve_config_path(config_path, String(get(paths, "output_root", "sampling_outputs"))) :
        normpath(abspath(output_root_override))

    for (label, required_path) in [
        ("geology catalog", geology_catalog_path),
        ("library catalog", library_catalog_path),
        ("Level 2 root", level2_root),
        ("PREDICT data root", predict_data_root),
    ]
        isempty(required_path) && error("A path for $label is required")
    end

    effective_raw = deepcopy(raw)
    effective_paths = get!(effective_raw, "paths", Dict{String, Any}())
    effective_paths["output_root"] = portable_path(output_root)
    config_hash = configuration_sha256(effective_raw)

    git = git_provenance(repo_root)
    require_clean_code = Bool(get(provenance, "require_clean_code", true))
    require_clean_code && git.dirty &&
        error("A clean Git worktree is required for production manifest generation")

    windows = String.(get(geometry, "windows", DEFAULT_WINDOWS))
    length(unique(windows)) == length(windows) || error("Window labels must be unique")

    cfg = SamplingConfig(
        config_path = config_path,
        configuration_hash = config_hash,
        design_version = design_version,
        repo_root = repo_root,
        code_commit = git.commit,
        code_dirty = git.dirty,
        require_clean_code = require_clean_code,
        geology_catalog_path = geology_catalog_path,
        library_catalog_path = library_catalog_path,
        level2_root = level2_root,
        predict_data_root = predict_data_root,
        predict_hash_inventory_path = predict_hash_inventory_path,
        output_root = output_root,
        windows = windows,
        n_slices = Int(get(geometry, "n_slices", 87)),
        expected_n_predict = Int(get(geometry, "n_predict", 2000)),
        phase1_n_independent = Int(get(phase1, "n_independent", 12)),
        phase1_include_representative = Bool(get(phase1, "include_representative", true)),
        phase1_include_low_stress = Bool(get(phase1, "include_low_stress", true)),
        phase1_include_high_stress = Bool(get(phase1, "include_high_stress", true)),
        phase2_n_selected_geologies = Int(get(phase2, "n_selected_geologies", 12)),
        phase2_n_additional_independent = Int(get(phase2, "n_additional_independent", 40)),
        base_seed = UInt64(get(sampling, "base_seed", 1729)),
        seed_method_version = String(get(sampling, "seed_method_version", SEED_METHOD_VERSION)),
        sampling_with_replacement = Bool(get(sampling, "with_replacement", true)),
        cross_window_assumption = String(get(sampling, "cross_window", "conditionally_independent")),
        along_strike_assumption = String(get(sampling, "along_strike", "conditionally_independent")),
        medoid_metric = String(get(medoid, "metric", "euclidean_log10k_3d")),
        state_target_count = Int(get(low_high, "target_count", 400)),
        predict_code_commit = String(get(provenance, "predict_code_commit", "")),
        predict_method_config_hash = String(get(provenance, "predict_method_config_hash", "")),
        recompute_medoids = Bool(get(medoid, "recompute", true)),
        require_stored_medoid_match = Bool(get(medoid, "require_stored_match", true)),
        qc_bootstrap_count = Int(get(qc, "bootstrap_count", 100)),
        qc_confidence_level = Float64(get(qc, "confidence_level", 0.95)),
        qc_independence_acceptance_confidence_level = Float64(get(
            qc, "independence_acceptance_confidence_level", 0.99)),
        qc_energy_max_points = Int(get(qc, "energy_max_points", 400)),
        optional_arrangement_test_enabled = Bool(get(
            optional_arrangement, "enabled", false)),
    )
    validate_sampling_config(cfg)
    return cfg
end

function validate_sampling_config(cfg::SamplingConfig)
    cfg.design_version == DESIGN_VERSION || error("Unexpected design version")
    cfg.n_slices > 0 || error("n_slices must be positive")
    cfg.expected_n_predict > 0 || error("n_predict must be positive")
    cfg.phase1_n_independent > 0 || error("Phase 1 requires at least one independent case")
    cfg.phase2_n_selected_geologies > 0 || error("Phase 2 geology count must be positive")
    cfg.phase2_n_additional_independent > 0 || error("Phase 2 additional count must be positive")
    cfg.sampling_with_replacement || error("$DESIGN_VERSION requires sampling with replacement")
    cfg.seed_method_version == SEED_METHOD_VERSION || error("Unsupported sampling seed method")
    cfg.cross_window_assumption == "conditionally_independent" ||
        error("Cross-window sampling must be conditionally_independent")
    cfg.along_strike_assumption == "conditionally_independent" ||
        error("Along-strike sampling must be conditionally_independent")
    cfg.medoid_metric == "euclidean_log10k_3d" || error("Unsupported medoid metric")
    5 * cfg.state_target_count == cfg.expected_n_predict ||
        error("Low/high target count must equal 20% of n_predict")
    occursin(r"^[0-9a-fA-F]{40}$", cfg.predict_code_commit) ||
        error("predict_code_commit must be a 40-character hexadecimal commit")
    occursin(r"^[0-9a-fA-F]{64}$", cfg.predict_method_config_hash) ||
        error("predict_method_config_hash must be a 64-character hexadecimal SHA-256")
    0 < cfg.qc_confidence_level < 1 || error("QC confidence level must lie in (0, 1)")
    cfg.qc_confidence_level <= cfg.qc_independence_acceptance_confidence_level < 1 ||
        error("Independence acceptance confidence must lie in [confidence_level, 1)")
    cfg.qc_bootstrap_count >= 0 || error("QC bootstrap count cannot be negative")
    cfg.qc_energy_max_points >= 0 || error("QC energy_max_points cannot be negative")
    !cfg.optional_arrangement_test_enabled ||
        error("The optional arrangement stress test is not implemented in the primary workflow")
    return cfg
end

"""Load ordered geology IDs from the Level 1 geology catalog."""
function load_geology_ids(cfg::SamplingConfig)
    rows = read_csv_dicts(cfg.geology_catalog_path)
    ids = String[]
    for row in rows
        id = required_csv_field(row, "geology_id")
        if haskey(row, "all_window_files_present")
            parse_bool(row["all_window_files_present"]) ||
                error("Geology $id is missing one or more PREDICT window libraries")
        end
        push!(ids, id)
    end
    length(unique(ids)) == length(ids) || error("Duplicate geology IDs in catalog")
    return sort(ids; by = geology_sort_key)
end

"""
    select_requested_geologies(cfg; phase, explicit_ids, deep_dive_csv, max_geologies)

Resolve a requested geology subset against the frozen Level 1 catalog. Phase 2
requires exactly the configured number of unique geologies, regardless of
whether they are provided by CSV or directly on the command line.
"""
function select_requested_geologies(cfg::SamplingConfig;
                                     phase::AbstractString = "phase1",
                                     explicit_ids::AbstractString = "",
                                     deep_dive_csv::AbstractString = "",
                                     max_geologies::Integer = 0)
    phase in ("phase1", "phase2") || error("Unsupported phase: $phase")
    max_geologies >= 0 || error("max_geologies cannot be negative")
    catalog_ids = load_geology_ids(cfg)
    catalog_set = Set(catalog_ids)
    explicit = strip(explicit_ids)
    deep_dive = strip(deep_dive_csv)
    !isempty(explicit) && !isempty(deep_dive) &&
        error("Use either explicit geology IDs or a deep-dive CSV, not both")
    phase == "phase1" && !isempty(deep_dive) &&
        error("A deep-dive CSV is valid only for Phase 2")

    selected_ids = copy(catalog_ids)
    if !isempty(explicit)
        requested = [strip(part) for part in split(explicit, ',') if !isempty(strip(part))]
        length(unique(requested)) == length(requested) ||
            error("Requested geology IDs must be unique")
        unknown = setdiff(Set(requested), catalog_set)
        isempty(unknown) ||
            error("Unknown requested geology IDs: $(join(sort!(collect(unknown)), ", "))")
        requested_set = Set(requested)
        selected_ids = [id for id in catalog_ids if id in requested_set]
    elseif !isempty(deep_dive)
        rows = read_csv_dicts(normpath(abspath(deep_dive)))
        requested = [required_csv_field(row, "geology_id") for row in rows]
        length(unique(requested)) == length(requested) ||
            error("Phase 2 deep-dive geology IDs must be unique")
        unknown = setdiff(Set(requested), catalog_set)
        isempty(unknown) ||
            error("Unknown Phase 2 geology IDs: $(join(sort!(collect(unknown)), ", "))")
        requested_set = Set(requested)
        selected_ids = [id for id in catalog_ids if id in requested_set]
    elseif phase == "phase2"
        error("Phase 2 requires a deep-dive CSV or explicit geology IDs")
    end

    max_geologies > 0 &&
        (selected_ids = selected_ids[1:min(max_geologies, length(selected_ids))])
    isempty(selected_ids) && error("No geologies selected")
    if phase == "phase2"
        length(selected_ids) == cfg.phase2_n_selected_geologies ||
            error("Phase 2 requires exactly $(cfg.phase2_n_selected_geologies) geologies")
    end
    return selected_ids
end
