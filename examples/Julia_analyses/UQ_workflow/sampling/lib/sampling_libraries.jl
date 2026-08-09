function int_vector(value, label::AbstractString)
    data = value isa Number ? [value] : vec(value)
    all(isfinite, Float64.(data)) || error("$label contains non-finite values")
    return Int.(round.(Float64.(data)))
end

function scalar_int(value, label::AbstractString)
    values = int_vector(value, label)
    length(values) == 1 || error("$label must be scalar")
    return only(values)
end

function scalar_float(value, label::AbstractString)
    data = value isa Number ? Float64(value) : Float64(only(vec(value)))
    isfinite(data) || error("$label must be finite")
    return data
end

function normalized_relative_path(path::AbstractString, root::AbstractString)
    relative = relpath(normpath(abspath(path)), normpath(abspath(root)))
    startswith(relative, "..") && error("Checkpoint lies outside PREDICT data root: $path")
    return portable_path(relative)
end

function read_checkpoint_hash_inventory(cfg::SamplingConfig)
    isempty(cfg.predict_hash_inventory_path) && return Dict{String, String}()
    rows = read_csv_dicts(cfg.predict_hash_inventory_path)
    inventory = Dict{String, String}()
    for row in rows
        relative = replace(required_csv_field(row, "relative_path"), '\\' => '/')
        digest = lowercase(required_csv_field(row, "sha256"))
        haskey(inventory, relative) && error("Duplicate hash-inventory path: $relative")
        inventory[relative] = digest
    end
    return inventory
end

"""Load and validate the complete six-window library set for one geology."""
function load_geology_libraries(cfg::SamplingConfig, geology_id::AbstractString)
    catalog_rows = [row for row in read_csv_dicts(cfg.library_catalog_path)
                    if required_csv_field(row, "geology_id") == geology_id]
    length(catalog_rows) == length(cfg.windows) ||
        error("Expected $(length(cfg.windows)) library rows for $geology_id, found $(length(catalog_rows))")
    by_window = Dict(required_csv_field(row, "window") => row for row in catalog_rows)
    Set(keys(by_window)) == Set(cfg.windows) ||
        error("Window catalog mismatch for $geology_id")
    hash_inventory = read_checkpoint_hash_inventory(cfg)

    libraries = Dict{String, PredictLibrary}()
    for window in cfg.windows
        libraries[window] = load_predict_library(cfg, by_window[window], hash_inventory)
        validate_library(libraries[window], cfg)
    end
    return libraries
end

"""Load one PREDICT checkpoint and its saved Level 2 state object."""
function load_predict_library(cfg::SamplingConfig,
                              catalog_row::Dict{String, String},
                              hash_inventory::Dict{String, String})
    geology_id = required_csv_field(catalog_row, "geology_id")
    window = required_csv_field(catalog_row, "window")
    checkpoint_path = normpath(abspath(required_csv_field(catalog_row, "mat_path")))
    isfile(checkpoint_path) || error("Missing PREDICT checkpoint: $checkpoint_path")

    state_path = joinpath(cfg.level2_root, geology_id, "window_states", window,
                          "$(window)_level2_state.mat")
    isfile(state_path) || error("Missing Level 2 state: $state_path")

    checkpoint = matread(checkpoint_path)
    state = matread(state_path)
    haskey(checkpoint, "perms") || error("Checkpoint lacks perms: $checkpoint_path")
    haskey(checkpoint, "meta") || error("Checkpoint lacks replay metadata: $checkpoint_path")
    haskey(state, "log_perms") || error("Level 2 state lacks log_perms: $state_path")
    haskey(state, "raw_perms") || error("Level 2 state lacks raw_perms: $state_path")

    K_mD = Matrix{Float64}(checkpoint["perms"])
    logK = log10.(K_mD)
    state_K = Matrix{Float64}(state["raw_perms"])
    size(state_K) == size(K_mD) || error("Level 2/checkpoint size mismatch for $geology_id $window")
    state_K == K_mD || error("Level 2 permeability rows differ from the source checkpoint")
    Matrix{Float64}(state["log_perms"]) == logK ||
        error("Level 2 log-permeability rows differ from the source checkpoint")

    String(state["geology_id"]) == geology_id || error("Level 2 geology ID mismatch")
    String(state["window"]) == window || error("Level 2 window mismatch")
    same_path(String(state["source_path"]), checkpoint_path) ||
        error("Level 2 source_path does not match catalog checkpoint")

    meta = checkpoint["meta"]
    meta isa AbstractDict || error("Checkpoint meta must be a dictionary")
    Bool(get(meta, "AcceptedSeedTracking", get(get(checkpoint, "checkpointInfo", Dict()),
                                                "AcceptedSeedTracking", false))) ||
        error("Accepted seed tracking is required for exact replay: $checkpoint_path")
    haskey(meta, "AcceptedSeeds") || error("Checkpoint lacks AcceptedSeeds")
    haskey(meta, "AcceptedAttemptIndices") || error("Checkpoint lacks AcceptedAttemptIndices")
    predict_seed = int_vector(meta["AcceptedSeeds"], "AcceptedSeeds")
    realization_id = int_vector(meta["AcceptedAttemptIndices"], "AcceptedAttemptIndices")
    n = size(K_mD, 1)
    length(predict_seed) == n || error("AcceptedSeeds length mismatch")
    length(realization_id) == n || error("AcceptedAttemptIndices length mismatch")

    checkpoint_relative_path = normalized_relative_path(checkpoint_path, cfg.predict_data_root)
    checkpoint_hash = file_sha256(checkpoint_path)
    if !isempty(hash_inventory)
        haskey(hash_inventory, checkpoint_relative_path) ||
            error("Checkpoint is absent from the frozen hash inventory: $checkpoint_relative_path")
        hash_inventory[checkpoint_relative_path] == checkpoint_hash ||
            error("Checkpoint SHA-256 mismatch: $checkpoint_relative_path")
    end

    low_rows = int_vector(state["low_indices"], "low_indices")
    high_rows = int_vector(state["high_indices"], "high_indices")
    stored_full = scalar_int(state["global_medoid_index"], "global_medoid_index")
    stored_low = scalar_int(state["low_medoid_index"], "low_medoid_index")
    stored_high = scalar_int(state["high_medoid_index"], "high_medoid_index")

    if cfg.recompute_medoids
        full_medoid = exact_logk_medoid(logK, realization_id)
        low_medoid = exact_logk_medoid(logK, low_rows, realization_id)
        high_medoid = exact_logk_medoid(logK, high_rows, realization_id)
    else
        full_medoid, low_medoid, high_medoid = stored_full, stored_low, stored_high
    end

    if cfg.require_stored_medoid_match
        full_medoid == stored_full || error("Recomputed full medoid differs from Level 2 for $geology_id $window")
        low_medoid == stored_low || error("Recomputed low medoid differs from Level 2 for $geology_id $window")
        high_medoid == stored_high || error("Recomputed high medoid differs from Level 2 for $geology_id $window")
    end

    library = PredictLibrary(
        geology_id,
        window,
        K_mD,
        logK,
        realization_id,
        predict_seed,
        trues(n),
        checkpoint_path,
        checkpoint_hash,
        checkpoint_relative_path,
        state_path,
        cfg.predict_code_commit,
        cfg.predict_method_config_hash,
        full_medoid,
        sort!(unique(low_rows)),
        sort!(unique(high_rows)),
        low_medoid,
        high_medoid,
        Float64.(vec(state["joint_rank_score"])),
        int_vector(state["cluster_assignments"], "cluster_assignments"),
        scalar_int(state["chosen_k"], "chosen_k"),
        scalar_float(state["best_silhouette"], "best_silhouette"),
        stored_full,
        stored_low,
        stored_high,
    )
    return library
end

"""Hard validation of one loaded PREDICT/Level 2 library."""
function validate_library(library::PredictLibrary, cfg::SamplingConfig)
    n = n_accepted(library)
    n == cfg.expected_n_predict ||
        error("$(library.geology_id) $(library.window_id) has $n accepted rows; expected $(cfg.expected_n_predict)")
    size(library.K_mD) == (n, 3) || error("K_mD must be n x 3")
    size(library.logK) == (n, 3) || error("logK must be n x 3")
    all(isfinite, library.K_mD) || error("K_mD contains non-finite values")
    all(isfinite, library.logK) || error("logK contains non-finite values")
    all(>(0), library.K_mD) || error("Permeability values must be positive")
    length(library.realization_id) == n || error("realization_id length mismatch")
    length(library.predict_seed) == n || error("predict_seed length mismatch")
    length(library.accepted_flag) == n || error("accepted_flag length mismatch")
    all(library.accepted_flag) || error("Library contains unaccepted rows")
    length(unique(library.realization_id)) == n || error("PREDICT realization IDs must be unique")
    length(unique(library.predict_seed)) == n || error("PREDICT replay seeds must be unique")
    length(library.low_library_rows) == cfg.state_target_count ||
        error("Low library must contain $(cfg.state_target_count) rows")
    length(library.high_library_rows) == cfg.state_target_count ||
        error("High library must contain $(cfg.state_target_count) rows")
    isempty(intersect(library.low_library_rows, library.high_library_rows)) ||
        error("Low and high libraries must not overlap")
    library.low_medoid_row in library.low_library_rows || error("Low medoid is outside low library")
    library.high_medoid_row in library.high_library_rows || error("High medoid is outside high library")
    1 <= library.full_medoid_row <= n || error("Full medoid is out of range")
    length(library.joint_rank_score) == n || error("joint_rank_score length mismatch")
    length(library.cluster_assignments) == n || error("cluster_assignments length mismatch")
    return true
end
