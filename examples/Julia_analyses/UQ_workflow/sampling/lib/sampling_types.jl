const DESIGN_VERSION = "independent_full_fault_v1"
const MANIFEST_SCHEMA_VERSION = "full_fault_sampling_manifest_v1"
const SEED_METHOD_VERSION = "sha256_identity_u63_counter_u64_rejection_v1"

"""Normalized configuration for the revised full-fault sampling design."""
Base.@kwdef struct SamplingConfig
    config_path::String
    configuration_hash::String
    design_version::String = DESIGN_VERSION
    manifest_schema_version::String = MANIFEST_SCHEMA_VERSION
    repo_root::String
    code_commit::String
    code_dirty::Bool
    require_clean_code::Bool = true
    geology_catalog_path::String
    library_catalog_path::String
    level2_root::String
    predict_data_root::String
    predict_hash_inventory_path::String = ""
    output_root::String
    windows::Vector{String}
    n_slices::Int = 87
    expected_n_predict::Int = 2000
    phase1_n_independent::Int = 12
    phase1_include_representative::Bool = true
    phase1_include_low_stress::Bool = true
    phase1_include_high_stress::Bool = true
    phase2_n_selected_geologies::Int = 12
    phase2_n_additional_independent::Int = 40
    base_seed::UInt64 = UInt64(1729)
    seed_method_version::String = SEED_METHOD_VERSION
    sampling_with_replacement::Bool = true
    cross_window_assumption::String = "conditionally_independent"
    along_strike_assumption::String = "conditionally_independent"
    medoid_metric::String = "euclidean_log10k_3d"
    state_target_count::Int = 400
    predict_code_commit::String
    predict_method_config_hash::String
    recompute_medoids::Bool = true
    require_stored_medoid_match::Bool = true
    qc_bootstrap_count::Int = 100
    qc_confidence_level::Float64 = 0.95
    qc_independence_acceptance_confidence_level::Float64 = 0.99
    qc_energy_max_points::Int = 400
    optional_arrangement_test_enabled::Bool = false
end

"""
One accepted PREDICT library and its Level 2 interpretation for a geology-window.

Rows are one-based and refer to the accepted PREDICT `perms` matrix. The
realization ID records the original accepted attempt index, while
`predict_seed` is the exact replay seed stored by PREDICT.
"""
struct PredictLibrary
    geology_id::String
    window_id::String
    K_mD::Matrix{Float64}
    logK::Matrix{Float64}
    realization_id::Vector{Int}
    predict_seed::Vector{Int}
    accepted_flag::BitVector
    checkpoint_path::String
    checkpoint_hash::String
    checkpoint_relative_path::String
    level2_state_path::String
    predict_code_commit::String
    predict_method_config_hash::String
    full_medoid_row::Int
    low_library_rows::Vector{Int}
    high_library_rows::Vector{Int}
    low_medoid_row::Int
    high_medoid_row::Int
    joint_rank_score::Vector{Float64}
    cluster_assignments::Vector{Int}
    chosen_k::Int
    best_silhouette::Float64
    stored_full_medoid_row::Int
    stored_low_medoid_row::Int
    stored_high_medoid_row::Int
end

n_accepted(library::PredictLibrary) = size(library.K_mD, 1)

"""One case to generate for a geology."""
struct CaseSpec
    phase::String
    case_id::String
    case_type::String
    replicate_id::Int
    use_for_probabilistic_uq::Bool
    is_benchmark::Bool
    is_stress_test::Bool
end

"""One window-slice assignment in the frozen sampling manifest."""
Base.@kwdef struct ManifestRow
    schema_version::String = MANIFEST_SCHEMA_VERSION
    design_version::String = DESIGN_VERSION
    geology_id::String
    phase::String
    case_id::String
    case_type::String
    replicate_id::Int
    window_id::String
    slice_id::Int
    source_library::String
    source_library_row::Int
    predict_realization_id::Int
    predict_seed::Int
    exact_replay_seed::Int
    sampling_seed::UInt64
    sampling_seed_method::String
    selection_method::String
    checkpoint_path::String
    checkpoint_relative_path::String
    checkpoint_hash::String
    level2_state_path::String
    code_commit::String
    code_dirty::Bool
    method_config_hash::String
    predict_code_commit::String
    predict_method_config_hash::String
    log_kxx::Float64
    log_kyy::Float64
    log_kzz::Float64
    perm_kxx_md::Float64
    perm_kyy_md::Float64
    perm_kzz_md::Float64
    use_for_probabilistic_uq::Bool
    is_benchmark::Bool
    is_stress_test::Bool
end

const MANIFEST_HEADER = [
    "schema_version",
    "design_version",
    "geology_id",
    "phase",
    "case_id",
    "case_type",
    "replicate_id",
    "window_id",
    "slice_id",
    "source_library",
    "source_library_row",
    "predict_realization_id",
    "predict_seed",
    "exact_replay_seed",
    "sampling_seed",
    "sampling_seed_method",
    "selection_method",
    "checkpoint_path",
    "checkpoint_relative_path",
    "checkpoint_hash",
    "level2_state_path",
    "code_commit",
    "code_dirty",
    "method_config_hash",
    "predict_code_commit",
    "predict_method_config_hash",
    "log_kxx",
    "log_kyy",
    "log_kzz",
    "perm_kxx_md",
    "perm_kyy_md",
    "perm_kzz_md",
    "use_for_probabilistic_uq",
    "is_benchmark",
    "is_stress_test",
]
