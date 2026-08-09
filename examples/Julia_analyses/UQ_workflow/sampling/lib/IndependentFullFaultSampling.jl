"""
    IndependentFullFaultSampling

Versioned full-fault sampling for the `independent_full_fault_v1` design.

The module builds explicit, reproducible assignment manifests. It deliberately
does not replay PREDICT or run property upscaling; those stages consume a
validated and frozen manifest later.
"""
module IndependentFullFaultSampling

using Dates
using MAT
using SHA
using Statistics
using TOML

include("sampling_types.jl")
include("sampling_provenance.jl")
include("sampling_config.jl")
include("sampling_medoids.jl")
include("sampling_libraries.jl")
include("sampling_manifests.jl")
include("sampling_validation.jl")
include("sampling_outputs.jl")
include("sampling_qc.jl")
include("sampling_mat_export.jl")

export DESIGN_VERSION,
       MANIFEST_SCHEMA_VERSION,
       SEED_METHOD_VERSION,
       SamplingConfig,
       PredictLibrary,
       CaseSpec,
       ManifestRow,
       read_sampling_config,
       load_geology_ids,
       select_requested_geologies,
       load_geology_libraries,
       exact_logk_medoid,
       stable_case_seed,
       stable_uniform_index,
       phase1_case_specs,
       phase2_case_specs,
       generate_case_manifest,
       generate_phase1_manifest,
       generate_phase2_manifest,
       validate_library,
       validate_case_manifest,
       validate_phase1_manifest,
       validate_phase2_manifest,
       write_geology_outputs,
       write_effective_config,
       write_ensemble_qc,
       read_manifest_csv,
       combine_phase_manifests,
       run_ensemble_qc,
       export_field_permeability_mat,
       validate_field_permeability_mat,
       fault_local_coordinate_contract,
       file_sha256,
       canonical_manifest_bytes

end
