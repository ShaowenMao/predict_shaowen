#!/usr/bin/env bash
# Plan, submit, or continue one independent-full-fault production phase.

set -euo pipefail

ACTION="${1:-plan}"
PHASE="${2:-phase1}"
if [[ "${ACTION}" != "plan" && "${ACTION}" != "submit" && "${ACTION}" != "continue" ]]; then
    echo "Usage: $0 plan|submit|continue phase1|phase2" >&2
    exit 2
fi
if [[ "${PHASE}" != "phase1" && "${PHASE}" != "phase2" ]]; then
    echo "Usage: $0 plan|submit|continue phase1|phase2" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
ORCHESTRATION_COMMIT="${ORCHESTRATION_COMMIT:-}"
PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT:?PREDICT_CODE_ROOT must be a clean checkout of the recorded PREDICT physics commit}"
HANDOFF_ROOT="${HANDOFF_ROOT:?HANDOFF_ROOT is required}"
PREDICT_ROOT="${PREDICT_ROOT:?PREDICT_ROOT is required}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT:-/tmp/${USER}/predict_shaowen}"
CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/checkpoint}"
MATLAB_SHORT_ROOT="${MATLAB_SHORT_ROOT:-${NODE_LOCAL_TMP_ROOT}/matlab}"
CASE_TEMP_ROOT="${CASE_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/case}"
RUN_ID_INPUT="${RUN_ID:-}"
RUN_ID="${RUN_ID:-independent_full_fault_v1_${PHASE}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DATA_ROOT}/production_runs/${RUN_ID}}"
FREEZE_ROOT="${HANDOFF_ROOT}/replay_upscaling_freeze"
SAMPLING_ROOT="${FREEZE_ROOT}/sampling"
SAMPLING_CSV="${SAMPLING_ROOT}/texas_field_slice_window_values.csv"
PERMEABILITY_INPUT="${SAMPLING_ROOT}/fault_permeability_independent_full_fault_v1.mat"
METHOD_CONFIG="${FREEZE_ROOT}/config/production_method_config.toml"
CHECKPOINT_MANIFEST_ROOT="${RUN_ROOT}/checkpoint_manifest"
CHECKPOINT_OUTPUT_ROOT="${RUN_ROOT}/checkpoint_pc"
CASE_WORK_ROOT="${RUN_ROOT}/case_work_manifest"
CASE_INPUT_ROOT="${RUN_ROOT}/case_inputs"
CASE_RESULT_ROOT="${RUN_ROOT}/case_results"
LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}"
STATUS_TOOL="${SCRIPT_DIR}/phase_production_status.py"
PRODUCTION_DIR="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production"

if [[ "${ACTION}" != "plan" && -z "${RUN_ID_INPUT}" ]]; then
    echo "Set an explicit versioned RUN_ID before submission." >&2
    exit 2
fi

SLURM_ACCOUNT="${SLURM_ACCOUNT:-mit_amf_advanced_cpu}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"
SLURM_PARTITION="${SLURM_PARTITION:-mit_normal}"
CHECKPOINT_MAX_CONCURRENT="${CHECKPOINT_MAX_CONCURRENT:-96}"
ASSEMBLY_MAX_CONCURRENT="${ASSEMBLY_MAX_CONCURRENT:-24}"
KR_MAX_CONCURRENT="${KR_MAX_CONCURRENT:-48}"
CHECKPOINT_WALLTIME="${CHECKPOINT_WALLTIME:-24:00:00}"
ASSEMBLY_WALLTIME="${ASSEMBLY_WALLTIME:-04:00:00}"
KR_WALLTIME="${KR_WALLTIME:-24:00:00}"
CHECKPOINT_MEMORY="${CHECKPOINT_MEMORY:-18G}"
ASSEMBLY_MEMORY="${ASSEMBLY_MEMORY:-16G}"
KR_MEMORY="${KR_MEMORY:-48G}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-0.005}"
SLURM_MAX_SUBMITTED_JOBS="${SLURM_MAX_SUBMITTED_JOBS:-400}"
EXTERNAL_CHECKPOINT_JOB_ID="${EXTERNAL_CHECKPOINT_JOB_ID:-}"
if [[ -n "${EXTERNAL_CHECKPOINT_JOB_ID}" && ! "${EXTERNAL_CHECKPOINT_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "EXTERNAL_CHECKPOINT_JOB_ID must be a numeric Slurm job ID." >&2
    exit 2
fi

if [[ "${PHASE}" == "phase1" ]]; then
    EXPECTED_GEOLOGIES=162
    EXPECTED_CASES=2430
    EXPECTED_ASSIGNMENTS=1268460
    GROUPS_PER_ARRAY_TASK="${GROUPS_PER_ARRAY_TASK:-5}"
    GEOLOGIES_PER_ARRAY_TASK="${GEOLOGIES_PER_ARRAY_TASK:-9}"
    CASES_PER_ARRAY_TASK="${CASES_PER_ARRAY_TASK:-16}"
else
    EXPECTED_GEOLOGIES=12
    EXPECTED_CASES=480
    EXPECTED_ASSIGNMENTS=250560
    GROUPS_PER_ARRAY_TASK="${GROUPS_PER_ARRAY_TASK:-1}"
    GEOLOGIES_PER_ARRAY_TASK="${GEOLOGIES_PER_ARRAY_TASK:-6}"
    CASES_PER_ARRAY_TASK="${CASES_PER_ARRAY_TASK:-8}"
fi

for required in \
    "${HANDOFF_ROOT}/handoff_metadata.json" \
    "${FREEZE_ROOT}/freeze_metadata.json" \
    "${SAMPLING_CSV}" \
    "${PERMEABILITY_INPUT}" \
    "${METHOD_CONFIG}" \
    "${STATUS_TOOL}"; do
    [[ -f "${required}" ]] || {
        echo "Missing required production input: ${required}" >&2
        exit 2
    }
done
[[ -d "${PREDICT_ROOT}" ]] || {
    echo "Missing PREDICT checkpoint root: ${PREDICT_ROOT}" >&2
    exit 2
}

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
python3 \
    "${RUNTIME_REPO}/examples/Julia_analyses/UQ_workflow/sampling/integration/verify_phase_handoff.py" \
    --handoff-root "${HANDOFF_ROOT}" \
    --phase "${PHASE}" \
    --expected-geologies "${EXPECTED_GEOLOGIES}" \
    --expected-cases "${EXPECTED_CASES}" \
    --expected-assignments "${EXPECTED_ASSIGNMENTS}" \
    --output-json "${RUN_ROOT}/handoff_verification.json"

read -r SAMPLING_COMMIT PHYSICS_COMMIT METHOD_CONFIG_SHA256 HANDOFF_SHA256 < <(
    python3 - \
        "${FREEZE_ROOT}/freeze_metadata.json" \
        "${HANDOFF_ROOT}/handoff_metadata.json" <<'PY'
import hashlib
import json
import sys

freeze = json.load(open(sys.argv[1], encoding="utf-8"))
handoff_path = sys.argv[2]
digest = hashlib.sha256(open(handoff_path, "rb").read()).hexdigest()
print(
    freeze["sampling_code_commit"],
    freeze["physics_commit"],
    freeze["production_method_config_sha256"],
    digest,
)
PY
)

runtime_commit="$(git -C "${RUNTIME_REPO}" rev-parse HEAD)"
ORCHESTRATION_COMMIT="${ORCHESTRATION_COMMIT:-${SAMPLING_COMMIT}}"
if [[ "${runtime_commit}" != "${ORCHESTRATION_COMMIT}" ]]; then
    echo "Orchestration commit mismatch at ${RUNTIME_REPO}: ${runtime_commit} != ${ORCHESTRATION_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${RUNTIME_REPO}" status --porcelain)" ]]; then
    echo "Workflow repository is dirty: ${RUNTIME_REPO}" >&2
    exit 2
fi

physics_commit="$(git -C "${PREDICT_CODE_ROOT}" rev-parse HEAD)"
if [[ "${physics_commit}" != "${PHYSICS_COMMIT}" ]]; then
    echo "PREDICT physics commit mismatch at ${PREDICT_CODE_ROOT}: ${physics_commit} != ${PHYSICS_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${PREDICT_CODE_ROOT}" status --porcelain)" ]]; then
    echo "PREDICT physics repository is dirty: ${PREDICT_CODE_ROOT}" >&2
    exit 2
fi

actual_method_config_sha256="$(sha256sum "${METHOD_CONFIG}" | awk '{print $1}')"
if [[ "${actual_method_config_sha256}" != "${METHOD_CONFIG_SHA256}" ]]; then
    echo "Method configuration hash mismatch: ${actual_method_config_sha256} != ${METHOD_CONFIG_SHA256}" >&2
    exit 2
fi

python3 - \
    "${RUN_ROOT}/phase_run_identity.json" \
    "${PHASE}" \
    "${RUN_ID}" \
    "${HANDOFF_ROOT}" \
    "${HANDOFF_SHA256}" \
    "${SAMPLING_COMMIT}" \
    "${PHYSICS_COMMIT}" \
    "${RUNTIME_REPO}" \
    "${ORCHESTRATION_COMMIT}" \
    "${PREDICT_CODE_ROOT}" \
    "${METHOD_CONFIG_SHA256}" \
    "${EXPECTED_GEOLOGIES}" \
    "${EXPECTED_CASES}" \
    "${EXPECTED_ASSIGNMENTS}" <<'PY'
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

(
    path_text,
    phase,
    run_id,
    handoff_root,
    handoff_sha256,
    sampling_commit,
    physics_commit,
    runtime_repo,
    orchestration_commit,
    predict_code_root,
    method_hash,
    geology_count,
    case_count,
    assignment_count,
) = sys.argv[1:]
path = Path(path_text)
immutable_identity = {
    "phase": phase,
    "run_id": run_id,
    "handoff_root": handoff_root,
    "handoff_metadata_sha256": handoff_sha256,
    "sampling_code_commit": sampling_commit,
    "physics_commit": physics_commit,
    "production_method_config_sha256": method_hash,
    "expected_geology_count": int(geology_count),
    "expected_case_count": int(case_count),
    "expected_assignment_count": int(assignment_count),
}
if path.is_file():
    existing = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in immutable_identity.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            "Run root belongs to a different immutable handoff or configuration: "
            + json.dumps(mismatches, sort_keys=True)
        )
else:
    identity = {
        "schema_version": "independent_full_fault_phase_run_identity_v2",
        **immutable_identity,
        "initial_runtime_repo": runtime_repo,
        "initial_orchestration_commit": orchestration_commit,
        "initial_predict_code_root": predict_code_root,
    }
    identity["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
PY

if [[ ! -f "${CHECKPOINT_MANIFEST_ROOT}/checkpoint_groups.csv" ]]; then
    python3 "${PRODUCTION_DIR}/build_checkpoint_work_manifest.py" \
        --freeze-root "${FREEZE_ROOT}" \
        --predict-root "${PREDICT_ROOT}" \
        --sampling-csv "${SAMPLING_CSV}" \
        --output-root "${CHECKPOINT_MANIFEST_ROOT}"
fi
if [[ ! -f "${CASE_WORK_ROOT}/case_work.csv" ]]; then
    python3 "${PRODUCTION_DIR}/build_case_work_manifest.py" \
        --sampling-csv "${SAMPLING_CSV}" \
        --assignment-to-task-csv "${FREEZE_ROOT}/manifests/assignment_to_task.csv" \
        --output-root "${CASE_WORK_ROOT}"
fi

status_args=(
    --run-root "${RUN_ROOT}"
    --checkpoint-chunk-size "${GROUPS_PER_ARRAY_TASK}"
    --assembly-chunk-size "${GEOLOGIES_PER_ARRAY_TASK}"
    --kr-chunk-size "${CASES_PER_ARRAY_TASK}"
    --max-replay-tolerance-log10 "${REPLAY_TOLERANCE_LOG10}"
)
eval "$(python3 "${STATUS_TOOL}" "${status_args[@]}" --shell)"
python3 "${STATUS_TOOL}" \
    "${status_args[@]}" \
    --output-json "${RUN_ROOT}/phase_status_before_submission.json"

if [[ "${CHECKPOINT_TOTAL}" -le 0 || "${ASSEMBLY_TOTAL}" -ne "${EXPECTED_GEOLOGIES}" || "${KR_TOTAL}" -ne "${EXPECTED_CASES}" ]]; then
    echo "Derived work-manifest counts do not match the phase contract." >&2
    exit 2
fi

cat <<EOF
Independent full-fault ${PHASE} production plan
  run_id: ${RUN_ID}
  handoff: ${HANDOFF_ROOT}
  frozen sampling commit: ${SAMPLING_COMMIT}
  orchestration commit: ${ORCHESTRATION_COMMIT}
  PREDICT physics commit: ${PHYSICS_COMMIT}
  workflow checkout: ${RUNTIME_REPO}
  PREDICT physics checkout: ${PREDICT_CODE_ROOT}
  method config SHA-256: ${METHOD_CONFIG_SHA256}
  checkpoint replay/Pc: ${CHECKPOINT_COMPLETE}/${CHECKPOINT_TOTAL} complete
  geology assembly: ${ASSEMBLY_COMPLETE}/${ASSEMBLY_TOTAL} complete
  dynamic Kr/final export: ${KR_COMPLETE}/${KR_TOTAL} complete
  checkpoint array: ${CHECKPOINT_MISSING_ARRAY_TASKS} pending tasks, ${GROUPS_PER_ARRAY_TASK} groups/task
  assembly array: ${ASSEMBLY_MISSING_ARRAY_TASKS} pending tasks, ${GEOLOGIES_PER_ARRAY_TASK} geologies/task
  Kr array: ${KR_MISSING_ARRAY_TASKS} pending tasks, ${CASES_PER_ARRAY_TASK} cases/task
  checkpoint concurrency: ${CHECKPOINT_MAX_CONCURRENT} one-CPU array tasks
  checkpoint temporary root: ${CHECKPOINT_TEMP_ROOT}
  case temporary root: ${CASE_TEMP_ROOT}
  external checkpoint bundle: ${EXTERNAL_CHECKPOINT_JOB_ID:-none}
  Kr concurrency: ${KR_MAX_CONCURRENT} six-CPU array tasks
EOF

if [[ "${ACTION}" == "plan" ]]; then
    exit 0
fi
if [[ "${ACTION}" == "submit" && -f "${RUN_ROOT}/phase_submission_latest.json" ]]; then
    echo "This run was already submitted; use the continue action." >&2
    exit 2
fi
active_jobs="$(
    squeue -u "${USER}" -h -o "%A|%j|%T" \
        | awk -F'|' -v run_id="${RUN_ID}" -v external_id="${EXTERNAL_CHECKPOINT_JOB_ID}" '
            index($2, run_id) > 0 && (external_id == "" || $1 != external_id) { print }
        ' \
        || true
)"
if [[ -n "${active_jobs}" ]]; then
    echo "Active jobs already exist for ${RUN_ID}:" >&2
    echo "${active_jobs}" >&2
    exit 2
fi
gate_count=0
if [[ "${CHECKPOINT_MISSING}" -gt 0 ]]; then
    gate_count=$((gate_count + 1))
fi
if [[ "${KR_MISSING}" -gt 0 ]]; then
    gate_count=$((gate_count + 1))
fi
checkpoint_submission_elements="${CHECKPOINT_MISSING_ARRAY_TASKS}"
if [[ -n "${EXTERNAL_CHECKPOINT_JOB_ID}" ]]; then
    checkpoint_submission_elements=0
fi
submitted_elements=$((
    checkpoint_submission_elements
    + ASSEMBLY_MISSING_ARRAY_TASKS
    + KR_MISSING_ARRAY_TASKS
    + gate_count
))
existing_job_elements="$(squeue -r -h -u "${USER}" | wc -l)"
existing_job_elements="${existing_job_elements//[[:space:]]/}"
if (( submitted_elements + existing_job_elements > SLURM_MAX_SUBMITTED_JOBS )); then
    echo "Submission requires ${submitted_elements} new Slurm elements and ${existing_job_elements} already exist; limit is ${SLURM_MAX_SUBMITTED_JOBS}." >&2
    echo "Increase chunk sizes without changing the scientific case definitions." >&2
    exit 2
fi

mkdir -p \
    "${LOG_ROOT}/checkpoint_pc" \
    "${LOG_ROOT}/checkpoint_gate" \
    "${LOG_ROOT}/assembly" \
    "${LOG_ROOT}/kr" \
    "${LOG_ROOT}/final_gate"

CHECKPOINT_JOB_ID=""
CHECKPOINT_GATE_JOB_ID=""
ASSEMBLY_JOB_ID=""
KR_JOB_ID=""
FINAL_GATE_JOB_ID=""

if [[ "${CHECKPOINT_MISSING}" -gt 0 ]]; then
    if [[ -n "${EXTERNAL_CHECKPOINT_JOB_ID}" ]]; then
        external_state="$(squeue -j "${EXTERNAL_CHECKPOINT_JOB_ID}" -h -o '%T')"
        if [[ "${external_state}" != "PENDING" && "${external_state}" != "RUNNING" ]]; then
            echo "External checkpoint job ${EXTERNAL_CHECKPOINT_JOB_ID} is not pending or running." >&2
            exit 2
        fi
        CHECKPOINT_JOB_ID="${EXTERNAL_CHECKPOINT_JOB_ID}"
        echo "Using external checkpoint node-bundle job ${CHECKPOINT_JOB_ID} (${external_state})."
    else
        submission="$(
            sbatch \
            --parsable \
            --account="${SLURM_ACCOUNT}" \
            --qos="${SLURM_QOS}" \
            --partition="${SLURM_PARTITION}" \
            --job-name="rpc_${RUN_ID}" \
            --time="${CHECKPOINT_WALLTIME}" \
            --cpus-per-task=1 \
            --mem="${CHECKPOINT_MEMORY}" \
            --array="${CHECKPOINT_ARRAY_SPEC}%${CHECKPOINT_MAX_CONCURRENT}" \
            --output="${LOG_ROOT}/checkpoint_pc/%x_%A_%a.out" \
            --error="${LOG_ROOT}/checkpoint_pc/%x_%A_%a.err" \
            --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT}",FREEZE_ROOT="${FREEZE_ROOT}",PREDICT_ROOT="${PREDICT_ROOT}",METHOD_CONFIG="${METHOD_CONFIG}",CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT}",COMPACT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT}",CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT}",MATLAB_SHORT_ROOT="${MATLAB_SHORT_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",GROUP_COUNT="${CHECKPOINT_TOTAL}",GROUPS_PER_ARRAY_TASK="${GROUPS_PER_ARRAY_TASK}" \
            "${PRODUCTION_DIR}/run_checkpoint_replay_pc_chunk.sh"
        )"
        CHECKPOINT_JOB_ID="${submission%%;*}"
    fi
    gate_submission="$(
        sbatch \
            --parsable \
            --account="${SLURM_ACCOUNT}" \
            --qos="${SLURM_QOS}" \
            --partition="${SLURM_PARTITION}" \
            --job-name="cgate_${RUN_ID}" \
            --time="04:00:00" \
            --cpus-per-task=1 \
            --mem="8G" \
            --dependency="afterany:${CHECKPOINT_JOB_ID}" \
            --output="${LOG_ROOT}/checkpoint_gate/%x_%j.out" \
            --error="${LOG_ROOT}/checkpoint_gate/%x_%j.err" \
            --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",DEFAULT_REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}" \
            "${PRODUCTION_DIR}/run_checkpoint_completion_gate.sh"
    )"
    CHECKPOINT_GATE_JOB_ID="${gate_submission%%;*}"
else
    RUNTIME_REPO="${RUNTIME_REPO}" \
    RUN_ROOT="${RUN_ROOT}" \
    PHYSICS_COMMIT="${PHYSICS_COMMIT}" \
    METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}" \
    DEFAULT_REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}" \
        bash "${PRODUCTION_DIR}/run_checkpoint_completion_gate.sh"
fi

assembly_dependency=()
if [[ -n "${CHECKPOINT_GATE_JOB_ID}" ]]; then
    assembly_dependency=(--dependency="afterok:${CHECKPOINT_GATE_JOB_ID}")
fi
if [[ "${ASSEMBLY_MISSING}" -gt 0 ]]; then
    assembly_submission="$(
        sbatch \
            --parsable \
            --account="${SLURM_ACCOUNT}" \
            --qos="${SLURM_QOS}" \
            --partition="${SLURM_PARTITION}" \
            --job-name="asm_${RUN_ID}" \
            --time="${ASSEMBLY_WALLTIME}" \
            --cpus-per-task=1 \
            --mem="${ASSEMBLY_MEMORY}" \
            --array="${ASSEMBLY_ARRAY_SPEC}%${ASSEMBLY_MAX_CONCURRENT}" \
            --output="${LOG_ROOT}/assembly/%x_%A_%a.out" \
            --error="${LOG_ROOT}/assembly/%x_%A_%a.err" \
            "${assembly_dependency[@]}" \
            --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",CASE_WORK_ROOT="${CASE_WORK_ROOT}",CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}",CASE_INPUT_ROOT="${CASE_INPUT_ROOT}",GEOLOGY_COUNT="${ASSEMBLY_TOTAL}",GEOLOGIES_PER_ARRAY_TASK="${GEOLOGIES_PER_ARRAY_TASK}" \
            "${PRODUCTION_DIR}/run_assemble_geology_cases_chunk.sh"
    )"
    ASSEMBLY_JOB_ID="${assembly_submission%%;*}"
fi

kr_dependency=()
if [[ -n "${ASSEMBLY_JOB_ID}" ]]; then
    kr_dependency=(--dependency="afterok:${ASSEMBLY_JOB_ID}")
elif [[ -n "${CHECKPOINT_GATE_JOB_ID}" ]]; then
    kr_dependency=(--dependency="afterok:${CHECKPOINT_GATE_JOB_ID}")
fi
if [[ "${KR_MISSING}" -gt 0 ]]; then
    kr_submission="$(
        sbatch \
            --parsable \
            --account="${SLURM_ACCOUNT}" \
            --qos="${SLURM_QOS}" \
            --partition="${SLURM_PARTITION}" \
            --job-name="kr_${RUN_ID}" \
            --time="${KR_WALLTIME}" \
            --cpus-per-task=6 \
            --mem="${KR_MEMORY}" \
            --array="${KR_ARRAY_SPEC}%${KR_MAX_CONCURRENT}" \
            --output="${LOG_ROOT}/kr/%x_%A_%a.out" \
            --error="${LOG_ROOT}/kr/%x_%A_%a.err" \
            "${kr_dependency[@]}" \
            --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT}",FREEZE_ROOT="${FREEZE_ROOT}",PREDICT_ROOT="${PREDICT_ROOT}",PERMEABILITY_INPUT="${PERMEABILITY_INPUT}",CASE_WORK_ROOT="${CASE_WORK_ROOT}",CASE_INPUT_ROOT="${CASE_INPUT_ROOT}",CASE_RESULT_ROOT="${CASE_RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT}",CASE_TEMP_ROOT="${CASE_TEMP_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",CASE_COUNT="${KR_TOTAL}",CASES_PER_ARRAY_TASK="${CASES_PER_ARRAY_TASK}" \
            "${PRODUCTION_DIR}/run_case_dynamic_kr_chunk.sh"
    )"
    KR_JOB_ID="${kr_submission%%;*}"
    final_submission="$(
        sbatch \
            --parsable \
            --account="${SLURM_ACCOUNT}" \
            --qos="${SLURM_QOS}" \
            --partition="${SLURM_PARTITION}" \
            --job-name="qagate_${RUN_ID}" \
            --time="08:00:00" \
            --cpus-per-task=1 \
            --mem="8G" \
            --dependency="afterany:${KR_JOB_ID}" \
            --output="${LOG_ROOT}/final_gate/%x_%j.out" \
            --error="${LOG_ROOT}/final_gate/%x_%j.err" \
            --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",MAX_SOURCE_LOG_PERMEABILITY_MISMATCH="${REPLAY_TOLERANCE_LOG10}" \
            "${PRODUCTION_DIR}/run_case_completion_gate.sh"
    )"
    FINAL_GATE_JOB_ID="${final_submission%%;*}"
else
    RUNTIME_REPO="${RUNTIME_REPO}" \
    RUN_ROOT="${RUN_ROOT}" \
    PHYSICS_COMMIT="${PHYSICS_COMMIT}" \
    METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}" \
    MAX_SOURCE_LOG_PERMEABILITY_MISMATCH="${REPLAY_TOLERANCE_LOG10}" \
        bash "${PRODUCTION_DIR}/run_case_completion_gate.sh"
fi

python3 - \
    "${RUN_ROOT}/phase_submission_latest.json" \
    "${ACTION}" \
    "${PHASE}" \
    "${RUN_ID}" \
    "${CHECKPOINT_JOB_ID}" \
    "${CHECKPOINT_GATE_JOB_ID}" \
    "${ASSEMBLY_JOB_ID}" \
    "${KR_JOB_ID}" \
    "${FINAL_GATE_JOB_ID}" \
    "${HANDOFF_ROOT}" \
    "${HANDOFF_SHA256}" \
    "${SAMPLING_COMMIT}" \
    "${ORCHESTRATION_COMMIT}" \
    "${RUNTIME_REPO}" \
    "${PHYSICS_COMMIT}" \
    "${PREDICT_CODE_ROOT}" \
    "${PREDICT_ROOT}" \
    "${METHOD_CONFIG_SHA256}" \
    "${REPLAY_TOLERANCE_LOG10}" \
    "${CHECKPOINT_COMPLETE}" \
    "${CHECKPOINT_MISSING}" <<'PY'
from datetime import datetime, timezone
import json
import sys

(
    output,
    action,
    phase,
    run_id,
    checkpoint_job,
    checkpoint_gate,
    assembly_job,
    kr_job,
    final_gate,
    handoff_root,
    handoff_sha256,
    sampling_commit,
    orchestration_commit,
    runtime_repo,
    physics_commit,
    predict_code_root,
    predict_root,
    method_config_sha256,
    replay_tolerance,
    checkpoint_complete,
    checkpoint_missing,
) = sys.argv[1:]
record = {
    "schema_version": "independent_full_fault_phase_submission_v2",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "action": action,
    "phase": phase,
    "run_id": run_id,
    "retry_scope": "missing_or_invalid_markers_only",
    "successful_checkpoint_markers_reused": int(checkpoint_complete),
    "checkpoint_markers_missing_before_submission": int(checkpoint_missing),
    "numerical_equivalence": {
        "metric": "maximum_absolute_log10_permeability_difference",
        "maximum_tolerance": float(replay_tolerance),
        "semantics": "maximum_allowed; stricter successful markers remain valid",
    },
    "provenance": {
        "handoff_root": handoff_root,
        "handoff_metadata_sha256": handoff_sha256,
        "sampling_code_commit": sampling_commit,
        "orchestration_code_commit": orchestration_commit,
        "runtime_repo": runtime_repo,
        "physics_commit": physics_commit,
        "predict_code_root": predict_code_root,
        "predict_data_root": predict_root,
        "production_method_config_sha256": method_config_sha256,
    },
    "jobs": {
        "checkpoint_array": checkpoint_job or None,
        "checkpoint_gate": checkpoint_gate or None,
        "assembly_array": assembly_job or None,
        "dynamic_kr_array": kr_job or None,
        "final_gate": final_gate or None,
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps(record, indent=2))
PY

echo "Submitted restartable ${PHASE} workflow for ${RUN_ID}."
