#!/usr/bin/env bash
# Submit the complete restartable 162-geology production workflow.

set -euo pipefail

ACTION="${1:-plan}"
if [[ "${ACTION}" != "plan" && "${ACTION}" != "submit" ]]; then
    echo "Usage: $0 plan|submit" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
ORCHESTRATION_COMMIT="${ORCHESTRATION_COMMIT:-}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
RUN_ID="${RUN_ID:-production_all1620_20260724_v1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DATA_ROOT}/production_runs/${RUN_ID}}"
QUALIFICATION_REPORT="${QUALIFICATION_REPORT:-${PROJECT_DATA_ROOT}/production_runs/production_qualification60_20260723_v1/case_completion_gate.json}"
POLICY_FILE="${POLICY_FILE:-${SCRIPT_DIR}/production_acceptance_policy.toml}"

PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-0.005}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-mit_amf_advanced_cpu}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"
SLURM_PARTITION="${SLURM_PARTITION:-mit_normal}"
CHECKPOINT_MAX_CONCURRENT="${CHECKPOINT_MAX_CONCURRENT:-96}"
KR_MAX_CONCURRENT="${KR_MAX_CONCURRENT:-24}"
CHECKPOINT_GROUPS_PER_ARRAY_TASK="${CHECKPOINT_GROUPS_PER_ARRAY_TASK:-5}"
ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK="${ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK:-6}"
KR_CASES_PER_ARRAY_TASK="${KR_CASES_PER_ARRAY_TASK:-10}"
SLURM_MAX_SUBMITTED_JOBS="${SLURM_MAX_SUBMITTED_JOBS:-400}"
RESUME="${RESUME:-0}"

CHECKPOINT_GROUP_COUNT=972
GEOLOGY_COUNT=162
CASE_COUNT=1620
for positive_integer in \
    "${CHECKPOINT_GROUPS_PER_ARRAY_TASK}" \
    "${ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK}" \
    "${KR_CASES_PER_ARRAY_TASK}" \
    "${SLURM_MAX_SUBMITTED_JOBS}"; do
    if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Chunk sizes and scheduler limits must be positive integers." >&2
        exit 2
    fi
done
CHECKPOINT_ARRAY_TASK_COUNT=$(( (CHECKPOINT_GROUP_COUNT + CHECKPOINT_GROUPS_PER_ARRAY_TASK - 1) / CHECKPOINT_GROUPS_PER_ARRAY_TASK ))
ASSEMBLY_ARRAY_TASK_COUNT=$(( (GEOLOGY_COUNT + ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK - 1) / ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK ))
KR_ARRAY_TASK_COUNT=$(( (CASE_COUNT + KR_CASES_PER_ARRAY_TASK - 1) / KR_CASES_PER_ARRAY_TASK ))
TOTAL_SUBMITTED_JOB_ELEMENTS=$(( CHECKPOINT_ARRAY_TASK_COUNT + ASSEMBLY_ARRAY_TASK_COUNT + KR_ARRAY_TASK_COUNT + 2 ))
if (( TOTAL_SUBMITTED_JOB_ELEMENTS > SLURM_MAX_SUBMITTED_JOBS )); then
    echo "Planned ${TOTAL_SUBMITTED_JOB_ELEMENTS} submitted jobs exceed the scheduler limit ${SLURM_MAX_SUBMITTED_JOBS}." >&2
    exit 2
fi

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

if [[ ! -f "${POLICY_FILE}" ]]; then
    echo "Missing production acceptance policy: ${POLICY_FILE}" >&2
    exit 2
fi
if [[ ! -f "${QUALIFICATION_REPORT}" ]]; then
    echo "Missing qualification report: ${QUALIFICATION_REPORT}" >&2
    exit 2
fi
if [[ ! -f "${FREEZE_ROOT}/freeze_metadata.json" ]]; then
    echo "Missing production freeze metadata: ${FREEZE_ROOT}" >&2
    exit 2
fi
if [[ "${ACTION}" == "submit" && -z "${ORCHESTRATION_COMMIT}" ]]; then
    echo "ORCHESTRATION_COMMIT is required for submission provenance." >&2
    exit 2
fi

python3 - \
    "${QUALIFICATION_REPORT}" \
    "${FREEZE_ROOT}/freeze_metadata.json" \
    "${PHYSICS_COMMIT}" \
    "${METHOD_CONFIG_SHA256}" <<'PY'
import json
import sys

qualification = json.load(open(sys.argv[1], encoding="utf-8"))
freeze = json.load(open(sys.argv[2], encoding="utf-8"))
physics_commit = sys.argv[3]
method_hash = sys.argv[4]

expected_qualification = {
    "status": "complete",
    "expected_case_count": 60,
    "expected_geology_count": 6,
    "input_markers_validated": 60,
    "result_markers_validated": 60,
    "expected_physics_commit": physics_commit,
    "expected_method_config_sha256": method_hash,
    "error_count": 0,
}
for field, expected in expected_qualification.items():
    actual = qualification.get(field)
    if actual != expected:
        raise SystemExit(
            f"Qualification field {field}={actual!r}; expected {expected!r}"
        )

expected_freeze = {
    "code_commit": physics_commit,
    "method_config_sha256": method_hash,
    "geology_count": 162,
    "geology_case_count": 1620,
    "slice_count": 87,
    "window_count": 6,
    "assignment_count": 845640,
    "unique_replay_pc_task_count": 519446,
}
for field, expected in expected_freeze.items():
    actual = freeze.get(field)
    if actual != expected:
        raise SystemExit(
            f"Freeze field {field}={actual!r}; expected {expected!r}"
        )
print("Qualification and full-production freeze checks passed.")
PY

policy_sha256="$(sha256sum "${POLICY_FILE}" | awk '{print $1}')"
qualification_sha256="$(sha256sum "${QUALIFICATION_REPORT}" | awk '{print $1}')"
freeze_metadata_sha256="$(
    sha256sum "${FREEZE_ROOT}/freeze_metadata.json" | awk '{print $1}'
)"

cat <<EOF
Production launch plan
  run_id: ${RUN_ID}
  run_root: ${RUN_ROOT}
  freeze_root: ${FREEZE_ROOT}
  policy_sha256: ${policy_sha256}
  qualification_sha256: ${qualification_sha256}
  replay_tolerance_log10: ${REPLAY_TOLERANCE_LOG10}
  Slurm: account=${SLURM_ACCOUNT}, qos=${SLURM_QOS}, partition=${SLURM_PARTITION}
  checkpoint: ${CHECKPOINT_ARRAY_TASK_COUNT} array tasks, ${CHECKPOINT_GROUPS_PER_ARRAY_TASK} groups/task, concurrency ${CHECKPOINT_MAX_CONCURRENT}
  assembly: ${ASSEMBLY_ARRAY_TASK_COUNT} array tasks, ${ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK} geologies/task
  dynamic Kr: ${KR_ARRAY_TASK_COUNT} array tasks, ${KR_CASES_PER_ARRAY_TASK} cases/task, concurrency ${KR_MAX_CONCURRENT}
  total submitted job elements: ${TOTAL_SUBMITTED_JOB_ELEMENTS}/${SLURM_MAX_SUBMITTED_JOBS}
EOF

if [[ "${ACTION}" == "plan" ]]; then
    if [[ -e "${RUN_ROOT}" ]]; then
        echo "WARNING: run root already exists: ${RUN_ROOT}" >&2
    fi
    exit 0
fi

if [[ -e "${RUN_ROOT}" && "${RESUME}" != "1" ]]; then
    echo "Run root already exists; set RESUME=1 only for a continuation run." >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"

export RUNTIME_REPO FREEZE_ROOT PROJECT_DATA_ROOT SCRATCH_ROOT RUN_ID RUN_ROOT
export PHYSICS_COMMIT METHOD_CONFIG_SHA256 REPLAY_TOLERANCE_LOG10
export SLURM_QOS
export MAX_CONCURRENT="${CHECKPOINT_MAX_CONCURRENT}"
export GROUPS_PER_ARRAY_TASK="${CHECKPOINT_GROUPS_PER_ARRAY_TASK}"

bash "${SCRIPT_DIR}/submit_checkpoint_replay_pc.sh" full
CHECKPOINT_ARRAY_JOB_ID="$(<"${RUN_ROOT}/checkpoint_array_job_id.txt")"

CHECKPOINT_GATE_LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}/checkpoint_gate"
mkdir -p "${CHECKPOINT_GATE_LOG_ROOT}"
checkpoint_gate_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="gate_${RUN_ID}" \
        --time="${CHECKPOINT_GATE_WALLTIME:-04:00:00}" \
        --cpus-per-task=1 \
        --mem="${CHECKPOINT_GATE_MEMORY:-8G}" \
        --dependency="afterany:${CHECKPOINT_ARRAY_JOB_ID}" \
        --output="${CHECKPOINT_GATE_LOG_ROOT}/%x_%j.out" \
        --error="${CHECKPOINT_GATE_LOG_ROOT}/%x_%j.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",DEFAULT_REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}" \
        "${SCRIPT_DIR}/run_checkpoint_completion_gate.sh"
)"
CHECKPOINT_GATE_JOB_ID="${checkpoint_gate_submission%%;*}"

export CHECKPOINT_JOB_ID="${CHECKPOINT_GATE_JOB_ID}"
export KR_MAX_CONCURRENT
export GEOLOGIES_PER_ARRAY_TASK="${ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK}"
export CASES_PER_ARRAY_TASK="${KR_CASES_PER_ARRAY_TASK}"
bash "${SCRIPT_DIR}/submit_case_assembly_kr.sh" full
ASSEMBLY_JOB_ID="$(<"${RUN_ROOT}/assembly_array_job_id.txt")"
KR_JOB_ID="$(<"${RUN_ROOT}/kr_array_job_id.txt")"

FINAL_GATE_LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}/final_gate"
mkdir -p "${FINAL_GATE_LOG_ROOT}"
final_gate_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="qa_${RUN_ID}" \
        --time="${FINAL_GATE_WALLTIME:-08:00:00}" \
        --cpus-per-task=1 \
        --mem="${FINAL_GATE_MEMORY:-8G}" \
        --dependency="afterany:${KR_JOB_ID}" \
        --output="${FINAL_GATE_LOG_ROOT}/%x_%j.out" \
        --error="${FINAL_GATE_LOG_ROOT}/%x_%j.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",MAX_SOURCE_LOG_PERMEABILITY_MISMATCH="${REPLAY_TOLERANCE_LOG10}" \
        "${SCRIPT_DIR}/run_case_completion_gate.sh"
)"
FINAL_GATE_JOB_ID="${final_gate_submission%%;*}"

python3 - \
    "${RUN_ROOT}/production_launch_manifest.json" \
    "${RUN_ID}" \
    "${RUN_ROOT}" \
    "${FREEZE_ROOT}" \
    "${ORCHESTRATION_COMMIT}" \
    "${policy_sha256}" \
    "${qualification_sha256}" \
    "${freeze_metadata_sha256}" \
    "${PHYSICS_COMMIT}" \
    "${METHOD_CONFIG_SHA256}" \
    "${REPLAY_TOLERANCE_LOG10}" \
    "${SLURM_ACCOUNT}" \
    "${SLURM_QOS}" \
    "${SLURM_PARTITION}" \
    "${CHECKPOINT_MAX_CONCURRENT}" \
    "${KR_MAX_CONCURRENT}" \
    "${CHECKPOINT_GROUPS_PER_ARRAY_TASK}" \
    "${ASSEMBLY_GEOLOGIES_PER_ARRAY_TASK}" \
    "${KR_CASES_PER_ARRAY_TASK}" \
    "${CHECKPOINT_ARRAY_TASK_COUNT}" \
    "${ASSEMBLY_ARRAY_TASK_COUNT}" \
    "${KR_ARRAY_TASK_COUNT}" \
    "${TOTAL_SUBMITTED_JOB_ELEMENTS}" \
    "${SLURM_MAX_SUBMITTED_JOBS}" \
    "${CHECKPOINT_ARRAY_JOB_ID}" \
    "${CHECKPOINT_GATE_JOB_ID}" \
    "${ASSEMBLY_JOB_ID}" \
    "${KR_JOB_ID}" \
    "${FINAL_GATE_JOB_ID}" <<'PY'
from datetime import datetime, timezone
import json
import sys

(
    output_path,
    run_id,
    run_root,
    freeze_root,
    orchestration_commit,
    policy_hash,
    qualification_hash,
    freeze_metadata_hash,
    physics_commit,
    method_hash,
    replay_tolerance,
    slurm_account,
    slurm_qos,
    slurm_partition,
    checkpoint_concurrency,
    kr_concurrency,
    checkpoint_groups_per_array_task,
    assembly_geologies_per_array_task,
    kr_cases_per_array_task,
    checkpoint_array_task_count,
    assembly_array_task_count,
    kr_array_task_count,
    total_submitted_job_elements,
    slurm_max_submitted_jobs,
    checkpoint_job_id,
    checkpoint_gate_job_id,
    assembly_job_id,
    kr_job_id,
    final_gate_job_id,
) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "status": "submitted",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "run_root": run_root,
    "freeze_root": freeze_root,
    "orchestration_commit": orchestration_commit,
    "production_acceptance_policy_sha256": policy_hash,
    "qualification_report_sha256": qualification_hash,
    "freeze_metadata_sha256": freeze_metadata_hash,
    "physics_commit": physics_commit,
    "method_config_sha256": method_hash,
    "replay_tolerance_log10": float(replay_tolerance),
    "expected_geology_count": 162,
    "expected_case_count": 1620,
    "expected_assignment_count": 845640,
    "expected_unique_replay_pc_task_count": 519446,
    "slurm": {
        "account": slurm_account,
        "qos": slurm_qos,
        "partition": slurm_partition,
        "checkpoint_max_concurrent": int(checkpoint_concurrency),
        "dynamic_kr_max_concurrent": int(kr_concurrency),
        "checkpoint_groups_per_array_task": int(
            checkpoint_groups_per_array_task
        ),
        "assembly_geologies_per_array_task": int(
            assembly_geologies_per_array_task
        ),
        "dynamic_kr_cases_per_array_task": int(kr_cases_per_array_task),
        "checkpoint_array_task_count": int(checkpoint_array_task_count),
        "assembly_array_task_count": int(assembly_array_task_count),
        "dynamic_kr_array_task_count": int(kr_array_task_count),
        "total_submitted_job_elements": int(total_submitted_job_elements),
        "scheduler_submitted_job_limit": int(slurm_max_submitted_jobs),
    },
    "jobs": {
        "checkpoint_array": checkpoint_job_id,
        "checkpoint_gate": checkpoint_gate_job_id,
        "assembly_array": assembly_job_id,
        "dynamic_kr_array": kr_job_id,
        "final_qa_gate": final_gate_job_id,
    },
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print(json.dumps(manifest, indent=2))
PY

echo "Full production chain submitted."
