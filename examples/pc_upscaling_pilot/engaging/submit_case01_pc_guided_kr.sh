#!/usr/bin/env bash
# Convenience Swi-medoid Kr wrapper for completed Case 01 replay/Pc results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEOLOGY_ID="${GEOLOGY_ID:-s05_c012}"
export CASE_IDS="${CASE_IDS:-1}"
export SOURCE_RUN_ID="${SOURCE_RUN_ID:-case01_full_codex_20260707_165751}"
export PC_RUN_ID="${PC_RUN_ID:-case01_full_native_swi_20260708_171125}"

exec "${SCRIPT_DIR}/submit_pc_guided_kr.sh"
