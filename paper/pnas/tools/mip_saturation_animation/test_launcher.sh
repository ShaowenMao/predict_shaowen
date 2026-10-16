#!/bin/bash
# Fast non-mutating-launch tests: no modules, Slurm submissions or simulations.
set -euo pipefail
script=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/render_job.sh
root=$(mktemp -d)
cleanup() {
    rm -f -- "$root/bundle/render_keys.txt"
    rmdir -- "$root/bundle" "$root"
}
trap cleanup EXIT
mkdir "$root/bundle"
for ending in '\n' '\r\n'; do
    printf "low_143${ending}high_165${ending}" > "$root/bundle/render_keys.txt"
    [[ $(MIPVIZ_VALIDATE_ONLY=1 SLURM_ARRAY_TASK_ID=0 bash "$script" "$root") == low_143 ]]
    [[ $(MIPVIZ_VALIDATE_ONLY=1 SLURM_ARRAY_TASK_ID=1 bash "$script" "$root") == high_165 ]]
    if MIPVIZ_VALIDATE_ONLY=1 SLURM_ARRAY_TASK_ID=2 bash "$script" "$root" 2>/dev/null; then
        exit 1
    fi
done
[[ $(MIPVIZ_VALIDATE_ONLY=1 bash "$script" "$root" high_172) == high_172 ]]
if MIPVIZ_VALIDATE_ONLY=1 bash "$script" "$root" ../bad 2>/dev/null; then
    exit 1
fi
printf 'PASS: LF/CRLF, both array endpoints, missing index, explicit core, invalid key\n'
