#!/bin/bash
#SBATCH --job-name=mipviz-render
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192M
#SBATCH --time=01:00:00
set -euo pipefail
root=$1
key=${2:-}
if [[ -z "$key" ]]; then
    key=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$root/bundle/render_keys.txt")
fi
[[ "$key" =~ ^(low|intermediate|high)_[0-9]{3}$ ]]
module load tex-live/20251104 julia/1.12.6
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYVISTA_OFF_SCREEN=true
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4 EGL_PLATFORM=surfaceless
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
work=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mipviz.${SLURM_JOB_ID}.XXXXXX")
export MPLCONFIGDIR="$work/matplotlib" JULIA_DEPOT_PATH="$work/julia_depot"
mkdir -p "$MPLCONFIGDIR" "$JULIA_DEPOT_PATH"
python="$root/venv/bin/python"
cd "$root/workflow"
srun --cpu-bind=cores --ntasks=1 --cpus-per-task=4 /usr/bin/time -v \
    "$python" -B run_core.py --bundle "$root/bundle" --core "$key" --output "$root/outputs/$key"
