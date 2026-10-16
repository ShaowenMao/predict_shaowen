#!/bin/bash
#SBATCH --job-name=mipviz-setup
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096M
#SBATCH --time=00:30:00
set -euo pipefail
root=$1
test ! -e "$root/venv"
/usr/bin/python3.12 -m venv "$root/venv"
"$root/venv/bin/python" -m pip install --disable-pip-version-check -r "$root/workflow/requirements.txt"
"$root/venv/bin/python" -m pip check
"$root/venv/bin/python" -m pip freeze > "$root/environment.txt"
