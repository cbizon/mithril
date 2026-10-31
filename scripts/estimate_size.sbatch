#!/bin/bash
#SBATCH --job-name=estimate
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:30:00
#SBATCH --array=0-7
# usage: sbatch --output=<dir>/%x-%A_%a.log scripts/estimate_size.sbatch <out dir> <target> <subject> <object> [extra args]
set -euo pipefail
out=$1; shift; target=$1; subject=$2; object=$3; shift 3
uv run python scripts/estimate_size.py probe --target "$target" --subject-class "$subject" --object-class "$object" \
    --seed ${SLURM_ARRAY_TASK_ID} --time-limit 10800 --out "$out/probes_${SLURM_ARRAY_TASK_ID}.pkl" "$@"
