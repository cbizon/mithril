#!/bin/bash
#SBATCH --job-name=select
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-14
# usage: sbatch [--dependency=afterok:<mine job>] --output=<dir>/%x-%A_%a.log scripts/select_sweep.sbatch <mined dir>
set -euo pipefail
floors=(0.01 0.02 0.03)
news=(0.001 0.0025 0.005 0.01 0.02)
i=${SLURM_ARRAY_TASK_ID}
uv run python scripts/select_rules.py --mined "$1" --recall-floor ${floors[$((i / 5))]} --min-new ${news[$((i % 5))]}
