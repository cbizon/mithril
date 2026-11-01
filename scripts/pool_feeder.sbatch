#!/bin/bash
# Feeder for a branch_pool work pool: a 1-CPU job that tops up the worker queue for
# ~3.5 h, then resubmits itself until the pool is drained (branch_pool.py writes DRAINED).
# Usage (from the repo root): sbatch scripts/pool_feeder.sbatch <pool> [feed options...]
#SBATCH --job-name=pool-feeder
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=data/1.0.2/pools/feeder-%j.log
set -euo pipefail
pool=$1
uv run python scripts/branch_pool.py feed --pool "$pool" --stop-after 12600 "${@:2}"
if [ ! -e "$pool/DRAINED" ]; then
    sbatch scripts/pool_feeder.sbatch "$@"
fi
