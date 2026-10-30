#!/bin/bash
#SBATCH --job-name=frontier
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=data/1.0.2/measure/%x-%j.log
# usage: sbatch scripts/measure_frontier.sbatch <out.tsv> <measure_frontier.py args...>
set -euo pipefail
out=$1; shift
uv run python scripts/measure_frontier.py --cache-gb 24 --out "$out" "$@"
