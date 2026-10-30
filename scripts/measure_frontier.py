"""Measure the support-pruned search frontier for one target.

Writes every evaluated body (with its support) to a TSV and prints frontier sizes at a
range of support thresholds, derived from a single low-threshold run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from biolink_rules.problem import load_problem
from biolink_rules.search import search


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--target", required=True, help="hop label, e.g. 'treats>'")
    p.add_argument("--subject-class", required=True)
    p.add_argument("--object-class", required=True)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--min-support", type=int, default=None)
    p.add_argument("--recall-floor", type=float, default=None, help="alternative to --min-support")
    p.add_argument("--time-limit", type=float, default=3600)
    p.add_argument("--cache-gb", type=float, default=4.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    n_target = problem.target.nvals
    min_support = args.min_support if args.min_support is not None else int(args.recall_floor * n_target)
    print(f"setup {time.time() - t0:.1f}s; target pairs {n_target:,}; min support {min_support:,}", flush=True)

    stats = search(problem.store, problem.hops, problem.classes, problem.target, problem.subject_class,
                   problem.object_class, args.hops, min_support, args.time_limit, args.cache_gb,
                   contexts=problem.contexts)
    print(f"evaluated {stats.evaluated:,} passed {stats.passed:,} in {stats.seconds:.1f}s "
          f"({stats.evaluated / max(stats.seconds, 1e-9):.1f} evals/s) complete={stats.complete}")

    with open(args.out, "w") as handle:
        handle.write("body\tsupport\tdepth\tparent\n")
        for _, body_name, parent_name, support, depth in stats.rows:
            handle.write(f"{body_name}\t{support}\t{depth}\t{parent_name}\n")

    supports = np.array([r[3] for r in stats.rows])
    support_of = {r[1]: r[3] for r in stats.rows}
    parent_of = np.array([support_of.get(r[2], np.inf) if r[2] else np.inf for r in stats.rows])
    print("threshold  passing  evaluated(=children of passing)")
    for s in sorted({min_support, 2 * min_support, 5 * min_support, 10 * min_support}):
        print(f"{s:>9}  {int((supports >= s).sum()):>7,}  {int((parent_of >= s).sum()):>7,}")


if __name__ == "__main__":
    main()
