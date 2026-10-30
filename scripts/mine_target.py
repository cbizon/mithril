"""Mine 2-hop rules for one target: search above a recall floor, then exact metrics.

Writes (in --out-dir): scored.pkl (rules with bodies, for scripts/select_rules.py)
and scored.tsv (human-readable, sorted by precision).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

from biolink_rules.problem import load_problem
from biolink_rules.score import Scored, Scorer
from biolink_rules.search import search


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--target", required=True)
    p.add_argument("--subject-class", required=True)
    p.add_argument("--object-class", required=True)
    p.add_argument("--search-floor", type=float, default=0.01, help="recall floor used to prune the search")
    p.add_argument("--min-precision", type=float, default=0.01, help="stop counting support below this precision")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    n_target = problem.target.nvals
    print(f"setup {time.time() - t0:.0f}s; target {problem.target_label}: {n_target:,} pairs", flush=True)

    floor = int(args.search_floor * n_target)
    stats = search(problem.store, problem.hops, problem.classes, problem.target, problem.subject_class,
                   problem.object_class, 2, floor, time_limit=6 * 3600, contexts=problem.contexts)
    survivors = [(body, name, sup) for body, name, _, sup, _ in stats.rows if sup >= floor]
    print(f"search: evaluated {stats.evaluated:,}, {len(survivors):,} clear {args.search_floor:.0%} recall "
          f"in {stats.seconds:.0f}s complete={stats.complete}", flush=True)

    scorer = Scorer(problem)
    scored: list[Scored] = []
    t1 = time.time()
    for i, (body, name, overlap) in enumerate(survivors, 1):
        support, capped = scorer.support(body, cap=int(overlap / args.min_precision))
        scored.append(Scored(body, name, overlap, support, capped, n_target))
        if i % 1000 == 0:
            print(f"[{time.time() - t1:7.0f}s] scored {i:,}/{len(survivors):,}", file=sys.stderr, flush=True)
    print(f"scored {len(scored):,} rules in {time.time() - t1:.0f}s", flush=True)

    meta = {"target": args.target, "subject_class": args.subject_class, "object_class": args.object_class,
            "target_label": problem.target_label, "target_pairs": n_target, "search_floor": args.search_floor,
            "min_precision": args.min_precision, "evaluated": stats.evaluated, "complete": stats.complete}
    with open(args.out_dir / "scored.pkl", "wb") as handle:
        pickle.dump({"meta": meta, "rules": scored}, handle)
    with open(args.out_dir / "scored.tsv", "w") as handle:
        handle.write("body\toverlap\tsupport\tsupport_capped\tprecision\trecall\tf1\n")
        for r in sorted(scored, key=lambda r: -r.precision):
            handle.write(f"{r.name}\t{r.overlap}\t{r.support}\t{r.capped}\t{r.precision:.5f}\t{r.recall:.5f}\t{r.f1:.5f}\n")
    print(f"wrote {args.out_dir / 'scored.pkl'}")


if __name__ == "__main__":
    main()
