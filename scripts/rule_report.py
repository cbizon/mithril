"""Mine 2-hop rules for one target and write what the top rules / rule sets look like.

Outputs (in --out-dir): scored.tsv (every rule clearing the recall floor, with exact
metrics) and report.md (top-30 by precision at 2% and 3% recall floors, top-30 by F1,
and greedy rule sets with cumulative union recall).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import graphblas as gb

from biolink_rules.problem import load_problem
from biolink_rules.score import Scored, Scorer, greedy_rule_set
from biolink_rules.search import search


def table(rows: list[Scored], extra=None) -> list[str]:
    head = "| # | rule body | overlap | support | precision | recall | F1 |" + (" new | union recall |" if extra else "")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for i, r in enumerate(rows, 1):
        sup = f">{r.support:,}" if r.capped else f"{r.support:,}"
        line = f"| {i} | `{r.name}` | {r.overlap:,} | {sup} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} |"
        if extra:
            new, union = extra[i - 1]
            line += f" {new:,} | {union:.3f} |"
        lines.append(line)
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--target", required=True)
    p.add_argument("--subject-class", required=True)
    p.add_argument("--object-class", required=True)
    p.add_argument("--search-floor", type=float, default=0.01, help="recall floor used to prune the search")
    p.add_argument("--k", type=int, default=30)
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
        support, capped = scorer.support(body, cap=100 * overlap)  # precision < 1% is irrelevant here
        scored.append(Scored(body, name, overlap, support, capped, n_target))
        if i % 500 == 0:
            print(f"[{time.time() - t1:7.0f}s] scored {i:,}/{len(survivors):,}", file=sys.stderr, flush=True)
    print(f"scored {len(scored):,} rules in {time.time() - t1:.0f}s", flush=True)

    with open(args.out_dir / "scored.tsv", "w") as handle:
        handle.write("body\toverlap\tsupport\tsupport_capped\tprecision\trecall\tf1\n")
        for r in sorted(scored, key=lambda r: -r.precision):
            handle.write(f"{r.name}\t{r.overlap}\t{r.support}\t{r.capped}\t{r.precision:.5f}\t{r.recall:.5f}\t{r.f1:.5f}\n")

    out = [f"# 2-hop rules for {problem.target_label}", "",
           f"Target pairs: {n_target:,} (text-mined and negated records excluded). "
           f"Search floor {args.search_floor:.0%} recall; {len(scored):,} rules scored. "
           "Support marked `>` was abandoned once precision was certainly below 1%.", ""]
    for pct in (0.02, 0.03):
        eligible = [r for r in scored if r.recall >= pct and not r.capped]
        top = sorted(eligible, key=lambda r: (-r.precision, -r.overlap))[: args.k]
        out += [f"## Top {args.k} by precision, recall >= {pct:.0%} ({len(eligible):,} eligible)", "", *table(top), ""]
        union = scorer.overlap_pairs(top[0].body) if top else None
        for r in top[1:]:
            union = union.ewise_add(scorer.overlap_pairs(r.body), gb.binary.pair).new()
        if top:
            out += [f"Sum of individual recalls {sum(r.recall for r in top):.3f}; union recall of these "
                    f"{len(top)} rules {union.nvals / n_target:.3f}.", ""]
        chosen = greedy_rule_set(scorer, eligible, args.k, min_new_fraction=0.005)
        out += [f"### Greedy rule set: by precision, recall >= {pct:.0%}, each adding >= 0.5% new target pairs",
                "", *table([c[0] for c in chosen], extra=[(c[1], c[2]) for c in chosen]), ""]
    top_f1 = sorted(scored, key=lambda r: -r.f1)[: args.k]
    out += [f"## Top {args.k} by F1", "", *table(top_f1), ""]
    (args.out_dir / "report.md").write_text("\n".join(out) + "\n")
    print(f"wrote {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
