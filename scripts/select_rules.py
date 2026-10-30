"""Select a rule set from a mined target (scripts/mine_target.py output).

One run = one variant: per-rule recall floor, minimum new-coverage per added rule, and
set size. Writes <out-dir>/select_floor<F>_new<N>.md and a one-line .tsv summary.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from biolink_rules.problem import load_problem
from biolink_rules.report import rule_table
from biolink_rules.score import Scorer, greedy_rule_set


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--mined", type=Path, required=True, help="directory holding scored.pkl")
    p.add_argument("--recall-floor", type=float, required=True)
    p.add_argument("--min-new", type=float, required=True, help="min new target pairs per rule, fraction of target")
    p.add_argument("--k", type=int, default=30)
    args = p.parse_args()

    with open(args.mined / "scored.pkl", "rb") as handle:
        mined = pickle.load(handle)
    meta, rules = mined["meta"], mined["rules"]
    problem = load_problem(args.data, args.config, meta["target"], meta["subject_class"], meta["object_class"])
    assert problem.target.nvals == meta["target_pairs"], "graph/config changed since mining"
    scorer = Scorer(problem)

    eligible = [r for r in rules if r.recall >= args.recall_floor and not r.capped]
    chosen = greedy_rule_set(scorer, eligible, args.k, min_new_fraction=args.min_new)
    stem = f"select_floor{args.recall_floor:g}_new{args.min_new:g}"
    final_recall = chosen[-1][2] if chosen else 0.0
    mean_precision = sum(c[0].precision for c in chosen) / len(chosen) if chosen else 0.0
    lines = [f"# {meta['target_label']}: rule set, recall >= {args.recall_floor:.1%}, "
             f"each rule adds >= {args.min_new:.2%} new target pairs", "",
             f"{len(eligible):,} eligible rules; {len(chosen)} chosen; union recall {final_recall:.3f}; "
             f"mean precision {mean_precision:.3f}.", "",
             *rule_table([c[0] for c in chosen], set_columns=[(c[1], c[2]) for c in chosen])]
    (args.mined / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (args.mined / f"{stem}.tsv").write_text(
        "recall_floor\tmin_new\teligible\tchosen\tunion_recall\tmean_precision\tmin_precision\n"
        f"{args.recall_floor}\t{args.min_new}\t{len(eligible)}\t{len(chosen)}\t{final_recall:.4f}\t"
        f"{mean_precision:.4f}\t{min((c[0].precision for c in chosen), default=0):.4f}\n")
    print(f"wrote {args.mined / stem}.md: {len(chosen)} rules, union recall {final_recall:.3f}")


if __name__ == "__main__":
    main()
