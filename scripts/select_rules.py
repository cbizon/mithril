"""Select a rule set from a mined target (scripts/mine_target.py output).

The set has two separately selected and separately judged parts:

- **discovery**: path-only rules. They can predict pairs with no edge between the
  endpoints, so per-rule recall floor, new coverage per added rule and union recall
  are meaningful.
- **conditioned**: rules requiring a direct edge (1-hop, or direct AND path). They only
  rank pairs that already carry a related edge, so recall says little about them; they
  are ranked by precision with a minimum overlap and a redundancy limit (each kept rule
  must have at least ``--min-novel`` of its overlap not already covered by earlier
  conditioned rules). No union recall is reported for them, nor for the combined set.

Writes <mined>/select_<variant>.md and a one-line .tsv summary.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from biolink_rules.problem import load_problem
from biolink_rules.report import rule_table
from biolink_rules.score import Scorer, greedy_rule_set, is_conditioned


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--mined", type=Path, required=True, help="directory holding scored.pkl")
    p.add_argument("--recall-floor", type=float, default=0.01, help="discovery: per-rule recall floor")
    p.add_argument("--min-new", type=float, default=0.005,
                   help="discovery: min new target pairs per rule, fraction of target")
    p.add_argument("--k", type=int, default=20, help="discovery rules")
    p.add_argument("--min-overlap", type=float, default=0.0025,
                   help="conditioned: per-rule overlap floor, fraction of target")
    p.add_argument("--min-novel", type=float, default=0.5,
                   help="conditioned: min fraction of a rule's overlap not covered by earlier conditioned rules")
    p.add_argument("--k-conditioned", type=int, default=10, help="conditioned rules")
    args = p.parse_args()

    with open(args.mined / "scored.pkl", "rb") as handle:
        mined = pickle.load(handle)
    meta, rules = mined["meta"], mined["rules"]
    # rebuild exactly the lattices used for mining (bodies reference their indices)
    config = Path(meta.get("config", "config/default.yaml"))
    problem = load_problem(args.data, config, meta["target"], meta["subject_class"], meta["object_class"])
    assert problem.target.nvals == meta["target_pairs"], "graph/config changed since mining"
    assert problem.classes.labels == meta["class_labels"], "class lattice differs from mining; re-mine"
    scorer = Scorer(problem)

    usable = [r for r in rules if not r.capped]
    paths = [r for r in usable if not is_conditioned(r) and r.recall >= args.recall_floor]
    direct = [r for r in usable if is_conditioned(r) and r.recall >= args.min_overlap]
    discovery = greedy_rule_set(scorer, paths, args.k, min_new_fraction=args.min_new)
    conditioned = greedy_rule_set(scorer, direct, args.k_conditioned, min_novel=args.min_novel)

    def mean_precision(chosen):
        return sum(c[0].precision for c in chosen) / len(chosen) if chosen else 0.0

    union_recall = discovery[-1][2] if discovery else 0.0
    stem = f"select_floor{args.recall_floor:g}_new{args.min_new:g}_novel{args.min_novel:g}"
    lines = [f"# {meta['target_label']}: rule set", "",
             f"## Discovery: path-only rules (recall >= {args.recall_floor:.1%}, "
             f"each adds >= {args.min_new:.2%} new target pairs)", "",
             f"{len(paths):,} eligible; {len(discovery)} chosen; union recall {union_recall:.3f}; "
             f"mean precision {mean_precision(discovery):.3f}.", "",
             *rule_table([c[0] for c in discovery], set_columns=[(c[1], c[2]) for c in discovery]), "",
             f"## Conditioned: rules requiring a direct edge (overlap >= {args.min_overlap:.2%} of target, "
             f">= {args.min_novel:.0%} of each rule's overlap not covered by earlier ones)", "",
             "These only score pairs that already have the direct edge; recall is not comparable "
             "to the discovery set and is not combined with it.", "",
             f"{len(direct):,} eligible; {len(conditioned)} chosen; "
             f"mean precision {mean_precision(conditioned):.3f}.", "",
             *rule_table([c[0] for c in conditioned])]
    (args.mined / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (args.mined / f"{stem}.tsv").write_text(
        "recall_floor\tmin_new\tmin_novel\tdiscovery\tunion_recall\tdiscovery_mean_precision\t"
        "discovery_min_precision\tconditioned\tconditioned_mean_precision\n"
        f"{args.recall_floor}\t{args.min_new}\t{args.min_novel}\t{len(discovery)}\t{union_recall:.4f}\t"
        f"{mean_precision(discovery):.4f}\t{min((c[0].precision for c in discovery), default=0):.4f}\t"
        f"{len(conditioned)}\t{mean_precision(conditioned):.4f}\n")
    print(f"wrote {args.mined / stem}.md: {len(discovery)} discovery (union recall {union_recall:.3f}), "
          f"{len(conditioned)} conditioned")


if __name__ == "__main__":
    main()
