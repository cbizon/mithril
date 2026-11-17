"""Rules for one target from path types (see biolink_rules.provenance).

    types    path types and direct edge types of every target pair -> <run>/types.npz
             (minutes)
    sketch   sketches of all predicted pairs, for one stripe of subjects
             -> <run>/sketch_<i>of<n>.npz   (independent jobs, CPU-hours in all; depends
             only on the config and the endpoint classes, not on the target, so runs
             with the same ones can share the files by symlink)
    search   every rule above the recall floor -- 2- and 3-hop paths, direct edges, and
             direct edge AND path -- with exact overlap and estimated support
             -> <run>/mined.npz, <run>/rules.tsv                              (minutes)
    select   rule sets; chosen rules get exact support -> <run>/select_*.md

The selected set has two separately chosen and separately judged parts:

- **discovery**: path-only rules. They can predict pairs with no edge between the
  endpoints, so per-rule recall floor, new coverage per added rule and union recall
  are meaningful.
- **conditioned**: rules requiring a direct edge (alone, or with a path). They only rank
  pairs that already carry a related edge, so recall says little about them; they are
  ranked by precision with a minimum overlap and a redundancy limit (each kept rule must
  have at least ``--min-novel`` of its overlap not covered by earlier conditioned rules).

Run the steps in that order from the repo root; ``--run`` names the output directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biolink_rules.counter import (
    PathTypeCounter,
    mask_array,
    merge_sketch_tables,
    pair_bits,
    sketch_table,
    type_table,
)
from biolink_rules.pathsearch import body_masks, body_name, body_of, load_mined, mine, save_mined
from biolink_rules.problem import load_problem
from biolink_rules.provenance import count_pairs, pair_types, quotient, type_sketches, typed_graph
from biolink_rules.report import rule_table
from biolink_rules.score import Scored, greedy_rule_set


def open_run(run: Path):
    meta = json.loads((run / "meta.json").read_text())
    problem = load_problem(Path(meta["data"]), Path(meta["config"]), meta["target"], meta["subject_class"],
                           meta["object_class"])
    q = quotient(problem.hops, problem.classes, problem.store.n_bits, problem.store.n_catsets)
    assert len(problem.pairs) == meta["target_pairs"], "graph/config changed since the run was started"
    assert (q.n_edge, q.n_node) == (meta["edge_types"], meta["node_types"]), "lattices changed since the run was started"
    return meta, problem, q


def endpoints(problem):
    """(subject nodes that have edges, object node mask)."""
    store, sets = problem.store, problem.classes.sets
    has_edges = np.zeros(store.n_nodes, dtype=bool)
    has_edges[store.src] = has_edges[store.dst] = True
    return np.flatnonzero(store.in_class(sets[problem.subject_class]) & has_edges), store.in_class(sets[problem.object_class])


def types(args) -> None:
    args.run.mkdir(parents=True, exist_ok=True)
    start = time.time()
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    q = quotient(problem.hops, problem.classes, problem.store.n_bits, problem.store.n_catsets)
    n = len(problem.pairs)
    (args.run / "meta.json").write_text(json.dumps({
        "data": str(args.data), "config": str(args.config), "target": args.target,
        "subject_class": args.subject_class, "object_class": args.object_class,
        "target_pairs": n, "edge_types": q.n_edge, "node_types": q.n_node}))
    pair, path_types, direct_pair, direct_type = pair_types(typed_graph(problem.store, q), q, problem.pairs)
    path_types, tptr, tpairs = type_table(pair, path_types, n)
    np.savez(args.run / "types.npz", types=path_types, tptr=tptr, tpairs=tpairs, direct_pair=direct_pair,
             direct_type=direct_type)
    print(f"{n:,} target pairs, {q.n_edge} edge types, {q.n_node} node types: {len(path_types):,} path types, "
          f"{len(tpairs):,} (pair, type) incidences, {len(np.unique(direct_pair)):,} pairs with a direct edge "
          f"in {time.time() - start:.0f}s", flush=True)


def sketch(args) -> None:
    start = time.time()
    _, problem, q = open_run(args.run)
    subjects, is_object = endpoints(problem)
    subjects = subjects[args.stripe::args.of]
    keys, tptr, codes = sketch_table(*type_sketches(typed_graph(problem.store, q), q, subjects, is_object))
    np.savez(args.run / f"sketch_{args.stripe}of{args.of}.npz", types=keys, tptr=tptr, codes=codes)
    print(f"stripe {args.stripe}/{args.of}: {len(subjects):,} subjects, {len(keys):,} sketches "
          f"in {time.time() - start:.0f}s", flush=True)


def load_sketch(run: Path):
    merged = run / "sketch.npz"
    if not merged.exists():
        parts = sorted(run.glob("sketch_*of*.npz"))
        of = {int(p.stem.split("of")[1]) for p in parts}
        if len(of) != 1 or len(parts) != of.pop():
            raise SystemExit(f"{run}: expected a complete set of sketch stripes, found {[p.name for p in parts]}")
        table = None
        for part in parts:
            z = np.load(part)
            piece = (z["types"], z["tptr"], z["codes"])
            table = piece if table is None else merge_sketch_tables(table, piece)
        np.savez(merged, types=table[0], tptr=table[1], codes=table[2])
    z = np.load(merged)
    return z["types"], z["tptr"], z["codes"]


def load_types(run: Path):
    z = np.load(run / "types.npz")
    return (z["types"], z["tptr"], z["tpairs"]), (z["direct_pair"], z["direct_type"])


def precision_estimate(mined) -> np.ndarray:
    return mined.overlap / np.maximum(mined.support, mined.overlap)


def search(args) -> None:
    start = time.time()
    meta, problem, q = open_run(args.run)
    table, direct = load_types(args.run)
    sketches = load_sketch(args.run)
    print(f"setup {time.time() - start:.0f}s", flush=True)
    start = time.time()
    n = meta["target_pairs"]
    mined = mine(problem, q, table, direct, sketches, max(1, int(args.recall_floor * n)), hops=tuple(args.hops),
                 conditioned=not args.no_conditioned)
    save_mined(args.run / "mined.npz", mined, recall_floor=args.recall_floor)
    rows = [(p, m, i) for m in mined for i, p in enumerate(precision_estimate(m)) if p >= args.min_precision]
    rows.sort(key=lambda row: -row[0])
    with open(args.run / "rules.tsv", "w") as handle:
        handle.write("body\toverlap\tsupport_estimate\tprecision_estimate\trecall\n")
        for p, m, i in rows:
            handle.write(f"{body_name(problem, body_of(problem, m, i))}\t{m.overlap[i]}\t{m.support[i]:.0f}\t"
                         f"{p:.5f}\t{m.overlap[i] / n:.5f}\n")
    for m in mined:
        kind = "path only" if m.direct < 0 else f"direct label {m.direct}"
        print(f"  {kind}, {m.hops} hops: evaluated {m.evaluated:,}, passed {len(m.overlap):,}", flush=True)
    print(f"search {time.time() - start:.0f}s: evaluated {sum(m.evaluated for m in mined):,}, "
          f"{sum(len(m.overlap) for m in mined):,} rules clear {args.recall_floor:.2%} recall, {len(rows):,} of them "
          f"with estimated precision >= {args.min_precision}", flush=True)


def select(args) -> None:
    meta, problem, q = open_run(args.run)
    table, (direct_pair, direct_type) = load_types(args.run)
    mined = load_mined(args.run / "mined.npz")
    n = meta["target_pairs"]
    overlap = PathTypeCounter(q, *table, n)
    graph = typed_graph(problem.store, q)
    subjects, is_object = endpoints(problem)

    def pairs_of(row):
        m, i = row
        masks = body_masks(problem, q, body_of(problem, m, i))
        pairs = overlap.value(*masks[1:])
        if masks[0]:
            having = np.zeros(n, dtype=bool)
            having[direct_pair[mask_array(masks[0], q.edge_base)[direct_type]]] = True
            pairs = pairs & pair_bits(having)
        return pairs

    def candidates(conditioned: bool, floor: float):
        """Rows above ``floor`` recall, by estimated precision then overlap."""
        rows = [(p, int(m.overlap[i]), g, i) for g, m in enumerate(mined) if (m.direct >= 0) == conditioned
                for i, p in enumerate(precision_estimate(m)) if m.overlap[i] >= floor * n]
        rows.sort(key=lambda row: (-row[0], -row[1]))
        return [(mined[g], i) for _, _, g, i in rows]

    def exact(chosen):
        rules = []
        for (m, i), _, _ in chosen:
            body = body_of(problem, m, i)
            support = count_pairs(graph, q, subjects, is_object, body_masks(problem, q, body))
            rules.append(Scored(body, body_name(problem, body), int(m.overlap[i]), support, n))
            print(f"{len(rules)}: {rules[-1].name}  precision {rules[-1].precision:.4f} "
                  f"(estimated {precision_estimate(m)[i]:.4f})", flush=True)
        return rules

    def mean_precision(rules):
        return sum(r.precision for r in rules) / len(rules) if rules else 0.0

    paths, direct = candidates(False, args.recall_floor), candidates(True, args.min_overlap)
    discovery = greedy_rule_set(paths, pairs_of, n, args.k, min_new_fraction=args.min_new)
    conditioned = greedy_rule_set(direct, pairs_of, n, args.k_conditioned, min_novel=args.min_novel)
    discovery_rules, conditioned_rules = exact(discovery), exact(conditioned)
    union_recall = discovery[-1][2] if discovery else 0.0
    lines = [f"# {problem.target_label}: rule set", "",
             "Rules are taken in order of estimated precision; support and precision below are exact.", "",
             f"## Discovery: path-only rules (recall >= {args.recall_floor:.1%}, "
             f"each adds >= {args.min_new:.2%} new target pairs)", "",
             f"{len(paths):,} eligible; {len(discovery)} chosen; union recall {union_recall:.3f}; "
             f"mean precision {mean_precision(discovery_rules):.3f}.", "",
             *rule_table(discovery_rules, set_columns=[c[1:] for c in discovery]), "",
             f"## Conditioned: rules requiring a direct edge (overlap >= {args.min_overlap:.2%} of target, "
             f">= {args.min_novel:.0%} of each rule's overlap not covered by earlier ones)", "",
             "These only score pairs that already have the direct edge; recall is not comparable "
             "to the discovery set and is not combined with it.", "",
             f"{len(direct):,} eligible; {len(conditioned)} chosen; "
             f"mean precision {mean_precision(conditioned_rules):.3f}.", "",
             *rule_table(conditioned_rules)]
    stem = f"select_floor{args.recall_floor:g}_new{args.min_new:g}_novel{args.min_novel:g}"
    (args.run / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.run / stem}.md: {len(discovery)} discovery (union recall {union_recall:.3f}, mean precision "
          f"{mean_precision(discovery_rules):.3f}), {len(conditioned)} conditioned "
          f"(mean precision {mean_precision(conditioned_rules):.3f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("types")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    s.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    s.add_argument("--target", required=True, help="hop label, e.g. 'treats>'")
    s.add_argument("--subject-class", required=True)
    s.add_argument("--object-class", required=True)
    s = sub.add_parser("sketch")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--stripe", type=int, default=0)
    s.add_argument("--of", type=int, default=1)
    s = sub.add_parser("search")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--recall-floor", type=float, default=0.01)
    s.add_argument("--hops", type=int, nargs="+", default=[2, 3], choices=(2, 3), help="path lengths")
    s.add_argument("--no-conditioned", action="store_true", help="skip rules that require a direct edge")
    s.add_argument("--min-precision", type=float, default=0.01, help="rules.tsv lists rules at or above this")
    s = sub.add_parser("select")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--recall-floor", type=float, default=0.01, help="discovery: per-rule recall floor")
    s.add_argument("--min-new", type=float, default=0.005,
                   help="discovery: min new target pairs per rule, fraction of target")
    s.add_argument("--k", type=int, default=20, help="discovery rules")
    s.add_argument("--min-overlap", type=float, default=0.0025,
                   help="conditioned: per-rule overlap floor, fraction of target")
    s.add_argument("--min-novel", type=float, default=0.5,
                   help="conditioned: min fraction of a rule's overlap not covered by earlier conditioned rules")
    s.add_argument("--k-conditioned", type=int, default=10, help="conditioned rules")
    args = p.parse_args()
    {"types": types, "sketch": sketch, "search": search, "select": select}[args.command](args)


if __name__ == "__main__":
    main()
