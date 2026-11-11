"""3-hop rules for one target from path types (see biolink_rules.provenance).

    provenance  path types of every target pair -> <run>/types.npz           (minutes)
    sketch      path-type sketches of all predicted pairs, for one stripe of subjects
                -> <run>/sketch_<i>of<n>.npz   (independent jobs; a few CPU-hours in all;
                depends only on the config and the endpoint classes, not on the target)
    search      every body above the recall floor, with exact overlap and estimated
                support -> <run>/mined.npz, <run>/rules.tsv                  (minutes)
    select      discovery rule set; chosen rules get exact support -> <run>/select_*.md

Run the steps in that order from the repo root; ``--run`` names the output directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from biolink_rules.pathsearch import Mined, bodies, mine_three_hop
from biolink_rules.problem import load_problem
from biolink_rules.provenance import (
    PathTypeCounter,
    merge_sketch_tables,
    pair_types,
    quotient,
    sketch_table,
    type_sketches,
    type_table,
    typed_graph,
)
from biolink_rules.report import rule_table
from biolink_rules.score import Scored
from biolink_rules.threehop import ThreeHop


def open_run(run: Path):
    meta = json.loads((run / "meta.json").read_text())
    problem = load_problem(Path(meta["data"]), Path(meta["config"]), meta["target"], meta["subject_class"],
                           meta["object_class"])
    store = problem.store
    q = quotient(problem.hops, problem.classes, 2 * (len(store.offsets) - 1), int(store.node_catset.max()) + 1)
    assert problem.target.nvals == meta["target_pairs"], "graph/config changed since the run was started"
    assert (q.n_edge, q.n_node) == (meta["edge_types"], meta["node_types"]), "lattices changed since the run was started"
    return meta, problem, q


def provenance(args) -> None:
    args.run.mkdir(parents=True, exist_ok=True)
    start = time.time()
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    store = problem.store
    q = quotient(problem.hops, problem.classes, 2 * (len(store.offsets) - 1), int(store.node_catset.max()) + 1)
    rows, cols, _ = problem.target.to_coo()  # sorted by row: pair index = position
    (args.run / "meta.json").write_text(json.dumps({
        "data": str(args.data), "config": str(args.config), "target": args.target,
        "subject_class": args.subject_class, "object_class": args.object_class,
        "target_pairs": len(rows), "edge_types": q.n_edge, "node_types": q.n_node}))
    graph = typed_graph(store, q)
    subjects, first = np.unique(rows, return_index=True)
    pair, types = pair_types(graph, q, subjects, np.append(first, len(rows)), cols)
    types, tptr, tpairs = type_table(pair, types, len(rows))
    np.savez(args.run / "types.npz", types=types, tptr=tptr, tpairs=tpairs)
    print(f"{len(rows):,} target pairs, {q.n_edge} edge types, {q.n_node} node types: {len(types):,} path types, "
          f"{len(tpairs):,} (pair, type) incidences in {time.time() - start:.0f}s", flush=True)


def sketch(args) -> None:
    start = time.time()
    _, problem, q = open_run(args.run)
    store, classes = problem.store, problem.classes
    graph = typed_graph(store, q)
    subjects = store.class_vector(classes.sets[problem.subject_class]).to_coo()[0]
    subjects = subjects[np.diff(graph.indptr)[subjects] > 0][args.stripe::args.of]
    is_object = np.zeros(store.n_nodes, dtype=bool)
    is_object[store.class_vector(classes.sets[problem.object_class]).to_coo()[0]] = True
    types, tptr, codes = sketch_table(*type_sketches(graph, q, subjects, is_object))
    np.savez(args.run / f"sketch_{args.stripe}of{args.of}.npz", types=types, tptr=tptr, codes=codes)
    print(f"stripe {args.stripe}/{args.of}: {len(subjects):,} subjects, {len(types):,} path types "
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


def counters(run: Path, meta: dict, q):
    z = np.load(run / "types.npz")
    overlap = PathTypeCounter(q, z["types"], z["tptr"], z["tpairs"], meta["target_pairs"])
    support = PathTypeCounter(q, *load_sketch(run), 0, kind="hll")
    return overlap, support


def body_name(problem, body) -> str:
    contexts, classes, a, b = problem.contexts, problem.classes, problem.subject_class, problem.object_class
    h1, c1, h2, c2, h3 = body
    return (f"{contexts.label(h1, a, c1)} | {classes.label(c1)} | {contexts.label(h2, c1, c2)} | "
            f"{classes.label(c2)} | {contexts.label(h3, c2, b)}")


def search(args) -> None:
    start = time.time()
    meta, problem, q = open_run(args.run)
    overlap, support = counters(args.run, meta, q)
    print(f"setup {time.time() - start:.0f}s", flush=True)
    start = time.time()
    min_support = max(1, int(args.recall_floor * meta["target_pairs"]))
    mined = mine_three_hop(problem, q, overlap, support, min_support)
    np.savez(args.run / "mined.npz", c1=mined.c1, c2=mined.c2, n1=mined.n1, n2=mined.n2, n3=mined.n3,
             overlap=mined.overlap, support=mined.support, evaluated=mined.evaluated, recall_floor=args.recall_floor)
    precision = mined.overlap / np.maximum(mined.support, mined.overlap)
    keep = np.flatnonzero(precision >= args.min_precision)
    keep = keep[np.argsort(-precision[keep], kind="stable")]
    with open(args.run / "rules.tsv", "w") as handle:
        handle.write("body\toverlap\tsupport_estimate\tprecision_estimate\trecall\n")
        for i, body in zip(keep, bodies(problem, mined, keep)):
            handle.write(f"{body_name(problem, body)}\t{mined.overlap[i]}\t{mined.support[i]:.0f}\t"
                         f"{precision[i]:.5f}\t{mined.overlap[i] / meta['target_pairs']:.5f}\n")
    print(f"search {time.time() - start:.0f}s: evaluated {mined.evaluated:,}, {len(mined.overlap):,} bodies clear "
          f"{args.recall_floor:.2%} recall, {len(keep):,} of them with estimated precision >= {args.min_precision}",
          flush=True)


def select(args) -> None:
    meta, problem, q = open_run(args.run)
    overlap, _ = counters(args.run, meta, q)
    z = np.load(args.run / "mined.npz")
    mined = Mined(z["c1"], z["c2"], z["n1"], z["n2"], z["n3"], z["overlap"], z["support"], int(z["evaluated"]))
    n_target = meta["target_pairs"]
    precision = mined.overlap / np.maximum(mined.support, mined.overlap)
    eligible = np.flatnonzero(mined.overlap >= args.recall_floor * n_target)
    order = eligible[np.lexsort((-mined.overlap[eligible], -precision[eligible]))]
    three = ThreeHop(problem.store, problem.classes.sets, problem.subject_class, problem.object_class)
    classes = problem.classes
    min_new = int(np.ceil(args.min_new * n_target))
    covered = np.zeros((n_target + 63) // 64, dtype=np.uint64)
    chosen, columns = [], []
    for i in order:
        (body,) = bodies(problem, mined, [i])
        h1, c1, h2, c2, h3 = body
        pairs = overlap.value(q.edge_mask(h1), q.node_mask(classes.sets[c1]), q.edge_mask(h2),
                              q.node_mask(classes.sets[c2]), q.edge_mask(h3))
        new = int(np.bitwise_count(pairs & ~covered).sum())
        if new < min_new:
            continue
        covered |= pairs
        exact, _ = three.support(body, cap=10**12)
        chosen.append((Scored((0, body), body_name(problem, body), int(mined.overlap[i]), exact, False, n_target),
                       float(mined.support[i])))
        columns.append((new, int(np.bitwise_count(covered).sum()) / n_target))
        print(f"{len(chosen)}: {chosen[-1][0].name}  precision {chosen[-1][0].precision:.4f} "
              f"(estimated {precision[i]:.4f})", flush=True)
        if len(chosen) == args.k:
            break
    rules = [c[0] for c in chosen]
    mean = sum(r.precision for r in rules) / len(rules) if rules else 0.0
    union = columns[-1][1] if columns else 0.0
    lines = [f"# {problem.target_label}: 3-hop discovery rules", "",
             f"Path-only rules with recall >= {args.recall_floor:.1%}, taken in order of estimated precision; each adds "
             f">= {args.min_new:.2%} new target pairs. Support and precision below are exact.", "",
             f"{len(eligible):,} eligible; {len(rules)} chosen; union recall {union:.3f}; mean precision {mean:.3f}.", "",
             *rule_table(rules, set_columns=columns), "",
             "Estimated vs exact support of the chosen rules: " +
             ", ".join(f"{est:,.0f}/{r.support:,}" for r, est in chosen)]
    stem = f"select_floor{args.recall_floor:g}_new{args.min_new:g}"
    (args.run / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.run / stem}.md: {len(rules)} rules, union recall {union:.3f}, mean precision {mean:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("provenance")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    s.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    s.add_argument("--target", required=True)
    s.add_argument("--subject-class", required=True)
    s.add_argument("--object-class", required=True)
    s = sub.add_parser("sketch")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--stripe", type=int, default=0)
    s.add_argument("--of", type=int, default=1)
    s = sub.add_parser("search")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--recall-floor", type=float, default=0.01)
    s.add_argument("--min-precision", type=float, default=0.01, help="rules.tsv lists bodies at or above this")
    s = sub.add_parser("select")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--recall-floor", type=float, default=0.01)
    s.add_argument("--min-new", type=float, default=0.005)
    s.add_argument("--k", type=int, default=20)
    args = p.parse_args()
    {"provenance": provenance, "sketch": sketch, "search": search, "select": select}[args.command](args)


if __name__ == "__main__":
    main()
