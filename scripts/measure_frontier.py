"""Phase-1 measurement: how big is the support-pruned search frontier for one target?

Writes every evaluated body (with its support) to a TSV and prints frontier sizes at a
range of support thresholds, derived from a single low-threshold run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from biolink_rules.lattice import class_lattice, hop_lattice, read_atoms, read_category_sets
from biolink_rules.search import Store, restrict, search


def find(lattice, label):
    matches = [i for i, labels in enumerate(lattice.labels) if label in labels]
    if len(matches) != 1:
        raise ValueError(f"{label!r} matches {len(matches)} lattice nodes")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--target", required=True, help="hop label, e.g. 'treats>'")
    p.add_argument("--subject-class", required=True)
    p.add_argument("--object-class", required=True)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--min-support", type=int, default=10)
    p.add_argument("--time-limit", type=float, default=3600)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    classes = class_lattice(read_category_sets(args.data / "ingest/category_sets.tsv"))
    atoms = read_atoms(args.data / "build/atoms.tsv")
    hops = hop_lattice(atoms)
    store = Store.load(args.data / "build", args.data / "ingest")
    t_index = find(hops, args.target)
    subject = store.class_vector(classes.sets[find(classes, args.subject_class)])
    object_ = store.class_vector(classes.sets[find(classes, args.object_class)])
    target = restrict(store.hop_matrix(hops.sets[t_index], target_only=True), rows=subject, cols=object_)
    print(f"setup {time.time() - t0:.1f}s; target pairs {target.nvals:,}", flush=True)

    stats = search(store, hops, classes, target, args.hops, args.min_support, args.time_limit)
    print(f"evaluated {stats.evaluated:,} passed {stats.passed:,} in {stats.seconds:.1f}s "
          f"({stats.evaluated / max(stats.seconds, 1e-9):.1f} evals/s) complete={stats.complete}")

    def name(body):
        return " | ".join(
            (classes if (len(body) == 3 and i == 1) else hops).label(c) for i, c in enumerate(body)
        )

    with open(args.out, "w") as handle:
        handle.write("body\tsupport\tdepth\tparent\n")
        for body, parent, support, depth in stats.rows:
            handle.write(f"{name(body)}\t{support}\t{depth}\t{name(parent) if parent else ''}\n")

    supports = np.array([r[2] for r in stats.rows])
    parents_support = {r[0]: r[2] for r in stats.rows}
    parent_of = np.array([parents_support[r[1]] if r[1] else np.inf for r in stats.rows])
    print("threshold  passing  evaluated(=children of passing)")
    for s in sorted({args.min_support, 20, 50, 100, 200, 500, 1000, 5000}):
        if s < args.min_support:
            continue
        # a body is evaluated at threshold s iff its (logged) parent passed s; exact only
        # for the first-visit parent, so 'evaluated' is a close lower bound
        print(f"{s:>9}  {int((supports >= s).sum()):>7,}  {int((parent_of >= s).sum()):>7,}")


if __name__ == "__main__":
    main()
