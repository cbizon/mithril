"""Evaluate externally-supplied (e.g. ARAGORN/AMIE) TRAPI rule templates with our exact metrics.

Each rule's query_graph is turned into a chain $source - x1 (- x2) - $target. All edges
between two consecutive chain nodes are AND-ed; edges directly between $source and
$target become a mask on the chain product. Edge predicates + qualifier constraints are
mapped to stored atoms with production semantics (``semantics.matched_atoms``).

This rule language (AND-ed parallel edges, non-simple chains) is outside what the path-type
engine mines, so this script carries its own boolean matrix products.

Counts are over distinct (source, target) pairs within the target's class pair, a != d,
with plain boolean products (no simple-path correction for chains of 3 hops).

For rules whose direct source->target edge matches treats edges, a 'non_leaky' variant
removes every atom matched by the target predicate from that direct edge.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import graphblas as gb
import numpy as np

from biolink_rules.lattice import ANY_CLASS, read_atoms
from biolink_rules.problem import find, load_problem
from biolink_rules.semantics import matched_atoms

SOURCE, TARGET = "$source", "$target"


def edge_query(edge: dict) -> tuple[str, tuple[tuple[str, str], ...]]:
    """TRAPI edge -> (predicate, sorted qualifier constraints) query predicate."""
    (predicate,) = edge["predicates"]
    constraints = []
    for constraint in edge.get("qualifier_constraints") or ():
        for q in constraint["qualifier_set"]:
            constraints.append((q["qualifier_type_id"].removeprefix("biolink:"), q["qualifier_value"]))
    return predicate, tuple(sorted(constraints))


def chain_of(query_graph: dict) -> tuple[list[str], dict[tuple[str, str], list[dict]], list[dict]]:
    """(node chain source..target, edges per consecutive pair, direct source-target edges).

    >>> qg = {"nodes": {"$source": {}, "$target": {}, "h": {}}, "edges": {
    ...   "e0": {"subject": "$source", "object": "$target", "predicates": ["p"]},
    ...   "e1": {"subject": "h", "object": "$target", "predicates": ["q"]},
    ...   "e2": {"subject": "$source", "object": "h", "predicates": ["r"]}}}
    >>> chain, pairs, direct = chain_of(qg)
    >>> chain, sorted(pairs), [e["predicates"] for e in direct]
    (['$source', 'h', '$target'], [('$source', 'h'), ('h', '$target')], [['p']])
    """
    direct, adjacency = [], defaultdict(set)
    by_pair: dict[frozenset, list[dict]] = defaultdict(list)
    for edge in query_graph["edges"].values():
        ends = frozenset((edge["subject"], edge["object"]))
        if ends == {SOURCE, TARGET}:
            direct.append(edge)
            continue
        by_pair[ends].append(edge)
        adjacency[edge["subject"]].add(edge["object"])
        adjacency[edge["object"]].add(edge["subject"])
    previous, queue = {SOURCE: None}, deque([SOURCE])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    if TARGET not in previous:
        raise ValueError("no source-target chain")
    chain = [TARGET]
    while chain[-1] != SOURCE:
        chain.append(previous[chain[-1]])
    chain.reverse()
    pairs = {(u, v): by_pair[frozenset((u, v))] for u, v in zip(chain, chain[1:])}
    if sum(len(e) for e in pairs.values()) != sum(len(e) for e in by_pair.values()):
        raise ValueError("rule is not a simple chain (+ direct edges)")
    return chain, pairs, direct


def oriented_bits(edge: dict, left: str, atoms: list[dict], exclude_atoms: set[int] = frozenset()) -> int:
    """Atom-orientation bitset for ``edge`` read from ``left`` to the other node."""
    bits = 0
    for atom, orientation in matched_atoms(edge_query(edge), atoms):
        if atom in exclude_atoms:
            continue
        forward = (orientation == "F") == (edge["subject"] == left)
        bits |= 1 << (2 * atom + (0 if forward else 1))
    return bits


def hop_matrix(store, bits: int) -> gb.Matrix:
    """Boolean N x N matrix of the (atom, orientation) set ``bits``."""
    rows, cols = [], []
    for bit in range(bits.bit_length()):
        if not bits >> bit & 1:
            continue
        atom, reverse = divmod(bit, 2)
        lo, hi = store.offsets[atom], store.offsets[atom + 1]
        s, d = store.src[lo:hi], store.dst[lo:hi]
        rows.append(d if reverse else s)
        cols.append(s if reverse else d)
    rows = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
    cols = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
    return gb.Matrix.from_coo(rows, cols, True, nrows=store.n_nodes, ncols=store.n_nodes)


def node_vector(store, catset_bits: int) -> gb.Vector:
    return gb.Vector.from_coo(np.flatnonzero(store.in_class(catset_bits)), True, size=store.n_nodes)


def edges_matrix(store, edges: list[dict], left: str, atoms, exclude_atoms=frozenset()) -> gb.Matrix:
    """AND of all edges' matrices, oriented left -> right."""
    result = None
    for edge in edges:
        m = hop_matrix(store, oriented_bits(edge, left, atoms, exclude_atoms))
        result = m if result is None else result.ewise_mult(m, gb.binary.pair).new()
    return result


def select_rows(m: gb.Matrix, vector: gb.Vector) -> gb.Matrix:
    return vector.diag().mxm(m, gb.semiring.any_pair).new()


def select_cols(m: gb.Matrix, vector: gb.Vector) -> gb.Matrix:
    return m.mxm(vector.diag(), gb.semiring.any_pair).new()


def evaluate(problem, atoms, rule: dict, exclude_direct_atoms: set[int], cap: int) -> dict:
    query_graph = rule["template"]["query_graph"]
    chain, pairs, direct = chain_of(query_graph)
    store, classes = problem.store, problem.classes

    def class_vector(node: str) -> gb.Vector:
        categories = query_graph["nodes"][node].get("categories") or []
        label = categories[0] if categories else ANY_CLASS
        return node_vector(store, classes.sets[find(classes, label)])

    a_vec = node_vector(store, classes.sets[problem.subject_class])
    b_vec = node_vector(store, classes.sets[problem.object_class])
    mats = []
    for i, (u, v) in enumerate(zip(chain, chain[1:])):
        m = edges_matrix(store, pairs[u, v], u, atoms)
        if i == 0:
            m = select_rows(m, a_vec.ewise_mult(class_vector(u), gb.binary.pair).new())
        right = class_vector(v)
        if v == TARGET:
            right = right.ewise_mult(b_vec, gb.binary.pair).new()
        mats.append(select_cols(m, right))
    mask = None
    if direct:
        mask = edges_matrix(store, direct, SOURCE, atoms, exclude_direct_atoms)
        mask = select_cols(select_rows(mask, a_vec), b_vec)

    target = gb.Matrix.from_coo(problem.pairs[:, 0], problem.pairs[:, 1], True, nrows=store.n_nodes,
                                ncols=store.n_nodes)
    support = overlap = 0
    capped = False
    rows = mats[0].reduce_rowwise(gb.monoid.any).new().to_coo()[0]
    tail = mats[1]
    for m in mats[2:]:
        tail = tail.mxm(m, gb.semiring.any_pair).new()
    block = 2048
    for start in range(0, len(rows), block):
        idx = rows[start:start + block]
        local = gb.Matrix(bool, len(idx), target.ncols)
        if mask is not None:
            local(mask[idx, :].new().S) << mats[0][idx, :].new().mxm(tail, gb.semiring.any_pair)
        else:
            local << mats[0][idx, :].new().mxm(tail, gb.semiring.any_pair)
        diagonal = gb.Matrix.from_coo(np.arange(len(idx)), idx, True, nrows=len(idx), ncols=target.ncols)
        support += local.nvals - local.ewise_mult(diagonal, gb.binary.pair).new().nvals
        overlap += local.ewise_mult(target[idx, :].new(), gb.binary.pair).new().nvals
        if support > cap:
            capped = True
            break
    return {"hops": len(chain) - 1, "support": support, "capped": capped, "overlap": overlap,
            "has_direct": bool(direct)}


def rule_text(rule: dict) -> str:
    query_graph = rule["template"]["query_graph"]
    names = {n: n.lstrip("$") + ":" + ((v.get("categories") or ["?"])[0].removeprefix("biolink:"))
             for n, v in query_graph["nodes"].items()}
    parts = []
    for e in query_graph["edges"].values():
        predicate, constraints = edge_query(e)
        q = "[" + ",".join(f"{f}={v}" for f, v in constraints) + "]" if constraints else ""
        parts.append(f"{names[e['subject']]} -{predicate.removeprefix('biolink:')}{q}-> {names[e['object']]}")
    return " ; ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--key", default='{"predicate": "biolink:treats"}')
    p.add_argument("--data", type=Path, default=Path("data/1.0.2"))
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--target", default="treats>")
    p.add_argument("--subject-class", default="biolink:ChemicalEntity")
    p.add_argument("--object-class", default="biolink:DiseaseOrPhenotypicFeature")
    p.add_argument("--target-predicate", default="biolink:treats")
    p.add_argument("--cap", type=int, default=500_000_000)
    p.add_argument("--substitute", action="append", default=[],
                   help="OLD=NEW predicate substitution; adds a 'sub:NEW' variant for rules using OLD")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.time()
    rules = json.loads(args.rules.read_text())[args.key]
    problem = load_problem(args.data, args.config, args.target, args.subject_class, args.object_class)
    atoms = read_atoms(args.data / "build/atoms.tsv")
    target_atoms = {a for a, _ in matched_atoms((args.target_predicate, ()), atoms)}
    n_target = len(problem.pairs)
    print(f"setup {time.time() - t0:.0f}s; target pairs {n_target:,}", flush=True)

    columns = ["rule", "variant", "aragorn_std_conf", "aragorn_head_cov", "aragorn_pos", "aragorn_body",
               "hops", "support", "support_capped", "overlap", "precision", "recall", "seconds", "rule_text"]
    with open(args.out, "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for i, rule in enumerate(rules, 1):
            variants = [("as_is", set())]
            direct_preds = [e for e in rule["template"]["query_graph"]["edges"].values()
                            if {e["subject"], e["object"]} == {SOURCE, TARGET}]
            if direct_preds and any(a in target_atoms for e in direct_preds
                                    for a, _ in matched_atoms(edge_query(e), atoms)):
                variants.append(("non_leaky", target_atoms))
            for old, new in (x.split("=") for x in args.substitute):
                edges = rule["template"]["query_graph"]["edges"].values()
                if any(e["predicates"] == [old] for e in edges):
                    swapped = json.loads(json.dumps(rule).replace(f'"{old}"', f'"{new}"'))
                    variants.append((f"sub:{new.removeprefix('biolink:')}", set(), swapped))
                    if len(variants) > 1 and variants[1][0] == "non_leaky":
                        variants.append((f"sub:{new.removeprefix('biolink:')}:non_leaky", target_atoms, swapped))
            for name, exclude, *alt in variants:
                t1 = time.time()
                rule_used = alt[0] if alt else rule
                r = evaluate(problem, atoms, rule_used, exclude, args.cap)
                precision = r["overlap"] / r["support"] if r["support"] else 0.0
                row = [i, name, rule["Std Confidence"], rule["Head Coverage"], rule["Positive Examples"],
                       rule["Body size"], r["hops"], r["support"], r["capped"], r["overlap"],
                       f"{precision:.4f}", f"{r['overlap'] / n_target:.4f}", f"{time.time() - t1:.1f}",
                       rule_text(rule_used)]
                handle.write("\t".join(map(str, row)) + "\n")
                handle.flush()
                print(f"rule {i} {name}: support {r['support']:,} overlap {r['overlap']:,} "
                      f"precision {precision:.4f} ({time.time() - t1:.0f}s)", flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
