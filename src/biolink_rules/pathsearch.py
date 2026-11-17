"""Exhaustive rule search over rolled-up path-type tables (see ``provenance``, ``counter``).

Top-down, one position at a time in a fixed order -- direct label, middle classes, then
h1, h2, h3 -- so each body is reached once; a body is expanded only if its overlap clears
``min_support`` (overlap can only shrink under specialization, so the pruning is exact).
Each level only rolls a table up one more position, and the last two levels run in one
compiled loop. Every path length is the same search: unused positions hold the none type
and a one-node lattice. No graph access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from biolink_rules.counter import (
    HLL_M,
    PathTypeCounter,
    _popcount,
    hll_estimate,
    mask_array,
    path_only,
    restrict_pairs,
    with_direct,
)
from biolink_rules.problem import Problem
from biolink_rules.provenance import Quotient


@nb.njit(cache=True)
def _walk_two(row_e2, row_e3, bits, mem_ptr2, mem2, kid_ptr2, kid2, root2, mem_ptr3, mem3, kid_ptr3, kid3, root3,
              n_edge, min_support):
    """Walk the h2 then h3 lattices under a fixed (c1, c2, h1); ``n_edge`` counts the none type. ``bits`` rows are pair
    sets keyed (e2, e3). Returns (h2 node, h3 node, count) of passing bodies in visit
    order (grouped by h2 node), and the number evaluated."""
    width = bits.shape[1]
    n2, n3 = len(mem_ptr2) - 1, len(mem_ptr3) - 1
    use2 = np.zeros(n_edge, dtype=np.bool_)
    table = np.zeros((n_edge, width), dtype=np.uint64)  # one row per e3 for the current h2
    used = np.zeros(n_edge, dtype=np.bool_)
    merged = np.empty(width, dtype=np.uint64)
    seen2 = np.zeros(n2, dtype=np.bool_)
    seen3 = np.zeros(n3, dtype=np.int64)
    stack2 = np.empty(n2, dtype=np.int64)
    stack3 = np.empty(n3, dtype=np.int64)
    out_x = np.empty(64, dtype=np.int32)
    out_y = np.empty(64, dtype=np.int32)
    out_n = np.empty(64, dtype=np.int32)
    n_out, evaluated, stamp = 0, 0, 0
    stack2[0] = root2
    seen2[root2] = True
    top2 = 1
    while top2 > 0:
        top2 -= 1
        x = stack2[top2]
        use2[:] = False
        for j in range(mem_ptr2[x], mem_ptr2[x + 1]):
            use2[mem2[j]] = True
        for e in range(n_edge):
            if used[e]:
                table[e, :] = 0
                used[e] = False
        for i in range(len(row_e2)):
            if use2[row_e2[i]]:
                e = row_e3[i]
                used[e] = True
                for w in range(width):
                    table[e, w] |= bits[i, w]
        stamp += 1
        stack3[0] = root3
        seen3[root3] = stamp
        top3 = 1
        root_passed = False
        while top3 > 0:
            top3 -= 1
            y = stack3[top3]
            merged[:] = 0
            for j in range(mem_ptr3[y], mem_ptr3[y + 1]):
                e = mem3[j]
                if used[e]:
                    for w in range(width):
                        merged[w] |= table[e, w]
            count = 0
            for w in range(width):
                count += _popcount(merged[w])
            evaluated += 1
            if count < min_support:
                continue
            if y == root3:
                root_passed = True
            if n_out == len(out_x):
                out_x = np.concatenate((out_x, out_x))
                out_y = np.concatenate((out_y, out_y))
                out_n = np.concatenate((out_n, out_n))
            out_x[n_out], out_y[n_out], out_n[n_out] = x, y, count
            n_out += 1
            for j in range(kid_ptr3[y], kid_ptr3[y + 1]):
                k = kid3[j]
                if seen3[k] != stamp:
                    seen3[k] = stamp
                    stack3[top3] = k
                    top3 += 1
        if root_passed:
            for j in range(kid_ptr2[x], kid_ptr2[x + 1]):
                k = kid2[j]
                if not seen2[k]:
                    seen2[k] = True
                    stack2[top2] = k
                    top2 += 1
    return out_x[:n_out], out_y[:n_out], out_n[:n_out], evaluated


@nb.njit(cache=True)
def _supports(xs, ys, row_e2, row_e3, reg, mem_ptr2, mem2, mem_ptr3, mem3, n_edge):
    """Distinct-pair estimates for bodies (xs[i], ys[i]) (grouped by x) from HLL rows keyed (e2, e3)."""
    out = np.empty(len(xs), dtype=np.float64)
    use2 = np.zeros(n_edge, dtype=np.bool_)
    table = np.zeros((n_edge, HLL_M), dtype=np.uint8)
    used = np.zeros(n_edge, dtype=np.bool_)
    merged = np.empty(HLL_M, dtype=np.uint8)
    current = -1
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        if x != current:
            current = x
            use2[:] = False
            for j in range(mem_ptr2[x], mem_ptr2[x + 1]):
                use2[mem2[j]] = True
            for e in range(n_edge):
                if used[e]:
                    table[e, :] = 0
                    used[e] = False
            for r in range(len(row_e2)):
                if use2[row_e2[r]]:
                    e = row_e3[r]
                    used[e] = True
                    for w in range(HLL_M):
                        if reg[r, w] > table[e, w]:
                            table[e, w] = reg[r, w]
        merged[:] = 0
        for j in range(mem_ptr3[y], mem_ptr3[y + 1]):
            e = mem3[j]
            if used[e]:
                for w in range(HLL_M):
                    if table[e, w] > merged[w]:
                        merged[w] = table[e, w]
        out[i] = hll_estimate(merged)
    return out


def _csr(lists):
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    flat = np.array([v for x in lists for v in x], dtype=np.int64)
    return ptr, flat


@dataclass
class Mined:
    """Passing bodies of one direct label and one path length, as parallel arrays.

    ``direct`` is a node of the direct lattice (-1: no direct hop); a body's path is
    (c1, c2, n1, n2, n3): class-lattice indices and node indices in the three hop context
    lattices (-1 / 0 in the positions a shorter path does not use).
    """

    direct: int
    hops: int
    c1: np.ndarray
    c2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    n3: np.ndarray
    overlap: np.ndarray
    support: np.ndarray  # estimated distinct predicted pairs (nan if no sketch was given)
    evaluated: int


def mine_paths(problem: Problem, q: Quotient, overlap: PathTypeCounter, support: PathTypeCounter | None,
               min_support: int, n_hops: int, direct: int = -1, progress=None) -> Mined:
    """Every path body of ``n_hops`` (0, 2 or 3) hops whose overlap reaches ``min_support``."""
    classes, contexts = problem.classes, problem.contexts
    a, b = problem.subject_class, problem.object_class
    root_bits = problem.hops.sets[problem.hops.root]
    node_mask = [q.node_mask(s) for s in classes.sets]
    none = ([1 << q.none_edge], [[]], 0, *_csr([[q.none_edge]]), *_csr([[]]))
    lattices: dict[tuple[int, int], tuple] = {}

    def lattice_of(left, right):
        if (left, right) not in lattices:
            lattice, _ = contexts.lattice(left, right)
            kids = lattice.children if contexts.near_duplicate >= 1.0 else contexts.kept_children(left, right)
            masks = [q.edge_mask(s) for s in lattice.sets]
            members = [[e for e in range(q.n_edge) if m >> e & 1] for m in masks]
            lattices[left, right] = (masks, kids, lattice.root, *_csr(members), *_csr(kids))
        return lattices[left, right]

    def hop_ends(c1, c2):
        """(left, right) classes of the three hops; None for an unused hop."""
        if n_hops == 3:
            return (a, c1), (c1, c2), (c2, b)
        return ((a, c1), (c1, b), None) if n_hops == 2 else (None, None, None)

    def linked(c1, c2):
        return all(root_bits & contexts.mask(*ends) for ends in hop_ends(c1, c2) if ends)

    out: list[tuple] = []
    evaluated = 0

    def visit(c1, c2) -> bool:
        """All bodies with these middle classes; True if the all-root body passes."""
        nonlocal evaluated
        k1 = node_mask[c1] if c1 >= 0 else 1 << q.none_node
        k2 = node_mask[c2] if c2 >= 0 else 1 << q.none_node
        (masks1, kids1, root1, *_), (_, _, root2, mem_ptr2, mem2, kid_ptr2, kid2), \
            (_, _, root3, mem_ptr3, mem3, kid_ptr3, kid3) = (lattice_of(*ends) if ends else none
                                                             for ends in hop_ends(c1, c2))
        stack, seen, passed_root = [root1], {root1}, False
        while stack:
            x1 = stack.pop()
            row_e2, row_e3, bits = overlap._by_first(k1, k2, masks1[x1])
            xs, ys, counts, n = _walk_two(row_e2, row_e3, bits, mem_ptr2, mem2, kid_ptr2, kid2, root2,
                                          mem_ptr3, mem3, kid_ptr3, kid3, root3, q.edge_base, min_support)
            evaluated += n
            if not len(xs):
                continue  # (x1, root, root) is evaluated first, so it failed
            if x1 == root1:
                passed_root = True
            if support is not None:
                s_e2, s_e3, reg = support._by_first(k1, k2, masks1[x1])
                est = _supports(xs, ys, s_e2, s_e3, reg, mem_ptr2, mem2, mem_ptr3, mem3, q.edge_base)
            else:
                est = np.full(len(xs), np.nan)
            out.append((c1, c2, x1, xs, ys, counts, est))
            for k in kids1[x1]:
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        if progress:
            progress(c1, c2, evaluated, sum(len(o[3]) for o in out))
        return passed_root

    def descend(visit_class, start):
        """Walk the class lattice from ``start``, going below a class only if it passes."""
        seen, stack = set(start), list(start)
        while stack:
            c = stack.pop()
            if visit_class(c):
                for child in classes.children[c]:
                    if child not in seen:
                        seen.add(child)
                        stack.append(child)

    def first_class(c1) -> bool:
        if not linked(c1, classes.root if n_hops == 3 else -1):
            return False
        if n_hops == 2:
            return visit(c1, -1)
        if not visit(c1, classes.root):
            return False
        descend(lambda c2: linked(c1, c2) and visit(c1, c2), classes.children[classes.root])
        return True

    if n_hops == 0:
        visit(-1, -1)
    else:
        descend(first_class, [classes.root])

    def column(i, dtype, repeat=False):
        if not out:
            return np.zeros(0, dtype=dtype)
        if repeat:
            return np.concatenate([np.full(len(o[3]), o[i], dtype=dtype) for o in out])
        return np.concatenate([o[i] for o in out]).astype(dtype)

    return Mined(direct, n_hops, column(0, np.int32, True), column(1, np.int32, True), column(2, np.int32, True),
                 column(3, np.int32), column(4, np.int32), column(5, np.int32), column(6, np.float64), evaluated)


def direct_lattice(problem: Problem):
    """Labels a direct hop between the endpoints may take: the (subject class, object
    class) hop context with everything the target matches removed, so a rule can never
    use the edge it predicts."""
    return problem.contexts.lattice(problem.subject_class, problem.object_class, exclude=problem.target_bits)[0]


def mine(problem: Problem, q: Quotient, table: tuple, direct: tuple, sketch: tuple | None, min_support: int,
         hops=(2, 3), conditioned: bool = True, progress=None) -> list[Mined]:
    """All rules above ``min_support``: path-only rules of the given lengths and, with
    ``conditioned``, the rules that also require a direct edge (alone = 1-hop rules).

    ``table`` = (path types, tptr, pairs) and ``direct`` = (pair, edge type) describe the
    target pairs (``provenance.pair_types``); ``sketch`` = (keys, tptr, codes) the whole
    graph (``provenance.type_sketches``), or None to skip support estimates.
    """
    n_pairs = len(problem.pairs)
    overlap = PathTypeCounter(q, *table, n_pairs)
    support = PathTypeCounter(q, *path_only(q, *sketch), 0, kind="hll") if sketch else None
    mined = [mine_paths(problem, q, overlap, support, min_support, n, progress=progress) for n in hops]
    lattice = direct_lattice(problem)
    if not conditioned or not lattice.sets:
        return mined
    seen, stack = {lattice.root}, [lattice.root]
    while stack:
        node = stack.pop()
        types = q.edge_mask(lattice.sets[node])
        keep = np.zeros(n_pairs, dtype=bool)
        keep[direct[0][mask_array(types, q.edge_base)[direct[1]]]] = True
        if keep.sum() < min_support:
            continue
        overlap = PathTypeCounter(q, *restrict_pairs(*table, keep), int(keep.sum()))
        support = PathTypeCounter(q, *with_direct(q, *sketch, types), 0, kind="hll") if sketch else None
        mined += [mine_paths(problem, q, overlap, support, min_support, n, node, progress) for n in (0, *hops)]
        for child in lattice.children[node]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return mined


def body_of(problem: Problem, mined: Mined, i: int) -> tuple:
    """Row ``i`` as ``(direct, path)``: ``direct`` is a hop bitset (0 = none); ``path`` is
    ``()``, ``(h1, c1, h2)`` or ``(h1, c1, h2, c2, h3)`` with effective hop bitsets."""
    contexts, a, b = problem.contexts, problem.subject_class, problem.object_class
    direct = direct_lattice(problem).sets[mined.direct] if mined.direct >= 0 else 0
    c1, c2 = int(mined.c1[i]), int(mined.c2[i])
    if mined.hops == 0:
        return direct, ()
    h1 = contexts.lattice(a, c1)[0].sets[mined.n1[i]]
    if mined.hops == 2:
        return direct, (h1, c1, contexts.lattice(c1, b)[0].sets[mined.n2[i]])
    return direct, (h1, c1, contexts.lattice(c1, c2)[0].sets[mined.n2[i]], c2,
                    contexts.lattice(c2, b)[0].sets[mined.n3[i]])


def body_name(problem: Problem, body: tuple) -> str:
    contexts, classes, a, b = problem.contexts, problem.classes, problem.subject_class, problem.object_class
    direct, path = body
    parts = [f"[{contexts.label(direct, a, b, exclude=problem.target_bits)}]"] if direct else []
    if len(path) == 3:
        h1, c1, h2 = path
        parts.append(f"{contexts.label(h1, a, c1)} | {classes.label(c1)} | {contexts.label(h2, c1, b)}")
    elif path:
        h1, c1, h2, c2, h3 = path
        parts.append(f"{contexts.label(h1, a, c1)} | {classes.label(c1)} | {contexts.label(h2, c1, c2)} | "
                     f"{classes.label(c2)} | {contexts.label(h3, c2, b)}")
    return " & ".join(parts)


def body_masks(problem: Problem, q: Quotient, body: tuple) -> tuple:
    """Type masks ``(direct, e1, k1, e2, k2, e3)`` of a body (direct 0 = none)."""
    direct, path = body
    sets = problem.classes.sets
    none_e, none_k = 1 << q.none_edge, 1 << q.none_node
    hops = [q.edge_mask(h) for h in path[0::2]] + [none_e] * (3 - len(path[0::2]))
    nodes = [q.node_mask(sets[c]) for c in path[1::2]] + [none_k] * (2 - len(path[1::2]))
    return q.edge_mask(direct), hops[0], nodes[0], hops[1], nodes[1], hops[2]


_COLUMNS = ("c1", "c2", "n1", "n2", "n3", "overlap", "support")


def save_mined(path, mined: list[Mined], **extra) -> None:
    rows = [len(m.overlap) for m in mined]
    np.savez(path, direct=[m.direct for m in mined], hops=[m.hops for m in mined], rows=rows,
             evaluated=[m.evaluated for m in mined],
             **{c: np.concatenate([getattr(m, c) for m in mined]) for c in _COLUMNS}, **extra)


def load_mined(path) -> list[Mined]:
    z = np.load(path)
    ends = np.cumsum(z["rows"])
    columns = {c: np.split(z[c], ends[:-1]) for c in _COLUMNS}
    return [Mined(int(z["direct"][i]), int(z["hops"][i]), *(columns[c][i] for c in _COLUMNS), int(z["evaluated"][i]))
            for i in range(len(ends))]
