"""Exhaustive 3-hop search over rolled-up path-type tables (see ``provenance``).

Visits exactly the bodies of the coordinate-ordered search tree (``search.body_space``:
classes first, then h1, h2, h3; a body is expanded only if it clears ``min_support``),
but each level only rolls a table up one more coordinate, and the last two levels run
in one compiled loop. No graph access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from biolink_rules.problem import Problem
from biolink_rules.provenance import HLL_M, PathTypeCounter, Quotient, _popcount, hll_estimate


@nb.njit(cache=True)
def _walk_two(row_e2, row_e3, bits, mem_ptr2, mem2, kid_ptr2, kid2, root2, mem_ptr3, mem3, kid_ptr3, kid3, root3,
              n_edge, min_support):
    """Walk the h2 then h3 lattices under a fixed (c1, c2, h1). ``bits`` rows are pair
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
    """Passing bodies as parallel arrays. A body is (c1, c2, n1, n2, n3): class-lattice
    indices and node indices in the three hop context lattices."""

    c1: np.ndarray
    c2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    n3: np.ndarray
    overlap: np.ndarray
    support: np.ndarray  # estimated distinct predicted pairs (nan if no sketch was given)
    evaluated: int


def mine_three_hop(problem: Problem, q: Quotient, overlap: PathTypeCounter, support: PathTypeCounter | None,
                   min_support: int, progress=None) -> Mined:
    classes, contexts = problem.classes, problem.contexts
    a, b = problem.subject_class, problem.object_class
    root_bits = problem.hops.sets[problem.hops.root]
    node_mask = [q.node_mask(s) for s in classes.sets]
    lattices: dict[tuple[int, int], tuple] = {}

    def lattice_of(left, right):
        if (left, right) not in lattices:
            lattice, _ = contexts.lattice(left, right)
            kids = lattice.children if contexts.near_duplicate >= 1.0 else contexts.kept_children(left, right)
            masks = [q.edge_mask(s) for s in lattice.sets]
            members = [[e for e in range(q.n_edge) if m >> e & 1] for m in masks]
            lattices[left, right] = (masks, kids, lattice.root, *_csr(members), *_csr(kids))
        return lattices[left, right]

    def linked(c1, c2):
        return bool(root_bits & contexts.mask(a, c1) and root_bits & contexts.mask(c1, c2)
                    and root_bits & contexts.mask(c2, b))

    out: list[tuple] = []
    evaluated = 0

    def visit(c1, c2) -> bool:
        """All bodies with these middle classes; True if the all-root body passes."""
        nonlocal evaluated
        k1, k2 = node_mask[c1], node_mask[c2]
        masks1, kids1, root1, *_ = lattice_of(a, c1)
        _, _, root2, mem_ptr2, mem2, kid_ptr2, kid2 = lattice_of(c1, c2)
        _, _, root3, mem_ptr3, mem3, kid_ptr3, kid3 = lattice_of(c2, b)
        stack, seen, passed_root = [root1], {root1}, False
        while stack:
            x1 = stack.pop()
            row_e2, row_e3, bits = overlap._by_first(k1, k2, masks1[x1])
            xs, ys, counts, n = _walk_two(row_e2, row_e3, bits, mem_ptr2, mem2, kid_ptr2, kid2, root2,
                                          mem_ptr3, mem3, kid_ptr3, kid3, root3, q.n_edge, min_support)
            evaluated += n
            if not len(xs):
                continue  # (x1, root, root) is evaluated first, so it failed
            if x1 == root1:
                passed_root = True
            if support is not None:
                s_e2, s_e3, reg = support._by_first(k1, k2, masks1[x1])
                est = _supports(xs, ys, s_e2, s_e3, reg, mem_ptr2, mem2, mem_ptr3, mem3, q.n_edge)
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

    seen1, stack1 = {classes.root}, [classes.root]
    while stack1:
        c1 = stack1.pop()
        if not linked(c1, classes.root):
            continue
        if not visit(c1, classes.root):
            continue
        for c in classes.children[c1]:
            if c not in seen1:
                seen1.add(c)
                stack1.append(c)
        seen2 = {classes.root}
        stack2 = []
        for c in classes.children[classes.root]:
            seen2.add(c)
            stack2.append(c)
        while stack2:
            c2 = stack2.pop()
            if not linked(c1, c2) or not visit(c1, c2):
                continue
            for c in classes.children[c2]:
                if c not in seen2:
                    seen2.add(c)
                    stack2.append(c)

    def column(i, dtype, repeat=False):
        if not out:
            return np.zeros(0, dtype=dtype)
        if repeat:
            return np.concatenate([np.full(len(o[3]), o[i], dtype=dtype) for o in out])
        return np.concatenate([o[i] for o in out]).astype(dtype)

    return Mined(column(0, np.int32, True), column(1, np.int32, True), column(2, np.int32, True),
                 column(3, np.int32), column(4, np.int32), column(5, np.int32), column(6, np.float64), evaluated)


def bodies(problem: Problem, mined: Mined, rows) -> list[tuple]:
    """Search bodies ``(h1, c1, h2, c2, h3)`` (effective hop bitsets) for the given rows."""
    contexts, a, b = problem.contexts, problem.subject_class, problem.object_class
    result = []
    for i in rows:
        c1, c2 = int(mined.c1[i]), int(mined.c2[i])
        result.append((contexts.lattice(a, c1)[0].sets[mined.n1[i]], c1,
                       contexts.lattice(c1, c2)[0].sets[mined.n2[i]], c2,
                       contexts.lattice(c2, b)[0].sets[mined.n3[i]]))
    return result
