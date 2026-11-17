"""Path-type provenance: walk the graph once, then count every rule body by set union.

A rule body has up to six positions: an optional *direct* hop between the two endpoints,
and a path of 0, 2 or 3 hops, ``(h1, c1, h2)`` or ``(h1, c1, h2, c2, h3)``, whose hops
are hop labels and whose middle nodes are classes. It joins a pair (a, d) iff the pair
has a direct edge in the direct label and a simple path a-b-d / a-b-c-d matching the
path position by position. Whether an edge matches a hop label depends only on which
labels its (atom, orientation) bit belongs to, and whether a node is in a class only on
which classes its category set belongs to. So:

- **Quotient.** Bits with the same label membership are one *edge type*; category sets
  with the same class membership are one *node type*. No body can tell two members of a
  type apart.
- **Path types.** ``(e1, k1, e2, k2, e3)``: the types along one concrete simple path.
  Shorter paths use the "none" type in the unused positions (2 hops: ``k2``, ``e3``; no
  path at all: every position), so one table holds every length.
- **Provenance.** For each target pair, its path types and the types of its direct
  edges, by enumeration from the pair's endpoints (``pair_types``).
- **Counting.** A body is a product of type sets; it joins a pair iff the pair's path
  types meet the product. The pairs joined are a union of pair sets -- exact, because
  unions (unlike counts) roll up (``counter``).
- **Support.** The pairs a body predicts over the whole graph are also a union, but
  those sets are too large to keep. One pass over all subjects keeps a HyperLogLog
  sketch per (direct edge type or none, path type) instead (``type_sketches``);
  sketches roll up exactly as pair sets do. ``count_pairs`` walks the graph for one
  body and gives its support exactly, for the few rules finally chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from mithril.counter import HLL_M, _mix, hll_register
from mithril.lattice import Lattice
from mithril.store import Store


@dataclass
class Quotient:
    """Edge types (classes of atom-orientation bits) and node types (classes of catsets).

    Type ``n_edge`` / ``n_node`` is the "none" type of an unused path position. A packed
    path type is ``(((e1 * K + k1) * E + e2) * K + k2) * E + e3`` with E = ``edge_base``,
    K = ``node_base``; a sketch key adds the direct edge type (or none) on top:
    ``direct * path_base + path type``.
    """

    edge_of_bit: np.ndarray  # bit -> edge type, -1 if no hop label matches the bit
    node_of_catset: np.ndarray  # catset -> node type
    n_edge: int
    n_node: int

    @property
    def none_edge(self) -> int:
        return self.n_edge

    @property
    def none_node(self) -> int:
        return self.n_node

    @property
    def edge_base(self) -> int:
        return self.n_edge + 1

    @property
    def node_base(self) -> int:
        return self.n_node + 1

    @property
    def path_base(self) -> int:
        return self.edge_base**3 * self.node_base**2

    def edge_mask(self, bits: int) -> int:
        """Edge-type bitmask of an atom-orientation bitset."""
        mask = 0
        while bits:
            low = bits & -bits
            e = int(self.edge_of_bit[low.bit_length() - 1])
            if e >= 0:
                mask |= 1 << e
            bits ^= low
        return mask

    def node_mask(self, catsets: int) -> int:
        mask = 0
        while catsets:
            low = catsets & -catsets
            mask |= 1 << int(self.node_of_catset[low.bit_length() - 1])
            catsets ^= low
        return mask

    def unpack(self, types: np.ndarray):
        """(e1, k1, e2, k2, e3) arrays of packed path types."""
        e3 = types % self.edge_base
        rest = types // self.edge_base
        k2 = rest % self.node_base
        rest //= self.node_base
        e2 = rest % self.edge_base
        rest //= self.edge_base
        return rest // self.node_base, rest % self.node_base, e2, k2, e3


def quotient(hops: Lattice, classes: Lattice, n_bits: int, n_catsets: int) -> Quotient:
    def group(sets, n):
        signature = [0] * n
        for i, s in enumerate(sets):
            while s:
                low = s & -s
                signature[low.bit_length() - 1] |= 1 << i
                s ^= low
        ids: dict[int, int] = {}
        out = np.full(n, -1, dtype=np.int32)
        for item, sig in enumerate(signature):
            if sig:
                out[item] = ids.setdefault(sig, len(ids))
        return out, len(ids)

    edge_of_bit, n_edge = group(hops.sets, n_bits)
    node_of_catset, n_node = group(classes.sets, n_catsets)
    return Quotient(edge_of_bit, node_of_catset, n_edge, n_node)


@dataclass
class TypedGraph:
    """Both-direction adjacency. Entry j of node u's row: neighbour ``nbr[j]``, the edge
    type of the hop u -> nbr[j] (``e_out``) and of the hop nbr[j] -> u (``e_in``); -1
    where no hop label matches. A row is sorted by (neighbour, e_out, e_in)."""

    indptr: np.ndarray
    nbr: np.ndarray
    e_out: np.ndarray
    e_in: np.ndarray
    node_type: np.ndarray


def typed_graph(store: Store, q: Quotient) -> TypedGraph:
    atom = np.repeat(np.arange(len(store.offsets) - 1), np.diff(store.offsets))
    fwd, rev = q.edge_of_bit[2 * atom], q.edge_of_bit[2 * atom + 1]
    u = np.concatenate([store.src, store.dst]).astype(np.int64)
    v = np.concatenate([store.dst, store.src]).astype(np.int64)
    e_out = np.concatenate([fwd, rev]).astype(np.int64)
    e_in = np.concatenate([rev, fwd]).astype(np.int64)
    keep = (e_out >= 0) | (e_in >= 0)
    u, v, e_out, e_in = u[keep], v[keep], e_out[keep], e_in[keep]
    n, width = store.n_nodes, q.n_edge + 1
    key = np.unique(((u * n + v) * width + (e_out + 1)) * width + (e_in + 1))
    e_in = (key % width - 1).astype(np.int32)
    key //= width
    e_out = (key % width - 1).astype(np.int32)
    key //= width
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(key // n, minlength=n), out=indptr[1:])
    return TypedGraph(indptr, (key % n).astype(np.int32), e_out, e_in,
                      q.node_of_catset[store.node_catset].astype(np.int32))


@nb.njit(cache=True)
def _grow(arr, need):
    if need <= len(arr):
        return arr
    out = np.empty(max(need, 2 * len(arr)), dtype=arr.dtype)
    out[:len(arr)] = arr
    return out


@nb.njit(cache=True)
def _pair_types(indptr, nbr, e_out, e_in, node_type, subjects, obj_ptr, obj_idx, n_edge, n_node):
    """For subject i and each of its objects (pair index = position in ``obj_idx``): the
    distinct path types of the pair as parallel arrays (pair, type), and the distinct
    types of its direct edges as (pair, edge type)."""
    n = len(indptr) - 1
    eb, kb = n_edge + 1, n_node + 1
    no_path = (((n_edge * kb + n_node) * eb + n_edge) * kb + n_node) * eb + n_edge
    two_tail = n_node * eb + n_edge  # (k2, e3) of a 2-hop path
    near_object = np.zeros(n, dtype=np.int32)  # stamp: c is adjacent to one of the subject's objects
    has_keys = np.zeros(n, dtype=np.int32)  # stamp: c has two-hop keys from this subject
    first_hop = np.zeros(n, dtype=np.int32)  # stamp: b is a neighbour of the subject ...
    run = np.zeros(n, dtype=np.int64)  # ... whose entries in the subject's row start here
    lo = np.zeros(n, dtype=np.int64)
    hi = np.zeros(n, dtype=np.int64)
    out_pair = np.empty(1 << 20, dtype=np.int32)
    out_type = np.empty(1 << 20, dtype=np.int64)
    n_out = 0
    direct_pair = np.empty(1 << 16, dtype=np.int32)
    direct_type = np.empty(1 << 16, dtype=np.int32)
    n_direct = 0
    buf = np.empty(1 << 16, dtype=np.int64)
    for si in range(len(subjects)):
        a, stamp = subjects[si], si + 1
        objs = obj_idx[obj_ptr[si]:obj_ptr[si + 1]]
        for i in range(indptr[a], indptr[a + 1]):
            if first_hop[nbr[i]] != stamp:
                first_hop[nbr[i]] = stamp
                run[nbr[i]] = i
        for d in objs:
            for j in range(indptr[d], indptr[d + 1]):
                if e_in[j] >= 0:
                    near_object[nbr[j]] = stamp
        # two-hop rows (c, key = (e1, k_b, e2), b) for c next to an object
        m = 0
        for i in range(indptr[a], indptr[a + 1]):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            for j in range(indptr[b], indptr[b + 1]):
                c = nbr[j]
                if c != a and e_out[j] >= 0 and near_object[c] == stamp:
                    m += 1
        row_ck = np.empty(m, dtype=np.int64)
        row_b = np.empty(m, dtype=np.int32)
        width = eb * kb * eb
        m = 0
        for i in range(indptr[a], indptr[a + 1]):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            left = (e_out[i] * kb + node_type[b]) * eb
            for j in range(indptr[b], indptr[b + 1]):
                c = nbr[j]
                if c != a and e_out[j] >= 0 and near_object[c] == stamp:
                    row_ck[m] = c * width + left + e_out[j]
                    row_b[m] = b
                    m += 1
        order = np.argsort(row_ck)
        # compact to distinct (c, key): first b and whether another b exists
        key = np.empty(m, dtype=np.int64)
        first_b = np.empty(m, dtype=np.int32)
        multi = np.zeros(m, dtype=np.bool_)
        r = -1
        prev = -1
        for t in range(m):
            ck = row_ck[order[t]]
            b = row_b[order[t]]
            if ck != prev:
                r += 1
                c = ck // width
                if has_keys[c] != stamp:
                    has_keys[c] = stamp
                    lo[c] = r
                hi[c] = r + 1
                key[r] = ck % width
                first_b[r] = b
                prev = ck
            elif b != first_b[r]:
                multi[r] = True
        for oi in range(len(objs)):
            d = objs[oi]
            pair = obj_ptr[si] + oi
            if first_hop[d] == stamp:  # direct edges a -> d
                i = run[d]
                last = -1
                while i < indptr[a + 1] and nbr[i] == d:
                    if e_out[i] >= 0 and e_out[i] != last:
                        last = e_out[i]
                        direct_pair = _grow(direct_pair, n_direct + 1)
                        direct_type = _grow(direct_type, n_direct + 1)
                        direct_pair[n_direct], direct_type[n_direct] = pair, last
                        n_direct += 1
                    i += 1
            buf[0] = no_path
            k = 1
            for j in range(indptr[d], indptr[d + 1]):
                c = nbr[j]
                if c == a or e_in[j] < 0:
                    continue
                if first_hop[c] == stamp:  # 2-hop paths a - c - d
                    i = run[c]
                    last = -1
                    while i < indptr[a + 1] and nbr[i] == c:
                        if e_out[i] >= 0 and e_out[i] != last:
                            last = e_out[i]
                            buf = _grow(buf, k + 1)
                            buf[k] = (((last * kb + node_type[c]) * eb + e_in[j]) * kb * eb) + two_tail
                            k += 1
                        i += 1
                if has_keys[c] != stamp:
                    continue
                tail = node_type[c] * eb + e_in[j]
                buf = _grow(buf, k + hi[c] - lo[c])
                for r in range(lo[c], hi[c]):
                    if multi[r] or first_b[r] != d:
                        buf[k] = key[r] * (kb * eb) + tail
                        k += 1
            types = np.unique(buf[:k])
            out_pair = _grow(out_pair, n_out + len(types))
            out_type = _grow(out_type, n_out + len(types))
            out_pair[n_out:n_out + len(types)] = pair
            out_type[n_out:n_out + len(types)] = types
            n_out += len(types)
    return out_pair[:n_out], out_type[:n_out], direct_pair[:n_direct], direct_type[:n_direct]


def pair_types(graph: TypedGraph, q: Quotient, pairs: np.ndarray):
    """Provenance of ``pairs`` (rows (a, d), sorted by a; no pair has a == d).

    Returns (pair index, packed path type) for every distinct path type of every pair --
    its simple 3-hop paths, its 2-hop paths, and the no-path type that every pair has --
    and (pair index, edge type) for the distinct types of its direct edges a -> d.
    """
    subjects, first = np.unique(pairs[:, 0], return_index=True)
    return _pair_types(graph.indptr, graph.nbr, graph.e_out, graph.e_in, graph.node_type, subjects.astype(np.int64),
                       np.append(first, len(pairs)).astype(np.int64), pairs[:, 1].astype(np.int64), q.n_edge, q.n_node)


# ---------------------------------------------------------------------------------------
# Whole-graph pass: a HyperLogLog sketch per (direct edge type or none, path type) of the
# pairs (a, d) having it, a over the given subjects and d over the object nodes.


@nb.njit(cache=True)
def _slot(keys, slots, key, n_used):
    """Index of ``key`` in an open-addressing table (inserted if new). Returns (index, n_used)."""
    size = len(keys)
    i = np.int64(_mix(np.uint64(key)) & np.uint64(size - 1))
    while True:
        if keys[i] == key:
            return slots[i], n_used
        if keys[i] < 0:
            keys[i] = key
            slots[i] = n_used
            return n_used, n_used + 1
        i = (i + 1) & (size - 1)


@nb.njit(cache=True)
def _mark(keys, slots, reg, n_types, key, idx, rank):
    """Raise register ``idx`` of the sketch of ``key`` to ``rank``. Returns (reg, n_types)."""
    slot, n_types = _slot(keys, slots, key, n_types)
    if n_types > reg.shape[0]:
        grown = np.zeros((2 * reg.shape[0], HLL_M), dtype=np.uint8)
        grown[:reg.shape[0]] = reg
        reg = grown
    if rank > reg[slot, idx]:
        reg[slot, idx] = rank
    return reg, n_types


@nb.njit(cache=True)
def _type_sketches(indptr, nbr, e_out, node_type, is_object, od_ptr, od_e, od_d, subjects, n_edge, n_node,
                   table_bits):
    """``od_*``: per node c, its hops c -> d to object nodes d, sorted by edge type.
    Returns (sketch keys, registers[len(keys), HLL_M])."""
    n = len(indptr) - 1
    eb, kb = n_edge + 1, n_node + 1
    path_base = eb * eb * eb * kb * kb
    no_direct = n_edge * path_base
    no_path = path_base - 1
    two_tail = n_node * eb + n_edge
    keys = np.full(1 << table_bits, -1, dtype=np.int64)
    slots = np.zeros(1 << table_bits, dtype=np.int32)
    reg = np.zeros((1 << 16, HLL_M), dtype=np.uint8)
    n_types = 0
    width = eb * kb * eb
    h_idx = np.empty(1 << 10, dtype=np.int32)
    h_rank = np.empty(1 << 10, dtype=np.uint8)
    hit = np.empty(1 << 10, dtype=np.int64)
    direct = np.zeros(n, dtype=np.int32)  # stamp: d has direct edges from the subject ...
    run = np.zeros(n, dtype=np.int64)  # ... starting here in the subject's row
    for si in range(len(subjects)):
        a, stamp = subjects[si], si + 1
        row_end = indptr[a + 1]
        for i in range(indptr[a], row_end):
            d = nbr[i]
            if e_out[i] < 0 or not is_object[d]:
                continue
            if direct[d] != stamp:
                direct[d] = stamp
                run[d] = i
            if i == run[d] or e_out[i] != e_out[i - 1]:  # no path: the direct edge alone
                idx, rank = hll_register(a, d, n)
                reg, n_types = _mark(keys, slots, reg, n_types, e_out[i] * path_base + no_path, idx, rank)
        # 2-hop paths a - b - d
        for i in range(indptr[a], row_end):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            left = (e_out[i] * kb + node_type[b]) * eb
            for j in range(od_ptr[b], od_ptr[b + 1]):
                d = od_d[j]
                if d == a:
                    continue
                path = (left + od_e[j]) * (kb * eb) + two_tail
                idx, rank = hll_register(a, d, n)
                reg, n_types = _mark(keys, slots, reg, n_types, no_direct + path, idx, rank)
                if direct[d] == stamp:
                    g = run[d]
                    while g < row_end and nbr[g] == d:
                        if g == run[d] or e_out[g] != e_out[g - 1]:
                            reg, n_types = _mark(keys, slots, reg, n_types, e_out[g] * path_base + path, idx, rank)
                        g += 1
        # 3-hop paths a - b - c - d: two-hop rows (c, key = (e1, k_b, e2), b), grouped by c
        m = 0
        for i in range(indptr[a], row_end):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            for j in range(indptr[b], indptr[b + 1]):
                c = nbr[j]
                if c != a and e_out[j] >= 0 and od_ptr[c + 1] > od_ptr[c]:
                    m += 1
        row_ck = np.empty(m, dtype=np.int64)
        row_b = np.empty(m, dtype=np.int32)
        m = 0
        for i in range(indptr[a], row_end):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            left = (e_out[i] * kb + node_type[b]) * eb
            for j in range(indptr[b], indptr[b + 1]):
                c = nbr[j]
                if c != a and e_out[j] >= 0 and od_ptr[c + 1] > od_ptr[c]:
                    row_ck[m] = c * width + left + e_out[j]
                    row_b[m] = b
                    m += 1
        order = np.argsort(row_ck)
        t = 0
        while t < m:
            c = row_ck[order[t]] // width
            lo, hi = od_ptr[c], od_ptr[c + 1]
            if hi - lo > len(h_idx):
                h_idx = np.empty(2 * (hi - lo), dtype=np.int32)
                h_rank = np.empty(2 * (hi - lo), dtype=np.uint8)
                hit = np.empty(2 * (hi - lo), dtype=np.int64)
            for j in range(lo, hi):
                h_idx[j - lo], h_rank[j - lo] = hll_register(a, od_d[j], n)
            tail_k = node_type[c] * eb
            while t < m and row_ck[order[t]] // width == c:
                ck = row_ck[order[t]]
                first_b = row_b[order[t]]
                multi = False
                t += 1
                while t < m and row_ck[order[t]] == ck:
                    if row_b[order[t]] != first_b:
                        multi = True
                    t += 1
                base = (ck % width) * (kb * eb) + tail_k
                j = lo
                while j < hi:
                    e3 = od_e[j]
                    slot, n_types = _slot(keys, slots, no_direct + base + e3, n_types)
                    if n_types > reg.shape[0]:
                        grown = np.zeros((2 * reg.shape[0], HLL_M), dtype=np.uint8)
                        grown[:reg.shape[0]] = reg
                        reg = grown
                    # The hot loop must not reassign ``reg`` (growing it there costs ~5x): pairs with
                    # a direct edge are noted in ``hit`` and marked under their direct types after.
                    r = reg[slot]
                    n_hit = 0
                    while j < hi and od_e[j] == e3:
                        d = od_d[j]
                        if d != a and (multi or d != first_b):
                            if h_rank[j - lo] > r[h_idx[j - lo]]:
                                r[h_idx[j - lo]] = h_rank[j - lo]
                            if direct[d] == stamp:
                                hit[n_hit] = j
                                n_hit += 1
                        j += 1
                    for x in range(n_hit):
                        jj = hit[x]
                        d = od_d[jj]
                        g = run[d]
                        while g < row_end and nbr[g] == d:
                            if g == run[d] or e_out[g] != e_out[g - 1]:
                                reg, n_types = _mark(keys, slots, reg, n_types,
                                                     e_out[g] * path_base + base + e3,
                                                     h_idx[jj - lo], h_rank[jj - lo])
                            g += 1
    types = np.empty(n_types, dtype=np.int64)
    for i in range(len(keys)):
        if keys[i] >= 0:
            types[slots[i]] = keys[i]
    return types, reg[:n_types]


def object_hops(graph: TypedGraph, is_object: np.ndarray):
    """Per node c, its hops to object nodes sorted by edge type: (ptr, edge type, object)."""
    u = np.repeat(np.arange(len(graph.indptr) - 1), np.diff(graph.indptr))
    keep = is_object[graph.nbr] & (graph.e_out >= 0)
    u, e, d = u[keep], graph.e_out[keep], graph.nbr[keep]
    order = np.lexsort((d, e, u))
    ptr = np.zeros(len(graph.indptr), dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=len(graph.indptr) - 1), out=ptr[1:])
    return ptr, e[order].astype(np.int32), d[order].astype(np.int32)


def type_sketches(graph: TypedGraph, q: Quotient, subjects: np.ndarray, is_object: np.ndarray, table_bits: int = 26):
    """(sketch keys, HLL registers) over pairs (a in subjects, d with ``is_object``, a != d).

    A pair with path type t counts under key (none, t) and, for each type g of its direct
    edges a -> d, under (g, t); every pair with a direct edge has the no-path type.
    """
    ptr, e, d = object_hops(graph, is_object)
    return _type_sketches(graph.indptr, graph.nbr, graph.e_out, graph.node_type, is_object, ptr, e, d,
                          subjects.astype(np.int64), q.n_edge, q.n_node, table_bits)


# ---------------------------------------------------------------------------------------
# Exact support of one body, by walking the graph with the body's type sets as filters.


@nb.njit(cache=True)
def _count_pairs(indptr, nbr, e_out, node_type, subjects, is_object, n_hops, use_direct, direct, e1, k1, e2, k2, e3):
    n = len(indptr) - 1
    allowed = np.zeros(n, dtype=np.int32)  # stamp: d has a direct edge of an allowed type
    counted = np.zeros(n, dtype=np.int32)
    reached = np.zeros(n, dtype=np.int32)  # stamp: c is two hops out ...
    first_b = np.zeros(n, dtype=np.int32)  # ... through this b ...
    multi = np.zeros(n, dtype=np.bool_)  # ... and through another one too
    middle = np.empty(n, dtype=np.int32)
    total = 0
    for si in range(len(subjects)):
        a, stamp = subjects[si], si + 1
        if use_direct:
            for i in range(indptr[a], indptr[a + 1]):
                d = nbr[i]
                if e_out[i] >= 0 and direct[e_out[i]] and is_object[d]:
                    allowed[d] = stamp
                    if n_hops == 0 and counted[d] != stamp:
                        counted[d] = stamp
                        total += 1
        if n_hops == 0:
            continue
        n_middle = 0
        for i in range(indptr[a], indptr[a + 1]):
            b = nbr[i]
            if e_out[i] < 0 or not e1[e_out[i]] or not k1[node_type[b]]:
                continue
            for j in range(indptr[b], indptr[b + 1]):
                c = nbr[j]
                if e_out[j] < 0 or not e2[e_out[j]] or c == a:
                    continue
                if n_hops == 2:
                    if is_object[c] and counted[c] != stamp and (not use_direct or allowed[c] == stamp):
                        counted[c] = stamp
                        total += 1
                elif k2[node_type[c]]:
                    if reached[c] != stamp:
                        reached[c] = stamp
                        first_b[c] = b
                        multi[c] = False
                        middle[n_middle] = c
                        n_middle += 1
                    elif first_b[c] != b:
                        multi[c] = True
        for x in range(n_middle):
            c = middle[x]
            for j in range(indptr[c], indptr[c + 1]):
                d = nbr[j]
                if (e_out[j] >= 0 and e3[e_out[j]] and d != a and is_object[d] and counted[d] != stamp
                        and (multi[c] or first_b[c] != d) and (not use_direct or allowed[d] == stamp)):
                    counted[d] = stamp
                    total += 1
    return total


def count_pairs(graph: TypedGraph, q: Quotient, subjects: np.ndarray, is_object: np.ndarray, masks: tuple) -> int:
    """Exact number of distinct pairs (a in subjects, d with ``is_object``, a != d) joined by
    a body given as type masks ``(direct, e1, k1, e2, k2, e3)``: ``direct`` is 0 for none;
    a 2-hop path has the none type as ``k2`` and ``e3``; no path has it everywhere."""
    from mithril.counter import mask_array

    direct, e1, k1, e2, k2, e3 = masks
    none_e, none_k = 1 << q.none_edge, 1 << q.none_node
    n_hops = 0 if e1 == none_e else 2 if e3 == none_e else 3
    assert n_hops or direct, "a body needs a direct hop or a path"
    assert (k2 == none_k) == (n_hops < 3) and (k1 == none_k) == (n_hops == 0)
    edge = [mask_array(m, q.edge_base) for m in (direct, e1, e2, e3)]
    node = [mask_array(m, q.node_base) for m in (k1, k2)]
    return int(_count_pairs(graph.indptr, graph.nbr, graph.e_out, graph.node_type, subjects.astype(np.int64), is_object,
                            n_hops, direct != 0, edge[0], edge[1], node[0], edge[2], node[1], edge[3]))
