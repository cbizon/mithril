"""Typed-path provenance: walk the graph once, then count every rule body by set union.

A 3-hop body ``(h1, c1, h2, c2, h3)`` joins a pair (a, d) iff some simple path
a-b-c-d has its first edge in h1, b in c1, second edge in h2, c in c2 and third edge in
h3. Whether an edge is in a hop label depends only on which labels its
(atom, orientation) bit belongs to, and whether a node is in a class only on which
classes its category set belongs to. So:

- **Quotient.** Bits with the same label membership are one *edge type*; category sets
  with the same class membership are one *node type*. No body can tell two members of a
  type apart.
- **Provenance.** For each pair, the set of *path types* ``(e1, k1, e2, k2, e3)`` of its
  simple paths. This is computed once, by enumeration from the pair's endpoints.
- **Counting.** A body is a product of type sets ``E1 x K1 x E2 x K2 x E3``; it joins
  the pair iff the pair's provenance meets the product. The number of pairs joined is
  the size of a union of pair sets -- exact, because unions (unlike counts) roll up.

- **Support.** How many pairs a body predicts over the whole graph is also the size of
  a union (of the pair sets of its path types), but those sets are too large to keep.
  One pass over all subjects keeps a HyperLogLog sketch per path type instead; sketches
  roll up by register-wise max exactly as pair sets roll up by union, giving a
  distinct-pair estimate (about 3% error) for every body. Exact support is then only
  needed for the few rules finally chosen (``threehop.ThreeHop.support``).

The graph is touched only in the two enumeration passes; every body evaluation
afterwards is bit operations on small tables (``PathTypeCounter``, ``pathsearch``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from biolink_rules.lattice import Lattice
from biolink_rules.search import Store


@dataclass
class Quotient:
    """Edge types (classes of atom-orientation bits) and node types (classes of catsets)."""

    edge_of_bit: np.ndarray  # bit -> edge type, -1 if no hop label matches the bit
    node_of_catset: np.ndarray  # catset -> node type
    n_edge: int
    n_node: int

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
    type of the hop u -> nbr[j] (``e_out``) and of the hop nbr[j] -> u (``e_in``)."""

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
    """For subject i and each of its objects (pair index = position in ``obj_idx``), the
    distinct path types of simple paths a-b-c-d, as parallel arrays (pair, type)."""
    n = len(indptr) - 1
    near_object = np.zeros(n, dtype=np.int32)  # stamp: c is adjacent to one of the subject's objects
    has_keys = np.zeros(n, dtype=np.int32)  # stamp: c has two-hop keys from this subject
    lo = np.zeros(n, dtype=np.int64)
    hi = np.zeros(n, dtype=np.int64)
    out_pair = np.empty(1 << 20, dtype=np.int32)
    out_type = np.empty(1 << 20, dtype=np.int64)
    n_out = 0
    buf = np.empty(1 << 16, dtype=np.int64)
    for si in range(len(subjects)):
        a, stamp = subjects[si], si + 1
        objs = obj_idx[obj_ptr[si]:obj_ptr[si + 1]]
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
        width = n_edge * n_node * n_edge
        m = 0
        for i in range(indptr[a], indptr[a + 1]):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            left = (e_out[i] * n_node + node_type[b]) * n_edge
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
            k = 0
            for j in range(indptr[d], indptr[d + 1]):
                c = nbr[j]
                if c == a or e_in[j] < 0 or has_keys[c] != stamp:
                    continue
                tail = node_type[c] * n_edge + e_in[j]
                buf = _grow(buf, k + hi[c] - lo[c])
                for r in range(lo[c], hi[c]):
                    if multi[r] or first_b[r] != d:
                        buf[k] = key[r] * (n_node * n_edge) + tail
                        k += 1
            if k == 0:
                continue
            types = np.unique(buf[:k])
            out_pair = _grow(out_pair, n_out + len(types))
            out_type = _grow(out_type, n_out + len(types))
            out_pair[n_out:n_out + len(types)] = obj_ptr[si] + oi
            out_type[n_out:n_out + len(types)] = types
            n_out += len(types)
    return out_pair[:n_out], out_type[:n_out]


def pair_types(graph: TypedGraph, q: Quotient, subjects: np.ndarray, obj_ptr: np.ndarray, obj_idx: np.ndarray):
    """Path types per pair. Pairs are (subjects[i], obj_idx[j]) for j in
    ``obj_ptr[i]:obj_ptr[i+1]``; returns (pair index j, packed type) arrays, distinct.
    A packed type is ``(((e1 * K + k1) * E + e2) * K + k2) * E + e3``."""
    return _pair_types(graph.indptr, graph.nbr, graph.e_out, graph.e_in, graph.node_type,
                       subjects.astype(np.int64), obj_ptr.astype(np.int64), obj_idx.astype(np.int64),
                       q.n_edge, q.n_node)


def unpack(types: np.ndarray, q: Quotient):
    """(e1, k1, e2, k2, e3) arrays of packed types."""
    e3 = types % q.n_edge
    rest = types // q.n_edge
    k2 = rest % q.n_node
    rest //= q.n_node
    e2 = rest % q.n_edge
    rest //= q.n_edge
    return rest // q.n_node, rest % q.n_node, e2, k2, e3


# ---------------------------------------------------------------------------------------
# Counting: every body is a union of pair sets, rolled up one coordinate at a time.


@nb.njit(cache=True)
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@nb.njit(cache=True)
def _merge_groups(order, group, tptr, tpairs, n_pairs):
    """Union the pair lists of the types ``order`` (sorted by ``group``) per group.
    Returns (group value per output row, row pointers, pairs)."""
    total = 0
    for t in order:
        total += tptr[t + 1] - tptr[t]
    pairs = np.empty(total, dtype=tpairs.dtype)
    values = np.empty(len(order), dtype=np.int64)
    ptr = np.empty(len(order) + 1, dtype=np.int64)
    seen = np.full(n_pairs, -1, dtype=np.int64)
    rows, k = 0, 0
    ptr[0] = 0
    for i in range(len(order)):
        t = order[i]
        if i == 0 or group[t] != values[rows - 1]:
            values[rows] = group[t]
            rows += 1
        for j in range(tptr[t], tptr[t + 1]):
            p = tpairs[j]
            if seen[p] != rows:
                seen[p] = rows
                pairs[k] = p
                k += 1
        ptr[rows] = k
    return values[:rows], ptr[:rows + 1], pairs[:k]


@nb.njit(cache=True)
def _dense_groups(use, group, ptr, pairs, n_words):
    """Dense pair bitsets, one per distinct ``group`` value among rows with ``use`` set
    (rows are sorted by group). Returns (group values, bits[rows, n_words])."""
    rows = 0
    prev = -1
    for i in range(len(use)):
        if use[i] and group[i] != prev:
            rows += 1
            prev = group[i]
    values = np.empty(rows, dtype=np.int64)
    bits = np.zeros((rows, n_words), dtype=np.uint64)
    r = -1
    prev = -1
    for i in range(len(use)):
        if not use[i]:
            continue
        if group[i] != prev:
            r += 1
            prev = group[i]
            values[r] = prev
        for j in range(ptr[i], ptr[i + 1]):
            p = pairs[j]
            bits[r, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
    return values, bits


@nb.njit(cache=True)
def _or_groups(use, group, bits, n_groups):
    """bits rows OR-ed per ``group`` value (0..n_groups-1) over rows with ``use`` set."""
    out = np.zeros((n_groups, bits.shape[1]), dtype=np.uint64)
    for i in range(len(use)):
        if use[i]:
            g = group[i]
            for w in range(bits.shape[1]):
                out[g, w] |= bits[i, w]
    return out


@nb.njit(cache=True)
def _union_count(bits, rows):
    total = 0
    for w in range(bits.shape[1]):
        x = np.uint64(0)
        for r in rows:
            x |= bits[r, w]
        total += _popcount(x)
    return total


@nb.njit(cache=True)
def _union(bits, rows):
    out = np.zeros(bits.shape[1], dtype=np.uint64)
    for r in rows:
        for w in range(bits.shape[1]):
            out[w] |= bits[r, w]
    return out


def mask_array(mask: int, n: int) -> np.ndarray:
    return np.array([mask >> i & 1 for i in range(n)], dtype=np.bool_)


class _Lru:
    def __init__(self, size: int):
        self.size, self.data = size, {}
        self.misses = 0

    def get(self, key, build):
        if key in self.data:
            self.data[key] = self.data.pop(key)
            return self.data[key]
        self.misses += 1
        value = self.data[key] = build()
        if len(self.data) > self.size:
            del self.data[next(iter(self.data))]
        return value


class PathTypeCounter:
    """Roll-up of per-path-type summaries to any 3-hop body.

    ``types`` are distinct packed path types; ``tptr``/``items`` hold a summary per type.
    Two kinds, both closed under union so they roll up exactly the same way:

    - ``kind="pairs"``: items are pair indices (< ``n``); a body's value is the exact
      number of those pairs it joins (``pairs`` gives the pair bitset).
    - ``kind="hll"``: items are HLL codes ``index | rank << HLL_P`` of the (a, d) pairs
      of the type; a body's value estimates how many distinct pairs it predicts.

    A body is five type masks ``(E1, K1, E2, K2, E3)`` (Python ints). Rolled-up tables
    are cached per prefix in the search's coordinate order: (K1, K2) -> rows keyed
    (e1, e2, e3); + E1 -> dense rows keyed (e2, e3); + E2 -> one dense row per e3; a
    body is then a union of a few rows.
    """

    def __init__(self, q: Quotient, types: np.ndarray, tptr: np.ndarray, items: np.ndarray, n: int,
                 kind: str = "pairs", cache=(4, 32, 2048)):
        assert kind in ("pairs", "hll")
        self.q, self.kind = q, kind
        self.n = n if kind == "pairs" else HLL_M
        self.width = (n + 63) // 64 if kind == "pairs" else HLL_M
        self.e1, self.k1, self.e2, self.k2, self.e3 = (x.astype(np.int64) for x in unpack(types, q))
        self.tptr, self.items = tptr, items
        # e1 varies fastest, so rows keyed (e1, e2, e3) come out with (e2, e3) runs contiguous
        self._group = (self.e2 * q.n_edge + self.e3) * q.n_edge + self.e1
        self._by_group = np.argsort(self._group, kind="stable")
        self._classes, self._first, self._second = (_Lru(size) for size in cache)
        self._edge_rows: dict[int, np.ndarray] = {}
        self._edge_use: dict[int, np.ndarray] = {}
        self._node_use: dict[int, np.ndarray] = {}

    def _use(self, cache, mask, n):
        if mask not in cache:
            cache[mask] = mask_array(mask, n)
        return cache[mask]

    def _by_classes(self, k1: int, k2: int):
        def build():
            q = self.q
            sel = (self._use(self._node_use, k1, q.n_node)[self.k1]
                   & self._use(self._node_use, k2, q.n_node)[self.k2])
            order = self._by_group[sel[self._by_group]]
            merge = _merge_groups if self.kind == "pairs" else _merge_groups_hll
            values, ptr, items = merge(order, self._group, self.tptr, self.items, self.n)
            return values % q.n_edge, values // q.n_edge, ptr, items

        return self._classes.get((k1, k2), build)

    def _by_first(self, k1: int, k2: int, e1: int):
        def build():
            row_e1, row_e23, ptr, items = self._by_classes(k1, k2)
            use = self._use(self._edge_use, e1, self.q.n_edge)[row_e1]
            dense = _dense_groups if self.kind == "pairs" else _dense_groups_hll
            values, rows = dense(use, row_e23, ptr, items, self.width)
            return values // self.q.n_edge, values % self.q.n_edge, rows

        return self._first.get((k1, k2, e1), build)

    def _by_second(self, k1: int, k2: int, e1: int, e2: int):
        def build():
            row_e2, row_e3, rows = self._by_first(k1, k2, e1)
            use = self._use(self._edge_use, e2, self.q.n_edge)[row_e2]
            combine = _or_groups if self.kind == "pairs" else _max_groups
            return combine(use, row_e3, rows, self.q.n_edge)

        return self._second.get((k1, k2, e1, e2), build)

    def _rows(self, e3: int) -> np.ndarray:
        if e3 not in self._edge_rows:
            self._edge_rows[e3] = np.flatnonzero(mask_array(e3, self.q.n_edge))
        return self._edge_rows[e3]

    def count(self, e1: int, k1: int, e2: int, k2: int, e3: int) -> float:
        table, rows = self._by_second(k1, k2, e1, e2), self._rows(e3)
        return int(_union_count(table, rows)) if self.kind == "pairs" else float(_union_estimate(table, rows))

    def value(self, e1: int, k1: int, e2: int, k2: int, e3: int) -> np.ndarray:
        """The body's summary itself: pair bitset (uint64 words) or HLL registers."""
        table, rows = self._by_second(k1, k2, e1, e2), self._rows(e3)
        return _union(table, rows) if self.kind == "pairs" else _max_rows(table, rows)


def type_table(pair: np.ndarray, types: np.ndarray, n_pairs: int):
    """(distinct types, tptr, tpairs) from parallel (pair, type) incidence arrays."""
    order = np.argsort(types, kind="stable")
    distinct, start = np.unique(types[order], return_index=True)
    tptr = np.append(start, len(types)).astype(np.int64)
    dtype = np.uint16 if n_pairs <= 1 << 16 else np.int32
    return distinct, tptr, pair[order].astype(dtype)


# ---------------------------------------------------------------------------------------
# Support: the pairs a body predicts over the whole graph. Per path type, a HyperLogLog
# sketch of its (a, d) pairs; sketches roll up by register-wise max exactly as pair sets
# roll up by union, so every body gets a distinct-pair estimate from the same tables.

HLL_P = 10  # 1024 registers: ~3% standard error
HLL_M = 1 << HLL_P


@nb.njit(cache=True)
def _mix(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True)
def hll_register(a, d, n_nodes):
    """(register index, rank) of the pair (a, d)."""
    h = _mix(np.uint64(a) * np.uint64(n_nodes) + np.uint64(d))
    rest = h >> np.uint64(HLL_P)
    rank = 1
    while rest & np.uint64(1) == 0 and rank <= 64 - HLL_P:
        rank += 1
        rest >>= np.uint64(1)
    return np.int64(h & np.uint64(HLL_M - 1)), rank


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
def _type_sketches(indptr, nbr, e_out, node_type, od_ptr, od_e, od_d, subjects, n_edge, n_node, table_bits):
    """HLL registers per path type over all (a, d): a in ``subjects``, d an object.

    ``od_*``: per node c, its hops c -> d to object nodes d, sorted by edge type.
    Returns (types, registers[len(types), HLL_M]).
    """
    n = len(indptr) - 1
    keys = np.full(1 << table_bits, -1, dtype=np.int64)
    slots = np.zeros(1 << table_bits, dtype=np.int32)
    reg = np.zeros((1 << 16, HLL_M), dtype=np.uint8)
    n_types = 0
    width = n_edge * n_node * n_edge
    h_idx = np.empty(1 << 10, dtype=np.int32)
    h_rank = np.empty(1 << 10, dtype=np.uint8)
    for si in range(len(subjects)):
        a = subjects[si]
        m = 0
        for i in range(indptr[a], indptr[a + 1]):
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
        for i in range(indptr[a], indptr[a + 1]):
            if e_out[i] < 0:
                continue
            b = nbr[i]
            left = (e_out[i] * n_node + node_type[b]) * n_edge
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
            for j in range(lo, hi):
                h_idx[j - lo], h_rank[j - lo] = hll_register(a, od_d[j], n)
            tail_k = node_type[c] * n_edge
            while t < m and row_ck[order[t]] // width == c:
                ck = row_ck[order[t]]
                first_b = row_b[order[t]]
                multi = False
                t += 1
                while t < m and row_ck[order[t]] == ck:
                    if row_b[order[t]] != first_b:
                        multi = True
                    t += 1
                base = (ck % width) * (n_node * n_edge) + tail_k
                j = lo
                while j < hi:
                    e3 = od_e[j]
                    slot, n_types = _slot(keys, slots, base + e3, n_types)
                    if n_types > reg.shape[0]:
                        grown = np.zeros((2 * reg.shape[0], HLL_M), dtype=np.uint8)
                        grown[:reg.shape[0]] = reg
                        reg = grown
                    r = reg[slot]
                    while j < hi and od_e[j] == e3:
                        d = od_d[j]
                        if d != a and (multi or d != first_b):
                            if h_rank[j - lo] > r[h_idx[j - lo]]:
                                r[h_idx[j - lo]] = h_rank[j - lo]
                        j += 1
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


def type_sketches(graph: TypedGraph, q: Quotient, subjects: np.ndarray, is_object: np.ndarray, table_bits: int = 25):
    """(types, HLL registers) over pairs (a in subjects, d with ``is_object``), simple paths."""
    ptr, e, d = object_hops(graph, is_object)
    return _type_sketches(graph.indptr, graph.nbr, graph.e_out, graph.node_type, ptr, e, d,
                          subjects.astype(np.int64), q.n_edge, q.n_node, table_bits)


@nb.njit(cache=True)
def _sigma(x):
    if x == 1.0:
        return np.inf
    y, z = 1.0, x
    while True:
        x *= x
        prev = z
        z += x * y
        y += y
        if z == prev:
            return z


@nb.njit(cache=True)
def _tau(x):
    if x == 0.0 or x == 1.0:
        return 0.0
    y, z = 1.0, 1.0 - x
    while True:
        x = np.sqrt(x)
        prev = z
        y *= 0.5
        z -= (1.0 - x) ** 2 * y
        if z == prev:
            return z / 3.0


@nb.njit(cache=True)
def hll_estimate(reg):
    """Distinct count from HLL registers (Ertl's improved estimator; no bias tables)."""
    q = 64 - HLL_P
    counts = np.zeros(q + 2, dtype=np.int64)
    for v in reg:
        counts[v] += 1
    m = float(len(reg))
    z = m * _tau(1.0 - counts[q + 1] / m)
    for k in range(q, 0, -1):
        z = 0.5 * (z + counts[k])
    z += m * _sigma(counts[0] / m)
    return 0.7213475204444817 * m * m / z


@nb.njit(cache=True)
def _merge_groups_hll(order, group, tptr, codes, m):
    """As ``_merge_groups`` for HLL code lists: per group, the max rank per register."""
    total = 0
    for t in order:
        total += tptr[t + 1] - tptr[t]
    out = np.empty(total, dtype=codes.dtype)
    values = np.empty(len(order), dtype=np.int64)
    ptr = np.empty(len(order) + 1, dtype=np.int64)
    seen = np.full(m, -1, dtype=np.int64)
    where = np.zeros(m, dtype=np.int64)
    rows, k = 0, 0
    ptr[0] = 0
    for i in range(len(order)):
        t = order[i]
        if i == 0 or group[t] != values[rows - 1]:
            values[rows] = group[t]
            rows += 1
        for j in range(tptr[t], tptr[t + 1]):
            code = codes[j]
            idx = code & (m - 1)
            if seen[idx] != rows:
                seen[idx] = rows
                where[idx] = k
                out[k] = code
                k += 1
            elif code > out[where[idx]]:
                out[where[idx]] = code
        ptr[rows] = k
    return values[:rows], ptr[:rows + 1], out[:k]


@nb.njit(cache=True)
def _dense_groups_hll(use, group, ptr, codes, m):
    rows = 0
    prev = -1
    for i in range(len(use)):
        if use[i] and group[i] != prev:
            rows += 1
            prev = group[i]
    values = np.empty(rows, dtype=np.int64)
    reg = np.zeros((rows, m), dtype=np.uint8)
    r = -1
    prev = -1
    for i in range(len(use)):
        if not use[i]:
            continue
        if group[i] != prev:
            r += 1
            prev = group[i]
            values[r] = prev
        for j in range(ptr[i], ptr[i + 1]):
            idx = codes[j] & (m - 1)
            rank = codes[j] >> HLL_P
            if rank > reg[r, idx]:
                reg[r, idx] = rank
    return values, reg


@nb.njit(cache=True)
def _max_groups(use, group, reg, n_groups):
    out = np.zeros((n_groups, reg.shape[1]), dtype=np.uint8)
    for i in range(len(use)):
        if use[i]:
            g = group[i]
            for w in range(reg.shape[1]):
                if reg[i, w] > out[g, w]:
                    out[g, w] = reg[i, w]
    return out


@nb.njit(cache=True)
def _max_rows(reg, rows):
    merged = np.zeros(reg.shape[1], dtype=np.uint8)
    for r in rows:
        for w in range(reg.shape[1]):
            if reg[r, w] > merged[w]:
                merged[w] = reg[r, w]
    return merged


@nb.njit(cache=True)
def _union_estimate(reg, rows):
    return hll_estimate(_max_rows(reg, rows))


def sketch_table(types: np.ndarray, reg: np.ndarray):
    """(types sorted, tptr, codes) from dense registers: the non-zero registers per type
    as codes ``index | rank << HLL_P``."""
    order = np.argsort(types)
    reg = reg[order]
    rows, idx = np.nonzero(reg)
    codes = (idx | (reg[rows, idx].astype(np.int64) << HLL_P)).astype(np.uint16)
    tptr = np.zeros(len(types) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(types)), out=tptr[1:])
    return types[order], tptr, codes


@nb.njit(cache=True)
def _merge_sketch_tables(types_a, ptr_a, codes_a, types_b, ptr_b, codes_b, m):
    types = np.empty(len(types_a) + len(types_b), dtype=np.int64)
    ptr = np.empty(len(types) + 1, dtype=np.int64)
    codes = np.empty(len(codes_a) + len(codes_b), dtype=codes_a.dtype)
    reg = np.zeros(m, dtype=np.int64)
    i, j, n, k = 0, 0, 0, 0
    ptr[0] = 0
    while i < len(types_a) or j < len(types_b):
        take_a = j >= len(types_b) or (i < len(types_a) and types_a[i] <= types_b[j])
        take_b = i >= len(types_a) or (j < len(types_b) and types_b[j] <= types_a[i])
        if take_a and take_b:  # same type in both: register-wise max
            types[n] = types_a[i]
            for x in range(ptr_a[i], ptr_a[i + 1]):
                reg[codes_a[x] & (m - 1)] = codes_a[x]
            for x in range(ptr_b[j], ptr_b[j + 1]):
                idx = codes_b[x] & (m - 1)
                if codes_b[x] > reg[idx]:
                    reg[idx] = codes_b[x]
            for idx in range(m):
                if reg[idx]:
                    codes[k] = reg[idx]
                    k += 1
                    reg[idx] = 0
            i += 1
            j += 1
        elif take_a:
            types[n] = types_a[i]
            for x in range(ptr_a[i], ptr_a[i + 1]):
                codes[k] = codes_a[x]
                k += 1
            i += 1
        else:
            types[n] = types_b[j]
            for x in range(ptr_b[j], ptr_b[j + 1]):
                codes[k] = codes_b[x]
                k += 1
            j += 1
        n += 1
        ptr[n] = k
    return types[:n], ptr[:n + 1], codes[:k]


def merge_sketch_tables(a, b):
    """Union of two (types, tptr, codes) sketch tables (e.g. from disjoint subject sets)."""
    return _merge_sketch_tables(*a, *b, HLL_M)
