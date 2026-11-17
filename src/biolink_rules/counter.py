"""Counting rule bodies from per-path-type summaries (see ``provenance``).

A body is a product of type sets, one per position; the pairs it joins are the union of
the pair sets of the path types inside the product. Two summaries of a pair set are
closed under union, so both roll up the same way, one position at a time:

- a **pair bitset** over the target pairs (union = OR; exact count = popcount);
- a **HyperLogLog sketch** of pairs over the whole graph (union = register-wise max;
  the count is an estimate, about 3% error with 1,024 registers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

if TYPE_CHECKING:
    from biolink_rules.provenance import Quotient

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


def pair_bits(member: np.ndarray) -> np.ndarray:
    """Pair bitset (uint64 words, pair p = bit p & 63 of word p >> 6) of a boolean array.

    >>> pair_bits(np.arange(70) % 65 == 1).tolist()
    [2, 4]
    """
    return np.packbits(np.pad(member, (0, -len(member) % 64)), bitorder="little").view(np.uint64)


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
    """Roll-up of per-path-type summaries to any body.

    ``types`` are distinct packed path types; ``tptr``/``items`` hold a summary per type.
    Two kinds, both closed under union so they roll up exactly the same way:

    - ``kind="pairs"``: items are pair indices (< ``n``); a body's value is the exact
      number of those pairs it joins (``pairs`` gives the pair bitset).
    - ``kind="hll"``: items are HLL codes ``index | rank << HLL_P`` of the (a, d) pairs
      of the type; a body's value estimates how many distinct pairs it predicts.

    A body is five type masks ``(E1, K1, E2, K2, E3)`` (Python ints; the "none" type
    of a position is bit ``n_edge`` / ``n_node``, see ``Quotient``). Rolled-up tables
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
        self.e1, self.k1, self.e2, self.k2, self.e3 = (x.astype(np.int64) for x in q.unpack(types))
        self.tptr, self.items = tptr, items
        # e1 varies fastest, so rows keyed (e1, e2, e3) come out with (e2, e3) runs contiguous
        self._group = (self.e2 * q.edge_base + self.e3) * q.edge_base + self.e1
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
            sel = (self._use(self._node_use, k1, q.node_base)[self.k1]
                   & self._use(self._node_use, k2, q.node_base)[self.k2])
            order = self._by_group[sel[self._by_group]]
            merge = _merge_groups if self.kind == "pairs" else _merge_groups_hll
            values, ptr, items = merge(order, self._group, self.tptr, self.items, self.n)
            return values % q.edge_base, values // q.edge_base, ptr, items

        return self._classes.get((k1, k2), build)

    def _by_first(self, k1: int, k2: int, e1: int):
        def build():
            row_e1, row_e23, ptr, items = self._by_classes(k1, k2)
            use = self._use(self._edge_use, e1, self.q.edge_base)[row_e1]
            dense = _dense_groups if self.kind == "pairs" else _dense_groups_hll
            values, rows = dense(use, row_e23, ptr, items, self.width)
            return values // self.q.edge_base, values % self.q.edge_base, rows

        return self._first.get((k1, k2, e1), build)

    def _by_second(self, k1: int, k2: int, e1: int, e2: int):
        def build():
            row_e2, row_e3, rows = self._by_first(k1, k2, e1)
            use = self._use(self._edge_use, e2, self.q.edge_base)[row_e2]
            combine = _or_groups if self.kind == "pairs" else _max_groups
            return combine(use, row_e3, rows, self.q.edge_base)

        return self._second.get((k1, k2, e1, e2), build)

    def _rows(self, e3: int) -> np.ndarray:
        if e3 not in self._edge_rows:
            self._edge_rows[e3] = np.flatnonzero(mask_array(e3, self.q.edge_base))
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


def path_only(q: Quotient, types: np.ndarray, tptr: np.ndarray, codes: np.ndarray):
    """The sketch table of path types regardless of direct edges: the rows whose direct
    part is "none" (the last block of the sorted types), as (path types, tptr, codes)."""
    lo = int(np.searchsorted(types, q.none_edge * q.path_base))
    return types[lo:] - q.none_edge * q.path_base, tptr[lo:] - tptr[lo], codes[tptr[lo]:]


def with_direct(q: Quotient, types: np.ndarray, tptr: np.ndarray, codes: np.ndarray, direct: int):
    """The sketch table of path types among pairs that have a direct edge whose type is
    in the mask ``direct``: (path types, tptr, codes)."""
    use = mask_array(direct, q.edge_base)[types // q.path_base]
    path = types % q.path_base
    order = np.flatnonzero(use)
    order = order[np.argsort(path[order], kind="stable")]
    return _merge_groups_hll(order, path, tptr, codes, HLL_M)


@nb.njit(cache=True)
def _restrict_pairs(tptr, tpairs, new_id):
    ptr = np.empty(len(tptr), dtype=np.int64)
    out = np.empty(len(tpairs), dtype=tpairs.dtype)
    k = 0
    ptr[0] = 0
    for t in range(len(tptr) - 1):
        for j in range(tptr[t], tptr[t + 1]):
            p = new_id[tpairs[j]]
            if p >= 0:
                out[k] = p
                k += 1
        ptr[t + 1] = k
    return ptr, out[:k]


def restrict_pairs(types: np.ndarray, tptr: np.ndarray, tpairs: np.ndarray, keep: np.ndarray):
    """A pair table restricted to the pairs with ``keep`` set, renumbered 0..: (types, tptr, tpairs)."""
    new_id = np.cumsum(keep) - 1
    new_id[~keep] = -1
    ptr, pairs = _restrict_pairs(tptr, tpairs, new_id)
    full = np.diff(ptr) > 0
    return types[full], np.append(ptr[:-1][full], ptr[-1]), pairs
