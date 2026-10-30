"""Target-centric top-down rule search (1 and 2 hops; 3 hops TODO).

A rule body for a target (A -T-> B) is a chain A -h1- c1 -h2- B where each ``h`` is a
hop-lattice node and each intermediate ``c`` a class-lattice node. The search starts at
the most general body and specializes one coordinate at a time. Overlap with the target
pair set is anti-monotone under specialization, so bodies below ``min_support`` are
pruned together with all their specializations. Each child's overlap is computed with
its parent's overlap pairs as the mask.

Every evaluated body is logged (including pruned ones), so the frontier for any higher
support threshold can be recomputed from one run.
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import graphblas as gb
import numpy as np

from biolink_rules.lattice import Lattice, hasse


@dataclass
class Store:
    """Built atom edge store plus node class membership."""

    offsets: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    target_ok: np.ndarray
    node_catset: np.ndarray

    @classmethod
    def load(cls, build_dir: Path, ingest_dir: Path) -> "Store":
        return cls(
            offsets=np.load(build_dir / "atom_offsets.npy"),
            src=np.load(build_dir / "edge_src.npy"),
            dst=np.load(build_dir / "edge_dst.npy"),
            target_ok=np.load(build_dir / "edge_target_ok.npy"),
            node_catset=np.load(ingest_dir / "node_catset.npy"),
        )

    @property
    def n_nodes(self) -> int:
        return len(self.node_catset)

    def hop_matrix(self, bits: int, target_only: bool = False) -> gb.Matrix:
        """Boolean N x N matrix of the (atom, orientation) set ``bits``."""
        rows, cols = [], []
        for bit in range(bits.bit_length()):
            if not bits >> bit & 1:
                continue
            atom, reverse = divmod(bit, 2)
            lo, hi = self.offsets[atom], self.offsets[atom + 1]
            s, d = self.src[lo:hi], self.dst[lo:hi]
            if target_only:
                ok = self.target_ok[lo:hi]
                s, d = s[ok], d[ok]
            rows.append(d if reverse else s)
            cols.append(s if reverse else d)
        rows = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
        return gb.Matrix.from_coo(rows, cols, True, nrows=self.n_nodes, ncols=self.n_nodes)

    def class_vector(self, catset_bits: int) -> gb.Vector:
        member = np.array([catset_bits >> int(c) & 1 for c in range(int(self.node_catset.max()) + 1)], dtype=bool)
        idx = np.flatnonzero(member[self.node_catset])
        return gb.Vector.from_coo(idx, True, size=self.n_nodes)


class Cache:
    """LRU cache bounded by an estimated memory budget (``size`` -> bytes of a value)."""

    def __init__(self, build, budget_bytes: int, size=lambda m: 16 * m.nvals + 1024):
        self.build, self.budget, self.size = build, budget_bytes, size
        self.data: OrderedDict = OrderedDict()
        self.used = 0

    def __call__(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key][0]
        value = self.build(key)
        nbytes = self.size(value)
        self.data[key] = (value, nbytes)
        self.used += nbytes
        while self.used > self.budget and len(self.data) > 1:
            _, (_, evicted) = self.data.popitem(last=False)
            self.used -= evicted
        return value


@dataclass
class SearchStats:
    evaluated: int = 0
    passed: int = 0
    seconds: float = 0.0
    complete: bool = True
    rows: list[tuple] = field(default_factory=list)


def restrict(matrix: gb.Matrix, rows: gb.Vector | None = None, cols: gb.Vector | None = None) -> gb.Matrix:
    """Keep only entries whose row (col) is in the given node vector."""
    if rows is not None:
        matrix = rows.diag().mxm(matrix, gb.semiring.any_pair).new()
    if cols is not None:
        matrix = matrix.mxm(cols.diag(), gb.semiring.any_pair).new()
    return matrix


class HopContexts:
    """Hop lattices collapsed within a (left class, right class) context.

    Two hop sets that differ only in atoms with no edge between the context classes
    give identical matrices once restricted to those classes, so they are one node.
    Hop coordinates in a search body are these *effective* bitsets.
    """

    def __init__(self, store: Store, hops: Lattice, classes: Lattice):
        self.hops, self.classes = hops, classes
        n_catsets = int(store.node_catset.max()) + 1
        atom = np.repeat(np.arange(len(store.offsets) - 1), np.diff(store.offsets))
        key = np.unique((atom.astype(np.int64) * n_catsets + store.node_catset[store.src]) * n_catsets
                        + store.node_catset[store.dst])
        self._atom, self._left, self._right = key // n_catsets**2, key // n_catsets % n_catsets, key % n_catsets
        self._member = [np.array([bits >> i & 1 for i in range(n_catsets)], dtype=bool) for bits in classes.sets]
        self._masks: dict[tuple[int, int], int] = {}
        self._lattices: dict[tuple[int, int], tuple[Lattice, dict[int, int]]] = {}

    def mask(self, left: int, right: int) -> int:
        """Atom-orientation bits with at least one edge from ``left`` to ``right``."""
        if (left, right) not in self._masks:
            lm, rm = self._member[left], self._member[right]
            forward = np.unique(self._atom[lm[self._left] & rm[self._right]])
            reverse = np.unique(self._atom[rm[self._left] & lm[self._right]])
            bits = 0
            for a in forward:
                bits |= 1 << (2 * int(a))
            for a in reverse:
                bits |= 1 << (2 * int(a) + 1)
            self._masks[left, right] = bits
        return self._masks[left, right]

    def lattice(self, left: int, right: int) -> tuple[Lattice, dict[int, int]]:
        """The context lattice and a bits -> node index map."""
        if (left, right) not in self._lattices:
            m = self.mask(left, right)
            labels: dict[int, list[str]] = {}
            for bits, names in zip(self.hops.sets, self.hops.labels):
                if bits & m:
                    labels.setdefault(bits & m, []).extend(names)
            sets = list(labels)
            children, root = hasse(sets) if sets else ([], 0)
            lattice = Lattice(sets=sets, labels=[sorted(labels[b]) for b in sets], children=children, root=root)
            self._lattices[left, right] = (lattice, {b: i for i, b in enumerate(sets)})
        return self._lattices[left, right]

    def children(self, bits: int, left: int, right: int) -> list[int]:
        lattice, index = self.lattice(left, right)
        return [lattice.sets[c] for c in lattice.children[index[bits]]]

    def label(self, bits: int, left: int, right: int) -> str:
        lattice, index = self.lattice(left, right)
        return lattice.label(index[bits])


def search(
    store: Store,
    hops: Lattice,
    classes: Lattice,
    target: gb.Matrix,
    subject_class: int,
    object_class: int,
    n_hops: int,
    min_support: int,
    time_limit: float,
    cache_gb: float = 4.0,
    contexts: HopContexts | None = None,
) -> SearchStats:
    """Depth-first specialization search for ``n_hops`` in (1, 2).

    A body is ``(h1,)`` or ``(h1, c1, h2)``: hops are effective bitsets in their class
    context (``HopContexts``), ``c1`` is a class-lattice index.
    """
    assert n_hops in (1, 2), "3-hop search not implemented yet"
    contexts = contexts or HopContexts(store, hops, classes)
    a, b = subject_class, object_class
    budget = int(cache_gb * 2**30)
    hop_m = Cache(store.hop_matrix, budget // 2)
    class_v = Cache(lambda c: store.class_vector(classes.sets[c]), 2**28)
    # second hop with its rows restricted to the intermediate class
    hop2_m = Cache(lambda key: restrict(hop_m(key[1]), rows=class_v(key[0])), budget // 2)

    def overlap(body, mask):
        result = gb.Matrix(bool, mask.nrows, mask.ncols)
        if n_hops == 1:
            result(mask.S) << hop_m(body[0])
        else:
            h1, c1, h2 = body
            result(mask.S) << hop_m(h1).mxm(hop2_m((c1, h2)), gb.semiring.any_pair)
        return result

    def children(body):
        if n_hops == 1:
            for h in contexts.children(body[0], a, b):
                yield (h,)
            return
        h1, c1, h2 = body
        for h in contexts.children(h1, a, c1):
            yield (h, c1, h2)
        for h in contexts.children(h2, c1, b):
            yield (h1, c1, h)
        for c in classes.children[c1]:
            n1, n2 = h1 & contexts.mask(a, c), h2 & contexts.mask(c, b)
            if n1 and n2:
                yield (n1, c, n2)

    def name(body):
        if n_hops == 1:
            return contexts.label(body[0], a, b)
        h1, c1, h2 = body
        return f"{contexts.label(h1, a, c1)} | {classes.label(c1)} | {contexts.label(h2, c1, b)}"

    if n_hops == 1:
        root = (hops.sets[hops.root] & contexts.mask(a, b),)
    else:
        c = classes.root
        root = (hops.sets[hops.root] & contexts.mask(a, c), c, hops.sets[hops.root] & contexts.mask(c, b))
    stats = SearchStats()
    start = time.time()
    visited = {root}
    root_overlap = overlap(root, target)
    stats.evaluated = 1
    stats.rows.append((root, name(root), "", root_overlap.nvals, 0))
    if root_overlap.nvals < min_support:
        return stats
    stats.passed = 1
    stack = [(root, root_overlap, 0)]
    while stack and time.time() - start < time_limit:
        body, mask, depth = stack.pop()
        for child in children(body):
            if child in visited:
                continue
            visited.add(child)
            child_overlap = overlap(child, mask)
            stats.evaluated += 1
            support = child_overlap.nvals
            stats.rows.append((child, name(child), name(body), support, depth + 1))
            if support >= min_support:
                stats.passed += 1
                stack.append((child, child_overlap, depth + 1))
            if stats.evaluated % 1000 == 0:
                elapsed = time.time() - start
                print(f"[{elapsed:8.0f}s] evaluated {stats.evaluated:,} passed {stats.passed:,} "
                      f"stack {len(stack):,} ({stats.evaluated / elapsed:.1f}/s)", file=sys.stderr, flush=True)
    stats.seconds = time.time() - start
    stats.complete = not stack
    return stats
