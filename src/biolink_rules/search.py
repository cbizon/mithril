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

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import graphblas as gb
import numpy as np

from biolink_rules.lattice import Lattice


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
    """Small LRU cache for matrices keyed by lattice coordinates."""

    def __init__(self, build, maxsize: int):
        self.build, self.maxsize, self.data = build, maxsize, OrderedDict()

    def __call__(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        value = self.build(key)
        self.data[key] = value
        if len(self.data) > self.maxsize:
            self.data.popitem(last=False)
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


def search(
    store: Store,
    hops: Lattice,
    classes: Lattice,
    target: gb.Matrix,
    n_hops: int,
    min_support: int,
    time_limit: float,
    cache_size: int = 256,
) -> SearchStats:
    """Depth-first specialization search for ``n_hops`` in (1, 2)."""
    assert n_hops in (1, 2), "3-hop search not implemented yet"
    hop_m = Cache(lambda b: store.hop_matrix(hops.sets[b]), cache_size)
    class_v = Cache(lambda c: store.class_vector(classes.sets[c]), cache_size)
    # second hop with its rows restricted to the intermediate class
    hop2_m = Cache(lambda key: restrict(hop_m(key[1]), rows=class_v(key[0])), cache_size)

    def overlap(body, mask):
        if n_hops == 1:
            result = gb.Matrix(bool, mask.nrows, mask.ncols)
            result(mask.S) << hop_m(body[0])
        else:
            h1, c1, h2 = body
            result = gb.Matrix(bool, mask.nrows, mask.ncols)
            result(mask.S) << hop_m(h1).mxm(hop2_m((c1, h2)), gb.semiring.any_pair)
        return result

    def children(body):
        for i, coord in enumerate(body):
            lattice = classes if (n_hops == 2 and i == 1) else hops
            for child in lattice.children[coord]:
                yield body[:i] + (child,) + body[i + 1:]

    root = (hops.root,) if n_hops == 1 else (hops.root, classes.root, hops.root)
    stats = SearchStats()
    start = time.time()
    visited = {root}
    root_overlap = overlap(root, target)
    stats.evaluated = 1
    stats.rows.append((root, None, root_overlap.nvals, 0))
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
            stats.rows.append((child, body, support, depth + 1))
            if support >= min_support:
                stats.passed += 1
                stack.append((child, child_overlap, depth + 1))
    stats.seconds = time.time() - start
    stats.complete = not stack
    return stats
