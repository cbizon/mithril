"""Exact 3-hop counts over simple (node-disjoint) paths.

For a body A -h1- c1 -h2- c2 -h3- B with boolean hop matrices restricted to the middle
classes (A1 = H1 cols in c1, A2 = H2 rows in c1 / cols in c2, A3 = H3 rows in c2), the
number of walks a-b-c-d is ``(A1·A2·A3)[a,d]`` in the plus-times semiring. With no
self-loops, a walk repeats a node only if c == a or b == d (or both), so

    simple = A1·A2·A3 − diag(A1·A2)·A3 − A1·diag(A2·A3) + A1 ∘ A2ᵀ ∘ A3

counts simple paths exactly, and (a, d) with a != d has a simple path iff simple > 0.
diag(A1·A2)[a] = rowsum(A1 ∘ A2ᵀ) and diag(A2·A3)[d] = rowsum(A2 ∘ A3ᵀ) are cheap.
The full product is formed in blocks of rows so memory stays bounded.
"""

from __future__ import annotations

import graphblas as gb
import numpy as np

from biolink_rules.search import Cache, Store, restrict

PLUS_TIMES = gb.semiring.plus_times[gb.dtypes.INT64]


def as_counts(matrix: gb.Matrix) -> gb.Matrix:
    return matrix.apply(gb.unary.one[gb.dtypes.INT64]).new()


class ThreeHop:
    """Overlap and support for 3-hop bodies ``(h1, c1, h2, c2, h3)``."""

    def __init__(self, store: Store, class_sets: list[int], subject_class: int, object_class: int,
                 cache_gb: float = 8.0):
        self.store = store
        self.class_v = Cache(lambda c: store.class_vector(class_sets[c]), 2**28)
        self.subject_v = self.class_v(subject_class)
        self.object_v = self.class_v(object_class)
        budget = int(cache_gb * 2**30) // 4
        self.a1 = Cache(lambda k: as_counts(restrict(store.hop_matrix(k[0]), cols=self.class_v(k[1]))), budget)
        self.a2 = Cache(lambda k: as_counts(restrict(store.hop_matrix(k[1]), rows=self.class_v(k[0]),
                                                     cols=self.class_v(k[2]))), budget)
        self.a3 = Cache(lambda k: as_counts(restrict(store.hop_matrix(k[1]), rows=self.class_v(k[0]))), budget)
        self.a3_object = Cache(lambda k: restrict(self.a3(k), cols=self.object_v), budget)

    def matrices(self, body):
        h1, c1, h2, c2, h3 = body
        return self.a1((h1, c1)), self.a2((c1, h2, c2)), self.a3((c2, h3))

    @staticmethod
    def diagonals(a1, a2, a3):
        diag_x = a1.ewise_mult(a2.T, gb.binary.times).new().reduce_rowwise(gb.monoid.plus).new()
        diag_y = a2.ewise_mult(a3.T, gb.binary.times).new().reduce_rowwise(gb.monoid.plus).new()
        return diag_x, diag_y

    @staticmethod
    def simple_counts(total, a1, a2, a3, diag_x, diag_y, mask):
        """total − t1 − t2 + t3 on ``mask``'s structure (entries of total only)."""
        t1 = gb.Matrix(gb.dtypes.INT64, total.nrows, total.ncols)
        t1(mask.S) << diag_x.diag().mxm(a3, PLUS_TIMES)
        t2 = gb.Matrix(gb.dtypes.INT64, total.nrows, total.ncols)
        t2(mask.S) << a1.mxm(diag_y.diag(), PLUS_TIMES)
        t3 = gb.Matrix(gb.dtypes.INT64, total.nrows, total.ncols)
        t3(mask.S) << a1.ewise_mult(a2.T, gb.binary.times).new().ewise_mult(a3, gb.binary.times)
        simple = total.ewise_add(t1, gb.binary.minus).new()
        simple = simple.ewise_add(t2, gb.binary.minus).new()
        simple = simple.ewise_add(t3, gb.binary.plus).new()
        return simple.select(">", 0).new()

    def _blocks(self, rows: np.ndarray, a1, a2, cap_rows: int = 65536):
        """Yield (row index block, X = A1[block] · A2) with adaptive block size."""
        start, block = 0, 256
        while start < len(rows):
            idx = rows[start:start + block]
            x = gb.Matrix(gb.dtypes.INT64, a1.nrows, a2.ncols)
            x[idx, :] << a1[idx, :].new().mxm(a2, PLUS_TIMES)
            yield idx, x
            start += block
            if x.nvals < 2_000_000:
                block = min(block * 2, cap_rows)
            elif x.nvals > 16_000_000:
                block = max(block // 2, 8)

    def overlap(self, body, mask: gb.Matrix) -> gb.Matrix:
        """Boolean matrix of ``mask`` pairs joined by at least one simple path."""
        a1, a2, a3 = self.matrices(body)
        diag_x, diag_y = self.diagonals(a1, a2, a3)
        rows = mask.reduce_rowwise(gb.monoid.any).new().to_coo()[0]
        result = gb.Matrix(bool, mask.nrows, mask.ncols)
        for idx, x in self._blocks(rows, a1, a2):
            block_mask = gb.Matrix(bool, mask.nrows, mask.ncols)
            block_mask[idx, :] << mask[idx, :]
            total = gb.Matrix(gb.dtypes.INT64, mask.nrows, mask.ncols)
            total(block_mask.S) << x.mxm(a3, PLUS_TIMES)
            simple = self.simple_counts(total, a1, a2, a3, diag_x, diag_y, block_mask)
            result(simple.S) << True
        return result

    def support(self, body, cap: int) -> tuple[int, bool]:
        """(distinct predicted (a, d), a != d, with a simple path; capped?)."""
        h1, c1, h2, c2, h3 = body
        a1 = restrict(self.a1((h1, c1)), rows=self.subject_v)
        a2 = self.a2((c1, h2, c2))
        a3 = self.a3_object((c2, h3))
        diag_x, diag_y = self.diagonals(a1, a2, a3)
        rows = a1.reduce_rowwise(gb.monoid.any).new().to_coo()[0]
        count = 0
        for idx, x in self._blocks(rows, a1, a2):
            total = x.mxm(a3, PLUS_TIMES).new()
            total = total.select("offdiag").new()
            simple = self.simple_counts(total, a1, a2, a3, diag_x, diag_y, total)
            count += simple.nvals
            if count > cap:
                return count, True
        return count, False
