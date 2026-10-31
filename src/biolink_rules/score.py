"""Exact rule metrics for search survivors, and rule-set selection.

Predictor support (number of distinct predicted (a, d) pairs) needs the full
product, so it is counted in row blocks and abandoned once it exceeds ``cap``.
"""

from __future__ import annotations

from dataclasses import dataclass

import graphblas as gb
import numpy as np

from biolink_rules.problem import Problem
from biolink_rules.search import Cache, restrict


@dataclass
class Scored:
    body: tuple
    name: str
    overlap: int
    support: int  # exact, or the count reached when capped
    capped: bool
    target_count: int

    @property
    def precision(self) -> float:
        return self.overlap / self.support if self.support else 0.0

    @property
    def recall(self) -> float:
        return self.overlap / self.target_count

    @property
    def f1(self) -> float:
        return 2 * self.overlap / (self.support + self.target_count)


class Scorer:
    """Exact pair counts for rule bodies ``(direct, path)``.

    ``direct`` is an atom-orientation bitset for an edge required directly between the
    endpoints (0 = none); ``path`` is a 2- or 3-hop search body (``()`` = none). A body
    predicts the endpoint pairs (a in subject class, d in object class) satisfying both.
    """

    def __init__(self, problem: Problem, cache_gb: float = 8.0):
        p = self.problem = problem
        store, classes = p.store, p.classes
        self.subject_v = store.class_vector(classes.sets[p.subject_class])
        self.object_v = store.class_vector(classes.sets[p.object_class])
        budget = int(cache_gb * 2**30) // 3
        self.first = Cache(lambda h: restrict(store.hop_matrix(h), rows=self.subject_v), budget)
        self.second = Cache(
            lambda key: restrict(store.hop_matrix(key[1]), rows=store.class_vector(classes.sets[key[0]]),
                                 cols=self.object_v),
            budget,
        )
        self.direct = Cache(lambda d: restrict(store.hop_matrix(d), rows=self.subject_v, cols=self.object_v)
                            .select("offdiag").new(), budget)
        self._three = None

    def three(self):
        if self._three is None:
            from biolink_rules.threehop import ThreeHop

            p = self.problem
            self._three = ThreeHop(p.store, p.classes.sets, p.subject_class, p.object_class)
        return self._three

    def path_pairs(self, path: tuple, mask: gb.Matrix) -> gb.Matrix:
        """Pairs of ``mask`` joined by the path (simple paths for 3 hops)."""
        if len(path) == 5:
            return self.three().overlap(path, mask)
        h1, c1, h2 = path
        result = gb.Matrix(bool, mask.nrows, mask.ncols)
        result(mask.S) << self.first(h1).mxm(self.second((c1, h2)), gb.semiring.any_pair)
        return result

    def path_support(self, path: tuple, cap: int) -> tuple[int, bool]:
        if len(path) == 5:
            return self.three().support(path, cap)
        h1, c1, h2 = path
        left, right = self.first(h1), self.second((c1, h2))
        rows = left.reduce_rowwise(gb.monoid.any).new().to_coo()[0]
        count, start, block = 0, 0, 512
        while start < len(rows):
            idx = rows[start:start + block]
            # keep global row indices so the a == d diagonal is the true diagonal
            product = gb.Matrix(bool, left.nrows, right.ncols)
            product[idx, :] << left[idx, :].new().mxm(right, gb.semiring.any_pair)
            product = product.select("offdiag").new()
            count += product.nvals
            if count > cap:
                return count, True
            start += block
            if product.nvals < 1_000_000:
                block = min(block * 2, 65536)
            elif product.nvals > 8_000_000:
                block = max(block // 2, 16)
        return count, False

    def support(self, body: tuple, cap: int) -> tuple[int, bool]:
        """(distinct predicted pairs, capped?) -- path-only counting stops above ``cap``."""
        direct, path = body
        if not path:
            return self.direct(direct).nvals, False
        if not direct:
            return self.path_support(path, cap)
        return self.path_pairs(path, self.direct(direct)).nvals, False

    def overlap_pairs(self, body: tuple) -> gb.Matrix:
        direct, path = body
        mask = self.problem.target
        if direct:
            mask = mask.ewise_mult(self.direct(direct), gb.binary.pair).new()
        return self.path_pairs(path, mask) if path else mask


def is_conditioned(rule: Scored) -> bool:
    """True for rules requiring a direct edge between the endpoints (1-hop or direct AND path).

    These only rank pairs that already carry a related edge, so they are selected and
    judged separately from path-only (discovery) rules.
    """
    return rule.body[0] != 0


def greedy_rule_set(
    scorer: Scorer, candidates: list[Scored], k: int, min_new_fraction: float = 0.0, min_novel: float = 0.0
) -> list[tuple[Scored, int, float]]:
    """Scan candidates by precision; keep a rule if it adds enough new target pairs.

    ``min_new_fraction`` is relative to the whole target (discovery sets);
    ``min_novel`` is relative to the rule's own overlap, i.e. a redundancy limit that
    says nothing about recall (conditioned sets).
    Returns (rule, new pairs added, cumulative union recall) for up to ``k`` rules.
    """
    target_count = scorer.problem.target.nvals
    min_new = int(np.ceil(min_new_fraction * target_count))
    covered = gb.Matrix(bool, scorer.problem.target.nrows, scorer.problem.target.ncols)
    chosen = []
    for rule in sorted(candidates, key=lambda r: (-r.precision, -r.overlap)):
        pairs = scorer.overlap_pairs(rule.body)
        new = pairs.nvals - pairs.ewise_mult(covered, gb.binary.pair).new().nvals
        if new < min_new or new < min_novel * pairs.nvals:
            continue
        covered = covered.ewise_add(pairs, gb.binary.pair).new()
        chosen.append((rule, new, covered.nvals / target_count))
        if len(chosen) == k:
            break
    return chosen
