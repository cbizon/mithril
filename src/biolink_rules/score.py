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
    """Counts predictor support and overlap pair sets for 1- and 2-hop bodies."""

    def __init__(self, problem: Problem, cache_gb: float = 8.0):
        p = self.problem = problem
        store, classes = p.store, p.classes
        self.subject_v = store.class_vector(classes.sets[p.subject_class])
        self.object_v = store.class_vector(classes.sets[p.object_class])
        budget = int(cache_gb * 2**30) // 2
        self.first = Cache(lambda h: restrict(store.hop_matrix(h), rows=self.subject_v), budget)
        self.second = Cache(
            lambda key: restrict(store.hop_matrix(key[1]), rows=store.class_vector(classes.sets[key[0]]),
                                 cols=self.object_v),
            budget,
        )

    def support(self, body: tuple, cap: int) -> tuple[int, bool]:
        """(distinct predicted pairs, capped?) -- stops counting once above ``cap``."""
        if len(body) == 1:
            m = restrict(self.first(body[0]), cols=self.object_v)
            return m.nvals, False
        h1, c1, h2 = body
        left, right = self.first(h1), self.second((c1, h2))
        rows = left.reduce_rowwise(gb.monoid.any).new().to_coo()[0]
        count, start, block = 0, 0, 512
        while start < len(rows):
            idx = rows[start:start + block]
            product = left[idx, :].new().mxm(right, gb.semiring.any_pair).new()
            count += product.nvals
            if count > cap:
                return count, True
            start += block
            if product.nvals < 1_000_000:
                block = min(block * 2, 65536)
            elif product.nvals > 8_000_000:
                block = max(block // 2, 16)
        return count, False

    def overlap_pairs(self, body: tuple) -> gb.Matrix:
        target = self.problem.target
        result = gb.Matrix(bool, target.nrows, target.ncols)
        if len(body) == 1:
            result(target.S) << self.first(body[0])
        else:
            h1, c1, h2 = body
            result(target.S) << self.first(h1).mxm(self.second((c1, h2)), gb.semiring.any_pair)
        return result


def greedy_rule_set(
    scorer: Scorer, candidates: list[Scored], k: int, min_new_fraction: float
) -> list[tuple[Scored, int, float]]:
    """Scan candidates by precision; keep a rule if it adds enough new target pairs.

    Returns (rule, new pairs added, cumulative union recall) for up to ``k`` rules.
    """
    target_count = scorer.problem.target.nvals
    min_new = int(np.ceil(min_new_fraction * target_count))
    covered = gb.Matrix(bool, scorer.problem.target.nrows, scorer.problem.target.ncols)
    chosen = []
    for rule in sorted(candidates, key=lambda r: (-r.precision, -r.overlap)):
        pairs = scorer.overlap_pairs(rule.body)
        new = pairs.nvals - pairs.ewise_mult(covered, gb.binary.pair).new().nvals
        if new < min_new:
            continue
        covered = covered.ewise_add(pairs, gb.binary.pair).new()
        chosen.append((rule, new, covered.nvals / target_count))
        if len(chosen) == k:
            break
    return chosen
