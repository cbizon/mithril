"""Rule metrics and rule-set selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scored:
    body: tuple
    name: str
    overlap: int
    support: int
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


def greedy_rule_set(candidates, pairs_of, n_target: int, k: int, min_new_fraction: float = 0.0,
                    min_novel: float = 0.0) -> list[tuple]:
    """Scan ``candidates`` in the given order; keep one if it adds enough new target pairs.

    ``pairs_of(candidate)`` is the bitset (uint64 words) of target pairs it joins.
    ``min_new_fraction`` is relative to the whole target (discovery sets); ``min_novel``
    is relative to the rule's own overlap, i.e. a redundancy limit that says nothing
    about recall (conditioned sets).
    Returns (candidate, new pairs added, cumulative union recall) for up to ``k`` rules.
    """
    min_new = int(np.ceil(min_new_fraction * n_target))
    covered = np.zeros((n_target + 63) // 64, dtype=np.uint64)
    chosen = []
    for candidate in candidates:
        if len(chosen) == k:
            break
        pairs = pairs_of(candidate)
        new = int(np.bitwise_count(pairs & ~covered).sum())
        if new < min_new or new < min_novel * int(np.bitwise_count(pairs).sum()):
            continue
        covered |= pairs
        chosen.append((candidate, new, int(np.bitwise_count(covered).sum()) / n_target))
    return chosen
