"""Rule-set selection keeps rules by the new target pairs they add."""

import numpy as np

from mithril.pathsearch import load_mined, mine, save_mined
from mithril.score import Scored, greedy_rule_set


def bits(*pairs):
    out = np.zeros(2, dtype=np.uint64)
    for p in pairs:
        out[p >> 6] |= np.uint64(1) << np.uint64(p & 63)
    return out


def test_greedy_novelty_rejects_redundant_rules():
    pairs = {"r": bits(0, 1, 70), "twin": bits(0, 1, 70), "half": bits(1, 2, 3, 4), "small": bits(99)}
    # an exact duplicate adds 0 new pairs: kept with no thresholds, dropped by novelty
    assert [c[0] for c in greedy_rule_set(["r", "twin"], pairs.get, 100, k=5)] == ["r", "twin"]
    chosen = greedy_rule_set(["r", "twin", "half", "small"], pairs.get, 100, k=5, min_novel=0.5)
    assert chosen == [("r", 3, 0.03), ("half", 3, 0.06), ("small", 1, 0.07)]
    # discovery: each rule must add 2% of the target; stop at k
    chosen = greedy_rule_set(["r", "twin", "small", "half"], pairs.get, 100, k=5, min_new_fraction=0.02)
    assert [c[0] for c in chosen] == ["r", "half"]
    assert [c[0] for c in greedy_rule_set(["r", "half", "small"], pairs.get, 100, k=1)] == ["r"]


def test_scored_metrics():
    rule = Scored((0, ()), "r", overlap=10, support=40, target_count=100)
    assert (rule.precision, rule.recall, rule.f1) == (0.25, 0.1, 20 / 140)


def test_mined_rules_round_trip(mining, tmp_path):
    m = mining
    mined = mine(m.problem, m.q, m.table, m.direct, m.sketch, 2)
    save_mined(tmp_path / "mined.npz", mined)
    loaded = load_mined(tmp_path / "mined.npz")
    assert len(loaded) == len(mined)
    for got, want in zip(loaded, mined):
        assert (got.direct, got.hops, got.evaluated) == (want.direct, want.hops, want.evaluated)
        for column in ("c1", "c2", "n1", "n2", "n3", "overlap", "support"):
            assert (getattr(got, column) == getattr(want, column)).all()
