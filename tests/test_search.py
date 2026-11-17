"""The table search must find exactly the search tree, at every path length and under
every direct label, with brute-force overlaps and supports."""

import numpy as np
import pytest

from biolink_rules.counter import HLL_M, hll_estimate, hll_register
from biolink_rules.pathsearch import body_masks, body_name, body_of, direct_lattice, mine
from biolink_rules.provenance import count_pairs
from brute import ordered_tree, pairs_of, predicted
from conftest import make_problem


def expected_rules(problem, m, min_support, hops=(2, 3)):
    """{(direct node, hops): ({body: overlap}, evaluated)} by brute force."""
    store, classes = problem.store, problem.classes
    cache = {}

    def joined(path):
        if path not in cache:
            cache[path] = predicted(store, classes, (0, path), m.a_nodes, m.b_nodes) & m.target_pairs
        return cache[path]

    out = {(-1, n): ordered_tree(problem, n, min_support, lambda path: len(joined(path))) for n in hops}
    lattice = direct_lattice(problem)
    seen, stack = {lattice.root}, [lattice.root]
    while stack:
        node = stack.pop()
        having = pairs_of(store, lattice.sets[node])
        if len(having & m.target_pairs) < min_support:
            continue
        for n in (0, *hops):
            out[node, n] = ordered_tree(problem, n, min_support, lambda path: len(joined(path) & having))
        for child in lattice.children[node]:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return out


@pytest.mark.parametrize("near_duplicate", [1.0, 0.6])
@pytest.mark.parametrize("min_support", [1, 3])
def test_search_finds_the_tree_at_every_length_and_direct_label(mining, near_duplicate, min_support):
    m = mining
    problem = make_problem(m.world, near_duplicate)
    lattice = direct_lattice(problem)
    expected = expected_rules(problem, m, min_support)
    mined = mine(problem, m.q, m.table, m.direct, m.sketch, min_support)
    assert {(x.direct, x.hops) for x in mined} == set(expected)
    assert {x.direct for x in mined} > {-1} and sum(len(x.overlap) for x in mined) > 100
    for x in mined:
        passing, evaluated = expected[x.direct, x.hops]
        direct = lattice.sets[x.direct] if x.direct >= 0 else 0
        found = {body_of(problem, x, i): int(x.overlap[i]) for i in range(len(x.overlap))}
        assert len(found) == len(x.overlap)  # each body once
        assert found == {(direct, path): n for path, n in passing.items()}
        assert x.evaluated == evaluated


def test_supports_are_the_sketch_and_the_exact_count_of_the_predicted_pairs(mining):
    m = mining
    problem, store = m.problem, m.problem.store
    mined = mine(problem, m.q, m.table, m.direct, m.sketch, 2)
    checked = set()
    for x in mined:
        for i in range(0, len(x.overlap), 7):
            body = body_of(problem, x, i)
            pairs = predicted(store, problem.classes, body, m.a_nodes, m.b_nodes)
            assert not body[0] & problem.target_bits  # the target edge is never a predictor
            assert x.overlap[i] == len(pairs & m.target_pairs), body_name(problem, body)
            registers = np.zeros(HLL_M, dtype=np.uint8)
            for a, d in pairs:
                idx, rank = hll_register(a, d, store.n_nodes)
                registers[idx] = max(registers[idx], rank)
            assert x.support[i] == pytest.approx(hll_estimate(registers)), body_name(problem, body)
            assert x.support[i] == pytest.approx(len(pairs), rel=0.1, abs=1)
            assert count_pairs(m.graph, m.q, m.subjects, m.is_object, body_masks(problem, m.q, body)) == len(pairs)
            checked.add((x.direct >= 0, x.hops))
    assert checked == {(False, 2), (False, 3), (True, 0), (True, 2), (True, 3)}


def test_search_without_a_sketch_or_direct_labels(mining):
    m = mining
    mined = mine(m.problem, m.q, m.table, m.direct, None, 2, hops=(2,), conditioned=False)
    assert [(x.direct, x.hops) for x in mined] == [(-1, 2)]
    assert len(mined[0].overlap) and np.isnan(mined[0].support).all()
