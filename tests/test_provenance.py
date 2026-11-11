"""Path-type provenance must reproduce brute-force 3-hop counts for every body."""

import numpy as np
import pytest

from biolink_rules.provenance import (
    HLL_M,
    PathTypeCounter,
    hll_estimate,
    hll_register,
    pair_types,
    quotient,
    sketch_table,
    type_sketches,
    type_table,
    typed_graph,
)
from biolink_rules.search import search
from test_threehop import setup, simple_predictions  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def provenance(setup):  # noqa: F811
    store, hops, classes, chem, dopf, target, a_nodes, b_nodes, target_pairs = setup
    q = quotient(hops, classes, 2 * (len(store.offsets) - 1), int(store.node_catset.max()) + 1)
    graph = typed_graph(store, q)
    rows, cols, _ = target.to_coo()
    subjects, start = np.unique(rows, return_index=True)
    pair, types = pair_types(graph, q, subjects, np.append(start, len(rows)), cols)
    overlap = PathTypeCounter(q, *type_table(pair, types, len(rows)), len(rows))
    is_object = np.zeros(store.n_nodes, dtype=bool)
    is_object[sorted(b_nodes)] = True
    sketch_types, registers = type_sketches(graph, q, np.array(sorted(a_nodes)), is_object, table_bits=12)
    support = PathTypeCounter(q, *sketch_table(sketch_types, registers), 0, kind="hll")
    return q, overlap, support, list(zip(rows.tolist(), cols.tolist()))


def masks(q, classes, body):
    h1, c1, h2, c2, h3 = body
    return q.edge_mask(h1), q.node_mask(classes.sets[c1]), q.edge_mask(h2), q.node_mask(classes.sets[c2]), q.edge_mask(h3)


def test_counts_match_brute_force_for_every_searched_body(setup, provenance):  # noqa: F811
    store, hops, classes, chem, dopf, target, a_nodes, b_nodes, target_pairs = setup
    q, overlap, support, pairs = provenance
    stats = search(store, hops, classes, target, chem, dopf, 3, min_support=1, time_limit=600, merge_equivalent=False)
    assert stats.complete and stats.evaluated > 100
    for body, _, _, searched, _ in stats.rows[:400]:
        predicted = simple_predictions(store, classes, body, a_nodes, b_nodes)
        m = masks(q, classes, body)
        assert overlap.count(*m) == searched == len(predicted & target_pairs), body
        bits = overlap.value(*m)
        joined = {pairs[i] for i in range(len(pairs)) if bits[i >> 6] >> np.uint64(i & 63) & np.uint64(1)}
        assert joined == predicted & target_pairs
        # the body's sketch is exactly the sketch of its predicted pairs
        expected = np.zeros(HLL_M, dtype=np.uint8)
        for a, d in predicted:
            idx, rank = hll_register(a, d, store.n_nodes)
            expected[idx] = max(expected[idx], rank)
        assert (support.value(*m) == expected).all(), body
        assert support.count(*m) == pytest.approx(len(predicted), rel=0.1, abs=1)


def test_hll_estimate_is_accurate():
    rng = np.random.default_rng(0)
    for n in (0, 1, 50, 2_000, 100_000):
        registers = np.zeros(HLL_M, dtype=np.uint8)
        for a, d in zip(rng.integers(0, 10**6, n), rng.integers(0, 10**6, n)):
            idx, rank = hll_register(a, d, 10**6)
            registers[idx] = max(registers[idx], rank)
        assert hll_estimate(registers) == pytest.approx(n, rel=0.1, abs=0.5)


def test_quotient_merges_only_indistinguishable_items(setup):  # noqa: F811
    store, hops, classes, *_ = setup
    n_bits = 2 * (len(store.offsets) - 1)
    q = quotient(hops, classes, n_bits, int(store.node_catset.max()) + 1)
    for s in hops.sets:  # every label is a union of whole edge types
        members = {int(q.edge_of_bit[b]) for b in range(n_bits) if s >> b & 1}
        others = {int(q.edge_of_bit[b]) for b in range(n_bits) if not s >> b & 1}
        assert not members & others


@pytest.mark.parametrize("near_duplicate", [1.0, 0.6])
@pytest.mark.parametrize("min_support", [1, 3])
def test_table_search_visits_the_coordinate_ordered_tree(setup, provenance, near_duplicate, min_support):  # noqa: F811
    from biolink_rules.pathsearch import bodies, mine_three_hop
    from biolink_rules.problem import Problem
    from biolink_rules.search import HopContexts, body_space, explore_ordered

    store, hops, classes, chem, dopf, target, a_nodes, b_nodes, target_pairs = setup
    q, overlap, support, _ = provenance
    contexts = HopContexts(store, hops, classes, near_duplicate)
    root, children, graph_overlap, _ = body_space(store, hops, classes, chem, dopf, 3, 0.1, contexts)
    root_overlap = graph_overlap(root, target)
    rows, evaluated, frontier = explore_ordered(children, graph_overlap, root, 0, root_overlap, 0, min_support)
    assert not frontier
    expected = {body: n for body, _, n in rows} | {root: root_overlap.nvals}

    problem = Problem(store, hops, classes, contexts, chem, dopf, target, "", 0)
    mined = mine_three_hop(problem, q, overlap, support, min_support)
    found = dict(zip(bodies(problem, mined, range(len(mined.overlap))), mined.overlap.tolist()))
    assert len(found) == len(mined.overlap)  # each body once
    assert found == expected
    assert mined.evaluated == evaluated + 1
    for body, estimate in zip(found, mined.support.tolist()):
        assert estimate == pytest.approx(support.count(*masks(q, classes, body)))


def test_sketch_tables_of_subject_stripes_merge_to_the_whole(setup):  # noqa: F811
    from biolink_rules.provenance import merge_sketch_tables

    store, hops, classes, chem, dopf, target, a_nodes, b_nodes, _ = setup
    q = quotient(hops, classes, 2 * (len(store.offsets) - 1), int(store.node_catset.max()) + 1)
    graph = typed_graph(store, q)
    is_object = np.zeros(store.n_nodes, dtype=bool)
    is_object[sorted(b_nodes)] = True
    subjects = np.array(sorted(a_nodes))
    whole = sketch_table(*type_sketches(graph, q, subjects, is_object, table_bits=12))
    parts = [sketch_table(*type_sketches(graph, q, subjects[i::3], is_object, table_bits=12)) for i in range(3)]
    merged = merge_sketch_tables(merge_sketch_tables(parts[0], parts[1]), parts[2])
    for got, want in zip(merged, whole):
        assert (got == want).all()
