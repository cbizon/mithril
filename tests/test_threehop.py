"""3-hop overlap/support must equal brute-force counts over simple paths."""

import itertools

import pytest

from biolink_rules.search import restrict, search
from biolink_rules.threehop import ThreeHop
from brute import members, pairs_of


def simple_predictions(store, classes, body, a_nodes, b_nodes):
    h1, c1, h2, c2, h3 = body
    m1, m2 = members(store, classes, c1), members(store, classes, c2)
    e1, e2, e3 = pairs_of(store, h1), pairs_of(store, h2), pairs_of(store, h3)
    out = set()
    for a, b in e1:
        if a not in a_nodes or b not in m1:
            continue
        for b2, c in e2:
            if b2 != b or c not in m2:
                continue
            for c2_, d in e3:
                if c2_ == c and d in b_nodes and len({a, b, c, d}) == 4:
                    out.add((a, d))
    return out


@pytest.fixture(scope="module")
def setup(world):
    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    target = restrict(store.hop_matrix(hops.sets[treats], target_only=True),
                      rows=store.class_vector(classes.sets[chem]), cols=store.class_vector(classes.sets[dopf]))
    a_nodes, b_nodes = members(store, classes, chem), members(store, classes, dopf)
    target_pairs = {(a, b) for a, b in pairs_of(store, hops.sets[treats]) if a in a_nodes and b in b_nodes}
    return store, hops, classes, chem, dopf, target, a_nodes, b_nodes, target_pairs


def test_three_hop_search_and_support_match_brute_force(setup):
    store, hops, classes, chem, dopf, target, a_nodes, b_nodes, target_pairs = setup
    stats = search(store, hops, classes, target, chem, dopf, 3, min_support=2, time_limit=600)
    assert stats.complete and stats.evaluated > 20
    three = ThreeHop(store, classes.sets, chem, dopf, cache_gb=0.1)
    for body, _, _, support, _ in stats.rows[:300]:
        predicted = simple_predictions(store, classes, body, a_nodes, b_nodes)
        assert support == len(predicted & target_pairs), body
        assert three.support(body, cap=10**9) == (len(predicted), False), body


def test_non_simple_walks_are_excluded():
    """Walk a-b-a-d (c == a) and a-d-c-d (b == d) must not count; a-b-c-d must."""
    import graphblas as gb
    from biolink_rules.threehop import ThreeHop, as_counts

    def m(edges, n=5):
        r, c = zip(*edges)
        return as_counts(gb.Matrix.from_coo(list(r), list(c), True, nrows=n, ncols=n))

    # nodes: a=0 b=1 c=2 d=3; edges allow walks 0-1-0-3 and 0-3-2-3 but no simple path
    a1, a2, a3 = m([(0, 1), (0, 3)]), m([(1, 0), (3, 2)]), m([(0, 3), (2, 3)])
    diag_x, diag_y = ThreeHop.diagonals(a1, a2, a3)
    mask = gb.Matrix.from_coo([0], [3], True, nrows=5, ncols=5)
    total = gb.Matrix(gb.dtypes.INT64, 5, 5)
    total(mask.S) << a1.mxm(a2, gb.semiring.plus_times).mxm(a3, gb.semiring.plus_times)
    assert total.nvals == 1
    assert ThreeHop.simple_counts(total, a1, a2, a3, diag_x, diag_y, mask).nvals == 0
    # add a genuine simple path 0-1-2-3
    a2 = m([(1, 0), (3, 2), (1, 2)])
    diag_x, diag_y = ThreeHop.diagonals(a1, a2, a3)
    total(mask.S) << a1.mxm(a2, gb.semiring.plus_times).mxm(a3, gb.semiring.plus_times)
    assert ThreeHop.simple_counts(total, a1, a2, a3, diag_x, diag_y, mask).nvals == 1


def test_three_hop_merging_is_exact_and_complete(setup):
    store, hops, classes, chem, dopf, target, a_nodes, b_nodes, target_pairs = setup
    raw = search(store, hops, classes, target, chem, dopf, 3, 2, 600, merge_equivalent=False)
    merged = search(store, hops, classes, target, chem, dopf, 3, 2, 600)
    assert merged.complete and merged.evaluated < raw.evaluated, (merged.evaluated, raw.evaluated)

    def passing(stats):
        return [frozenset(simple_predictions(store, classes, r[0], a_nodes, b_nodes)) for r in stats.rows if r[3] >= 2]

    merged_sets = passing(merged)
    assert set(passing(raw)) <= set(merged_sets)
    for row in merged.rows:  # supports of merged representatives are still exact
        assert row[3] == len(simple_predictions(store, classes, row[0], a_nodes, b_nodes) & target_pairs)
