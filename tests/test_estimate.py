"""Knuth probe estimates must match the exactly enumerated coordinate-ordered tree."""

import statistics

import pytest

from biolink_rules.search import HopContexts, body_space, estimate_size, restrict, search


@pytest.fixture(scope="module")
def setup(world):
    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    target = restrict(store.hop_matrix(hops.sets[treats], target_only=True),
                      rows=store.class_vector(classes.sets[chem]), cols=store.class_vector(classes.sets[dopf]))
    return store, hops, classes, chem, dopf, target


def ordered_tree(setup, n_hops, min_support):
    """Exact (evaluated, passed, distinct passing bodies) of the coordinate-ordered tree."""
    store, hops, classes, chem, dopf, target = setup
    root, children, overlap, _ = body_space(store, hops, classes, chem, dopf, n_hops, 0.1,
                                            HopContexts(store, hops, classes))
    evaluated, passed, bodies = 0, 0, {root}

    def walk(body, mask, last):
        nonlocal evaluated, passed
        for coord, child in children(body):
            if coord < last:
                continue
            evaluated += 1
            child_overlap = overlap(child, mask)
            if child_overlap.nvals >= min_support:
                passed += 1
                bodies.add(child)
                walk(child, child_overlap, coord)

    walk(root, overlap(root, target), 0)
    return evaluated, passed, bodies


@pytest.mark.parametrize("n_hops,min_support,bias,memo_depth", [
    (2, 1, 0.0, -1),  # plain Knuth
    (3, 2, 0.0, -1),
    (2, 1, 1.0, -1),  # overlap-biased
    (3, 2, 2.0, -1),
    (3, 2, 0.0, 3),  # adaptive
    (3, 2, 2.0, 50),
])
def test_estimate_is_unbiased_for_ordered_tree(setup, n_hops, min_support, bias, memo_depth):
    store, hops, classes, chem, dopf, target = setup
    evaluated, passed, _ = ordered_tree(setup, n_hops, min_support)
    probes = estimate_size(store, hops, classes, target, chem, dopf, n_hops, min_support, probes=3000, seed=1,
                           cache_gb=0.1, bias=bias, memo_depth=memo_depth)
    for field, exact in (("evaluated", evaluated), ("passed", passed)):
        totals = [sum(p[field].values()) for p in probes]
        assert abs(statistics.mean(totals) - exact) < 3 * statistics.stdev(totals) / len(totals) ** 0.5, field
    est_evaluated = statistics.mean(sum(p["evaluated"].values()) for p in probes)
    by_classes = {}
    for p in probes:
        for k, v in p["by_classes"].items():
            by_classes[k] = by_classes.get(k, 0) + v / len(probes)
    assert sum(by_classes.values()) == pytest.approx(est_evaluated)


@pytest.mark.parametrize("n_hops,min_support", [(2, 1), (3, 2)])
def test_ordered_refinement_reaches_every_passing_body(setup, n_hops, min_support):
    store, hops, classes, chem, dopf, target = setup
    _, _, bodies = ordered_tree(setup, n_hops, min_support)
    stats = search(store, hops, classes, target, chem, dopf, n_hops, min_support, 600, merge_equivalent=False)
    assert stats.complete
    assert {row[0] for row in stats.rows if row[3] >= min_support} == bodies
