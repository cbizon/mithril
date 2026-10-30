"""Search supports must equal brute-force counts under the same lattice semantics."""

import pytest

from biolink_rules.search import restrict, search
from brute import members, pairs_of

@pytest.mark.parametrize("n_hops", [1, 2])
def test_search_supports_match_brute_force(world, n_hops):
    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    a_nodes, b_nodes = members(store, classes, chem), members(store, classes, dopf)
    target_pairs = {(a, b) for a, b in pairs_of(store, hops.sets[treats]) if a in a_nodes and b in b_nodes}
    target = restrict(
        store.hop_matrix(hops.sets[treats], target_only=True),
        rows=store.class_vector(classes.sets[chem]),
        cols=store.class_vector(classes.sets[dopf]),
    )
    assert target.nvals == len(target_pairs)

    stats = search(store, hops, classes, target, chem, dopf, n_hops, min_support=1, time_limit=600)
    assert stats.complete
    assert stats.evaluated >= 10
    for body, _, _, support, _ in stats.rows:
        if n_hops == 1:
            predicted = pairs_of(store, body[0])
        else:
            h1, c1, h2 = body
            mid = members(store, classes, c1)
            first, second = pairs_of(store, h1), pairs_of(store, h2)
            predicted = {(a, d) for a, b in first if b in mid for b2, d in second if b2 == b}
        assert support == len(predicted & target_pairs), body


def predicted_pairs(store, classes, body, a_nodes, b_nodes):
    h1, c1, h2 = body
    mid = members(store, classes, c1)
    return frozenset((a, d) for a, b in pairs_of(store, h1) if a in a_nodes and b in mid
                     for b2, d in pairs_of(store, h2) if b2 == b and d in b_nodes)


def test_merging_equivalent_bodies_is_exact_and_complete(world):
    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    target = restrict(store.hop_matrix(hops.sets[treats], target_only=True),
                      rows=store.class_vector(classes.sets[chem]), cols=store.class_vector(classes.sets[dopf]))
    a_nodes, b_nodes = members(store, classes, chem), members(store, classes, dopf)

    raw = search(store, hops, classes, target, chem, dopf, 2, 1, 600, merge_equivalent=False)
    merged = search(store, hops, classes, target, chem, dopf, 2, 1, 600)
    assert merged.complete and merged.evaluated < raw.evaluated, (merged.evaluated, raw.evaluated)
    raw_sets = {predicted_pairs(store, classes, r[0], a_nodes, b_nodes) for r in raw.rows if r[3] >= 1}
    merged_sets = [predicted_pairs(store, classes, r[0], a_nodes, b_nodes) for r in merged.rows if r[3] >= 1]
    # every distinct passing rule of the raw search is still found
    assert raw_sets <= set(merged_sets)
