"""Scorer support/overlap must equal brute-force pair counts."""


from biolink_rules.problem import Problem
from biolink_rules.score import Scorer
from biolink_rules.search import HopContexts, restrict, search
from brute import members, pairs_of


def test_scorer_matches_brute_force(world):
    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    target = restrict(store.hop_matrix(hops.sets[treats], target_only=True),
                      rows=store.class_vector(classes.sets[chem]), cols=store.class_vector(classes.sets[dopf]))
    problem = Problem(store, hops, classes, HopContexts(store, hops, classes), chem, dopf, target, "t")
    scorer = Scorer(problem, cache_gb=0.1)
    a_nodes, b_nodes = members(store, classes, chem), members(store, classes, dopf)
    target_pairs = {(a, b) for a, b in pairs_of(store, hops.sets[treats]) if a in a_nodes and b in b_nodes}

    stats = search(store, hops, classes, target, chem, dopf, 2, min_support=1, time_limit=600)
    for body, *_ in stats.rows:
        h1, c1, h2 = body
        mid = members(store, classes, c1)
        predicted = {(a, d) for a, b in pairs_of(store, h1) if a in a_nodes and b in mid
                     for b2, d in pairs_of(store, h2) if b2 == b and d in b_nodes}
        assert scorer.support(body, cap=10**9) == (len(predicted), False), body
        assert scorer.overlap_pairs(body).nvals == len(predicted & target_pairs), body
        if len(predicted) > 2:
            count, capped = scorer.support(body, cap=1)
            assert capped and count > 1
