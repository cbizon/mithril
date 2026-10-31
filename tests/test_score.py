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
    problem = Problem(store, hops, classes, HopContexts(store, hops, classes), chem, dopf, target, "t",
                      hops.sets[treats])
    scorer = Scorer(problem, cache_gb=0.1)
    a_nodes, b_nodes = members(store, classes, chem), members(store, classes, dopf)
    target_pairs = {(a, b) for a, b in pairs_of(store, hops.sets[treats]) if a in a_nodes and b in b_nodes}

    stats = search(store, hops, classes, target, chem, dopf, 2, min_support=1, time_limit=600)
    for body, *_ in stats.rows:
        h1, c1, h2 = body
        mid = members(store, classes, c1)
        predicted = {(a, d) for a, b in pairs_of(store, h1) if a in a_nodes and b in mid
                     for b2, d in pairs_of(store, h2) if b2 == b and d in b_nodes}
        predicted = {(a, d) for a, d in predicted if a != d}
        assert scorer.support((0, body), cap=10**9) == (len(predicted), False), body
        assert scorer.overlap_pairs((0, body)).nvals == len(predicted & target_pairs), body
        if len(predicted) > 2:
            count, capped = scorer.support((0, body), cap=1)
            assert capped and count > 1


def test_direct_and_conjunctive_rules_match_brute_force(world):
    from biolink_rules.conjunctive import mine_conjunctions, mine_one_hop
    from biolink_rules.score import Scored

    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    target = restrict(store.hop_matrix(hops.sets[treats], target_only=True),
                      rows=store.class_vector(classes.sets[chem]), cols=store.class_vector(classes.sets[dopf]))
    problem = Problem(store, hops, classes, HopContexts(store, hops, classes), chem, dopf, target, "t",
                      hops.sets[treats])
    scorer = Scorer(problem, cache_gb=0.1)
    a_nodes, b_nodes = members(store, classes, chem), members(store, classes, dopf)
    target_pairs = {(a, b) for a, b in pairs_of(store, hops.sets[treats]) if a in a_nodes and b in b_nodes}

    def direct_pairs(bits):
        return {(a, d) for a, d in pairs_of(store, bits) if a in a_nodes and d in b_nodes and a != d}

    one_hop = mine_one_hop(problem, scorer, floor=1)
    assert one_hop
    for rule in one_hop:
        bits = rule.body[0]
        assert bits & problem.target_bits == 0  # the target edge is never a predictor
        predicted = direct_pairs(bits)
        assert (rule.support, rule.overlap) == (len(predicted), len(predicted & target_pairs))

    stats = search(store, hops, classes, target, chem, dopf, 2, min_support=1, time_limit=600)
    paths = [Scored((0, body), name, sup, 0, False, target.nvals) for body, name, _, sup, _ in stats.rows if sup >= 1]
    conj = mine_conjunctions(problem, scorer, paths, floor=1)
    assert conj
    for rule in conj:
        bits, (h1, c1, h2) = rule.body
        mid = members(store, classes, c1)
        path = {(a, d) for a, b in pairs_of(store, h1) if a in a_nodes and b in mid
                for b2, d in pairs_of(store, h2) if b2 == b and d in b_nodes}
        predicted = path & direct_pairs(bits)
        assert (rule.support, rule.overlap) == (len(predicted), len(predicted & target_pairs)), rule.name


def test_greedy_novelty_rejects_redundant_rules(world):
    from biolink_rules.score import Scored, greedy_rule_set

    store, hops, classes = world
    (treats,) = [i for i, labels in enumerate(hops.labels) if "treats>" in labels]
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (dopf,) = [i for i, labels in enumerate(classes.labels) if "biolink:DiseaseOrPhenotypicFeature" in labels]
    target = restrict(store.hop_matrix(hops.sets[treats], target_only=True),
                      rows=store.class_vector(classes.sets[chem]), cols=store.class_vector(classes.sets[dopf]))
    problem = Problem(store, hops, classes, HopContexts(store, hops, classes), chem, dopf, target, "t",
                      hops.sets[treats])
    scorer = Scorer(problem, cache_gb=0.1)
    stats = search(store, hops, classes, target, chem, dopf, 2, min_support=1, time_limit=600)
    body = (0, stats.rows[0][0])
    overlap = scorer.overlap_pairs(body).nvals
    rule = Scored(body, "r", overlap, overlap * 2, False, target.nvals)
    twin = Scored(body, "twin", overlap, overlap * 2, False, target.nvals)
    # an exact duplicate adds 0 new pairs: kept with no thresholds, dropped by novelty
    assert len(greedy_rule_set(scorer, [rule, twin], k=5)) == 2
    chosen = greedy_rule_set(scorer, [rule, twin], k=5, min_novel=0.5)
    assert [c[0].name for c in chosen] == ["r"]
    assert chosen[0][1:] == (overlap, overlap / target.nvals)
