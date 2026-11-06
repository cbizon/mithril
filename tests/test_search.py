"""Search supports must equal brute-force counts under the same lattice semantics."""

import json

import pytest

from biolink_rules.search import HopContexts, restrict, search, set_bits
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


@pytest.mark.parametrize("share", [0.0, 0.6, 0.95])
def test_near_duplicate_stepping_keeps_lattice_reachable(world, share):
    store, hops, classes = world
    plain = HopContexts(store, hops, classes)
    pruned = HopContexts(store, hops, classes, near_duplicate=share)
    for left in range(len(classes.sets)):
        for right in range(len(classes.sets)):
            if not plain.mask(left, right):
                continue
            lattice, _ = pruned.lattice(left, right)
            kids = pruned.kept_children(left, right)
            counts = pruned.edge_counts(left, right)
            weight = [sum(counts.get(b, 0) for b in set_bits(bits)) for bits in lattice.sets]
            near = {i for i in range(len(lattice.sets)) if i != lattice.root and lattice.children[i]
                    and max(weight[c] for c in lattice.children[i]) >= share * weight[i]}
            seen, stack = {lattice.root}, [lattice.root]
            while stack:
                node = stack.pop()
                for c in kids[node]:
                    assert c not in near
                    assert lattice.sets[c] != lattice.sets[node] and lattice.sets[c] & ~lattice.sets[node] == 0
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
            # every node that is not stepped over stays reachable from the root
            assert seen == set(range(len(lattice.sets))) - near


def test_near_duplicate_off_is_the_plain_lattice(world):
    store, hops, classes = world
    contexts = HopContexts(store, hops, classes, near_duplicate=1.0)
    for left in range(len(classes.sets)):
        for right in range(len(classes.sets)):
            if contexts.mask(left, right):
                lattice, _ = contexts.lattice(left, right)
                for bits in lattice.sets:
                    expected = [lattice.sets[c] for c in lattice.children[lattice.sets.index(bits)]]
                    assert contexts.children(bits, left, right) == expected


def test_near_duplicate_label_is_stepped_over(tmp_path):
    from biolink_rules.build import build
    from biolink_rules.ingest import ingest
    from biolink_rules.lattice import class_lattice, hop_lattice, read_atoms, read_category_sets
    from biolink_rules.search import Store

    kg = tmp_path / "kg"
    kg.mkdir()
    chems = [{"id": f"C{i}", "category": ["biolink:NamedThing", "biolink:ChemicalEntity"]} for i in range(20)]
    genes = [{"id": f"G{i}", "category": ["biolink:NamedThing", "biolink:BiologicalEntity", "biolink:Gene"]}
             for i in range(20)]
    edges = ([("expression", "increased", i) for i in range(10)] + [("expression", "decreased", 10)]
             + [("activity", "increased", i) for i in range(11, 16)])
    (kg / "nodes.jsonl").write_text("".join(json.dumps(n) + "\n" for n in chems + genes))
    (kg / "edges.jsonl").write_text("".join(json.dumps(
        {"subject": f"C{i}", "object": f"G{i}", "predicate": "biolink:affects",
         "object_aspect_qualifier": aspect, "object_direction_qualifier": direction}) + "\n"
        for aspect, direction, i in edges))
    config = tmp_path / "config.yaml"
    config.write_text("exclude_hop_predicates: []\ntarget_record_exclusions: []\n")
    ingest(kg, tmp_path / "ingest")
    build(tmp_path / "ingest", config, tmp_path / "build")
    store = Store.load(tmp_path / "build", tmp_path / "ingest")
    classes = class_lattice(read_category_sets(tmp_path / "ingest/category_sets.tsv"))
    hops = hop_lattice(read_atoms(tmp_path / "build/atoms.tsv"))
    (chem,) = [i for i, labels in enumerate(classes.labels) if "biolink:ChemicalEntity" in labels]
    (gene,) = [i for i, labels in enumerate(classes.labels) if "biolink:Gene" in labels]

    def reachable(contexts):
        lattice, _ = contexts.lattice(chem, gene)
        names, stack, seen = set(), [lattice.sets[lattice.root]], set()
        while stack:
            bits = stack.pop()
            if bits in seen:
                continue
            seen.add(bits)
            names.update(lattice.labels[lattice.sets.index(bits)])
            stack += contexts.children(bits, chem, gene)
        return names

    plain = reachable(HopContexts(store, hops, classes))
    # affects[aspect=expression] has 11 edges, 10 of them in its child [..., dir=increased]
    pruned = reachable(HopContexts(store, hops, classes, near_duplicate=0.9))
    assert "affects[aspect=expression]>" in plain
    assert "affects[aspect=expression]>" not in pruned
    assert "affects[aspect=expression,dir=increased]>" in pruned
    assert "affects[aspect=expression,dir=decreased]>" in pruned
    assert pruned < plain
