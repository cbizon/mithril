"""Search supports must equal brute-force counts under the same lattice semantics."""

import json
import random

import pytest

from biolink_rules.build import build
from biolink_rules.ingest import ingest
from biolink_rules.lattice import class_lattice, hop_lattice, read_atoms, read_category_sets
from biolink_rules.search import Store, restrict, search

PREDICATES = ["biolink:treats", "biolink:affects", "biolink:interacts_with", "biolink:has_phenotype", "biolink:causes"]
CATEGORIES = [
    ["biolink:NamedThing", "biolink:ChemicalEntity", "biolink:SmallMolecule"],
    ["biolink:NamedThing", "biolink:BiologicalEntity", "biolink:Gene"],
    ["biolink:NamedThing", "biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"],
    ["biolink:NamedThing", "biolink:DiseaseOrPhenotypicFeature", "biolink:PhenotypicFeature"],
]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rng = random.Random(7)
    tmp = tmp_path_factory.mktemp("search")
    kg = tmp / "kg"
    kg.mkdir()
    nodes = [{"id": f"N{i}", "category": CATEGORIES[i % 4]} for i in range(24)]
    edges = []
    for _ in range(160):
        s, o = rng.sample(range(24), 2)
        edges.append({"subject": f"N{s}", "object": f"N{o}", "predicate": rng.choice(PREDICATES)})
    (kg / "nodes.jsonl").write_text("".join(json.dumps(n) + "\n" for n in nodes))
    (kg / "edges.jsonl").write_text("".join(json.dumps(e) + "\n" for e in edges))
    config = tmp / "config.yaml"
    config.write_text("exclude_hop_predicates: []\ntarget_record_exclusions: []\n")
    ingest(kg, tmp / "ingest")
    build(tmp / "ingest", config, tmp / "build")
    store = Store.load(tmp / "build", tmp / "ingest")
    classes = class_lattice(read_category_sets(tmp / "ingest/category_sets.tsv"))
    hops = hop_lattice(read_atoms(tmp / "build/atoms.tsv"))
    return store, hops, classes


def pairs_of(store, bits):
    """Set of (left, right) node pairs for a hop set, by brute force."""
    out = set()
    for bit in range(bits.bit_length()):
        if bits >> bit & 1:
            atom, reverse = divmod(bit, 2)
            for k in range(store.offsets[atom], store.offsets[atom + 1]):
                s, d = int(store.src[k]), int(store.dst[k])
                out.add((d, s) if reverse else (s, d))
    return out


def members(store, classes, c):
    return {n for n, cs in enumerate(store.node_catset) if classes.sets[c] >> int(cs) & 1}


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
