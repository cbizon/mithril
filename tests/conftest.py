"""Shared small random graph for search/score tests."""

import json
import random

import pytest

from biolink_rules.build import build
from biolink_rules.ingest import ingest
from biolink_rules.lattice import class_lattice, hop_lattice, read_atoms, read_category_sets
from biolink_rules.search import Store

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
