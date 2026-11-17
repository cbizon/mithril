"""Shared small random graph for search/score tests."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from mithril.build import build
from mithril.ingest import ingest
from mithril.lattice import class_lattice, hop_lattice, read_atoms, read_category_sets
from mithril.store import Store

PREDICATES = ["biolink:treats", "biolink:affects", "biolink:interacts_with", "biolink:has_phenotype", "biolink:causes"]
CATEGORIES = [
    ["biolink:NamedThing", "biolink:ChemicalEntity", "biolink:SmallMolecule"],
    ["biolink:NamedThing", "biolink:BiologicalEntity", "biolink:Gene"],
    ["biolink:NamedThing", "biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"],
    ["biolink:NamedThing", "biolink:DiseaseOrPhenotypicFeature", "biolink:PhenotypicFeature"],
]


def make_world(tmp):
    """Ingest + build the random graph under ``tmp`` (ingest/, build/, config.yaml)."""
    rng = random.Random(7)
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


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("search")
    make_world(tmp)
    store = Store.load(tmp / "build", tmp / "ingest")
    classes = class_lattice(read_category_sets(tmp / "ingest/category_sets.tsv"))
    hops = hop_lattice(read_atoms(tmp / "build/atoms.tsv"))
    return store, hops, classes


def make_problem(world, near_duplicate: float = 1.0):
    """Chemical -treats-> DiseaseOrPhenotypicFeature on the shared graph."""
    from mithril.contexts import HopContexts
    from mithril.problem import Problem, find

    store, hops, classes = world
    chem, dopf = find(classes, "biolink:ChemicalEntity"), find(classes, "biolink:DiseaseOrPhenotypicFeature")
    target_bits = hops.sets[find(hops, "treats>")]
    pairs = store.target_pairs(target_bits, store.in_class(classes.sets[chem]), store.in_class(classes.sets[dopf]))
    return Problem(store, hops, classes, HopContexts(store, hops, classes, near_duplicate), chem, dopf, pairs, "t",
                   target_bits)


@pytest.fixture(scope="module")
def mining(world):
    """The problem with its provenance: everything the counters and the search need."""
    from mithril.counter import sketch_table, type_table
    from mithril.provenance import pair_types, quotient, type_sketches, typed_graph

    problem = make_problem(world)
    store, classes = problem.store, problem.classes
    q = quotient(problem.hops, classes, store.n_bits, store.n_catsets)
    graph = typed_graph(store, q)
    pair, types, direct_pair, direct_type = pair_types(graph, q, problem.pairs)
    is_object = store.in_class(classes.sets[problem.object_class])
    subjects = np.flatnonzero(store.in_class(classes.sets[problem.subject_class]))
    return SimpleNamespace(
        world=world, problem=problem, q=q, graph=graph, subjects=subjects, is_object=is_object,
        table=type_table(pair, types, len(problem.pairs)), direct=(direct_pair, direct_type),
        sketch=sketch_table(*type_sketches(graph, q, subjects, is_object, table_bits=16)),
        a_nodes=set(subjects.tolist()), b_nodes=set(np.flatnonzero(is_object).tolist()),
        target_pairs={(int(a), int(d)) for a, d in problem.pairs})
