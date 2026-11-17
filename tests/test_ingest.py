import json

import numpy as np
import pytest

from mithril.ingest import EDGE_TYPE_FIELDS, edge_type_key, ingest


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def kg_dir(tmp_path):
    kg = tmp_path / "kg"
    kg.mkdir()
    write_jsonl(kg / "nodes.jsonl", [
        {"id": "CHEBI:1", "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity"]},
        {"id": "NCBIGene:1", "category": ["biolink:Gene"]},
        # same category set as CHEBI:1 in a different order -> same catset
        {"id": "CHEBI:2", "category": ["biolink:ChemicalEntity", "biolink:SmallMolecule"]},
    ])
    affects = {
        "predicate": "biolink:affects",
        "qualified_predicate": "biolink:causes",
        "object_direction_qualifier": "decreased",
        "object_aspect_qualifier": "activity",
        "species_context_qualifier": "NCBITaxon:9606",
        "sources": [
            {"resource_id": "infores:chembl", "resource_role": "primary_knowledge_source"},
            {"resource_id": "infores:agg", "resource_role": "aggregator_knowledge_source"},
        ],
    }
    write_jsonl(kg / "edges.jsonl", [
        {"subject": "CHEBI:1", "object": "NCBIGene:1", **affects},
        # differs only in an entity-valued context qualifier -> same edge type
        {"subject": "CHEBI:2", "object": "NCBIGene:1", **affects, "species_context_qualifier": "NCBITaxon:10090"},
        {"subject": "CHEBI:1", "object": "NCBIGene:1", "predicate": "biolink:affects", "negated": True},
        {"subject": "CHEBI:1", "object": "MONDO:404", "predicate": "biolink:treats"},
    ])
    return kg


def read_tsv(path):
    header, *rows = path.read_text().splitlines()
    return [dict(zip(header.split("\t"), row.split("\t"))) for row in rows]


def test_ingest_arrays_and_tables(kg_dir, tmp_path):
    out = tmp_path / "out"
    meta = ingest(kg_dir, out)

    assert meta["node_count"] == 3
    assert meta["category_set_count"] == 2
    assert meta["edge_count"] == 3
    assert meta["edge_type_count"] == 2
    assert meta["dangling_edges_dropped"] == 1

    assert (out / "node_ids.txt").read_text().split() == ["CHEBI:1", "NCBIGene:1", "CHEBI:2"]
    assert np.load(out / "node_catset.npy").tolist() == [0, 1, 0]
    assert np.load(out / "edge_subject.npy").tolist() == [0, 2, 0]
    assert np.load(out / "edge_object.npy").tolist() == [1, 1, 1]
    assert np.load(out / "edge_type.npy").tolist() == [0, 0, 1]

    catsets = read_tsv(out / "category_sets.tsv")
    assert catsets[0]["categories"] == "biolink:ChemicalEntity|biolink:SmallMolecule"
    assert catsets[0]["node_count"] == "2"

    types = read_tsv(out / "edge_types.tsv")
    assert types[0]["edge_count"] == "2"
    assert types[0]["qualified_predicate"] == "biolink:causes"
    assert types[0]["object_direction_qualifier"] == "decreased"
    assert types[0]["primary_knowledge_source"] == "infores:chembl"
    assert types[1]["negated"] == "true"
    assert "species_context_qualifier" not in types[0]


def test_edge_type_key_prefers_explicit_primary_source():
    key = edge_type_key({"predicate": "biolink:treats", "primary_knowledge_source": "infores:x"})
    assert key[EDGE_TYPE_FIELDS.index("primary_knowledge_source")] == "infores:x"
    assert key[EDGE_TYPE_FIELDS.index("predicate")] == "biolink:treats"
