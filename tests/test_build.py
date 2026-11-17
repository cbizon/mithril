import json

import numpy as np
import pytest

from mithril.build import build
from mithril.ingest import ingest


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def built(tmp_path):
    kg = tmp_path / "kg"
    kg.mkdir()
    write_jsonl(kg / "nodes.jsonl", [
        {"id": "A", "category": ["biolink:SmallMolecule"]},
        {"id": "B", "category": ["biolink:Disease"]},
        {"id": "C", "category": ["biolink:Disease"]},
    ])
    manual = {"agent_type": "manual_agent", "knowledge_level": "knowledge_assertion"}
    mined = {"agent_type": "text_mining_agent", "knowledge_level": "not_provided"}
    write_jsonl(kg / "edges.jsonl", [
        # A treats B twice: once text-mined, once curated -> one pair, target evidence
        {"subject": "A", "object": "B", "predicate": "biolink:treats", **mined},
        {"subject": "A", "object": "B", "predicate": "biolink:treats", **manual},
        # A treats C only text-mined -> predictor pair, not target evidence
        {"subject": "A", "object": "C", "predicate": "biolink:treats", **mined},
        # negated record -> not target evidence
        {"subject": "B", "object": "C", "predicate": "biolink:related_to", "negated": True, **manual},
        {"subject": "B", "object": "C", "predicate": "biolink:subclass_of", **manual},
        {"subject": "C", "object": "C", "predicate": "biolink:related_to", **manual},
    ])
    config = tmp_path / "config.yaml"
    config.write_text(
        "exclude_hop_predicates: [biolink:subclass_of]\n"
        "target_record_exclusions:\n"
        "  - {agent_type: text_mining_agent}\n"
        "  - {negated: 'true'}\n"
    )
    ingest(kg, tmp_path / "ingest")
    out = tmp_path / "build"
    meta = build(tmp_path / "ingest", config, out)
    return meta, out


def test_build_counts(built):
    meta, _ = built
    assert meta["records_in"] == 6
    assert meta["records_excluded_hop_predicate"] == 1
    assert meta["records_self_loop"] == 1
    assert meta["atom_count"] == 2
    assert meta["pairs"] == 3
    assert meta["target_pairs"] == 1


def test_build_arrays(built):
    _, out = built
    offsets = np.load(out / "atom_offsets.npy").tolist()
    src = np.load(out / "edge_src.npy").tolist()
    dst = np.load(out / "edge_dst.npy").tolist()
    ok = np.load(out / "edge_target_ok.npy").tolist()
    # atom 0 = treats: (A,B) target evidence, (A,C) not; atom 1 = related_to (negated) B->C
    assert offsets == [0, 2, 3]
    assert list(zip(src, dst, ok)) == [(0, 1, True), (0, 2, False), (1, 2, False)]
    rows = (out / "atoms.tsv").read_text().splitlines()
    assert rows[1].split("\t")[1] == "biolink:treats"
    assert rows[1].split("\t")[-2:] == ["2", "1"]
    # self-loop removal can leave an atom with no edges
    assert rows[2].split("\t")[-2:] == ["1", "0"]


def test_unknown_exclusion_field_fails(tmp_path, built):
    _, out = built
    config = tmp_path / "bad.yaml"
    config.write_text("target_record_exclusions:\n  - {agent: x}\n")
    with pytest.raises(ValueError, match="agent"):
        build(out.parent / "ingest", config, tmp_path / "bad")


def test_config_extends(tmp_path):
    from mithril.build import load_config
    (tmp_path / "base.yaml").write_text("a: 1\nb: [x]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nb: [y]\nc: 3\n")
    assert load_config(tmp_path / "child.yaml") == {"a": 1, "b": ["y"], "c": 3}
