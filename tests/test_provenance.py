"""Path-type provenance and its counters must reproduce brute-force pair sets."""

import json

import numpy as np
import pytest

from biolink_rules.counter import (
    HLL_M,
    PathTypeCounter,
    hll_estimate,
    hll_register,
    merge_sketch_tables,
    path_only,
    sketch_table,
    type_table,
)
from biolink_rules.pathsearch import body_masks, body_of, mine
from biolink_rules.provenance import count_pairs, pair_types, quotient, type_sketches, typed_graph
from brute import predicted


def test_pair_bitsets_and_sketches_are_those_of_the_predicted_pairs(mining):
    m = mining
    problem, store, q = m.problem, m.problem.store, m.q
    overlap = PathTypeCounter(q, *m.table, len(problem.pairs))
    support = PathTypeCounter(q, *path_only(q, *m.sketch), 0, kind="hll")
    pairs = [(int(a), int(d)) for a, d in problem.pairs]
    for x in mine(problem, q, m.table, m.direct, None, 1, conditioned=False):
        for i in range(0, len(x.overlap), 5):
            body = body_of(problem, x, i)
            want = predicted(store, problem.classes, body, m.a_nodes, m.b_nodes)
            masks = body_masks(problem, q, body)[1:]
            bits = overlap.value(*masks)
            joined = {pairs[p] for p in range(len(pairs)) if bits[p >> 6] >> np.uint64(p & 63) & np.uint64(1)}
            assert joined == want & m.target_pairs
            assert overlap.count(*masks) == len(joined)
            expected = np.zeros(HLL_M, dtype=np.uint8)
            for a, d in want:
                idx, rank = hll_register(a, d, store.n_nodes)
                expected[idx] = max(expected[idx], rank)
            assert (support.value(*masks) == expected).all()


def test_direct_edge_types_are_those_of_the_pairs_edges(mining):
    m = mining
    store, q = m.problem.store, m.q
    pair, edge_type = m.direct
    found = {(int(p), int(e)) for p, e in zip(pair, edge_type)}
    expected = set()
    atom = np.repeat(np.arange(len(store.offsets) - 1), np.diff(store.offsets))
    index = {(int(a), int(d)): i for i, (a, d) in enumerate(m.problem.pairs)}
    for k in range(len(store.src)):
        s, d = int(store.src[k]), int(store.dst[k])
        for bit, ends in ((2 * atom[k], (s, d)), (2 * atom[k] + 1, (d, s))):
            if ends in index and q.edge_of_bit[bit] >= 0:
                expected.add((index[ends], int(q.edge_of_bit[bit])))
    assert found == expected and found


def test_non_simple_walks_are_excluded(tmp_path):
    """Walks a-b-a-d (c == a) and a-d-c-d (b == d) must not count; a-b-c-d must."""
    from biolink_rules.build import build
    from biolink_rules.ingest import ingest
    from biolink_rules.lattice import class_lattice, hop_lattice, read_atoms, read_category_sets
    from biolink_rules.store import Store

    def provenance(name, edges):
        tmp = tmp_path / name
        kg = tmp / "kg"
        kg.mkdir(parents=True)
        (kg / "nodes.jsonl").write_text("".join(
            json.dumps({"id": f"N{i}", "category": ["biolink:NamedThing"]}) + "\n" for i in range(4)))
        (kg / "edges.jsonl").write_text("".join(
            json.dumps({"subject": f"N{s}", "object": f"N{o}", "predicate": "biolink:affects"}) + "\n"
            for s, o in edges))
        (tmp / "config.yaml").write_text("exclude_hop_predicates: []\ntarget_record_exclusions: []\n")
        ingest(kg, tmp / "ingest")
        build(tmp / "ingest", tmp / "config.yaml", tmp / "build")
        store = Store.load(tmp / "build", tmp / "ingest")
        hops = hop_lattice(read_atoms(tmp / "build/atoms.tsv"))
        classes = class_lattice(read_category_sets(tmp / "ingest/category_sets.tsv"))
        q = quotient(hops, classes, store.n_bits, store.n_catsets)
        graph = typed_graph(store, q)
        _, types, _, _ = pair_types(graph, q, np.array([[0, 3]]))
        hop, node = (1 << q.n_edge) - 1, (1 << q.n_node) - 1
        everyone = np.ones(store.n_nodes, dtype=bool)
        three_hop = count_pairs(graph, q, np.array([0]), everyone, (0, hop, node, hop, node, hop))
        return {int(e3) != q.none_edge for e3 in q.unpack(types)[4]}, three_hop

    # 0-1, 0-3, 2-3: the only 3-step walks from 0 to 3 are 0-1-0-3 and 0-3-2-3
    has_three_hop, predicted_pairs = provenance("walks", [(0, 1), (0, 3), (2, 3)])
    assert has_three_hop == {False} and predicted_pairs == 0
    # adding 1-2 gives the simple path 0-1-2-3 (and, in the whole graph, 0-3-2-1)
    has_three_hop, predicted_pairs = provenance("path", [(0, 1), (0, 3), (2, 3), (1, 2)])
    assert has_three_hop == {False, True} and predicted_pairs == 2


def test_hll_estimate_is_accurate():
    rng = np.random.default_rng(0)
    for n in (0, 1, 50, 2_000, 100_000):
        registers = np.zeros(HLL_M, dtype=np.uint8)
        for a, d in zip(rng.integers(0, 10**6, n), rng.integers(0, 10**6, n)):
            idx, rank = hll_register(a, d, 10**6)
            registers[idx] = max(registers[idx], rank)
        assert hll_estimate(registers) == pytest.approx(n, rel=0.1, abs=0.5)


def test_quotient_merges_only_indistinguishable_items(mining):
    store, hops, q = mining.problem.store, mining.problem.hops, mining.q
    for s in hops.sets:  # every label is a union of whole edge types
        members = {int(q.edge_of_bit[b]) for b in range(store.n_bits) if s >> b & 1}
        others = {int(q.edge_of_bit[b]) for b in range(store.n_bits) if not s >> b & 1}
        assert not members & others


def test_sketch_tables_of_subject_stripes_merge_to_the_whole(mining):
    m = mining
    parts = [sketch_table(*type_sketches(m.graph, m.q, m.subjects[i::3], m.is_object, table_bits=16))
             for i in range(3)]
    merged = merge_sketch_tables(merge_sketch_tables(parts[0], parts[1]), parts[2])
    for got, want in zip(merged, m.sketch):
        assert (got == want).all()


def test_type_table_groups_pairs_by_type():
    types, tptr, pairs = type_table(np.array([0, 0, 1, 2]), np.array([7, 5, 7, 7]), 3)
    assert types.tolist() == [5, 7] and tptr.tolist() == [0, 1, 4] and pairs.tolist() == [0, 0, 1, 2]
