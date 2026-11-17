"""Stage 0a: stream a KGX graph once into compact integer arrays.

Everything downstream (vocabulary choice, pseudo-predicate encoding, matrix
building) works from this output, so the 27 GB JSONL is parsed exactly once.

Each edge is reduced to (subject index, object index, edge type id). An edge
type is the interned tuple of the edge's predicate plus every enumerated-value
qualifier and provenance field in ``EDGE_TYPE_FIELDS``. Entity-valued context
qualifiers (species, anatomy, disease context, stage, ...) are deliberately
left out: they have huge cardinality and are not part of the rule semantics.

Each node is reduced to an interned category set. KGX ``category`` lists hold
the full ancestor closure in no particular order, so no ordering is assumed.
"""

from __future__ import annotations

import json
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import orjson

EDGE_TYPE_FIELDS = (
    "predicate",
    "qualified_predicate",
    "object_direction_qualifier",
    "object_aspect_qualifier",
    "subject_direction_qualifier",
    "subject_aspect_qualifier",
    "subject_form_or_variant_qualifier",
    "object_form_or_variant_qualifier",
    "causal_mechanism_qualifier",
    "negated",
    "knowledge_level",
    "agent_type",
    "primary_knowledge_source",
)


def field_value(value) -> str:
    """Normalize a KGX field value to a string ('' when absent).

    >>> field_value(None), field_value(True), field_value(["b", "a"]), field_value("x")
    ('', 'true', 'a|b', 'x')
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "|".join(sorted(str(v) for v in value))
    return str(value)


def primary_knowledge_source(edge: dict) -> str | None:
    """Return the edge's primary knowledge source infores, if any."""
    if "primary_knowledge_source" in edge:
        return edge["primary_knowledge_source"]
    for source in edge.get("sources") or ():
        if source.get("resource_role") == "primary_knowledge_source":
            return source["resource_id"]
    return None


def edge_type_key(edge: dict) -> tuple[str, ...]:
    """Return the interning key for an edge, ordered as ``EDGE_TYPE_FIELDS``."""
    values = dict(edge, primary_knowledge_source=primary_knowledge_source(edge))
    return tuple(field_value(values.get(field)) for field in EDGE_TYPE_FIELDS)


def read_nodes(nodes_path: Path) -> tuple[list[str], np.ndarray, list[tuple[str, ...]]]:
    """Read nodes.jsonl into (node ids, per-node category-set id, category sets)."""
    node_ids: list[str] = []
    catset_of_node = array("i")
    catset_index: dict[tuple[str, ...], int] = {}
    with open(nodes_path, "rb") as handle:
        for line in handle:
            node = orjson.loads(line)
            catset = tuple(sorted(node.get("category") or ()))
            node_ids.append(node["id"])
            catset_of_node.append(catset_index.setdefault(catset, len(catset_index)))
    catsets = sorted(catset_index, key=catset_index.get)
    return node_ids, np.frombuffer(catset_of_node, dtype=np.int32), catsets


def read_edges(
    edges_path: Path, node_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, ...]], Counter, int]:
    """Read edges.jsonl into subject/object/type arrays.

    Edges whose subject or object is missing from the node file are dropped
    and counted (returned as ``dangling``) so the caller can report them.
    """
    subjects, objects, types = array("i"), array("i"), array("i")
    type_index: dict[tuple[str, ...], int] = {}
    type_counts: Counter = Counter()
    dangling = 0
    with open(edges_path, "rb") as handle:
        for line in handle:
            edge = orjson.loads(line)
            subject = node_index.get(edge["subject"])
            object_ = node_index.get(edge["object"])
            if subject is None or object_ is None:
                dangling += 1
                continue
            type_id = type_index.setdefault(edge_type_key(edge), len(type_index))
            type_counts[type_id] += 1
            subjects.append(subject)
            objects.append(object_)
            types.append(type_id)
    edge_types = sorted(type_index, key=type_index.get)
    return (
        np.frombuffer(subjects, dtype=np.int32),
        np.frombuffer(objects, dtype=np.int32),
        np.frombuffer(types, dtype=np.int32),
        edge_types,
        type_counts,
        dangling,
    )


def ingest(kg_dir: Path, out_dir: Path) -> dict:
    """Ingest ``kg_dir/{nodes,edges}.jsonl`` into ``out_dir``; return the meta dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids, catset_of_node, catsets = read_nodes(kg_dir / "nodes.jsonl")
    (out_dir / "node_ids.txt").write_text("\n".join(node_ids) + "\n")
    np.save(out_dir / "node_catset.npy", catset_of_node)
    catset_counts = np.bincount(catset_of_node, minlength=len(catsets))
    with open(out_dir / "category_sets.tsv", "w") as handle:
        handle.write("catset_id\tnode_count\tcategories\n")
        for catset_id, catset in enumerate(catsets):
            handle.write(f"{catset_id}\t{catset_counts[catset_id]}\t{'|'.join(catset)}\n")

    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    subjects, objects, types, edge_types, type_counts, dangling = read_edges(
        kg_dir / "edges.jsonl", node_index
    )
    np.save(out_dir / "edge_subject.npy", subjects)
    np.save(out_dir / "edge_object.npy", objects)
    np.save(out_dir / "edge_type.npy", types)
    with open(out_dir / "edge_types.tsv", "w") as handle:
        handle.write("\t".join(("edge_type_id", "edge_count", *EDGE_TYPE_FIELDS)) + "\n")
        for type_id, key in enumerate(edge_types):
            handle.write("\t".join((str(type_id), str(type_counts[type_id]), *key)) + "\n")

    meta = {
        "kg_dir": str(kg_dir.resolve()),
        "node_count": len(node_ids),
        "category_set_count": len(catsets),
        "edge_count": int(len(types)),
        "edge_type_count": len(edge_types),
        "dangling_edges_dropped": dangling,
        "edge_type_fields": list(EDGE_TYPE_FIELDS),
    }
    release_metadata = kg_dir / "release-metadata.json"
    if release_metadata.exists():
        meta["release"] = json.loads(release_metadata.read_text())
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
