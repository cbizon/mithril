"""Stage 0b: ingest output + config -> deduplicated edges grouped by semantic atom.

An *atom* is a stored predicate plus the qualifier values in ``ATOM_QUALIFIER_FIELDS``
(provenance dropped). Edges are kept in their stored orientation: query-time
predicate/qualifier expansion (``semantics``) decides which atoms, in which
orientation, a query predicate matches -- exactly as production does.

Per atom we keep each distinct (subject, object) pair once, with a flag saying
whether at least one of its records is acceptable target evidence. Self-loops are
dropped (they cannot be part of a simple path, and a == d is not a prediction).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from biolink_rules.ingest import EDGE_TYPE_FIELDS

ATOM_QUALIFIER_FIELDS = (
    "qualified_predicate",
    "object_direction_qualifier",
    "object_aspect_qualifier",
    "subject_direction_qualifier",
    "subject_aspect_qualifier",
    "subject_form_or_variant_qualifier",
    "object_form_or_variant_qualifier",
    "causal_mechanism_qualifier",
)
ATOM_FIELDS = ("predicate", *ATOM_QUALIFIER_FIELDS)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def read_edge_types(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def record_is_target_evidence(edge_type: dict[str, str], exclusions: list[dict]) -> bool:
    """True unless the edge type matches one of the exclusion entries.

    >>> record_is_target_evidence({"agent_type": "text_mining_agent", "negated": ""},
    ...                           [{"agent_type": "text_mining_agent"}])
    False
    >>> record_is_target_evidence({"agent_type": "manual_agent", "negated": ""},
    ...                           [{"agent_type": "text_mining_agent"}])
    True
    """
    for exclusion in exclusions:
        unknown = set(exclusion) - set(EDGE_TYPE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown target exclusion field(s): {sorted(unknown)}")
        if all(edge_type[field] == str(value) for field, value in exclusion.items()):
            return False
    return True


def classify_edge_types(edge_types: list[dict[str, str]], config: dict):
    """Map each ingest edge type to (atom id or -1 if excluded hop, target-evidence flag).

    Returns (type_to_atom, type_target_ok, atoms) where ``atoms`` is the ordered list
    of atom keys (tuples ordered as ``ATOM_FIELDS``).
    """
    excluded = set(config.get("exclude_hop_predicates") or ())
    exclusions = config.get("target_record_exclusions") or []
    atom_index: dict[tuple[str, ...], int] = {}
    type_to_atom = np.full(len(edge_types), -1, dtype=np.int32)
    type_target_ok = np.zeros(len(edge_types), dtype=bool)
    for type_id, edge_type in enumerate(edge_types):
        assert int(edge_type["edge_type_id"]) == type_id
        type_target_ok[type_id] = record_is_target_evidence(edge_type, exclusions)
        if edge_type["predicate"] in excluded:
            continue
        key = tuple(edge_type[field] for field in ATOM_FIELDS)
        type_to_atom[type_id] = atom_index.setdefault(key, len(atom_index))
    atoms = sorted(atom_index, key=atom_index.get)
    return type_to_atom, type_target_ok, atoms


def dedupe_atom_edges(atom: np.ndarray, src: np.ndarray, dst: np.ndarray, target_ok: np.ndarray):
    """Collapse duplicate (atom, src, dst) rows; a pair is target evidence if any record is.

    Returns arrays sorted by (atom, src, dst).
    """
    order = np.lexsort((dst, src, atom))
    atom, src, dst, target_ok = atom[order], src[order], dst[order], target_ok[order]
    starts = np.ones(len(atom), dtype=bool)
    starts[1:] = (atom[1:] != atom[:-1]) | (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    start_idx = np.flatnonzero(starts)
    merged_ok = np.maximum.reduceat(target_ok.astype(np.uint8), start_idx).astype(bool) if len(atom) else target_ok
    return atom[start_idx], src[start_idx], dst[start_idx], merged_ok


def build(ingest_dir: Path, config_path: Path, out_dir: Path) -> dict:
    """Build the atom edge store in ``out_dir``; return the meta dict."""
    config = load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_types = read_edge_types(ingest_dir / "edge_types.tsv")
    type_to_atom, type_target_ok, atoms = classify_edge_types(edge_types, config)

    types = np.load(ingest_dir / "edge_type.npy")
    src = np.load(ingest_dir / "edge_subject.npy")
    dst = np.load(ingest_dir / "edge_object.npy")
    atom = type_to_atom[types]
    target_ok = type_target_ok[types]

    hop = atom >= 0
    self_loop = src == dst
    keep = hop & ~self_loop
    atom, src, dst, target_ok = dedupe_atom_edges(atom[keep], src[keep], dst[keep], target_ok[keep])

    offsets = np.searchsorted(atom, np.arange(len(atoms) + 1)).astype(np.int64)
    np.save(out_dir / "atom_offsets.npy", offsets)
    np.save(out_dir / "edge_src.npy", src)
    np.save(out_dir / "edge_dst.npy", dst)
    np.save(out_dir / "edge_target_ok.npy", target_ok)

    n_edges = np.diff(offsets)
    n_target = np.bincount(atom, weights=target_ok, minlength=len(atoms)).astype(np.int64)
    with open(out_dir / "atoms.tsv", "w") as handle:
        handle.write("\t".join(("atom_id", *ATOM_FIELDS, "pair_count", "target_pair_count")) + "\n")
        for atom_id, key in enumerate(atoms):
            handle.write("\t".join((str(atom_id), *key, str(n_edges[atom_id]), str(n_target[atom_id]))) + "\n")

    meta = {
        "ingest_dir": str(ingest_dir.resolve()),
        "config": config,
        "records_in": int(len(types)),
        "records_excluded_hop_predicate": int((~hop).sum()),
        "records_self_loop": int((hop & self_loop).sum()),
        "atom_count": len(atoms),
        "pairs": int(len(src)),
        "target_pairs": int(target_ok.sum()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
