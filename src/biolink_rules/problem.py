"""Load everything needed to mine rules for one target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import graphblas as gb

from biolink_rules.build import load_config
from biolink_rules.lattice import (
    DEFAULT_EXACT_VALUE_FIELDS,
    DEFAULT_QUERY_QUALIFIER_FIELDS,
    Lattice,
    class_lattice,
    hop_lattice,
    read_atoms,
    read_category_sets,
)
from biolink_rules.search import HopContexts, Store, restrict


@dataclass
class Problem:
    store: Store
    hops: Lattice
    classes: Lattice
    contexts: HopContexts
    subject_class: int
    object_class: int
    target: gb.Matrix
    target_label: str
    target_bits: int  # atom-orientations matched by the target hop label


def find(lattice: Lattice, label: str) -> int:
    matches = [i for i, labels in enumerate(lattice.labels) if label in labels]
    if len(matches) != 1:
        raise ValueError(f"{label!r} matches {len(matches)} lattice nodes")
    return matches[0]


def load_problem(data_dir: Path, config_path: Path, target: str, subject_class: str, object_class: str) -> Problem:
    """``target`` is a hop label such as ``'treats>'``; classes are Biolink CURIEs."""
    config = load_config(config_path)
    qualifiers = config.get("query_qualifiers") or {}
    excluded = set(config.get("middle_class_exclusions") or ())
    for endpoint in (subject_class, object_class):
        if endpoint in excluded:
            raise ValueError(f"{endpoint} is a target endpoint but listed in middle_class_exclusions")
    classes = class_lattice(read_category_sets(data_dir / "ingest/category_sets.tsv"), exclude=excluded)
    hops = hop_lattice(
        read_atoms(data_dir / "build/atoms.tsv"),
        fields=tuple(qualifiers.get("fields", DEFAULT_QUERY_QUALIFIER_FIELDS)),
        exact_value_fields=tuple(qualifiers.get("exact_value_fields", DEFAULT_EXACT_VALUE_FIELDS)),
        ignored_fields={k: tuple(v) for k, v in (qualifiers.get("ignored_fields") or {}).items()},
    )
    store = Store.load(data_dir / "build", data_dir / "ingest")
    a, b = find(classes, subject_class), find(classes, object_class)
    target_bits = hops.sets[find(hops, target)]
    target_matrix = restrict(
        store.hop_matrix(target_bits, target_only=True),
        rows=store.class_vector(classes.sets[a]),
        cols=store.class_vector(classes.sets[b]),
    )
    return Problem(store, hops, classes, HopContexts(store, hops, classes), a, b, target_matrix,
                   f"{subject_class.removeprefix('biolink:')} {target} {object_class.removeprefix('biolink:')}",
                   target_bits)
