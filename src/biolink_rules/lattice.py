"""Refinement lattices for the search, built by set inclusion over the data.

Hop lattice: a *hop label* is (query predicate, qualifier constraints, direction). Its
meaning in this graph is the set of (atom, orientation) pairs it matches under
production semantics (``semantics``). Labels with identical match sets collapse into
one node; children are maximal proper subsets. The root is ``related_to`` (all atoms,
both orientations).

Class lattice: a class's meaning is the set of category-sets (hence nodes) carrying
it. Same collapse/children construction; the root is ``NamedThing``.

Sets are Python ints used as bitmasks. Hop bit ``2*atom + 0`` is the atom matched
forward (stored subject on the left of the hop), ``2*atom + 1`` reversed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import cache
from itertools import product
from pathlib import Path

from biolink_rules.build import ATOM_QUALIFIER_FIELDS
from biolink_rules.semantics import expanded_predicates, qualifiers_match, toolkit


@dataclass
class Lattice:
    """Distinct non-empty sets, their labels, and Hasse-diagram children."""

    sets: list[int]
    labels: list[list[str]]
    children: list[list[int]] = field(default_factory=list)
    root: int = 0

    def label(self, node: int) -> str:
        """Shortest label as the display name."""
        return min(self.labels[node], key=lambda s: (len(s), s))


def hasse(sets: list[int]) -> tuple[list[list[int]], int]:
    """Children (maximal proper subsets) for each set; returns (children, root index).

    >>> hasse([0b111, 0b011, 0b001, 0b100])
    ([[1, 3], [2], [], []], 0)
    """
    order = sorted(range(len(sets)), key=lambda i: -sets[i].bit_count())
    children: list[list[int]] = [[] for _ in sets]
    for pos, i in enumerate(order):
        below = [j for j in order[pos + 1:] if sets[j] != sets[i] and sets[j] & ~sets[i] == 0]
        maximal = []
        for j in below:  # 'below' is size-descending, so supersets come first
            if not any(sets[j] & ~sets[k] == 0 for k in maximal):
                maximal.append(j)
        children[i] = sorted(maximal)
    roots = [i for i in range(len(sets)) if not any(i in c for c in children)]
    if len(roots) != 1:
        raise ValueError(f"Lattice has {len(roots)} roots; expected 1")
    return children, roots[0]


def collapse(labelled: dict[str, int]) -> Lattice:
    """Group labels by set, drop the empty set, and build the Hasse diagram."""
    by_set: dict[int, list[str]] = {}
    for name, bits in labelled.items():
        if bits:
            by_set.setdefault(bits, []).append(name)
    sets = list(by_set)
    children, root = hasse(sets)
    return Lattice(sets=sets, labels=[sorted(by_set[s]) for s in sets], children=children, root=root)


# --------------------------------------------------------------------------- classes

ANY_CLASS = "(any)"


def read_category_sets(path: Path) -> list[set[str]]:
    with open(path, newline="") as handle:
        return [set(row["categories"].split("|")) for row in csv.DictReader(handle, delimiter="\t")]


def class_lattice(category_sets: list[set[str]], exclude: set[str] = frozenset()) -> Lattice:
    """Lattice over classes present in the data; a class's set is its catset ids.

    The root is ``ANY_CLASS`` (an unconstrained query node). It collapses with
    ``NamedThing`` unless some nodes lack ``NamedThing`` (1.0.2 has 75 typed only as
    the mixin ``GenomicEntity``).
    """
    labelled: dict[str, int] = {ANY_CLASS: (1 << len(category_sets)) - 1}
    for catset_id, categories in enumerate(category_sets):
        for category in categories - exclude:
            labelled[category] = labelled.get(category, 0) | (1 << catset_id)
    return collapse(labelled)


# --------------------------------------------------------------------------- hops


def read_atoms(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@cache
def enum_value_ancestors(value: str) -> frozenset[str]:
    """The value plus its ancestors in every Biolink enum that contains it."""
    tk = toolkit()
    result = {value}
    for enum_name in tk.view.all_enums():
        if tk.is_permissible_value_of_enum(enum_name=enum_name, value=value):
            result.update(tk.get_permissible_value_ancestors(permissible_value=value, enum_name=enum_name) or ())
    return frozenset(result)


@cache
def predicate_ancestors(predicate: str) -> frozenset[str]:
    return frozenset(toolkit().get_ancestors(predicate, formatted=True, mixin=True, reflexive=True))


def qualifier_generalizations(atom: dict[str, str]) -> set[tuple[tuple[str, str], ...]]:
    """Every constraint set (sorted (field, value) tuple) that the atom satisfies."""
    options = []
    for name in ATOM_QUALIFIER_FIELDS:
        value = atom.get(name, "")
        if not value:
            continue
        values = predicate_ancestors(value) if name == "qualified_predicate" else enum_value_ancestors(value)
        options.append([None, *((name, v) for v in sorted(values))])
    return {tuple(sorted(c for c in combo if c)) for combo in product(*options)}


def format_hop_label(predicate: str, constraints: tuple[tuple[str, str], ...], direction: str) -> str:
    """e.g. ``affects[object_direction_qualifier=increased]>`` (``>`` F, ``<`` R)."""
    name = predicate.removeprefix("biolink:")
    if constraints:
        name += "[" + ",".join(f"{f}={v.removeprefix('biolink:')}" for f, v in constraints) + "]"
    return name + (">" if direction == "F" else "<")


def flip(bits: int, n_atoms: int) -> int:
    """Swap forward/reverse bits of every atom."""
    even = sum(1 << (2 * i) for i in range(n_atoms))
    return ((bits & even) << 1) | ((bits >> 1) & even)


def hop_lattice(atoms: list[dict[str, str]]) -> Lattice:
    """Lattice over all hop labels with a non-empty match set in this graph."""
    gens = [qualifier_generalizations(a) for a in atoms]
    query_predicates = set().union(*(predicate_ancestors(a["predicate"]) for a in atoms))

    labelled: dict[str, int] = {}
    for predicate in sorted(query_predicates):
        forward, reverse = expanded_predicates(predicate)
        f_atoms = [i for i, a in enumerate(atoms) if forward is None or a["predicate"] in forward]
        r_atoms = [i for i, a in enumerate(atoms) if forward is None or a["predicate"] in reverse]
        constraint_sets = set().union(*(gens[i] for i in {*f_atoms, *r_atoms}))
        for constraints in constraint_sets:
            bits = 0
            for i in f_atoms:
                if constraints in gens[i]:
                    bits |= 1 << (2 * i)
            for i in r_atoms:
                if constraints in gens[i]:
                    bits |= 1 << (2 * i + 1)
            labelled[format_hop_label(predicate, constraints, "F")] = bits
            labelled[format_hop_label(predicate, constraints, "R")] = flip(bits, len(atoms))
    return collapse(labelled)


def check_hop_semantics(atoms: list[dict[str, str]], sample: list[tuple[str, tuple]]) -> None:
    """Cross-check generated constraint sets against ``semantics.qualifiers_match``."""
    for predicate, constraints in sample:
        for atom, gen in zip(atoms, (qualifier_generalizations(a) for a in atoms)):
            assert (constraints in gen) == qualifiers_match((predicate, constraints), atom), (constraints, atom)
