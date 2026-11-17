"""Production query semantics: which stored atoms a query predicate matches.

Follows the reasoner-transpiler rules that Translator query engines use (the same
rules as Gandalf's ``PredicateExpander`` / ``QualifierExpander``), so that rule counts
agree with what those engines return, quirks included (e.g. an ``overlaps`` query does
not match stored non-canonical ``part_of`` edges):

- ``biolink:related_to`` matches any predicate in either orientation.
- A predicate matches itself and those of its descendants that are canonical or
  symmetric, on edges stored in the query's orientation.
- On edges stored in the opposite orientation it matches its inverse (and itself, if
  symmetric), and their canonical-or-symmetric descendants.
- A qualifier value matches itself and its descendants in every Biolink enum that has
  it; a ``qualified_predicate`` value matches itself and its descendant predicates.

A *query predicate* is a predicate CURIE plus qualifier constraints, given as a
tuple of (qualifier field, value) pairs, e.g.
``("biolink:affects", (("qualified_predicate", "biolink:causes"),
("object_direction_qualifier", "increased")))``.
Orientation ``"F"`` means the stored subject is the query subject; ``"R"`` means the
stored edge is matched in reverse.
"""

from __future__ import annotations

from functools import cache

from bmt import Toolkit

QueryPredicate = tuple[str, tuple[tuple[str, str], ...]]


@cache
def toolkit() -> Toolkit:
    return Toolkit()


@cache
def canonical_or_symmetric(predicate: str) -> bool:
    annotations = toolkit().get_element(predicate).annotations or {}
    return bool(annotations.get("canonical_predicate", False)) or toolkit().is_symmetric(predicate)


def with_descendants(predicates: list[str]) -> frozenset[str]:
    """The predicates plus their canonical-or-symmetric descendants."""
    found = set(predicates)
    for predicate in predicates:
        found.update(d for d in toolkit().get_descendants(predicate, formatted=True) if canonical_or_symmetric(d))
    return frozenset(found)


@cache
def expanded_predicates(predicate: str) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """(forward, reverse) stored-predicate sets; ``None`` means "any predicate"."""
    if predicate == "biolink:related_to":
        return None, None
    tk = toolkit()
    inverse = [tk.get_inverse_predicate(predicate, formatted=True)] if tk.has_inverse(predicate) else []
    if tk.is_symmetric(predicate):
        inverse.append(predicate)
    return with_descendants([predicate]), with_descendants(inverse)


@cache
def expanded_qualifier_values(field: str, value: str) -> frozenset[str]:
    """All stored qualifier values matched by a constraint ``field = value``."""
    tk = toolkit()
    if field == "qualified_predicate":
        return frozenset([value, *tk.get_descendants(value, formatted=True)])
    found = {value}
    for enum in tk.view.all_enums():
        if tk.is_permissible_value_of_enum(enum_name=enum, value=value):
            found.update(tk.get_permissible_value_descendants(permissible_value=value, enum_name=enum))
    return frozenset(found)


def qualifiers_match(query: QueryPredicate, atom: dict[str, str]) -> bool:
    """True if the atom satisfies every qualifier constraint of the query."""
    return all(atom.get(field, "") in expanded_qualifier_values(field, value) for field, value in query[1])


def match_orientations(query: QueryPredicate, atom: dict[str, str]) -> tuple[str, ...]:
    """Orientations ("F", "R") in which a stored atom matches the query predicate."""
    if not qualifiers_match(query, atom):
        return ()
    forward, reverse = expanded_predicates(query[0])
    stored = atom["predicate"]
    if forward is None:
        return ("F", "R")
    return tuple(o for o, preds in (("F", forward), ("R", reverse)) if stored in preds)


def matched_atoms(query: QueryPredicate, atoms: list[dict[str, str]]) -> list[tuple[int, str]]:
    """All (atom index, orientation) pairs the query predicate matches."""
    return [(i, o) for i, atom in enumerate(atoms) for o in match_orientations(query, atom)]
