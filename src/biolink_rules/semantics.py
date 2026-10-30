"""Production query semantics: which stored atoms a query predicate matches.

Delegates to Gandalf's ``PredicateExpander`` / ``QualifierExpander`` so that rule
counts agree with what Translator query engines return, quirks included (e.g. an
``overlaps`` query does not match stored non-canonical ``part_of`` edges).

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
from gandalf.search.expanders import PredicateExpander, QualifierExpander

QueryPredicate = tuple[str, tuple[tuple[str, str], ...]]


@cache
def toolkit() -> Toolkit:
    return Toolkit()


@cache
def predicate_expander() -> PredicateExpander:
    return PredicateExpander(toolkit())


@cache
def qualifier_expander() -> QualifierExpander:
    return QualifierExpander(toolkit())


@cache
def expanded_predicates(predicate: str) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """(forward, reverse) stored-predicate sets; ``None`` means "any predicate"."""
    forward, inverse = predicate_expander().expand_predicates([predicate])
    if forward == [] and inverse == []:
        return None, None
    return frozenset(forward), frozenset(inverse or ())


@cache
def expanded_qualifier_values(field: str, value: str) -> frozenset[str]:
    """All stored qualifier values matched by a constraint ``field = value``."""
    expander = qualifier_expander()
    if field == "qualified_predicate":
        return frozenset(expander.get_predicate_value_descendants(value))
    return frozenset(expander.get_value_descendants(value))


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
