"""Query semantics against the real Biolink model (via Gandalf's expanders)."""

import pytest

from biolink_rules.semantics import match_orientations, matched_atoms


def atom(predicate, **qualifiers):
    return {"predicate": predicate, **qualifiers}


@pytest.mark.parametrize(
    "query_predicate, stored, expected",
    [
        ("biolink:treats_or_applied_or_studied_to_treat", "biolink:treats", ("F",)),
        ("biolink:treats", "biolink:treats_or_applied_or_studied_to_treat", ()),
        ("biolink:interacts_with", "biolink:directly_physically_interacts_with", ("F", "R")),
        # production quirk: non-canonical part_of is not found by an overlaps query
        ("biolink:overlaps", "biolink:part_of", ()),
        ("biolink:part_of", "biolink:has_part", ("R",)),
        ("biolink:related_to", "biolink:treats", ("F", "R")),
    ],
)
def test_predicate_orientations(query_predicate, stored, expected):
    assert match_orientations((query_predicate, ()), atom(stored)) == expected


def test_qualifier_constraints_expand_downward():
    stored = atom(
        "biolink:affects",
        qualified_predicate="biolink:causes",
        object_direction_qualifier="increased",
        object_aspect_qualifier="expression",
    )
    general = ("biolink:affects", (("object_aspect_qualifier", "activity_or_abundance"),))
    via_parent_predicate = ("biolink:affects", (("qualified_predicate", "biolink:contributes_to"),))
    wrong_direction = ("biolink:affects", (("object_direction_qualifier", "decreased"),))
    assert match_orientations(general, stored) == ("F",)
    assert match_orientations(via_parent_predicate, stored) == ("F",)
    assert match_orientations(wrong_direction, stored) == ()
    # a qualified query never matches an unqualified edge
    assert match_orientations(general, atom("biolink:affects")) == ()


def test_matched_atoms_lists_index_and_orientation():
    atoms = [atom("biolink:treats"), atom("biolink:coexists_with"), atom("biolink:causes")]
    assert matched_atoms(("biolink:treats_or_applied_or_studied_to_treat", ()), atoms) == [(0, "F")]
    assert matched_atoms(("biolink:coexists_with", ()), atoms) == [(1, "F"), (1, "R")]
