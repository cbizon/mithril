from biolink_rules.lattice import (
    ANY_CLASS,
    class_lattice,
    flip,
    format_hop_label,
    hop_lattice,
    qualifier_generalizations,
)


def node_with_label(lattice, label):
    (node,) = [i for i, labels in enumerate(lattice.labels) if label in labels]
    return node


def assert_children_are_proper_subsets(lattice):
    for parent, kids in enumerate(lattice.children):
        for kid in kids:
            assert lattice.sets[kid] != lattice.sets[parent]
            assert lattice.sets[kid] & ~lattice.sets[parent] == 0


def test_class_lattice_collapses_identical_sets_and_has_any_root():
    catsets = [
        {"biolink:NamedThing", "biolink:ChemicalEntity", "biolink:SmallMolecule"},
        {"biolink:NamedThing", "biolink:Disease", "biolink:DiseaseOrPhenotypicFeature"},
        {"biolink:GenomicEntity"},  # a node without NamedThing, as in 1.0.2
    ]
    lattice = class_lattice(catsets)
    assert lattice.labels[lattice.root] == [ANY_CLASS]
    # ChemicalEntity and SmallMolecule cover the same nodes here -> one node
    assert node_with_label(lattice, "biolink:ChemicalEntity") == node_with_label(lattice, "biolink:SmallMolecule")
    named = node_with_label(lattice, "biolink:NamedThing")
    assert set(lattice.children[lattice.root]) == {named, node_with_label(lattice, "biolink:GenomicEntity")}
    assert_children_are_proper_subsets(lattice)


def test_flip_swaps_orientation_bits():
    assert flip(0b01, 1) == 0b10
    assert flip(0b1001, 2) == 0b0110


def test_qualifier_generalizations_include_ancestors_and_absence():
    gens = qualifier_generalizations({
        "predicate": "biolink:affects",
        "qualified_predicate": "biolink:causes",
        "object_aspect_qualifier": "expression",
    })
    assert () in gens
    assert (("object_aspect_qualifier", "abundance"),) in gens
    assert (("object_aspect_qualifier", "activity_or_abundance"), ("qualified_predicate", "biolink:causes")) in gens
    assert (("object_aspect_qualifier", "activity"),) not in gens
    # qualified_predicate is exact-valued: no ancestors such as contributes_to / related_to
    assert not any(v != "biolink:causes" for c in gens for f, v in c if f == "qualified_predicate")


def test_qualifier_generalizations_ignore_unlisted_fields():
    gens = qualifier_generalizations({"predicate": "biolink:affects", "causal_mechanism_qualifier": "inhibition"})
    assert gens == {()}


def test_exact_valued_label_still_matches_descendants():
    atoms = [
        {"predicate": "biolink:associated_with", "qualified_predicate": "biolink:causes"},
        {"predicate": "biolink:associated_with", "qualified_predicate": "biolink:contributes_to"},
    ]
    lattice = hop_lattice(atoms)
    label = format_hop_label("biolink:associated_with", (("qualified_predicate", "biolink:contributes_to"),), "F")
    node = node_with_label(lattice, label)
    # production semantics: contributes_to also matches causes (forward and, symmetric, reverse)
    assert lattice.sets[node] == 0b1111
    assert not any("qp=related_to" in l for labels in lattice.labels for l in labels)


def test_hop_lattice_structure():
    atoms = [
        {"predicate": "biolink:treats"},
        {"predicate": "biolink:in_clinical_trials_for"},
        {"predicate": "biolink:affects", "object_direction_qualifier": "increased", "object_aspect_qualifier": "expression"},
        {"predicate": "biolink:affects", "object_direction_qualifier": "decreased", "object_aspect_qualifier": "activity"},
    ]
    lattice = hop_lattice(atoms)
    assert lattice.sets[lattice.root] == (1 << 8) - 1  # related_to: every atom, both ways
    assert_children_are_proper_subsets(lattice)

    treats_f = node_with_label(lattice, format_hop_label("biolink:treats", (), "F"))
    assert lattice.sets[treats_f] == 1 << 0
    broad = node_with_label(lattice, format_hop_label("biolink:treats_or_applied_or_studied_to_treat", (), "F"))
    assert lattice.sets[broad] == (1 << 0) | (1 << 2)
    affects_up = node_with_label(
        lattice, format_hop_label("biolink:affects", (("object_direction_qualifier", "increased"),), "R")
    )
    assert lattice.sets[affects_up] == 1 << (2 * 2 + 1)
