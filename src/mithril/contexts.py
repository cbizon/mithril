"""Hop lattices within a (left class, right class) context."""

from __future__ import annotations

import numpy as np

from mithril.lattice import Lattice, hasse
from mithril.store import Store


class HopContexts:
    """Hop lattices collapsed within a (left class, right class) context.

    Two hop sets that differ only in atoms with no edge between the context classes
    give identical matrices once restricted to those classes, so they are one node.
    Hop coordinates in a search body are these *effective* bitsets.

    Near-duplicates: a node one of whose children holds at least ``near_duplicate`` of
    its edges within the context is not a refinement target (refining it would repeat
    the child's subtree); ``children`` steps over it to its nearest kept descendants.
    Edges are counted per atom (a pair carried by two atoms counts twice). 1.0 = off.
    """

    def __init__(self, store: Store, hops: Lattice, classes: Lattice, near_duplicate: float = 1.0):
        self.hops, self.classes, self.near_duplicate = hops, classes, near_duplicate
        n_catsets = int(store.node_catset.max()) + 1
        atom = np.repeat(np.arange(len(store.offsets) - 1), np.diff(store.offsets))
        key, count = np.unique((atom.astype(np.int64) * n_catsets + store.node_catset[store.src]) * n_catsets
                               + store.node_catset[store.dst], return_counts=True)
        self._atom, self._left, self._right = key // n_catsets**2, key // n_catsets % n_catsets, key % n_catsets
        self._count = count
        self._member = [np.array([bits >> i & 1 for i in range(n_catsets)], dtype=bool) for bits in classes.sets]
        self._masks: dict[tuple[int, int], int] = {}
        self._lattices: dict[tuple[int, int, int], tuple[Lattice, dict[int, int]]] = {}
        self._kept_children: dict[tuple[int, int, int], list[list[int]]] = {}

    def mask(self, left: int, right: int) -> int:
        """Atom-orientation bits with at least one edge from ``left`` to ``right``."""
        if (left, right) not in self._masks:
            lm, rm = self._member[left], self._member[right]
            forward = np.unique(self._atom[lm[self._left] & rm[self._right]])
            reverse = np.unique(self._atom[rm[self._left] & lm[self._right]])
            bits = 0
            for a in forward:
                bits |= 1 << (2 * int(a))
            for a in reverse:
                bits |= 1 << (2 * int(a) + 1)
            self._masks[left, right] = bits
        return self._masks[left, right]

    def _build(self, m: int) -> tuple[Lattice, dict[int, int]]:
        labels: dict[int, list[str]] = {}
        for bits, names in zip(self.hops.sets, self.hops.labels):
            if bits & m:
                labels.setdefault(bits & m, []).extend(names)
        sets = list(labels)
        children, root = hasse(sets) if sets else ([], 0)
        lattice = Lattice(sets=sets, labels=[sorted(labels[b]) for b in sets], children=children, root=root)
        return lattice, {b: i for i, b in enumerate(sets)}

    def edge_counts(self, left: int, right: int) -> dict[int, int]:
        """Per atom-orientation bit: number of its edges from ``left`` to ``right``."""
        lm, rm = self._member[left], self._member[right]
        counts: dict[int, int] = {}
        for bit, sel in ((0, lm[self._left] & rm[self._right]), (1, rm[self._left] & lm[self._right])):
            for a, n in zip(self._atom[sel].tolist(), self._count[sel].tolist()):
                counts[2 * a + bit] = counts.get(2 * a + bit, 0) + n
        return counts

    def kept_children(self, left: int, right: int, exclude: int = 0) -> list[list[int]]:
        """Children per node of the context lattice with near-duplicate nodes stepped over."""
        key = (left, right, exclude)
        if key not in self._kept_children:
            lattice, _ = self.lattice(left, right, exclude)
            counts = self.edge_counts(left, right)

            def weight(bits):
                return sum(counts.get(b, 0) for b in set_bits(bits))

            weights = [weight(b) for b in lattice.sets]
            kept = [i == lattice.root or not lattice.children[i]
                    or max(weights[c] for c in lattice.children[i]) < self.near_duplicate * weights[i]
                    for i in range(len(lattice.sets))]

            def reach(i, out):
                for c in lattice.children[i]:
                    if kept[c]:
                        out.add(c)
                    else:
                        reach(c, out)
                return out

            result = []
            for i in range(len(lattice.sets)):
                below = reach(i, set())
                sets = lattice.sets
                # only maximal ones: drop any that is a subset of another reached node
                result.append(sorted(c for c in below
                                     if not any(d != c and sets[c] & ~sets[d] == 0 for d in below)))
            self._kept_children[key] = result
        return self._kept_children[key]

    def lattice(self, left: int, right: int, exclude: int = 0) -> tuple[Lattice, dict[int, int]]:
        """The context lattice (optionally without the atom-orientations in ``exclude``)
        and a bits -> node index map."""
        if (left, right, exclude) not in self._lattices:
            self._lattices[left, right, exclude] = self._build(self.mask(left, right) & ~exclude)
        return self._lattices[left, right, exclude]

    def children(self, bits: int, left: int, right: int, exclude: int = 0) -> list[int]:
        lattice, index = self.lattice(left, right, exclude)
        children = (lattice.children if self.near_duplicate >= 1.0
                    else self.kept_children(left, right, exclude))
        return [lattice.sets[c] for c in children[index[bits]]]

    def label(self, bits: int, left: int, right: int, exclude: int = 0) -> str:
        lattice, index = self.lattice(left, right, exclude)
        return lattice.label(index[bits])


def set_bits(bits: int):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low
