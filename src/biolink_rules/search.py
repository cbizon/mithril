"""Target-centric top-down rule search (1, 2 and 3 hops).

A rule body for a target (A -T-> B) is a chain A -h1- c1 -h2- B (or A -h1- c1 -h2- c2
-h3- B) where each ``h`` is a hop-lattice node and each intermediate ``c`` a
class-lattice node. The search starts at
the most general body and specializes one coordinate at a time. Overlap with the target
pair set is anti-monotone under specialization, so bodies below ``min_support`` are
pruned together with all their specializations. Each child's overlap is computed with
its parent's overlap pairs as the mask.

Every evaluated body is logged (including pruned ones), so the frontier for any higher
support threshold can be recomputed from one run.
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import graphblas as gb
import numpy as np

from biolink_rules.lattice import Lattice, hasse


@dataclass
class Store:
    """Built atom edge store plus node class membership."""

    offsets: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    target_ok: np.ndarray
    node_catset: np.ndarray

    @classmethod
    def load(cls, build_dir: Path, ingest_dir: Path) -> "Store":
        return cls(
            offsets=np.load(build_dir / "atom_offsets.npy"),
            src=np.load(build_dir / "edge_src.npy"),
            dst=np.load(build_dir / "edge_dst.npy"),
            target_ok=np.load(build_dir / "edge_target_ok.npy"),
            node_catset=np.load(ingest_dir / "node_catset.npy"),
        )

    @property
    def n_nodes(self) -> int:
        return len(self.node_catset)

    def hop_matrix(self, bits: int, target_only: bool = False) -> gb.Matrix:
        """Boolean N x N matrix of the (atom, orientation) set ``bits``."""
        rows, cols = [], []
        for bit in range(bits.bit_length()):
            if not bits >> bit & 1:
                continue
            atom, reverse = divmod(bit, 2)
            lo, hi = self.offsets[atom], self.offsets[atom + 1]
            s, d = self.src[lo:hi], self.dst[lo:hi]
            if target_only:
                ok = self.target_ok[lo:hi]
                s, d = s[ok], d[ok]
            rows.append(d if reverse else s)
            cols.append(s if reverse else d)
        rows = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
        return gb.Matrix.from_coo(rows, cols, True, nrows=self.n_nodes, ncols=self.n_nodes)

    def class_vector(self, catset_bits: int) -> gb.Vector:
        member = np.array([catset_bits >> int(c) & 1 for c in range(int(self.node_catset.max()) + 1)], dtype=bool)
        idx = np.flatnonzero(member[self.node_catset])
        return gb.Vector.from_coo(idx, True, size=self.n_nodes)


class Cache:
    """LRU cache bounded by an estimated memory budget (``size`` -> bytes of a value)."""

    def __init__(self, build, budget_bytes: int, size=lambda m: 16 * m.nvals + 1024):
        self.build, self.budget, self.size = build, budget_bytes, size
        self.data: OrderedDict = OrderedDict()
        self.used = 0

    def __call__(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key][0]
        value = self.build(key)
        nbytes = self.size(value)
        self.data[key] = (value, nbytes)
        self.used += nbytes
        while self.used > self.budget and len(self.data) > 1:
            _, (_, evicted) = self.data.popitem(last=False)
            self.used -= evicted
        return value


@dataclass
class SearchStats:
    evaluated: int = 0
    passed: int = 0
    seconds: float = 0.0
    complete: bool = True
    rows: list[tuple] = field(default_factory=list)


def restrict(matrix: gb.Matrix, rows: gb.Vector | None = None, cols: gb.Vector | None = None) -> gb.Matrix:
    """Keep only entries whose row (col) is in the given node vector."""
    if rows is not None:
        matrix = rows.diag().mxm(matrix, gb.semiring.any_pair).new()
    if cols is not None:
        matrix = matrix.mxm(cols.diag(), gb.semiring.any_pair).new()
    return matrix


class HopContexts:
    """Hop lattices collapsed within a (left class, right class) context.

    Two hop sets that differ only in atoms with no edge between the context classes
    give identical matrices once restricted to those classes, so they are one node.
    Hop coordinates in a search body are these *effective* bitsets.
    """

    def __init__(self, store: Store, hops: Lattice, classes: Lattice):
        self.hops, self.classes = hops, classes
        n_catsets = int(store.node_catset.max()) + 1
        atom = np.repeat(np.arange(len(store.offsets) - 1), np.diff(store.offsets))
        key = np.unique((atom.astype(np.int64) * n_catsets + store.node_catset[store.src]) * n_catsets
                        + store.node_catset[store.dst])
        self._atom, self._left, self._right = key // n_catsets**2, key // n_catsets % n_catsets, key % n_catsets
        self._member = [np.array([bits >> i & 1 for i in range(n_catsets)], dtype=bool) for bits in classes.sets]
        self._masks: dict[tuple[int, int], int] = {}
        self._lattices: dict[tuple[int, int, int], tuple[Lattice, dict[int, int]]] = {}

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

    def lattice(self, left: int, right: int, exclude: int = 0) -> tuple[Lattice, dict[int, int]]:
        """The context lattice (optionally without the atom-orientations in ``exclude``)
        and a bits -> node index map."""
        if (left, right, exclude) not in self._lattices:
            self._lattices[left, right, exclude] = self._build(self.mask(left, right) & ~exclude)
        return self._lattices[left, right, exclude]

    def endpoint_tables(self, subject: int, object_: int) -> tuple[dict[int, int], dict[int, int]]:
        """Per atom-orientation bit: catsets reachable from ``subject`` / reaching ``object_``.

        ``from_subject[bit]`` is the catset bitmask of right-hand ends of the bit's edges
        whose left end is in the subject class; ``to_object[bit]`` the left-hand ends of
        edges whose right end is in the object class.
        """
        sm, om = self._member[subject], self._member[object_]
        from_subject: dict[int, int] = {}
        to_object: dict[int, int] = {}
        for atom, left, right in zip(self._atom.tolist(), self._left.tolist(), self._right.tolist()):
            for bit, l, r in ((2 * atom, left, right), (2 * atom + 1, right, left)):
                if sm[l]:
                    from_subject[bit] = from_subject.get(bit, 0) | (1 << r)
                if om[r]:
                    to_object[bit] = to_object.get(bit, 0) | (1 << l)
        return from_subject, to_object

    def pair_table(self) -> dict[int, dict[int, int]]:
        """Per atom-orientation bit: left catset -> bitmask of right catsets it links to."""
        if not hasattr(self, "_pairs"):
            table: dict[int, dict[int, int]] = {}
            for atom, left, right in zip(self._atom.tolist(), self._left.tolist(), self._right.tolist()):
                for bit, l, r in ((2 * atom, left, right), (2 * atom + 1, right, left)):
                    row = table.setdefault(bit, {})
                    row[l] = row.get(l, 0) | (1 << r)
            self._pairs = table
        return self._pairs

    def children(self, bits: int, left: int, right: int, exclude: int = 0) -> list[int]:
        lattice, index = self.lattice(left, right, exclude)
        return [lattice.sets[c] for c in lattice.children[index[bits]]]

    def label(self, bits: int, left: int, right: int, exclude: int = 0) -> str:
        lattice, index = self.lattice(left, right, exclude)
        return lattice.label(index[bits])


def set_bits(bits: int):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def canonical_two_hop(h1: int, c_set: int, h2: int, from_subject: dict[int, int], to_object: dict[int, int]):
    """Canonical key of a 2-hop body: only middle catsets with a first-hop edge from the
    subject class AND a second-hop edge to the object class matter, and only hop atoms
    touching those catsets. Iterated to a fixpoint. Bodies with equal keys have
    identical predicted pair sets.
    """
    while True:
        reach = 0
        for bit in set_bits(h1):
            reach |= from_subject.get(bit, 0)
        back = 0
        for bit in set_bits(h2):
            back |= to_object.get(bit, 0)
        middle = c_set & reach & back
        n1 = sum(1 << bit for bit in set_bits(h1) if from_subject.get(bit, 0) & middle)
        n2 = sum(1 << bit for bit in set_bits(h2) if to_object.get(bit, 0) & middle)
        if (n1, middle, n2) == (h1, c_set, h2):
            return h1, c_set, h2
        h1, c_set, h2 = n1, middle, n2


def canonical_three_hop(h1, c1_set, h2, c2_set, h3, from_subject, to_object, pairs):
    """Canonical key of a 3-hop body (see ``canonical_two_hop``): middle catsets must lie
    on a subject -> c1 -> c2 -> object chain at catset level; hops keep only atoms whose
    edges touch the surviving catsets. Iterated to a fixpoint."""
    while True:
        reach = 0
        for bit in set_bits(h1):
            reach |= from_subject.get(bit, 0)
        back = 0
        for bit in set_bits(h3):
            back |= to_object.get(bit, 0)
        m1_cand, m2_cand = c1_set & reach, c2_set & back
        m1 = 0
        for bit in set_bits(h2):
            for left, rights in pairs.get(bit, {}).items():
                if m1_cand >> left & 1 and rights & m2_cand:
                    m1 |= 1 << left
        m2 = 0
        for bit in set_bits(h2):
            for left, rights in pairs.get(bit, {}).items():
                if m1 >> left & 1:
                    m2 |= rights & m2_cand
        n1 = sum(1 << bit for bit in set_bits(h1) if from_subject.get(bit, 0) & m1)
        n3 = sum(1 << bit for bit in set_bits(h3) if to_object.get(bit, 0) & m2)
        n2 = sum(1 << bit for bit in set_bits(h2)
                 if any(m1 >> left & 1 and rights & m2 for left, rights in pairs.get(bit, {}).items()))
        if (n1, m1, n2, m2, n3) == (h1, c1_set, h2, c2_set, h3):
            return h1, c1_set, h2, c2_set, h3
        h1, c1_set, h2, c2_set, h3 = n1, m1, n2, m2, n3


def body_space(store: Store, hops: Lattice, classes: Lattice, subject_class: int, object_class: int,
               n_hops: int, cache_gb: float, contexts: HopContexts):
    """The specialization space for ``n_hops`` in (1, 2, 3): ``(root, children, overlap, name)``.

    A body is ``(h1,)``, ``(h1, c1, h2)`` or ``(h1, c1, h2, c2, h3)``: hops are effective
    bitsets in their class context (``HopContexts``), ``c`` are class-lattice indices.
    ``children(body)`` yields ``(coordinate, child)``; coordinates number the refinable
    positions classes first (3 hops: c1=0, c2=1, h1=2, h2=3, h3=4; 2 hops: c1=0, h1=1,
    h2=2), so refining in non-decreasing coordinate order reaches every body.
    ``overlap(body, mask)`` is the boolean matrix of ``mask`` pairs the body joins (simple
    paths for 3 hops).
    """
    assert n_hops in (1, 2, 3)
    if n_hops == 3:
        from biolink_rules.threehop import ThreeHop

        three = ThreeHop(store, classes.sets, subject_class, object_class, cache_gb)
    a, b = subject_class, object_class
    budget = int(cache_gb * 2**30)
    hop_m = Cache(store.hop_matrix, budget // 2)
    class_v = Cache(lambda c: store.class_vector(classes.sets[c]), 2**28)
    # second hop with its rows restricted to the intermediate class
    hop2_m = Cache(lambda key: restrict(hop_m(key[1]), rows=class_v(key[0])), budget // 2)

    def overlap(body, mask):
        result = gb.Matrix(bool, mask.nrows, mask.ncols)
        if n_hops == 1:
            result(mask.S) << hop_m(body[0])
        elif n_hops == 2:
            h1, c1, h2 = body
            result(mask.S) << hop_m(h1).mxm(hop2_m((c1, h2)), gb.semiring.any_pair)
        else:
            return three.overlap(body, mask)
        return result

    def children(body):
        if n_hops == 1:
            for h in contexts.children(body[0], a, b):
                yield 0, (h,)
            return
        if n_hops == 3:
            yield from children_3(body)
            return
        h1, c1, h2 = body
        for c in classes.children[c1]:
            n1, n2 = h1 & contexts.mask(a, c), h2 & contexts.mask(c, b)
            if n1 and n2:
                yield 0, (n1, c, n2)
        for h in contexts.children(h1, a, c1):
            yield 1, (h, c1, h2)
        for h in contexts.children(h2, c1, b):
            yield 2, (h1, c1, h)

    def children_3(body):
        h1, c1, h2, c2, h3 = body
        for c in classes.children[c1]:
            n1, n2 = h1 & contexts.mask(a, c), h2 & contexts.mask(c, c2)
            if n1 and n2:
                yield 0, (n1, c, n2, c2, h3)
        for c in classes.children[c2]:
            n2, n3 = h2 & contexts.mask(c1, c), h3 & contexts.mask(c, b)
            if n2 and n3:
                yield 1, (h1, c1, n2, c, n3)
        for h in contexts.children(h1, a, c1):
            yield 2, (h, c1, h2, c2, h3)
        for h in contexts.children(h2, c1, c2):
            yield 3, (h1, c1, h, c2, h3)
        for h in contexts.children(h3, c2, b):
            yield 4, (h1, c1, h2, c2, h)

    def name(body):
        if n_hops == 1:
            return contexts.label(body[0], a, b)
        if n_hops == 3:
            h1, c1, h2, c2, h3 = body
            return (f"{contexts.label(h1, a, c1)} | {classes.label(c1)} | {contexts.label(h2, c1, c2)} | "
                    f"{classes.label(c2)} | {contexts.label(h3, c2, b)}")
        h1, c1, h2 = body
        return f"{contexts.label(h1, a, c1)} | {classes.label(c1)} | {contexts.label(h2, c1, b)}"

    if n_hops == 1:
        root = (hops.sets[hops.root] & contexts.mask(a, b),)
    elif n_hops == 3:
        c, r = classes.root, hops.sets[hops.root]
        root = (r & contexts.mask(a, c), c, r & contexts.mask(c, c), c, r & contexts.mask(c, b))
    else:
        c = classes.root
        root = (hops.sets[hops.root] & contexts.mask(a, c), c, hops.sets[hops.root] & contexts.mask(c, b))
    return root, children, overlap, name


def search(
    store: Store,
    hops: Lattice,
    classes: Lattice,
    target: gb.Matrix,
    subject_class: int,
    object_class: int,
    n_hops: int,
    min_support: int,
    time_limit: float,
    cache_gb: float = 4.0,
    contexts: HopContexts | None = None,
    merge_equivalent: bool = True,
) -> SearchStats:
    """Depth-first specialization search over ``body_space``.

    With ``merge_equivalent``, 2- and 3-hop bodies with the same canonical key
    (``canonical_two_hop`` / ``canonical_three_hop``) are evaluated and reported once, labelled with the most
    specific middle class seen; every raw state is still explored so nothing reachable
    only through an equivalent class is lost.
    """
    contexts = contexts or HopContexts(store, hops, classes)
    root, children, overlap, name = body_space(store, hops, classes, subject_class, object_class, n_hops,
                                               cache_gb, contexts)
    from_subject, to_object = contexts.endpoint_tables(subject_class, object_class) if n_hops > 1 else ({}, {})
    pairs = contexts.pair_table() if n_hops == 3 else {}

    def key_of(body):
        if n_hops == 1 or not merge_equivalent:
            return body
        if n_hops == 3:
            h1, c1, h2, c2, h3 = body
            return canonical_three_hop(h1, classes.sets[c1], h2, classes.sets[c2], h3, from_subject, to_object, pairs)
        h1, c1, h2 = body
        return canonical_two_hop(h1, classes.sets[c1], h2, from_subject, to_object)

    def specificity(body):
        return sum(classes.sets[c].bit_count() for c in body[1::2])

    stats = SearchStats()
    start = time.time()
    root_overlap = overlap(root, target)
    stats.evaluated = 1
    stats.rows.append((root, name(root), "", root_overlap.nvals, 0))
    row_of = {key_of(root): 0}  # canonical key -> index in stats.rows
    if root_overlap.nvals < min_support:
        return stats
    stats.passed = 1
    explored = {root}
    stack = [(root, root_overlap, 0)]
    while stack and time.time() - start < time_limit:
        body, mask, depth = stack.pop()
        for _, child in children(body):
            if child in explored:
                continue
            explored.add(child)
            key = key_of(child)
            if key in row_of:
                row = row_of[key]
                support = stats.rows[row][3]
                if specificity(child) < specificity(stats.rows[row][0]):
                    _, _, parent_name, _, d = stats.rows[row]
                    stats.rows[row] = (child, name(child), parent_name, support, d)
                if support >= min_support:
                    stack.append((child, mask, depth + 1))  # parent's overlap is a valid superset mask
                continue
            child_overlap = overlap(child, mask)
            stats.evaluated += 1
            support = child_overlap.nvals
            row_of[key] = len(stats.rows)
            stats.rows.append((child, name(child), name(body), support, depth + 1))
            if support >= min_support:
                stats.passed += 1
                stack.append((child, child_overlap, depth + 1))
            if stats.evaluated % 1000 == 0:
                elapsed = time.time() - start
                print(f"[{elapsed:8.0f}s] evaluated {stats.evaluated:,} passed {stats.passed:,} "
                      f"stack {len(stack):,} explored {len(explored):,} ({stats.evaluated / elapsed:.1f}/s)",
                      file=sys.stderr, flush=True)
    stats.seconds = time.time() - start
    stats.complete = not stack
    return stats


def estimate_size(
    store: Store,
    hops: Lattice,
    classes: Lattice,
    target: gb.Matrix,
    subject_class: int,
    object_class: int,
    n_hops: int,
    min_support: int,
    probes: int,
    seed: int = 0,
    cache_gb: float = 4.0,
    contexts: HopContexts | None = None,
    time_limit: float = float("inf"),
    progress=None,
    bias: float = 0.0,
    mix: float = 0.1,
    memo_depth: int = 1,
) -> list[dict]:
    """Knuth random-probe estimate of the support-pruned search size, one dict per probe.

    Each probe walks from the root, refining in non-decreasing coordinate order (so a
    body is reached by about one route rather than one per refinement ordering), and
    picks a random passing child at each step. With ``w`` the product of the inverse
    selection probabilities so far, a step that evaluates ``k`` children of which ``p``
    pass adds ``w*k`` evaluated and ``w*p`` passing bodies at the next depth; the mean
    over probes is an unbiased estimate for the coordinate-ordered tree.

    ``bias=0`` picks children uniformly (Knuth's estimator). The tree is very unbalanced
    (huge subtrees under a few broad bodies), which makes uniform probes rarely reach
    the bulk of it and the estimate heavy-tailed. ``bias>0`` importance-samples: child
    ``i`` with target overlap ``s_i`` is picked with probability
    ``(1-mix) * s_i**bias / sum(s**bias) + mix / p``; still unbiased, since every child
    keeps a positive probability and is reweighted by its inverse.

    At bodies of depth <= ``memo_depth`` the expansion is cached and the selection is
    *adaptive*: each probe records the size of the subtree it entered (its own
    estimate below that child), and later probes pick children in proportion to the
    mean recorded size (unvisited children get the largest mean seen so far, which
    encourages exploring them), mixed with ``mix`` uniform. Probabilities depend only on
    earlier probes, so each probe stays unbiased given the history. That tree
    still repeats bodies reachable through several class or hop parents, so it
    overestimates the deduplicated search somewhat.

    Probe dicts: ``evaluated`` / ``passed`` (depth -> estimate), ``by_classes``
    (middle-class tuple -> estimated evaluations), ``seconds``, ``evaluations`` (actually done).
    No new probe starts after ``time_limit`` seconds; ``progress(probe)`` is called after each.
    """
    contexts = contexts or HopContexts(store, hops, classes)
    root, children, overlap, _ = body_space(store, hops, classes, subject_class, object_class, n_hops,
                                               cache_gb, contexts)
    rng = np.random.default_rng(seed)
    root_overlap = overlap(root, target)
    if root_overlap.nvals < min_support:
        return []
    # bodies at depth <= memo_depth: (#children, passing [(coord, body, overlap)], heuristic
    # probabilities, per-child [sum of recorded subtree sizes, count])
    top: dict[tuple, tuple[int, list, np.ndarray, np.ndarray]] = {}

    def learned(prob, record):
        seen = record[:, 1] > 0
        if not seen.any():
            return prob
        size = np.where(seen, record[:, 0] / np.maximum(record[:, 1], 1), (record[seen, 0] / record[seen, 1]).max())
        return (1 - mix) * size / size.sum() + mix / len(size)

    def expand(body, mask, last):
        kids = [(coord, child) for coord, child in children(body) if coord >= last]
        passing = []
        for coord, child in kids:
            child_overlap = overlap(child, mask)
            if child_overlap.nvals >= min_support:
                passing.append((coord, child, child_overlap))
        if not passing:
            return len(kids), passing, np.zeros(0)
        if bias == 0:
            return len(kids), passing, np.full(len(passing), 1 / len(passing))
        score = np.array([o.nvals for _, _, o in passing], dtype=float) ** bias
        return len(kids), passing, (1 - mix) * score / score.sum() + mix / len(passing)

    results = []
    deadline = time.time() + time_limit
    while len(results) < probes and time.time() < deadline:
        start = time.time()
        evaluated: dict[int, float] = {}
        passed: dict[int, float] = {}
        by_classes: dict[tuple, float] = {}
        body, mask, last, depth, weight, done = root, root_overlap, 0, 0, 1.0, 0
        steps = []  # (weight, children evaluated, memo record or None, picked index)
        while True:
            key = (body, last)
            record = None
            if depth <= memo_depth:
                if key not in top:
                    n_kids, passing, prob = expand(body, mask, last)
                    done += n_kids
                    top[key] = (n_kids, passing, prob, np.zeros((len(passing), 2)))
                n_kids, passing, prob, record = top[key]
                prob = learned(prob, record)
            else:
                n_kids, passing, prob = expand(body, mask, last)
                done += n_kids
            evaluated[depth + 1] = evaluated.get(depth + 1, 0.0) + weight * n_kids
            passed[depth + 1] = passed.get(depth + 1, 0.0) + weight * len(passing)
            middle = body[1::2]
            by_classes[middle] = by_classes.get(middle, 0.0) + weight * n_kids
            if not passing:
                steps.append((weight, n_kids, None, -1))
                break
            pick = rng.choice(len(passing), p=prob)
            steps.append((weight, n_kids, record, pick))
            last, body, mask = passing[pick]
            weight /= prob[pick]
            depth += 1
        # subtree size entered at step j = 1 + (estimated evaluations from step j+1 on) / weight there
        below = 0.0
        for j in range(len(steps) - 1, -1, -1):
            w, n_kids, record, pick = steps[j]
            if record is not None:
                child_weight = steps[j + 1][0]
                record[pick] += (1 + below / child_weight, 1)
            below += w * n_kids
        results.append({"evaluated": evaluated, "passed": passed, "by_classes": by_classes,
                        "seconds": time.time() - start, "evaluations": done})
        if progress:
            progress(results[-1])
    return results


def explore_ordered(children, overlap, body, last: int, mask: gb.Matrix, depth: int, min_support: int,
                    deadline: float = float("inf")):
    """Exhaustive search of the coordinate-ordered subtree below ``body`` (see ``body_space``).

    ``body`` has already been evaluated (``mask`` is its overlap) and refined last in
    coordinate ``last``. Children are refined in non-decreasing coordinate order, so
    disjoint subtrees can be searched independently. Expands at least one body, then
    stops starting new expansions once ``deadline`` has passed. Returns (passing rows ``(body, depth, overlap)``, number evaluated,
    unexpanded frontier ``[(body, last, overlap matrix, depth)]``; empty when complete).
    """
    stack = [(body, last, mask, depth)]
    seen = set()
    rows, evaluated = [], 0
    while stack:
        body, last, mask, depth = stack.pop()
        for coord, child in children(body):
            if coord < last or (child, coord) in seen:
                continue
            seen.add((child, coord))
            child_overlap = overlap(child, mask)
            evaluated += 1
            if child_overlap.nvals >= min_support:
                rows.append((child, depth + 1, child_overlap.nvals))
                stack.append((child, coord, child_overlap, depth + 1))
        if time.time() >= deadline:
            break
    return rows, evaluated, stack
