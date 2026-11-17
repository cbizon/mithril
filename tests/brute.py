"""Brute-force reference: predicted pair sets and the search tree, straight from the definitions."""

from collections import defaultdict


def pairs_of(store, bits):
    """Set of (left, right) node pairs for a hop set, by brute force."""
    out = set()
    for bit in range(bits.bit_length()):
        if bits >> bit & 1:
            atom, reverse = divmod(bit, 2)
            for k in range(store.offsets[atom], store.offsets[atom + 1]):
                s, d = int(store.src[k]), int(store.dst[k])
                out.add((d, s) if reverse else (s, d))
    return out


def members(store, classes, c):
    return {n for n, cs in enumerate(store.node_catset) if classes.sets[c] >> int(cs) & 1}


def predicted(store, classes, body, a_nodes, b_nodes):
    """Pairs (a, d), a != d, joined by ``body = (direct, path)``: a direct edge in the hop
    set ``direct`` (0: not required) and a simple path matching ``path`` (``()``: none)."""
    direct, path = body
    if not path:
        out = {(a, d) for a in a_nodes for d in b_nodes}
    elif len(path) == 3:
        h1, c1, h2 = path
        middle = members(store, classes, c1)
        second = defaultdict(set)
        for b, d in pairs_of(store, h2):
            second[b].add(d)
        out = {(a, d) for a, b in pairs_of(store, h1) if b in middle for d in second[b]}
    else:
        h1, c1, h2, c2, h3 = path
        m1, m2 = members(store, classes, c1), members(store, classes, c2)
        second, third = defaultdict(set), defaultdict(set)
        for b, c in pairs_of(store, h2):
            second[b].add(c)
        for c, d in pairs_of(store, h3):
            third[c].add(d)
        out = {(a, d) for a, b in pairs_of(store, h1) if b in m1 for c in second[b] if c in m2
               for d in third[c] if len({a, b, c, d}) == 4}
    if direct:
        out &= pairs_of(store, direct)
    return {(a, d) for a, d in out if a in a_nodes and d in b_nodes and a != d}


def ordered_tree(problem, n_hops, min_support, overlap_of):
    """The search tree by definition: from the most general path, specialize one position
    at a time in non-decreasing position order (classes, then hops), expanding a path
    only if ``overlap_of(path) >= min_support``. Returns ({passing path: overlap}, evaluated)."""
    classes, contexts, a, b = problem.classes, problem.contexts, problem.subject_class, problem.object_class
    top, c = problem.hops.sets[problem.hops.root], classes.root

    def children(path):
        if n_hops == 2:
            h1, c1, h2 = path
            for k in classes.children[c1]:
                n1, n2 = h1 & contexts.mask(a, k), h2 & contexts.mask(k, b)
                if n1 and n2:
                    yield 0, (n1, k, n2)
            for h in contexts.children(h1, a, c1):
                yield 1, (h, c1, h2)
            for h in contexts.children(h2, c1, b):
                yield 2, (h1, c1, h)
        elif n_hops == 3:
            h1, c1, h2, c2, h3 = path
            for k in classes.children[c1]:
                n1, n2 = h1 & contexts.mask(a, k), h2 & contexts.mask(k, c2)
                if n1 and n2:
                    yield 0, (n1, k, n2, c2, h3)
            for k in classes.children[c2]:
                n2, n3 = h2 & contexts.mask(c1, k), h3 & contexts.mask(k, b)
                if n2 and n3:
                    yield 1, (h1, c1, n2, k, n3)
            for h in contexts.children(h1, a, c1):
                yield 2, (h, c1, h2, c2, h3)
            for h in contexts.children(h2, c1, c2):
                yield 3, (h1, c1, h, c2, h3)
            for h in contexts.children(h3, c2, b):
                yield 4, (h1, c1, h2, c2, h)

    root = {0: (), 2: (top & contexts.mask(a, c), c, top & contexts.mask(c, b)),
            3: (top & contexts.mask(a, c), c, top & contexts.mask(c, c), c, top & contexts.mask(c, b))}[n_hops]
    passing, evaluated, seen = {}, 1, set()
    overlap = overlap_of(root)
    if overlap < min_support:
        return passing, evaluated
    passing[root] = overlap
    stack = [(root, 0)]
    while stack:
        path, last = stack.pop()
        for position, child in children(path):
            if position < last or (child, position) in seen:
                continue
            seen.add((child, position))
            evaluated += 1
            overlap = overlap_of(child)
            if overlap >= min_support:
                passing[child] = overlap
                stack.append((child, position))
    return passing, evaluated
