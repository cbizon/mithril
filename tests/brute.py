"""Brute-force helpers for search/score tests."""


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
