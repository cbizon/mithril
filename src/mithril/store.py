"""Built edge store: edges grouped by semantic atom, plus node category sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Store:
    """Edges ``src[k] -> dst[k]`` of atom i are rows ``offsets[i]:offsets[i+1]``. Hop sets
    are bitsets over (atom, orientation): bit ``2 * atom`` is the stored direction, bit
    ``2 * atom + 1`` the reverse."""

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

    @property
    def n_bits(self) -> int:
        return 2 * (len(self.offsets) - 1)

    @property
    def n_catsets(self) -> int:
        return int(self.node_catset.max()) + 1

    def in_class(self, catset_bits: int) -> np.ndarray:
        """Per node: is its category set in the class (a bitset of category sets)?"""
        member = np.array([catset_bits >> c & 1 for c in range(self.n_catsets)], dtype=bool)
        return member[self.node_catset]

    def target_pairs(self, bits: int, subjects: np.ndarray, objects: np.ndarray) -> np.ndarray:
        """Distinct (subject, object) pairs, sorted, joined by an edge of the hop set
        ``bits`` that counts as target evidence; ``subjects``/``objects`` are node masks."""
        keys = []
        for bit in range(bits.bit_length()):
            if not bits >> bit & 1:
                continue
            atom, reverse = divmod(bit, 2)
            lo, hi = self.offsets[atom], self.offsets[atom + 1]
            ok = self.target_ok[lo:hi]
            s, d = self.src[lo:hi][ok], self.dst[lo:hi][ok]
            left, right = (d, s) if reverse else (s, d)
            keep = subjects[left] & objects[right]
            keys.append(left[keep].astype(np.int64) * self.n_nodes + right[keep])
        key = np.unique(np.concatenate(keys)) if keys else np.zeros(0, dtype=np.int64)
        return np.stack([key // self.n_nodes, key % self.n_nodes], axis=1)
