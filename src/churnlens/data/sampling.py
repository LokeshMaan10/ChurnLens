"""Deterministic, stream-friendly user sampling.

A user is kept iff int(sha256(msno + str(seed)), 16) % 100 < pct. The decision
depends only on the msno, so the same users survive in every table, both
snapshots, and any chunking or file order. No shuffling, no in-memory sample.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.compute as pc

FULL_PCT = 100


def keep_user(msno: str, seed: int, pct: int) -> bool:
    """Return True if this msno falls in the sample."""
    digest = hashlib.sha256(f"{msno}{seed}".encode()).hexdigest()
    return int(digest, 16) % 100 < pct


class UserSampler:
    """Applies the hash rule to Arrow columns, caching each msno's decision."""

    def __init__(self, seed: int, pct: int) -> None:
        self.seed = seed
        self.pct = pct
        self._cache: dict[str, bool] = {}

    @property
    def keeps_everyone(self) -> bool:
        """True in full mode, where hashing can be skipped entirely."""
        return self.pct >= FULL_PCT

    def mask(self, msno: pa.ChunkedArray | pa.Array) -> pa.Array:
        """Boolean mask of rows whose msno is in the sample."""
        if self.keeps_everyone:
            return pa.array([True] * len(msno), type=pa.bool_())
        kept = [u for u in pc.unique(msno).to_pylist() if u is not None and self._decide(u)]
        return pc.is_in(msno, value_set=pa.array(kept, type=pa.string()))

    def _decide(self, msno: str) -> bool:
        decision = self._cache.get(msno)
        if decision is None:
            decision = keep_user(msno, self.seed, self.pct)
            self._cache[msno] = decision
        return decision
