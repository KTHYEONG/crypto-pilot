"""Stationary block-bootstrap index kernel shared by MHS evidence and sizing.

One RNG draw protocol serves every seeded bootstrap in ``src.mhs`` so CI bounds,
deployment-readiness probabilities and exposure curves stay bit-identical
across refactors. Callers own seeding, chunk sizing, the block-count budget and
the output reduction; this module owns only index composition.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np

BOOTSTRAP_BLOCK_COUNT_SAFETY_FACTOR: Final[float] = 6.0
BOOTSTRAP_BLOCK_COUNT_SLACK: Final[int] = 16


def stationary_bootstrap_max_blocks(path_len: int, mean_block: int) -> int:
    """Block-count budget per replicate that makes a vectorized shortfall negligible.

    Drawing ``SAFETY_FACTOR`` times the expected block count (plus a fixed slack for
    short paths) makes the drawn lengths sum below ``path_len`` astronomically
    unlikely; the budget never exceeds ``path_len`` because every geometric length
    is at least 1, so ``path_len`` blocks always cover the path.

    Args:
        path_len: Replicate path length in observations (>= 1).
        mean_block: Mean block length (>= 1).
    Returns:
        ``min(path_len, ceil(path_len * SAFETY_FACTOR / mean_block) + SLACK)``.
    Raises:
        ValueError: ``path_len < 1`` or ``mean_block < 1``.
    """
    if isinstance(path_len, bool) or not isinstance(path_len, (int, np.integer)) or int(path_len) < 1:
        raise ValueError(f"path_len must be >= 1, got {path_len!r}")
    if isinstance(mean_block, bool) or not isinstance(mean_block, (int, np.integer)) or int(mean_block) < 1:
        raise ValueError(f"mean_block must be >= 1, got {mean_block!r}")
    n = int(path_len)
    mb = int(mean_block)
    return min(n, int(np.ceil(n * BOOTSTRAP_BLOCK_COUNT_SAFETY_FACTOR / mb)) + BOOTSTRAP_BLOCK_COUNT_SLACK)


def stationary_bootstrap_scalar_indices(
    rng: np.random.Generator, source_len: int, p_block: float,
) -> np.ndarray:
    """Source indices of one scalar block-bootstrap replicate of length ``source_len``.

    This is the original while-loop law kept as the reference fallback: block starts
    are uniform, a block grows while ``rng.random() > p_block`` (capped at
    ``source_len``), is truncated at the array end (not circular), then truncated to
    the remaining path length. It serves the vectorized shortfall fallback and the
    degenerate ``p_block == 0`` configuration, where every block grows to full length.

    Args:
        rng: Generator advanced in place; per block exactly one scalar ``integers(0,
            source_len)`` call followed by one scalar ``random()`` call per length
            increment attempt.
        source_len: Number of source observations and the replicate length (>= 1).
        p_block: Block termination probability in [0, 1].
    Returns:
        1-D int64 array of length ``source_len`` with values in ``[0, source_len)``.
    Raises:
        ValueError: ``source_len < 1`` or ``p_block`` outside [0, 1].
    """
    if isinstance(source_len, bool) or not isinstance(source_len, (int, np.integer)) or int(source_len) < 1:
        raise ValueError(f"source_len must be >= 1, got {source_len!r}")
    n = int(source_len)
    p = float(p_block)
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"p_block must be in [0, 1], got {p_block!r}")
    parts: list[np.ndarray] = []
    filled = 0
    while filled < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p:
            length += 1
        length = min(length, n - filled)
        seg = np.arange(start, min(start + length, n), dtype=np.int64)
        parts.append(seg)
        filled += int(seg.size)
    return np.concatenate(parts)[:n]


@dataclass(frozen=True, slots=True)
class BootstrapIndexChunk:
    """One chunk of replicate source-index paths.

    Attributes:
        row_start: Absolute replicate index of ``indices[0]``.
        indices: int64 matrix ``(rows, path_len)``; row ``i`` is replicate
            ``row_start + i``. Values lie in ``[0, source_len)``.
    """

    row_start: int
    indices: np.ndarray


def iter_stationary_bootstrap_index_chunks(
    rng: np.random.Generator,
    *,
    source_len: int,
    path_len: int,
    n_replicates: int,
    mean_block: int,
    chunk_size: int,
    max_blocks: int,
) -> Iterator[BootstrapIndexChunk]:
    """Lazily compose stationary block-bootstrap index paths chunk by chunk.

    Block lengths are ``geometric(1 / mean_block)`` and block starts uniform on the
    source; blocks wrap circularly over the source and the last block is truncated
    at ``path_len``. Per chunk the generator draws, in this exact order, one
    ``geometric`` matrix ``(rows, max_blocks)``, one ``integers`` matrix of the same
    shape, then -- for each row whose drawn lengths sum below ``path_len``, in
    ascending row order -- one scalar fallback replicate
    (``stationary_bootstrap_scalar_indices``). The next chunk is drawn only when the
    consumer resumes iteration, so a caller-owned generator observes exactly the same
    draw sequence as the historical inline loops, and peak memory stays at one
    ``(chunk_size, path_len)`` index matrix.

    Args:
        rng: Generator advanced in place; seeding is the caller's responsibility.
        source_len: Number of source observations (>= 1).
        path_len: Length of every replicate path (>= 1).
        n_replicates: Total replicates (>= 0; zero yields nothing and draws nothing).
        mean_block: Mean block length (>= 1).
        chunk_size: Replicates per chunk (>= 1); the last chunk may be smaller.
        max_blocks: Blocks drawn per replicate (>= 1). ``max_blocks >= path_len``
            guarantees no shortfall.
    Yields:
        ``BootstrapIndexChunk`` in ascending ``row_start`` order covering
        ``[0, n_replicates)`` exactly once.
    Raises:
        ValueError: Any size argument below its bound, or ``max_blocks < path_len``
            with ``source_len != path_len`` (the scalar fallback is defined only for
            equal source and path lengths).
    """
    for name, value, bound in (
        ("source_len", source_len, 1),
        ("path_len", path_len, 1),
        ("n_replicates", n_replicates, 0),
        ("mean_block", mean_block, 1),
        ("chunk_size", chunk_size, 1),
        ("max_blocks", max_blocks, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < bound:
            raise ValueError(f"{name} must be >= {bound}, got {value!r}")
    n_src = int(source_len)
    n_path = int(path_len)
    n_rep = int(n_replicates)
    mb = int(mean_block)
    csize = int(chunk_size)
    n_blocks = int(max_blocks)
    if n_blocks < n_path and n_src != n_path:
        raise ValueError(
            f"max_blocks ({n_blocks}) < path_len ({n_path}) requires source_len == path_len, "
            f"got source_len={n_src}",
        )
    p_block = 1.0 / mb
    for r0 in range(0, n_rep, csize):
        k = min(r0 + csize, n_rep) - r0
        lengths = rng.geometric(p_block, size=(k, n_blocks))
        starts = rng.integers(0, n_src, size=(k, n_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n_path
        short_rows = np.flatnonzero(short).tolist()
        fallback = [stationary_bootstrap_scalar_indices(rng, n_src, p_block) for _r in short_rows]
        valid = ~short
        ends_trunc = np.minimum(ends, n_path)
        used = ends_trunc - np.concatenate(
            [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
        )
        u = used[valid].ravel()
        s = starts[valid].ravel()
        keep = u > 0
        u = u[keep]
        s = s[keep]
        block_start = np.cumsum(u) - u
        offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
        flat = (np.repeat(s, u) + offsets) % n_src
        if not short_rows:
            yield BootstrapIndexChunk(r0, flat.reshape(k, n_path))
        else:
            indices = np.empty((k, n_path), dtype=np.int64)
            if valid.any():
                indices[valid] = flat.reshape(int(valid.sum()), n_path)
            for r, row in zip(short_rows, fallback, strict=True):
                indices[r] = row
            yield BootstrapIndexChunk(r0, indices)
