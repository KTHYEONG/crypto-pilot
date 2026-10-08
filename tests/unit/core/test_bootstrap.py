"""Invariant scenarios for the shared stationary block-bootstrap index kernel."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.bootstrap import (
    BootstrapIndexChunk,
    iter_stationary_bootstrap_index_chunks,
    stationary_bootstrap_max_blocks,
    stationary_bootstrap_scalar_indices,
)


def _collect(rng: np.random.Generator, **kwargs: object) -> list[BootstrapIndexChunk]:
    return list(iter_stationary_bootstrap_index_chunks(rng, **kwargs))  # type: ignore[arg-type]


def test_max_blocks_matches_historical_formula() -> None:
    """The block-count budget equals the historical inline expression."""
    for path_len, mean_block in ((1, 1), (37, 168), (50, 1), (400, 24), (525600, 168)):
        expected = min(path_len, int(np.ceil(path_len * 6.0 / mean_block)) + 16)
        result = stationary_bootstrap_max_blocks(path_len, mean_block)
        assert result == expected
        assert result <= path_len


def test_max_blocks_rejects_degenerate_inputs() -> None:
    """Zero path length or mean block fails closed."""
    with pytest.raises(ValueError, match="mean_block must be"):
        stationary_bootstrap_max_blocks(10, 0)
    with pytest.raises(ValueError, match="path_len must be"):
        stationary_bootstrap_max_blocks(0, 5)


def test_kernel_indices_pinned_golden() -> None:
    """Seeded kernel indices reproduce the pinned matrix bit-for-bit."""
    rng = np.random.default_rng(11)
    chunks = _collect(
        rng, source_len=40, path_len=9, n_replicates=4, mean_block=3, chunk_size=4, max_blocks=9,
    )
    assert len(chunks) == 1
    assert chunks[0].row_start == 0
    assert chunks[0].indices.tolist() == [
        [34, 26, 27, 3, 4, 5, 8, 22, 36],
        [1, 2, 3, 6, 7, 8, 9, 17, 18],
        [25, 26, 27, 28, 29, 27, 28, 29, 30],
        [39, 0, 1, 2, 3, 4, 5, 6, 7],
    ]
    assert rng.random() == 0.8458901064450575


def test_chunk_partition_covers_replicates_once() -> None:
    """Chunk row_starts tile [0, n_replicates) with int64 in-range matrices."""
    rng = np.random.default_rng(3)
    chunks = _collect(
        rng, source_len=50, path_len=50, n_replicates=300, mean_block=5, chunk_size=128,
        max_blocks=50,
    )
    assert [c.row_start for c in chunks] == [0, 128, 256]
    assert [c.indices.shape[0] for c in chunks] == [128, 128, 44]
    for chunk in chunks:
        assert chunk.indices.dtype == np.int64
        assert chunk.indices.shape[1] == 50
        assert bool(((chunk.indices >= 0) & (chunk.indices < 50)).all())


def test_chunking_is_deterministic_per_fixed_chunk_size() -> None:
    """Same seed and chunk size give identical indices; chunk 0 matches a manual replay."""
    kwargs: dict[str, object] = {
        "source_len": 50, "path_len": 50, "n_replicates": 130, "mean_block": 5,
        "chunk_size": 128, "max_blocks": 50,
    }
    first = np.concatenate([c.indices for c in _collect(np.random.default_rng(9), **kwargs)])
    second = np.concatenate([c.indices for c in _collect(np.random.default_rng(9), **kwargs)])
    assert np.array_equal(first, second)

    replay = np.random.default_rng(9)
    p_block = 1.0 / 5
    lengths = replay.geometric(p_block, size=(128, 50))
    starts = replay.integers(0, 50, size=(128, 50))
    ends = np.minimum(np.cumsum(lengths, axis=1), 50)
    used = ends - np.concatenate([np.zeros((128, 1), dtype=np.int64), ends[:, :-1]], axis=1)
    u = used.ravel()
    s = starts.ravel()
    keep = u > 0
    u = u[keep]
    s = s[keep]
    block_start = np.cumsum(u) - u
    offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
    assert np.array_equal(first[:128], ((np.repeat(s, u) + offsets) % 50).reshape(128, 50))


def test_shortfall_fallback_interleaves_before_next_chunk() -> None:
    """Every short row replays as scalar fallback draws between the chunk vector draws."""
    rng = np.random.default_rng(1)
    chunks = _collect(
        rng, source_len=60, path_len=60, n_replicates=10, mean_block=4, chunk_size=5,
        max_blocks=3,
    )
    assert len(chunks) == 2
    replay = np.random.default_rng(1)
    expected: list[np.ndarray] = []
    for _ in range(2):
        replay.geometric(0.25, size=(5, 3))
        replay.integers(0, 60, size=(5, 3))
        expected.extend(stationary_bootstrap_scalar_indices(replay, 60, 0.25) for _ in range(5))
    assert np.array_equal(np.concatenate([c.indices for c in chunks]), np.stack(expected))


def test_mixed_short_and_valid_rows_keep_row_positions() -> None:
    """A mixed chunk holds both short and valid rows, all full-length and in range."""
    rng = np.random.default_rng(7)
    chunks = _collect(
        rng, source_len=60, path_len=60, n_replicates=300, mean_block=4, chunk_size=128,
        max_blocks=14,
    )
    replay = np.random.default_rng(7)
    lengths = replay.geometric(1.0 / 4, size=(128, 14))
    replay.integers(0, 60, size=(128, 14))
    short = np.cumsum(lengths, axis=1)[:, -1] < 60
    assert bool(short.any())
    assert bool((~short).any())
    for chunk in chunks:
        assert chunk.indices.shape[1] == 60
        assert bool(((chunk.indices >= 0) & (chunk.indices < 60)).all())


def test_lazy_generator_does_not_pre_draw() -> None:
    """Consuming one chunk advances the caller rng by exactly one chunk's draws."""
    rng = np.random.default_rng(5)
    stream = iter_stationary_bootstrap_index_chunks(
        rng, source_len=50, path_len=50, n_replicates=10, mean_block=5, chunk_size=5,
        max_blocks=50,
    )
    next(stream)
    replay = np.random.default_rng(5)
    replay.geometric(1.0 / 5, size=(5, 50))
    replay.integers(0, 50, size=(5, 50))
    assert rng.random() == replay.random()


def test_zero_replicates_draw_nothing() -> None:
    """Zero replicates yield nothing and leave the generator state untouched."""
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    assert _collect(
        rng, source_len=50, path_len=50, n_replicates=0, mean_block=5, chunk_size=128,
        max_blocks=50,
    ) == []
    assert rng.bit_generator.state == before


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"source_len": 40, "path_len": 9, "n_replicates": 4, "mean_block": 3, "chunk_size": 4, "max_blocks": 8}, "requires source_len"),
        ({"source_len": 0, "path_len": 9, "n_replicates": 4, "mean_block": 3, "chunk_size": 4, "max_blocks": 9}, "source_len must be"),
        ({"source_len": 40, "path_len": 0, "n_replicates": 4, "mean_block": 3, "chunk_size": 4, "max_blocks": 9}, "path_len must be"),
        ({"source_len": 40, "path_len": 9, "n_replicates": -1, "mean_block": 3, "chunk_size": 4, "max_blocks": 9}, "n_replicates must be"),
        ({"source_len": 40, "path_len": 9, "n_replicates": 4, "mean_block": 0, "chunk_size": 4, "max_blocks": 9}, "mean_block must be"),
        ({"source_len": 40, "path_len": 9, "n_replicates": 4, "mean_block": 3, "chunk_size": 0, "max_blocks": 9}, "chunk_size must be"),
        ({"source_len": 40, "path_len": 9, "n_replicates": 4, "mean_block": 3, "chunk_size": 4, "max_blocks": 0}, "max_blocks must be"),
    ],
)
def test_invalid_geometry_rejected_without_drawing(kwargs: dict[str, object], match: str) -> None:
    """Bad geometry fails closed before the first draw."""
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match=match):
        list(iter_stationary_bootstrap_index_chunks(rng, **kwargs))  # type: ignore[arg-type]
    assert rng.bit_generator.state == before


def test_scalar_indices_truncate_at_array_end() -> None:
    """Scalar indices concatenate non-circular arange segments truncated to length."""
    out = stationary_bootstrap_scalar_indices(np.random.default_rng(2), 5, 0.0)
    assert out.dtype == np.int64
    assert out.shape == (5,)
    assert bool(((out >= 0) & (out < 5)).all())
    pos = 0
    while pos < 5:
        start = int(out[pos])
        run = 1
        while pos + run < 5 and out[pos + run] == start + run:
            run += 1
        assert out[pos : pos + run].tolist() == list(range(start, 5))[:run]
        pos += run


@pytest.mark.parametrize("p_block", [1.5, -0.1])
def test_scalar_indices_reject_invalid_probability(p_block: float) -> None:
    """Out-of-range termination probabilities fail closed."""
    with pytest.raises(ValueError, match="p_block must be"):
        stationary_bootstrap_scalar_indices(np.random.default_rng(2), 5, p_block)
    with pytest.raises(ValueError, match="source_len must be"):
        stationary_bootstrap_scalar_indices(np.random.default_rng(2), 0, 0.5)
