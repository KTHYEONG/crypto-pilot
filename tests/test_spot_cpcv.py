from __future__ import annotations

from src.spot_strategy.opt_spot_utils.cv_utils import (
    build_cpcv_test_paths,
    build_cpcv_test_paths_with_fallback,
)


def test_cpcv_six_blocks_yields_fifteen_paths() -> None:
    n = 1200
    paths = build_cpcv_test_paths(n, 6, 2)
    assert len(paths) == 15
    for p in paths:
        assert len(p) >= 1
        for s, e in p:
            assert 0 <= s < e <= n


def test_cpcv_fallback_prefers_config_primary_six_two() -> None:
    n = 500
    paths = build_cpcv_test_paths(n, 4, 2)
    assert len(paths) == 6
    paths_fb, nb, k = build_cpcv_test_paths_with_fallback(n)
    assert nb == 6
    assert k == 2
    assert len(paths_fb) == 15


def test_cpcv_embargo_trims_block_starts() -> None:
    n = 1200
    paths0 = build_cpcv_test_paths(n, 6, 2, embargo=0)
    paths7 = build_cpcv_test_paths(n, 6, 2, embargo=7)
    assert len(paths0) == len(paths7) == 15
    for p0, p7 in zip(paths0, paths7):
        assert len(p0) == len(p7)
        for (s0, e0), (s7, e7) in zip(p0, p7):
            assert e0 == e7
            assert s7 == s0 + 7
            assert 0 <= s7 < e7 <= n


def test_cpcv_embargo_too_large_returns_empty() -> None:
    n = 120
    assert build_cpcv_test_paths(n, 6, 2, embargo=25) == []
