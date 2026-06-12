from __future__ import annotations

import inspect

import pytest

from src.domain.futures.strategy.config import CandidateStrategyConfig, resolve_purge_and_embargo_bars
from src.domain.futures.strategy.walk_forward import WFFold, build_l1_swf_folds, build_walk_forward_folds


def _cfg(**kwargs: object) -> CandidateStrategyConfig:
    return CandidateStrategyConfig(**kwargs)  # type: ignore[arg-type]


def _build_l1_swf_folds(**kwargs: int | float) -> tuple[WFFold, ...]:
    params = inspect.signature(build_l1_swf_folds).parameters
    if "l1_start_bars" in params:
        kwargs["l1_start_bars"] = kwargs.pop("warmup_bars")
    return build_l1_swf_folds(**kwargs)  # type: ignore[arg-type]


def test_single_scheme_returns_one_fold() -> None:
    cfg = _cfg(wf_scheme="single", wf_n_folds=4, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=1000, cfg=cfg)
    assert len(folds) == 1


def test_single_fold_matches_candidate_split() -> None:
    from src.domain.futures.strategy_runtime.bridge import _candidate_ml_split_indices

    cfg = _cfg(wf_scheme="single", wf_n_folds=1, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=1000, cfg=cfg)
    purge_bars, embargo_bars = resolve_purge_and_embargo_bars(cfg)
    fs, fe, _cs, _ce, os_, oe = _candidate_ml_split_indices(
        n_bars=1000,
        fit_fraction=cfg.ml_fit_fraction,
        calibration_fraction=cfg.ml_calibration_fraction,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    assert folds[0].fit_start == fs
    assert folds[0].fit_end == fe
    assert folds[0].oos_start == os_
    assert folds[0].oos_end == oe


def test_anchored_scheme_returns_n_folds() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=4, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=2000, cfg=cfg)
    assert len(folds) >= 1
    assert len(folds) <= 4


def test_anchored_fit_starts_at_zero() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=3, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=2000, cfg=cfg)
    for fold in folds:
        assert fold.fit_start == 0


def test_rolling_fit_moves_forward() -> None:
    cfg = _cfg(wf_scheme="rolling", wf_n_folds=3, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=3000, cfg=cfg)
    if len(folds) >= 2:
        assert folds[1].fit_start >= folds[0].fit_start


def test_oos_windows_non_overlapping() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=4, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=2000, cfg=cfg)
    for i in range(1, len(folds)):
        assert folds[i].oos_start >= folds[i - 1].oos_end


def test_purge_gap_between_fit_and_cal() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=2, purge_bars=10, embargo_bars=5, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=1500, cfg=cfg)
    assert cfg.purge_bars is not None
    for fold in folds:
        assert fold.cal_start >= fold.fit_end + cfg.purge_bars


def test_embargo_gap_between_cal_and_oos() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=2, purge_bars=10, embargo_bars=10, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=1500, cfg=cfg)
    assert cfg.embargo_bars is not None
    for fold in folds:
        assert fold.oos_start >= fold.cal_end + cfg.embargo_bars


def test_fallback_to_single_when_too_few_bars() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=10, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=50, cfg=cfg)
    assert len(folds) >= 1


def test_wffold_fields_are_valid() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=3, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=2000, cfg=cfg)
    for fold in folds:
        assert isinstance(fold, WFFold)
        assert fold.fit_start < fold.fit_end
        assert fold.cal_start < fold.cal_end
        assert fold.oos_start < fold.oos_end


# ---------------------------------------------------------------------------
# build_l1_swf_folds 테스트
# ---------------------------------------------------------------------------


def test_build_l1_swf_folds_causal_and_expanding() -> None:
    """S1: initial-train block 이후 L1 시작점부터 expanding fit."""
    # Arrange
    folds = _build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=2000,
        l1_end_bars=8000,
        purge_bars=50,
        embargo_bars=20,
    )

    # Assert
    assert len(folds) == 5
    for fold in folds:
        assert fold.fit_end < fold.oos_start  # strictly causal
        assert fold.fit_start == 2000         # expanding from L1 start
    assert folds[0].oos_start == 3000        # first block reserved for initial train
    assert folds[-1].oos_end == 8000


def test_build_l1_swf_folds_oos_bounds() -> None:
    """S2: warm-up 구간은 학습 이벤트에서 제외되고 OOS는 initial-train 뒤에서 시작."""
    # Arrange
    folds = _build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=2000,
        l1_end_bars=8000,
        purge_bars=50,
        embargo_bars=20,
    )

    # Assert
    assert min(fold.fit_start for fold in folds) == 2000
    assert folds[0].oos_start > 2000
    assert folds[-1].oos_end == 8000


def test_build_l1_swf_folds_purge_gap() -> None:
    """S3: purge 간격 정확성."""
    # Arrange
    folds = _build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=2000,
        l1_end_bars=8000,
        purge_bars=100,
        embargo_bars=0,
    )

    # Assert
    for fold in folds:
        assert fold.fit_end == fold.oos_start - 100


def test_build_l1_swf_folds_insufficient_bars_fallback() -> None:
    """S4: initial-train + OOS block을 만들 수 없으면 ValueError."""
    with pytest.raises(ValueError, match=r".+"):
        _build_l1_swf_folds(
            n_bars=100,
            warmup_bars=90,
            l1_end_bars=95,
            n_folds=5,
            purge_bars=2,
            embargo_bars=1,
        )


def test_build_l1_swf_folds_equal_partition() -> None:
    """S5: L1 구간은 initial-train 1개 + OOS 5개 동일 block으로 분할."""
    # Arrange
    folds = _build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=1000,
        l1_end_bars=7000,
        purge_bars=10,
        embargo_bars=5,
    )
    # available = 6000, block_len = 1000, first block = initial train only

    # Assert
    assert len(folds) == 5
    for fold in folds:
        assert (fold.oos_end - fold.oos_start) == 1000
