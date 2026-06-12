from __future__ import annotations

from src.domain.futures.strategy.config import CandidateStrategyConfig, resolve_purge_and_embargo_bars
from src.domain.futures.strategy.walk_forward import WFFold, build_l1_swf_folds, build_walk_forward_folds


def _cfg(**kwargs: object) -> CandidateStrategyConfig:
    return CandidateStrategyConfig(**kwargs)  # type: ignore[arg-type]


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
    for fold in folds:
        assert fold.cal_start >= fold.fit_end + cfg.purge_bars


def test_embargo_gap_between_cal_and_oos() -> None:
    cfg = _cfg(wf_scheme="anchored", wf_n_folds=2, purge_bars=10, embargo_bars=10, wf_enabled=True)
    folds = build_walk_forward_folds(n_bars=1500, cfg=cfg)
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
    """S1: 인과 보장 + expanding fit."""
    # Arrange
    folds = build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=2000,
        l1_end_bars=7000,
        purge_bars=50,
        embargo_bars=20,
    )

    # Assert
    assert len(folds) == 5
    for fold in folds:
        assert fold.fit_end < fold.oos_start  # strictly causal
        assert fold.fit_start == 0            # expanding from bar 0


def test_build_l1_swf_folds_oos_bounds() -> None:
    """S2: OOS 파티션 — l1_end_bars 상한, warmup_bars 하한."""
    # Arrange
    folds = build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=2000,
        l1_end_bars=7000,
        purge_bars=50,
        embargo_bars=20,
    )

    # Assert
    assert folds[0].oos_start == 2000   # warmup_bars
    assert folds[-1].oos_end == 7000    # l1_end_bars (n_bars=10000 미사용)


def test_build_l1_swf_folds_purge_gap() -> None:
    """S3: purge 간격 정확성."""
    # Arrange
    folds = build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=2000,
        l1_end_bars=7000,
        purge_bars=100,
        embargo_bars=0,
    )

    # Assert
    for fold in folds:
        assert fold.fit_end == fold.oos_start - 100


def test_build_l1_swf_folds_insufficient_bars_fallback() -> None:
    """S4: bar 부족 → single fallback."""
    # Arrange / Act
    folds = build_l1_swf_folds(
        n_bars=100,
        warmup_bars=90,
        l1_end_bars=95,
        n_folds=5,
        purge_bars=2,
        embargo_bars=1,
    )

    # Assert
    assert len(folds) == 1  # fallback, no ValueError


def test_build_l1_swf_folds_equal_partition() -> None:
    """S5: fold 등간격 (마지막 제외)."""
    # Arrange
    folds = build_l1_swf_folds(
        n_bars=10000,
        n_folds=5,
        warmup_bars=1000,
        l1_end_bars=6000,
        purge_bars=10,
        embargo_bars=5,
    )
    # available = 5000, fold_len = 1000

    # Assert
    assert len(folds) == 5
    for fold in folds[:-1]:
        assert (fold.oos_end - fold.oos_start) == 1000
