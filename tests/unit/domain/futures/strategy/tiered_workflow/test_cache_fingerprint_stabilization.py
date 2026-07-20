from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_events_df(n=10) -> pd.DataFrame:
    dt = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "datetime": dt, "symbol": ["SYM0"] * n,
        "side": [1] * n, "expected_gross_bps": [50.0] * n,
        "strategy_id": ["t"] * n, "native_tf": ["1h"] * n,
    })


def test_deterministic_fingerprint_same_returns_same():
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _deterministic_df_fingerprint,
    )
    df = _sample_events_df(10)
    fp1 = _deterministic_df_fingerprint(df, salt="test")
    fp2 = _deterministic_df_fingerprint(df, salt="test")
    fp3 = _deterministic_df_fingerprint(df.copy(), salt="test")
    assert fp1 == fp2 == fp3
    assert len(fp1) == 16


def test_deterministic_fingerprint_different_returns_different():
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _deterministic_df_fingerprint,
    )
    df1 = _sample_events_df(10)
    df2 = df1.copy()
    df2.iloc[-1, df2.columns.get_loc("side")] = -1
    assert _deterministic_df_fingerprint(df1, salt="t") != _deterministic_df_fingerprint(df2, salt="t")


def test_deterministic_fingerprint_empty_df_no_crash():
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _deterministic_df_fingerprint,
    )
    empty = pd.DataFrame()
    fp = _deterministic_df_fingerprint(empty, salt="test")
    assert len(fp) == 16


def test_deterministic_fingerprint_salt_changes_output():
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _deterministic_df_fingerprint,
    )
    df = _sample_events_df(5)
    assert _deterministic_df_fingerprint(df, salt="a") != _deterministic_df_fingerprint(df, salt="b")


def test_should_load_cache_accepts_when_rss_low(mocker):
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _should_load_cache,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._get_rss_mb",
        return_value=5000.0,
    )
    assert _should_load_cache(10, threshold_mb=11500, expansion_ratio=15.0) is True


def test_should_load_cache_rejects_when_rss_high(mocker):
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _should_load_cache,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._get_rss_mb",
        return_value=11000.0,
    )
    assert _should_load_cache(100, threshold_mb=11500, expansion_ratio=15.0) is False


def test_should_load_cache_returns_true_on_resource_error(mocker):
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _should_load_cache,
    )
    # RSS unknown (negative) and small cache -> allowed
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._get_rss_mb",
        return_value=-1.0,
    )
    assert _should_load_cache(64.0, threshold_mb=100, expansion_ratio=10.0) is True


def test_should_load_cache_rejects_large_cache_on_unknown_rss(mocker):
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        _should_load_cache,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._get_rss_mb",
        return_value=-1.0,
    )
    assert _should_load_cache(100, threshold_mb=100, expansion_ratio=10.0) is False


def test_l1_cache_hit_cross_process_deterministic(tmp_path, mocker):
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    cache_dir = str(tmp_path / "l1_det")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"L1_RESULT_CACHE_ENABLED": True, "L1_RESULT_CACHE_DIR": cache_dir},
        clear=False,
    )

    n_bars, n_sym = 200, 5
    base_dt = pd.date_range("2026-01-01", periods=n_bars, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    ones = np.ones((n_bars, n_sym), dtype=np.float64)
    mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned = AlignedMarketData(
        datetimes=base_dt, symbols=("BTCUSDT", *(f"S{i}" for i in range(n_sym - 1))),
        open_2d=ones * 100, high_2d=ones * 105, low_2d=ones * 95,
        close_2d=ones * 100, volume_2d=ones * 1000,
        funding_2d=np.zeros((n_bars, n_sym)), active_mask=mask,
        warm_mask=mask, entry_block_mask=mask, kill_mask=~mask,
    )
    lbl = _sample_events_df(10)
    outer_folds = (WFFold(fit_start=0, fit_end=50, cal_start=40, cal_end=50, oos_start=50, oos_end=80),)
    cfg = CandidateStrategyConfig()

    mocker.patch(
        "src.domain.futures.strategy.event_grid_contracts.normalize_native_l1_events",
        return_value=mocker.MagicMock(eligible_events=lbl, audit=None),
    )
    _fake = Layer1Result(
        signals_per_fold=(), oos_stacked={"S0": [0.1]},
        pooled_ic=0.0, pooled_tstat=0.0, breadth=0.0,
        valid_coverage=0.0, fold_pass_ratio=0.0, gate_passed=True,
        n_valid=1, n_total=1,
    )
    l1_spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf",
        return_value=_fake,
    )

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_per_tf_l1

    result1 = run_per_tf_l1(tf="1h", labeled_events=lbl, aligned=aligned, outer_folds=outer_folds, cfg=cfg, seed=42)
    result2 = run_per_tf_l1(tf="1h", labeled_events=lbl, aligned=aligned, outer_folds=outer_folds, cfg=cfg, seed=42)
    assert l1_spy.call_count == 1
    assert result1.tf == result2.tf == "1h"


def test_l1_cache_gc_collect_called_on_hit(tmp_path, mocker):
    import gc

    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.walk_forward import WFFold
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    cache_dir = str(tmp_path / "l1_gc")
    mocker.patch.dict(
        OPT_FUTURES_CONFIG,
        {"L1_RESULT_CACHE_ENABLED": True, "L1_RESULT_CACHE_DIR": cache_dir},
        clear=False,
    )

    n_bars, n_sym = 200, 5
    base_dt = pd.date_range("2026-01-01", periods=n_bars, freq="1h", tz="UTC").to_numpy(dtype="datetime64[ns]")
    ones = np.ones((n_bars, n_sym), dtype=np.float64)
    mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned = AlignedMarketData(
        datetimes=base_dt, symbols=("BTCUSDT", *(f"S{i}" for i in range(n_sym - 1))),
        open_2d=ones * 100, high_2d=ones * 105, low_2d=ones * 95,
        close_2d=ones * 100, volume_2d=ones * 1000,
        funding_2d=np.zeros((n_bars, n_sym)), active_mask=mask,
        warm_mask=mask, entry_block_mask=mask, kill_mask=~mask,
    )
    lbl = _sample_events_df(10)
    outer_folds = (WFFold(fit_start=0, fit_end=50, cal_start=40, cal_end=50, oos_start=50, oos_end=80),)
    cfg = CandidateStrategyConfig()

    mocker.patch(
        "src.domain.futures.strategy.event_grid_contracts.normalize_native_l1_events",
        return_value=mocker.MagicMock(eligible_events=lbl, audit=None),
    )
    _fake = Layer1Result(
        signals_per_fold=(), oos_stacked={"S0": [0.1]},
        pooled_ic=0.0, pooled_tstat=0.0, breadth=0.0,
        valid_coverage=0.0, fold_pass_ratio=0.0, gate_passed=True,
        n_valid=1, n_total=1,
    )
    l1_spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf",
        return_value=_fake,
    )

    gc_spy = mocker.patch.object(gc, "collect", wraps=gc.collect)

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_per_tf_l1

    run_per_tf_l1(tf="1h", labeled_events=lbl, aligned=aligned, outer_folds=outer_folds, cfg=cfg, seed=42)
    run_per_tf_l1(tf="1h", labeled_events=lbl, aligned=aligned, outer_folds=outer_folds, cfg=cfg, seed=42)
    assert l1_spy.call_count == 1
    assert gc_spy.call_count >= 2
