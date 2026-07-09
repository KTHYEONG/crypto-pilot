from __future__ import annotations

import importlib
import sys
from concurrent.futures import Future

import numba
import numpy as np
import pandas as pd
import pytest

from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.signals.rules import build_rule_signal_panels, candidate_panels_to_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig


def _make_aligned(t: int = 400, n: int = 8) -> AlignedMarketData:
    rng = np.random.default_rng(7)
    base = 100.0 + 0.25 * np.arange(t, dtype=np.float64)
    close = np.column_stack([base * (1.0 + 0.01 * np.sin(0.05 * np.arange(t))) for _ in range(n)])
    close = np.maximum(close + rng.normal(0.0, 0.1, size=(t, n)), 1.0)
    dt = np.arange(
        np.datetime64("2026-01-01T00"),
        np.datetime64("2026-03-08T16"),
        np.timedelta64(4, "h"),
        dtype="datetime64[ns]",
    )[:t]
    symbols = (*tuple(f"SYM{i}USDT" for i in range(n - 1)), "BTCUSDT")
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt,
        symbols=symbols,
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((t, n), 1_000.0),
        funding_2d=np.full((t, n), 0.0001),
        active_mask=mask,
        warm_mask=mask,
        entry_block_mask=np.zeros_like(mask),
        kill_mask=np.zeros_like(mask),
        oi_2d=np.full((t, n), 10_000.0),
        lsr_2d=np.full((t, n), 1.2),
        taker_buy_2d=np.full((t, n), 500.0),
        trades_2d=np.full((t, n), 100.0),
        execution_cost_bps_2d=np.full((t, n), 2.5),
    )


class _InlineExecutor:
    def __init__(self, *_: object, **__: object) -> None:
        self._closed = False

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        self._closed = True

    def submit(self, fn: object, /, *args: object, **kwargs: object) -> Future:
        fut: Future = Future()
        fut.set_result(fn(*args, **kwargs))  # type: ignore[misc]
        return fut


def test_build_rule_signal_panels_and_candidate_events_cover_main_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aligned = _make_aligned()
    cfg = CandidateStrategyConfig()
    families = (
        "btc_regime_pullback",
        "trend_pullback_continuation",
        "mtf_trend_pullback",
        "trend_pullback_quality_v2",
    )

    monkeypatch.setattr("src.domain.futures.signals.rules.ThreadPoolExecutor", _InlineExecutor)

    panels = build_rule_signal_panels(aligned=aligned, cfg=cfg, family_filter=families)
    assert isinstance(panels, tuple)
    assert len(panels) > 0

    panel_t = aligned.close_2d.shape[0]
    panel_scores = np.where(np.arange(panel_t)[:, None] % 6 < 3, 0.9, 0.0).astype(np.float64)
    panel_sides = np.where(np.arange(panel_t)[:, None] % 6 < 3, 1, 0).astype(np.int8)
    panel_cost = np.full((panel_t, 1), 2.5, dtype=np.float64)

    panel = CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=("BTCUSDT",),
        signed_score_2d=panel_scores,
        side_hint_2d=panel_sides,
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((panel_t, 1), dtype=np.float64),
        valid_mask_2d=np.ones((panel_t, 1), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )

    empty_events = candidate_panels_to_events((), min_abs_score=0.0)
    masked_panel = CandidateSignalPanel(
        family="trend_ma",
        variant="masked",
        params={"fast": 12, "slow": 72},
        datetimes=aligned.datetimes,
        symbols=("BTCUSDT",),
        signed_score_2d=np.zeros((panel_t, 1), dtype=np.float64),
        side_hint_2d=np.zeros((panel_t, 1), dtype=np.int8),
        expected_holding_bars=3,
        min_holding_bars=1,
        stop_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.zeros((panel_t, 1), dtype=np.float64),
        valid_mask_2d=np.ones((panel_t, 1), dtype=np.bool_),
        metadata={"recipe_id": "r1"},
    )
    masked_events = candidate_panels_to_events((masked_panel,), min_abs_score=0.0, execution_cost_bps_2d=panel_cost)
    flipped_events = candidate_panels_to_events(
        (panel,),
        min_abs_score=0.0,
        side_flip_variants=("trend_ma:ema_12_72",),
        execution_cost_bps_2d=panel_cost,
    )
    events = candidate_panels_to_events((panel,), min_abs_score=0.0, execution_cost_bps_2d=panel_cost)

    assert empty_events.empty
    assert masked_events.empty
    assert not flipped_events.empty
    assert isinstance(events, pd.DataFrame)
    assert not events.empty
    assert set(events["family"]) == {"trend_ma"}


def test_cross_sectional_robust_zscore_python_fallback_covers_numba_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")

    def _identity_njit(*args: object, **kwargs: object) -> object:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda fn: fn

    monkeypatch.setattr(numba, "njit", _identity_njit)
    sys.modules.pop("src.domain.futures.signals.rules", None)
    reloaded = importlib.import_module("src.domain.futures.signals.rules")
    raw_scores = np.array([1.0, 2.0, np.nan, 4.0, 9.0], dtype=np.float64)
    groups = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    result = reloaded._cross_sectional_robust_zscore(raw_scores, groups)

    assert result.shape == raw_scores.shape
    assert np.isfinite(result[0])
    assert np.isfinite(result[3])
