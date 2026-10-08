"""Cross-site liquid-half universe eligibility parity (single-source constants)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.lab.mhs.backtest.market_data as bt_market
import src.lab.mhs.evaluation.fold_weights as fold_weights_mod
import src.lab.mhs.pipeline.stages.selection as selection_mod
import src.market_data.services.mhs_execution as mhs_execution
from src.core.marks import _pit_execution_mask
from src.core.panel import liquid_half_eligibility as real_eligibility
from src.core.params import (PANEL_MIN_HISTORY_BARS, UNIVERSE_ELIGIBILITY_LOOKBACK_BARS, UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS)
from src.lab.mhs.params import FOLD_PANEL_WARMUP_HOURS

_CANONICAL = (UNIVERSE_ELIGIBILITY_LOOKBACK_BARS, UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS)


class _HaltEligibilityError(Exception):
    pass


def _spy(module, recorded: list) -> object:
    signature = inspect.signature(real_eligibility)

    def _recording(frame, *args, **kwargs):
        bound = signature.bind(frame, *args, **kwargs)
        bound.apply_defaults()
        recorded.append((
            (bound.arguments["lookback_bars"], bound.arguments["min_history_bars"]),
            tuple(frame.shape),
        ))
        raise _HaltEligibilityError

    _recording.__name__ = getattr(module.liquid_half_eligibility, "__name__", "liquid_half_eligibility")
    return _recording


def _seeded_quote_volume(n_rows: int, symbols: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_rows, freq="1h", tz="UTC")
    base = np.abs(rng.normal(1e6, 2e5, (n_rows, len(symbols)))) + np.arange(len(symbols)) * 1e5
    return pd.DataFrame(base, index=idx, columns=symbols)


def _parity_frame() -> pd.DataFrame:
    symbols = [f"P{i:02d}USDT" for i in range(8)]
    frame = _seeded_quote_volume(UNIVERSE_ELIGIBILITY_LOOKBACK_BARS + 200, symbols, 23)
    for j in range(len(symbols)):
        frame.iloc[: j * 10, j] = np.nan
    frame.iloc[UNIVERSE_ELIGIBILITY_LOOKBACK_BARS + 100, 0] = np.nan
    return frame


def test_constants_pinned() -> None:
    assert UNIVERSE_ELIGIBILITY_LOOKBACK_BARS == 720
    assert UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS == 720
    assert PANEL_MIN_HISTORY_BARS == UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS
    assert FOLD_PANEL_WARMUP_HOURS == 912
    for value in (
        UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
        PANEL_MIN_HISTORY_BARS,
        FOLD_PANEL_WARMUP_HOURS,
    ):
        assert isinstance(value, int)
    assert mhs_execution.DYNAMIC_GAP_EXCLUSION_HOURS == 720.0
    assert isinstance(mhs_execution.DYNAMIC_GAP_EXCLUSION_HOURS, float)


def test_constants_satisfy_callee_precondition() -> None:
    quote_vol = _seeded_quote_volume(
        UNIVERSE_ELIGIBILITY_LOOKBACK_BARS + 5, ["AUSDT", "BUSDT", "CUSDT"], 7,
    )
    eligible = real_eligibility(
        quote_vol,
        lookback_bars=UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        min_history_bars=UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )
    assert eligible.index.equals(quote_vol.index)
    assert list(eligible.columns) == list(quote_vol.columns)
    assert bool((eligible.dtypes.map(str) == "bool").all())


def test_fold_weights_site_uses_canonical_pair(mhs_market, monkeypatch) -> None:
    from src.core.marks import _load_funding_series
    from src.quant.universe.pit_universe import symbol_partition
    from tests.fixtures.mhs_requests import research_baseline
    from tests.lab.mhs.test_evaluation_appresearch import _FOLD, _START

    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = _load_funding_series(symbols)
    request = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
    )
    recorded: list = []
    monkeypatch.setattr(fold_weights_mod, "liquid_half_eligibility", _spy(fold_weights_mod, recorded))
    with pytest.raises(_HaltEligibilityError):
        fold_weights_mod._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)
    assert len(recorded) == 1
    assert recorded[0][0] == _CANONICAL


def test_selection_site_uses_canonical_pair(monkeypatch) -> None:
    import dataclasses

    from src.lab.mhs.contracts import MhsDiagnosticRequest
    from src.lab.mhs.pipeline.context import PipelineContext
    from src.lab.mhs.telemetry import StageTelemetry

    grid = pd.date_range("2021-01-01", periods=10, freq="1h", tz="UTC")
    syms = ["AAAUSDT", "BBBUSDT"]
    quote_vol = _seeded_quote_volume(len(grid), syms, 11)
    assert quote_vol.index.equals(grid)
    ctx = PipelineContext(
        config=dataclasses.replace(MhsDiagnosticRequest(), fold_safe_horizon_selection=False),
        resolved_end=None,
        start=grid[0],
        end=grid[-1],
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=grid,
        close=pd.DataFrame(1.0, index=grid, columns=syms),
        opens=pd.DataFrame(1.0, index=grid, columns=syms),
        quote_vol=quote_vol,
        taker_buy_quote=None,
        symbols=syms,
    )
    ctx.bar_funding = pd.DataFrame(0.0001, index=grid, columns=syms)
    recorded: list = []
    monkeypatch.setattr(selection_mod, "liquid_half_eligibility", _spy(selection_mod, recorded))
    with pytest.raises(_HaltEligibilityError):
        selection_mod.select_horizons(ctx, StageTelemetry(log_run=False))
    assert len(recorded) == 1
    assert recorded[0][0] == _CANONICAL


def test_process_backtest_site_uses_canonical_pair(monkeypatch) -> None:
    n_rows = UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS + 30
    grid = pd.date_range("2021-01-01", periods=n_rows, freq="1h", tz="UTC")
    syms = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT",
            "MHSEUSDT", "MHSGUSDT", "MHSHUSDT", "MHSIUSDT"]
    rng = np.random.default_rng(23)
    close = pd.DataFrame(
        100.0 + np.cumsum(rng.normal(0, 0.5, (n_rows, len(syms))), axis=0),
        index=grid, columns=syms,
    )
    panel = {
        "close": close,
        "open": close.copy(),
        "high": close + 0.3,
        "low": close - 0.3,
        "quote_vol": pd.DataFrame(
            rng.uniform(1e6, 2e6, (n_rows, len(syms))), index=grid, columns=syms,
        ),
        "taker_buy_quote": pd.DataFrame(
            rng.uniform(4e5, 6e5, (n_rows, len(syms))), index=grid, columns=syms,
        ),
    }
    funding = {s: pd.Series(rng.normal(0, 1e-5, n_rows), index=grid) for s in syms}
    monkeypatch.setattr(bt_market, "load_base_panel", lambda *a, **k: panel)
    monkeypatch.setattr(
        bt_market, "_load_funding_series",
        lambda syms_: ({s: funding[s] for s in syms_ if s in funding}, {}),
    )
    recorded: list = []
    monkeypatch.setattr(bt_market, "liquid_half_eligibility", _spy(bt_market, recorded))
    with pytest.raises(_HaltEligibilityError):
        bt_market.load_process_market_data(grid[0], grid[-1])
    assert len(recorded) == 1
    assert recorded[0][0] == _CANONICAL


def test_execution_plan_site_uses_canonical_pair(tmp_path, monkeypatch) -> None:
    idx = pd.date_range("2025-01-01", periods=2200, freq="1h", tz="UTC")
    quote = pd.DataFrame({f"S{i:02d}": float(i + 1) for i in range(16)}, index=idx)
    close = pd.DataFrame(
        {symbol: 100.0 + (i + 1) * pd.Series(range(len(idx)), index=idx)
         for i, symbol in enumerate(quote.columns)},
    )
    monkeypatch.setattr(
        mhs_execution, "load_base_panel",
        lambda *args, **kwargs: {"close": close, "quote_vol": quote},
    )
    monkeypatch.setattr(mhs_execution, "funding_path", lambda symbol: tmp_path / f"{symbol}.parquet")
    for symbol in quote.columns:
        (tmp_path / f"{symbol}.parquet").touch()
    recorded: list = []
    real = mhs_execution.liquid_half_eligibility
    monkeypatch.setattr(mhs_execution, "liquid_half_eligibility", _spy(mhs_execution, recorded))
    with pytest.raises(_HaltEligibilityError):
        mhs_execution.build_mhs_execution_plan("2025-01-01", "2025-03-30", execution_universe_size=8)
    assert len(recorded) == 1
    assert recorded[0][0] == _CANONICAL
    monkeypatch.setattr(mhs_execution, "liquid_half_eligibility", real)
    plan = mhs_execution.build_mhs_execution_plan("2025-01-01", "2025-03-30", execution_universe_size=8)
    assert len(plan.symbols) > 0


def test_cross_site_eligibility_parity() -> None:
    quote_vol = _parity_frame()
    pairs = {
        "fold": _CANONICAL,
        "selection": _CANONICAL,
        "process": _CANONICAL,
        "execution_plan": _CANONICAL,
    }
    assert all(pair == _CANONICAL for pair in pairs.values())
    masks = {
        site: real_eligibility(quote_vol, lookback_bars=lb, min_history_bars=mh)
        for site, (lb, mh) in pairs.items()
    }
    reference = masks["fold"]
    for site, mask in masks.items():
        assert mask.index.equals(reference.index), site
        pd.testing.assert_frame_equal(mask, reference)


def test_roster_admission_aligned_with_eligibility_history() -> None:
    symbols = [f"R{i:02d}USDT" for i in range(10)]
    quote_vol = _seeded_quote_volume(UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS + 20, symbols, 41)
    eligible = real_eligibility(
        quote_vol,
        lookback_bars=UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        min_history_bars=UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )
    roster = _pit_execution_mask(quote_vol, eligible, 8)
    start = UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS - 1
    assert not roster.iloc[:start].to_numpy().any()
    assert bool(roster.iloc[start].any())


def test_eligibility_causality_preserved() -> None:
    quote_vol = _parity_frame()
    horizon = UNIVERSE_ELIGIBILITY_LOOKBACK_BARS + 50
    shocked = quote_vol.copy()
    rng = np.random.default_rng(99)
    shocked.iloc[horizon + 1 :] = np.abs(rng.normal(5e6, 1e6, shocked.iloc[horizon + 1 :].shape))
    base = real_eligibility(
        quote_vol,
        lookback_bars=UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        min_history_bars=UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )
    moved = real_eligibility(
        shocked,
        lookback_bars=UNIVERSE_ELIGIBILITY_LOOKBACK_BARS,
        min_history_bars=UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS,
    )
    pd.testing.assert_frame_equal(base.iloc[: horizon + 1], moved.iloc[: horizon + 1])
