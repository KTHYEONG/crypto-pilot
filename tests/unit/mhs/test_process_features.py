"""Invariant guards for decision-label process features and candidate books."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.books import rank_weight_book
from src.mhs.features import FEATURE_REGISTRY, _finite
from src.mhs.funding import funding_carry_signal
from src.mhs.horizons import horizon_log_return
from src.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    PROCESS_FEATURE_CANDIDATES,
    PROCESS_FUNDING_CARRY_CANDIDATES_HOURS,
    PROCESS_MIN_SYMBOLS,
)
import src.mhs.backtest.market_data as bt_market
from src.mhs.backtest.market_data import build_candidate_member_books
from src.mhs.process_features import build_process_feature_grid
from src.mhs.regime import beta_neutralize_weights, causal_market_beta

_REGISTRY = {spec.name: spec for spec in FEATURE_REGISTRY}


def _deterministic_panels(n_bars: int = 1800, n_symbols: int = 9, seed: int = 8127):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n_bars, freq="h", tz="UTC")
    columns = [f"S{i:02d}USDT" for i in range(n_symbols)]
    close = pd.DataFrame(
        np.exp(rng.normal(0, 0.01, (n_bars, n_symbols)).cumsum(axis=0)),
        index=index,
        columns=columns,
    )
    quote_vol = pd.DataFrame(
        np.abs(rng.normal(1e6, 1e5, (n_bars, n_symbols))), index=index, columns=columns
    )
    taker_buy_quote = quote_vol * rng.uniform(0.4, 0.6, (n_bars, n_symbols))
    panels = {
        "close": close,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "quote_vol": quote_vol,
        "taker_buy_quote": taker_buy_quote,
    }
    decisions = index[::24]
    return panels, decisions


def test_process_feature_grid_matches_registered_builders() -> None:
    panels, decisions = _deterministic_panels()
    for name in PROCESS_FEATURE_CANDIDATES:
        spec = _REGISTRY[name]
        expected = spec.builder(panels).reindex(decisions)
        actual = build_process_feature_grid(spec, panels, decisions)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert list(actual.columns) == list(panels["close"].columns)
        assert str(actual.dtypes.iloc[0]) == "float64"


def test_process_feature_grid_block_width_independent() -> None:
    panels, decisions = _deterministic_panels(n_symbols=10)
    for name in ("mom_168h", "xs_idio_mom_336h", "flow_imb_720h", "amihud"):
        spec = _REGISTRY[name]
        reference = build_process_feature_grid(spec, panels, decisions, column_block_size=32)
        for width in (1, 3, 4, 10):
            actual = build_process_feature_grid(spec, panels, decisions, column_block_size=width)
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)


def test_process_feature_grid_uses_canonical_idiosyncratic_market() -> None:
    panels, decisions = _deterministic_panels(n_bars=1800, n_symbols=8, seed=5)
    drift = np.zeros((len(panels["close"]), 8))
    drift[:, :4] = 0.004
    drift[:, 4:] = -0.004
    close = panels["close"] * np.exp(pd.DataFrame(drift, index=panels["close"].index, columns=panels["close"].columns))
    panels = dict(panels)
    panels["close"] = close
    spec = _REGISTRY["xs_idio_mom_336h"]
    actual = build_process_feature_grid(spec, panels, decisions, column_block_size=3)
    expected = spec.builder(panels).reindex(decisions)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    block_frames = []
    for left in range(0, 8, 3):
        local = {k: v.iloc[:, left : left + 3] for k, v in panels.items()}
        raw = horizon_log_return(np.log(local["close"]), 336)
        market = raw.mean(axis=1)
        mean_r = raw.rolling(336, min_periods=336).mean()
        mean_m = market.rolling(336, min_periods=336).mean()
        mean_rm = raw.mul(market, axis=0).rolling(336, min_periods=336).mean()
        mean_m2 = market.pow(2).rolling(336, min_periods=336).mean()
        beta = (mean_rm - mean_r.mul(mean_m, axis=0)).div((mean_m2 - mean_m.pow(2)).replace(0, np.nan), axis=0)
        residual = raw - beta.mul(market, axis=0)
        residual_vol = residual.rolling(336, min_periods=336).std(ddof=1) * np.sqrt(336)
        block_frames.append(_finite(residual.div(residual_vol.replace(0, np.nan))).reindex(decisions))
    wrong = pd.concat(block_frames, axis=1).reindex(columns=list(panels["close"].columns))
    assert not wrong.equals(expected)


def test_process_feature_grid_listing_and_missing_history() -> None:
    panels, decisions = _deterministic_panels()
    for frame in panels.values():
        frame.iloc[:400, :2] = np.nan
        frame.iloc[1100:1105, 4] = np.nan
    panels["close"].iloc[:, 7] = panels["close"].iloc[0, 7]
    for name in PROCESS_FEATURE_CANDIDATES:
        spec = _REGISTRY[name]
        expected = spec.builder(panels).reindex(decisions)
        actual = build_process_feature_grid(spec, panels, decisions, column_block_size=4)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_process_feature_grid_future_perturbation_invariance() -> None:
    panels, decisions = _deterministic_panels()
    cutoff = panels["close"].index[1400]
    spec = _REGISTRY["mom_168h"]
    before = build_process_feature_grid(spec, panels, decisions)
    shocked = {k: v.copy() for k, v in panels.items()}
    for frame in shocked.values():
        frame.loc[frame.index > cutoff] *= 17.0
    after = build_process_feature_grid(spec, shocked, decisions)
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff], check_exact=True)


def test_process_feature_grid_rejects_invalid_inputs() -> None:
    panels, decisions = _deterministic_panels()
    spec = _REGISTRY["mom_168h"]
    for bad in (True, False, 0, -1, "32"):
        with pytest.raises(ValueError, match="column_block_size"):
            build_process_feature_grid(spec, panels, decisions, column_block_size=bad)
    incomplete = {k: v for k, v in panels.items() if k != "close"}
    with pytest.raises(ValueError, match="required_columns"):
        build_process_feature_grid(spec, incomplete, decisions)
    shifted = dict(panels)
    shifted["close"] = panels["close"].copy()
    shifted["close"].index = panels["close"].index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="labels"):
        build_process_feature_grid(spec, shifted, decisions)
    shuffled = dict(panels)
    shuffled["close"] = panels["close"].iloc[:, ::-1]
    with pytest.raises(ValueError, match="order"):
        build_process_feature_grid(spec, shuffled, decisions)
    from src.mhs.features import FeatureSpec

    outsider = FeatureSpec(name="nope", required_columns=("close",), min_coverage=0.0, builder=lambda p: p["close"])
    with pytest.raises(ValueError, match="unsupported"):
        build_process_feature_grid(outsider, panels, decisions)


def test_decision_first_rank_matches_full_hour_reference() -> None:
    panels, decisions = _deterministic_panels(n_bars=1000, n_symbols=10)
    eligible = pd.DataFrame(True, index=panels["close"].index, columns=panels["close"].columns)
    eligible.iloc[:30, :] = False
    eligible.iloc[100, :] = False
    mask_grid = eligible.reindex(decisions).fillna(False)
    for name, sign in (("mom_168h", 1), ("flow_imb_168h", 1)):
        spec = _REGISTRY[name]
        feature_grid = build_process_feature_grid(spec, panels, decisions, column_block_size=4)
        feature_grid.iloc[3, :] = 0.25
        feature_grid.iloc[5, :] = np.nan
        tuned = rank_weight_book(feature_grid, mask_grid, sign, PROCESS_MIN_SYMBOLS)
        full = spec.builder(panels)
        legacy = rank_weight_book(full, eligible, sign, PROCESS_MIN_SYMBOLS).reindex(decisions).fillna(0.0)
        mask_rows = mask_grid.sum(axis=1) >= PROCESS_MIN_SYMBOLS
        pd.testing.assert_frame_equal(tuned.loc[mask_rows], legacy.loc[mask_rows], check_exact=True)


def test_candidate_member_books_match_prechange_reference() -> None:
    panels, decisions = _deterministic_panels(n_bars=1000, n_symbols=10)
    rng = np.random.default_rng(3)
    funding = pd.DataFrame(
        rng.normal(0, 1e-4, panels["close"].shape), index=panels["close"].index, columns=panels["close"].columns
    )
    funding.iloc[:50, :2] = np.nan
    eligible = panels["close"].notna()
    mask = panels["close"].notna()
    books = build_candidate_member_books(panels, funding, eligible, mask, decisions)
    expected_keys = list(PROCESS_FEATURE_CANDIDATES) + [f"funding_carry_{h}h" for h in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS]
    assert list(books.keys()) == expected_keys
    beta_grid = causal_market_beta(
        np.log(panels["close"]), eligible, CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS
    ).reindex(decisions)
    mask_grid = mask.reindex(decisions).fillna(False)
    for name in PROCESS_FEATURE_CANDIDATES:
        feature = _REGISTRY[name].builder(panels)
        grid = feature.reindex(decisions)
        finite = pd.DataFrame(
            np.isfinite(grid.to_numpy(dtype="float64")),
            index=grid.index,
            columns=list(grid.columns),
        )
        validated = mask_grid & finite
        legacy = beta_neutralize_weights(
            rank_weight_book(grid, validated, 1, PROCESS_MIN_SYMBOLS),
            beta_grid,
            validated,
            PROCESS_MIN_SYMBOLS,
        )
        pd.testing.assert_frame_equal(books[name], legacy, check_exact=True)
    for lookback in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS:
        signal_grid = funding_carry_signal(funding, lookback).reindex(decisions)
        finite_signal = pd.DataFrame(
            np.isfinite(signal_grid.to_numpy(dtype="float64")),
            index=signal_grid.index,
            columns=list(signal_grid.columns),
        )
        validated_carry = mask_grid & finite_signal
        legacy = beta_neutralize_weights(
            rank_weight_book(signal_grid, validated_carry, -1, PROCESS_MIN_SYMBOLS),
            beta_grid,
            validated_carry,
            PROCESS_MIN_SYMBOLS,
        )
        pd.testing.assert_frame_equal(books[f"funding_carry_{lookback}h"], legacy, check_exact=True)


def test_funding_carry_books_match_full_history_reference() -> None:
    panels, decisions = _deterministic_panels(n_bars=1200, n_symbols=10)
    rng = np.random.default_rng(11)
    funding = pd.DataFrame(
        rng.normal(0, 1e-4, panels["close"].shape), index=panels["close"].index, columns=panels["close"].columns
    )
    eligible = panels["close"].notna()
    books = build_candidate_member_books(panels, funding, eligible, eligible, decisions)
    beta_grid = causal_market_beta(
        np.log(panels["close"]), eligible, CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS
    ).reindex(decisions)
    mask_grid = eligible.reindex(decisions).fillna(False)
    for lookback in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS:
        signal_grid = funding_carry_signal(funding, lookback).reindex(decisions)
        finite_signal = pd.DataFrame(
            np.isfinite(signal_grid.to_numpy(dtype="float64")),
            index=signal_grid.index,
            columns=list(signal_grid.columns),
        )
        validated = mask_grid & finite_signal
        expected = beta_neutralize_weights(
            rank_weight_book(signal_grid, validated, -1, PROCESS_MIN_SYMBOLS),
            beta_grid,
            validated,
            PROCESS_MIN_SYMBOLS,
        )
        pd.testing.assert_frame_equal(books[f"funding_carry_{lookback}h"], expected, check_exact=True)


def test_feature_preparation_does_not_mutate_sources() -> None:
    panels, decisions = _deterministic_panels()
    rng = np.random.default_rng(13)
    funding = pd.DataFrame(
        rng.normal(0, 1e-4, panels["close"].shape), index=panels["close"].index, columns=panels["close"].columns
    )
    eligible = panels["close"].notna()
    snapshots = {k: (v.index.copy(), list(v.columns), v.to_numpy(dtype="float64").copy()) for k, v in panels.items()}
    funding_snapshot = funding.to_numpy(dtype="float64").copy()
    mask_snapshot = eligible.to_numpy().copy()
    for name in PROCESS_FEATURE_CANDIDATES:
        build_process_feature_grid(_REGISTRY[name], panels, decisions, column_block_size=4)
    build_candidate_member_books(panels, funding, eligible, eligible, decisions)
    for key, (index, columns, values) in snapshots.items():
        assert panels[key].index.equals(index)
        assert list(panels[key].columns) == columns
        assert np.array_equal(panels[key].to_numpy(dtype="float64"), values, equal_nan=True)
    assert np.array_equal(funding.to_numpy(dtype="float64"), funding_snapshot, equal_nan=True)
    assert np.array_equal(eligible.to_numpy(), mask_snapshot)


def test_loader_releases_expired_buffers() -> None:

    n_bars = 800
    symbols = [f"W{i:02d}USDT" for i in range(6)]
    grid = pd.date_range("2021-01-01", periods=n_bars, freq="h", tz="UTC")
    rng = np.random.default_rng(21)
    close_vals = 100.0 + np.cumsum(rng.normal(0, 0.5, (n_bars, len(symbols))), axis=0)
    base = pd.DataFrame(close_vals, index=grid, columns=symbols)
    quote_vals = pd.DataFrame(rng.uniform(1e6, 2e6, (n_bars, len(symbols))), index=grid, columns=symbols)
    fake_panel = {
        "close": base,
        "open": base.copy(),
        "high": base + 0.3,
        "low": base - 0.3,
        "quote_vol": quote_vals,
        "taker_buy_quote": quote_vals * 0.5,
    }
    close_ref = weakref.ref(fake_panel["close"])
    quote_ref = weakref.ref(fake_panel["quote_vol"])
    funding = {s: pd.Series(rng.normal(0, 1e-5, n_bars), index=grid) for s in symbols}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(bt_market, "load_base_panel", lambda *a, _p=fake_panel, **k: _p)
    monkeypatch.setattr(bt_market, "_load_funding_series", lambda syms: ({s: funding[s] for s in syms}, {}))
    monkeypatch.setattr(bt_market, "apply_dynamic_gap_exclusion", lambda mask, *a, **k: (mask, {}))
    try:
        start = grid[0]
        end = grid[0] + pd.Timedelta(days=30)
        data = bt_market.load_process_market_data(start, end)
    finally:
        monkeypatch.undo()
    del fake_panel, base, quote_vals
    gc.collect()
    assert close_ref() is None
    assert quote_ref() is None
    assert data.opens_1h.shape[1] == len(symbols)
    assert data.bar_funding_1h.shape[1] == len(symbols)
    assert data.execution_mask.index.equals(data.decision_grid)


def test_process_feature_grid_rejects_corrupt_source_labels() -> None:
    panels, decisions = _deterministic_panels(n_bars=900, n_symbols=4)
    spec = _REGISTRY["mom_168h"]
    duplicated = {k: pd.concat([v, v.iloc[:5]]).sort_index() for k, v in panels.items()}
    full = dict(panels)
    full.update(duplicated)
    with pytest.raises(DataIntegrityError, match="unique"):
        build_process_feature_grid(spec, full, decisions)
    bad_labels = decisions.copy()
    bad_labels = bad_labels.insert(0, pd.NaT)
    with pytest.raises(DataIntegrityError, match="NaT"):
        build_process_feature_grid(spec, panels, bad_labels)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_process_feature_grid(spec, panels, list(decisions))  # type: ignore[arg-type]
