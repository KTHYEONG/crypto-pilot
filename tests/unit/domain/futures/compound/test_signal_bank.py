from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pytest

from src.domain.futures.compound.bar_engine import (
    aggregate_timeframe_bars,
    build_multi_timeframe_bars,
)
from src.domain.futures.compound.contracts import (
    CausalityError,
    InsufficientCoverageError,
    MarketFeatureCube,
    RawSignalPanel,
)
from src.domain.futures.compound.signal_bank import (
    _compute_basis_gap,
    _compute_flow_imbalance_taker,
    _compute_funding_carry_reversion,
    _compute_open_interest_confirmation,
    _compute_raw_signal,
    _compute_smart_money_divergence,
    _compute_volatility_squeeze_keltner,
    _compute_xs_rank_signal,
    _compute_xs_reversal,
    _default_catalog,
    _rolling_mad_z,
    _rolling_mad_z_numba_kernel,
    _rolling_mad_z_numpy,
    _rolling_mad_z_single_sort_kernel,
    build_raw_signal_panel,
    estimate_signal_panel_peak_bytes,
)

if TYPE_CHECKING:
    pass

HOUR_NS = 3_600_000_000_000


def _make_market(
    n_bars: int, n_syms: int, close: NDArray[np.float32] | None = None,
) -> MarketFeatureCube:
    if close is None:
        t = np.arange(n_bars, dtype=np.float64)[:, None]
        close = (100.0 + 10.0 * np.sin(t / 96.0) + 0.01 * t).astype(np.float32)
        close = np.repeat(close, n_syms, axis=1)
    ts = np.arange(n_bars, dtype=np.int64) * HOUR_NS + 1_700_000_000_000_000_000
    return MarketFeatureCube(
        timestamps_ns=ts,
        symbols=tuple(f"SYM{i}USDT" for i in range(n_syms)),
        fields_2d={
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_volume": np.full((n_bars, n_syms), 1e6, np.float32),
            "funding": np.full((n_bars, n_syms), 1e-4, np.float32),
            "premium": np.zeros((n_bars, n_syms), np.float32),
            "mark": close,
            "index": close,
            "taker_buy_quote": np.full((n_bars, n_syms), 6e5, np.float32),
        },
        available_2d={k: np.ones((n_bars, n_syms), np.bool_) for k in
                      ("open", "high", "low", "close", "quote_volume", "funding",
                       "premium", "mark", "index", "taker_buy_quote")},
        eligible_2d=np.ones((n_bars, n_syms), np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, np.float32),
        data_manifest_hash="test",
    )


@pytest.fixture
def synthetic_market() -> MarketFeatureCube:
    return _make_market(24 * 90, 3)


def test_aggregate_4h_ohlc_exact(synthetic_market: MarketFeatureCube) -> None:
    cube = aggregate_timeframe_bars(synthetic_market, "4h")
    o = synthetic_market.fields_2d["open"]
    h = synthetic_market.fields_2d["high"]
    l = synthetic_market.fields_2d["low"]
    c = synthetic_market.fields_2d["close"]

    n_4h = 24 * 90 // 4
    assert cube.timestamps_ns.shape[0] == n_4h
    assert cube.open_2d.shape == (n_4h, 3)
    assert cube.high_2d.shape == (n_4h, 3)
    assert cube.low_2d.shape == (n_4h, 3)
    assert cube.close_2d.shape == (n_4h, 3)
    assert cube.quote_volume_2d.shape == (n_4h, 3)
    assert cube.complete_2d.shape == (n_4h, 3)

    np.testing.assert_allclose(cube.open_2d[0], o[0], rtol=1e-6)
    np.testing.assert_allclose(cube.close_2d[0], c[3], rtol=1e-6)
    np.testing.assert_allclose(cube.high_2d[0], np.max(h[:4], axis=0), rtol=1e-6)
    np.testing.assert_allclose(cube.low_2d[0], np.min(l[:4], axis=0), rtol=1e-6)
    np.testing.assert_allclose(cube.quote_volume_2d[0], 4e6, rtol=1e-6)

    assert bool(cube.complete_2d[0, 0]) is True

    ts_expected = synthetic_market.timestamps_ns[3::4][:n_4h]
    np.testing.assert_array_equal(cube.timestamps_ns, ts_expected)


def test_aggregate_1d_ohlc_exact(synthetic_market: MarketFeatureCube) -> None:
    cube = aggregate_timeframe_bars(synthetic_market, "1d")
    o = synthetic_market.fields_2d["open"]
    h = synthetic_market.fields_2d["high"]
    l = synthetic_market.fields_2d["low"]
    c = synthetic_market.fields_2d["close"]

    n_1d = 24 * 90 // 24
    assert cube.timestamps_ns.shape[0] == n_1d

    np.testing.assert_allclose(cube.open_2d[0], o[0], rtol=1e-6)
    np.testing.assert_allclose(cube.close_2d[0], c[23], rtol=1e-6)
    np.testing.assert_allclose(cube.high_2d[0], np.max(h[:24], axis=0), rtol=1e-6)
    np.testing.assert_allclose(cube.low_2d[0], np.min(l[:24], axis=0), rtol=1e-6)

    ts_expected = synthetic_market.timestamps_ns[23::24][:n_1d]
    np.testing.assert_array_equal(cube.timestamps_ns, ts_expected)


def test_incomplete_bar_marks_invalid_and_no_lookahead(synthetic_market: MarketFeatureCube) -> None:
    close = synthetic_market.fields_2d["close"].copy()
    nan_bar = 24 * 30 + 2
    close[nan_bar, :] = np.nan
    tampered = _make_market(
        n_bars=24 * 90, n_syms=3, close=close,
    )

    bars = build_multi_timeframe_bars(tampered)
    cube_4h = bars.cubes["4h"]

    bad_4h_idx = nan_bar // 4
    assert not bool(cube_4h.complete_2d[bad_4h_idx, 0])

    panel = build_raw_signal_panel(
        bars,
        eligible_2d=np.ones((bars.decision_timestamps_ns.size, 3), np.bool_),
    )
    assert not bool(panel.valid_3d[bad_4h_idx, 0, 0])

    ts_original = synthetic_market.timestamps_ns.copy()
    ts_original[-1] = ts_original[0] - HOUR_NS
    bad_market = MarketFeatureCube(
        timestamps_ns=ts_original,
        symbols=synthetic_market.symbols,
        fields_2d=synthetic_market.fields_2d,
        available_2d=synthetic_market.available_2d,
        eligible_2d=synthetic_market.eligible_2d,
        entry_block_2d=synthetic_market.entry_block_2d,
        exit_required_2d=synthetic_market.exit_required_2d,
        capacity_usdt_2d=synthetic_market.capacity_usdt_2d,
        execution_cost_bps_2d=synthetic_market.execution_cost_bps_2d,
        data_manifest_hash="test",
    )
    with pytest.raises(CausalityError, match="monotonic"):
        aggregate_timeframe_bars(bad_market, "4h")


def test_lookahead_immunity(synthetic_market: MarketFeatureCube) -> None:
    bars_a = build_multi_timeframe_bars(synthetic_market)
    eligible = np.ones((bars_a.decision_timestamps_ns.size, 3), np.bool_)
    panel_a = build_raw_signal_panel(bars_a, eligible_2d=eligible)

    c = synthetic_market.fields_2d["close"].copy()
    tamper_start = 24 * 80
    c[tamper_start:] *= 1000.0
    tampered_market = _make_market(n_bars=24 * 90, n_syms=3, close=c)

    bars_b = build_multi_timeframe_bars(tampered_market)
    panel_b = build_raw_signal_panel(bars_b, eligible_2d=eligible)

    cut = tamper_start // 4 - 2
    np.testing.assert_allclose(panel_a.z_3d[:cut], panel_b.z_3d[:cut], atol=1e-5)


def test_rolling_mad_z_differs_from_global(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    cube_4h = bars.cubes["4h"]
    raw = np.full((cube_4h.close_2d.shape[0], 3), np.nan, dtype=np.float64)
    raw[180:] = np.random.default_rng(42).normal(0, 1, (cube_4h.close_2d.shape[0] - 180, 3))

    rolling_z = _rolling_mad_z(raw, window=180, min_periods=90)
    global_med = np.nanmedian(raw, axis=0, keepdims=True)
    global_mad = np.nanmedian(np.abs(raw - global_med), axis=0, keepdims=True)
    global_mad = np.where(global_mad < 1e-12, 1e-12, global_mad)
    global_z = (raw - global_med) / (1.4826 * global_mad)

    non_nan_mask = (~np.isnan(rolling_z)) & (~np.isnan(global_z))
    assert non_nan_mask.any()
    diff = np.abs(rolling_z[non_nan_mask] - global_z[non_nan_mask])
    assert np.any(diff > 0.01)


def test_rolling_mad_numba_matches_numpy_exactly() -> None:
    rng = np.random.default_rng(42)
    n_t, n_s = 438, 5
    window, min_per = 120, 60

    arr = rng.standard_normal((n_t, n_s)).astype(np.float64)
    numba_z = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window, min_per)
    numpy_z = _rolling_mad_z_numpy(arr, window, min_per)
    mask = np.isfinite(numpy_z)
    assert np.allclose(numba_z[mask], numpy_z[mask], atol=1e-12)
    assert np.array_equal(np.isnan(numba_z), np.isnan(numpy_z))

    arr_nan = arr.copy()
    arr_nan[50:80, 2] = np.nan
    nz = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr_nan), window, min_per)
    npz = _rolling_mad_z_numpy(arr_nan, window, min_per)
    m = np.isfinite(npz)
    assert np.allclose(nz[m], npz[m], atol=1e-12)
    assert np.array_equal(np.isnan(nz), np.isnan(npz))

    const = np.full((n_t, n_s), 42.0, dtype=np.float64)
    const_z = _rolling_mad_z_numba_kernel(np.ascontiguousarray(const), window, min_per)
    assert np.all(np.isnan(const_z))

    future = arr.copy()
    future[400:] += 999.0
    fz = _rolling_mad_z_numba_kernel(np.ascontiguousarray(future), window, min_per)
    fpz = _rolling_mad_z_numpy(future, window, min_per)
    pre = 395
    assert np.allclose(fz[:pre], fpz[:pre], atol=1e-12, equal_nan=True)

    wrapper_z = _rolling_mad_z(arr, window, min_per)
    assert np.allclose(wrapper_z, numpy_z, atol=1e-12, equal_nan=True)

    wrap_nan = _rolling_mad_z(arr_nan, window, min_per)
    assert np.allclose(wrap_nan, npz, atol=1e-12, equal_nan=True)


def test_non_monotonic_timestamps_raises_causality_error() -> None:
    ts = np.array([3, 1, 2], dtype=np.int64) * HOUR_NS
    market = MarketFeatureCube(
        timestamps_ns=ts,
        symbols=("A",),
        fields_2d={"close": np.array([[1.0], [2.0], [3.0]], dtype=np.float32)},
        available_2d={"close": np.ones((3, 1), np.bool_)},
        eligible_2d=np.ones((3, 1), np.bool_),
        entry_block_2d=np.zeros((3, 1), np.bool_),
        exit_required_2d=np.zeros((3, 1), np.bool_),
        capacity_usdt_2d=np.full((3, 1), 1e6, np.float64),
        execution_cost_bps_2d=np.full((3, 1), 12.0, np.float32),
        data_manifest_hash="test",
    )
    with pytest.raises(CausalityError, match="monotonic"):
        build_multi_timeframe_bars(market)


def test_insufficient_coverage_raises_error() -> None:
    market = _make_market(24 * 3, 2)
    with pytest.raises(InsufficientCoverageError):
        build_multi_timeframe_bars(market)


def test_bar_engine_to_signal_bank_pipeline_shapes_and_masks(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)

    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)

    assert isinstance(panel, RawSignalPanel)
    assert panel.decision_timestamps_ns.shape == (T,)
    assert len(panel.symbols) == N
    catalog = _default_catalog()
    assert len(panel.descriptors) == len(catalog)
    K = len(catalog)
    assert panel.z_3d.shape == (T, N, K)
    assert panel.valid_3d.shape == (T, N, K)
    assert panel.sigma_2d.shape == (T, N)

    assert panel.z_3d.dtype == np.float32
    finite_mask = np.isfinite(panel.z_3d)
    if np.any(finite_mask):
        assert np.all(panel.z_3d[finite_mask] >= -3.0)
        assert np.all(panel.z_3d[finite_mask] <= 3.0 + 1e-6)

    valid_implies_eligible = np.all(panel.valid_3d <= eligible[:, :, None])
    assert valid_implies_eligible

    desc_ids = tuple(d.signal_id for d in panel.descriptors)
    expected_ids = tuple(d.signal_id for d in catalog)
    assert desc_ids == expected_ids

    assert np.all(np.isfinite(panel.sigma_2d))
    assert np.all(panel.sigma_2d > 0)


def test_trend_ema_fast_positive_in_uptrend() -> None:
    n_bars = 24 * 90
    t = np.arange(n_bars, dtype=np.float64)[:, None]
    uptrend_close = (100.0 + 0.05 * t).astype(np.float32)
    uptrend_close = np.repeat(uptrend_close, 2, axis=1)
    market = _make_market(n_bars, 2, close=uptrend_close)

    bars = build_multi_timeframe_bars(market)
    T = bars.decision_timestamps_ns.size
    eligible = np.ones((T, 2), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)

    trend_ema_fast_idx = next(i for i, d in enumerate(panel.descriptors)
                              if d.signal_id == "trend_ema:fast")
    late_scores = panel.z_3d[-20:, :, trend_ema_fast_idx]
    assert np.mean(late_scores) > 0.1


def test_reversal_st_has_eight_speeds() -> None:
    catalog = _default_catalog()
    rev_st = [d for d in catalog if d.family == "reversal_st"]
    assert len(rev_st) == 8
    speeds = [d.speed for d in rev_st]
    assert "fast" in speeds
    assert "extreme_slow" in speeds

def test_compute_basis_gap_speed_dependent_smoothing() -> None:
    n_t, n_s = 1000, 5
    mark = np.ones((n_t, n_s), dtype=np.float32)
    index_arr = np.ones((n_t, n_s), dtype=np.float32)
    switch = n_t // 2
    mark[switch:] = 1.01
    fast = _compute_basis_gap(mark, index_arr, 24)
    very_slow = _compute_basis_gap(mark, index_arr, 648)
    assert not np.allclose(fast, very_slow, equal_nan=True)


def test_compute_xs_reversal_ranks_top_and_bottom_performers() -> None:
    n_t, n_s = 10, 12
    close = np.zeros((n_t, n_s), dtype=np.float32)
    close[0] = 100.0
    vals = np.array([101.0 + i for i in range(n_s)], dtype=np.float32)
    close[1] = vals
    eligible = np.ones((n_t, n_s), dtype=np.bool_)
    result = _compute_xs_reversal(close, 1, eligible)
    row = result[1]
    assert row[0] > row[-1]
    assert row[0] > 0
    assert row[-1] < 0


def test_compute_xs_rank_signal_sign_flip_is_exact_negation() -> None:
    n_t, n_s = 10, 12
    close = np.zeros((n_t, n_s), dtype=np.float32)
    close[0] = 100.0
    vals = np.array([101.0 + i for i in range(n_s)], dtype=np.float32)
    close[1] = vals
    eligible = np.ones((n_t, n_s), dtype=np.bool_)
    rev = _compute_xs_rank_signal(close, 1, eligible, sign=-1.0)
    mom = _compute_xs_rank_signal(close, 1, eligible, sign=+1.0)
    assert np.allclose(rev, -mom, equal_nan=True)


def test_compute_xs_reversal_below_min_cross_section_is_nan() -> None:
    n_t, n_s = 10, 9
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal((n_t, n_s)).cumsum(axis=0).astype(np.float32)
    eligible = np.ones((n_t, n_s), dtype=np.bool_)
    result = _compute_xs_reversal(close, 1, eligible)
    assert np.all(np.isnan(result))


def test_compute_xs_reversal_exactly_10_eligible() -> None:
    n_t, n_s = 10, 11
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal((n_t, n_s)).cumsum(axis=0).astype(np.float32)
    eligible = np.ones((n_t, n_s), dtype=np.bool_)
    eligible[:, -1] = False
    result = _compute_xs_reversal(close, 1, eligible)
    assert np.any(np.isfinite(result[1]))


def test_default_catalog_has_eight_families_sixty_recipes() -> None:
    catalog = _default_catalog()
    assert len(catalog) == 60
    families = {d.family for d in catalog}
    assert families == {
        "trend_ema", "momentum_ts", "breakout_donchian", "reversal_st",
        "funding_carry_reversion", "flow_imbalance_taker",
        "volatility_squeeze_keltner", "open_interest_confirmation",
    }


def test_default_catalog_flow_imbalance_taker_included() -> None:
    catalog = _default_catalog()
    ft = [d for d in catalog if d.family == "flow_imbalance_taker"]
    assert len(ft) == 8
    ft_old = [d for d in catalog if d.family == "flow_taker"]
    assert len(ft_old) == 0


def test_build_raw_signal_panel_default_catalog_has_60_signals_with_eight_families(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert len(panel.descriptors) == 60
    families = {d.family for d in panel.descriptors}
    assert families == {
        "trend_ema", "momentum_ts", "breakout_donchian", "reversal_st",
        "funding_carry_reversion", "flow_imbalance_taker",
        "volatility_squeeze_keltner", "open_interest_confirmation",
    }
    trend = [d for d in panel.descriptors if d.family == "trend_ema"]
    assert len(trend) == 8
    rev_st = [d for d in panel.descriptors if d.family == "reversal_st"]
    assert len(rev_st) == 8
    vol_sq = [d for d in panel.descriptors if d.family == "volatility_squeeze_keltner"]
    assert len(vol_sq) == 6
    oi = [d for d in panel.descriptors if d.family == "open_interest_confirmation"]
    assert len(oi) == 6


def test_default_catalog_60_recipes_matched_horizons() -> None:
    catalog = _default_catalog()
    assert len(catalog) == 60
    for desc in catalog:
        assert desc.target_horizon_hours > 0
        assert desc.target_horizon_hours == desc.lookback_hours, (
            f"{desc.signal_id}: target_horizon_hours={desc.target_horizon_hours} "
            f"!= lookback_hours={desc.lookback_hours}"
        )


def test_compute_smart_money_divergence_sign_and_nan_handling() -> None:
    n_t, n_s = 200, 3
    top_trader = np.full((n_t, n_s), 2.0, dtype=np.float32)
    retail = np.full((n_t, n_s), 1.0, dtype=np.float32)
    result = _compute_smart_money_divergence(top_trader, retail, 24)
    assert result.shape == (n_t, n_s)
    valid = result[42:]
    assert np.all(valid < 0), "contrarian sign: top_trader > retail -> negative"

    top_trader_invalid = top_trader.copy()
    top_trader_invalid[50] = -1.0
    top_trader_invalid[60] = 0.0
    top_trader_invalid[70, 1] = np.nan
    _compute_smart_money_divergence(top_trader_invalid, retail, 24)


def test_build_raw_signal_panel_funding_carry_reversion_wiring() -> None:
    n_bars = 24 * 90
    t = np.arange(n_bars, dtype=np.float64)[:, None]
    close = (100.0 + 10.0 * np.sin(t / 96.0) + 0.01 * t).astype(np.float32)
    close = np.repeat(close, 3, axis=1)
    ts = np.arange(n_bars, dtype=np.int64) * HOUR_NS + 1_700_000_000_000_000_000

    rng = np.random.default_rng(42)
    n_syms = 3
    funding = 2e-4 * rng.random((n_bars, n_syms)).astype(np.float32)

    market = MarketFeatureCube(
        timestamps_ns=ts,
        symbols=tuple(f"SYM{i}USDT" for i in range(n_syms)),
        fields_2d={
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_volume": np.full((n_bars, n_syms), 1e6, np.float32),
            "funding": funding,
            "premium": np.zeros((n_bars, n_syms), np.float32),
            "mark": close,
            "index": close,
            "taker_buy_quote": np.full((n_bars, n_syms), 6e5, np.float32),
        },
        available_2d={k: np.ones((n_bars, n_syms), np.bool_) for k in
                      ("open", "high", "low", "close", "quote_volume", "funding",
                       "premium", "mark", "index", "taker_buy_quote")},
        eligible_2d=np.ones((n_bars, n_syms), np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, np.float32),
        data_manifest_hash="test",
    )

    bars = build_multi_timeframe_bars(market)
    T = bars.decision_timestamps_ns.size
    eligible = np.ones((T, n_syms), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)

    fcr_fast_idx = next(i for i, d in enumerate(panel.descriptors)
                        if d.signal_id == "funding_carry_reversion:fast")
    fcr_medium_idx = next(i for i, d in enumerate(panel.descriptors)
                          if d.signal_id == "funding_carry_reversion:medium")
    assert np.any(panel.valid_3d[:, :, fcr_fast_idx])
    assert np.any(panel.valid_3d[:, :, fcr_medium_idx])
    fast_scores = panel.z_3d[:, :, fcr_fast_idx]
    medium_scores = panel.z_3d[:, :, fcr_medium_idx]
    assert np.any(np.isfinite(fast_scores))
    assert np.any(np.isfinite(medium_scores))
    assert panel.descriptors[fcr_fast_idx].native_timeframe == "1h"
    assert panel.descriptors[fcr_medium_idx].native_timeframe == "1h"


def test_p2_pipeline_60_signals_120_symbols_supported(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert isinstance(panel, RawSignalPanel)
    assert panel.z_3d.shape[-1] == 60
    assert panel.z_3d.dtype == np.float32
    finite_mask = np.isfinite(panel.z_3d)
    if np.any(finite_mask):
        assert np.all(panel.z_3d[finite_mask] >= -3.0)
        assert np.all(panel.z_3d[finite_mask] <= 3.0 + 1e-6)


def test_signal_panel_numba_identity_over_two_calls(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    a = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=1)
    b = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=1)

    assert a.descriptors == b.descriptors
    finite = np.isfinite(a.z_3d) & np.isfinite(b.z_3d)
    if np.any(finite):
        np.testing.assert_array_equal(a.z_3d[finite], b.z_3d[finite])
    np.testing.assert_array_equal(np.isnan(a.z_3d), np.isnan(b.z_3d))
    np.testing.assert_array_equal(a.valid_3d, b.valid_3d)
    np.testing.assert_array_equal(a.sigma_2d, b.sigma_2d)


def test_signal_panel_numba_thread_bounds(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    with pytest.raises(ValueError, match="numba_threads"):
        build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=0)
    with pytest.raises(ValueError, match="numba_threads"):
        build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=7)

    panel = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=1)
    assert isinstance(panel, RawSignalPanel)


def test_recipe_failure_is_isolated(synthetic_market: MarketFeatureCube, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)

    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    catalog = _default_catalog()
    failing_id = catalog[0].signal_id

    def failing_raw_signal(desc, bars_, eligible_):
        if desc.signal_id == failing_id:
            import logging as _lg
            _lg.getLogger("src.domain.futures.compound.signal_bank").error("[DATA] injected failure signal_id=%s", desc.signal_id)
            return None
        return _compute_raw_signal(desc, bars_, eligible_)

    import src.domain.futures.compound.signal_bank as sb_mod
    orig = sb_mod._compute_raw_signal
    sb_mod._compute_raw_signal = failing_raw_signal
    try:
        panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    finally:
        sb_mod._compute_raw_signal = orig

    assert np.all(np.isnan(panel.z_3d[:, :, 0]))
    assert not np.any(panel.valid_3d[:, :, 0])
    other_valid = np.any(panel.valid_3d[:, :, 1:])
    assert other_valid
    assert failing_id in caplog.text


def test_engine_invokes_signal_panel_and_logs(synthetic_market: MarketFeatureCube, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert isinstance(panel, RawSignalPanel)
    assert panel.descriptors == _default_catalog()
    assert "[PERF][L1] signal_panel elapsed_s=" in caplog.text
    assert "numba_threads=" in caplog.text
    assert "numba_fallbacks=" in caplog.text


def test_signal_panel_resource_bounds_and_restoration(synthetic_market: MarketFeatureCube) -> None:
    from numba import get_num_threads

    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    prior = get_num_threads()
    _ = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=1)
    assert get_num_threads() == prior

    estimate = estimate_signal_panel_peak_bytes(
        current_rss_bytes=500_000_000,
        n_bars=4380,
        n_symbols=120,
        n_recipes=27,
        max_native_rows=17520,
        numba_threads=4,
    )
    assert estimate > 500_000_000

    base = {
        "current_rss_bytes": 500_000_000,
        "n_bars": 4380, "n_symbols": 120, "n_recipes": 27,
        "max_native_rows": 17520, "numba_threads": 4,
    }
    monotonic_keys = ("n_recipes", "n_symbols", "n_bars", "max_native_rows", "numba_threads")
    for key in monotonic_keys:
        overrides = base.copy()
        overrides[key] = overrides[key] * 2
        higher = estimate_signal_panel_peak_bytes(**overrides)
        assert higher >= estimate, f"monotonic failed for {key}"

    with pytest.raises(ValueError, match="positive"):
        estimate_signal_panel_peak_bytes(
            current_rss_bytes=500_000_000,
            n_bars=4380, n_symbols=120, n_recipes=27,
            max_native_rows=17520, numba_threads=0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        estimate_signal_panel_peak_bytes(
            current_rss_bytes=-1,
            n_bars=4380, n_symbols=120, n_recipes=27,
            max_native_rows=17520, numba_threads=4,
        )
    with pytest.raises(ValueError, match="non-negative"):
        estimate_signal_panel_peak_bytes(
            current_rss_bytes=500_000_000,
            n_bars=-1, n_symbols=120, n_recipes=27,
            max_native_rows=17520, numba_threads=4,
        )


def test_signal_panel_fallback_and_memory_guards_fail_closed(
    synthetic_market: MarketFeatureCube, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.domain.futures.compound.signal_bank as sb_mod

    def fail_kernel(*_args: object) -> NDArray[np.float64]:
        raise RuntimeError("injected numba failure")

    monkeypatch.setattr(sb_mod, "_rolling_mad_z_numba_kernel", fail_kernel)
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert isinstance(panel, RawSignalPanel)

    with pytest.raises(MemoryError, match="preflight"):
        build_raw_signal_panel(bars, eligible_2d=eligible, max_rss_mb=1)

    call_count: list[int] = [0]

    class MockProcess:
        @staticmethod
        def memory_info() -> object:
            call_count[0] += 1
            if call_count[0] == 1:
                return type("MI", (), {"rss": 500_000_000})()
            return type("MI", (), {"rss": 100 * 1024 * 1024 * 1024})()

    monkeypatch.setattr(sb_mod.psutil, "Process", lambda: MockProcess())
    with pytest.raises(MemoryError, match="runtime RSS"):
        build_raw_signal_panel(bars, eligible_2d=eligible, max_rss_mb=12_000)


def test_engine_wires_guarded_six_thread_l1_panel(synthetic_market: MarketFeatureCube) -> None:

    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    panel = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=6, max_rss_mb=12_000)
    assert isinstance(panel, RawSignalPanel)
    assert panel.z_3d.shape[0] == T
    assert panel.z_3d.shape[1] == N
    assert panel.z_3d.shape[2] == 60
    assert panel.valid_3d.shape == (T, N, 60)
    assert panel.sigma_2d.shape == (T, N)
    finite_sigma = np.isfinite(panel.sigma_2d)
    assert np.all(finite_sigma)
    assert np.all(panel.sigma_2d > 0)
    assert panel.sigma_2d.dtype == np.float32


def test_volatility_squeeze_keltner_tight_range_gives_negative_squeeze() -> None:
    n_t, n_s = 400, 3
    rng = np.random.default_rng(42)
    close = 100.0 + 0.1 * rng.standard_normal((n_t, n_s)).cumsum(axis=0)
    high = close + np.abs(rng.normal(0, 0.2, (n_t, n_s)))
    low = close - np.abs(rng.normal(0, 0.2, (n_t, n_s)))
    close = close.astype(np.float32)
    high = high.astype(np.float32)
    low = low.astype(np.float32)
    result = _compute_volatility_squeeze_keltner(high, low, close, 20)
    assert result.shape == (n_t, n_s)
    assert np.any(np.isfinite(result))


def test_funding_carry_reversion_high_funding_gives_negative_signal() -> None:
    n_t, n_s = 400, 3
    rng = np.random.default_rng(42)
    base = 0.0001
    funding = np.full((n_t, n_s), base, dtype=np.float32)
    funding[:n_t // 2] = base + 2e-5 * rng.standard_normal((n_t // 2, n_s)).astype(np.float32)
    funding[n_t // 2:] = 3 * base + 2e-5 * rng.standard_normal((n_t - n_t // 2, n_s)).astype(np.float32)
    premium = np.zeros((n_t, n_s), dtype=np.float32)
    result = _compute_funding_carry_reversion(funding, premium, 24)
    assert result.shape == (n_t, n_s)
    valid = result[200:]
    assert np.any(np.isfinite(valid))


def test_flow_imbalance_taker_buy_pressure_gives_positive_signal() -> None:
    n_t, n_s = 400, 3
    rng = np.random.default_rng(42)
    taker_buy = (500_000 + 400_000 * rng.random((n_t, n_s))).astype(np.float32)
    volume = np.full((n_t, n_s), 1_000_000, dtype=np.float32)
    result = _compute_flow_imbalance_taker(taker_buy, volume, 24)
    assert result.shape == (n_t, n_s)
    valid = result[200:]
    assert np.any(np.isfinite(valid))


def test_open_interest_confirmation_rising_oi_gives_positive_signal() -> None:
    n_t, n_s = 400, 3
    rng = np.random.default_rng(42)
    ramp = 1000.0 + np.arange(n_t, dtype=np.float32) * 0.5
    oi = np.column_stack([ramp + rng.normal(0, 1.0, n_t) for _ in range(n_s)]).astype(np.float32)
    volume = np.full((n_t, n_s), 1_000_000, dtype=np.float32)
    result = _compute_open_interest_confirmation(oi, volume, 24)
    assert result.shape == (n_t, n_s)
    valid = result[200:]
    assert np.any(np.isfinite(valid))


def test_open_interest_confirmation_missing_data_does_not_crash_pipeline(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    oi_sigs = [i for i, d in enumerate(panel.descriptors) if d.family == "open_interest_confirmation"]
    for idx in oi_sigs:
        assert not np.any(panel.valid_3d[:, :, idx]), "open_interest not in test data, should be all invalid"


def test_default_catalog_60_recipes_unique_ids() -> None:
    catalog = _default_catalog()
    assert len(catalog) == 60
    ids = {d.signal_id for d in catalog}
    assert len(ids) == 60, "all signal_ids must be unique"


def test_default_catalog_all_speeds_have_expected_signal_ids() -> None:
    catalog = _default_catalog()
    expected = set()
    for family in ("trend_ema", "momentum_ts", "breakout_donchian", "reversal_st", "funding_carry_reversion", "flow_imbalance_taker"):
        for speed in ("fast", "medium", "moderate", "slow", "very_slow", "ultra_slow", "super_slow", "extreme_slow"):
            expected.add(f"{family}:{speed}")
    for family in ("volatility_squeeze_keltner", "open_interest_confirmation"):
        for speed in ("fast", "medium", "moderate", "slow", "very_slow", "ultra_slow"):
            expected.add(f"{family}:{speed}")
    actual = {d.signal_id for d in catalog}
    assert actual == expected


def test_build_raw_signal_panel_120_universe_symbols(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    symbols = bars.cubes["4h"].symbols
    eligible = np.ones((T, len(symbols)), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert panel.z_3d.shape[1] == len(symbols)
    assert panel.z_3d.shape[2] == 60
    assert len(panel.symbols) == len(symbols)


def test_signal_bank_dynamic_masking_coverage(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    eligible[T // 2:, :N // 2] = False
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert panel.valid_3d.shape == (T, N, 60)
    late_none = ~panel.valid_3d[T // 2:, :N // 2, :].any(axis=(0, 2))
    assert late_none.any(), "masked symbols should have no valid entries in later bars"


def test_deep_optimization_mad_zero_allocation_identity() -> None:
    rng = np.random.default_rng(42)
    n_t, n_s = 1000, 10
    window, min_per = 252, 126
    arr = rng.standard_normal((n_t, n_s)).astype(np.float64)
    arr[50:80, 2] = np.nan
    arr[200:250, 5] = np.nan

    numba_z = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window, min_per)
    numpy_z = _rolling_mad_z_numpy(arr, window, min_per)

    mask = np.isfinite(numpy_z)
    assert np.allclose(numba_z[mask], numpy_z[mask], atol=1e-12)
    assert np.array_equal(np.isnan(numba_z), np.isnan(numpy_z))

    const = np.full((n_t, n_s), 42.0, dtype=np.float64)
    const_z = _rolling_mad_z_numba_kernel(np.ascontiguousarray(const), window, min_per)
    assert np.all(np.isnan(const_z))


def test_signal_bank_thread_safety(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)

    panel = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=6)
    assert isinstance(panel, RawSignalPanel)
    assert panel.z_3d.shape == (T, N, 60)
    assert panel.valid_3d.shape == (T, N, 60)
    assert panel.sigma_2d.shape == (T, N)


def test_signal_bank_large_universe_tpe_off(synthetic_market: MarketFeatureCube) -> None:
    market_30 = _make_market(24 * 30, 30)
    bars = build_multi_timeframe_bars(market_30)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    assert N >= 30
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=6)
    assert isinstance(panel, RawSignalPanel)
    assert panel.z_3d.shape == (T, N, 60)
    assert panel.valid_3d.shape == (T, N, 60)


# ── H2-SINGLE-SORT-MERGE: bit-exact MAD-z kernel equivalence ──────────


def test_single_sort_vs_numba_bit_exact() -> None:
    rng = np.random.default_rng(42)
    n_t, n_s = 100, 5
    window, min_per = 20, 10
    arr = rng.standard_normal((n_t, n_s)).astype(np.float64)

    ref = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window, min_per)
    h2 = _rolling_mad_z_single_sort_kernel(np.ascontiguousarray(arr), window, min_per)

    mask = np.isfinite(ref) & np.isfinite(h2)
    assert np.allclose(ref[mask], h2[mask], atol=1e-15)
    assert np.array_equal(np.isnan(ref), np.isnan(h2))

    arr_nan = arr.copy()
    arr_nan[30:50, 2] = np.nan
    ref_n = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr_nan), window, min_per)
    h2_n = _rolling_mad_z_single_sort_kernel(np.ascontiguousarray(arr_nan), window, min_per)
    m = np.isfinite(ref_n) & np.isfinite(h2_n)
    assert np.allclose(ref_n[m], h2_n[m], atol=1e-15)
    assert np.array_equal(np.isnan(ref_n), np.isnan(h2_n))

    const = np.full((n_t, n_s), 42.0, dtype=np.float64)
    cz = _rolling_mad_z_single_sort_kernel(np.ascontiguousarray(const), window, min_per)
    assert np.all(np.isnan(cz))


def test_single_sort_small_window_manual() -> None:
    arr = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]], dtype=np.float64).T
    window, min_per = 3, 2
    h2 = _rolling_mad_z_single_sort_kernel(np.ascontiguousarray(arr), window, min_per)
    ref = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window, min_per)
    assert np.allclose(ref, h2, atol=1e-15, equal_nan=True)
    mask = np.isfinite(ref) & np.isfinite(h2)
    assert np.all(np.abs(ref[mask] - h2[mask]) < 1e-15)


def test_single_sort_all_nan() -> None:
    arr = np.full((50, 10), np.nan, dtype=np.float64)
    h2 = _rolling_mad_z_single_sort_kernel(arr, 20, 10)
    assert np.all(np.isnan(h2))


def test_single_sort_single_valid() -> None:
    arr = np.full((50, 10), np.nan, dtype=np.float64)
    arr[5, 2] = 42.0
    h2 = _rolling_mad_z_single_sort_kernel(arr, 20, 10)
    assert np.all(np.isnan(h2))


def test_single_sort_numerical_precision() -> None:
    rng = np.random.default_rng(42)
    n_t, n_s = 200, 8
    window, min_per = 60, 30
    arr = rng.standard_normal((n_t, n_s)).astype(np.float64)
    h2 = _rolling_mad_z_single_sort_kernel(np.ascontiguousarray(arr), window, min_per)
    ref = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window, min_per)
    mask = np.isfinite(ref) & np.isfinite(h2)
    assert np.allclose(ref[mask], h2[mask], atol=1e-14)


def test_single_sort_1h_data() -> None:
    rng = np.random.default_rng(42)
    n_t, n_s = 21768, 51
    arr = rng.standard_normal((n_t, n_s)).astype(np.float64)
    arr[rng.random((n_t, n_s)) < 0.03] = np.nan
    window, min_per = 540, 180
    h2 = _rolling_mad_z_single_sort_kernel(np.ascontiguousarray(arr), window, min_per)
    ref = _rolling_mad_z_numba_kernel(np.ascontiguousarray(arr), window, min_per)
    mask = np.isfinite(ref) & np.isfinite(h2)
    assert np.allclose(ref[mask], h2[mask], atol=1e-14)
    assert np.array_equal(np.isnan(ref), np.isnan(h2))


def test_signal_panel_output_identity(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible, numba_threads=1)
    assert isinstance(panel, RawSignalPanel)
    assert panel.z_3d.shape == (T, N, 60)
    assert panel.valid_3d.shape == (T, N, 60)
    finite = np.isfinite(panel.z_3d)
    if np.any(finite):
        assert np.all(panel.z_3d[finite] >= -3.0)
        assert np.all(panel.z_3d[finite] <= 3.0 + 1e-6)
