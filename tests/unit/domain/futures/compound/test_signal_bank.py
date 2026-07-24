from __future__ import annotations

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
    _compute_xs_rank_signal,
    _compute_xs_reversal,
    _default_catalog,
    _rolling_mad_z,
    build_raw_signal_panel,
)

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


def test_reversal_st_only_fast_speed() -> None:
    catalog = _default_catalog()
    rev_st = [d for d in catalog if d.family == "reversal_st"]
    assert len(rev_st) == 1
    assert rev_st[0].speed == "fast"
    assert rev_st[0].lookback_hours == 24

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


def test_default_catalog_xs_reversal_has_fast_and_medium() -> None:
    catalog = _default_catalog()
    xs = [d for d in catalog if d.family == "xs_reversal"]
    assert len(xs) == 2
    speeds = {d.speed for d in xs}
    assert speeds == {"fast", "medium"}


def test_default_catalog_flow_taker_excluded() -> None:
    catalog = _default_catalog()
    ft = [d for d in catalog if d.family == "flow_taker"]
    assert len(ft) == 0


def test_build_raw_signal_panel_default_catalog_has_25_signals_with_new_families(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert len(panel.descriptors) == 25
    families = {d.family for d in panel.descriptors}
    assert "xs_reversal" in families
    assert "xs_momentum_slow" in families
    assert "flow_taker" not in families
    xs = [d for d in panel.descriptors if d.family == "xs_reversal"]
    assert len(xs) == 2
    msm = [d for d in panel.descriptors if d.family == "xs_momentum_slow"]
    assert len(msm) == 2


def test_signal_bank_v4_default_catalog_matched_horizons() -> None:
    catalog = _default_catalog()
    assert len(catalog) == 25
    for desc in catalog:
        assert desc.target_horizon_hours > 0
        assert desc.target_horizon_hours == desc.lookback_hours, (
            f"{desc.signal_id}: target_horizon_hours={desc.target_horizon_hours} "
            f"!= lookback_hours={desc.lookback_hours}"
        )


def test_p2_pipeline_handles_updated_catalog_size_without_hardcoded_25(synthetic_market: MarketFeatureCube) -> None:
    bars = build_multi_timeframe_bars(synthetic_market)
    T = bars.decision_timestamps_ns.size
    N = len(bars.cubes["4h"].symbols)
    eligible = np.ones((T, N), np.bool_)
    panel = build_raw_signal_panel(bars, eligible_2d=eligible)
    assert isinstance(panel, RawSignalPanel)
    assert panel.z_3d.shape[-1] == 25
    assert panel.z_3d.dtype == np.float32
    finite_mask = np.isfinite(panel.z_3d)
    if np.any(finite_mask):
        assert np.all(panel.z_3d[finite_mask] >= -3.0)
        assert np.all(panel.z_3d[finite_mask] <= 3.0 + 1e-6)
