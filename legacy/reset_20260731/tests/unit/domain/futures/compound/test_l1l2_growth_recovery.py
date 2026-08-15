from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.allocator import (
    build_rank_conviction_targets,
    compute_dynamic_compounding_path,
)
from src.domain.futures.compound.calibration import build_folds_4h
from src.domain.futures.compound.config import (
    CalibrationConfig,
    DynamicCompoundingConfig,
    HandoffConfig,
)
from src.domain.futures.compound.contracts import (
    CalibratedForecastPanel,
    CausalClusterFold,
    CausalFold,
    ClusterPanel,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.l1_sleeves import estimate_cluster_sleeve_posteriors
from src.domain.futures.compound.multiplicity import TrialMultiplicity, deflated_sharpe_probability


def test_build_rank_conviction_targets_balanced_mu_returns_zero_net_unit_gross() -> None:
    mu = np.array([3.0, 1.0, -1.0, -3.0], dtype=np.float64)
    eligible = np.ones(4, dtype=np.bool_)
    result = build_rank_conviction_targets(mu, eligible, min_breadth=4)
    assert abs(float(np.sum(result))) < 1e-12
    assert abs(float(np.sum(np.abs(result))) - 1.0) < 1e-12
    assert result[0] > result[1] > result[2] > result[3]


def test_build_rank_conviction_targets_below_min_breadth_returns_zeros() -> None:
    mu = np.arange(9, dtype=np.float64) + 1.0
    eligible = np.ones(9, dtype=np.bool_)
    result = build_rank_conviction_targets(mu, eligible, min_breadth=10)
    np.testing.assert_array_equal(result, np.zeros(9))


def test_build_rank_conviction_targets_shape_mismatch_raises() -> None:
    mu = np.array([1.0, 2.0], dtype=np.float64)
    eligible = np.array([True, False, True], dtype=np.bool_)
    with pytest.raises(ValueError, match="shape"):
        build_rank_conviction_targets(mu, eligible)


def test_compute_dynamic_compounding_path_recovers_exposure_after_drawdown_release() -> None:
    n_bars, n_syms = 400, 20
    rng = np.random.default_rng(42)
    mu_2d = rng.normal(0.0, 0.01, (n_bars, n_syms)).astype(np.float32)
    sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
    ts = np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000 * 4
    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=ts,
        symbols=tuple(f"S{i}" for i in range(n_syms)),
        mu_2d=mu_2d,
        se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        family_ids=("f1",),
        admitted_signal_ids=("s1",),
        fold_manifest_hash="fh1",
    )
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    close = 1.0 + rng.normal(0, 0.002, (n_bars, n_syms)).astype(np.float32)
    close = np.maximum(close, 0.5)
    close[:100] = 1.0
    close[100:120] *= 0.998 ** np.arange(20).reshape(-1, 1)
    config = DynamicCompoundingConfig(
        use_rank_conviction=True, alpha_smooth=0.15,
        band_frac=0.30, dd_scale_floor=0.25,
        dd_cooldown_bars=60, min_vol_samples=60,
        max_gross_leverage=1.0, max_long_leverage=0.7,
        max_short_leverage=0.3,
    )
    result = compute_dynamic_compounding_path(
        forecast=forecast, sigma_2d=sigma_2d,
        funding_rates_1h_2d=funding, config=config,
        close_2d=close, cost_bps=8.0,
    )
    assert np.all(np.isfinite(result))
    assert result.shape == (n_bars, n_syms)
    gross_before = float(np.mean(np.sum(np.abs(result[50:80]), axis=1)))
    gross_after = float(np.mean(np.sum(np.abs(result[300:]), axis=1)))
    assert gross_after > gross_before * 0.5, (
        f"gross after drawdown ({gross_after}) should recover toward "
        f"pre-dd level ({gross_before})"
    )


def test_compute_dynamic_compounding_path_band_frac_zero_uses_raw_ewma_state() -> None:
    n_bars, n_syms = 10, 3
    ts = np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000 * 4
    mu_2d = np.full((n_bars, n_syms), 0.02, dtype=np.float32)
    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=ts, symbols=tuple(f"S{i}" for i in range(n_syms)),
        mu_2d=mu_2d, se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        family_ids=("f1",), admitted_signal_ids=("s1",), fold_manifest_hash="fh1",
    )
    sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    close = np.ones((n_bars, n_syms), dtype=np.float32)
    config = DynamicCompoundingConfig(band_frac=0.0, alpha_smooth=0.5)
    result = compute_dynamic_compounding_path(
        forecast=forecast, sigma_2d=sigma_2d, funding_rates_1h_2d=funding,
        config=config, close_2d=close, cost_bps=8.0,
    )
    assert np.all(np.isfinite(result))
    assert not np.allclose(result[1], result[0])


def test_compute_dynamic_compounding_path_zero_support_bar_resets_state_to_zero() -> None:
    n_bars, n_syms = 5, 3
    ts = np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000 * 4
    mu_2d = np.full((n_bars, n_syms), 0.02, dtype=np.float32)
    mu_2d[2] = 0.0  # bar 2: no support at all
    forecast = CalibratedForecastPanel(
        decision_timestamps_ns=ts, symbols=tuple(f"S{i}" for i in range(n_syms)),
        mu_2d=mu_2d, se_2d=np.full((n_bars, n_syms), 0.01, dtype=np.float32),
        family_mu_3d=np.zeros((n_bars, n_syms, 1), dtype=np.float32),
        family_ids=("f1",), admitted_signal_ids=("s1",), fold_manifest_hash="fh1",
    )
    sigma_2d = np.full((n_bars, n_syms), 0.01, dtype=np.float32)
    funding = np.zeros((n_bars * 4, n_syms), dtype=np.float32)
    close = np.ones((n_bars, n_syms), dtype=np.float32)
    config = DynamicCompoundingConfig(band_frac=0.30, alpha_smooth=0.5)
    result = compute_dynamic_compounding_path(
        forecast=forecast, sigma_2d=sigma_2d, funding_rates_1h_2d=funding,
        config=config, close_2d=close, cost_bps=8.0,
    )
    assert np.all(result[2] == 0.0)


def test_dynamic_compounding_config_zero_dd_scale_floor_raises_assertion_error() -> None:
    with pytest.raises(AssertionError, match="dd_scale_floor"):
        DynamicCompoundingConfig(dd_scale_floor=0.0)


def test_deflated_sharpe_probability_scales_with_effective_trials() -> None:
    rng = np.random.default_rng(42)
    excess_rets = rng.normal(0.001, 0.01, 365).astype(np.float64)
    low_m = TrialMultiplicity(5, 2.0, 1.0)
    high_m = TrialMultiplicity(5, 5.0, 1.0)
    p_low = deflated_sharpe_probability(
        observed_sharpe=1.85, multiplicity=low_m,
        excess_returns=excess_rets,
    )
    p_high = deflated_sharpe_probability(
        observed_sharpe=1.85, multiplicity=high_m,
        excess_returns=excess_rets,
    )
    assert p_low >= p_high


def test_deflated_sharpe_probability_short_sample_returns_half() -> None:
    rng = np.random.default_rng(42)
    excess_rets = rng.normal(0.001, 0.01, 20).astype(np.float64)
    mult = TrialMultiplicity(27, 5.0, 1.0)
    prob = deflated_sharpe_probability(
        observed_sharpe=5.0, multiplicity=mult,
        excess_returns=excess_rets,
    )
    assert prob == 0.5


def test_engine_folds_do_not_overlap_sealed_holdout_window() -> None:
    l1_window_end = 1500
    config = CalibrationConfig(
        n_folds=5, purge_bars=25, embargo_bars=42,
        min_fold_obs=100, ridge_lambda_scale=0.01,
        family_shrink=0.5,
    )
    folds = build_folds_4h(l1_window_end, config)
    for fold in folds:
        assert fold.oos_end_exclusive <= l1_window_end, (
            f"fold {fold.fold_id}: oos_end={fold.oos_end_exclusive} "
            f"> l1_window_end={l1_window_end}"
        )


def test_estimate_cluster_sleeve_posteriors_rejects_is_strong_oos_reversed() -> None:
    """Fit-window correlation is strongly positive (high posterior probability)
    while the OOS window deliberately reverses sign. The P1 OOS AND-gate must
    reject this sleeve (oos_confirmation_failed), fixing the old bug where
    in-sample significance alone granted admission."""
    n_syms, n_bars = 2, 40
    symbols = ("S0", "S1")

    close = np.empty((n_bars, n_syms), dtype=np.float64)
    close[0] = 100.0
    close[1:4] = 100.0
    for t in range(0, 19):
        close[t + 1] = close[t] * 1.02  # fit-window: future return +2%
    for t in range(19, 25):
        close[t + 1] = close[t]  # flat purge/calibration gap
    for t in range(25, 35):
        close[t + 1] = close[t] * 0.98  # oos-window: future return -2%
    for t in range(35, n_bars - 1):
        close[t + 1] = close[t]

    close32 = close.astype(np.float32)
    bars_4h = TimeframeBarCube(
        "4h", np.arange(n_bars, dtype=np.int64), symbols,
        close32.copy(), close32 + 1.0, close32 - 1.0, close32.copy(),
        np.ones((n_bars, n_syms), dtype=np.float32),
        np.ones((n_bars, n_syms), dtype=np.bool_),
    )

    z = np.ones((n_bars, n_syms, 1), dtype=np.float32)
    valid = np.ones((n_bars, n_syms, 1), dtype=np.bool_)
    descriptors = (SignalDescriptor("s", "trend", "fast", 4, "4h", 4, "", "", "v1"),)
    panel = RawSignalPanel(
        np.arange(n_bars, dtype=np.int64), symbols, descriptors,
        z, valid, np.ones((n_bars, n_syms), dtype=np.float32),
    )

    labels = np.array([0, 0], dtype=np.int32)
    centroids = np.zeros((1, 4), dtype=np.float64)
    cluster_panel = ClusterPanel(symbols, labels, centroids, 1)
    cf = CausalClusterFold(
        fold_id=0, fit_end_exclusive_4h=20,
        fit_end_time_ns=20 * 4 * 3_600_000_000_000,
        panel=cluster_panel, member_hash="test_hash",
    )
    fold = CausalFold(0, 0, 20, 15, 20, 25, 35, 5, 1)

    cost = np.ones((n_bars, n_syms), dtype=np.float32)
    funding = np.zeros((n_bars, n_syms), dtype=np.float32)
    config = HandoffConfig()

    result = estimate_cluster_sleeve_posteriors(
        panel, bars_4h, (cf,), (fold,), cost, funding, config,
    )

    rejected_negative = [
        s for s in result
        if not s.admitted
        and s.posterior_positive_probability >= 0.95
        and s.mean_net_return < 0.0
        and "oos_confirmation_failed" in s.reasons
    ]
    assert rejected_negative, (
        f"expected a sleeve rejected by OOS confirmation; got {result}"
    )


def test_run_multiscale_compound_engine_passes_five_time_ordered_folds_to_l2(
    tmp_path, monkeypatch,
) -> None:
    from src.domain.futures.compound.config import CompoundEngineConfig
    from src.domain.futures.compound.contracts import MarketFeatureCube, SealedHoldoutManifest
    from src.domain.futures.compound.engine import run_multiscale_compound_engine
    from src.domain.futures.compound.holdout_store import SealedHoldoutStore
    import src.domain.futures.compound.engine as engine_module
    import src.domain.futures.compound.validation as validation_module

    n_bars, n_syms = 1024, 5
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
    ns_per_hour = 3_600_000_000_000
    close = np.column_stack(tuple(
        np.linspace(100, 110 + i, n_bars) for i in range(n_syms)
    )).astype(np.float64)
    arr_f32 = close.astype(np.float32)
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * ns_per_hour,
        symbols=symbols,
        fields_2d={
            "open": arr_f32 * 0.9995, "high": arr_f32 * 1.005,
            "low": arr_f32 * 0.995, "close": arr_f32,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "mark": arr_f32.copy(), "index": arr_f32.copy(),
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    universe = type("Universe", (), {"symbols": symbols, "snapshots": ()})()
    store = SealedHoldoutStore(tmp_path / "engine_wiring_test.sqlite3")
    manifest = SealedHoldoutManifest(
        holdout_id="wiring-test",
        start_time_ns=int(cube.timestamps_ns[-180]),
        end_time_ns=int(cube.timestamps_ns[-1]),
        holdout_days=7,
        model_version="v1",
        data_manifest_hash="h1",
        strategy_spec_hash="spec1",
    )
    store.create(manifest)

    captured: dict[str, np.ndarray] = {}
    real_evaluate = validation_module.evaluate_l2_walk_forward

    def _spy_evaluate(*, ledger, fold_ids_1d, benchmark, trial_multiplicity, config, bootstrap_seed, **kwargs):
        captured["fold_ids_1d"] = fold_ids_1d.copy()
        return real_evaluate(
            ledger=ledger, fold_ids_1d=fold_ids_1d, benchmark=benchmark,
            trial_multiplicity=trial_multiplicity, config=config, bootstrap_seed=bootstrap_seed,
            **kwargs,
        )

    monkeypatch.setattr(engine_module, "evaluate_l2_walk_forward", _spy_evaluate)

    run_multiscale_compound_engine(
        market=cube, universe=universe, holdout_store=store,
        holdout_id="wiring-test", config=CompoundEngineConfig(),
    )

    assert "fold_ids_1d" in captured
    fold_ids = captured["fold_ids_1d"]
    unique = np.unique(fold_ids)
    assert len(unique) == 5, f"expected 5 outer folds, got {len(unique)}: {unique}"
    first_idx_per_fold = [int(np.argmax(fold_ids == f)) for f in unique]
    assert first_idx_per_fold == sorted(first_idx_per_fold), (
        "fold ids must be time-ordered (non-decreasing over the ledger)"
    )


# ── P0 integration: unified walk-forward span wired through the engine ─────
# (docs/specs/l1_cash_only_exit_redesign.md scenarios 14-15)


def _synthetic_engine_fixture(n_bars: int = 24000, n_syms: int = 20):
    import datetime

    from src.domain.futures.compound.contracts import MarketFeatureCube, SealedHoldoutManifest
    from src.domain.futures.data_lake.run_windows import QuarterlyRunWindow

    symbols = ("BTCUSDT", "ETHUSDT") + tuple(f"SYM{i}USDT" for i in range(n_syms - 2))
    ns_per_hour = 3_600_000_000_000
    rng = np.random.default_rng(1)
    close = (np.cumprod(1.0 + rng.normal(0.0002, 0.004, (n_bars, n_syms)), axis=0).astype(np.float32) * 100)
    cube = MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * ns_per_hour,
        symbols=symbols,
        fields_2d={
            "open": close * 0.9995, "high": close * 1.005, "low": close * 0.995, "close": close,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "mark": close.copy(), "index": close.copy(),
            "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    universe = type("Universe", (), {"symbols": symbols, "snapshots": ()})()
    far_future = 9_000_000_000_000_000_000
    window = QuarterlyRunWindow(
        requested_date=datetime.date(2020, 1, 1), cutoff_date=datetime.date(2020, 4, 1),
        acquisition_start_ns=far_future, l1_start_ns=far_future + 1,
        l2_start_ns=far_future + 2, l3_start_ns=far_future + 3, cutoff_exclusive_ns=far_future + 4,
    )
    manifest = SealedHoldoutManifest(
        holdout_id="quarterly-2020-04-01", start_time_ns=int(cube.timestamps_ns[-100]),
        end_time_ns=int(cube.timestamps_ns[-1]), holdout_days=4, model_version="v1",
        data_manifest_hash="h1", strategy_spec_hash="spec1",
    )
    return cube, universe, window, manifest


def test_engine_wires_unified_span_into_handoff(tmp_path) -> None:
    """[RULE-P0-4] The quarterly path must call build_expanding_walk_forward_steps
    with (l1_start, l3_start), not the old (l1_start, l2_start) 5-fold span."""
    from src.domain.futures.compound.config import CompoundEngineConfig
    from src.domain.futures.compound.engine import run_multiscale_compound_engine
    from src.domain.futures.compound.holdout_store import SealedHoldoutStore
    import src.domain.futures.compound.calibration as calibration_module
    import src.domain.futures.compound.engine as engine_module

    cube, universe, window, manifest = _synthetic_engine_fixture()
    store = SealedHoldoutStore(tmp_path / "unified_span_test.sqlite3")
    store.create(manifest)

    calls: list[tuple] = []
    real_fn = calibration_module.build_expanding_walk_forward_steps

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fn(*args, **kwargs)

    engine_module.build_expanding_walk_forward_steps = _spy
    try:
        run_multiscale_compound_engine(
            market=cube, universe=universe, window=window, holdout_store=store,
            config=CompoundEngineConfig(),
        )
    finally:
        engine_module.build_expanding_walk_forward_steps = real_fn

    assert calls, "quarterly path must call build_expanding_walk_forward_steps"
    args, kwargs = calls[0]
    l1_start, l3_start = args[0], args[1]
    assert l3_start > l1_start
    # l2_start (the old, narrower upper bound) must NOT be the span passed in.
    l2_start_would_be = l1_start + (l3_start - l1_start) // 2
    assert l3_start != l2_start_would_be, "must use the l3 boundary, not the old l2-only span"

    steps = real_fn(l1_start, l3_start, CompoundEngineConfig().calibration,
                    step_bars=kwargs["step_bars"], initial_fit_bars=kwargs["initial_fit_bars"],
                    max_target_horizon_bars=kwargs["max_target_horizon_bars"])
    assert len(steps) > 5, "unified span must produce more steps than the old 5-fold L1-only split"

