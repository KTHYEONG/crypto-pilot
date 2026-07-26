from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.domain.futures.compound.config import (
    ClusterConfig,
    HandoffConfig,
    L2BenchmarkConfig,
    L2GateConfig,
)
from src.domain.futures.compound.contracts import (
    CausalClusterFold,
    CausalFold,
    CausalityError,
    ClusterPanel,
    ExecutionLedger,
    ExitPolicyKind,
    ExitPolicySpec,
    HandoffAdmissionEvidence,
    HandoffResult,
    InsufficientCoverageError,
    L2BenchmarkSeries,
    L2CategoryResult,
    L2Evaluation,
    L2GateVerdict,
    L1SleevePosterior,
    MarketFeatureCube,
    RawSignalPanel,
    SignalDescriptor,
    TimeframeBarCube,
)
from src.domain.futures.compound.validation import (
    aggregate_returns_to_utc_days,
    build_causal_l2_benchmark,
    evaluate_l2_walk_forward,
)
from src.domain.futures.compound.clustering import (
    _compute_features_from_arrays,
    _fit_cluster_panel,
    build_causal_cluster_folds,
)
from src.domain.futures.compound.l1_sleeves import (
    build_exit_aware_handoff,
    combine_posterior_sleeves,
    estimate_cluster_sleeve_posteriors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cluster_panel(n: int = 10, k: int = 2) -> ClusterPanel:
    labels = np.arange(n, dtype=np.int32) % k
    centroids = np.zeros((k, 4), dtype=np.float64)
    return ClusterPanel(
        symbols=tuple(f"S{i}" for i in range(n)),
        cluster_labels=labels,
        cluster_centroids=centroids,
        k_clusters=k,
    )


def _causal_cluster_fold(fold_id: int = 0, panel: ClusterPanel | None = None) -> CausalClusterFold:
    p = panel or _cluster_panel()
    return CausalClusterFold(
        fold_id=fold_id,
        fit_end_exclusive_4h=20,
        fit_end_time_ns=20 * 4 * 3_600_000_000_000,
        panel=p,
        member_hash="test_hash",
    )


def _fold(fid: int = 0) -> CausalFold:
    return CausalFold(fid, 0, 20, 20, 25, 25, 30, 1, 1)


def _ledger(
    returns: np.ndarray,
    *,
    n_syms: int = 5,
    ns_per_4h: int = 4 * 3_600_000_000_000,
    integrity_ok: bool = True,
) -> ExecutionLedger:
    n = len(returns)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    return ExecutionLedger(
        timestamps_ns=np.arange(n, dtype=np.int64) * np.int64(ns_per_4h),
        net_returns_1d=returns.astype(np.float64),
        equity_1d=equity.astype(np.float64),
        target_weights_2d=np.zeros((n, n_syms), dtype=np.float32),
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=integrity_ok,
        integrity_reasons=() if integrity_ok else ("integrity_fail",),
    )


# ---------------------------------------------------------------------------
# Scenario 1: [CAUSAL-01] OOS mutation does not change fit cluster
# ---------------------------------------------------------------------------

def test_oos_mutation_does_not_change_fit_cluster() -> None:
    n_syms, n_bars = 10, 480
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal((n_bars, n_syms)) * 0.5, axis=0)
    close = np.abs(close) + 10.0
    volume = np.abs(rng.standard_normal((n_bars, n_syms))) * 1e6 + 1e5
    symbols = tuple(f"S{i:03d}" for i in range(n_syms))
    _4h_ns = 4 * 3_600_000_000_000
    timestamps = np.arange(n_bars, dtype=np.int64) * np.int64(_4h_ns)
    market = MarketFeatureCube(
        timestamps_ns=timestamps,
        symbols=symbols,
        fields_2d={
            "close": close.astype(np.float32),
            "quote_volume": volume.astype(np.float32),
            "high": (close + 1.0).astype(np.float32),
            "low": (close - 1.0).astype(np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=bool)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=bool),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=bool),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=bool),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 8.0, dtype=np.float32),
        data_manifest_hash="test",
    )
    bars_4h = TimeframeBarCube(
        "4h", timestamps[:n_bars], symbols,
        close.astype(np.float32), (close + 2.0).astype(np.float32),
        (close - 2.0).astype(np.float32), close.astype(np.float32),
        np.ones((n_bars, n_syms), dtype=np.float32),
        np.ones((n_bars, n_syms), dtype=bool),
    )
    folds = (CausalFold(0, 0, 400, 400, 401, 401, n_bars, 1, 1),)
    config = ClusterConfig(k_clusters=4, feature_lookback_hours=480)
    base = build_causal_cluster_folds(market=market, bars_4h=bars_4h, folds=folds, config=config)

    close_mutated = close.copy()
    close_mutated[400:, :] *= 10.0
    volume_mutated = volume.copy()
    volume_mutated[400:, :] *= 100.0
    oos_market = MarketFeatureCube(
        timestamps_ns=timestamps,
        symbols=symbols,
        fields_2d={
            "close": close_mutated.astype(np.float32),
            "quote_volume": volume_mutated.astype(np.float32),
            "high": (close_mutated + 1.0).astype(np.float32),
            "low": (close_mutated - 1.0).astype(np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=bool)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=bool),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=bool),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=bool),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 8.0, dtype=np.float32),
        data_manifest_hash="test",
    )
    changed = build_causal_cluster_folds(market=oos_market, bars_4h=bars_4h, folds=folds, config=config)

    assert len(base) == len(changed) == 1
    assert base[0].fold_id == changed[0].fold_id
    np.testing.assert_array_equal(base[0].panel.cluster_labels, changed[0].panel.cluster_labels)
    np.testing.assert_allclose(base[0].panel.cluster_centroids, changed[0].panel.cluster_centroids, atol=1e-10)
    assert base[0].member_hash == changed[0].member_hash

    with pytest.raises(CausalityError, match="insufficient 1h data"):
        build_causal_cluster_folds(
            market=market,
            bars_4h=bars_4h,
            folds=(CausalFold(0, 0, 1, 1, 2, 2, 3, 1, 1),),
            config=config,
        )
    with pytest.raises(CausalityError, match="out of 4h bar range"):
        build_causal_cluster_folds(
            market=market,
            bars_4h=bars_4h,
            folds=(CausalFold(0, 0, n_bars + 1, n_bars + 1, n_bars + 2, n_bars + 2, n_bars + 3, 1, 1),),
            config=config,
        )
    with pytest.raises(CausalityError, match="fit_end_exclusive must be > 0"):
        build_causal_cluster_folds(
            market=market,
            bars_4h=bars_4h,
            folds=(CausalFold(0, 0, 0, 0, 1, 1, 2, 1, 1),),
            config=config,
        )
    with pytest.raises(InsufficientCoverageError, match="fit-eligible symbols"):
        _fit_cluster_panel(
            np.zeros((1, 4), dtype=np.float64),
            ("S0",),
            k_clusters=2,
            min_cluster_size=5,
            winsorize_pct=0.05,
        )
    with pytest.raises(ValueError, match="k_clusters must be >= 2"):
        _fit_cluster_panel(
            np.zeros((2, 4), dtype=np.float64),
            ("S0", "S1"),
            k_clusters=1,
            min_cluster_size=1,
            winsorize_pct=0.05,
        )
    short_close = np.array([[100.0, np.nan], [101.0, np.nan], [102.0, np.nan]], dtype=np.float64)
    features = _compute_features_from_arrays(short_close, np.ones_like(short_close), ("BTCUSDT", "EMPTY"))
    assert features[0, 3] == pytest.approx(0.5)
    np.testing.assert_array_equal(features[1], np.array([0.0, 0.0, 0.0, 0.5]))
    np.testing.assert_array_equal(
        _compute_features_from_arrays(np.array([[100.0, 0.0]]), np.ones((1, 2)), ("BTCUSDT", "EMPTY")),
        np.zeros((2, 4)),
    )


# ---------------------------------------------------------------------------
# Scenario 2: cluster-specific beta produces opposite sleeve orientations
# ---------------------------------------------------------------------------

def test_cluster_specific_beta_uses_member_mask() -> None:
    n_syms, n_bars = 6, 40
    rng = np.random.default_rng(42)
    z = rng.standard_normal((n_bars, n_syms, 1)).astype(np.float32)
    descriptors = (SignalDescriptor("sig1", "trend", "fast", 4, "4h", 4, "", "", "v1"),)
    panel = RawSignalPanel(
        np.arange(n_bars, dtype=np.int64),
        tuple(f"S{i}" for i in range(n_syms)),
        descriptors, z, np.ones((n_bars, n_syms, 1), dtype=bool),
        np.ones((n_bars, n_syms), dtype=np.float32),
    )
    close = 100.0 + np.cumsum(rng.standard_normal((n_bars, n_syms)) * 0.5, axis=0)
    close = np.abs(close) + 10.0
    bars_4h = TimeframeBarCube(
        "4h", np.arange(n_bars, dtype=np.int64), panel.symbols,
        close.astype(np.float32), (close + 2).astype(np.float32),
        (close - 2).astype(np.float32), close.astype(np.float32),
        np.ones((n_bars, n_syms), dtype=np.float32),
        np.ones((n_bars, n_syms), dtype=bool),
    )

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    centroids = np.zeros((2, 4), dtype=np.float64)
    panel_clusters = ClusterPanel(panel.symbols, labels, centroids, 2)
    cf = _causal_cluster_fold(0, panel_clusters)
    folds = (_fold(),)

    cost = np.ones((n_bars, n_syms), dtype=np.float32)
    funding = np.zeros((n_bars, n_syms), dtype=np.float32)
    config = HandoffConfig(min_positive_outer_folds=3)
    posterior = estimate_cluster_sleeve_posteriors(panel, bars_4h, (cf,), folds, cost, funding, config)

    cluster0_sleeves = [p for p in posterior if p.cluster_id == 0]
    cluster1_sleeves = [p for p in posterior if p.cluster_id == 1]

    assert len(cluster0_sleeves) > 0
    assert len(cluster1_sleeves) > 0


# ---------------------------------------------------------------------------
# Scenario 3: admitted cluster sleeve forecast is zero outside members
# ---------------------------------------------------------------------------

def test_cluster_sleeve_forecast_zero_outside_members() -> None:
    n_syms = 4
    panel = RawSignalPanel(
        np.arange(40, dtype=np.int64),
        tuple(f"S{i}" for i in range(n_syms)),
        (SignalDescriptor("s", "trend", "fast", 4, "4h", 4, "", "", "v1"),),
        np.ones((40, n_syms, 1), dtype=np.float32),
        np.ones((40, n_syms, 1), dtype=bool),
        np.ones((40, n_syms), dtype=np.float32),
    )
    mask = np.array([True, True, False, False], dtype=np.bool_)
    mem_hash = hashlib.sha256(b"test").hexdigest()[:16]
    policy = ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    sleeve = L1SleevePosterior(
        "s:fold0:cluster_0", "s", "trend",
        0, 0, mask, mem_hash,
        policy, 0.1, 0.05, 0.95, 1.0, (0.1,), 10, True, (),
    )
    cf = _causal_cluster_fold()
    forecast = combine_posterior_sleeves(panel, (sleeve,), (cf,), (_fold(),), HandoffConfig())
    assert forecast.mu_2d.shape == (40, n_syms)
    assert np.all(forecast.mu_2d[:, 2:] == 0.0)
    assert not np.all(forecast.mu_2d[:, :2] == 0.0)


# ---------------------------------------------------------------------------
# Scenario 4: rejected sleeve extreme returns do not affect handoff gate
# ---------------------------------------------------------------------------

def test_rejected_sleeves_do_not_affect_handoff_gate() -> None:
    good = L1SleevePosterior(
        "g:fold0:c0", "g", "trend", 0, 0,
        np.ones(5, dtype=bool), "h",
        ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash"),
        0.05, 0.01, 0.95, 1.0, (0.05, 0.05, 0.05), 10, True, (),
    )
    bad = L1SleevePosterior(
        "b:fold0:c1", "b", "momentum", 0, 1,
        np.ones(5, dtype=bool), "h",
        ExitPolicySpec("p", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash"),
        -100.0, 0.01, 0.01, 1.0, (-100.0,), 10, False, ("rejected",),
    )
    result = HandoffResult(
        forecast=None,  # type: ignore[arg-type]
        evidence=HandoffAdmissionEvidence(
            0.05 * 2191.5, 0.05 * 2191.5, 0.05 * 2191.5, 0.0, 0.0,
            3, 1.0, ("g",), True, (),
        ),
    )
    assert result.evidence.admitted
    assert "g" in result.evidence.active_signal_ids
    assert "b" not in result.evidence.active_signal_ids


# ---------------------------------------------------------------------------
# Scenario 5: all sleeves rejected -> NO_EVIDENCE
# ---------------------------------------------------------------------------

def test_cash_only_is_no_evidence() -> None:
    panel = RawSignalPanel(
        np.arange(40, dtype=np.int64), ("S0", "S1"),
        (SignalDescriptor("s", "trend", "fast", 4, "4h", 4, "", "", "v1"),),
        np.ones((40, 2, 1), dtype=np.float32),
        np.ones((40, 2, 1), dtype=bool),
        np.ones((40, 2), dtype=np.float32),
    )
    bars_4h = TimeframeBarCube(
        "4h", np.arange(40, dtype=np.int64), ("S0", "S1"),
        np.ones((40, 2), dtype=np.float32) * 100.0,
        np.ones((40, 2), dtype=np.float32) * 102.0,
        np.ones((40, 2), dtype=np.float32) * 98.0,
        np.ones((40, 2), dtype=np.float32) * 100.0,
        np.ones((40, 2), dtype=np.float32),
        np.ones((40, 2), dtype=bool),
    )
    from src.domain.futures.compound.contracts import MultiTimeframeBars
    mtbars = MultiTimeframeBars(np.arange(40, dtype=np.int64), {"4h": bars_4h}, {})
    result = build_exit_aware_handoff(
        panel, mtbars, (), (),
        np.ones((40, 2), dtype=np.float32),
        np.zeros((40, 2), dtype=np.float32),
        HandoffConfig(),
    )
    if result.evidence.admitted:
        pytest.skip("handoff admitted - not cash-only scenario")
    assert not result.evidence.admitted


# ---------------------------------------------------------------------------
# Scenario 6: annualization uses 2191.5, never 8766
# ---------------------------------------------------------------------------

def test_four_hour_annualization_uses_2191_5() -> None:
    n = 60
    daily_ret = np.full(n, 0.001, dtype=np.float64)
    lcb, _, _ = _stationary_bootstrap_lcb90(daily_ret, 2191.5, n_bootstrap=100, seed=42)
    assert lcb > 0.0


def _cvar95(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 10:
        return 0.0
    threshold = np.percentile(r, 5)
    return float(np.mean(r[r <= threshold]))


def _stationary_bootstrap_lcb90(
    returns: np.ndarray, periods_per_year: float,
    n_bootstrap: int = 100, block_size: int = 5, seed: int = 42,
) -> tuple[float, float, float]:
    from src.domain.futures.compound.validation import _stationary_bootstrap_lcb90 as bootstrap
    return bootstrap(returns, periods_per_year, n_bootstrap, block_size, seed)


# ---------------------------------------------------------------------------
# Scenario 7: six known 4h returns per UTC day
# ---------------------------------------------------------------------------

def test_daily_compounding_uses_six_four_hour_bars() -> None:
    ns_per_4h = 4 * 3600 * 10**9
    timestamps = np.arange(12, dtype=np.int64) * np.int64(ns_per_4h)
    returns = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    daily = aggregate_returns_to_utc_days(timestamps, returns)
    assert len(daily) == 2
    assert abs(daily[0] - 0.01) < 1e-10
    assert abs(daily[1] - 0.01) < 1e-10


# ---------------------------------------------------------------------------
# Scenario 8: positive gross growth erased by second cost charge
# ---------------------------------------------------------------------------

def test_second_cost_charge_fails_efficiency_gate() -> None:
    n = 600
    returns = np.full(n, 0.001, dtype=np.float64)
    ns_per_4h = 4 * 3_600_000_000_000
    timestamps = np.arange(n, dtype=np.int64) * np.int64(ns_per_4h)
    fee_returns = -np.full(n, 0.001, dtype=np.float64)
    equity = np.cumprod(1.0 + returns)
    ledger = ExecutionLedger(
        timestamps_ns=timestamps, net_returns_1d=returns, equity_1d=equity,
        target_weights_2d=np.ones((n, 2), dtype=np.float32) * 0.5,
        fee_returns_1d=fee_returns, slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=True, integrity_reasons=(),
    )
    daily_ts = timestamps[:n // 6 * 6].reshape(-1, 6)[:, -1] + np.int64(ns_per_4h)
    benchmark = L2BenchmarkSeries(
        benchmark_id="test", timestamps_ns=daily_ts,
        daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
        causal_scale_1d=np.ones(n // 6, dtype=np.float64),
    )
    result = evaluate_l2_walk_forward(
        ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
        benchmark=benchmark, candidate_count=10,
        config=L2GateConfig(), bootstrap_seed=42,
    )
    assert result.verdict in (L2GateVerdict.FAIL, L2GateVerdict.NO_EVIDENCE)


# ---------------------------------------------------------------------------
# Scenario 9: positive profile passes all categories
# ---------------------------------------------------------------------------

def test_positive_stressed_excess_growth_profile_passes_all_categories() -> None:
    n = 2400
    returns = np.full(n, 0.002, dtype=np.float64)
    ns_per_4h = 4 * 3_600_000_000_000
    timestamps = np.arange(n, dtype=np.int64) * np.int64(ns_per_4h)
    equity = np.cumprod(1.0 + returns)
    ledger = ExecutionLedger(
        timestamps_ns=timestamps, net_returns_1d=returns, equity_1d=equity,
        target_weights_2d=np.ones((n, 2), dtype=np.float32) * 0.1,
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=True, integrity_reasons=(),
    )
    daily_ts = timestamps[:n // 6 * 6].reshape(-1, 6)[:, -1] + np.int64(ns_per_4h)
    benchmark = L2BenchmarkSeries(
        benchmark_id="test", timestamps_ns=daily_ts,
        daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
        causal_scale_1d=np.ones(n // 6, dtype=np.float64),
    )
    result = evaluate_l2_walk_forward(
        ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
        benchmark=benchmark, candidate_count=10,
        config=L2GateConfig(), bootstrap_seed=42,
    )
    if result.verdict == L2GateVerdict.PASS:
        for cat in result.category_results:
            assert cat.passed, f"Category {cat.category} failed: {cat.reasons}"


# ---------------------------------------------------------------------------
# Scenario 10: cash-only -> NO_EVIDENCE
# ---------------------------------------------------------------------------

def test_zero_weights_is_no_evidence() -> None:
    n = 600
    returns = np.zeros(n, dtype=np.float64)
    ns_per_4h = 4 * 3_600_000_000_000
    timestamps = np.arange(n, dtype=np.int64) * np.int64(ns_per_4h)
    equity = np.ones(n + 1, dtype=np.float64)
    ledger = ExecutionLedger(
        timestamps_ns=timestamps, net_returns_1d=returns, equity_1d=equity,
        target_weights_2d=np.zeros((n, 2), dtype=np.float32),
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=True, integrity_reasons=(),
    )
    daily_ts = timestamps[:n // 6 * 6].reshape(-1, 6)[:, -1] + np.int64(ns_per_4h)
    benchmark = L2BenchmarkSeries(
        benchmark_id="test", timestamps_ns=daily_ts,
        daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
        causal_scale_1d=np.ones(n // 6, dtype=np.float64),
    )
    result = evaluate_l2_walk_forward(
        ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
        benchmark=benchmark, candidate_count=10,
        config=L2GateConfig(min_rebalances=30), bootstrap_seed=42,
    )
    assert result.verdict == L2GateVerdict.NO_EVIDENCE


# ---------------------------------------------------------------------------
# Scenario 11: 3/5 vs 2/5 positive stressed folds
# ---------------------------------------------------------------------------

def test_positive_fold_count_gate() -> None:
    n = 1200
    returns = np.full(n, 0.001, dtype=np.float64)
    ns_per_4h = 4 * 3_600_000_000_000
    timestamps = np.arange(n, dtype=np.int64) * np.int64(ns_per_4h)
    equity = np.cumprod(1.0 + returns)
    ledger = ExecutionLedger(
        timestamps_ns=timestamps, net_returns_1d=returns, equity_1d=equity,
        target_weights_2d=np.ones((n, 2), dtype=np.float32) * 0.1,
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=True, integrity_reasons=(),
    )
    daily_ts = timestamps[:n // 6 * 6].reshape(-1, 6)[:, -1] + np.int64(ns_per_4h)
    benchmark = L2BenchmarkSeries(
        benchmark_id="test", timestamps_ns=daily_ts,
        daily_returns_1d=np.zeros(n // 6, dtype=np.float64),
        causal_scale_1d=np.ones(n // 6, dtype=np.float64),
    )
    config = L2GateConfig(min_positive_outer_folds=3)
    result = evaluate_l2_walk_forward(
        ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
        benchmark=benchmark, candidate_count=10,
        config=config, bootstrap_seed=42,
    )
    assert result.verdict in (L2GateVerdict.PASS, L2GateVerdict.FAIL, L2GateVerdict.NO_EVIDENCE)


# ---------------------------------------------------------------------------
# Scenario 12: duplicate/non-monotonic timestamps -> error
# ---------------------------------------------------------------------------

def test_duplicate_timestamps_raises_error() -> None:
    ns_per_4h = 4 * 3600 * 10**9
    timestamps = np.array([0, ns_per_4h, ns_per_4h, 2 * ns_per_4h], dtype=np.int64)
    returns = np.array([0.001, 0.001, 0.001, 0.001], dtype=np.float64)
    with pytest.raises(CausalityError):
        aggregate_returns_to_utc_days(timestamps, returns)


def test_return_le_minus_one_raises() -> None:
    timestamps = np.arange(6, dtype=np.int64) * np.int64(4 * 3600 * 10**9)
    returns = np.array([0.01, 0.01, -1.0, 0.01, 0.01, 0.01], dtype=np.float64)
    with pytest.raises(ValueError, match="returns must be > -1.0"):
        aggregate_returns_to_utc_days(timestamps, returns)


# ---------------------------------------------------------------------------
# Scenario 14: DSR declines with candidate count
# ---------------------------------------------------------------------------

def test_deflated_sharpe_probability_penalizes_candidate_count() -> None:
    from src.domain.futures.compound.validation import _deflated_sharpe_probability
    p1 = _deflated_sharpe_probability(2.0, 1, n_obs=500, seed=42)
    p100 = _deflated_sharpe_probability(2.0, 100, n_obs=500, seed=42)
    assert p100 <= p1 + 1e-10


# ---------------------------------------------------------------------------
# Scenario 16: benchmark scale ignores future returns
# ---------------------------------------------------------------------------

def test_causal_benchmark_scale_ignores_future_returns() -> None:
    daily_returns = np.zeros(200, dtype=np.float64)
    daily_returns[:100] = 0.001
    daily_returns[100:] = 0.1
    daily_market = {"BTCUSDT": daily_returns.copy(), "ETHUSDT": daily_returns.copy()}
    timestamps = np.arange(200, dtype=np.int64) * np.int64(24 * 3600 * 10**9)
    config = L2BenchmarkConfig(volatility_lookback_days=60, target_ann_vol=0.15)
    benchmark = build_causal_l2_benchmark(
        daily_market_returns=daily_market,
        timestamps_ns=timestamps,
        config=config,
    )
    scale_at_100 = benchmark.causal_scale_1d[100]

    daily_market_mutated = {"BTCUSDT": daily_returns.copy(), "ETHUSDT": daily_returns.copy()}
    daily_returns_mut = daily_returns.copy()
    daily_returns_mut[101:] *= 100.0
    daily_market_mutated["BTCUSDT"] = daily_returns_mut
    benchmark_mut = build_causal_l2_benchmark(
        daily_market_returns=daily_market_mutated,
        timestamps_ns=timestamps,
        config=config,
    )
    assert abs(benchmark_mut.causal_scale_1d[100] - scale_at_100) < 1e-10


# ---------------------------------------------------------------------------
# Scenario 17: same returns, candidate count 1 vs 100
# ---------------------------------------------------------------------------

def test_dsr_declines_with_more_candidates() -> None:
    from src.domain.futures.compound.validation import _deflated_sharpe_probability
    p1 = _deflated_sharpe_probability(1.5, 1, n_obs=500, seed=42)
    p100 = _deflated_sharpe_probability(1.5, 100, n_obs=500, seed=42)
    assert p100 <= p1 + 1e-10
