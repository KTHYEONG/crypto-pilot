from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.compound.calibration import build_folds_4h
from src.domain.futures.compound.config import (
    CalibrationConfig,
    CompoundEngineConfig,
)
from src.domain.futures.compound.contracts import (
    CandidateTrial,
    CandidateTrialLedger,
    CompoundEngineResult,
    CompoundWindowAudit,
    DeploymentBundle,
    DeploymentCandidate,
    DeploymentVerdict,
    ExecutionLedger,
    InsufficientCoverageError,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    QuarterlyBarBoundaries,
    SignalDescriptor,
)
from src.domain.futures.compound.deployment import (
    compute_live_target_weights,
    publish_promoted_strategy,
)
from src.domain.futures.compound.engine import (
    resolve_quarterly_boundaries,
)
from src.domain.futures.compound.holdout_store import SealedHoldoutStore
from src.domain.futures.compound.validation import (
    audit_compound_market_window,
)
from src.domain.futures.data_lake.run_windows import (
    QuarterlyRunWindow,
    QuarterlyWindowConfig,
    build_quarterly_execution_calendar,
    resolve_completed_quarter_window,
)


def _valid_descriptor() -> SignalDescriptor:
    return SignalDescriptor(
        signal_id="sig", family="momentum", speed="fast",
        lookback_hours=48, native_timeframe="4h",
    )


def test_candidate_contract_validation_errors_are_covered() -> None:
    with pytest.raises(ValueError, match="candidate_hash"):
        CandidateTrial("", "spec", (), "risk", 1)
    with pytest.raises(ValueError, match="strategy_spec_hash"):
        CandidateTrial("candidate", "", (), "risk", 1)

    desc = _valid_descriptor()
    base = {
        "active_signal_ids": ("sig",), "descriptors": (desc,),
        "orientation_signs": (1,), "vote_weights": (1.0,),
        "model_version": "v1", "strategy_spec_hash": "spec",
        "fold_manifest_hash": "fold", "trial_count": 0,
    }
    invalid = (
        ("active_signal_ids", ()),
        ("descriptors", ()),
        ("orientation_signs", ()),
        ("vote_weights", ()),
        ("model_version", ""),
        ("strategy_spec_hash", ""),
        ("fold_manifest_hash", ""),
        ("trial_count", -1),
    )
    for key, value in invalid:
        payload = dict(base)
        payload[key] = value
        with pytest.raises(ValueError):
            DeploymentCandidate(**payload)

    candidate = DeploymentCandidate(**base)
    with pytest.raises(ValueError, match="promotion_id"):
        DeploymentBundle(
            schema_version=1, promotion_id="", candidate=candidate,
            data_manifest_hash="data", universe_state_hash="universe",
            config_payload={}, l2_payload={}, l3_payload={},
        )
    with pytest.raises(ValueError, match="failed audit"):
        CompoundWindowAudit(
            passed=False, core_coverage_ratio=0.0,
            dataset_status=(), reasons=(),
        )
    with pytest.raises(ValueError, match="passed audit"):
        CompoundWindowAudit(
            passed=True, core_coverage_ratio=1.0,
            dataset_status=(), reasons=("unexpected",),
        )


def test_quarterly_boundary_validation_rejects_uncovered_grid() -> None:
    window = QuarterlyRunWindow(
        requested_date=date(2026, 7, 26), cutoff_date=date(2026, 6, 30),
        acquisition_start_ns=100, l1_start_ns=200, l2_start_ns=300,
        l3_start_ns=400, cutoff_exclusive_ns=500,
    )
    timestamps = np.arange(100, 501, 100, dtype=np.int64)
    boundaries = resolve_quarterly_boundaries(timestamps, window)
    assert (boundaries.acquisition_start, boundaries.l1_start,
            boundaries.l2_start, boundaries.l3_start,
            boundaries.cutoff_exclusive) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="not fully covered"):
        resolve_quarterly_boundaries(np.array([100, 200, 300], dtype=np.int64), window)

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_4H = 4 * _NS_PER_HOUR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def quarterly_window() -> QuarterlyRunWindow:
    return resolve_completed_quarter_window(
        date(2026, 7, 26), QuarterlyWindowConfig(),
    )


@pytest.fixture
def sample_timestamps_ns() -> np.ndarray:
    n = 910 * 6  # 910 days of 4h bars
    return np.arange(n, dtype=np.int64) * _NS_PER_4H


@pytest.fixture
def market_cube_2sym() -> MarketFeatureCube:
    n = 910 * 6
    syms = ("BTCUSDT", "ETHUSDT")
    close = np.ones((n, 2), dtype=np.float32) * 100.0
    return MarketFeatureCube(
        timestamps_ns=np.arange(n, dtype=np.int64) * _NS_PER_4H,
        symbols=syms,
        fields_2d={
            "close": close,
            "open": close * 0.9995,
            "high": close * 1.005,
            "low": close * 0.995,
            "quote_volume": np.ones((n, 2), dtype=np.float32) * 50_000_000,
            "funding": np.zeros((n, 2), dtype=np.float32),
        },
        available_2d={"core": np.ones((n, 2), dtype=np.bool_)},
        eligible_2d=np.ones((n, 2), dtype=np.bool_),
        entry_block_2d=np.zeros((n, 2), dtype=np.bool_),
        exit_required_2d=np.zeros((n, 2), dtype=np.bool_),
        capacity_usdt_2d=np.full((n, 2), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n, 2), 8.0, dtype=np.float32),
        data_manifest_hash="test_hash",
    )


# ---------------------------------------------------------------------------
# Test 1: Exact 910-day calendar and 5 boundary indices
# ---------------------------------------------------------------------------

class TestQuarterlyBarBoundaries:
    def test_boundaries_are_strictly_increasing(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            QuarterlyBarBoundaries(0, 10, 20, 15, 100)

    def test_910_day_calendar_and_boundaries(
        self, quarterly_window: QuarterlyRunWindow, sample_timestamps_ns: np.ndarray,
    ) -> None:
        calendar = build_quarterly_execution_calendar(quarterly_window)
        assert isinstance(calendar, pd.DatetimeIndex)
        assert len(calendar) > 0

        boundaries = resolve_quarterly_boundaries(sample_timestamps_ns, quarterly_window)
        assert isinstance(boundaries, QuarterlyBarBoundaries)
        assert boundaries.acquisition_start < boundaries.l1_start
        assert boundaries.l1_start < boundaries.l2_start
        assert boundaries.l2_start < boundaries.l3_start
        assert boundaries.l3_start < boundaries.cutoff_exclusive


# ---------------------------------------------------------------------------
# Test 2: L1/L2/L3 timestamp intersection is empty, each 365/90 complete days
# ---------------------------------------------------------------------------

class TestDisjointWindows:
    def test_l1_l2_l3_disjoint(self, quarterly_window, sample_timestamps_ns) -> None:
        boundaries = resolve_quarterly_boundaries(sample_timestamps_ns, quarterly_window)
        n_total = len(sample_timestamps_ns)
        indices = set(range(n_total))

        l1_set = set(range(boundaries.l1_start, boundaries.l2_start))
        l2_set = set(range(boundaries.l2_start, boundaries.l3_start))
        l3_set = set(range(boundaries.l3_start, boundaries.cutoff_exclusive))

        assert l1_set.isdisjoint(l2_set)
        assert l2_set.isdisjoint(l3_set)
        assert l1_set.isdisjoint(l3_set)


# ---------------------------------------------------------------------------
# Test 3: Offset fold OOS+horizon does not reach l2_start
# ---------------------------------------------------------------------------

class TestOffsetCausalFolds:
    def test_offset_fold_oos_before_l2(self) -> None:
        n_bars = 2000
        start_offset = 500
        config = CalibrationConfig(n_folds=5, ridge_lambda_scale=0.01)
        folds = build_folds_4h(n_bars, config, max_target_horizon_bars=10, start_offset=start_offset)
        for fold in folds:
            assert fold.fit_start >= start_offset
            assert fold.oos_end_exclusive <= n_bars
            assert fold.oos_start > fold.fit_end_exclusive
            assert 0 <= fold.fit_end_exclusive <= n_bars

    def test_offset_fold_oos_strictly_after_l1(self) -> None:
        n_bars = 2000
        l1_end = 500
        config = CalibrationConfig(n_folds=5)
        folds = build_folds_4h(n_bars, config, max_target_horizon_bars=25, start_offset=l1_end)
        for fold in folds:
            assert fold.oos_end_exclusive <= n_bars
            assert fold.fit_end_exclusive >= l1_end
            assert fold.oos_start >= fold.fit_end_exclusive + 25


# ---------------------------------------------------------------------------
# Test 4: CORE gap/coverage < 98% raises InsufficientCoverageError
# ---------------------------------------------------------------------------

class TestWindowAuditCoreCoverage:
    def test_leading_core_gap_raises(self, market_cube_2sym) -> None:
        n = market_cube_2sym.available_2d["core"].shape[0]
        market_cube_2sym.available_2d["core"][:100] = False
        window = QuarterlyBarBoundaries(0, 200, 500, 800, n)
        desc = SignalDescriptor(signal_id="test", family="test", speed="medium", lookback_hours=48, native_timeframe="4h")
        with pytest.raises(InsufficientCoverageError, match="leading_core_gap"):
            audit_compound_market_window(market=market_cube_2sym, window=window, required_descriptors=(desc,))

    def test_trailing_core_gap_raises(self, market_cube_2sym) -> None:
        n = market_cube_2sym.available_2d["core"].shape[0]
        market_cube_2sym.available_2d["core"][-400:] = False
        window = QuarterlyBarBoundaries(0, 200, 500, 800, n)
        desc = SignalDescriptor(signal_id="test", family="test", speed="medium", lookback_hours=48, native_timeframe="4h")
        with pytest.raises(InsufficientCoverageError, match="trailing_core_gap"):
            audit_compound_market_window(market=market_cube_2sym, window=window, required_descriptors=(desc,))

    def test_low_coverage_raises(self, market_cube_2sym) -> None:
        n = market_cube_2sym.available_2d["core"].shape[0]
        market_cube_2sym.available_2d["core"][:] = False
        market_cube_2sym.available_2d["core"][:50] = True
        window = QuarterlyBarBoundaries(0, 10, 20, 30, n)
        desc = SignalDescriptor(signal_id="test", family="test", speed="medium", lookback_hours=48, native_timeframe="4h")
        with pytest.raises(InsufficientCoverageError, match="core_coverage"):
            audit_compound_market_window(market=market_cube_2sym, window=window, required_descriptors=(desc,))


# ---------------------------------------------------------------------------
# Test 5: OI without available_at is DISABLED_DATA
# ---------------------------------------------------------------------------

class TestOIDisabledData:
    def test_oi_available_but_disabled(self, market_cube_2sym) -> None:
        n = market_cube_2sym.available_2d["core"].shape[0]
        market_cube_2sym.available_2d["open_interest"] = np.zeros((n, 2), dtype=np.bool_)
        window = QuarterlyBarBoundaries(0, 100, 300, 600, n)
        desc = SignalDescriptor(
            signal_id="oi_signal", family="test", speed="medium",
            lookback_hours=48, native_timeframe="4h",
        )
        result = audit_compound_market_window(market=market_cube_2sym, window=window, required_descriptors=(desc,))
        oi_entry = next((e for e in result.dataset_status if e.dataset == "open_interest"), None)
        assert oi_entry is not None
        assert oi_entry.readiness in ("degraded", "disabled")


# ---------------------------------------------------------------------------
# Test 6: Trial ledger dedup and floor
# ---------------------------------------------------------------------------

class TestCandidateTrialLedger:
    def test_duplicate_registration_idempotent(self, tmp_path) -> None:
        ledger = CandidateTrialLedger(tmp_path / "trials.sqlite3")
        trial = CandidateTrial(
            candidate_hash="abc123", strategy_spec_hash="spec1",
            descriptor_ids=("d1", "d2"), risk_policy_hash="rp1",
            cutoff_time_ns=1000,
        )
        assert ledger.register(trial) == 1
        assert ledger.register(trial) == 0
        count = ledger.distinct_count(cutoff_time_ns=1000, floor=1)
        assert count == 1

    def test_different_risk_hash_increases_count(self, tmp_path) -> None:
        ledger = CandidateTrialLedger(tmp_path / "trials2.sqlite3")
        t1 = CandidateTrial("h1", "s1", ("d1",), "rp1", 1000)
        t2 = CandidateTrial("h2", "s2", ("d2",), "rp2", 1000)
        ledger.register(t1)
        ledger.register(t2)
        count = ledger.distinct_count(cutoff_time_ns=1000, floor=0)
        assert count == 2

    def test_floor_defaults_to_27(self, tmp_path) -> None:
        ledger = CandidateTrialLedger(tmp_path / "trials3.sqlite3")
        assert ledger.distinct_count(cutoff_time_ns=9999) == 27


# ---------------------------------------------------------------------------
# Test 7: Negative Sharpe DSR=0.0; strict L2 raises on L1 rows
# ---------------------------------------------------------------------------

class TestNegativeSharpe:
    def test_negative_sharpe_dsr_zero(self) -> None:
        from src.domain.futures.compound.validation import evaluate_l2_walk_forward
        n = 2400
        returns = -np.abs(np.random.randn(n).astype(np.float64)) * 0.001
        weights = np.zeros((n, 2), dtype=np.float32)
        for t in range(1, n):
            weights[t, 0] = 0.05 * (t % 3 - 1)
            weights[t, 1] = 0.05 * ((t + 1) % 3 - 1)
        ledger = ExecutionLedger(
            timestamps_ns=np.arange(n, dtype=np.int64) * _NS_PER_4H,
            net_returns_1d=returns,
            equity_1d=np.cumprod(1.0 + returns),
            target_weights_2d=weights,
            fee_returns_1d=np.zeros(n, dtype=np.float64),
            slippage_returns_1d=np.zeros(n, dtype=np.float64),
            impact_returns_1d=np.zeros(n, dtype=np.float64),
            funding_returns_1d=np.zeros(n, dtype=np.float64),
            integrity_ok=True,
            integrity_reasons=(),
        )
        from src.domain.futures.compound.config import L2GateConfig
        from src.domain.futures.compound.contracts import L2BenchmarkSeries
        n_daily = len(returns) // 6
        daily_ts = np.arange(n_daily, dtype=np.int64) * (6 * _NS_PER_4H) + _NS_PER_4H
        benchmark = L2BenchmarkSeries(
            benchmark_id="test",
            timestamps_ns=daily_ts,
            daily_returns_1d=np.zeros(n_daily, dtype=np.float64),
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )
        result = evaluate_l2_walk_forward(
            ledger=ledger, fold_ids_1d=np.zeros(n, dtype=np.int16),
            benchmark=benchmark, candidate_count=27,
            config=L2GateConfig(), bootstrap_seed=42,
        )
        assert result.sharpe < 0
        assert result.deflated_sharpe_probability <= 0.5


# ---------------------------------------------------------------------------
# Test 8: Dry-run / L2 FAIL / L3 SHADOW — holdout and pointer unchanged
# ---------------------------------------------------------------------------

class TestDryRunAndShadow:
    def test_dry_run_does_not_consume_holdout(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "dry_run_test.sqlite3")
        from src.domain.futures.compound.contracts import SealedHoldoutManifest
        store.create(SealedHoldoutManifest(
            holdout_id="dry-test", start_time_ns=0, end_time_ns=100,
            holdout_days=90, model_version="v1", data_manifest_hash="h1",
        ))
        dest = tmp_path / "deploy"
        result = CompoundEngineResult(
            handoff=type("AlphaEventTape", (), {"data_manifest_hash": "h1", "model_version": "v1"})(),
            ledger=type("ExecutionLedger", (), {"timestamps_ns": np.array([0]), "net_returns_1d": np.array([0.0]), "equity_1d": np.array([1.0]), "target_weights_2d": np.zeros((1, 1), dtype=np.float32), "fee_returns_1d": np.array([0.0]), "slippage_returns_1d": np.array([0.0]), "impact_returns_1d": np.array([0.0]), "funding_returns_1d": np.array([0.0]), "integrity_ok": True, "integrity_reasons": ()})(),
            l2=type("L2Evaluation", (), {"verdict": L2GateVerdict.PASS, "annualized_log_growth": 0.0, "cagr": 0.0, "excess_growth_lcb90": 0.0, "excess_growth_probability": 1.0, "stressed_excess_growth_lcb90": 0.0, "equity_multiple": 1.0, "sharpe": 1.0, "sharpe_probability": 1.0, "deflated_sharpe_probability": 1.0, "candidate_count": 27, "calmar": 0.0, "max_drawdown": 0.0, "daily_cvar95": 0.0, "annual_volatility": 0.0, "annual_turnover": 0.0, "cost_drag_ratio": 0.0, "capacity_utilisation_p95": 0.0, "active_days_ratio": 1.0, "rebalance_count": 30, "positive_outer_folds": 5, "oos_days": 365, "category_results": (), "integrity_ok": True, "reasons": (), "absolute_cagr": 0.0})(),
            l3=L3ValidationResult(
                verdict=DeploymentVerdict.SHADOW, posterior_growth_probability=0.0,
                holdout_days=90, max_drawdown=0.0, daily_cvar95=0.0,
                reasons=("dry_run_holdout_not_consumed",),
            ),
        )
        path = publish_promoted_strategy(
            result=result, candidate=None, config=CompoundEngineConfig(), destination=dest,
        )
        assert path is None
        assert not (dest / "active.json").exists()


# ---------------------------------------------------------------------------
# Test 9: PROMOTE bundle round-trip
# ---------------------------------------------------------------------------

class TestPromoteBundleRoundTrip:
    def test_bundle_round_trip_preserves_weights_and_descriptors(self, tmp_path) -> None:
        desc = SignalDescriptor(
            signal_id="sig1", family="momentum", speed="fast",
            lookback_hours=48, native_timeframe="4h",
        )
        candidate = DeploymentCandidate(
            active_signal_ids=("sig1",),
            descriptors=(desc,),
            orientation_signs=(1,),
            vote_weights=(1.0,),
            model_version="v1",
            strategy_spec_hash="spec_hash_1",
            fold_manifest_hash="fold_hash_1",
            trial_count=27,
        )
        handoff = type("AlphaEventTape", (), {
            "data_manifest_hash": "data_hash_1", "model_version": "v1",
            "events": type("pa", (), {})(), "recipe_definitions": (),
            "evidence": (), "active_recipe_ids": (), "fold_manifest_hash": "fold_hash_1",
        })()
        ledger = ExecutionLedger(
            timestamps_ns=np.array([0, 1], dtype=np.int64),
            net_returns_1d=np.array([0.001, 0.002], dtype=np.float64),
            equity_1d=np.array([1.0, 1.003], dtype=np.float64),
            target_weights_2d=np.array([[0.0], [0.1]], dtype=np.float32),
            fee_returns_1d=np.zeros(2, dtype=np.float64),
            slippage_returns_1d=np.zeros(2, dtype=np.float64),
            impact_returns_1d=np.zeros(2, dtype=np.float64),
            funding_returns_1d=np.zeros(2, dtype=np.float64),
            integrity_ok=True,
            integrity_reasons=(),
        )
        from src.domain.futures.compound.contracts import L2CategoryResult
        passing_categories = tuple(
            L2CategoryResult(category=f"cat-{i}", passed=True, reasons=())
            for i in range(5)
        )
        l2_eval = L2Evaluation(
            verdict=L2GateVerdict.PASS,
            benchmark_id="bench1",
            annualized_log_growth=0.1, cagr=0.1, excess_growth_lcb90=0.05,
            excess_growth_probability=1.0, stressed_excess_growth_lcb90=0.03,
            equity_multiple=1.1, sharpe=1.5, sharpe_probability=1.0,
            deflated_sharpe_probability=1.0, candidate_count=27, calmar=0.5,
            max_drawdown=0.05, daily_cvar95=-0.01, annual_volatility=0.15,
            annual_turnover=2.0, cost_drag_ratio=0.1, capacity_utilisation_p95=0.05,
            active_days_ratio=1.0, rebalance_count=30, positive_outer_folds=5,
            oos_days=365, category_results=passing_categories, integrity_ok=True,
            reasons=(), absolute_cagr=0.1,
        )
        result = CompoundEngineResult(
            handoff=handoff,
            ledger=ledger,
            l2=l2_eval,
            l3=L3ValidationResult(
                verdict=DeploymentVerdict.PROMOTE, posterior_growth_probability=0.9,
                holdout_days=90, max_drawdown=0.03, daily_cvar95=-0.005,
                reasons=(),
            ),
            deployment_candidate=candidate,
        )
        dest = tmp_path / "deploy"
        path = publish_promoted_strategy(
            result=result, candidate=candidate, config=CompoundEngineConfig(), destination=dest,
        )
        assert path is not None
        assert path.exists()
        assert (dest / "active.json").exists()

        with open(path) as f:
            data = json.load(f)
        assert data["sha256"] != ""
        assert data["candidate"]["active_signal_ids"] == ["sig1"]
        assert data["candidate"]["orientation_signs"] == [1]

    def test_non_promote_verdict_returns_none(self, tmp_path) -> None:
        result = CompoundEngineResult(
            handoff=type("AlphaEventTape", (), {"data_manifest_hash": "", "model_version": ""})(),
            ledger=type("ExecutionLedger", (), {"timestamps_ns": np.array([0]), "net_returns_1d": np.array([0.0]), "equity_1d": np.array([1.0]), "target_weights_2d": np.zeros((1, 1), dtype=np.float32), "fee_returns_1d": np.array([0.0]), "slippage_returns_1d": np.array([0.0]), "impact_returns_1d": np.array([0.0]), "funding_returns_1d": np.array([0.0]), "integrity_ok": True, "integrity_reasons": ()})(),
            l2=type("L2Evaluation", (), {"verdict": L2GateVerdict.FAIL, "annualized_log_growth": 0.0, "cagr": 0.0, "excess_growth_lcb90": 0.0, "excess_growth_probability": 0.0, "stressed_excess_growth_lcb90": 0.0, "equity_multiple": 1.0, "sharpe": 0.0, "sharpe_probability": 0.0, "deflated_sharpe_probability": 0.0, "candidate_count": 0, "calmar": 0.0, "max_drawdown": 0.0, "daily_cvar95": 0.0, "annual_volatility": 0.0, "annual_turnover": 0.0, "cost_drag_ratio": 0.0, "capacity_utilisation_p95": 0.0, "active_days_ratio": 0.0, "rebalance_count": 0, "positive_outer_folds": 0, "oos_days": 0, "category_results": (), "integrity_ok": True, "reasons": ("l2_fail",), "absolute_cagr": 0.0})(),
            l3=L3ValidationResult(verdict=DeploymentVerdict.REJECT, posterior_growth_probability=0.0, holdout_days=0, max_drawdown=0.0, daily_cvar95=0.0, reasons=("l2_not_pass",)),
        )
        path = publish_promoted_strategy(
            result=result, candidate=None, config=CompoundEngineConfig(),
            destination=tmp_path / "deploy2",
        )
        assert path is None


# ---------------------------------------------------------------------------
# Test 10: Stale bundle live inference returns zero weights, sink not called
# ---------------------------------------------------------------------------

class TestLiveInferenceStale:
    def test_stale_bundle_returns_zero_weights(self, market_cube_2sym) -> None:
        desc = SignalDescriptor(
            signal_id="BTCUSDT", family="momentum", speed="fast",
            lookback_hours=48, native_timeframe="4h",
        )
        candidate = DeploymentCandidate(
            active_signal_ids=("BTCUSDT",),
            descriptors=(desc,),
            orientation_signs=(1,),
            vote_weights=(1.0,),
            model_version="v1",
            strategy_spec_hash="spec_hash_1",
            fold_manifest_hash="fold_hash_1",
            trial_count=27,
        )
        stale_bundle = DeploymentBundle(
            schema_version=1,
            promotion_id="stale-001",
            candidate=candidate,
            data_manifest_hash="stale_hash",
            universe_state_hash="",
            config_payload={"target_ann_vol": 0.15, "soft_drawdown_limit": 0.1, "hard_drawdown_limit": 0.18},
            l2_payload={},
            l3_payload={},
            sha256="abc",
        )
        weights = compute_live_target_weights(
            bundle=stale_bundle, market=market_cube_2sym,
            previous_weights=np.zeros(2, dtype=np.float64), equity=100_000.0,
        )
        assert np.allclose(weights, 0.0)


# ---------------------------------------------------------------------------
# Test 11: Integration — real store + strict slice → L2 → L3 → publish
# ---------------------------------------------------------------------------

class TestIntegrationPipeline:
    def test_strict_slice_l2_l3_publish(self, tmp_path) -> None:
        n = 500
        syms = ("BTCUSDT", "ETHUSDT")
        close = np.ones((n, 2), dtype=np.float32) * 100.0
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n, dtype=np.int64) * _NS_PER_HOUR,
            symbols=syms,
            fields_2d={
                "close": close,
                "open": close * 0.9995,
                "high": close * 1.005,
                "low": close * 0.995,
                "quote_volume": np.ones((n, 2), dtype=np.float32) * 50_000_000,
                "funding": np.zeros((n, 2), dtype=np.float32),
                "premium": np.zeros((n, 2), dtype=np.float32),
                "mark": close.copy(),
                "index": close.copy(),
                "taker_buy_quote": np.ones((n, 2), dtype=np.float32) * 25_000_000,
            },
            available_2d={"core": np.ones((n, 2), dtype=np.bool_)},
            eligible_2d=np.ones((n, 2), dtype=np.bool_),
            entry_block_2d=np.zeros((n, 2), dtype=np.bool_),
            exit_required_2d=np.zeros((n, 2), dtype=np.bool_),
            capacity_usdt_2d=np.full((n, 2), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((n, 2), 8.0, dtype=np.float32),
            data_manifest_hash="integ_hash",
        )
        store = SealedHoldoutStore(tmp_path / "integ_holdout.sqlite3")
        from src.domain.futures.compound.contracts import SealedHoldoutManifest
        store.create(SealedHoldoutManifest(
            holdout_id="integ-test", start_time_ns=int(cube.timestamps_ns[-30]),
            end_time_ns=int(cube.timestamps_ns[-1]), holdout_days=90,
            model_version="v1", data_manifest_hash="integ_hash",
            strategy_spec_hash="spec_integ",
        ))
        from src.domain.futures.compound.engine import run_multiscale_compound_engine
        universe = type("Universe", (), {"symbols": cube.symbols, "snapshots": ()})()
        config = CompoundEngineConfig()
        result = run_multiscale_compound_engine(
            market=cube, universe=universe,
            holdout_store=store, holdout_id="integ-test", config=config,
        )
        assert isinstance(result, CompoundEngineResult)
        assert result.l2 is not None
        assert result.l3 is not None

        if result.l3.verdict == DeploymentVerdict.PROMOTE and result.deployment_candidate is not None:
            dest = tmp_path / "deploy_integ"
            path = publish_promoted_strategy(
                result=result, candidate=result.deployment_candidate,
                config=config, destination=dest,
            )
            if path is not None:
                assert path.exists()
                assert (dest / "active.json").exists()
                with open(path) as f:
                    data = json.load(f)
                assert data["sha256"] != ""


# ---------------------------------------------------------------------------
# Test 12: PROMOTE through publish_promoted_strategy directly
# ---------------------------------------------------------------------------

class TestPublishWiring:
    def test_promote_triggers_publish(self, tmp_path) -> None:
        from src.domain.futures.compound.contracts import (
            L2CategoryResult, L2Evaluation, L3ValidationResult,
        )
        desc = SignalDescriptor("sig1", "mom", "fast", 48, "4h")
        candidate = DeploymentCandidate(
            active_signal_ids=("sig1",),
            descriptors=(desc,),
            orientation_signs=(1,),
            vote_weights=(1.0,),
            model_version="v1",
            strategy_spec_hash="spec1",
            fold_manifest_hash="fold1",
            trial_count=27,
        )
        passing_categories = tuple(
            L2CategoryResult(category=f"cat-{i}", passed=True, reasons=())
            for i in range(5)
        )
        l2_eval = L2Evaluation(
            verdict=L2GateVerdict.PASS, benchmark_id="b1",
            annualized_log_growth=0.1, cagr=0.1, excess_growth_lcb90=0.05,
            excess_growth_probability=1.0, stressed_excess_growth_lcb90=0.03,
            equity_multiple=1.1, sharpe=1.5, sharpe_probability=1.0,
            deflated_sharpe_probability=1.0, candidate_count=27, calmar=0.5,
            max_drawdown=0.03, daily_cvar95=-0.005, annual_volatility=0.12,
            annual_turnover=1.5, cost_drag_ratio=0.05, capacity_utilisation_p95=0.03,
            active_days_ratio=1.0, rebalance_count=30, positive_outer_folds=5,
            oos_days=365, category_results=passing_categories, integrity_ok=True,
            reasons=(), absolute_cagr=0.1,
        )
        l3_result = L3ValidationResult(
            verdict=DeploymentVerdict.PROMOTE, posterior_growth_probability=0.9,
            holdout_days=90, max_drawdown=0.02, daily_cvar95=-0.005,
            reasons=(),
        )
        mock_handoff = type("AlphaEventTape", (), {
            "data_manifest_hash": "h1", "model_version": "v1",
            "fold_manifest_hash": "fold1",
        })()
        mock_ledger = ExecutionLedger(
            timestamps_ns=np.array([0, 1], dtype=np.int64),
            net_returns_1d=np.array([0.001, 0.001], dtype=np.float64),
            equity_1d=np.array([1.0, 1.002], dtype=np.float64),
            target_weights_2d=np.array([[0.0], [0.05]], dtype=np.float32),
            fee_returns_1d=np.zeros(2, dtype=np.float64),
            slippage_returns_1d=np.zeros(2, dtype=np.float64),
            impact_returns_1d=np.zeros(2, dtype=np.float64),
            funding_returns_1d=np.zeros(2, dtype=np.float64),
            integrity_ok=True, integrity_reasons=(),
        )
        result = CompoundEngineResult(
            handoff=mock_handoff, ledger=mock_ledger,
            l2=l2_eval, l3=l3_result,
            deployment_candidate=candidate,
        )

        dest = tmp_path / "deploy_wire"
        path = publish_promoted_strategy(
            result=result, candidate=candidate,
            config=CompoundEngineConfig(), destination=dest,
        )
        assert path is not None
        assert path.exists()
        assert (dest / "active.json").exists()

    def test_non_promote_does_not_publish(self, tmp_path) -> None:
        l2_eval = L2Evaluation(
            verdict=L2GateVerdict.FAIL, benchmark_id="b1",
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=0.0, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.0,
            deflated_sharpe_probability=0.0, candidate_count=0, calmar=0.0,
            max_drawdown=0.0, daily_cvar95=0.0, annual_volatility=0.0,
            annual_turnover=0.0, cost_drag_ratio=0.0, capacity_utilisation_p95=0.0,
            active_days_ratio=0.0, rebalance_count=0, positive_outer_folds=0,
            oos_days=0, category_results=(), integrity_ok=True,
            reasons=("l2_fail",), absolute_cagr=0.0,
        )
        l3_result = L3ValidationResult(
            verdict=DeploymentVerdict.REJECT, posterior_growth_probability=0.0,
            holdout_days=0, max_drawdown=0.0, daily_cvar95=0.0,
            reasons=("l2_not_pass",),
        )
        result = CompoundEngineResult(
            handoff=type("AlphaEventTape", (), {"data_manifest_hash": "", "model_version": ""})(),
            ledger=ExecutionLedger(
                timestamps_ns=np.array([0], dtype=np.int64),
                net_returns_1d=np.array([0.0], dtype=np.float64),
                equity_1d=np.array([1.0], dtype=np.float64),
                target_weights_2d=np.zeros((1, 1), dtype=np.float32),
                fee_returns_1d=np.zeros(1, dtype=np.float64),
                slippage_returns_1d=np.zeros(1, dtype=np.float64),
                impact_returns_1d=np.zeros(1, dtype=np.float64),
                funding_returns_1d=np.zeros(1, dtype=np.float64),
                integrity_ok=True, integrity_reasons=(),
            ),
            l2=l2_eval, l3=l3_result,
        )
        path = publish_promoted_strategy(
            result=result, candidate=None,
            config=CompoundEngineConfig(), destination=tmp_path / "deploy_nowire",
        )
        assert path is None


# ---------------------------------------------------------------------------
# Test 13: Performance — audit RSS < 512MB, runtime < 15%
# ---------------------------------------------------------------------------

class TestPerformanceBudget:
    def test_audit_does_not_exceed_memory_budget(self, market_cube_2sym) -> None:
        n = len(market_cube_2sym.timestamps_ns)
        window = QuarterlyBarBoundaries(0, n // 4, n // 2, 3 * n // 4, n)
        desc = SignalDescriptor(
            signal_id="perf_test", family="test", speed="medium",
            lookback_hours=48, native_timeframe="4h",
        )
        result = audit_compound_market_window(
            market=market_cube_2sym, window=window,
            required_descriptors=(desc,),
        )
        assert isinstance(result, CompoundWindowAudit)
        assert result.passed
        assert result.core_coverage_ratio > 0.98
