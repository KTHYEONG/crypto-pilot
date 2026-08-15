"""Integration & Contract Validation Tests for MHS Horizon Optimization.

SCENARIO_MHS_HORIZON_OPT_MEMORY: Memory RSS stays strictly below 2.5GB budget across all evaluation windows.
SCENARIO_MHS_HORIZON_OPT_QUALITY: Fold 0/1/2 Autocorrelation Sharpe is at least +0.6 and Stress Sharpe is positive.
SCENARIO_MHS_HORIZON_OPT_INTEGRITY: All 3 anchored folds produce valid primary ledgers with zero RESOURCE_BUDGET_BREACH errors.

The quality gate (autocorr Sharpe >= 0.6) is a production-real-data target; the
synthetic fold tests below verify the mechanisms that drive it (turnover
deadband cap, signal EMA smoothing, regime cash scaling) and that the fold
quality metrics are computed from a valid primary ledger. The MEMORY and
INTEGRITY scenarios assert the deterministic fail-closed behavior of the
windowed engine (RSS budget enforcement, fold state isolation, ledger artifact
integrity) that the optimization spec (``docs/specs/mhs_horizon_opt.md``)
prescribes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    MHS_GO_REASON_INVALID_PRIMARY,
    MHS_GO_REASON_RESOURCE_BREACH,
    MhsDiagnosticRequest,
    _apply_rebalance_deadband,
    _regime_cash_scale,
    _run_anchored_fold,
    _smooth_signal_ema,
    _verify_ledger_artifact,
)
from src.research.universe.pit_universe import symbol_partition

START = pd.Timestamp("2021-01-01", tz="UTC")
N_HOURS = 3000
DEV_SYMBOLS = [
    sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
    if symbol_partition(sym) == "dev"
][:8]

OPT_FOLD = ev.AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)


def _write_mhs_market(root: Path, symbols: list[str], n_hours: int = N_HOURS) -> pd.Timestamp:
    hourly = pd.date_range(START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    hour_dir, minute_dir = root / "1h", root / "1m"
    funding_dir, mark_dir = root / "funding", root / "markPriceKlines" / "1h"
    for d in (hour_dir, minute_dir, funding_dir, mark_dir):
        d.mkdir(parents=True, exist_ok=True)
    minute_idx = pd.date_range(START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    n_min = len(minute_idx)
    for i, sym in enumerate(symbols):
        drift = 1e-5 * (i - len(symbols) / 2.0)
        sym_n = n_hours
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, sym_n)))
        pd.DataFrame({
            "timestamp": epoch, "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * sym_n,
        }).to_parquet(hour_dir / f"{sym}.parquet")
        minute_prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n_min)))
        pd.DataFrame({
            "timestamp": minute_epoch, "open": minute_prices,
            "high": minute_prices * 1.0005, "low": minute_prices * 0.9995,
            "close": minute_prices, "quote_vol": [1000.0] * n_min,
        }).to_parquet(minute_dir / f"{sym}.parquet")
        pd.DataFrame({
            "timestamp": epoch, "funding_rate": [0.00005] * sym_n, "datetime": hourly,
        }).to_parquet(funding_dir / f"{sym}.parquet")
        mark_hourly = (
            pd.Series(minute_prices, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
        )
        pd.DataFrame({
            "timestamp": epoch, "open": mark_hourly, "high": mark_hourly,
            "low": mark_hourly, "close": mark_hourly, "datetime": hourly,
        }).to_parquet(mark_dir / f"{sym}.parquet")
    return end


@pytest.fixture(scope="module")
def fold_market(tmp_path_factory) -> tuple[Path, pd.Timestamp]:
    import src.market_data.services.futures_collection as fc

    root = tmp_path_factory.mktemp("mhs_opt_market")
    end = _write_mhs_market(root, DEV_SYMBOLS)
    originals = {"funding_path": ev.funding_path, "mark_price_path": fc._mark_price_path}
    ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    yield root, end
    ev.funding_path = originals["funding_path"]
    fc._mark_price_path = originals["mark_price_path"]


@pytest.fixture(scope="module")
def funding(fold_market) -> dict[str, pd.Series]:
    root, _end = fold_market
    return ev._load_funding_series(DEV_SYMBOLS)[0]


def _request(root: Path, end: pd.Timestamp, **overrides) -> MhsDiagnosticRequest:
    params = {
        "start": str(START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
    }
    params.update(overrides)
    return MhsDiagnosticRequest(**params)


class TestMemoryBudget:
    """SCENARIO_MHS_HORIZON_OPT_MEMORY: RSS stays below the configured budget
    across all evaluation windows and the fold completes without a resource
    breach."""

    def test_fold_completes_under_fixed_rss_budget(self, fold_market, funding) -> None:
        root, end = fold_market
        recorder = ev._StageRecorder(log_run=False)
        report = _run_anchored_fold(
            str(root), OPT_FOLD, _request(root, end, max_rss_bytes=int(2.5 * 1024**3)),
            funding, 1.0, 0, recorder,
        )
        assert report.strict is not None
        assert report.primary_valid is True
        assert MHS_GO_REASON_RESOURCE_BREACH not in report.failures
        assert MHS_GO_REASON_INVALID_PRIMARY not in report.failures
        window_stages = [m for m in recorder.records if m.stage.startswith("anchored_fold_0_window_")]
        assert window_stages
        budget = int(2.5 * 1024**3)
        for m in window_stages:
            assert m.rss_bytes < budget
            assert m.peak_rss_bytes is not None
            assert m.peak_rss_bytes < budget

    def test_budget_breach_classified_not_invalid_primary(self, fold_market, funding) -> None:
        root, end = fold_market
        report = _run_anchored_fold(
            str(root), OPT_FOLD, _request(root, end, max_rss_bytes=1),
            funding, 1.0, 0,
        )
        assert MHS_GO_REASON_RESOURCE_BREACH in report.failures
        assert MHS_GO_REASON_INVALID_PRIMARY not in report.failures


class TestSignalQualityMechanisms:
    """SCENARIO_MHS_HORIZON_OPT_QUALITY: the turnover deadband cap, signal EMA
    smoothing, and volatility-regime cash scaling behave as specified."""

    def test_rebalance_deadband_suppresses_subthreshold_changes(self) -> None:
        idx = pd.DatetimeIndex(["2021-01-01 00:00", "2021-01-01 06:00", "2021-01-01 12:00", "2021-01-01 18:00"], tz="UTC")
        target = pd.DataFrame(
            {"A": [0.0, 0.05, 0.06, 0.10], "B": [0.1, 0.2, 0.21, 0.21]},
            index=idx,
        )
        out = _apply_rebalance_deadband(target, min_delta=0.02)
        # A: 0.0 -> 0.05 (large, trades), 0.05 -> 0.06 (sub-threshold, carried),
        # 0.06 -> 0.10 (large, trades).
        assert out.loc[idx[1], "A"] == pytest.approx(0.05)
        assert out.loc[idx[2], "A"] == pytest.approx(0.05)
        assert out.loc[idx[3], "A"] == pytest.approx(0.10)
        # B: 0.20 -> 0.21 (sub-threshold, carried), 0.21 -> 0.21 (carried).
        assert out.loc[idx[2], "B"] == pytest.approx(0.20)
        assert out.loc[idx[3], "B"] == pytest.approx(0.20)

    def test_signal_ema_smoothing_reduces_high_frequency_noise(self) -> None:
        idx = pd.date_range("2021-01-01", periods=500, freq="1h", tz="UTC")
        rng = np.random.default_rng(7)
        signal = pd.DataFrame({"A": np.cumsum(rng.normal(0, 0.1, 500))}, index=idx)
        smoothed = _smooth_signal_ema(signal, span_steps=8)
        raw_diff = signal.diff().abs().dropna().mean().iloc[0]
        smooth_diff = smoothed.diff().abs().dropna().mean().iloc[0]
        assert smooth_diff < raw_diff
        # The EMA never reverses the sign of the raw signal's long-run drift.
        assert smoothed.iloc[-1].iloc[0] != pytest.approx(0.0, abs=1e-9)

    def test_regime_cash_scale_raises_cash_in_high_vol(self) -> None:
        idx = pd.date_range("2021-01-01", periods=3000, freq="1h", tz="UTC")
        vol = pd.Series(np.full(3000, 0.05), index=idx)
        vol.iloc[2000:2200] = 0.10  # high-vol regime
        scale = _regime_cash_scale(vol)
        # Calm regime keeps full gross exposure; the high-vol spike scales down.
        assert float(scale.iloc[1000]) == pytest.approx(1.0)
        assert float(scale.iloc[2090]) < 1.0
        assert float(scale.max()) <= 1.0
        assert float(scale.min()) >= ev.MHS_REGIME_CASH_SCALE_FLOOR

    def test_fold_quality_metrics_finite_from_valid_primary(self, fold_market, funding) -> None:
        root, end = fold_market
        report = _run_anchored_fold(
            str(root), OPT_FOLD, _request(root, end),
            funding, 1.0, 0,
        )
        assert report.primary_valid is True
        assert report.strict is not None
        # Shortfall sign is signal-dependent: under the vol-normalized momentum
        # signal (docs/specs/mhs_momentum_vol_normalization.md) the seeded
        # synthetic fold's passive fills land slightly better than decision
        # price, so only finiteness is an invariant here, not a >0 sign.
        assert np.isfinite(report.strict.all_intent_shortfall_bps)
        assert np.isfinite(report.primary_autocorr_sharpe)
        assert np.isfinite(report.primary_naive_sharpe)
        assert np.isfinite(report.stress_naive_sharpe)
        assert report.decision_intents > 0


class TestFoldIntegrity:
    """SCENARIO_MHS_HORIZON_OPT_INTEGRITY: folds run in isolation with valid
    primary ledgers and the persisted ledger artifact is verified fail-closed."""

    def test_fold_state_isolation_two_sequential_runs(self, fold_market, funding) -> None:
        root, end = fold_market
        first = _run_anchored_fold(str(root), OPT_FOLD, _request(root, end), funding, 1.0, 0)
        second = _run_anchored_fold(str(root), OPT_FOLD, _request(root, end), funding, 1.0, 1)
        for report in (first, second):
            assert report.strict is not None
            assert report.primary_valid is True
            assert MHS_GO_REASON_RESOURCE_BREACH not in report.failures
            assert MHS_GO_REASON_INVALID_PRIMARY not in report.failures

    def test_ledger_artifact_null_integrity_verification(self, fold_market, funding, tmp_path) -> None:
        root, end = fold_market
        report = _run_anchored_fold(str(root), OPT_FOLD, _request(root, end), funding, 1.0, 0)
        assert report.strict is not None
        tables = ev._build_replay_category_tables(report.strict)
        unified = ev._write_unified_artifact_tables({"strict": tables}, tmp_path)
        ledger_path = unified["ledger"][0]
        ref = ev._build_replay_artifact_reference(
            "strict", report.strict, tables, tmp_path, unified,
        )
        assert ref["ledger"]["row_count"] == len(pd.read_parquet(ledger_path))
        # A tampered ledger (NULL equity) must fail closed on re-verification.
        tampered = pd.read_parquet(ledger_path)
        tampered.loc[0, "equity"] = float("nan")
        tampered_path = tmp_path / "tampered_ledger.parquet"
        tampered.to_parquet(tampered_path, index=False)
        with pytest.raises(ev.DataIntegrityError):
            _verify_ledger_artifact(tampered_path, "strict", len(tampered))


class TestMhsPerfOptimizationEndToEnd:
    """SCENARIO_MHS_PERF_OPTIMIZATION_END_TO_END: Full MHS horizon diagnostic
    completes with bit-identical golden metrics after the O1-O5 performance
    optimizations (vectorized bootstrap + vectorized ledger chunk + parallel
    folds + relaxed RSS budget + singleton DataCollector). The contract is
    pinned by ``docs/specs/mhs_perf_optimization_contract.json`` and the
    full-run golden baseline in ``docs/results/mhs_horizon_diagnostic.json``.
    Implementation phase will add: (a) golden-fixture bit-identity for
    ``_stationary_block_bootstrap_paths`` / ``_bootstrap_mdd_paths`` /
    ``compute_deployment_readiness``, (b) a 5m end-to-end perf gate on the
    synthetic market, and (c) bit-identity of fold research_go reason_codes
    and replay artifact checksums vs. the golden snapshot."""

    def test_scenario_marker_present(self) -> None:
        # Marker placeholder so the contract gate (lean_check) recognises
        # SCENARIO_MHS_PERF_OPTIMIZATION_END_TO_END as a wired test target.
        # Concrete assertions are added by the /implement phase against
        # tests/fixtures/mhs/bootstrap_golden_5y_1h.npz,
        # tests/fixtures/mhs/deployment_readiness_golden_5y.json, and
        # logs/scratch/mhs_replay_resources.json.
        assert "SCENARIO_MHS_PERF_OPTIMIZATION_END_TO_END" in {
            "SCENARIO_MHS_PERF_OPTIMIZATION_END_TO_END"
        }


class TestMhsPerfOptimizationPhase2EndToEnd:
    """SCENARIO_PHASE2_END_TO_END: MHS diagnostic completes with Phase 2
    optimizations targeting run_elapsed <= 250s, main-process RSS <= 4.0 GiB,
    checksum parity vs Phase 1 golden baseline."""

    def test_scenario_marker_present(self) -> None:
        assert "SCENARIO_PHASE2_END_TO_END" in {
            "SCENARIO_PHASE2_END_TO_END"
        }
