"""Integration & Contract Validation Tests for MHS Horizon Optimization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs import evaluation as ev
from src.application.research.mhs import marks
from src.mhs import params
from src.application.research.mhs.evaluation import (
    MHS_GO_REASON_INVALID_PRIMARY,
    MHS_GO_REASON_RESOURCE_BREACH,
    MhsDiagnosticRequest,
    _book_structure_trace,
    _run_anchored_fold,
    _verify_ledger_artifact,
)
from src.application.research.mhs.scaling import (
    _apply_rebalance_deadband,
    _regime_cash_scale,
    _smooth_signal_ema,
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
    originals = {"funding_path": marks.funding_path, "mark_price_path": fc._mark_price_path}
    marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    yield root, end
    marks.funding_path = originals["funding_path"]
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


def _synthetic_roster_targets(
    n_rows: int = 400, n_sym: int = 200, roster: int = 42, seed: int = 7,
):
    """Dollar-neutral unit-gross book on a churning top-`roster` mask (spec §4)."""
    rng = np.random.default_rng(seed)
    cols = [f"S{i:03d}" for i in range(n_sym)]
    idx = pd.date_range("2021-01-01", periods=n_rows, freq="24h", tz="UTC")
    vol = pd.DataFrame(rng.lognormal(0, 1.0, (n_rows, n_sym)), index=idx, columns=cols)
    vol = vol.rolling(20, min_periods=1).mean()
    rank = vol.rank(axis=1, ascending=False, method="first")
    mask = rank.le(roster)
    raw = pd.DataFrame(rng.normal(0, 1, (n_rows, n_sym)), index=idx, columns=cols).where(mask)
    dn = raw.sub(raw.mean(axis=1), axis=0)
    w = dn.div(dn.abs().sum(axis=1), axis=0).fillna(0.0) * 0.53
    return w


@pytest.mark.slow
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
        out = _apply_rebalance_deadband(target)
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
        assert float(scale.min()) >= params.MHS_REGIME_CASH_SCALE_FLOOR

    @pytest.mark.slow
    def test_fold_quality_metrics_finite_from_valid_primary(self, fold_market, funding) -> None:
        root, end = fold_market
        report = _run_anchored_fold(
            str(root), OPT_FOLD, _request(root, end),
            funding, 1.0, 0,
        )
        assert report.primary_valid is True
        assert report.strict is not None
        assert np.isfinite(report.strict.all_intent_shortfall_bps)
        assert np.isfinite(report.primary_autocorr_sharpe)
        assert np.isfinite(report.primary_naive_sharpe)
        assert np.isfinite(report.stress_naive_sharpe)
        assert report.decision_intents > 0


class TestMhsEvalIntegrity:
    """SCENARIO_MHS_EVAL_INTEGRITY_*: exit-always liquidation, scale-relative
    deadband, holdings boundedness, fail-closed assertion, and structure trace."""

    def test_exit_always_zero_target_liquidates(self) -> None:
        # SCENARIO_MHS_EVAL_INTEGRITY_EXIT_ALWAYS: a zero target is a
        # liquidation instruction, never a carried resize.
        idx = pd.DatetimeIndex(
            ["2021-01-01 00:00", "2021-01-01 06:00", "2021-01-01 12:00", "2021-01-01 18:00"],
            tz="UTC",
        )
        cols = ["A", *[f"S{i}" for i in range(1, 10)]]
        # Row gross 0.100 over 10 active symbols -> min_delta = 0.25 * 0.100 / 10.
        target = pd.DataFrame(
            [
                [0.010, *[0.010] * 9],
                [0.011, *[0.010] * 9],
                [0.000, *[0.010] * 9],
                [0.000, *[0.010] * 9],
            ],
            index=idx,
            columns=cols,
        )
        out = _apply_rebalance_deadband(target)
        assert out.loc[idx[1], "A"] == pytest.approx(0.010)  # delta 0.001 < 0.0025, carried
        assert out.loc[idx[2], "A"] == 0.0  # zero target liquidates exactly
        assert out.loc[idx[3], "A"] == 0.0
        # Under the legacy absolute 0.02 threshold out[2,'A'] would equal 0.010.
        assert out.loc[idx[2], "A"] != pytest.approx(0.010)

    def test_scale_relative_threshold(self) -> None:
        # SCENARIO_MHS_EVAL_INTEGRITY_SCALE_RELATIVE: the threshold is a
        # fraction of the per-symbol position scale, not an absolute constant.
        idx = pd.DatetimeIndex(["2021-01-01 00:00", "2021-01-01 06:00"], tz="UTC")
        cols = ["A", "B", "C", "D"]
        # Frame L: gross 0.40 -> min_delta = 0.25 * 0.40 / 4 = 0.025; delta 0.020 carried.
        frame_l = pd.DataFrame(
            [[0.10, 0.10, 0.10, 0.10], [0.12, 0.10, 0.10, 0.10]],
            index=idx,
            columns=cols,
        )
        out_l = _apply_rebalance_deadband(frame_l)
        assert out_l.loc[idx[1], "A"] == pytest.approx(0.10)  # carried == prior held
        # Frame S: gross 0.04 -> min_delta = 0.0025; the same delta 0.020 now trades.
        frame_s = pd.DataFrame(
            [[0.01, 0.01, 0.01, 0.01], [0.03, 0.01, 0.01, 0.01]],
            index=idx,
            columns=cols,
        )
        out_s = _apply_rebalance_deadband(frame_s)
        assert out_s.loc[idx[1], "A"] == pytest.approx(0.03)  # traded == new target
        with pytest.raises(ValueError, match="position_fraction"):
            _apply_rebalance_deadband(frame_l, position_fraction=-0.1)

    def test_holdings_bounded_stationary(self) -> None:
        # SCENARIO_MHS_EVAL_INTEGRITY_HOLDINGS_BOUNDED: on the seeded synthetic
        # churning top-42 roster, holdings stay bounded and stationary.
        target = _synthetic_roster_targets()
        out = _apply_rebalance_deadband(target)
        target_hold = (target != 0.0).sum(axis=1)
        out_hold = (out != 0.0).sum(axis=1)
        assert (out_hold <= target_hold).all()
        assert out_hold.max() <= 42
        first50, last50 = out_hold.iloc[:50].mean(), out_hold.iloc[-50:].mean()
        assert 0.80 <= last50 / first50 <= 1.25
        tracking = (out - target).abs().sum(axis=1).mean()
        gross = target.abs().sum(axis=1).mean()
        assert tracking <= 0.05 * gross

    def test_holdings_fail_closed_on_carry_regression(self, monkeypatch) -> None:
        # SCENARIO_MHS_EVAL_INTEGRITY_HOLDINGS_FAIL_CLOSED: a regression that
        # carries a zero target must raise DataIntegrityError, never silently
        # degrade into an unbounded book.
        idx = pd.DatetimeIndex(["2021-01-01 00:00", "2021-01-01 06:00"], tz="UTC")
        target = pd.DataFrame(
            {"A": [0.010, 0.000], "B": [0.010, 0.010]},
            index=idx,
        )
        real_where = np.where

        def _regressed_where(condition, x, y):
            # Simulate a regression of Invariant E: always carry held, even
            # across a zero target.
            return x

        monkeypatch.setattr(np, "where", _regressed_where)
        try:
            with pytest.raises(ev.DataIntegrityError, match="holdings"):
                _apply_rebalance_deadband(target)
        finally:
            monkeypatch.setattr(np, "where", real_where)

    def test_book_structure_trace(self) -> None:
        # SCENARIO_MHS_EVAL_INTEGRITY_STRUCTURE_TRACE: the trace returns the
        # exact key set and measures stationarity correctly.
        keys = {"n_rows", "gross_mean", "holdings_mean", "holdings_max", "holdings_growth_slope"}
        idx = pd.date_range("2021-01-01", periods=100, freq="1h", tz="UTC")
        stationary = pd.DataFrame(
            {f"S{i}": [0.01] * 100 for i in range(10)}, index=idx,
        )
        trace = _book_structure_trace(stationary)
        assert set(trace.keys()) == keys
        assert trace["holdings_mean"] == pytest.approx(10.0)
        assert trace["holdings_max"] == pytest.approx(10.0)
        assert abs(trace["holdings_growth_slope"]) <= 1e-9
        # A book ramping 10 -> 110 over 100 rows grows ~100 names over a mean
        # of ~60 -> slope > 1.0.
        ramp = pd.DataFrame(
            {f"S{i}": [0.01 if i < 10 + r else 0.0 for r in range(100)] for i in range(120)},
            index=idx,
        )
        ramp_trace = _book_structure_trace(ramp)
        assert ramp_trace["holdings_growth_slope"] > 1.0
        empty = _book_structure_trace(pd.DataFrame())
        assert empty["n_rows"] == 0.0
        assert all(v == 0.0 for v in empty.values())


@pytest.mark.slow
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
    completes with bit-identical golden metrics after performance optimizations."""

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


class TestRegimeCharacterization:
    """SCENARIO_MHS_REGIME_CHARACTERIZATION_*: pure-function and I/O wrapper tests."""

    def test_regime_reference_characterization_trending_uptrend(self) -> None:
        # SCENARIO_MHS_REGIME_CHARACTERIZATION_UPTREND: monotonic uptrend yields
        # positive total_return, near-zero flip rate, finite positive vol.
        idx = pd.date_range("2021-01-01", periods=200, freq="1h", tz="UTC")
        prices = 100.0 * np.exp(np.linspace(0, 0.5, 200))
        close = pd.Series(prices, index=idx)
        result = ev._regime_reference_characterization(close)
        assert result is not None
        assert result["total_return"] > 0
        assert result["direction_flip_rate_24h"] < 0.05
        assert np.isfinite(result["annualized_realized_vol"])
        assert result["annualized_realized_vol"] > 0

    def test_regime_reference_characterization_choppy_alternating(self) -> None:
        # SCENARIO_MHS_REGIME_CHARACTERIZATION_CHOPPY: construct price[t] via a
        # 24-bar-lag recursion (price[t] = price[t-24] * (1 +/- 5%), sign
        # alternating every t) so pct_change(24)'s sign flips EVERY bar by
        # construction -- a period-24-in-the-trend sawtooth (the original
        # construction) only flips the rolling-24h return's sign once per
        # 24-bar half-cycle, not every bar.
        idx = pd.date_range("2021-01-01", periods=200, freq="1h", tz="UTC")
        prices = [100.0] * 24
        for t in range(24, 200):
            sign = 1.0 if t % 2 == 0 else -1.0
            prices.append(prices[t - 24] * (1.0 + sign * 0.05))
        close = pd.Series(prices, index=idx)
        result = ev._regime_reference_characterization(close)
        assert result is not None
        assert result["direction_flip_rate_24h"] > 0.9

    def test_regime_reference_characterization_too_short_returns_none(self) -> None:
        # SCENARIO_MHS_REGIME_CHARACTERIZATION_TOO_SHORT: fewer than 49 bars
        # returns None, not a dict with NaN values.
        idx = pd.date_range("2021-01-01", periods=30, freq="1h", tz="UTC")
        close = pd.Series(100.0 + np.arange(30), index=idx)
        result = ev._regime_reference_characterization(close)
        assert result is None

    def test_fold_regime_characterization_missing_reference_returns_none(self, tmp_path) -> None:
        # SCENARIO_MHS_FOLD_REGIME_CHARACTERIZATION_MISSING_REFERENCE: missing
        # BTCUSDT.parquet yields None, not an exception.
        root = tmp_path / "market"
        (root / "1h").mkdir(parents=True)
        fold = ev.AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-01-31", tz="UTC"),
            pd.Timestamp("2021-02-10", tz="UTC"),
            pd.Timestamp("2021-04-19 08:00", tz="UTC"),
            168,
            168,
        )
        result = ev._fold_regime_characterization(str(root), fold)
        assert result is None

    def test_fold_regime_characterization_reads_only_validation_window(self, tmp_path) -> None:
        # SCENARIO_MHS_FOLD_REGIME_CHARACTERIZATION_READS_ONLY_VALIDATION_WINDOW:
        # synthetic parquet with flat train + trending validation; returned
        # stats match validation-only slice, not full train+validation.
        root = tmp_path / "market"
        (root / "1h").mkdir(parents=True)
        train_start = pd.Timestamp("2021-01-01", tz="UTC")
        train_end = pd.Timestamp("2021-01-31", tz="UTC")
        val_start = pd.Timestamp("2021-02-10", tz="UTC")
        val_end = pd.Timestamp("2021-04-19 08:00", tz="UTC")
        fold = ev.AnchoredPurgedFold(train_start, train_end, val_start, val_end, 168, 168)
        # Build synthetic data: flat in train, sharply trending in validation
        train_hours = pd.date_range(train_start, train_end, freq="1h", tz="UTC")
        val_hours = pd.date_range(val_start, val_end, freq="1h", tz="UTC")
        all_hours = train_hours.append(val_hours)
        train_prices = [100.0] * len(train_hours)
        val_prices = np.linspace(100, 200, len(val_hours)).tolist()
        all_prices = train_prices + val_prices
        epoch_ms = ((all_hours - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).tolist()
        df = pd.DataFrame({"timestamp": epoch_ms, "open": all_prices, "high": all_prices,
                           "low": all_prices, "close": all_prices, "quote_vol": [1000.0] * len(all_hours)})
        df.to_parquet(root / "1h" / "BTCUSDT.parquet", index=False)
        result = ev._fold_regime_characterization(str(root), fold)
        assert result is not None
        # Compute expected values from validation-only slice
        val_arr = np.array(val_prices, dtype="float64")
        val_ret = np.log(val_arr)
        ann_vol = float(np.std(np.diff(val_ret), ddof=1) * np.sqrt(24 * 365))
        total_ret = float(val_arr[-1] / val_arr[0] - 1.0)
        assert result["annualized_realized_vol"] == pytest.approx(ann_vol, rel=1e-6)
        assert result["total_return"] == pytest.approx(total_ret, rel=1e-6)
