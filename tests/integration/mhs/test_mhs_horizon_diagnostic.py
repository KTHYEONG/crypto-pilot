from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    _bootstrap_ci as _real_bootstrap_ci,
    _placebo_sharpe_percentile as _real_placebo_sharpe_percentile,
    MhsDiagnosticRequest,
    MhsHorizonDiagnosticReport,
    run_mhs_horizon_diagnostic,
)
from src.cli.commands.research.mhs import add_mhs_commands
from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.research.universe.pit_universe import symbol_partition

START = pd.Timestamp("2021-01-01", tz="UTC")
N_HOURS = 2000
DEV_SYMBOLS = [
    sym for sym in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
    if symbol_partition(sym) == "dev"
][:8]


def _write_mhs_market(
    root: Path,
    symbols: list[str],
    late_listings: dict[str, pd.Timestamp] | None = None,
    n_hours: int = N_HOURS,
) -> pd.Timestamp:
    late_listings = late_listings or {}
    hourly = pd.date_range(START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    hour_dir = root / "1h"
    minute_dir = root / "1m"
    five_dir = root / "5m"
    funding_dir = root / "funding"
    mark_dir = root / "markPriceKlines" / "1h"
    hour_dir.mkdir(parents=True, exist_ok=True)
    minute_dir.mkdir(parents=True, exist_ok=True)
    five_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)
    mark_dir.mkdir(parents=True, exist_ok=True)

    n = len(hourly)
    minute_idx = pd.date_range(START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    n_min = len(minute_idx)

    for i, sym in enumerate(symbols):
        sym_start = late_listings.get(sym, START)
        sym_hourly = hourly[hourly >= sym_start]
        sym_epoch = (sym_hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        sym_n = len(sym_hourly)
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, sym_n)))
        pd.DataFrame(
            {
                "timestamp": sym_epoch,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "quote_vol": [1000.0] * sym_n,
            },
        ).to_parquet(hour_dir / f"{sym}.parquet")

        sym_minute = minute_idx[minute_idx >= sym_start]
        sym_minute_epoch = (sym_minute - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        sym_n_min = len(sym_minute)
        minute_prices = 100.0 * np.exp(
            np.cumsum(rng.normal(drift, 0.002, sym_n_min)),
        )
        pd.DataFrame(
            {
                "timestamp": sym_minute_epoch,
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
                "quote_vol": [1000.0] * sym_n_min,
            },
        ).to_parquet(minute_dir / f"{sym}.parquet")

        # 5-minute execution grid for the default ``execution_timeframe="5m"``
        # runs (SCENARIO_MHS_ANNUALIZATION_04), derived from the same minute
        # path so the 1m/5m ledgers share one underlying price process.
        five_frame = pd.DataFrame(
            {
                "open": minute_prices,
                "high": minute_prices * 1.0005,
                "low": minute_prices * 0.9995,
                "close": minute_prices,
            },
            index=sym_minute,
        ).resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"},
        ).dropna()
        pd.DataFrame(
            {
                "timestamp": (five_frame.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
                "open": five_frame["open"].to_numpy(),
                "high": five_frame["high"].to_numpy(),
                "low": five_frame["low"].to_numpy(),
                "close": five_frame["close"].to_numpy(),
                "quote_vol": np.full(len(five_frame), 5000.0),
            },
        ).to_parquet(five_dir / f"{sym}.parquet")

        pd.DataFrame(
            {
                "timestamp": sym_epoch,
                "funding_rate": [0.00005] * sym_n,
                "datetime": sym_hourly,
            },
        ).to_parquet(funding_dir / f"{sym}.parquet")

        mark_hourly = (
            pd.Series(minute_prices, index=sym_minute)
            .resample("1h")
            .last()
            .reindex(sym_hourly)
            .to_numpy()
        )
        pd.DataFrame(
            {
                "timestamp": sym_epoch,
                "open": mark_hourly,
                "high": mark_hourly,
                "low": mark_hourly,
                "close": mark_hourly,
                "datetime": sym_hourly,
            },
        ).to_parquet(mark_dir / f"{sym}.parquet")
    return end


@pytest.fixture(scope="module")
def synthetic_market(tmp_path_factory) -> tuple[Path, pd.Timestamp]:
    import src.market_data.services.futures_collection as fc

    root = tmp_path_factory.mktemp("mhs_market")
    end = _write_mhs_market(root, DEV_SYMBOLS)
    originals = {
        "funding_path": ev.funding_path,
        "mark_price_path": fc._mark_price_path,
        "_BOOTSTRAP_REPLICATES": ev._BOOTSTRAP_REPLICATES,
        "_BOOTSTRAP_MEAN_BLOCK": ev._BOOTSTRAP_MEAN_BLOCK,
        "_BOOTSTRAP_SEED": ev._BOOTSTRAP_SEED,
    }
    ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    ev._BOOTSTRAP_REPLICATES = 20
    ev._BOOTSTRAP_MEAN_BLOCK = 24
    ev._BOOTSTRAP_SEED = 20260807
    yield root, end
    for name, value in originals.items():
        if name == "mark_price_path":
            fc._mark_price_path = value
        else:
            setattr(ev, name, value)


@pytest.fixture(scope="module")
def report(synthetic_market):
    root, end = synthetic_market
    return run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(start=str(START), end=str(end), data_root=str(root), execution_timeframe="1m", log_run=False),
    )


@pytest.fixture(scope="module")
def touch_report(synthetic_market):
    root, end = synthetic_market
    return run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(
            start=str(START), end=str(end), data_root=str(root),
            execution_timeframe="1m", log_run=False, touch_diagnostic=True,
        ),
    )


@pytest.fixture(scope="module")
def annualization_report(synthetic_market):
    """SCENARIO_MHS_ANNUALIZATION_04: a full diagnostic on the default 5m
    execution grid, used to prove the corrected hourly-grid annualization of
    every real-execution-ledger headline metric against the pre-fix formulas
    applied to the same raw ledger.

    Module-scoped sibling fixtures (``fold_market``/``late_market_report``)
    re-point ``fc._mark_price_path``/``ev.funding_path`` at their own roots and
    only restore them at module teardown, so this fixture re-asserts the
    synthetic-market paths itself before running the diagnostic.
    """
    import src.market_data.services.futures_collection as fc

    root, end = synthetic_market
    originals = {
        "funding_path": ev.funding_path,
        "mark_price_path": fc._mark_price_path,
    }
    ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
    fc._mark_price_path = (
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
    )
    try:
        return run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                execution_timeframe="5m", log_run=False,
            ),
        )
    finally:
        ev.funding_path = originals["funding_path"]
        fc._mark_price_path = originals["mark_price_path"]


@pytest.fixture(scope="module")
def calibrated_report(synthetic_market):
    """Full diagnostic run with spies on the R1/R3 wiring seams: records the
    ``ema_span`` passed to every ``_book_weights`` call and which callers invoke
    the regime cash scale and turnover deadband.

    The top-level book replays and anchored folds now run in forked workers
    (Phase 3, P10/P14), so the capture stores use ``multiprocessing.Manager``
    proxies: a fork child's writes are reflected in the parent's proxy instead
    of mutating a copy-on-write shadow.
    """
    from multiprocessing import Manager

    root, end = synthetic_market
    mgr = Manager()
    captured = mgr.dict({
        "ema_spans": mgr.dict(),
        "regime_callers": mgr.list(),
        "deadband_callers": mgr.list(),
    })
    real_book_weights = ev._book_weights
    real_regime = ev._regime_cash_scale
    real_deadband = ev._apply_rebalance_deadband

    def _book_weights(log_close, eligible, spec, step_grid, ema_span=None):
        spans = captured["ema_spans"].get(spec.band.name)
        if spans is None:
            spans = mgr.list()
            captured["ema_spans"][spec.band.name] = spans
        spans.append(ema_span)
        return real_book_weights(log_close, eligible, spec, step_grid, ema_span=ema_span)

    def _regime(*args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        captured["regime_callers"].append(caller)
        return real_regime(*args, **kwargs)

    def _deadband(*args, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        captured["deadband_callers"].append(caller)
        return real_deadband(*args, **kwargs)

    ev._book_weights = _book_weights
    ev._regime_cash_scale = _regime
    ev._apply_rebalance_deadband = _deadband
    try:
        report = run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                execution_timeframe="1m", log_run=False,
            ),
        )
    finally:
        ev._book_weights = real_book_weights
        ev._regime_cash_scale = real_regime
        ev._apply_rebalance_deadband = real_deadband
    yield report, captured
    mgr.shutdown()


class TestQualityCalibrationWiring:
    """Quality calibration wiring contract: top-level books apply matching signal calibration."""

    @staticmethod
    def _book_signal_ema_span(spec) -> int:
        return max(1, round(spec.horizon_hours / spec.step_hours * ev.MHS_SIGNAL_EMA_HORIZON_SPAN))

    def test_top_level_book_weights_use_ema_span_matching_fold_path(self, calibrated_report) -> None:
        _report, captured = calibrated_report
        fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
        slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
        assert captured["ema_spans"]["fast_reversal"]
        assert captured["ema_spans"]["slow_momentum"]
        # Every _book_weights call passes the same sign-aware EMA span.
        for span in captured["ema_spans"]["fast_reversal"]:
            assert span is None
        for span in captured["ema_spans"]["slow_momentum"]:
            assert span == self._book_signal_ema_span(slow)
        assert self._book_signal_ema_span(slow) >= 1

    def test_top_level_blend_applies_regime_cash_scale_and_deadband(self, calibrated_report) -> None:
        report, captured = calibrated_report
        assert report.books, "top-level books must run on the synthetic market"
        assert captured["regime_callers"].count("run_mhs_horizon_diagnostic") == 1
        assert captured["deadband_callers"].count("_book_outcome") == 3

    def test_top_level_books_diagnostic_fields_populated_after_quality_calibration(self, report) -> None:
        for name in ("fast_reversal", "slow_momentum"):
            book = report.books[name]
            assert book.primary is not None
            assert len(book.prescreen) > 0
            assert book.tail.event_window_bars > 0
        assert report.blend is not None
        assert report.blend.primary is not None
        assert len(report.blend.prescreen) > 0
        assert report.blend.tail.event_window_bars > 0

    def test_run_mhs_horizon_diagnostic_xs_ic_regression_unchanged_after_reorder(self, report, synthetic_market) -> None:
        root, end = synthetic_market
        panel = ev.load_base_panel(
            root, "1h", ("close", "open", "quote_vol"), START, end,
            partition="dev", min_bars=2000,
        )
        log_close = np.log(panel["close"])
        opens = panel["open"]
        signal_48h = ev.horizon_log_return(log_close, 48)
        assert report.xs_rank_ic == ev._xs_rank_ic(signal_48h, opens, forward_bars=48)
        assert report.date_clustered_regression == ev._date_clustered_ols(opens, signal_48h, forward_bars=48)

    def test_run_mhs_horizon_diagnostic_log_close_released_before_book_replay(self) -> None:
        src = inspect.getsource(ev.run_mhs_horizon_diagnostic)
        assert "del log_close" in src
        assert "signal_48h" in src
        assert src.index("del log_close") < src.index("_run_books_concurrent(")


class TestMhsHorizonDiagnostic:
    """MHS-10-DIAGNOSTIC-HOLDOUT-SEALED: dev-only diagnostic on a synthetic panel."""

    def test_produces_frozen_books_and_separate_evidence_paths(self, report) -> None:
        assert report.status == "COMPLETE"
        assert set(report.books) == {"fast_reversal", "slow_momentum"}
        assert report.blend is not None
        assert report.eligible_symbols >= 8
        assert report.trials_attempted > 0
        assert report.execution_tiers_bps == pytest.approx((2.64, 4.18, 6.07))
        assert report.blend_target_gross > 0.0

    def test_mhs_3m_01_default_execution_timeframe(self) -> None:
        """MHS-3M-01-DEFAULT: production requests default to 3m."""
        from src.application.research.mhs.evaluation import MhsDiagnosticRequest

        assert MhsDiagnosticRequest().execution_timeframe == "3m"

    def test_diagnostic_ensemble_separate_from_executable_tranche(self, report) -> None:
        fast = report.books["fast_reversal"]
        assert fast.phase.n_phases > 0
        assert fast.primary.ledger is not None
        assert len(fast.primary.ledger.equity) > 0
        for tier in (2.64, 4.18, 6.07):
            assert tier in fast.prescreen

    def test_holdout_partition_raises(self, synthetic_market) -> None:
        root, end = synthetic_market
        with pytest.raises(RuntimeError):
            run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    partition="holdout", execution_timeframe="1m", log_run=False,
                ),
            )

    def test_end_past_cutoff_raises(self, synthetic_market) -> None:
        root, end = synthetic_market
        with pytest.raises(RuntimeError):
            run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START),
                    end=str(HOLDOUT_CUTOFF + pd.Timedelta(days=1)),
                    data_root=str(root),
                    execution_timeframe="1m", log_run=False,
                ),
            )


class TestResourceTelemetry:
    """MHS-31-RESOURCE-TELEMETRY-ORDER: the report carries ordered non-negative
    stage elapsed/RSS records without changing its GO decision."""

    STAGE_ORDER = (
        "base_1h_panel",
        "funding_alignment",
        "minute_market_mark_funding",
        "blend_participation",
        "statistical_diagnostics",
        "final_return",
    )

    def test_ordered_non_negative_stage_records(self, report) -> None:
        measurements = report.resource_measurements
        assert measurements
        stages = [m.stage for m in measurements]
        assert stages[0] == "base_1h_panel"
        assert stages[-1] == "final_return"
        assert "minute_market_mark_funding" in stages
        assert any(s.startswith("replay_") for s in stages)
        assert "blend_participation" in stages
        assert "statistical_diagnostics" in stages
        fold_stages = [s for s in stages if s.startswith("anchored_fold_") and "_window_" not in s]
        assert len(fold_stages) == 3
        # Anchored folds are recorded in their declared order.
        assert fold_stages == [f"anchored_fold_{i}" for i in range(3)]
        # Key stages follow the execution order.
        positions = {s: stages.index(s) for s in self.STAGE_ORDER if s in stages}
        assert list(positions.values()) == sorted(positions.values())
        for m in measurements:
            assert m.stage
            assert m.elapsed_ms >= 0
            assert m.rss_bytes > 0

    def test_stage_records_serialize_and_go_gate_unchanged(self, report) -> None:
        payload = report.to_payload()
        records = payload["resource_measurements"]
        assert isinstance(records, list)
        assert records
        for record in records:
            assert set(record) == {
                "stage", "elapsed_ms", "rss_bytes", "grid_bars", "n_symbols",
                "fill_count", "window_start", "window_end", "active_symbols",
                "peak_rss_bytes",
            }
        assert report.research_go.eligible is False


class TestWindowExecutionTelemetry:
    """MHS-31-RESOURCE-TELEMETRY-ORDER: per-window telemetry is ordered and
    carries non-negative elapsed time, positive RSS, and window provenance
    without changing the GO decision."""

    def test_window_stages_ordered_and_carry_provenance(self, report) -> None:
        stages = [m.stage for m in report.resource_measurements]
        window_stages = [s for s in stages if s.startswith("execution_window_")]
        assert window_stages, "window execution stages must be recorded"
        for m in report.resource_measurements:
            if not m.stage.startswith("execution_window_"):
                continue
            assert m.elapsed_ms >= 0
            assert m.rss_bytes > 0
            assert m.peak_rss_bytes is not None
            assert m.peak_rss_bytes >= m.rss_bytes
            assert m.window_start is not None
            assert m.window_end is not None
            assert m.active_symbols is not None
            assert m.active_symbols >= 0
            assert m.grid_bars is not None
            assert m.grid_bars > 0
        # Window stages run between the minute-market stage and blend
        # participation, preserving the existing stage order.
        first_window = stages.index(window_stages[0])
        assert "minute_market_mark_funding" in stages
        assert "blend_participation" in stages
        assert stages.index("minute_market_mark_funding") < first_window
        assert first_window < stages.index("blend_participation")

    def test_window_telemetry_does_not_change_go_gate(self, report) -> None:
        assert report.research_go.eligible is False
        assert "UNSPECIFIED_POLICY" in report.research_go.reason_codes


class TestStrictSimulatedPrimary:
    """MHS-19-STRICT-SIMULATED-PRIMARY: the realistic immediate-taker bound is
    the primary evidence, with a cost-stressed x3 stress bound."""

    def test_prescreen_and_primary_are_separate(self, report) -> None:
        blend = report.blend
        assert blend is not None
        assert 4.18 in blend.prescreen
        assert 6.07 in blend.prescreen
        assert blend.primary.fill_source == "OHLCV_IMMEDIATE_TAKER"
        assert blend.primary.ledger.mark_source == "MARK_PRICE"
        assert blend.primary_naive_sharpe is not None
        assert blend.primary_max_drawdown <= 0.0 or np.isnan(blend.primary_max_drawdown)
        assert blend.stress.fill_source == "OHLCV_IMMEDIATE_TAKER"
        assert report.fill_source == "OHLCV_IMMEDIATE_TAKER"
        assert blend.primary.simulated_fills is not None
        # The executable tranche actually traded through the immediate taker.
        assert len(blend.primary.simulated_fills) > 0

    def test_phase_and_tail_diagnostics_reported(self, report) -> None:
        fast = report.books["fast_reversal"]
        assert fast.tail.event_window_bars > 0
        assert set(fast.tail.winsor_curve) == {10, 20, 30, 50}


class TestTouchDiagnostic:
    """MHS execution fill-model realism Phase 1: ``touch_diagnostic=True``
    adds an opt-in ``OHLCV_TOUCH_PROXY`` replay leg alongside the strict/stress
    pair; the default path stays touch-free."""

    def test_touch_default_off(self, report) -> None:
        """SCENARIO_MHS_TOUCH_DEFAULT_OFF: every book report on the default
        path carries ``touch=None``/``touch_naive_sharpe=None``."""
        for book in (report.books["fast_reversal"], report.books["slow_momentum"], report.blend):
            assert book.touch is None
            assert book.touch_naive_sharpe is None

    def test_touch_weak_dominance_over_strict(self, touch_report) -> None:
        """SCENARIO_MHS_TOUCH_WEAK_DOMINANCE: the touch crossing condition
        ``adverse <= decision_price`` is a superset of strict's
        ``adverse < decision_price``, so on any input touch fill count weakly
        dominates the (demoted) strict patient-reference bound. The primary is
        now the immediate-taker bound, so strict lives on as
        ``patient_reference``."""
        for book in (touch_report.books["slow_momentum"], touch_report.blend):
            assert book.touch is not None
            assert book.touch.fill_source == "OHLCV_TOUCH_PROXY"
            assert book.touch_naive_sharpe is not None
            assert book.patient_reference is not None
            assert book.patient_reference.fill_source == "OHLCV_STRICT_PROXY"
            assert book.touch.fill_count >= book.patient_reference.fill_count
            assert book.touch.unfilled_count <= book.patient_reference.unfilled_count


class TestFreezeBeforeFinalOos:
    """MHS-24-FREEZE-BEFORE-FINAL-OOS: Phase 1 has no unseal path."""

    def test_cli_registers_no_unseal_flag(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="portfolio_command", required=True)
        add_mhs_commands(sub)
        args = parser.parse_args(["mhs-horizon-diagnostic"])
        assert callable(args.handler)
        assert not hasattr(args, "unseal_holdout")

    def test_forward_observation_is_frozen(self) -> None:
        from src.mhs.execution import ForwardExecutionObservation

        obs = ForwardExecutionObservation(
            symbol="BTCUSDT", signal_time=pd.Timestamp("2026-01-01", tz="UTC"),
            intent_time=pd.Timestamp("2026-01-01", tz="UTC"),
            submit_time=None, fill_time=None, side=1,
            requested_quantity=0.1, filled_quantity=0.0,
            limit_price=None, fill_price=None, best_bid=None, best_ask=None,
            top_n_depth_notional=None, trade_print_notional=None,
            reject_reason=None, cancel_replace_count=0, latency_ms=None,
        )
        with pytest.raises(Exception, match="cannot assign"):
            obs.filled_quantity = 0.1

    def test_mhs_5m_03_signal_preservation(self, report) -> None:
        """MHS-5M-03-SIGNAL-PRESERVATION: signal and replay universes are reported separately."""
        assert isinstance(report.execution_symbols, tuple)
        assert report.execution_symbols


class TestMarkPriceCacheRequired:
    """MHS-MARK-03-CACHE-REQUIRED-INTEGRATION: a complete causal mark cache
    labels every replay MARK_PRICE and the report agrees across fast/slow/blend."""

    def test_fast_slow_blend_and_report_are_mark_price(self, report) -> None:
        assert report.mark_source == "MARK_PRICE"
        for book in (report.books["fast_reversal"], report.books["slow_momentum"], report.blend):
            assert book.primary.ledger.mark_source == "MARK_PRICE"
            assert book.primary.mark_source == "MARK_PRICE"
            assert book.stress.ledger.mark_source == "MARK_PRICE"
            assert book.stress.mark_source == "MARK_PRICE"

    def test_strict_and_stress_share_the_same_mark_source(self, report) -> None:
        assert report.mark_source == report.blend.primary.mark_source
        assert report.blend.primary.mark_source == report.blend.stress.mark_source


class TestNoSilentMarkFallback:
    """MHS-MARK-04-NO-SILENT-FALLBACK: a cache gap is never silently replaced by
    OHLCV closes under cache_required; explicit fallback stays labelled."""

    def _gapped_market(self, root: Path, tmp_path: Path, monkeypatch) -> None:
        import src.market_data.services.futures_collection as fc

        gap_dir = tmp_path / "markPriceKlines" / "1h"
        gap_dir.mkdir(parents=True, exist_ok=True)
        gap_start = pd.Timestamp("2021-02-01", tz="UTC")
        gap_end = pd.Timestamp("2021-02-10", tz="UTC")
        for sym in DEV_SYMBOLS:
            frame = pd.read_parquet(root / "markPriceKlines" / "1h" / f"{sym}.parquet")
            drop = (frame["datetime"] >= gap_start) & (frame["datetime"] < gap_end)
            frame.loc[drop, "close"] = float("nan")
            frame.loc[drop, "high"] = float("nan")
            frame.loc[drop, "low"] = float("nan")
            frame.loc[drop, "open"] = float("nan")
            frame.to_parquet(gap_dir / f"{sym}.parquet")
        monkeypatch.setattr(
            fc,
            "_mark_price_path",
            lambda symbol, timeframe: gap_dir / f"{symbol}.parquet",
        )

    def test_cache_required_fails_closed_with_typed_rejection(
        self, synthetic_market, tmp_path, monkeypatch,
    ) -> None:
        """MHS-MARK-04-NO-SILENT-FALLBACK: a cache gap is never silently
        replaced by OHLCV closes under cache_required. The expected strict
        replay failure becomes a typed book-level rejection in the terminal
        report instead of escaping the diagnostic process."""
        root, end = synthetic_market
        self._gapped_market(root, tmp_path, monkeypatch)
        report = run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            ),
        )
        assert report.status == "COMPLETE"
        failed = [b for b in report.books.values() if b.failure is not None]
        assert failed, "cache_required mark gap must reject every book"
        for book in report.books.values():
            assert book.primary is None
            assert book.primary_autocorr_sharpe is None
        assert report.research_go.eligible is False

    def test_explicit_ohlcv_fallback_completes_as_fallback(
        self, synthetic_market, tmp_path, monkeypatch,
    ) -> None:
        root, end = synthetic_market
        self._gapped_market(root, tmp_path, monkeypatch)
        report = run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                mark_mode="ohlcv_close_fallback", execution_timeframe="1m", log_run=False,
            ),
        )
        assert report.mark_source == "OHLCV_CLOSE_FALLBACK"
        assert report.blend is not None
        assert report.blend.primary.ledger.mark_source == "OHLCV_CLOSE_FALLBACK"


class TestMarkPriceGoValidityIntegration:
    """MHS-MARK-05-GO-VALIDITY: an invalid primary never yields a Research GO."""

    def test_research_go_requires_valid_primary(self) -> None:
        from src.mhs.evaluation import compute_deployment_readiness

        idx = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
        equity = pd.Series(np.cumprod(1.0 + np.full(5, 0.001)), index=idx)
        invalid = compute_deployment_readiness(equity, 8760.0, primary_valid=False, n_bootstrap=2, mean_block_bars=1)
        assert invalid.research_go_eligible is False
        assert invalid.execution_go_eligible is False
        assert invalid.pilot_go_eligible is False
        assert invalid.scale_go_eligible is False


class TestMarkModeCli:
    """MHS-MARK-06-CLI-MODE: CLI defaults to cache_required, accepts only the
    two named modes, and exposes neither --collect-mark nor --unseal-holdout."""

    def test_cli_defaults_to_cache_required(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="portfolio_command", required=True)
        add_mhs_commands(sub)
        args = parser.parse_args(["mhs-horizon-diagnostic"])
        assert args.mark_mode == "cache_required"
        assert not hasattr(args, "collect_mark")
        assert not hasattr(args, "unseal_holdout")

    def test_cli_accepts_only_two_named_modes(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="portfolio_command", required=True)
        add_mhs_commands(sub)
        fallback = parser.parse_args(["mhs-horizon-diagnostic", "--mark-mode", "ohlcv_close_fallback"])
        assert fallback.mark_mode == "ohlcv_close_fallback"
        with pytest.raises(SystemExit):
            parser.parse_args(["mhs-horizon-diagnostic", "--mark-mode", "bogus"])


LATE_SYMBOL = "MHSAUSDT"
LATE_START = pd.Timestamp("2021-02-01", tz="UTC")


class TestPitExecutionGrid:
    """MHS-25-FULL-PERIOD-PIT-GRID: a late-listed symbol never clips the replay
    start; pre-listing NaNs are retained and no order is emitted before the
    symbol is PIT eligible."""

    @pytest.fixture(scope="module")
    def late_market_report(self, tmp_path_factory):
        import src.market_data.services.futures_collection as fc

        root = tmp_path_factory.mktemp("mhs_late_market")
        end = _write_mhs_market(
            root, DEV_SYMBOLS,
            late_listings={LATE_SYMBOL: LATE_START},
            n_hours=3000,
        )
        originals = {
            "funding_path": ev.funding_path,
            "mark_price_path": fc._mark_price_path,
            "_BOOTSTRAP_REPLICATES": ev._BOOTSTRAP_REPLICATES,
            "_BOOTSTRAP_MEAN_BLOCK": ev._BOOTSTRAP_MEAN_BLOCK,
            "_BOOTSTRAP_SEED": ev._BOOTSTRAP_SEED,
            "_placebo_sharpe_percentile": ev._placebo_sharpe_percentile,
            "_bootstrap_ci": ev._bootstrap_ci,
        }
        ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        ev._BOOTSTRAP_REPLICATES = 20
        ev._BOOTSTRAP_MEAN_BLOCK = 24
        ev._BOOTSTRAP_SEED = 20260807
        # The 500-placebo / bootstrap stats are not the subject of this test.
        ev._placebo_sharpe_percentile = lambda *a, **k: None
        ev._bootstrap_ci = lambda *a, **k: None
        try:
            yield run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    execution_timeframe="1m", log_run=False,
                ),
            )
        finally:
            for name, value in originals.items():
                if name == "mark_price_path":
                    fc._mark_price_path = value
                else:
                    setattr(ev, name, value)

    def test_replay_grid_starts_at_requested_start(self, late_market_report) -> None:
        report = late_market_report
        assert report.blend is not None
        # The requested evaluation grid is the replay grid: it must begin at the
        # requested start, not at the late symbol's first-observed timestamp.
        assert report.blend.primary.ledger.equity.index[0] == START
        assert report.blend.stress.ledger.equity.index[0] == START

    def test_no_pre_listing_order_for_late_symbol(self, late_market_report) -> None:
        report = late_market_report
        assert LATE_SYMBOL in report.execution_symbols
        fills = report.blend.primary.simulated_fills
        pre_listing = fills[
            (fills["symbol"] == LATE_SYMBOL) & (fills["timestamp"] < LATE_START)
        ]
        assert pre_listing.empty

    def test_late_symbol_kept_as_nan_then_traded(self, late_market_report) -> None:
        report = late_market_report
        units = report.blend.primary.simulated_units
        if LATE_SYMBOL in units.columns and len(units):
            before = units[units.index < LATE_START][LATE_SYMBOL]
            assert before.isna().all() or (before == 0.0).all()


class TestAnchoredFoldGoGate:
    """MHS-26-ANCHOR-FOLD-GO-GATE: all three replayed folds are reported; an
    incomplete fold, negative strict Sharpe, non-positive stress Sharpe, or
    relevant termination produces Research GO false and reason codes."""

    def test_three_folds_reported_and_go_false(self, report) -> None:
        assert len(report.folds) == 3
        assert len(report.anchored_folds) == 3
        for fold_report in report.folds:
            assert fold_report.validation_start.startswith("2023-01-08") or fold_report.validation_start.startswith("2024-01-08") or fold_report.validation_start.startswith("2025-01-08")
            assert fold_report.validation_end.startswith("2023-12-31") or fold_report.validation_end.startswith("2024-12-31") or fold_report.validation_end.startswith("2025-12-31")
        assert report.research_go.eligible is False
        assert "INCOMPLETE_ANCHORED_FOLD" in report.research_go.reason_codes
        assert report.research_go.evaluated_folds == 3
        # The gate boolean routes to deployment readiness, never primary_valid alone.
        assert report.deployment_readiness.research_go_eligible is False
        # The cap-30 / annual-return policies are not source-registered.
        assert "UNSPECIFIED_POLICY" in report.research_go.reason_codes

    def test_fold_metrics_exposed(self, report) -> None:
        fold_report = report.folds[0]
        assert isinstance(fold_report.failures, tuple)
        assert isinstance(fold_report.termination_counts, dict)
        assert fold_report.strict_elapsed_seconds >= 0.0
        assert fold_report.stress_elapsed_seconds >= 0.0

    def test_gate_collects_each_fail_closed_reason(self) -> None:
        from src.application.research.mhs.evaluation import (
            MhsFoldReport,
            _mhs_research_go,
        )
        from src.mhs.execution import ExecutionSpec, strategy_aware_execution_replay

        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        replay = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )

        def fold(index: int, failures: tuple[str, ...]) -> MhsFoldReport:
            return MhsFoldReport(
                fold_index=index,
                validation_start="2023-01-08 00:00:00+00:00",
                validation_end="2023-12-31 00:00:00+00:00",
                strict=replay,
                stress=replay,
                primary_valid=True,
                primary_autocorr_sharpe=1.0,
                primary_naive_sharpe=1.0,
                primary_net_ann=0.1,
                primary_geometric_cagr=0.1,
                primary_max_drawdown=-0.02,
                stress_naive_sharpe=0.1,
                decision_intents=1,
                termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
                failures=failures,
                strict_elapsed_seconds=0.01,
                stress_elapsed_seconds=0.01,
            )

        from src.application.research.mhs.evaluation import (
            MHS_GO_REASON_EXECUTION_GAP,
            MHS_GO_REASON_INCOMPLETE_FOLD,
            MHS_GO_REASON_PRIMARY_SHARPE,
            MHS_GO_REASON_STRESS_SHARPE,
        )

        gap = _mhs_research_go((fold(0, (MHS_GO_REASON_EXECUTION_GAP,)),))
        assert gap.eligible is False
        assert MHS_GO_REASON_EXECUTION_GAP in gap.reason_codes

        sharpe = _mhs_research_go((fold(0, (MHS_GO_REASON_PRIMARY_SHARPE,)),))
        assert MHS_GO_REASON_PRIMARY_SHARPE in sharpe.reason_codes

        stress = _mhs_research_go((fold(0, (MHS_GO_REASON_STRESS_SHARPE,)),))
        assert MHS_GO_REASON_STRESS_SHARPE in stress.reason_codes

        incomplete = _mhs_research_go(
            (
                MhsFoldReport(
                    fold_index=0,
                    validation_start="2023-01-08 00:00:00+00:00",
                    validation_end="2023-12-31 00:00:00+00:00",
                    strict=None,
                    stress=None,
                    primary_valid=False,
                    primary_autocorr_sharpe=float("nan"),
                    primary_naive_sharpe=float("nan"),
                    primary_net_ann=float("nan"),
                    primary_geometric_cagr=float("nan"),
                    primary_max_drawdown=float("nan"),
                    stress_naive_sharpe=float("nan"),
                    decision_intents=0,
                    termination_counts={},
                    failures=(MHS_GO_REASON_INCOMPLETE_FOLD,),
                    strict_elapsed_seconds=0.0,
                    stress_elapsed_seconds=0.0,
                ),
            ),
        )
        assert MHS_GO_REASON_INCOMPLETE_FOLD in incomplete.reason_codes
        assert incomplete.eligible is False

class TestFoldSafeHorizonEfficiency:
    """SCENARIO_MHS_HORIZON_SEARCH_EFF_06_FULL_DIAGNOSTIC_REPORT_UNCHANGED: the
    discovery weight cache (Q3) is a pure performance change -- running the
    diagnostic with ``fold_safe_horizon_selection=True`` through the cached
    path must produce a report byte-identical to the pre-caching-change
    baseline (the same run with ``precomputed_candidate_weights`` stripped from
    the fold-scoped qualification calls, which is exactly what the pre-change
    code did)."""

    @pytest.fixture(scope="module")
    def fold_safe_report(self, synthetic_market) -> MhsHorizonDiagnosticReport:
        root, end = synthetic_market
        return run_mhs_horizon_diagnostic(
            MhsDiagnosticRequest(
                start=str(START), end=str(end), data_root=str(root),
                execution_timeframe="1m", log_run=False,
                fold_safe_horizon_selection=True,
            ),
        )

    @pytest.fixture(scope="module")
    def fold_safe_baseline_report(self, synthetic_market) -> MhsHorizonDiagnosticReport:
        root, end = synthetic_market
        real_fn = ev.fold_train_only_discovery_qualification

        def _no_cache(*args, **kwargs):
            kwargs.pop("precomputed_candidate_weights", None)
            return real_fn(*args, **kwargs)

        ev.fold_train_only_discovery_qualification = _no_cache
        try:
            return run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    execution_timeframe="1m", log_run=False,
                    fold_safe_horizon_selection=True,
                ),
            )
        finally:
            ev.fold_train_only_discovery_qualification = real_fn

    def test_cached_path_report_byte_identical_to_baseline(
        self, fold_safe_report, fold_safe_baseline_report,
    ) -> None:
        assert fold_safe_report.status == "COMPLETE"
        assert fold_safe_baseline_report.status == "COMPLETE"
        assert len(fold_safe_report.folds) == len(fold_safe_baseline_report.folds) == 3
        for cached_fold, baseline_fold in zip(
            fold_safe_report.folds, fold_safe_baseline_report.folds, strict=True,
        ):
            assert cached_fold.slow_horizon_hours == baseline_fold.slow_horizon_hours
            assert cached_fold.slow_horizon_source == baseline_fold.slow_horizon_source
            assert cached_fold.fast_horizon_hours == baseline_fold.fast_horizon_hours
            assert cached_fold.fast_horizon_source == baseline_fold.fast_horizon_source
        assert fold_safe_report.research_go == fold_safe_baseline_report.research_go
        assert fold_safe_report.blend.primary_autocorr_sharpe == fold_safe_baseline_report.blend.primary_autocorr_sharpe

class TestMhsPerfOptimizationO3FoldParity:
    """SCENARIO_O3_FOLD_PARITY: the three anchored folds run in parallel worker
    processes produce bit-identical per-fold evidence to a sequential baseline
    (same primary_autocorr_sharpe, stress_naive_sharpe, decision_intents,
    termination_counts, primary_valid, and equity series)."""

    @pytest.fixture(scope="module")
    def fold_parity_request(self, tmp_path_factory) -> tuple[Path, pd.Timestamp]:
        import src.market_data.services.futures_collection as fc

        root = tmp_path_factory.mktemp("mhs_fold_parity")
        end = _write_mhs_market(root, DEV_SYMBOLS)
        originals = {"funding_path": ev.funding_path, "mark_price_path": fc._mark_price_path}
        ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        yield root, end
        ev.funding_path = originals["funding_path"]
        fc._mark_price_path = originals["mark_price_path"]

    def test_parallel_folds_match_sequential_folds(self, fold_parity_request) -> None:
        from src.application.research.mhs.evaluation import (
            _run_anchored_fold,
            _run_folds_parallel,
        )

        root, end = fold_parity_request
        funding, _ = ev._load_funding_series(DEV_SYMBOLS)
        request = MhsDiagnosticRequest(
            start=str(START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        )
        sequential = tuple(
            _run_anchored_fold(str(root), fold, request, funding, 1.0, idx, None)
            for idx, fold in enumerate(ev.phase_1_anchored_purged_folds())
        )
        parallel = _run_folds_parallel(str(root), request, funding, 1.0, None)
        assert len(sequential) == len(parallel) == 3

        def _same_float(a: float, b: float) -> bool:
            return (a == b) or (np.isnan(a) and np.isnan(b))

        for seq, par in zip(sequential, parallel, strict=True):
            assert seq.fold_index == par.fold_index
            assert _same_float(seq.primary_autocorr_sharpe, par.primary_autocorr_sharpe)
            assert _same_float(seq.primary_naive_sharpe, par.primary_naive_sharpe)
            assert _same_float(seq.stress_naive_sharpe, par.stress_naive_sharpe)
            assert seq.decision_intents == par.decision_intents
            assert seq.primary_valid == par.primary_valid
            assert dict(seq.termination_counts) == dict(par.termination_counts)
            assert seq.failures == par.failures
            assert (seq.strict is None) == (par.strict is None)
            assert (seq.stress is None) == (par.stress is None)
            if seq.strict is not None and par.strict is not None:
                assert seq.strict.ledger.equity.equals(par.strict.ledger.equity)
                assert len(seq.strict.simulated_fills) == len(par.strict.simulated_fills)
                assert len(seq.stress.simulated_fills) == len(par.stress.simulated_fills)


class TestTypedArtifactRoundtrip:
    """MHS-29-TYPED-ARTIFACT-ROUNDTRIP: persisted ledger and times artifacts
    retain an explicit UTC timestamp and round-trip without strings or
    NaN-equity concealment."""

    def test_ledger_and_times_round_trip_as_utc(self, report, tmp_path) -> None:
        from src.application.research.mhs.evaluation import (
            MhsOutputTier,
            load_mhs_replay_artifact,
            persist_mhs_horizon_diagnostic_report,
        )

        out = tmp_path / "mhs_report.json"
        persist_mhs_horizon_diagnostic_report(report, out, tier=MhsOutputTier.FULL)
        artifact_dir = out.parent / "mhs_report_artifacts" / "_full"

        parquet_files = list(artifact_dir.glob("*.parquet"))
        assert len(parquet_files) == 5
        assert {p.name for p in parquet_files} == {
            "fills.parquet",
            "units.parquet",
            "notional_weights.parquet",
            "ledger.parquet",
            "times.parquet",
        }

        ledger = load_mhs_replay_artifact(artifact_dir, "blend_primary", "ledger")
        assert pd.api.types.is_datetime64_any_dtype(ledger["timestamp"])
        assert ledger["timestamp"].dt.tz is not None
        assert str(ledger["timestamp"].dt.tz) == "UTC"
        assert np.isfinite(ledger["equity"].to_numpy()).all()
        idx = pd.DatetimeIndex(pd.to_datetime(ledger["timestamp"], utc=True))
        assert idx.tz is not None

        times = load_mhs_replay_artifact(artifact_dir, "blend_primary", "times")
        assert pd.api.types.is_datetime64_any_dtype(times["submit_time"])
        assert pd.api.types.is_datetime64_any_dtype(times["fill_time"])
        assert times["submit_time"].dt.tz is not None

    def test_json_reference_carries_schema_and_checksum(self, report, tmp_path) -> None:
        import json

        from src.application.research.mhs.evaluation import (
            MhsOutputTier,
            load_mhs_replay_artifact,
            persist_mhs_horizon_diagnostic_report,
        )

        out = tmp_path / "mhs_report.json"
        persist_mhs_horizon_diagnostic_report(report, out, tier=MhsOutputTier.FULL)
        report_json = out.parent / "mhs_report_artifacts" / "_full" / "report.json"
        payload = json.loads(report_json.read_text())
        ledger_ref = payload["blend"]["primary"]["ledger"]
        assert ledger_ref["schema_version"] == 1
        assert ledger_ref["row_count"] > 0
        assert ledger_ref["time_bounds"]["start"] is not None
        assert ledger_ref["checksum_sha256"]
        fills_ref = payload["blend"]["primary"]["fills"]
        assert fills_ref["row_count"] == len(report.blend.primary.simulated_fills)
        # The summary JSON references the 5 unified files and the replay_id mapping.
        assert set(payload["artifacts"]) == {
            "fills", "units", "notional_weights", "ledger", "times",
        }
        assert "blend_primary" in payload["replay_ids"]
        roundtrip = load_mhs_replay_artifact(
            out.parent / "mhs_report_artifacts" / "_full", "blend_primary", "ledger"
        )
        assert len(roundtrip) == ledger_ref["row_count"]

    def test_empty_replay_artifacts_round_trip(self, tmp_path) -> None:
        from src.application.research.mhs.evaluation import (
            _build_replay_category_tables,
            _write_unified_artifact_tables,
            load_mhs_replay_artifact,
        )
        from src.mhs.execution import ExecutionSpec, strategy_aware_execution_replay

        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        px.loc["2021-01-01 12:10":, "A"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        replay = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert replay.simulated_fills.empty
        _write_unified_artifact_tables({"empty": _build_replay_category_tables(replay)}, tmp_path)
        units = load_mhs_replay_artifact(tmp_path, "empty", "units")
        assert "timestamp" in units.columns
        assert pd.api.types.is_datetime64_any_dtype(units["timestamp"])
        ledger = load_mhs_replay_artifact(tmp_path, "empty", "ledger")
        assert "timestamp" in ledger.columns
        assert len(units) == 0

    def test_completed_fold_artifacts_persisted(self, report, tmp_path) -> None:
        from dataclasses import replace

        from src.application.research.mhs.evaluation import (
            MHS_GO_REASON_UNSPECIFIED_POLICY,
            MhsFoldReport,
            MhsOutputTier,
            load_mhs_replay_artifact,
            persist_mhs_horizon_diagnostic_report,
        )
        from src.mhs.execution import ExecutionSpec, strategy_aware_execution_replay

        idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
        px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
        replay = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        fold_report = MhsFoldReport(
            fold_index=0,
            validation_start="2023-01-08 00:00:00+00:00",
            validation_end="2023-12-31 00:00:00+00:00",
            strict=replay,
            stress=replay,
            primary_valid=True,
            primary_autocorr_sharpe=1.0,
            primary_naive_sharpe=1.0,
            primary_net_ann=0.1,
            primary_geometric_cagr=0.1,
            primary_max_drawdown=-0.02,
            stress_naive_sharpe=0.1,
            decision_intents=1,
            termination_counts={},
            failures=(MHS_GO_REASON_UNSPECIFIED_POLICY,),
            strict_elapsed_seconds=0.01,
            stress_elapsed_seconds=0.01,
        )
        patched = replace(report, folds=(fold_report,))
        out = tmp_path / "fold_report.json"
        persist_mhs_horizon_diagnostic_report(patched, out, tier=MhsOutputTier.FULL)
        artifact_dir = out.parent / "fold_report_artifacts" / "_full"
        parquet_files = list(artifact_dir.glob("*.parquet"))
        assert len(parquet_files) == 5
        assert {p.name for p in parquet_files} == {
            "fills.parquet", "units.parquet", "notional_weights.parquet",
            "ledger.parquet", "times.parquet",
        }
        strict_ledger = load_mhs_replay_artifact(artifact_dir, "fold0_strict", "ledger")
        assert "timestamp" in strict_ledger.columns


class TestFullModeBackwardCompat:
    """MHS-OUTPUT-TIERING: FULL tier reproduces the pre-tiering 5-category
    unified Parquet tables and row counts with no per-fill data loss."""

    def test_full_mode_persists_unified_tables(self, report, tmp_path) -> None:
        # FULL_MODE_BACKWARD_COMPAT
        import pandas as pd

        from src.application.research.mhs.evaluation import (
            MhsOutputTier,
            persist_mhs_horizon_diagnostic_report,
        )

        out = tmp_path / "mhs_report.json"
        report_json = persist_mhs_horizon_diagnostic_report(
            report, out, tier=MhsOutputTier.FULL,
        )
        assert report_json is not None
        artifact_dir = out.parent / "mhs_report_artifacts" / "_full"
        assert (artifact_dir / "report.json").exists()

        for category in ("fills", "units", "notional_weights", "ledger", "times"):
            table = pd.read_parquet(artifact_dir / f"{category}.parquet")
            assert "replay_id" in table.columns
            assert table["replay_id"].nunique() == len(pd.unique(table["replay_id"]))

        fills = pd.read_parquet(artifact_dir / "fills.parquet")
        expected_fills = sum(
            len(replay.simulated_fills)
            for book_report in report.books.values()
            for replay in (
                book_report.primary, book_report.stress,
                book_report.patient_reference, book_report.pre_vol_target_reference,
            )
            if replay is not None
        )
        if report.blend is not None:
            for replay in (
                report.blend.primary, report.blend.stress,
                report.blend.patient_reference, report.blend.pre_vol_target_reference,
            ):
                if replay is not None:
                    expected_fills += len(replay.simulated_fills)
        for fold_report in report.folds:
            for replay in (fold_report.strict, fold_report.stress):
                if replay is not None:
                    expected_fills += len(replay.simulated_fills)
        assert len(fills) == expected_fills
        assert fills["replay_id"].notna().all()

        ledger = pd.read_parquet(artifact_dir / "ledger.parquet")
        assert "timestamp" in ledger.columns
        assert pd.api.types.is_datetime64_any_dtype(ledger["timestamp"])
        assert np.isfinite(ledger["equity"].to_numpy()).all()


FOLD_WINDOW_FOLD = ev.AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)


class TestFoldWindowTelemetryOracle:
    """MHS-31-FOLD-WINDOW-TELEMETRY: a synthetic fold records monotonically
    ordered per-window telemetry and is numerically equivalent to the
    single-panel oracle."""

    @pytest.fixture(scope="module")
    def fold_market(self, tmp_path_factory) -> tuple[Path, pd.Timestamp]:
        import src.market_data.services.futures_collection as fc

        root = tmp_path_factory.mktemp("mhs_fold_market")
        end = _write_mhs_market(root, DEV_SYMBOLS, n_hours=3000)
        originals = {
            "funding_path": ev.funding_path,
            "mark_price_path": fc._mark_price_path,
        }
        ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        yield root, end
        ev.funding_path = originals["funding_path"]
        fc._mark_price_path = originals["mark_price_path"]

    def _single_panel_oracle(
        self, root: Path, funding_by_symbol: dict[str, pd.Series],
    ):
        """Replicate the fold's pre-replay decision construction via the shared
        ``_build_fold_target_weights`` builder and run the dense single-panel
        oracle (``strategy_aware_execution_replay``) over the whole validation
        window."""
        from src.mhs.contracts import ExecutionSpec
        from src.mhs.execution import strategy_aware_execution_replay

        fold = FOLD_WINDOW_FOLD
        vs, ve = fold.validation_start, fold.validation_end
        request = MhsDiagnosticRequest(
            start=str(fold.train_start), end=str(fold.validation_end),
            data_root=str(root), mark_mode="cache_required",
            execution_timeframe="1m", log_run=False,
        )
        target_weights, signal_available_at, roster, _grid_1h = ev._build_fold_target_weights(
            str(root), fold, request, funding_by_symbol,
        )
        target_replay = target_weights[roster]
        minute_grid = pd.date_range(vs, ve, freq="1min", tz="UTC")
        target_replay, signal_available_at, _censored = ev._truncate_replayable_decisions(
            target_replay, signal_available_at, minute_grid, ExecutionSpec(),
        )
        minute_frames = ev._align_minute_frames(
            ev._load_window_minute_frames(str(root), list(target_replay.columns), vs, ve, "1m"),
            "1m", vs, ve,
        )
        assert minute_frames is not None
        highs, lows, closes = minute_frames
        marks = ev.DataCollector().load_mark_price_panel(
            list(closes.columns), "1h", minute_grid, max_stale_hours=0,
        )
        mper = minute_grid[1] - minute_grid[0]
        mfwin = {
            s: funding_by_symbol[s].loc[
                (funding_by_symbol[s].index >= minute_grid[0])
                & (funding_by_symbol[s].index < minute_grid[-1] + mper)
            ]
            for s in list(closes.columns)
        }
        minute_funding = ev.bar_funding_panel(mfwin, minute_grid).reindex(columns=list(closes.columns))
        oracle = strategy_aware_execution_replay(
            target_replay, signal_available_at, highs, lows, closes, marks,
            minute_funding, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
        )
        return oracle

    def test_fold_window_telemetry_monotonic_and_oracle_equivalent(self, fold_market) -> None:
        root, end = fold_market
        symbols = list(DEV_SYMBOLS)
        funding_by_symbol, _ = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        )
        recorder = ev._StageRecorder(log_run=False)
        fold_report = ev._run_anchored_fold(
            str(root), FOLD_WINDOW_FOLD, request, funding_by_symbol, 1.0, 0, recorder,
        )
        assert fold_report.strict is not None
        assert fold_report.strict.event_snapshots_retained is False
        assert fold_report.stress is not None
        assert fold_report.stress.event_snapshots_retained is False

        # The fold runs two replay generations: the reference pass under
        # ``_window_`` and the rescaled primary/stress pair sharing one
        # interleaved stream under ``_window_rescaled_``; all windows are
        # recorded in chronological order.
        reference_windows = [
            (m.window_start, m.window_end)
            for m in recorder.records
            if m.stage.startswith("anchored_fold_0_window_")
            and not m.stage.startswith("anchored_fold_0_window_rescaled_")
        ]
        rescaled_windows = [
            (m.window_start, m.window_end)
            for m in recorder.records
            if m.stage.startswith("anchored_fold_0_window_rescaled_")
        ]
        assert reference_windows, "fold reference window telemetry must be recorded"
        assert rescaled_windows, "fold rescaled window telemetry must be recorded"
        for seen in (reference_windows, rescaled_windows):
            # Each generation's windows are chronologically ordered: starts and
            # ends are non-decreasing and span the validation window.
            starts = [pd.Timestamp(s) for s, _ in seen]
            ends = [pd.Timestamp(e) for _, e in seen]
            assert starts == sorted(starts)
            assert ends == sorted(ends)
            assert starts[0] <= FOLD_WINDOW_FOLD.validation_start
            assert ends[-1] >= FOLD_WINDOW_FOLD.validation_end
        for m in recorder.records:
            if not m.stage.startswith("anchored_fold_0_window_"):
                continue
            assert m.window_start is not None
            assert m.window_end is not None
            assert m.grid_bars is not None
            assert m.grid_bars > 0
            assert m.active_symbols is not None
            assert m.active_symbols >= 0
            assert m.peak_rss_bytes is not None
            assert m.peak_rss_bytes >= m.rss_bytes

        # Numerical equivalence to the single-panel oracle on the same inputs.
        oracle = self._single_panel_oracle(root, funding_by_symbol)
        windowed = fold_report.strict
        fill_o = oracle.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        fill_w = windowed.simulated_fills.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        assert len(fill_o) == len(fill_w)
        for col in ("timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps", "reason"):
            assert fill_o[col].tolist() == fill_w[col].tolist()
        for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
            np.testing.assert_allclose(
                getattr(oracle.ledger, field).to_numpy(),
                getattr(windowed.ledger, field).to_numpy(),
                rtol=1e-12, atol=1e-12,
            )
        assert oracle.ledger.primary_valid == windowed.ledger.primary_valid
        assert oracle.ledger.invalid_reasons == windowed.ledger.invalid_reasons
        assert dict(oracle.termination_counts) == dict(windowed.termination_counts)
        assert oracle.fill_count == windowed.fill_count


class TestMhsExecutionAnnualization:
    """SCENARIO_MHS_ANNUALIZATION_04: the real-execution-ledger headline
    metrics must be annualized on the native execution-timeframe grid via the
    hourly resample (C1/C2), not with the hourly-grid constant applied to the
    raw 5m ledger. The pre-fix formulas are kept here as an explicit oracle so
    the regression guard survives independently of the fixed code."""

    @staticmethod
    def _pre_fix_metrics(ledger):
        net = ledger.net_returns
        equity = ledger.equity
        pre_naive = float(net.mean() / net.std(ddof=1) * np.sqrt(ev._PERIODS_PER_YEAR_1H))
        n = len(equity)
        pre_cagr = float(
            (equity.iloc[-1] / equity.iloc[0]) ** (ev._PERIODS_PER_YEAR_1H / n) - 1.0,
        )
        pre_net_ann = float(net.mean() * ev._PERIODS_PER_YEAR_1H)
        pre_turnover = float(ledger.fill_turnover.mean() * ev._PERIODS_PER_YEAR_1H)
        return pre_naive, pre_cagr, pre_net_ann, pre_turnover

    def test_headline_metrics_larger_in_magnitude_holding_sign(self, annualization_report) -> None:
        blend = annualization_report.blend
        assert blend is not None
        assert blend.primary is not None, f"blend primary missing: failure={blend.failure!r}"
        ledger = blend.primary.ledger
        pre_naive, pre_cagr, pre_net_ann, pre_turnover = self._pre_fix_metrics(ledger)
        post = (
            blend.primary_naive_sharpe,
            blend.primary_geometric_cagr,
            blend.primary_net_ann,
            blend.primary_annualized_turnover,
        )
        for pre, value, label in zip(
            (pre_naive, pre_cagr, pre_net_ann, pre_turnover),
            post,
            ("naive_sharpe", "geometric_cagr", "net_ann", "annualized_turnover"),
            strict=False,
        ):
            assert value is not None, label
            assert abs(value) > abs(pre), f"{label}: post={value} pre={pre}"
            assert np.sign(value) == np.sign(pre), f"{label}: post={value} pre={pre}"

    def test_autocorr_sharpe_and_mdd_still_read_raw_ledger(self, annualization_report) -> None:
        # The frequency-independent metrics must stay wired to the raw ledger:
        # any future resample there would shift MDD off the tick-level trough
        # and silently change the daily autocorr Sharpe.
        blend = annualization_report.blend
        assert blend is not None
        assert blend.primary is not None, f"blend primary missing: failure={blend.failure!r}"
        ledger = blend.primary.ledger
        assert blend.primary_autocorr_sharpe == ev._daily_autocorr_sharpe(ledger)
        assert blend.primary_max_drawdown == ev._mdd(ledger.equity)


_SUBPROCESS_SCRIPT = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pandas as pd
import src.market_data.services.futures_collection as fc
from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import MhsDiagnosticRequest, run_mhs_horizon_diagnostic, persist_mhs_horizon_diagnostic_report

root = Path(sys.argv[2])
out = Path(sys.argv[3])
ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
fc._mark_price_path = lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
ev._BOOTSTRAP_REPLICATES = 20
ev._BOOTSTRAP_MEAN_BLOCK = 24
ev._BOOTSTRAP_SEED = 20260807
report = run_mhs_horizon_diagnostic(
    MhsDiagnosticRequest(
        start="2021-01-01", end=str(ev.pd.Timestamp(sys.argv[4])),
        data_root=str(root), mark_mode="cache_required",
        execution_timeframe="1m", log_run=False,
        max_rss_bytes=int(sys.argv[5]),
    ),
)
persist_mhs_horizon_diagnostic_report(report, out)
payload = json.loads(out.read_text())
sys.stdout.write(json.dumps({
    "status": report.status,
    "persisted": out.exists(),
    "go_eligible": report.research_go.eligible,
    "reasons": list(report.research_go.reason_codes),
    "stage_count": len(report.resource_measurements),
}))
"""


class TestTerminalPersistenceSubprocess:
    """MHS-28-TERMINAL-FAIL-CLOSED-REPORT: the full pipeline reaches report
    persistence in a fresh process under a fixed RSS budget, proving a
    resource breach yields a serializable terminal rejection rather than an
    uncaught process error or a missing report."""

    @pytest.mark.slow
    def test_full_pipeline_persists_terminal_report_under_fixed_rss(self, tmp_path) -> None:
        import subprocess
        import sys

        root = tmp_path / "market"
        end = _write_mhs_market(root, DEV_SYMBOLS)
        out = tmp_path / "terminal.json"
        script = _SUBPROCESS_SCRIPT
        proc = subprocess.run(  # noqa: S603 - fully static interpreter + fixed script
            [sys.executable, "-c", script, str(Path(__file__).resolve().parents[3]), str(root), str(out), str(end), "1"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert out.exists(), "terminal report must be persisted even under a fixed RSS budget"
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        assert result["status"] == "COMPLETE"
        assert result["persisted"] is True
        assert result["go_eligible"] is False
        assert "RESOURCE_BUDGET_BREACH" in result["reasons"]
        assert result["stage_count"] > 0
        payload = json.loads(out.read_text())
        assert payload["status"] == "COMPLETE"
        assert any(
            b.get("failure", {}).get("reason") == "RESOURCE_BUDGET_BREACH"
            for b in payload["books"].values()
        )


class TestMhsRefactorBitIdenticalReport:
    """SCENARIO_MHS_REFACTOR_07: the memory/scheduling refactor
    (fork-COW shared payloads, RAM-aware worker counts, candidate-weight
    dedup, byte-budgeted window materialization, dead-cache removal) must
    never alter a computed field of ``MhsHorizonDiagnosticReport``. The
    synthetic-market full pipeline is compared bit-identically to the
    pre-refactor golden payload (docs/results/mhs_refactor_baseline.json,
    generated from the pre-refactor commit)."""

    _GOLDEN = Path(__file__).resolve().parents[3] / "docs" / "results" / "mhs_refactor_baseline.json"

    @pytest.fixture(scope="module")
    def refactor_report(self, synthetic_market) -> MhsHorizonDiagnosticReport:
        import src.market_data.services.futures_collection as fc

        root, end = synthetic_market
        originals = {
            "funding_path": ev.funding_path,
            "mark_price_path": fc._mark_price_path,
            "_BOOTSTRAP_REPLICATES": ev._BOOTSTRAP_REPLICATES,
            "_BOOTSTRAP_MEAN_BLOCK": ev._BOOTSTRAP_MEAN_BLOCK,
            "_BOOTSTRAP_SEED": ev._BOOTSTRAP_SEED,
            "_bootstrap_ci": ev._bootstrap_ci,
            "_placebo_sharpe_percentile": ev._placebo_sharpe_percentile,
        }

        ev.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        ev._BOOTSTRAP_REPLICATES = 20
        ev._BOOTSTRAP_MEAN_BLOCK = 24
        ev._BOOTSTRAP_SEED = 20260807
        # Sibling module-scoped fixtures stub these for their own speed; the
        # golden bit-identity contract requires the real implementations.
        ev._bootstrap_ci = _real_bootstrap_ci
        ev._placebo_sharpe_percentile = _real_placebo_sharpe_percentile
        try:
            return run_mhs_horizon_diagnostic(
                MhsDiagnosticRequest(
                    start=str(START), end=str(end), data_root=str(root),
                    execution_timeframe="1m", log_run=False, discovery_gate=True,
                ),
            )
        finally:
            for name, value in originals.items():
                if name == "mark_price_path":
                    fc._mark_price_path = value
                else:
                    setattr(ev, name, value)

    def _scalar(self, x):
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x

    def _project(self, report: MhsHorizonDiagnosticReport) -> dict:
        out = {
            "status": report.status,
            "research_go": {
                "eligible": bool(report.research_go.eligible),
                "reason_codes": sorted(report.research_go.reason_codes),
            },
            "blend_target_gross": self._scalar(report.blend_target_gross),
            "blend_cash_fraction": self._scalar(report.blend_cash_fraction),
            "eligible_symbols": report.eligible_symbols,
            "realized_execution_roster_size": self._scalar(report.realized_execution_roster_size),
            "deflated_sharpe_ratio": self._scalar(report.deflated_sharpe_ratio),
            "xs_rank_ic": self._scalar(report.xs_rank_ic),
            "horizon_diagnostics": {
                k: self._scalar(v) for k, v in report.horizon_diagnostics.items()
            },
            "bootstrap_ci": (
                None
                if report.bootstrap_ci is None
                else [self._scalar(x) for x in report.bootstrap_ci]
            ),
            "placebo_sharpe_percentile": self._scalar(report.placebo_sharpe_percentile),
        }
        for bname in ("fast_reversal", "slow_momentum"):
            book = report.books[bname]
            out[bname] = {
                "horizon_hours": book.horizon_hours,
                "primary_naive_sharpe": self._scalar(book.primary_naive_sharpe),
                "primary_net_ann": self._scalar(book.primary_net_ann),
                "primary_geometric_cagr": self._scalar(book.primary_geometric_cagr),
                "primary_max_drawdown": self._scalar(book.primary_max_drawdown),
                "primary_autocorr_sharpe": self._scalar(book.primary_autocorr_sharpe),
                "stress_naive_sharpe": self._scalar(book.stress_naive_sharpe),
                "failure": None if book.failure is None else book.failure.reason,
            }
        blend = report.blend
        out["blend"] = {
            "horizon_hours": blend.horizon_hours,
            "primary_naive_sharpe": self._scalar(blend.primary_naive_sharpe),
            "primary_net_ann": self._scalar(blend.primary_net_ann),
            "primary_geometric_cagr": self._scalar(blend.primary_geometric_cagr),
            "primary_max_drawdown": self._scalar(blend.primary_max_drawdown),
            "primary_autocorr_sharpe": self._scalar(blend.primary_autocorr_sharpe),
            "stress_naive_sharpe": self._scalar(blend.stress_naive_sharpe),
            "failure": None if blend.failure is None else blend.failure.reason,
        }
        out["folds"] = []
        for fold in report.folds:
            out["folds"].append({
                "fold_index": fold.fold_index,
                "primary_autocorr_sharpe": self._scalar(fold.primary_autocorr_sharpe),
                "primary_naive_sharpe": self._scalar(fold.primary_naive_sharpe),
                "primary_net_ann": self._scalar(fold.primary_net_ann),
                "primary_geometric_cagr": self._scalar(fold.primary_geometric_cagr),
                "primary_max_drawdown": self._scalar(fold.primary_max_drawdown),
                "stress_naive_sharpe": self._scalar(fold.stress_naive_sharpe),
                "decision_intents": fold.decision_intents,
                "failures": sorted(fold.failures),
                "slow_horizon_hours": fold.slow_horizon_hours,
                "fast_horizon_hours": fold.fast_horizon_hours,
            })
        if report.discovery_qualification is not None:
            out["discovery_qualification"] = {}
            for k, v in report.discovery_qualification.items():
                if v is None:
                    out["discovery_qualification"][k] = None
                else:
                    out["discovery_qualification"][k] = {
                        "selected_horizon": self._scalar(v.selected_horizon),
                        "admitted": bool(v.admitted),
                        "qualification_net_t": self._scalar(v.qualification_net_t),
                        "qualification_adjusted_net_t": self._scalar(v.qualification_adjusted_net_t),
                        "qualification_regime_scaled_net_t": self._scalar(
                            v.qualification_regime_scaled_net_t,
                        ),
                    }
        return out

    def test_report_fields_bit_identical_to_pre_refactor_golden(
        self, refactor_report,
    ) -> None:
        golden = json.loads(self._GOLDEN.read_text())
        projected = self._project(refactor_report)

        def _nan_equal(a, b) -> bool:
            if isinstance(a, float) and isinstance(b, float):
                return a == b or (np.isnan(a) and np.isnan(b))
            return a == b

        def _deep_equal(a, b) -> bool:
            if isinstance(a, dict) and isinstance(b, dict):
                return set(a) == set(b) and all(_deep_equal(a[k], b[k]) for k in a)
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b, strict=True))
            return _nan_equal(a, b)

        assert _deep_equal(projected, golden)

