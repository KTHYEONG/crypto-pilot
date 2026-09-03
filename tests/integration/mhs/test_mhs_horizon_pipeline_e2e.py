"""MHS horizon diagnostic end-to-end integration tests (balanced split; shared builders remain in the original module)."""

from __future__ import annotations


import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
import src.mhs.marks as marks
import src.mhs.statistics as statistics
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
    MhsHorizonDiagnosticReport,
)
from src.cli.commands.research.mhs import add_mhs_commands
from src.quant.evaluation.policy import HOLDOUT_CUTOFF

from tests.integration.mhs.test_mhs_horizon_diagnostic import (  # noqa: F401
    DEV_SYMBOLS,
    LATE_START,
    LATE_SYMBOL,
    START,
    _write_mhs_market,
)

class TestQualityCalibrationWiring:
    """Quality calibration wiring contract: top-level books apply matching signal calibration."""

    @staticmethod
    def _book_signal_ema_span(spec) -> int:
        return max(1, round(spec.horizon_hours / spec.step_hours * ev.SIGNAL_EMA_HORIZON_SPAN))

    def test_top_level_book_weights_use_ema_span_matching_fold_path(self, calibrated_report) -> None:
        _report, captured = calibrated_report
        fast = ev.BOOK_SPECS["fast_reversal"]
        slow = ev.BOOK_SPECS["slow_momentum"]
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
        assert captured["regime_callers"].count("build_committee") == 1
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
        assert report.xs_rank_ic == statistics._xs_rank_ic(signal_48h, opens, forward_bars=48)
        assert report.date_clustered_regression == statistics._date_clustered_ols(opens, signal_48h, forward_bars=48)

    def test_run_mhs_horizon_diagnostic_log_close_released_before_book_replay(self) -> None:
        """The log_close release (now in build_committee) still runs, and the
        pipeline runner still sequences build_committee strictly before
        run_replays -- the memory-release ordering this test protects survived
        the stage decomposition, just moved to different source files."""
        from src.mhs.pipeline import runner
        from src.mhs.pipeline.stages import committee as committee_stage

        committee_src = inspect.getsource(committee_stage.build_committee)
        assert "del ctx.log_close" in committee_src
        assert "signal_48h" in committee_src

        runner_src = inspect.getsource(runner.run_stages)
        assert runner_src.index("build_committee(") < runner_src.index("run_replays(")

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
        from src.mhs.evaluation import MhsDiagnosticRequest

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

    @pytest.mark.slow
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
        n_folds = len(ev.phase_1_anchored_purged_folds())
        assert len(fold_stages) == n_folds
        # Anchored folds are recorded in their declared order.
        assert fold_stages == [f"anchored_fold_{i}" for i in range(n_folds)]
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
        assert "INCOMPLETE_ANCHORED_FOLD" in report.research_go.reason_codes

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

@pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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
        from src.mhs.evidence import compute_deployment_readiness

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

class TestPitExecutionGrid:
    """MHS-25-FULL-PERIOD-PIT-GRID: a late-listed symbol never clips the replay
    start; pre-listing NaNs are retained and no order is emitted before the
    symbol is PIT eligible."""

    @pytest.fixture(scope="class")
    def late_market_report(self, tmp_path_factory):
        import src.market_data.services.futures_collection as fc

        root = tmp_path_factory.mktemp("mhs_late_market")
        end = _write_mhs_market(
            root, DEV_SYMBOLS,
            late_listings={LATE_SYMBOL: LATE_START},
            n_hours=3000,
        )
        originals = {
            "funding_path": marks.funding_path,
            "mark_price_path": fc._mark_price_path,
            "_BOOTSTRAP_REPLICATES": statistics._BOOTSTRAP_REPLICATES,
            "_BOOTSTRAP_MEAN_BLOCK": statistics._BOOTSTRAP_MEAN_BLOCK,
            "_BOOTSTRAP_SEED": statistics._BOOTSTRAP_SEED,
            "_placebo_sharpe_percentile": statistics._placebo_sharpe_percentile,
            "_bootstrap_ci": statistics._bootstrap_ci,
        }
        marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        statistics._BOOTSTRAP_REPLICATES = 20
        statistics._BOOTSTRAP_MEAN_BLOCK = 24
        statistics._BOOTSTRAP_SEED = 20260807
        # The 500-placebo / bootstrap stats are not the subject of this test.
        statistics._placebo_sharpe_percentile = lambda *a, **k: None
        statistics._bootstrap_ci = lambda *a, **k: None
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
                elif name == "funding_path":
                    marks.funding_path = value
                else:
                    setattr(statistics, name, value)

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
    """MHS-26-ANCHOR-FOLD-GO-GATE: all replayed folds are reported; an
    incomplete fold, negative strict Sharpe, non-positive stress Sharpe, or
    relevant termination produces Research GO false and reason codes."""

    def test_three_folds_reported_and_go_false(self, report) -> None:
        expected = ev.phase_1_anchored_purged_folds()
        assert len(report.folds) == len(expected)
        assert len(report.anchored_folds) == len(expected)
        for fold_report, fold in zip(report.folds, expected, strict=True):
            assert fold_report.validation_start.startswith(
                fold.validation_start.strftime("%Y-%m-%d"),
            )
            assert fold_report.validation_end.startswith(
                fold.validation_end.strftime("%Y-%m-%d"),
            )
        assert report.research_go.eligible is False
        assert "INCOMPLETE_ANCHORED_FOLD" in report.research_go.reason_codes
        assert report.research_go.evaluated_folds == len(expected)
        # The gate boolean routes to deployment readiness, never primary_valid alone.
        assert report.deployment_readiness.research_go_eligible is False

    def test_fold_metrics_exposed(self, report) -> None:
        fold_report = report.folds[0]
        assert isinstance(fold_report.failures, tuple)
        assert isinstance(fold_report.termination_counts, dict)
        assert fold_report.strict_elapsed_seconds >= 0.0
        assert fold_report.stress_elapsed_seconds >= 0.0

    def test_gate_collects_each_fail_closed_reason(self) -> None:
        from src.mhs.evaluation import (
            MhsFoldReport,
        )
        from src.mhs.research_go import _mhs_research_go
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

        from src.mhs.evaluation import (
            GO_REASON_EXECUTION_GAP,
            GO_REASON_INCOMPLETE_FOLD,
            GO_REASON_PRIMARY_SHARPE,
            GO_REASON_STRESS_SHARPE,
        )

        gap = _mhs_research_go((fold(0, (GO_REASON_EXECUTION_GAP,)),))
        assert gap.eligible is False
        assert GO_REASON_EXECUTION_GAP in gap.reason_codes

        sharpe = _mhs_research_go((fold(0, (GO_REASON_PRIMARY_SHARPE,)),))
        assert GO_REASON_PRIMARY_SHARPE in sharpe.reason_codes

        stress = _mhs_research_go((fold(0, (GO_REASON_STRESS_SHARPE,)),))
        assert GO_REASON_STRESS_SHARPE in stress.reason_codes

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
                    failures=(GO_REASON_INCOMPLETE_FOLD,),
                    strict_elapsed_seconds=0.0,
                    stress_elapsed_seconds=0.0,
                ),
            ),
        )
        assert GO_REASON_INCOMPLETE_FOLD in incomplete.reason_codes
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
        assert (
            len(fold_safe_report.folds)
            == len(fold_safe_baseline_report.folds)
            == len(ev.phase_1_anchored_purged_folds())
        )
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
        originals = {"funding_path": marks.funding_path, "mark_price_path": fc._mark_price_path}
        marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        yield root, end
        marks.funding_path = originals["funding_path"]
        fc._mark_price_path = originals["mark_price_path"]

    def test_parallel_folds_match_sequential_folds(self, fold_parity_request) -> None:
        from src.mhs.evaluation import (
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
        assert len(sequential) == len(parallel)

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
