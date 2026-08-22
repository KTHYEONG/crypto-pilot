"""MHS horizon diagnostic end-to-end integration tests (balanced split; shared builders remain in the original module)."""

from __future__ import annotations


import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.marks as marks
import src.application.research.mhs.marks as mhs_marks
import src.application.research.mhs.statistics as statistics
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
    MhsHorizonDiagnosticReport,
    run_mhs_horizon_diagnostic,
)

from tests.integration.mhs.test_mhs_horizon_diagnostic import (  # noqa: F401
    DEV_SYMBOLS,
    FOLD_WINDOW_FOLD,
    START,
    _SUBPROCESS_SCRIPT,
    _write_mhs_market,
)

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
            GO_REASON_UNSPECIFIED_POLICY,
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
            failures=(GO_REASON_UNSPECIFIED_POLICY,),
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

class TestFoldWindowTelemetryOracle:
    """MHS-31-FOLD-WINDOW-TELEMETRY: a synthetic fold records monotonically
    ordered per-window telemetry and is numerically equivalent to the
    single-panel oracle."""

    @pytest.fixture(scope="class")
    def fold_market(self, tmp_path_factory) -> tuple[Path, pd.Timestamp]:
        import src.market_data.services.futures_collection as fc

        root = tmp_path_factory.mktemp("mhs_fold_market")
        end = _write_mhs_market(root, DEV_SYMBOLS, n_hours=3000)
        originals = {
            "funding_path": marks.funding_path,
            "mark_price_path": fc._mark_price_path,
        }
        marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        yield root, end
        marks.funding_path = originals["funding_path"]
        fc._mark_price_path = originals["mark_price_path"]

    def _single_panel_oracle(
        self, root: Path, funding_by_symbol: dict[str, pd.Series],
    ):
        """Replicate the fold's pre-replay decision construction via the shared
        ``_build_fold_target_weights`` builder and run the dense single-panel
        oracle (``strategy_aware_execution_replay``) over the whole validation
        window."""
        from src.mhs.types import ExecutionSpec
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
        minute_frames = mhs_marks._align_minute_frames(
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
        assert blend.primary_autocorr_sharpe == statistics._daily_autocorr_sharpe(ledger)
        assert blend.primary_max_drawdown == statistics._mdd(ledger.equity)

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
            "funding_path": marks.funding_path,
            "mark_price_path": fc._mark_price_path,
            "_BOOTSTRAP_REPLICATES": statistics._BOOTSTRAP_REPLICATES,
            "_BOOTSTRAP_MEAN_BLOCK": statistics._BOOTSTRAP_MEAN_BLOCK,
            "_BOOTSTRAP_SEED": statistics._BOOTSTRAP_SEED,
            "_bootstrap_ci": statistics._bootstrap_ci,
            "_placebo_sharpe_percentile": statistics._placebo_sharpe_percentile,
        }

        marks.funding_path = lambda sym: root / "funding" / f"{sym}.parquet"
        fc._mark_price_path = (
            lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet"
        )
        statistics._BOOTSTRAP_REPLICATES = 20
        statistics._BOOTSTRAP_MEAN_BLOCK = 24
        statistics._BOOTSTRAP_SEED = 20260807
        # Sibling module-scoped fixtures stub these for their own speed; the
        # golden bit-identity contract requires the real implementations.
        statistics._bootstrap_ci = statistics._bootstrap_ci
        statistics._placebo_sharpe_percentile = statistics._placebo_sharpe_percentile
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
                elif name == "funding_path":
                    marks.funding_path = value
                else:
                    setattr(statistics, name, value)

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
