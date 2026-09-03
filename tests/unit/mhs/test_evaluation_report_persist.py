"""MHS evaluation core tests (second-level split by domain)."""

"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import json
import dataclasses
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
import src.mhs.report.persist as persist_mod
from src.common.errors import DataIntegrityError
from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _FOLD,
    _START,
    _assert_books_equal,
    _assert_regime_vol_mean_roster_masked,
    _build_book_outcome_args,
    _build_books_concurrent_args,
    _build_compact_report,
    _deployment_readiness,
    _dispatch_spec,
    _gap_mixed_replay,
    _passing_fold_report,
    _perf_opt_placebo_inputs,
    _pre_change_slow_book,
    _reference_bootstrap_ci,
    _reference_participation_warnings,
    _reference_placebo_percentile,
    _reference_resolve_ns_scalar,
    _reference_weights,
    _roster_mask_panel_inputs,
    _sequential_book_reports,
    _signal_disagreement_panel,
    _slow_book_panel_inputs,
    _synthetic_ledger,
    _write_3m_cache,
    _write_quote_volume_market,
)

def test_mhs_output_tier_enum_values() -> None:
    assert ev.MhsOutputTier.COMPACT.value == "compact"
    assert ev.MhsOutputTier.FULL.value == "full"
    assert ev.MhsOutputTier("compact") is ev.MhsOutputTier.COMPACT
    assert ev.MhsOutputTier("full") is ev.MhsOutputTier.FULL

def test_regression_existing_report_fields_unchanged() -> None:
    # SCENARIO_REGRESSION_EXISTING_REPORT_FIELDS_UNCHANGED: the two-pass
    # change must not rename or drop any existing MhsBookReport/MhsFoldReport
    # field (which doubles as the JSON key via to_payload()) -- only
    # pre_vol_target_reference/pre_vol_target_reference_naive_sharpe are new,
    # following the exact patient_reference field-addition precedent.
    book_fields = {f.name for f in dataclasses.fields(ev.MhsBookReport)}
    for field_name in (
        "name", "band", "horizon_hours", "step_hours", "tranche_count",
        "n_symbols", "phase", "prescreen", "tail", "primary", "stress",
        "primary_autocorr_sharpe", "primary_naive_sharpe", "primary_net_ann",
        "primary_geometric_cagr", "primary_max_drawdown",
        "primary_annualized_turnover", "stress_naive_sharpe",
        "terminal_censored_decisions", "failure", "touch", "touch_naive_sharpe",
        "ladder", "ladder_naive_sharpe", "patient_reference",
        "patient_reference_naive_sharpe",
    ):
        assert field_name in book_fields
    assert "pre_vol_target_reference" in book_fields
    assert "pre_vol_target_reference_naive_sharpe" in book_fields

    fold_fields = {f.name for f in dataclasses.fields(ev.MhsFoldReport)}
    for field_name in (
        "fold_index", "validation_start", "validation_end", "strict", "stress",
        "primary_valid", "primary_autocorr_sharpe", "primary_naive_sharpe",
        "primary_net_ann", "primary_geometric_cagr", "primary_max_drawdown",
        "stress_naive_sharpe", "decision_intents", "termination_counts",
        "failures", "strict_elapsed_seconds", "stress_elapsed_seconds",
        "terminal_censored_decisions",
    ):
        assert field_name in fold_fields

    report = _build_compact_report()
    book = report.books["fast_reversal"]
    for key in ("primary", "patient_reference", "patient_reference_naive_sharpe"):
        assert key in dataclasses.asdict(book)
    assert "pre_vol_target_reference" in dataclasses.asdict(book)
    payload = report.to_payload()
    assert "pre_vol_target_reference" in payload["books"]["fast_reversal"]

def test_daily_resample_ledger_fidelity() -> None:
    # COMPACT_DAILY_LEDGER_FIDELITY: the daily rollup preserves the source
    # ledger's per-day first/max/min/last equity and the cross-day return.
    idx = pd.date_range("2021-01-01", periods=48 * 3, freq="30min", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "equity": equity.to_numpy(),
            "fill_turnover": 0.0,
        }
    )
    frame.loc[2, "fill_turnover"] = 0.5
    frame.loc[5, "fill_turnover"] = 0.25
    daily = ev._daily_resample_ledger(frame)
    assert len(daily) == 3
    assert list(daily.columns) == [
        "date", "equity_open", "equity_high", "equity_low", "equity_close",
        "daily_turnover", "daily_fill_count", "daily_return",
    ]
    d0 = daily.iloc[0]
    day0 = idx.normalize()[0]
    day0_mask = idx < day0 + pd.Timedelta("1D")
    day0_eq = frame.loc[day0_mask, "equity"]
    assert d0["equity_open"] == pytest.approx(day0_eq.iloc[0], rel=1e-6)
    assert d0["equity_high"] == pytest.approx(day0_eq.max(), rel=1e-6)
    assert d0["equity_low"] == pytest.approx(day0_eq.min(), rel=1e-6)
    assert d0["equity_close"] == pytest.approx(day0_eq.iloc[-1], rel=1e-6)
    assert d0["daily_turnover"] == pytest.approx(0.75, rel=1e-6)
    assert d0["daily_fill_count"] == 2
    assert np.isnan(d0["daily_return"])
    d1 = daily.iloc[1]
    day1_mask = (idx >= day0 + pd.Timedelta("1D")) & (idx < day0 + pd.Timedelta("2D"))
    day1_eq = frame.loc[day1_mask, "equity"]
    assert d1["equity_open"] == pytest.approx(day1_eq.iloc[0], rel=1e-6)
    assert d1["daily_return"] == pytest.approx(day1_eq.iloc[-1] / d0["equity_close"] - 1.0, rel=1e-6)

def test_daily_resample_ledger_fails_closed_on_bad_equity() -> None:
    idx = pd.date_range("2021-01-01", periods=48, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "equity": [100.0] * 47 + [np.nan],
            "fill_turnover": 0.0,
        }
    )
    with pytest.raises(DataIntegrityError, match="equity"):
        ev._daily_resample_ledger(frame)

def test_compact_json_stripped_and_wired(tmp_path) -> None:
    # COMPACT_JSON_STRIPPED: compact persist drops per-replay SHA-256/schema
    # references while retaining only row counts and the scalar report fields.
    report = _build_compact_report()
    out = tmp_path / "mhs_report.json"
    persisted = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )
    assert persisted == out
    payload = json.loads(out.read_text())
    raw = json.dumps(payload)
    assert "checksum_sha256" not in raw
    assert "schema_version" not in raw
    assert "time_bounds" not in raw
    ref = payload["books"]["fast_reversal"]["primary"]
    assert set(ref) == {"fills", "units", "notional_weights", "ledger", "times"}
    assert all(set(v) == {"row_count"} for v in ref.values())
    assert ref["ledger"]["row_count"] == len(report.books["fast_reversal"].primary.ledger.equity)
    assert ref["fills"]["row_count"] == len(report.books["fast_reversal"].primary.simulated_fills)
    assert payload["status"] == "COMPLETE"
    assert "daily_ledger" in payload["artifacts"]
    assert set(payload["artifacts"]["fills"]) == {"file", "row_count"}
    assert "fast_reversal_primary" in payload["replay_ids"]

def test_compact_size_budget(tmp_path) -> None:
    # COMPACT_SIZE_BUDGET: compact artifacts stay far below the git-friendly
    # budgets (daily ledger < 500KB, JSON < 20KB) for a small replay workload.
    report = _build_compact_report()
    out = tmp_path / "mhs_report.json"
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.COMPACT)
    artifact_dir = out.parent / "mhs_report_artifacts"
    daily_path = artifact_dir / "daily_ledger.parquet"
    assert daily_path.exists()
    assert daily_path.stat().st_size < 500 * 1024
    assert out.stat().st_size < 20 * 1024
    daily = pd.read_parquet(daily_path)
    assert "replay_id" in daily.columns
    assert daily["replay_id"].eq("fast_reversal_primary").all()
    assert len(daily) == 4
    assert daily["equity_close"].gt(0).all()

def test_compact_failure_escalates_past_artifacts(tmp_path, monkeypatch) -> None:
    # A non-DataIntegrityError resample failure logs and returns None without
    # writing compact artifacts (fail-closed escalation).
    report = _build_compact_report()

    def _boom(_table):
        raise RuntimeError("boom")

    monkeypatch.setattr(persist_mod, "_daily_resample_ledger", _boom)
    out = tmp_path / "mhs_report.json"
    persisted = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )
    assert persisted is None
    assert not out.exists()

def test_gitignore_full_subdir_only() -> None:
    # GITIGNORE_FULL_SUBDIR: only the _full/ audit subdirectory is gitignored;
    # the compact daily ledger path and summary JSON stay trackable.
    gitignore = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docs/results/mhs_horizon_diagnostic_artifacts/_full/" in gitignore
    assert "docs/results/mhs_horizon_diagnostic_artifacts/" not in gitignore
    assert "docs/results/mhs_horizon_diagnostic.json" not in gitignore
    assert "docs/results/mhs_horizon_diagnostic_artifacts/daily_ledger.parquet" not in gitignore

def test_persist_wires_run_history_append_for_compact_and_full(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_05: ``persist_mhs_horizon_diagnostic_report``
    calls ``append_run_history_record`` exactly once per COMPACT/FULL tier."""
    report = _build_compact_report()
    calls: list[tuple[str, str]] = []

    def _spy_append(record, history_dir):
        calls.append((record["output_tier"], str(history_dir)))
        return Path(history_dir) / "active.jsonl"

    monkeypatch.setattr(persist_mod, "append_run_history_record", _spy_append)
    out = tmp_path / "mhs_report.json"
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.COMPACT)
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.FULL)

    assert len(calls) == 2
    assert [tier for tier, _ in calls] == ["compact", "full"]
    assert all(history_dir.endswith("mhs_run_history") for _, history_dir in calls)

def test_persist_still_appends_when_compact_resample_fails(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_05 (COMPACT-None branch): the COMPACT path that
    returns ``None`` (resample failure escalated past artifacts) still appends
    a history record."""
    report = _build_compact_report()
    calls: list = []

    def _boom(_table):
        raise RuntimeError("boom")

    def _spy_append(record, history_dir):
        calls.append(record)
        return Path(history_dir) / "active.jsonl"

    monkeypatch.setattr(persist_mod, "_daily_resample_ledger", _boom)
    monkeypatch.setattr(persist_mod, "append_run_history_record", _spy_append)
    out = tmp_path / "mhs_report.json"
    persisted = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )

    assert persisted is None
    assert len(calls) == 1
    assert calls[0]["output_tier"] == "compact"

def test_persist_isolates_history_append_failure(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_06: an exception from ``append_run_history_record``
    never propagates and never changes the persist return value."""
    report = _build_compact_report()
    out = tmp_path / "mhs_report.json"

    baseline = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )

    def _boom(record, history_dir):
        raise RuntimeError("history boom")

    monkeypatch.setattr(persist_mod, "append_run_history_record", _boom)
    isolated = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )

    assert isolated == baseline
