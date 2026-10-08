"""Run-history registry append contract."""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import json
import sqlite3
from pathlib import Path

from src.lab.mhs.run_history import append_run_history_record


def _history_records(registry: Path) -> list[dict[str, object]]:
    with sqlite3.connect(registry) as conn:
        rows = conn.execute(
            "SELECT record_json FROM history_records ORDER BY ordinal",
        ).fetchall()
    return [json.loads(str(row[0])) for row in rows]


def test_append_creates_history_dir_and_registry(tmp_path) -> None:
    record = {"run_id": "abc", "status": "COMPLETE", "perf": {"run_elapsed_seconds": 1.5}}
    history_dir = tmp_path / "history"
    registry = append_run_history_record(record, history_dir)

    assert registry.name == "registry.sqlite3"
    assert registry == history_dir / "registry.sqlite3"
    assert history_dir.is_dir()
    assert _history_records(registry) == [record]


def test_append_keeps_all_records_in_registry(tmp_path) -> None:
    history_dir = tmp_path / "history"

    append_run_history_record({"run_id": "first"}, history_dir)
    append_run_history_record({"run_id": "second"}, history_dir)
    append_run_history_record({"run_id": "third"}, history_dir)

    assert _history_records(history_dir / "registry.sqlite3") == [
        {"run_id": "first"}, {"run_id": "second"}, {"run_id": "third"},
    ]


def test_append_does_not_mutate_legacy_archives(tmp_path) -> None:
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    (history_dir / "mhs_run_history_100.jsonl").write_text('{"run_id": "oldest"}\n', encoding="utf-8")
    (history_dir / "mhs_run_history_200.jsonl").write_text('{"run_id": "mid"}\n', encoding="utf-8")
    (history_dir / "mhs_run_history_300.jsonl").write_text('{"run_id": "recent"}\n', encoding="utf-8")
    (history_dir / "active.jsonl").write_text('{"run_id": "pre-rotation"}\n', encoding="utf-8")

    append_run_history_record({"run_id": "trigger"}, history_dir)

    assert {p.name for p in history_dir.glob("mhs_run_history_*.jsonl")} == {
        "mhs_run_history_100.jsonl", "mhs_run_history_200.jsonl", "mhs_run_history_300.jsonl",
    }
    assert _history_records(history_dir / "registry.sqlite3") == [{"run_id": "trigger"}]


def test_append_no_rotation_when_under_budget(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "a"}, history_dir)
    append_run_history_record({"run_id": "b"}, history_dir)
    assert not list(history_dir.glob("mhs_run_history_*.jsonl"))
    assert _history_records(history_dir / "registry.sqlite3") == [{"run_id": "a"}, {"run_id": "b"}]


def test_latest_snapshot_tracks_most_recent_record(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "first"}, history_dir)
    append_run_history_record({"run_id": "second"}, history_dir)
    records = _history_records(history_dir / "registry.sqlite3")
    assert records[-1]["run_id"] == "second"


class TestFillMarkParityRunHistoryRecord:
    """SCENARIO_MHS_FILL_MARK_PARITY_06: fill_mark_parity in run history record."""

    def test_scenario_mhs_exposure_ceiling_07_census_persisted_in_record(self) -> None:
        """SCENARIO_MHS_FILL_MARK_PARITY_06 / SCENARIO_MHS_EXPOSURE_CEILING_07:
        the run-history flags payload carries exposure_scale_two_sided and its
        value matches the fixture request (research_baseline() default stays
        False at the contract layer, I5)."""
        from src.lab.mhs.evidence import DeploymentReadinessResult

        from src.lab.mhs.contracts import (
            MhsOutputTier,
            MhsResearchGoResult,
        )
        from src.lab.mhs.report.persist import build_mhs_run_history_record
        from src.lab.mhs.report.schema import MhsHorizonDiagnosticReport

        report = MhsHorizonDiagnosticReport(
            feature="mhs",
            status="COMPLETE",
            start="2021-01-01",
            end="2025-01-01",
            resolved_end="2025-01-01",
            partition="dev",
            execution_tiers_bps=(2.64, 4.18, 6.07),
            books={},
            blend=None,
            blend_target_gross=0.0,
            blend_cash_fraction=1.0,
            eligible_symbols=10,
            trials_attempted=70,
            deflated_sharpe_ratio=None,
            xs_rank_ic={},
            date_clustered_regression={},
            horizon_diagnostics={},
            bootstrap_ci=None,
            placebo_sharpe_percentile=None,
            deployment_readiness=DeploymentReadinessResult(
                geometric_cagr=0.5, max_drawdown=-0.2, calmar=2.5,
                expected_shortfall=0.0, worst_1d=0.0, worst_7d=0.0, worst_event=0.0,
                time_under_water_bars=0, recovery_bars=None,
                probability_final_wealth_below_initial=0.0,
                probability_mdd_over_20pct=0.0, probability_mdd_over_30pct=0.0,
                leverage_ruin_probabilities={}, concentration={}, participation_warnings={},
                research_go_eligible=False, execution_go_eligible=False,
                pilot_go_eligible=False, scale_go_eligible=False,
            ),
            synthetic_stress={},
            participation_warnings={},
            termination_counts={},
            unsupported_assumptions=(),
            anchored_folds=(),
            folds=(),
            research_go=MhsResearchGoResult(
                eligible=False, reason_codes=(), evaluated_folds=0, folds_passed=0,
            ),
            fill_source="OHLCV",
            mark_source="MARK",
            execution_timeframe="3m",
            execution_universe_size=30,
            execution_symbols=(),
            run_elapsed_seconds=1.0,
        )
        request = research_baseline()
        record = build_mhs_run_history_record(report, request, MhsOutputTier.COMPACT, None)
        assert record["flags"]["exposure_scale_two_sided"] is False


def test_SCENARIO_MHS_EVID_03_TRIALS_NEVER_UNDERSTATED(tmp_path) -> None:
    """SCENARIO_MHS_EVID_03_TRIALS_NEVER_UNDERSTATED: the DSR trials denominator
    accumulates the history's distinct admissible trial configurations on top
    of the registered constant floor; an unreadable history falls back with an
    explicit 'constant_fallback' provenance."""
    from src.core.params import SEARCH_TRIALS_ATTEMPTED
    from src.lab.mhs.run_history import derive_trials_attempted

    def _trial(run_id: str, universe_size: int) -> dict[str, object]:
        return {
            "run_id": run_id,
            "status": "COMPLETE",
            "flags": {"execution_universe_size": universe_size},
            "blend": {"primary_naive_sharpe": 1.0},
            "research_go": {"reason_codes": [], "data_integrity_reason_codes": []},
        }

    # 6 distinct flag configurations: constant floor + observed history.
    small_dir = tmp_path / "history_small"
    for index in range(6):
        append_run_history_record(
            _trial(f"r{index}", 30 + (index % 6)), small_dir
        )
    assert derive_trials_attempted(small_dir) == (
        SEARCH_TRIALS_ATTEMPTED + 6,
        "constant_plus_ledger",
    )

    # 200 distinct configurations: every observation adds on top of the floor.
    big_dir = tmp_path / "history_big"
    for index in range(200):
        append_run_history_record(_trial(f"r{index}", index), big_dir)
    assert derive_trials_attempted(big_dir) == (
        SEARCH_TRIALS_ATTEMPTED + 200,
        "constant_plus_ledger",
    )

    # Missing directory: unreadable -> conservative fallback, never a guess.
    missing = derive_trials_attempted(tmp_path / "does_not_exist")
    assert missing == (SEARCH_TRIALS_ATTEMPTED, "constant_fallback")

    # The floor holds regardless of source.
    for count, _source in (
        derive_trials_attempted(small_dir),
        derive_trials_attempted(big_dir),
        missing,
    ):
        assert count >= SEARCH_TRIALS_ATTEMPTED
