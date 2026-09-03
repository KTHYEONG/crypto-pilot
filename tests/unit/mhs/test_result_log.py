"""Run-history ledger contract: append / rotate / prune / latest snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from src.mhs.run_history import (
    RUN_HISTORY_MAX_SHARDS,
    RUN_HISTORY_SHARD_MAX_BYTES,
    append_run_history_record,
    mhs_run_history_dir,
)


def test_append_creates_history_dir_active_line_and_latest(tmp_path) -> None:
    record = {"run_id": "abc", "status": "COMPLETE", "perf": {"run_elapsed_seconds": 1.5}}
    history_dir = tmp_path / "history"
    active = append_run_history_record(record, history_dir)

    assert active.name == "active.jsonl"
    assert active == history_dir / "active.jsonl"
    assert history_dir.is_dir()
    lines = active.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record
    latest = history_dir / "latest.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8")) == record


def test_append_rotates_active_shard_when_over_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.mhs.run_history.RUN_HISTORY_SHARD_MAX_BYTES", 1)
    history_dir = tmp_path / "history"

    append_run_history_record({"run_id": "first"}, history_dir)
    append_run_history_record({"run_id": "second"}, history_dir)
    append_run_history_record({"run_id": "third"}, history_dir)

    archives = sorted(history_dir.glob("mhs_run_history_*.jsonl"))
    assert len(archives) == 2
    assert json.loads(archives[0].read_text(encoding="utf-8").splitlines()[0])["run_id"] == "first"
    assert json.loads(archives[1].read_text(encoding="utf-8").splitlines()[0])["run_id"] == "second"

    active_lines = (history_dir / "active.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(active_lines) == 1
    assert json.loads(active_lines[0])["run_id"] == "third"


def test_append_prunes_oldest_archive_at_retention_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.mhs.run_history.RUN_HISTORY_SHARD_MAX_BYTES", 1)
    monkeypatch.setattr("src.mhs.run_history.RUN_HISTORY_MAX_SHARDS", 3)
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    (history_dir / "mhs_run_history_100.jsonl").write_text('{"run_id": "oldest"}\n', encoding="utf-8")
    (history_dir / "mhs_run_history_200.jsonl").write_text('{"run_id": "mid"}\n', encoding="utf-8")
    (history_dir / "mhs_run_history_300.jsonl").write_text('{"run_id": "recent"}\n', encoding="utf-8")
    (history_dir / "active.jsonl").write_text('{"run_id": "pre-rotation"}\n', encoding="utf-8")

    append_run_history_record({"run_id": "trigger"}, history_dir)

    archives = sorted(history_dir.glob("mhs_run_history_*.jsonl"))
    assert len(archives) == 3
    names = [p.name for p in archives]
    assert "mhs_run_history_100.jsonl" not in names
    assert "mhs_run_history_200.jsonl" in names
    assert "mhs_run_history_300.jsonl" in names
    active_lines = (history_dir / "active.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(active_lines) == 1
    assert json.loads(active_lines[0])["run_id"] == "trigger"


def test_append_no_rotation_when_under_budget(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "a"}, history_dir)
    append_run_history_record({"run_id": "b"}, history_dir)
    assert not list(history_dir.glob("mhs_run_history_*.jsonl"))
    lines = (history_dir / "active.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "a"
    assert json.loads(lines[1])["run_id"] == "b"


def test_latest_snapshot_tracks_most_recent_record(tmp_path) -> None:
    history_dir = tmp_path / "history"
    append_run_history_record({"run_id": "first"}, history_dir)
    append_run_history_record({"run_id": "second"}, history_dir)
    latest = json.loads((history_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "second"


def test_mhs_run_history_dir_derives_from_target_parent() -> None:
    assert mhs_run_history_dir(Path("results/report.json")) == Path("results/mhs_run_history")


def test_shard_constants_are_fixed_bounds() -> None:
    assert RUN_HISTORY_SHARD_MAX_BYTES == 262144
    assert RUN_HISTORY_MAX_SHARDS == 12


class TestFillMarkParityRunHistoryRecord:
    """SCENARIO_MHS_FILL_MARK_PARITY_06: fill_mark_parity in run history record."""

    def test_scenario_mhs_exposure_ceiling_07_census_persisted_in_record(self) -> None:
        """SCENARIO_MHS_FILL_MARK_PARITY_06 / SCENARIO_MHS_EXPOSURE_CEILING_07:
        the run-history flags payload carries exposure_scale_two_sided and its
        value matches the fixture request (MhsDiagnosticRequest() default stays
        False at the contract layer, I5)."""
        from src.mhs.evidence import DeploymentReadinessResult

        from src.mhs.evaluation import (
            MhsDiagnosticRequest,
            MhsHorizonDiagnosticReport,
            MhsOutputTier,
            MhsResearchGoResult,
            build_mhs_run_history_record,
        )

        census = {
            "band": 0.0488,
            "cells_over_band": 7,
            "eligible_cells_removed": 7,
            "symbols": {"FROZEN": 7},
        }
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
            fill_mark_parity=census,
        )
        request = MhsDiagnosticRequest()
        record = build_mhs_run_history_record(report, request, MhsOutputTier.COMPACT, None)
        assert record["fill_mark_parity"]["cells_over_band"] == 7
        assert record["fill_mark_parity"]["band"] == 0.0488
        assert record["flags"]["fill_mark_parity_gate"] is True
        assert record["flags"]["exposure_scale_two_sided"] is False


def test_SCENARIO_MHS_EVID_03_TRIALS_NEVER_UNDERSTATED(tmp_path) -> None:
    """SCENARIO_MHS_EVID_03_TRIALS_NEVER_UNDERSTATED: the DSR trials denominator
    accumulates the history's distinct admissible trial configurations on top
    of the registered constant floor; an unreadable history falls back with an
    explicit 'constant_fallback' provenance."""
    from src.mhs.params import SEARCH_TRIALS_ATTEMPTED
    from src.mhs.run_history import derive_trials_attempted

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
