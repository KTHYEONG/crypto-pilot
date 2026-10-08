"""Consulted data horizon completeness over the canonical run-history registry (I-SOURCE-COMPLETE).

Every independent consultation record must raise the horizon: sealed ceilings,
canonical registry ``history_records`` and registered ``runs``, procedure
evaluation events, and the process research journal. Existing-but-corrupt
evidence fails closed instead of degrading into "no evidence".
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.backtests.contracts import RunRegistration
from src.backtests.migration import migrate_legacy_backtests
from src.backtests.registry import initialize_registry, register_run
from src.common.errors import DataIntegrityError
from src.common.paths import BACKTESTS_DIR, BASE_DIR
from src.mhs.backtest.journal import initialize_research_journal
from src.core.params import MHS_FINAL_OOS_CUTOFF_2026H1
from src.mhs.preregistration import (
    EVENT_EVALUATION,
    LEGACY_PROCEDURE_REGISTRY_PATH,
    PROCEDURE_REGISTRY_PATH,
    _append_event,
    consulted_data_horizon,
    register_procedure,
)
from src.mhs.run_history import append_run_history_record, consulted_registry_horizon

_FORWARD_LOOK: dict[str, Any] = {
    "status": "COMPLETE",
    "start": "2021-01-01 00:00:00+00:00",
    "end": "2026-09-30",
    "resolved_end": "2026-09-30 23:59:59+00:00",
    "flags": {"forward_registration_digest": "a" * 32},
    "params_snapshot": {},
    "blend": {"primary_naive_sharpe": 1.0},
}


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _request() -> Any:
    from src.mhs.contracts import MhsDiagnosticRequest

    return MhsDiagnosticRequest()


def _registry(home: Path) -> Path:
    initialize_registry(home / "registry.sqlite3")
    return home / "registry.sqlite3"


def _register_run(registry: Path, request: dict[str, Any], *, run_id: str | None = None) -> str:
    identifier = run_id if run_id is not None else uuid4().hex
    register_run(
        registry,
        RunRegistration(
            run_id=identifier,
            strategy_id="s",
            registered_at="2026-10-01T00:00:00+00:00",
            request=request,
            managed_directory=None,
        ),
    )
    return identifier


def _corrupt(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "registry.sqlite3").write_text("not-a-db", encoding="utf-8")


def test_sealed_ceiling_floor_without_any_evidence(tmp_path: Path) -> None:
    horizon = consulted_data_horizon(tmp_path / "history", tmp_path / "reg.jsonl")
    assert horizon == MHS_FINAL_OOS_CUTOFF_2026H1


def test_registry_forward_look_advances_horizon(tmp_path: Path) -> None:
    """Regression of the b_horizon_gap contamination: the sqlite registry is evidence."""
    history = tmp_path / "history"
    append_run_history_record(_FORWARD_LOOK, history)
    assert consulted_data_horizon(history, tmp_path / "reg.jsonl") == _utc("2026-09-30 23:59:59")


def test_backdated_registration_inside_consulted_window_is_rejected(tmp_path: Path) -> None:
    history = tmp_path / "history"
    registry = tmp_path / "reg.jsonl"
    append_run_history_record(_FORWARD_LOOK, history)
    with pytest.raises(DataIntegrityError, match="precedes consulted data horizon"):
        register_procedure(
            _request(), now=_utc("2026-08-01"), registry_path=registry, history_dir=history
        )
    assert not registry.exists()


def test_end_fallback_when_resolved_end_is_null(tmp_path: Path) -> None:
    history = tmp_path / "history"
    append_run_history_record({"resolved_end": None, "end": "2026-08-15"}, history)
    assert consulted_data_horizon(history, tmp_path / "reg.jsonl") == _utc("2026-08-15")


def test_status_and_admission_blindness(tmp_path: Path) -> None:
    """A failed or non-trial look still read its data, so both count."""
    history = tmp_path / "history"
    append_run_history_record({"status": "FAILED", "resolved_end": "2026-08-20"}, history)
    append_run_history_record(
        {"status": "COMPLETE", "resolved_end": "2026-08-10", "blend": {"primary_naive_sharpe": float("nan")}},
        history,
    )
    assert consulted_data_horizon(history, tmp_path / "reg.jsonl") == _utc("2026-08-20")


def test_registered_backtest_run_counts_as_consulted(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "home")
    _register_run(registry, {"start": "2021-01-01", "end": "2026-09-15T00:00:00+00:00"})
    assert consulted_data_horizon(tmp_path / "home", tmp_path / "reg.jsonl") == _utc("2026-09-15")


@pytest.mark.parametrize("registered_request", [{"start": "2021-01-01"}, {"start": "2021-01-01", "end": None}])
def test_run_without_end_fails_closed_naming_the_run(tmp_path: Path, registered_request: dict[str, Any]) -> None:
    home = tmp_path / "home"
    registry = _registry(home)
    _register_run(registry, {"start": "2021-01-01", "end": "2026-09-15T00:00:00+00:00"})
    _register_run(registry, registered_request, run_id="f" * 32)
    procedure_registry = tmp_path / "reg.jsonl"
    with pytest.raises(DataIntegrityError, match="f" * 32):
        consulted_data_horizon(home, procedure_registry)
    with pytest.raises(DataIntegrityError, match="f" * 32):
        register_procedure(
            _request(), now=_utc("2027-01-01"), registry_path=procedure_registry, history_dir=home
        )
    assert not procedure_registry.exists()


def test_migrated_legacy_run_carries_its_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-2026"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"status": "COMPLETE", "start": "2021-01-01T00:00:00+00:00", "end": "2026-09-20T00:00:00+00:00"}),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    registry = home / "registry.sqlite3"
    migrate_legacy_backtests(
        registry_path=registry, history_directories=(), run_directories=(run_dir,), dry_run=False
    )
    assert consulted_registry_horizon(registry) == _utc("2026-09-20")


def test_registry_without_a_runs_table_fails_closed(tmp_path: Path) -> None:
    """``initialize_registry`` always creates ``runs``, so its absence is corruption."""
    import sqlite3

    home = tmp_path / "home"
    registry = _registry(home)
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute("DROP TABLE runs")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError, match=r"registry\.sqlite3"):
        consulted_registry_horizon(registry)


def test_history_record_without_any_end_makes_no_temporal_claim(tmp_path: Path) -> None:
    history = tmp_path / "history"
    append_run_history_record({"status": "COMPLETE", "blend": {"primary_naive_sharpe": 1.0}}, history)
    assert consulted_data_horizon(history, tmp_path / "reg.jsonl") == MHS_FINAL_OOS_CUTOFF_2026H1


@pytest.mark.parametrize("payload", ["{bad-json", "[1, 2]"])
def test_unreadable_run_request_fails_closed(tmp_path: Path, payload: str) -> None:
    import sqlite3

    home = tmp_path / "home"
    registry = _registry(home)
    conn = sqlite3.connect(str(registry))
    try:
        conn.execute(
            "INSERT INTO runs (run_id, strategy_id, registered_at, request_json, managed_directory)"
            " VALUES (?, ?, ?, ?, ?)",
            ("d" * 32, "s", "2026-10-01T00:00:00+00:00", payload, None),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError, match="d" * 32):
        consulted_registry_horizon(registry)


def test_evaluation_events_and_journal_count(tmp_path: Path) -> None:
    history = tmp_path / "history"
    registry = tmp_path / "reg.jsonl"
    _append_event(
        registry,
        {
            "event": EVENT_EVALUATION,
            "procedure_digest": "b" * 32,
            "resolved_end": "2026-12-31T23:59:59+00:00",
            "at": "2027-01-05T00:00:00+00:00",
        },
    )
    journal = tmp_path / "journal.db"
    initialize_research_journal(
        journal, now=_utc("2027-03-01"), legacy_consulted_through=_utc("2027-03-31"),
        legacy_history_complete=False,
    )
    assert consulted_data_horizon(history, registry) == _utc("2026-12-31 23:59:59")
    assert consulted_data_horizon(history, registry, journal_path=journal) == _utc("2027-03-31")


def test_missing_supplied_journal_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError):
        consulted_data_horizon(tmp_path / "history", tmp_path / "reg.jsonl", journal_path=tmp_path / "missing.db")


def test_corrupt_registry_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _corrupt(home)
    procedure_registry = tmp_path / "reg.jsonl"
    with pytest.raises(DataIntegrityError):
        consulted_data_horizon(home, procedure_registry)
    with pytest.raises(DataIntegrityError):
        register_procedure(
            _request(), now=_utc("2027-01-01"), registry_path=procedure_registry, history_dir=home
        )
    assert not procedure_registry.exists()


def test_unparseable_history_end_fails_closed(tmp_path: Path) -> None:
    history = tmp_path / "history"
    append_run_history_record({"status": "COMPLETE", "resolved_end": "not-a-date"}, history)
    with pytest.raises(DataIntegrityError) as unparseable:
        consulted_data_horizon(history, tmp_path / "reg.jsonl")
    assert "not-a-date" in str(unparseable.value)
    assert "registry.sqlite3" in str(unparseable.value)


def test_unparseable_run_request_end_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    registry = _registry(home)
    _register_run(registry, {"start": "2021-01-01", "end": "whenever"}, run_id="c" * 32)
    with pytest.raises(DataIntegrityError, match="c" * 32):
        consulted_registry_horizon(registry)


def test_horizon_is_monotone_under_every_append(tmp_path: Path) -> None:
    """I-HORIZON-MONOTONE and I-SOURCE-COMPLETE over mixed evidence sources."""
    rng = np.random.default_rng(7)
    span = pd.date_range("2024-01-01", "2027-12-31", freq="D", tz="UTC")
    history = tmp_path / "history"
    procedure_registry = tmp_path / "reg.jsonl"
    home = tmp_path / "home"
    registry = _registry(home)
    ends: list[pd.Timestamp] = []
    observed: list[pd.Timestamp] = [
        consulted_data_horizon(history, procedure_registry),
        consulted_data_horizon(history, procedure_registry),
    ]
    for step in range(30):
        end = span[int(rng.integers(len(span)))]
        ends.append(end)
        if step % 3 == 0:
            append_run_history_record({"status": "COMPLETE", "resolved_end": end.isoformat()}, history)
        elif step % 3 == 1:
            _register_run(registry, {"start": "2021-01-01", "end": end.isoformat()})
        else:
            _append_event(
                procedure_registry,
                {"event": EVENT_EVALUATION, "procedure_digest": "d" * 32,
                 "resolved_end": end.isoformat(), "at": "2027-12-31T00:00:00+00:00"},
            )
        observed.append(consulted_data_horizon(history, procedure_registry))
    assert all(later >= earlier for earlier, later in itertools.pairwise(observed))
    assert observed[-1] == max([MHS_FINAL_OOS_CUTOFF_2026H1, *ends])


def test_perturbation_invariance_of_status_flags_and_snapshot(tmp_path: Path) -> None:
    """Only the dated end of a look can move the horizon (perturbation invariance)."""
    rng = np.random.default_rng(11)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for index in range(5):
        record = {
            "status": "COMPLETE",
            "resolved_end": f"2026-0{index + 1}-15T00:00:00+00:00",
            "flags": {"u": index},
            "params_snapshot": {},
            "blend": {"primary_naive_sharpe": 1.0},
        }
        append_run_history_record(record, first)
        append_run_history_record(
            {
                **record,
                "status": str(rng.choice(["COMPLETE", "FAILED", "RUNNING"])),
                "flags": {"u": int(rng.integers(5))},
                "params_snapshot": {"K": int(rng.integers(5))},
                "blend": {"primary_naive_sharpe": float(rng.choice([1.0, float("nan")]))},
            },
            second,
        )
    horizon = consulted_data_horizon(first, tmp_path / "reg.jsonl")
    assert consulted_data_horizon(second, tmp_path / "reg.jsonl") == horizon

    append_run_history_record({"status": "COMPLETE", "resolved_end": "2024-01-01"}, first)
    assert consulted_data_horizon(first, tmp_path / "reg.jsonl") == horizon


def test_legacy_jsonl_is_not_evidence_until_imported(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    (history / "active.jsonl").write_text(
        json.dumps({"status": "COMPLETE", "end": "2026-09-30"}) + "\n", encoding="utf-8"
    )
    assert consulted_data_horizon(history, tmp_path / "reg.jsonl") == MHS_FINAL_OOS_CUTOFF_2026H1
    migrate_legacy_backtests(
        registry_path=history / "registry.sqlite3", history_directories=(history,),
        run_directories=(), dry_run=False,
    )
    assert consulted_data_horizon(history, tmp_path / "reg.jsonl") == _utc("2026-09-30")


def test_paths_are_cwd_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.run_history as run_history

    home = tmp_path / "home"
    append_run_history_record({"status": "COMPLETE", "resolved_end": "2026-09-30T00:00:00+00:00"}, home)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_history, "canonical_history_registry", lambda: home / "registry.sqlite3")
    assert consulted_data_horizon(None, tmp_path / "reg.jsonl") == _utc("2026-09-30")
    assert PROCEDURE_REGISTRY_PATH.is_absolute()
    assert PROCEDURE_REGISTRY_PATH == BACKTESTS_DIR / "procedure_registry.jsonl"
    assert LEGACY_PROCEDURE_REGISTRY_PATH == BASE_DIR / "docs" / "decisions" / "mhs_procedure_registry.jsonl"
