"""Invariant guards for process procedure registration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.journal import (
    consulted_process_horizon,
    initialize_research_journal,
    load_process_evaluation_plan,
    process_procedure_digest,
    record_research_consultation,
    reserve_research_attempt,
)
from src.mhs.preregistration import register_process_procedure

from tests.unit.mhs.test_research_journal import (
    E1,
    JUDGE,
    REG_NOW,
    _journal,
    _plan,
    _procedure,
)

INIT_NOW = pd.Timestamp("2026-08-01", tz="UTC")
LEGACY_OLD = pd.Timestamp("2024-01-01", tz="UTC")


def _bounds(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "hist", tmp_path / "reg.jsonl"


def test_registration_advances_boundary_without_unsealing_history(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    history, registry = _bounds(tmp_path)
    record_research_consultation(
        journal, procedure_digest=None, start=LEGACY_OLD, end=pd.Timestamp("2026-08-10", tz="UTC"),
        now=INIT_NOW, source="legacy",
    )
    plan = _plan()
    registered = register_process_procedure(plan, now=pd.Timestamp("2026-08-11", tz="UTC"), journal_path=journal,
                                            legacy_history_dir=history, legacy_registry_path=registry)
    assert registered.registration_digest is not None
    assert registered.judging_start is not None
    assert registered.judging_start > consulted_process_horizon(journal)
    assert load_process_evaluation_plan(journal) == registered
    journal2 = tmp_path / "j2.db"
    initialize_research_journal(journal2, now=INIT_NOW, legacy_consulted_through=LEGACY_OLD, legacy_history_complete=False)
    (tmp_path / "hist2").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hist2" / "active.jsonl").write_text(
        '{"resolved_end": "2026-08-20T00:00:00+00:00", "end": "2026-08-20T00:00:00+00:00"}\n', encoding="utf-8"
    )
    late = register_process_procedure(
        _plan(family_id="fam-late"), now=pd.Timestamp("2026-08-21", tz="UTC"), journal_path=journal2,
        legacy_history_dir=tmp_path / "hist2", legacy_registry_path=tmp_path / "reg2.jsonl",
    )
    assert late.registration_digest is not None
    historical = register_process_procedure(
        _plan(role="historical", judging_start=None, family_id="fam-hist"),
        now=pd.Timestamp("2026-08-21", tz="UTC"), journal_path=journal2,
        legacy_history_dir=tmp_path / "hist2", legacy_registry_path=tmp_path / "reg2.jsonl",
    )
    assert historical.registration_digest is not None


def test_frozen_runtime_mismatch_blocks_registration(tmp_path: Path) -> None:
    import dataclasses

    journal = _journal(tmp_path / "j.db")
    history, registry = _bounds(tmp_path)
    forged = dataclasses.replace(_plan(), procedure_digest="ff" * 32)
    with pytest.raises(DataIntegrityError):
        register_process_procedure(forged, now=REG_NOW, journal_path=journal,
                                   legacy_history_dir=history, legacy_registry_path=registry)
    with pytest.raises(ValueError, match="tz-aware"):
        register_process_procedure(_plan(), now=pd.Timestamp("2026-08-02"), journal_path=journal,
                                   legacy_history_dir=history, legacy_registry_path=registry)
    with pytest.raises(DataIntegrityError):
        register_process_procedure(_plan(family_id="fam-old"), now=pd.Timestamp("2026-06-30 23:59:59", tz="UTC"),
                                   journal_path=journal, legacy_history_dir=history, legacy_registry_path=registry)
    with pytest.raises(DataIntegrityError):
        register_process_procedure(
            _plan(family_id="fam-past", judging_start=pd.Timestamp("2026-06-30 23:59:59", tz="UTC")),
            now=REG_NOW, journal_path=journal, legacy_history_dir=history, legacy_registry_path=registry,
        )
    with pytest.raises(DataIntegrityError):
        register_process_procedure(_plan(family_id="fam-missing"), now=REG_NOW, journal_path=tmp_path / "missing.db",
                                   legacy_history_dir=history, legacy_registry_path=registry)
    assert process_procedure_digest(_procedure()) != process_procedure_digest(_procedure(member_ids=("ZZZ",)))


def test_training_may_precede_registration_while_judging_cannot(tmp_path: Path) -> None:
    journal = _journal(tmp_path / "j.db")
    history, registry = _bounds(tmp_path)
    plan = _plan()
    registered = register_process_procedure(plan, now=REG_NOW, journal_path=journal,
                                            legacy_history_dir=history, legacy_registry_path=registry)
    attempt = reserve_research_attempt(
        journal, registered, start=pd.Timestamp("2026-01-01", tz="UTC"), end=E1,
        now=REG_NOW, attempt_id="warm",
    )
    assert attempt.requested_start < REG_NOW
    assert attempt.context.interval_start == JUDGE
    record_research_consultation(
        journal, procedure_digest=registered.procedure_digest, start=JUDGE,
        end=pd.Timestamp("2026-09-15", tz="UTC"), now=REG_NOW, source="external",
    )
    assert consulted_process_horizon(journal) >= E1
    with pytest.raises(DataIntegrityError):
        register_process_procedure(
            _plan(family_id="fam-late", judging_start=pd.Timestamp("2026-09-10", tz="UTC")),
            now=pd.Timestamp("2026-10-01", tz="UTC"), journal_path=journal,
            legacy_history_dir=history, legacy_registry_path=registry,
        )
