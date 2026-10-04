"""One-time relocation of the procedure registry out of the documentation tree.

The registry is append-only runtime evidence, never documentation: it must be
moved byte-for-byte, never merged, duplicated or dropped, and a both-present
state must fail closed before any read or append.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs import preregistration as prereg
from src.mhs.preregistration import (
    EVENT_EVALUATION,
    EVENT_REGISTRATION,
    ProcedureRegistration,
    _append_event,
    load_registrations,
    migrate_legacy_procedure_registry,
    record_forward_evaluation,
    register_procedure,
)


def _registration_event(digest: str) -> dict[str, object]:
    return {
        "event": EVENT_REGISTRATION,
        "procedure_digest": digest,
        "frozen_at": "2026-09-17T00:00:00+00:00",
        "data_horizon": "2026-06-30T23:59:59+00:00",
        "procedure": {"k": digest[0]},
    }


def _evaluation_event(digest: str, resolved_end: str) -> dict[str, object]:
    return {
        "event": EVENT_EVALUATION,
        "procedure_digest": digest,
        "resolved_end": resolved_end,
        "at": "2027-01-05T00:00:00+00:00",
    }


def _request() -> Any:
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig

    return MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))


@pytest.fixture(autouse=True)
def registry_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    target = tmp_path / "backtests" / "procedure_registry.jsonl"
    legacy = tmp_path / "docs" / "decisions" / "mhs_procedure_registry.jsonl"
    monkeypatch.setattr(prereg, "PROCEDURE_REGISTRY_PATH", target)
    monkeypatch.setattr(prereg, "LEGACY_PROCEDURE_REGISTRY_PATH", legacy)
    return target, legacy


def _seed_legacy(legacy: Path) -> bytes:
    legacy.parent.mkdir(parents=True, exist_ok=True)
    for digest in ("a" * 32, "b" * 32):
        _append_event(legacy, _registration_event(digest))
    return legacy.read_bytes()


def test_legacy_registry_moves_once_byte_identically(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    payload = _seed_legacy(legacy)

    loaded = load_registrations()

    assert target.read_bytes() == payload
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    assert not legacy.exists()
    assert [r.procedure_digest for r in loaded] == ["a" * 32, "b" * 32]
    assert migrate_legacy_procedure_registry() is False


def test_both_registries_present_fails_closed(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    legacy_bytes = _seed_legacy(legacy)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_registration_event("c" * 32), sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    target_bytes = target.read_bytes()

    for action in (
        lambda: load_registrations(),
        lambda: migrate_legacy_procedure_registry(),
        lambda: register_procedure(
            _request(), now=pd.Timestamp("2026-10-01", tz="UTC"), history_dir=tmp_path / "history"
        ),
    ):
        with pytest.raises(DataIntegrityError) as ambiguous:
            action()
        assert str(legacy) in str(ambiguous.value)
        assert str(target) in str(ambiguous.value)

    assert legacy.read_bytes() == legacy_bytes
    assert target.read_bytes() == target_bytes


def test_append_path_migrates_before_writing(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    legacy_lines = _seed_legacy(legacy).decode("utf-8").splitlines(keepends=True)
    registration = ProcedureRegistration(
        "a" * 32, pd.Timestamp("2026-09-17", tz="UTC"), pd.Timestamp("2026-06-30 23:59:59", tz="UTC"), {}
    )

    record_forward_evaluation(
        registration, pd.Timestamp("2026-12-31", tz="UTC"),
        now=pd.Timestamp("2027-01-05", tz="UTC"),
    )

    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[: len(legacy_lines)] == legacy_lines
    assert len(lines) == len(legacy_lines) + 1
    assert '"event": "evaluation"' in lines[-1]
    assert not legacy.exists()


def test_migration_fails_closed_when_the_moved_bytes_differ(tmp_path: Path, registry_paths, monkeypatch) -> None:
    """A move that does not preserve the bytes is corruption, never a silent success."""
    target, legacy = registry_paths
    _seed_legacy(legacy)

    real_replace = os.replace

    def _corrupting_replace(source: Path, destination: Path) -> None:
        Path(source).write_text('{"event": "evaluation"}\n', encoding="utf-8")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", _corrupting_replace)
    with pytest.raises(DataIntegrityError, match="changed content"):
        migrate_legacy_procedure_registry()

    # The legacy evidence is retained and the target is not silently accepted.
    assert legacy.is_file()
    assert target.read_text(encoding="utf-8") == '{"event": "evaluation"}\n'


def test_no_legacy_file_is_a_no_op(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    assert migrate_legacy_procedure_registry() is False
    assert not legacy.exists()
    assert not target.exists()
    assert not target.parent.exists()
    assert load_registrations() == ()


def test_explicit_paths_never_migrate(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    payload = _seed_legacy(legacy)

    assert load_registrations(tmp_path / "other.jsonl") == ()

    assert legacy.read_bytes() == payload
    assert not target.exists()


def test_migration_logs_one_info_line(tmp_path: Path, registry_paths, caplog) -> None:
    _seed_legacy(registry_paths[1])
    with caplog.at_level(logging.INFO, logger="MhsPreregistration"):
        assert migrate_legacy_procedure_registry() is True
    assert sum(
        1 for record in caplog.records if "[DATA] procedure_registry migrated" in record.message
    ) == 1


def test_evaluation_events_survive_the_move(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    _seed_legacy(legacy)
    assert prereg.consulted_data_horizon(tmp_path / "history") == pd.Timestamp(
        "2026-06-30T23:59:59+00:00"
    )
    _append_event(target, _evaluation_event("a" * 32, "2026-12-31T23:59:59+00:00"))
    assert prereg.consulted_data_horizon(tmp_path / "history") == pd.Timestamp(
        "2026-12-31T23:59:59+00:00"
    )
