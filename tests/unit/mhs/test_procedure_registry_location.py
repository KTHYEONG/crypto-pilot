"""Registry destination ownership: explicit migration, fail-closed reads.

The procedure registry is append-only runtime evidence. Migration out of the
documentation tree is an explicit operator action only; reads fail closed
while the legacy file still exists and never trigger a migration.
"""

from __future__ import annotations

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

    return MhsDiagnosticRequest()


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


def test_reads_fail_closed_on_unmigrated_legacy_without_migrating(
    tmp_path: Path, registry_paths
) -> None:
    target, legacy = registry_paths
    payload = _seed_legacy(legacy)

    for action in (
        lambda: load_registrations(),
        lambda: load_registrations(target),
        lambda: prereg.find_registration("a" * 32),
        lambda: prereg.consulted_data_horizon(tmp_path / "history"),
    ):
        with pytest.raises(DataIntegrityError) as excinfo:
            action()
        message = str(excinfo.value)
        assert str(legacy) in message
        assert str(target) in message
        assert "ops procedure-registry-migrate" in message

    assert legacy.read_bytes() == payload
    assert not target.exists()


def test_append_fails_closed_on_unmigrated_legacy(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    payload = _seed_legacy(legacy)
    registration = ProcedureRegistration(
        "a" * 32, pd.Timestamp("2026-09-17", tz="UTC"), pd.Timestamp("2026-06-30 23:59:59", tz="UTC"), {}
    )

    with pytest.raises(DataIntegrityError):
        record_forward_evaluation(
            registration, pd.Timestamp("2026-12-31", tz="UTC"),
            now=pd.Timestamp("2027-01-05", tz="UTC"), registry_path=target,
        )

    assert legacy.read_bytes() == payload
    assert not target.exists()


def test_explicit_paths_never_migrate(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    payload = _seed_legacy(legacy)

    assert load_registrations(tmp_path / "other.jsonl") == ()

    assert legacy.read_bytes() == payload
    assert not target.exists()


def test_explicit_migration_moves_once_byte_identically(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    payload = _seed_legacy(legacy)

    assert migrate_legacy_procedure_registry(legacy_path=legacy, target_path=target) is True
    assert target.read_bytes() == payload
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    assert not legacy.exists()
    assert [r.procedure_digest for r in load_registrations()] == ["a" * 32, "b" * 32]
    assert migrate_legacy_procedure_registry(legacy_path=legacy, target_path=target) is False


def test_migration_requires_both_paths(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    _seed_legacy(legacy)

    with pytest.raises(TypeError):
        migrate_legacy_procedure_registry()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        migrate_legacy_procedure_registry(legacy_path=legacy)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        migrate_legacy_procedure_registry(legacy_path=None, target_path=target)  # type: ignore[arg-type]

    assert legacy.is_file()
    assert not target.exists()


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
        lambda: migrate_legacy_procedure_registry(legacy_path=legacy, target_path=target),
        lambda: register_procedure(
            _request(), now=pd.Timestamp("2026-10-01", tz="UTC"),
            registry_path=target, history_dir=tmp_path / "history",
        ),
    ):
        with pytest.raises(DataIntegrityError) as ambiguous:
            action()
        assert str(legacy) in str(ambiguous.value)

    assert legacy.read_bytes() == legacy_bytes
    assert target.read_bytes() == target_bytes


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
        migrate_legacy_procedure_registry(legacy_path=legacy, target_path=target)

    # The legacy evidence is retained and the target is not silently accepted.
    assert legacy.is_file()
    assert target.read_text(encoding="utf-8") == '{"event": "evaluation"}\n'


def test_no_legacy_file_is_a_no_op(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    assert migrate_legacy_procedure_registry(legacy_path=legacy, target_path=target) is False
    assert not legacy.exists()
    assert not target.exists()
    assert not target.parent.exists()
    assert load_registrations() == ()


def test_migration_logs_one_info_line(tmp_path: Path, registry_paths, caplog) -> None:
    _seed_legacy(registry_paths[1])
    with caplog.at_level(logging.INFO, logger="MhsPreregistration"):
        assert migrate_legacy_procedure_registry(legacy_path=registry_paths[1], target_path=registry_paths[0]) is True
    assert sum(
        1 for record in caplog.records if "[DATA] procedure_registry migrated" in record.message
    ) == 1


def test_evaluation_events_survive_the_move(tmp_path: Path, registry_paths) -> None:
    target, legacy = registry_paths
    _seed_legacy(legacy)
    assert migrate_legacy_procedure_registry(legacy_path=legacy, target_path=target) is True
    assert prereg.consulted_data_horizon(tmp_path / "history") == pd.Timestamp(
        "2026-06-30T23:59:59+00:00"
    )
    _append_event(target, _evaluation_event("a" * 32, "2026-12-31T23:59:59+00:00"))
    assert prereg.consulted_data_horizon(tmp_path / "history") == pd.Timestamp(
        "2026-12-31T23:59:59+00:00"
    )


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{not json", "line 1 is not valid JSON"),
        (json.dumps({"event": "evaluation", "procedure_hash": "a" * 32}), "line 1 has no valid resolved_end"),
        (json.dumps({"event": "evaluation", "resolved_end": "not-a-date"}), "line 1 has no valid resolved_end"),
    ],
)
def test_corrupt_procedure_events_fail_closed_as_integrity_errors(tmp_path: Path, line: str, message: str) -> None:
    registry = tmp_path / "procedures.jsonl"
    registry.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=message):
        prereg.consulted_data_horizon(tmp_path / "history", registry)
