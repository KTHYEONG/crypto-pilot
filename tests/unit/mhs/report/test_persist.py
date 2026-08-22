"""src.mhs.report.persist: tier-dispatch contract.

Behavioral coverage of the compact tier itself (byte-identity, touch/ladder
stubbing) lives in tests/unit/mhs/test_report_persist_compact.py; this module
covers ``persist_mhs_report``'s own dispatch and path contract, which no
other test targets directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.mhs.report.persist as persist_mod
from src.application.research.mhs.contracts import MhsOutputTier
from src.mhs.report.persist import mhs_horizon_diagnostic_report_path, persist_mhs_report


def test_report_path_is_source_controlled() -> None:
    assert mhs_horizon_diagnostic_report_path() == str(
        Path("docs/results") / "mhs_horizon_diagnostic.json"
    )


def _patch_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(persist_mod, "build_mhs_run_history_record", lambda *a, **k: {})
    monkeypatch.setattr(persist_mod, "append_run_history_record", lambda *a, **k: None)


def test_persist_mhs_report_dispatches_compact_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_full",
        lambda report, target: calls.append("full") or target,
    )
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_compact",
        lambda report, target: calls.append("compact") or target,
    )
    _patch_history(monkeypatch)

    target = tmp_path / "report.json"
    result = persist_mhs_report(report=object(), target=target)  # type: ignore[arg-type]

    assert calls == ["compact"]
    assert result == target
    assert target.parent.exists()


def test_persist_mhs_report_dispatches_full_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_full",
        lambda report, target: calls.append("full") or target,
    )
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_compact",
        lambda report, target: calls.append("compact") or target,
    )
    _patch_history(monkeypatch)

    target = tmp_path / "report.json"
    persist_mhs_report(report=object(), target=target, tier=MhsOutputTier.FULL)  # type: ignore[arg-type]

    assert calls == ["full"]


def test_persist_mhs_report_swallows_run_history_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run-history append failure is observational and never breaks the
    returned persisted path."""
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_compact", lambda report, target: target,
    )
    monkeypatch.setattr(persist_mod, "build_mhs_run_history_record", lambda *a, **k: {})

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("history backend unavailable")

    monkeypatch.setattr(persist_mod, "append_run_history_record", _boom)

    target = tmp_path / "report.json"
    result = persist_mhs_report(report=object(), target=target)  # type: ignore[arg-type]

    assert result == target
