from __future__ import annotations

from pathlib import Path

import pytest

from src.application.futures.runner.legacy_retirement import (
    LegacyRetirementReport,
    retire_legacy_storage,
)


def test_retirement_blocks_delete_before_audit(tmp_path: Path) -> None:
    target = tmp_path / "legacy"
    target.mkdir()
    report = LegacyRetirementReport(True, False, True, (), (target,))
    with pytest.raises(RuntimeError, match="retirement preflight"):
        retire_legacy_storage(report=report, approved=True)
    assert target.exists()


def test_retirement_deletes_exact_approved_targets(tmp_path: Path) -> None:
    target = tmp_path / "legacy"
    target.mkdir()
    report = LegacyRetirementReport(True, True, True, (), (target,))
    assert retire_legacy_storage(report=report, approved=True) == (target,)
    assert not target.exists()
