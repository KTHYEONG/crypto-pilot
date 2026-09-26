"""Worker argument contract: the procedure identity is forwarded, never invented or malformed."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.application.mhs_worker as worker

_BUDGET = ["--total-tree-pss-bytes", "1", "--replay-tree-pss-bytes", "1", "--min-available-bytes", "1"]
_WINDOW = ["--start", "2026-01-01T00:00:00Z", "--end", "2026-02-01T00:00:00Z"]


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [*_WINDOW, "--result-output", str(tmp_path / "r.json"), *_BUDGET, *extra]


@pytest.mark.parametrize(
    "extra",
    [
        ("--procedure-code-digest", "ABC"),
        ("--procedure-code-digest", "a" * 64),
        ("--evidence-root", "e", "--registry-path", "r", "--run-id", "x"),
        ("--evidence-root", "e"),
    ],
    ids=["malformed_digest", "standalone_with_digest", "managed_without_digest", "partial_managed"],
)
def test_worker_rejects_invalid_procedure_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]) -> None:
    calls: list[object] = []
    monkeypatch.setattr(worker, "execute_mhs_backtest", calls.append)
    with pytest.raises(SystemExit, match="invalid worker arguments"):
        worker.main(_argv(tmp_path, *extra))
    assert calls == []


def test_worker_forwards_supervisor_digest_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[worker.MhsBacktestRequest] = []
    monkeypatch.setattr(worker, "execute_mhs_backtest", calls.append)
    digest = "0123456789abcdef" * 4
    rc = worker.main(_argv(tmp_path, "--evidence-root", "e", "--registry-path", "r", "--run-id", "x", "--procedure-code-digest", digest))
    assert rc == 0
    assert len(calls) == 1
    assert calls[0].procedure_code_digest == digest
