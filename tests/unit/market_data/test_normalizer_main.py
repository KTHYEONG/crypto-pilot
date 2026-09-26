"""Invariant guards for the normalizer process entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.paths import LIVE_CAPTURE_DIR
from src.market_data.streams.normalizer_main import BACKUP_STATUS_PATH, main


def test_main_runs_until_shutdown_with_defaults(tmp_path: Path, monkeypatch: Any) -> None:
    """Patched run_normalizer receives production defaults and main returns 0."""
    import src.market_data.streams.normalizer_main as module

    seen: dict[str, Any] = {}

    def _fake_run_normalizer(*args: Any, **kwargs: Any) -> None:
        seen["args"] = args
        seen.update(kwargs)

    monkeypatch.setattr(module, "configure_normalizer_logging", lambda *a, **k: None)
    monkeypatch.setattr(module, "run_normalizer", _fake_run_normalizer)
    assert main([]) == 0
    assert seen["args"][0] == LIVE_CAPTURE_DIR
    assert seen["backup_status_path"] == BACKUP_STATUS_PATH


def test_log_directory_failure_never_blocks(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    """An unwritable --log-dir falls back to stdout-only with a warning."""
    import logging

    import src.market_data.streams.normalizer_main as module

    calls: list[str] = []

    def _fake_run_normalizer(*args: Any, **kwargs: Any) -> None:
        calls.append("ran")

    monkeypatch.setattr(module, "run_normalizer", _fake_run_normalizer)
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-directory")
    with caplog.at_level(logging.WARNING):
        assert main(["--capture-root", str(tmp_path), "--log-dir", str(blocker / "child")]) == 0
    assert calls == ["ran"]
    assert "FILE_LOG_UNAVAILABLE" in caplog.text


def test_logging_is_idempotent(tmp_path) -> None:
    """A second call with the same directory reuses the file handler."""
    import logging

    from src.market_data.streams.normalizer_main import configure_normalizer_logging

    first = configure_normalizer_logging(tmp_path / "logs")
    second = configure_normalizer_logging(tmp_path / "logs")
    assert first == second
    root = logging.getLogger()
    assert sum(getattr(h, "baseFilename", "") == str(first) for h in root.handlers) == 1


def test_process_entry_wires_defaults(tmp_path, monkeypatch) -> None:
    """run_normalizer_process wires defaults and the shared clock."""
    import src.market_data.streams.normalizer_main as module

    seen: dict = {}

    def _fake_run(*args, **kwargs) -> None:
        seen["args"] = args
        seen.update(kwargs)

    monkeypatch.setattr(module, "run_normalizer", _fake_run)
    module.run_normalizer_process(capture_root=tmp_path)
    assert seen["args"][0] == tmp_path
    assert seen["backup_status_path"] == module.BACKUP_STATUS_PATH


def test_logging_without_handlers_adds_stdout(tmp_path) -> None:
    """A bare root logger gains a stdout handler plus the rotating file."""
    import logging

    from src.market_data.streams.normalizer_main import configure_normalizer_logging

    root = logging.getLogger()
    saved = root.handlers[:]
    for handler in saved:
        root.removeHandler(handler)
    try:
        path = configure_normalizer_logging(tmp_path / "logs")
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
    assert path is not None
    assert path.exists()
