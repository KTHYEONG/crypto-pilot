"""Guards for the dedicated market-recorder process entrypoint."""

from __future__ import annotations

import contextlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from src.market_data.streams import recorder_main


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    async def _fake_run(config, *, capture_root, liquidations_dir, shutdown):
        seen.update(capture_root=capture_root, liquidations_dir=liquidations_dir, shutdown=shutdown)

    monkeypatch.setattr(recorder_main, "run_market_recorder", _fake_run)
    monkeypatch.setattr(recorder_main, "install_shutdown_handlers", lambda flag: seen.setdefault("installed", flag))
    return seen


def _hermetic_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    target = tmp_path / "logs-default"
    monkeypatch.setattr(recorder_main, "RECORDER_LOG_DIR", target)
    return target


def _remove_file_handlers(log_file: Path) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, RotatingFileHandler):
            try:
                if Path(str(handler.baseFilename)).resolve() == log_file.resolve():
                    root.removeHandler(handler)
                    handler.close()
            except OSError:
                continue


def test_run_recorder_uses_defaults_and_installs_shutdown_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    recorder_main.run_recorder()
    assert seen["capture_root"] == recorder_main.LIVE_CAPTURE_DIR
    assert seen["liquidations_dir"] == recorder_main.default_liquidations_dir()
    assert seen["installed"] is seen["shutdown"]


def test_main_passes_explicit_roots_and_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture(monkeypatch)
    default_dir = _hermetic_log_dir(monkeypatch, tmp_path)
    log_dir = tmp_path / "x"
    rc = recorder_main.main(
        ["--capture-root", str(tmp_path / "c"), "--liquidations-dir", str(tmp_path / "l"), "--log-dir", str(log_dir)]
    )
    try:
        assert rc == 0
        assert seen["capture_root"] == tmp_path / "c"
        assert seen["liquidations_dir"] == tmp_path / "l"
        assert (log_dir / "recorder.log").exists()
        assert not default_dir.exists()
    finally:
        _remove_file_handlers(log_dir / "recorder.log")
        _remove_file_handlers(default_dir / "recorder.log")


def test_main_without_args_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture(monkeypatch)
    default_dir = _hermetic_log_dir(monkeypatch, tmp_path)
    try:
        assert recorder_main.main([]) == 0
        assert seen["capture_root"] == recorder_main.LIVE_CAPTURE_DIR
        assert (default_dir / "recorder.log").exists()
    finally:
        _remove_file_handlers(default_dir / "recorder.log")


def test_configure_recorder_logging_writes_file_log(tmp_path: Path) -> None:
    log_file = tmp_path / "recorder.log"
    try:
        result = recorder_main.configure_recorder_logging(tmp_path)
        assert result == log_file
        logging.getLogger().info("recorder-file-log-probe")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "recorder-file-log-probe" in log_file.read_text(encoding="utf-8")
    finally:
        _remove_file_handlers(log_file)


def test_configure_recorder_logging_attaches_stdout_handler_when_missing(tmp_path: Path) -> None:
    root = logging.getLogger()
    saved = list(root.handlers)
    for handler in saved:
        root.removeHandler(handler)
    log_file = tmp_path / "recorder.log"
    try:
        result = recorder_main.configure_recorder_logging(tmp_path)
        assert result == log_file
        assert any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
            for h in root.handlers
        )
        assert log_file.exists()
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()
        for handler in saved:
            root.addHandler(handler)


def test_configure_recorder_logging_is_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "recorder.log"
    try:
        recorder_main.configure_recorder_logging(tmp_path)
        recorder_main.configure_recorder_logging(tmp_path)
        matches = [
            h
            for h in logging.getLogger().handlers
            if isinstance(h, RotatingFileHandler)
            and Path(str(h.baseFilename)).resolve() == log_file.resolve()
        ]
        assert len(matches) == 1
    finally:
        _remove_file_handlers(log_file)


def test_configure_recorder_logging_falls_back_to_stdout_when_unwritable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-dir", encoding="utf-8")
    bad_dir = blocker / "logs"
    with caplog.at_level(logging.WARNING):
        result = recorder_main.configure_recorder_logging(bad_dir)
    assert result is None
    assert any("FILE_LOG_UNAVAILABLE" in rec.message for rec in caplog.records)


def test_main_honours_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture(monkeypatch)
    default_dir = tmp_path / "default"
    monkeypatch.setattr(recorder_main, "RECORDER_LOG_DIR", default_dir)
    custom = tmp_path / "x"
    try:
        assert recorder_main.main(["--log-dir", str(custom)]) == 0
        assert seen["capture_root"] == recorder_main.LIVE_CAPTURE_DIR
        assert (custom / "recorder.log").exists()
        assert not default_dir.exists()
    finally:
        _remove_file_handlers(custom / "recorder.log")
