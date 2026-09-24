"""Guards for the dedicated market-recorder process entrypoint."""

from __future__ import annotations

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


def test_run_recorder_uses_defaults_and_installs_shutdown_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    recorder_main.run_recorder()
    assert seen["capture_root"] == recorder_main.LIVE_CAPTURE_DIR
    assert seen["liquidations_dir"] == recorder_main.default_liquidations_dir()
    assert seen["installed"] is seen["shutdown"]


def test_main_passes_explicit_roots_and_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture(monkeypatch)
    rc = recorder_main.main(["--capture-root", str(tmp_path / "c"), "--liquidations-dir", str(tmp_path / "l")])
    assert rc == 0
    assert seen["capture_root"] == tmp_path / "c"
    assert seen["liquidations_dir"] == tmp_path / "l"


def test_main_without_args_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    assert recorder_main.main([]) == 0
    assert seen["capture_root"] == recorder_main.LIVE_CAPTURE_DIR
