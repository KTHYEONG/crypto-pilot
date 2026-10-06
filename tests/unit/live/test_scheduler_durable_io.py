"""Durability of persisted daemon state."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from src.live.scheduler import DaemonState, _load_daemon_state, _save_daemon_state


def test_save_daemon_state_fsyncs_file_and_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_fsync = os.fsync
    synced_modes: list[int] = []

    def _spy(fd: int) -> None:
        synced_modes.append(os.fstat(fd).st_mode)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    state_path = tmp_path / "daemon_state.json"
    state = DaemonState(
        last_processed_decision_time=pd.Timestamp("2026-08-24", tz="UTC"),
        pending_decision_time=pd.Timestamp("2026-08-25", tz="UTC"),
        attempts=2,
    )
    _save_daemon_state(state_path, state)
    assert any(stat.S_ISREG(mode) for mode in synced_modes)
    assert any(stat.S_ISDIR(mode) for mode in synced_modes)
    assert _load_daemon_state(state_path) == state
