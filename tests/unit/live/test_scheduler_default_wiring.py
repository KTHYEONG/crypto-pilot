"""Production step wiring must accept the single positional decision time run_daemon passes."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.live import scheduler
from src.live.settings import LiveSettings


@pytest.mark.parametrize("name", ["signal", "refresh", "venue", "prefetch"])
def test_default_step_fns_bind_decision_time_positionally(name: str, tmp_path: Path) -> None:
    fns = scheduler.default_step_fns(LiveSettings(), tmp_path / "w.parquet")
    bound = inspect.signature(fns[name]).bind(pd.Timestamp("2026-09-26", tz="UTC"))
    assert len(bound.args) == 1


def test_default_refresh_receives_decision_time_in_its_own_parameter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake(settings: LiveSettings, weights_path: Path, decision_time: pd.Timestamp) -> str:
        seen.update(settings=settings, weights_path=weights_path, decision_time=decision_time)
        return "ok"

    monkeypatch.setattr(scheduler, "_default_data_refresh", fake)
    monkeypatch.setattr(scheduler, "_default_funding_prefetch", fake)
    settings = LiveSettings()
    target = pd.Timestamp("2026-09-26", tz="UTC")
    for name in ("refresh", "prefetch"):
        seen.clear()
        assert scheduler.default_step_fns(settings, tmp_path / "w.parquet")[name](target) == "ok"
        assert seen["settings"] is settings
        assert seen["decision_time"] == target
