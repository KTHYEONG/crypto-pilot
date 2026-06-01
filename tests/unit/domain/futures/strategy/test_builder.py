from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.domain.futures.strategy.builder import build_strategy_alpha
from src.domain.futures.strategy.config import StrategyConfig


def test_build_strategy_alpha_rejects_non_candidate_name() -> None:
    cfg = StrategyConfig(name="rule_baseline")
    illegal_cfg = object.__new__(StrategyConfig)
    object.__setattr__(illegal_cfg, "name", "momentum")
    object.__setattr__(illegal_cfg, "blend", cfg.blend)
    object.__setattr__(illegal_cfg, "regime", cfg.regime)
    object.__setattr__(illegal_cfg, "candidate", cfg.candidate)

    with pytest.raises(ValueError, match="unsupported active strategy name"):
        build_strategy_alpha({}, [], "4h", illegal_cfg)


def test_build_strategy_alpha_delegates_candidate_path(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = pd.DataFrame(
        {
            "alpha_long": [0.1],
            "alpha_short": [0.2],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-01 00:00:00"), "BTCUSDT")],
            names=["datetime", "symbol"],
        ),
    )

    class _Result:
        alpha_panel = expected

    captured: dict[str, Any] = {}

    def _fake_run_candidate_strategy_for_universe(**kwargs: Any) -> _Result:
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(
        "src.domain.futures.strategy_runtime.bridge.run_candidate_strategy_for_universe",
        _fake_run_candidate_strategy_for_universe,
    )

    maps = {"BTCUSDT": {"4h": pd.DataFrame({"datetime": [pd.Timestamp("2026-01-01")], "close": [1.0]})}}
    cfg = StrategyConfig(name="candidate_ml")
    out = build_strategy_alpha(maps, ["BTCUSDT"], "4h", cfg)

    assert out.equals(expected)
    assert captured["symbols"] == ["BTCUSDT"]
    assert captured["tf"] == "4h"
    assert captured["strategy_cfg"] == cfg
    assert captured["preloaded_data_maps"] is maps
