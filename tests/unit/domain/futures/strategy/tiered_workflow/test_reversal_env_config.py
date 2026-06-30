"""Spec: futures-l2-reversal-economic-replay, Scenario 1-2."""
from __future__ import annotations

import os

import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import _reversal_config_from_env


class TestReversalConfigFromEnv:
    def test_env_overrides_threshold_and_persistence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L2_REVERSAL_DD_THRESHOLD", "0.10")
        monkeypatch.setenv("L2_REVERSAL_PERSISTENCE_BARS", "2")
        cfg = _reversal_config_from_env()
        assert cfg.reversal_dd_threshold == 0.10
        assert cfg.reversal_persistence_bars == 2

    def test_reversal_config_from_env_rejects_invalid_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L2_REVERSAL_DD_THRESHOLD", "1.5")
        with pytest.raises(ValueError, match="reversal_dd_threshold"):
            _reversal_config_from_env()

    def test_reversal_config_from_env_returns_defaults_when_no_env(self) -> None:
        for key in ("L2_REVERSAL_DD_WINDOW", "L2_REVERSAL_DD_THRESHOLD", "L2_REVERSAL_MOM_FAST",
                     "L2_REVERSAL_MOM_SLOW", "L2_REVERSAL_RISK_OFF_FLOOR", "L2_REVERSAL_PERSISTENCE_BARS"):
            os.environ.pop(key, None)
        cfg = _reversal_config_from_env()
        assert cfg.reversal_dd_window == 90
        assert cfg.reversal_dd_threshold == 0.12
        assert cfg.reversal_mom_fast == 20
        assert cfg.reversal_mom_slow == 120
        assert cfg.reversal_risk_off_floor == 0.05
        assert cfg.reversal_persistence_bars == 3
