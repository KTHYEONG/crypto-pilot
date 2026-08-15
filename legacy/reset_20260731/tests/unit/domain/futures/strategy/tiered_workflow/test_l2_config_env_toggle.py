from __future__ import annotations

from typing import Any

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig


def test_from_mapping_env_enables_diag_attribution(monkeypatch: Any) -> None:
    monkeypatch.setenv("L2_DIAG_ATTR", "1")
    cfg = Layer2AllocationConfig.from_mapping({})
    assert cfg.l2_diag_attribution_enabled is True


def test_from_mapping_env_disabled_by_default() -> None:
    cfg = Layer2AllocationConfig.from_mapping({})
    assert cfg.l2_diag_attribution_enabled is False
