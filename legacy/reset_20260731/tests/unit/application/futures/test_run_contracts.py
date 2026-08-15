from __future__ import annotations

import pytest

from src.application.futures.run_contracts import FuturesRunConfig
from src.domain.futures.alpha_foundry.contracts import AlphaFoundryRuntimeConfig


def test_default_alpha_foundry_is_off() -> None:
    cfg = FuturesRunConfig(
        timeframe="4h",
        date=None,
        trials=1,
        phase="l2",
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
    )
    assert cfg.l0_runtime.mode == "off"
    assert cfg.alpha_foundry is cfg.l0_runtime


def test_custom_l0_runtime() -> None:
    runtime = AlphaFoundryRuntimeConfig(mode="gate")
    cfg = FuturesRunConfig(
        timeframe="4h",
        date=None,
        trials=1,
        phase="l0",
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
        l0_runtime=runtime,
    )
    assert cfg.l0_runtime.mode == "gate"
    assert cfg.seed == 42


def test_frozen_cannot_be_mutated() -> None:
    cfg = FuturesRunConfig(
        timeframe="4h",
        date=None,
        trials=1,
        phase="l2",
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
    )
    with pytest.raises(AttributeError):
        cfg.timeframe = "1h"  # type: ignore[misc]
