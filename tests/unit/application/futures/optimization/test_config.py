from __future__ import annotations

from src.application.futures.optimization.config import FuturesRunConfig
from src.application.futures.run_contracts import FuturesRunConfig as CanonicalFuturesRunConfig


def test_optimization_config_reexports_canonical_run_config() -> None:
    assert FuturesRunConfig is CanonicalFuturesRunConfig


def test_optimization_config_run_config_exposes_l0_runtime() -> None:
    config = FuturesRunConfig(
        timeframe="4h",
        date="2026-05-01",
        trials=1,
        phase="l1",
        sync="skip",
        refresh_universe=False,
        sync_metrics=False,
    )

    assert config.alpha_foundry is config.l0_runtime
