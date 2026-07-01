from __future__ import annotations

from typing import Any

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import MarketDataBundle


def run_regime_stage(
    run_config: FuturesRunConfig,
    data_bundle: MarketDataBundle,
) -> tuple[Any, Any] | None:
    raise NotImplementedError("run_regime_stage: delegate to opt_main_futures regime logic")
