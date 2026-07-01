from __future__ import annotations

from typing import Any

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import (
    MarketDataBundle,
    RunnerResult,
    RunWindow,
    TimeframeProbeResult,
)


def run_strategy_stage(
    run_config: FuturesRunConfig,
    window: RunWindow,
    data_bundle: MarketDataBundle,
    universe_result: Any,
    *,
    layered_window: Any | None = None,
    probe_result: TimeframeProbeResult | None = None,
) -> Any | RunnerResult | None:
    raise NotImplementedError("run_strategy_stage: delegate to opt_main_futures strategy logic")
