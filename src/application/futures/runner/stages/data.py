from __future__ import annotations

from typing import Any

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import MarketDataBundle, RunWindow


def run_data_stage(
    run_config: FuturesRunConfig,
    window: RunWindow,
    universe_result: Any,
    *,
    layered_window: Any | None = None,
) -> MarketDataBundle:
    raise NotImplementedError("run_data_stage: delegate to opt_main_futures data logic")
