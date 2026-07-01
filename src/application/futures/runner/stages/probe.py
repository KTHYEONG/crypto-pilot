from __future__ import annotations

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import MarketDataBundle, TimeframeProbeResult


def run_tf_probe_stage(
    run_config: FuturesRunConfig,
    data_bundle: MarketDataBundle,
) -> TimeframeProbeResult | None:
    raise NotImplementedError("run_tf_probe_stage: delegate to opt_main_futures probe logic")
