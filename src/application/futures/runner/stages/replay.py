from __future__ import annotations

from typing import Any

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import RunWindow


def run_replay_stage(
    run_config: FuturesRunConfig,
    window: RunWindow,
    data_bundle: Any,
    *,
    seed: int,
) -> Any:
    raise NotImplementedError("run_replay_stage: delegate to opt_main_futures replay logic")
