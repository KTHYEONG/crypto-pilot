from __future__ import annotations

from typing import Any

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import RunnerResult, RunWindow


def run_optimization_stage(
    run_config: FuturesRunConfig,
    window: RunWindow,
    data_bundle: Any,
    *,
    seed: int,
    resume: bool,
) -> RunnerResult:
    raise NotImplementedError("run_optimization_stage: delegate to opt_main_futures optimization logic")
