from __future__ import annotations

import logging
from typing import Any, cast

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import RunnerResult

_logger = logging.getLogger(__name__)


def run_pipeline(
    run_config: FuturesRunConfig,
    *,
    seed: int = 42,
    resume: bool = False,
) -> RunnerResult:
    from src.application.futures.runner.active_pipeline import run_pipeline as run_active_pipeline

    _logger.info("Delegating to active futures pipeline: phase=%s timeframe=%s", run_config.phase, run_config.timeframe)
    result = run_active_pipeline(cast(Any, run_config), seed=seed, resume=resume)
    return RunnerResult(exit_code=result.exit_code, reason=result.reason)
