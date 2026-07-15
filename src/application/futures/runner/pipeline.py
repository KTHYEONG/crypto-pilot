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
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Delegate runner orchestration to the active futures pipeline."""
    from src.application.futures.runner.active_pipeline import run_pipeline as run_active_pipeline

    _logger.info("Delegating to active futures pipeline: phase=%s timeframe=%s", run_config.phase, run_config.timeframe)
    return run_active_pipeline(cast(Any, run_config), seed=seed, resume=resume)
