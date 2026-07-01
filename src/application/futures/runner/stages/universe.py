from __future__ import annotations

from typing import Any

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import RunWindow


def run_universe_stage(
    run_config: FuturesRunConfig,
    window: RunWindow,
    *,
    layered_window: Any | None = None,
) -> Any:
    raise NotImplementedError("run_universe_stage: delegate to opt_main_futures universe logic")
