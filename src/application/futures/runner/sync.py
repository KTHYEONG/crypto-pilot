from __future__ import annotations

from src.application.futures.runner.config import FuturesRunConfig
from src.application.futures.runner.models import RunWindow


def ensure_universe_ledger_sync(run_config: FuturesRunConfig, window: RunWindow) -> None:
    raise NotImplementedError("ensure_universe_ledger_sync: delegate to opt_main_futures sync logic")


def ensure_cached_symbol_data(
    run_config: FuturesRunConfig,
    window: RunWindow,
    symbols: tuple[str, ...],
    *,
    require_exec_1m: bool,
) -> None:
    raise NotImplementedError("ensure_cached_symbol_data: delegate to opt_main_futures cache logic")
