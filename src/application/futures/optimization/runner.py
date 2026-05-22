from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.application.futures.optimization.config import (
    FuturesRunConfig,
    build_run_config_from_args,
)
from src.execution import opt_main_futures


@dataclass(slots=True, frozen=True)
class RunnerResult:
    """Backward-compatible runner completion status."""

    exit_code: int
    reason: str


def build_arg_parser() -> argparse.ArgumentParser:
    """Return active futures CLI parser."""
    return opt_main_futures.build_arg_parser()


class FuturesOptimizationRunner:
    """Backward-compatible wrapper delegating to execution orchestrator."""

    def run(
        self,
        run_config: FuturesRunConfig,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> RunnerResult:
        """Execute active pipeline through execution-layer orchestrator."""
        result = opt_main_futures.run_pipeline(run_config, seed=seed, resume=resume)
        return RunnerResult(exit_code=result.exit_code, reason=result.reason)


def run_from_cli(argv: list[str] | None = None) -> int:
    """Run CLI via execution orchestrator."""
    return opt_main_futures.run_from_cli(argv)


def build_config_from_namespace(args: argparse.Namespace) -> FuturesRunConfig:
    """Build validated run config from argparse namespace."""
    payload = vars(args).copy()
    if bool(payload.get("quick_backtest", False)):
        payload["mode"] = "quick-backtest"
    symbols_raw = payload.get("symbols")
    if symbols_raw:
        payload["symbols"] = tuple(str(symbols_raw).split(","))
    return build_run_config_from_args(payload)
