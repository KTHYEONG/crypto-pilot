from __future__ import annotations

import argparse

from src.application.futures.runner.cli import (
    build_arg_parser as _build_arg_parser,
    run_from_cli as _run_from_cli,
)
from src.application.futures.runner.config import (
    FuturesRunConfig,
    build_run_config_from_args,
)
from src.application.futures.runner.models import RunnerResult
from src.application.futures.runner.pipeline import run_pipeline

__all__ = [
    "RunnerResult",
    "FuturesOptimizationRunner",
    "build_arg_parser",
    "build_config_from_namespace",
    "run_from_cli",
]


def build_arg_parser() -> argparse.ArgumentParser:
    return _build_arg_parser()


class FuturesOptimizationRunner:
    """Backward-compatible wrapper delegating to runner package."""

    def run(
        self,
        run_config: FuturesRunConfig,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> RunnerResult:
        result = run_pipeline(run_config, seed=seed, resume=resume)
        return RunnerResult(exit_code=result.exit_code, reason=result.reason)


def run_from_cli(argv: list[str] | None = None) -> int:
    return _run_from_cli(argv)


def build_config_from_namespace(args: argparse.Namespace) -> FuturesRunConfig:
    return build_run_config_from_args(args)
