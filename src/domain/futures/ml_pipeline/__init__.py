"""ML alpha pipeline bridge for legacy implementation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "MLPipelineOutput",
    "merge_ml_output_into_is_and_oos",
    "run_ml_pipeline_for_universe",
]


def __getattr__(name: str) -> Any:
    if name == "run_ml_pipeline_for_universe":
        from src.domain.futures.strategy_runtime.bridge import run_ml_pipeline_for_universe as _run
        return _run
    if name == "MLPipelineOutput":
        from src.domain.futures.strategy_runtime.bridge import MLPipelineOutput as _MLPipelineOutput

        return _MLPipelineOutput
    if name == "merge_ml_output_into_is_and_oos":
        from src.domain.futures.strategy_runtime.bridge import (
            merge_ml_output_into_is_and_oos as _merge,
        )

        return _merge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
