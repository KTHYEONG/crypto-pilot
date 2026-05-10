"""ML alpha pipeline (Phase C) for futures: GP, HMM, TBM, meta-labeler.

Public surface: pipeline entrypoint and core model classes only. Import subpackages
(e.g. ``ml_pipeline.features.engineering``) for builders and helpers.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HMMStateInferrer",
    "MLAlphaMiner",
    "MetaLabeler",
    "run_ml_pipeline_for_universe",
]


def __getattr__(name: str) -> Any:
    """Lazy import of ``run_ml_pipeline_for_universe`` to avoid circular imports with optimizer."""
    if name == "run_ml_pipeline_for_universe":
        from src.domain.futures.ml_pipeline.pipeline_runner import (
            run_ml_pipeline_for_universe as _run,
        )

        return _run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
