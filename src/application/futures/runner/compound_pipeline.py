"""Small compatibility adapter for direct compound-engine integration tests."""

from __future__ import annotations

from typing import Any, Literal, cast

from src.domain.futures.compound.contracts import CompoundPipelineOutcome


def run_compound_pipeline(*, aligned: Any, universe: Any, settings: Any) -> CompoundPipelineOutcome:
    """Return the explicit deployment mode without reintroducing legacy routing."""
    mode = str(getattr(settings, "mode", "shadow"))
    if mode not in {"legacy", "shadow", "active"}:
        raise ValueError(f"invalid compound pipeline mode: {mode}")
    return CompoundPipelineOutcome(
        mode=cast(Literal["legacy", "shadow", "active"], mode),
        engine_result=None,
        order_routed=False,
        reason="direct_compound_only",
    )


__all__ = ["CompoundPipelineOutcome", "run_compound_pipeline"]
