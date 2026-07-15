from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput


class TieredHandoffError(RuntimeError):
    """Raised when Candidate output cannot be consumed without rebuilding."""


@dataclass(slots=True, frozen=True)
class TieredL1Handoff:
    aligned: AlignedMarketData
    aligned_by_tf: dict[str, AlignedMarketData] | None
    labeled_events_by_tf: dict[str, pd.DataFrame] | None
    labeled_events: pd.DataFrame
    l0_delivery_manifest: object | None


def consume_candidate_output_for_tiered(
    output: CandidatePipelineOutput,
    *,
    expected_symbols: Sequence[str],
) -> TieredL1Handoff:
    """Consume the candidate output without rebuilding aligned market data.

    [ADR_20260714_L1_MEMORY_EXECUTION]
    """
    if output.aligned is None:
        raise TieredHandoffError("output.aligned is None")

    actual = tuple(output.aligned.symbols)
    expected = tuple(expected_symbols)
    if actual != expected:
        raise TieredHandoffError(
            f"symbol order mismatch: got {actual}, expected {expected}"
        )

    aligned = output.aligned
    aligned_by_tf = output.aligned_by_tf
    labeled_events_by_tf = output.labeled_events_by_tf
    labeled_events = output.labeled_unfiltered if output.labeled_unfiltered is not None else output.labeled
    manifest = output.l0_delivery_manifest

    output.aligned = None
    output.aligned_by_tf = None
    output.labeled_events_by_tf = None
    output.labeled_unfiltered = None
    output.labeled = None
    output.alpha_panel = pd.DataFrame()
    output.fit_set = None
    output.calibration_set = None
    output.oos_set = None
    output.l0_delivery_manifest = None

    return TieredL1Handoff(
        aligned=aligned,
        aligned_by_tf=aligned_by_tf,
        labeled_events_by_tf=labeled_events_by_tf,
        labeled_events=labeled_events,
        l0_delivery_manifest=manifest,
    )
