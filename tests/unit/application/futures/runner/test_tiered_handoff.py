from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.application.futures.runner.tiered_handoff import (
    TieredHandoffError,
    TieredL1Handoff,
    consume_candidate_output_for_tiered,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput

# AlignedMarketData is a concrete class, but for identity-check tests we can use MagicMock
# that responds to isinstance checks properly if needed.
# The handoff contract only requires reference-preservation, not isinstance checking.


def test_tiered_handoff_moves_without_copy() -> None:
    aligned = MagicMock()
    aligned.symbols = ("BTCUSDT", "ETHUSDT")
    labeled = pd.DataFrame({"native_tf": ["4h"], "l0_recipe_id": ["r1"], "realized_return": [0.01]})
    output = CandidatePipelineOutput(
        alpha_panel=pd.DataFrame({"BTCUSDT": [1.0]}),
        aligned=aligned,
        labeled=labeled.copy(),
        labeled_unfiltered=labeled,
        fit_set=object(),
        calibration_set=object(),
        oos_set=object(),
        l0_delivery_manifest=MagicMock(),
    )

    handoff = consume_candidate_output_for_tiered(
        output,
        expected_symbols=("BTCUSDT", "ETHUSDT"),
    )

    assert handoff.aligned is aligned
    assert handoff.labeled_events is labeled
    assert output.aligned is None
    assert output.labeled_unfiltered is None
    assert output.alpha_panel.empty
    assert output.fit_set is None


def test_tiered_handoff_rejects_symbol_reorder() -> None:
    aligned = MagicMock()
    aligned.symbols = ("ETHUSDT", "BTCUSDT")
    output = CandidatePipelineOutput(
        aligned=aligned,
        labeled_unfiltered=pd.DataFrame({"native_tf": ["4h"]}),
    )

    with pytest.raises(TieredHandoffError, match="symbol order"):
        consume_candidate_output_for_tiered(
            output,
            expected_symbols=("BTCUSDT", "ETHUSDT"),
        )


def test_tiered_handoff_raises_when_aligned_none() -> None:
    output = CandidatePipelineOutput()

    with pytest.raises(TieredHandoffError, match="aligned is None"):
        consume_candidate_output_for_tiered(
            output,
            expected_symbols=("BTCUSDT",),
        )


def test_consume_candidate_output_threads_aligned_by_tf() -> None:
    a2h = MagicMock(spec=["symbols"])
    a2h.symbols = ("BTCUSDT",)
    a4h = MagicMock(spec=["symbols"])
    a4h.symbols = ("BTCUSDT",)
    labeled = pd.DataFrame({"native_tf": ["4h"], "l0_recipe_id": ["r1"], "realized_return": [0.01]})
    output = CandidatePipelineOutput(
        alpha_panel=pd.DataFrame({"BTCUSDT": [1.0]}),
        aligned=a4h,
        aligned_by_tf={"2h": a2h, "4h": a4h},
        labeled=labeled,
        labeled_unfiltered=labeled,
        fit_set=object(),
        calibration_set=object(),
        oos_set=object(),
        l0_delivery_manifest=MagicMock(),
    )

    handoff = consume_candidate_output_for_tiered(output, expected_symbols=("BTCUSDT",))

    assert handoff.aligned_by_tf is not None
    assert handoff.aligned_by_tf["2h"] is a2h
    assert handoff.aligned_by_tf["4h"] is a4h
    assert output.aligned_by_tf is None  # destructive handoff pattern


def test_consume_candidate_output_handles_missing_aligned_by_tf() -> None:
    aligned = MagicMock(spec=["symbols"])
    aligned.symbols = ("BTCUSDT",)
    labeled = pd.DataFrame({"native_tf": ["4h"]})
    output = CandidatePipelineOutput(
        alpha_panel=pd.DataFrame({"BTCUSDT": [1.0]}),
        aligned=aligned,
        aligned_by_tf=None,
        labeled=labeled,
        labeled_unfiltered=labeled,
        l0_delivery_manifest=MagicMock(),
    )

    handoff = consume_candidate_output_for_tiered(output, expected_symbols=("BTCUSDT",))

    assert handoff.aligned_by_tf is None


def test_consume_output_aligned_by_tf_none_does_not_alter_error_path() -> None:
    output = CandidatePipelineOutput(aligned_by_tf=None)

    with pytest.raises(TieredHandoffError, match="aligned is None"):
        consume_candidate_output_for_tiered(output, expected_symbols=("BTCUSDT",))


def test_resolve_aligned_for_tf_per_tf_grids_are_distinct() -> None:
    from src.domain.futures.strategy.tiered_workflow.pipeline import _resolve_aligned_for_tf

    base = cast(AlignedMarketData, MagicMock(spec=AlignedMarketData, datetimes=list(range(6949))))
    per_tf: dict[str, AlignedMarketData] = {
        "2h": cast(AlignedMarketData, MagicMock(spec=AlignedMarketData, datetimes=list(range(11736)))),
        "1d": cast(AlignedMarketData, MagicMock(spec=AlignedMarketData, datetimes=list(range(978)))),
    }

    resolved_2h = _resolve_aligned_for_tf("2h", base, per_tf)
    resolved_1d = _resolve_aligned_for_tf("1d", base, per_tf)
    resolved_6h = _resolve_aligned_for_tf("6h", base, per_tf)

    assert len(resolved_2h.datetimes) == 11736
    assert len(resolved_1d.datetimes) == 978
    assert resolved_6h is base


def test_tiered_handoff_uses_labeled_as_fallback() -> None:
    aligned = MagicMock()
    aligned.symbols = ("BTCUSDT",)
    labeled = pd.DataFrame({"native_tf": ["4h"]})
    output = CandidatePipelineOutput(
        aligned=aligned,
        labeled=labeled,
        labeled_unfiltered=None,
    )

    handoff = consume_candidate_output_for_tiered(
        output,
        expected_symbols=("BTCUSDT",),
    )

    assert handoff.labeled_events is labeled


def test_tiered_l1_handoff_dataclass() -> None:
    aligned = MagicMock()
    labeled = pd.DataFrame()
    handoff = TieredL1Handoff(
        aligned=aligned,
        aligned_by_tf=None,
        labeled_events_by_tf=None,
        labeled_events=labeled,
        l0_delivery_manifest=None,
    )
    assert handoff.aligned is aligned
    assert handoff.aligned_by_tf is None
    assert handoff.l0_delivery_manifest is None
