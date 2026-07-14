from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.application.futures.runner.tiered_handoff import (
    TieredHandoffError,
    TieredL1Handoff,
    consume_candidate_output_for_tiered,
)
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput


def test_tiered_handoff_moves_without_copy() -> None:
    aligned = MagicMock()
    aligned.symbols = ("BTCUSDT", "ETHUSDT")
    labeled = pd.DataFrame(
        {"native_tf": ["4h"], "l0_recipe_id": ["r1"], "realized_return": [0.01]}
    )
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
        labeled_events=labeled,
        l0_delivery_manifest=None,
    )
    assert handoff.aligned is aligned
    assert handoff.l0_delivery_manifest is None
