from __future__ import annotations

import pandas as pd
import pytest

from src.application.futures.optimization.strategy_service import (
    assert_candidate_output_ready,
    pick_strategy_data_maps,
)
from src.domain.futures.strategy_runtime.bridge import CandidatePipelineOutput


def test_pick_strategy_data_maps_strips_is_start_idx() -> None:
    frame = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01"]), "close": [1.0]})
    picked = pick_strategy_data_maps(
        {"BTCUSDT": {"4h": frame}},
        {"BTCUSDT": {"4h": frame, "is_start_idx_4h": 10, "x": 1}},
        ["BTCUSDT"],
        "4h",
    )
    assert "is_start_idx_4h" not in picked["BTCUSDT"]


def test_assert_candidate_output_ready_requires_target_weight() -> None:
    out = CandidatePipelineOutput(
        alpha_panel=pd.DataFrame({"alpha_long": [0.1]}),
    )
    with pytest.raises(RuntimeError, match="target_weight"):
        assert_candidate_output_ready(
            candidate_out=out,
            oos_data_maps={"BTCUSDT": {"4h": pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01"])})}},
            valid_symbols=["BTCUSDT"],
            tf="4h",
        )


def test_assert_candidate_output_ready_accepts_nonzero_target_weight() -> None:
    panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01"]),
            "symbol": ["BTCUSDT"],
            "target_weight": [0.1],
        }
    ).set_index(["datetime", "symbol"])
    out = CandidatePipelineOutput(alpha_panel=panel)
    oos = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-01-01"]),
                    "target_weight": [0.1],
                }
            )
        }
    }
    report = assert_candidate_output_ready(
        candidate_out=out,
        oos_data_maps=oos,
        valid_symbols=["BTCUSDT"],
        tf="4h",
    )
    assert report.merged_symbols == 1
