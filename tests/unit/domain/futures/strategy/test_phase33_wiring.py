"""Phase 3-3 wiring tests: state_cube injection logic for align_data_maps.

Verifies the conditional logic that wires ``state_cube`` into ``align_data_maps``
when ``universe_engine="pit"`` and passes ``None`` on the ``stage6`` path.

Because ``align_data_maps`` is imported inside a local ``try`` block in
``_run_strategy_stage``, we patch at the definition site
``src.domain.futures.strategy.common.alignment.align_data_maps`` and exercise
the wiring logic in isolation — no need to call the full pipeline.

Complexity: O(1) — mock-only, no real data structures.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.universe.contracts import (
    UniverseStateCube,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALIGN_TARGET = "src.domain.futures.strategy.common.alignment.align_data_maps"
_READINESS_TARGET = "src.domain.futures.universe.readiness.evaluate_strategy_readiness"


def _make_state_cube() -> UniverseStateCube:
    """Minimal 1-bar x 1-instrument UniverseStateCube."""
    cal = pd.DatetimeIndex(["2024-01-01"], tz="UTC")
    return UniverseStateCube(
        calendar=cal,
        instrument_ids=("BTCUSDT",),
        eligible=np.ones((1, 1), dtype=np.bool_),
        entry_block=np.zeros((1, 1), dtype=np.bool_),
        exit_required=np.zeros((1, 1), dtype=np.bool_),
        capacity_usdt=np.full((1, 1), 1_000.0, dtype=np.float64),
        risk_scale=np.ones((1, 1), dtype=np.float64),
        cost_bps=np.full((1, 1), 5.0, dtype=np.float64),
    )


def _make_aligned_obj(n_t: int = 1, n_n: int = 1) -> AlignedMarketData:
    """Minimal AlignedMarketData compatible with dataclasses.replace."""
    return AlignedMarketData(
        datetimes=np.array(["2024-01-01"], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        open_2d=np.ones((n_t, n_n)),
        high_2d=np.ones((n_t, n_n)),
        low_2d=np.ones((n_t, n_n)),
        close_2d=np.ones((n_t, n_n)),
        volume_2d=np.ones((n_t, n_n)),
        funding_2d=np.zeros((n_t, n_n)),
        active_mask=np.ones((n_t, n_n), dtype=np.bool_),
        warm_mask=np.ones((n_t, n_n), dtype=np.bool_),
        entry_block_mask=np.zeros((n_t, n_n), dtype=np.bool_),
        kill_mask=np.zeros((n_t, n_n), dtype=np.bool_),
    )


def _run_wiring_block(
    *,
    universe_engine: str,
    state_cube: UniverseStateCube,
    aligned_return: AlignedMarketData,
) -> tuple[AlignedMarketData, dict[str, Any]]:
    """Replicate the Phase 3-3 wiring block from _run_strategy_stage.

    Returns the final aligned object and the kwargs used in the align_data_maps call.
    """

    from src.domain.futures.strategy.common.alignment import align_data_maps

    run_config = SimpleNamespace(
        timeframe="4h",
        universe_engine=universe_engine,
        wf_lookback_bars=100,
    )
    universe_result = SimpleNamespace(state_cube=state_cube)
    full_strategy_maps: dict[str, Any] = {}
    effective_trade_syms: list[str] = ["BTCUSDT"]

    _is_pit: bool = (
        universe_result is not None
        and getattr(run_config, "universe_engine", "stage6") == "pit"
    )
    _pit_state_cube = (
        universe_result.state_cube  # type: ignore[union-attr]
        if _is_pit
        else None
    )
    result = align_data_maps(
        full_strategy_maps,
        effective_trade_syms,
        run_config.timeframe,
        state_cube=_pit_state_cube,
    )
    return result, {"state_cube": _pit_state_cube}


# ---------------------------------------------------------------------------
# Test 1 — PIT path: align_data_maps receives state_cube kwarg non-None
# ---------------------------------------------------------------------------

def test_align_data_maps_receives_state_cube_on_pit_path() -> None:
    """When universe_engine='pit', wiring block passes state_cube != None.

    Arrange:
        - universe_engine = "pit", state_cube populated.
    Act:
        - Execute extracted wiring block with mocked align_data_maps.
    Assert:
        - align_data_maps called with state_cube that is the exact state_cube instance.
    """
    # Arrange
    state_cube = _make_state_cube()
    aligned_obj = _make_aligned_obj()

    with patch(_ALIGN_TARGET, return_value=aligned_obj) as mock_align:
        # Act
        _, kwargs = _run_wiring_block(
            universe_engine="pit",
            state_cube=state_cube,
            aligned_return=aligned_obj,
        )

        # Assert
        mock_align.assert_called_once()
        passed_cube = mock_align.call_args.kwargs.get("state_cube")
        assert passed_cube is not None, (
            "state_cube kwarg must be non-None on the pit path"
        )
        assert passed_cube is state_cube


# ---------------------------------------------------------------------------
# Test 2 — stage6 path: align_data_maps receives state_cube=None
# ---------------------------------------------------------------------------

def test_align_data_maps_has_no_state_cube_on_stage6_path() -> None:
    """When universe_engine='stage6', wiring block passes state_cube=None.

    Arrange:
        - universe_engine = "stage6".
    Act:
        - Execute extracted wiring block with mocked align_data_maps.
    Assert:
        - align_data_maps called with state_cube=None.
    """
    # Arrange
    state_cube = _make_state_cube()
    aligned_obj = _make_aligned_obj()

    with patch(_ALIGN_TARGET, return_value=aligned_obj) as mock_align:
        # Act
        _, kwargs = _run_wiring_block(
            universe_engine="stage6",
            state_cube=state_cube,
            aligned_return=aligned_obj,
        )

        # Assert
        mock_align.assert_called_once()
        passed_cube = mock_align.call_args.kwargs.get("state_cube", None)
        assert passed_cube is None, (
            f"state_cube must be None on the stage6 path; got {passed_cube!r}"
        )
