"""Tests for PIT state_cube integration in align_data_maps (Phase 3-1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.universe.contracts import UniverseStateCube

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_BARS: int = 250  # compute_multi_alignment_info requires eff_ref_len >= 200


def _make_data_maps(
    symbols: list[str],
    tf: str,
    n_bars: int = _MIN_BARS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build minimal data_maps with required OHLCV + datetime columns."""
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    freq = "4h" if tf == "4h" else tf
    datetimes = pd.date_range(base_ts, periods=n_bars, freq=freq)

    maps: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        df = pd.DataFrame(
            {
                "datetime": datetimes,
                "open": np.ones(n_bars, dtype=np.float64),
                "high": np.ones(n_bars, dtype=np.float64),
                "low": np.ones(n_bars, dtype=np.float64),
                "close": np.ones(n_bars, dtype=np.float64),
                "volume": np.ones(n_bars, dtype=np.float64),
            }
        )
        maps[sym] = {tf: df}
    return maps


def _make_state_cube(
    symbols: list[str],
    n_bars: int = _MIN_BARS,
    eligible_flags: dict[str, bool] | None = None,
    entry_block_flags: dict[str, bool] | None = None,
) -> UniverseStateCube:
    """Build a UniverseStateCube fixture.

    Args:
        symbols: Symbol list (used as instrument_ids directly).
        n_bars: Number of calendar bars.
        eligible_flags: Per-symbol eligible value; default True for all.
        entry_block_flags: Per-symbol entry_block value; default False for all.
    """
    eligible_flags = eligible_flags or {}
    entry_block_flags = entry_block_flags or {}

    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    calendar = pd.date_range(base_ts, periods=n_bars, freq="4h", tz="UTC")

    n = len(symbols)
    eligible = np.ones((n_bars, n), dtype=np.bool_)
    entry_block = np.zeros((n_bars, n), dtype=np.bool_)
    exit_required = np.zeros((n_bars, n), dtype=np.bool_)
    capacity_usdt = np.full((n_bars, n), 10_000.0, dtype=np.float64)
    risk_scale = np.ones((n_bars, n), dtype=np.float64)
    cost_bps = np.full((n_bars, n), 5.0, dtype=np.float64)

    for col, sym in enumerate(symbols):
        if sym in eligible_flags:
            eligible[:, col] = eligible_flags[sym]
        if sym in entry_block_flags:
            entry_block[:, col] = entry_block_flags[sym]

    return UniverseStateCube(
        calendar=calendar,
        instrument_ids=tuple(symbols),
        eligible=eligible,
        entry_block=entry_block,
        exit_required=exit_required,
        capacity_usdt=capacity_usdt,
        risk_scale=risk_scale,
        cost_bps=cost_bps,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlignDataMapsWithStateCube:
    """Phase 3-1: state_cube join into active_mask / entry_block_mask."""

    def test_align_data_maps_with_state_cube_fills_active_mask(self) -> None:
        """state_cube eligible=True→sym0, False→sym1 must propagate into active_mask."""
        # Arrange
        symbols = ["SYM0", "SYM1"]
        tf = "4h"
        n_bars = _MIN_BARS
        data_maps = _make_data_maps(symbols, tf, n_bars)
        cube = _make_state_cube(
            symbols,
            n_bars,
            eligible_flags={"SYM0": True, "SYM1": False},
        )

        # Act
        from src.domain.futures.strategy.common.alignment import align_data_maps

        aligned = align_data_maps(data_maps, symbols, tf, state_cube=cube)

        # Assert
        sym0_col = list(aligned.symbols).index("SYM0")
        sym1_col = list(aligned.symbols).index("SYM1")
        assert aligned.active_mask[:, sym0_col].all(), "SYM0 must be fully active"
        assert not aligned.active_mask[:, sym1_col].any(), "SYM1 must be fully inactive"

    def test_align_data_maps_without_state_cube_uses_all_pass_default(self) -> None:
        """Without state_cube, active_mask must default to all True (existing behavior)."""
        # Arrange
        symbols = ["SYM0", "SYM1"]
        tf = "4h"
        data_maps = _make_data_maps(symbols, tf)

        # Act
        from src.domain.futures.strategy.common.alignment import align_data_maps

        aligned = align_data_maps(data_maps, symbols, tf)

        # Assert — no state_cube means all-pass active mask
        assert aligned.active_mask.all(), "default active_mask must be all True"

    def test_align_data_maps_state_cube_entry_block(self) -> None:
        """entry_block=True for SYM0 must propagate into aligned.entry_block_mask[:,col]."""
        # Arrange
        symbols = ["SYM0", "SYM1"]
        tf = "4h"
        n_bars = _MIN_BARS
        data_maps = _make_data_maps(symbols, tf, n_bars)
        cube = _make_state_cube(
            symbols,
            n_bars,
            entry_block_flags={"SYM0": True, "SYM1": False},
        )

        # Act
        from src.domain.futures.strategy.common.alignment import align_data_maps

        aligned = align_data_maps(data_maps, symbols, tf, state_cube=cube)

        # Assert
        sym0_col = list(aligned.symbols).index("SYM0")
        sym1_col = list(aligned.symbols).index("SYM1")
        assert aligned.entry_block_mask[:, sym0_col].all(), "SYM0 entry_block must be all True"
        assert not aligned.entry_block_mask[:, sym1_col].any(), "SYM1 entry_block must be all False"

    def test_align_data_maps_state_cube_unknown_symbol_is_skipped(self) -> None:
        """Symbols absent from state_cube instrument_ids must not raise — default masks preserved."""
        # Arrange
        symbols = ["SYM0"]
        tf = "4h"
        n_bars = _MIN_BARS
        data_maps = _make_data_maps(symbols, tf, n_bars)
        # cube only has SYM_UNKNOWN — SYM0 is absent
        cube = _make_state_cube(
            ["SYM_UNKNOWN"],
            n_bars,
            eligible_flags={"SYM_UNKNOWN": False},
        )

        # Act
        from src.domain.futures.strategy.common.alignment import align_data_maps

        aligned = align_data_maps(data_maps, symbols, tf, state_cube=cube)

        # Assert — SYM0 not in cube → active_mask defaults to all True
        assert aligned.active_mask.all(), "missing cube symbol must keep default active_mask=True"

    def test_align_data_maps_state_cube_bypasses_cache(self) -> None:
        """Calling with state_cube=None and then state_cube=... must return different active_masks."""
        # Arrange
        symbols = ["SYM0"]
        tf = "4h"
        n_bars = _MIN_BARS
        data_maps = _make_data_maps(symbols, tf, n_bars)
        cube = _make_state_cube(symbols, n_bars, eligible_flags={"SYM0": False})

        from src.domain.futures.strategy.common.alignment import align_data_maps

        # Act — first call without cube (populates cache)
        aligned_no_cube = align_data_maps(data_maps, symbols, tf)
        # Second call with cube (must NOT use cache)
        aligned_with_cube = align_data_maps(data_maps, symbols, tf, state_cube=cube)

        # Assert
        assert aligned_no_cube.active_mask.all()
        assert not aligned_with_cube.active_mask.any()

    @pytest.mark.xfail(
        reason="P2 semantic cleanup for PIT cube ADV/capacity split is not implemented yet",
        strict=True,
    )
    def test_align_data_maps_state_cube_keeps_market_adv_usdt(self) -> None:
        """state_cube capacity_usdt must not overwrite market ADV semantics."""
        symbols = ["SYM0"]
        tf = "4h"
        n_bars = _MIN_BARS
        data_maps = _make_data_maps(symbols, tf, n_bars)
        data_maps["SYM0"][tf] = data_maps["SYM0"][tf].assign(adv_usdt=np.full(n_bars, 123.0, dtype=np.float64))
        cube = _make_state_cube(symbols, n_bars, eligible_flags={"SYM0": True})
        cube.capacity_usdt[:, 0] = 999.0

        from src.domain.futures.strategy.common.alignment import align_data_maps

        aligned = align_data_maps(data_maps, symbols, tf, state_cube=cube)

        np.testing.assert_allclose(aligned.adv_usdt_2d[:, 0], 123.0)
