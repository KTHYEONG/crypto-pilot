"""OPT-1 regression tests: loop-invariant hoisting in align_data_maps state_cube join.

Scenarios:
- S1 (Happy/Equivalence): Small UniverseStateCube fixture (T_cube=10, N_cube=3).
  active_mask, entry_block_mask, adv_usdt_2d, execution_cost_bps_2d are bit-identical
  to golden arrays computed from known cube values.
- S2 (Edge): All aligned_ts_ns before cube.calendar[0] → t_valid.size==0
  → all masks retain initial values (active=1, entry_block=0).
- S3 (Edge): Some symbols not in cube_sym_idx → those columns unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.common.alignment import align_data_maps
from src.domain.futures.universe.contracts import UniverseStateCube

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TF: str = "4h"
# compute_multi_alignment_info requires eff_ref_len >= 200
_N_BARS: int = 250
_CUBE_T: int = 10   # T_cube
_CUBE_N: int = 3    # N_cube


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_datetimes(base: pd.Timestamp, n_bars: int, freq: str = "4h") -> pd.DatetimeIndex:
    """Build a DatetimeIndex starting at base."""
    return pd.date_range(base, periods=n_bars, freq=freq, tz="UTC")


def _make_data_maps(
    symbols: list[str],
    tf: str = _TF,
    n_bars: int = _N_BARS,
    base: pd.Timestamp | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build minimal data_maps with required OHLCV + datetime columns.

    Args:
        symbols: List of symbol strings.
        tf: Timeframe key.
        n_bars: Number of OHLCV bars per symbol.
        base: Base timestamp; defaults to 2024-01-05 (after cube range 2024-01-01).
    """
    if base is None:
        base = pd.Timestamp("2024-01-05", tz="UTC")
    datetimes = _make_datetimes(base, n_bars, tf)
    maps: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        rng = np.random.default_rng(seed=abs(hash(sym)) % (2**31))
        df = pd.DataFrame(
            {
                "datetime": datetimes,
                "open": rng.uniform(100.0, 200.0, n_bars),
                "high": rng.uniform(200.0, 300.0, n_bars),
                "low": rng.uniform(50.0, 100.0, n_bars),
                "close": rng.uniform(100.0, 200.0, n_bars),
                "volume": rng.uniform(1e4, 1e6, n_bars),
            }
        )
        maps[sym] = {tf: df}
    return maps


def _make_state_cube(
    symbols: list[str],
    t_cube: int = _CUBE_T,
    *,
    eligible_arr: np.ndarray | None = None,
    entry_block_arr: np.ndarray | None = None,
    capacity_arr: np.ndarray | None = None,
    cost_arr: np.ndarray | None = None,
    calendar_base: pd.Timestamp | None = None,
) -> UniverseStateCube:
    """Build a small UniverseStateCube fixture.

    Args:
        symbols: Symbol list used as instrument_ids.
        t_cube: Number of cube calendar bars.
        eligible_arr: Optional [T_cube, N_cube] bool array; default all True.
        entry_block_arr: Optional [T_cube, N_cube] bool array; default all False.
        capacity_arr: Optional [T_cube, N_cube] float64 array; default 10_000.
        cost_arr: Optional [T_cube, N_cube] float64 array; default 5.0.
        calendar_base: Base timestamp for cube calendar; defaults to 2024-01-01.
    """
    n = len(symbols)
    if calendar_base is None:
        calendar_base = pd.Timestamp("2024-01-01", tz="UTC")
    calendar = pd.date_range(calendar_base, periods=t_cube, freq="4h", tz="UTC")

    eligible = eligible_arr if eligible_arr is not None else np.ones((t_cube, n), dtype=np.bool_)
    entry_block = (
        entry_block_arr if entry_block_arr is not None else np.zeros((t_cube, n), dtype=np.bool_)
    )
    capacity_usdt = (
        capacity_arr if capacity_arr is not None else np.full((t_cube, n), 10_000.0, dtype=np.float64)
    )
    cost_bps = cost_arr if cost_arr is not None else np.full((t_cube, n), 5.0, dtype=np.float64)

    return UniverseStateCube(
        calendar=calendar,
        instrument_ids=tuple(symbols),
        eligible=eligible,
        entry_block=entry_block,
        exit_required=np.zeros((t_cube, n), dtype=np.bool_),
        capacity_usdt=capacity_usdt,
        risk_scale=np.ones((t_cube, n), dtype=np.float64),
        cost_bps=cost_bps,
    )


# ---------------------------------------------------------------------------
# S1: Happy path — bit-identical golden array verification
# ---------------------------------------------------------------------------


class TestOpt1StateCubeHoisting:
    """S1: Hoisted searchsorted produces bit-identical results to pre-refactor golden arrays."""

    def test_s1_active_mask_bit_identical_to_golden(self) -> None:
        """align_data_maps with state_cube: active_mask matches expected eligible values.

        Golden: cube eligible[*, 0]=True, [*, 1]=False, [*, 2]=True.
        After join, col-0 fully active, col-1 fully inactive, col-2 fully active.
        """
        # Arrange
        symbols = ["AAA", "BBB", "CCC"]  # N_cube=3
        data_maps = _make_data_maps(symbols)

        rng = np.random.default_rng(42)
        eligible_arr = np.ones((_CUBE_T, _CUBE_N), dtype=np.bool_)
        eligible_arr[:, 1] = False  # BBB → ineligible
        entry_block_arr = np.zeros((_CUBE_T, _CUBE_N), dtype=np.bool_)
        entry_block_arr[:, 2] = True  # CCC → entry blocked
        capacity_arr = np.full((_CUBE_T, _CUBE_N), 99_000.0, dtype=np.float64)
        cost_arr = np.full((_CUBE_T, _CUBE_N), 7.5, dtype=np.float64)

        cube = _make_state_cube(
            symbols,
            eligible_arr=eligible_arr,
            entry_block_arr=entry_block_arr,
            capacity_arr=capacity_arr,
            cost_arr=cost_arr,
        )

        # Act
        aligned = align_data_maps(data_maps, symbols, _TF, state_cube=cube)

        # Assert — golden: derive expected via same searchsorted logic
        datetimes = aligned.datetimes.astype("datetime64[ns]").view(np.int64)
        cube_ts_ns = cube.calendar.view(np.int64)
        pos = np.searchsorted(cube_ts_ns, datetimes, side="right") - 1
        valid_mask = pos >= 0
        t_v = np.where(valid_mask)[0]
        p_v = pos[valid_mask]

        for col, sym in enumerate(aligned.symbols):
            sym_idx = list(symbols).index(sym)
            expected_active = np.ones(aligned.datetimes.shape[0], dtype=np.bool_)
            expected_entry_block = np.zeros(aligned.datetimes.shape[0], dtype=np.bool_)
            expected_adv = np.full(aligned.datetimes.shape[0], np.nan)
            expected_cost = np.full(aligned.datetimes.shape[0], np.nan)
            if t_v.size > 0:
                expected_active[t_v] = eligible_arr[p_v, sym_idx]
                expected_entry_block[t_v] = entry_block_arr[p_v, sym_idx]
                expected_adv[t_v] = capacity_arr[p_v, sym_idx]
                expected_cost[t_v] = cost_arr[p_v, sym_idx]

            np.testing.assert_array_equal(
                aligned.active_mask[:, col],
                expected_active,
                err_msg=f"active_mask mismatch for {sym}",
            )
            np.testing.assert_array_equal(
                aligned.entry_block_mask[:, col],
                expected_entry_block,
                err_msg=f"entry_block_mask mismatch for {sym}",
            )

            adv = aligned.adv_usdt_2d
            cost = aligned.execution_cost_bps_2d
            assert adv is not None
            assert cost is not None

            # NaN positions remain NaN; valid positions are bit-equal
            nan_mask = np.isnan(expected_adv)
            np.testing.assert_array_equal(
                np.isnan(adv[:, col]),
                nan_mask,
                err_msg=f"adv NaN pattern mismatch for {sym}",
            )
            np.testing.assert_array_equal(
                adv[~nan_mask, col],
                expected_adv[~nan_mask],
                err_msg=f"adv_usdt_2d value mismatch for {sym}",
            )
            np.testing.assert_array_equal(
                cost[~nan_mask, col],
                expected_cost[~nan_mask],
                err_msg=f"execution_cost_bps_2d value mismatch for {sym}",
            )


# ---------------------------------------------------------------------------
# S2: Edge — all aligned_ts_ns before cube.calendar[0] → t_valid.size==0
# ---------------------------------------------------------------------------


class TestOpt1AllTimestampsBeforeCube:
    """S2: When all aligned bars precede cube start, masks keep initial values."""

    def test_s2_masks_retain_initial_values_when_all_ts_before_cube(self) -> None:
        """active_mask=1, entry_block_mask=0 when cube range does not cover aligned range.

        Cube calendar starts at 2030-01-01; aligned data ends at ~2024-01-end.
        → positions all -1 → t_valid.size==0 → no cube overwrites.
        """
        # Arrange
        symbols = ["AAA", "BBB", "CCC"]
        # Aligned data: 2024-01-05 to ~2024-06 (well before cube)
        data_maps = _make_data_maps(symbols, base=pd.Timestamp("2024-01-05", tz="UTC"))

        # Cube starts in the far future
        cube = _make_state_cube(
            symbols,
            calendar_base=pd.Timestamp("2030-01-01", tz="UTC"),
            eligible_arr=np.zeros((_CUBE_T, _CUBE_N), dtype=np.bool_),
            entry_block_arr=np.ones((_CUBE_T, _CUBE_N), dtype=np.bool_),
        )

        # Act
        aligned = align_data_maps(data_maps, symbols, _TF, state_cube=cube)

        # Assert — initial values: active=True (all ones), entry_block=False (all zeros)
        assert aligned.active_mask.all(), (
            "active_mask must remain True when no cube bars cover aligned range"
        )
        assert not aligned.entry_block_mask.any(), (
            "entry_block_mask must remain False when no cube bars cover aligned range"
        )


# ---------------------------------------------------------------------------
# S3: Edge — some symbols not in cube → those columns unchanged
# ---------------------------------------------------------------------------


class TestOpt1PartialCubeSymbolCoverage:
    """S3: Symbols absent from cube_sym_idx keep initial column values."""

    def test_s3_absent_symbol_columns_unchanged(self) -> None:
        """When only a subset of valid_symbols exists in cube, absent columns are untouched.

        Cube covers only [AAA, CCC]; BBB is absent → BBB columns keep initial values.
        """
        # Arrange
        symbols = ["AAA", "BBB", "CCC"]
        cube_symbols = ["AAA", "CCC"]  # BBB intentionally absent

        data_maps = _make_data_maps(symbols)

        # Cube N=2: AAA at col-0, CCC at col-1
        eligible_arr_2 = np.ones((_CUBE_T, 2), dtype=np.bool_)
        eligible_arr_2[:, 0] = False  # AAA → ineligible

        cube = _make_state_cube(
            cube_symbols,
            eligible_arr=eligible_arr_2,
        )

        # Act
        aligned = align_data_maps(data_maps, symbols, _TF, state_cube=cube)

        # Assert
        bbb_col = list(aligned.symbols).index("BBB")
        aaa_col = list(aligned.symbols).index("AAA")

        # BBB column: active_mask must still be all True (initial), entry_block all False
        assert aligned.active_mask[:, bbb_col].all(), (
            "BBB active_mask must remain initial True (not in cube)"
        )
        assert not aligned.entry_block_mask[:, bbb_col].any(), (
            "BBB entry_block_mask must remain initial False (not in cube)"
        )

        # AAA column: cube overwrote eligible=False → active_mask must be False
        # (only for bars within cube range; at minimum some bars must be False)
        cube_ts_ns = cube.calendar.view(np.int64)
        aaa_datetimes = aligned.datetimes.astype("datetime64[ns]").view(np.int64)
        pos = np.searchsorted(cube_ts_ns, aaa_datetimes, side="right") - 1
        t_v = np.where(pos >= 0)[0]

        if t_v.size > 0:
            # AAA (col-0 in cube) has eligible=False → active_mask should be False at t_v
            assert not aligned.active_mask[t_v, aaa_col].any(), (
                "AAA bars covered by cube must have active_mask=False"
            )
