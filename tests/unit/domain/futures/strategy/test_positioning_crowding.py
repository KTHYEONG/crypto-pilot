from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.market_regime import (
    compute_crowding_dampener_mult,
    compute_crowding_persistent_mask_2d,
    compute_positioning_crowding_z_2d,
)


def _make_aligned_fixture(
    oi_2d: NDArray[np.float64] | None = None,
    lsr_2d: NDArray[np.float64] | None = None,
    n_bars: int = 300,
    n_sym: int = 2,
) -> AlignedMarketData:
    close = np.full((n_bars, n_sym), 100.0, dtype=np.float64)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(n_bars).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("A", "B"),
        open_2d=close.copy(),
        high_2d=close * 1.01,
        low_2d=close * 0.99,
        close_2d=close,
        volume_2d=np.full((n_bars, n_sym), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((n_bars, n_sym), dtype=np.float64),
        active_mask=np.ones((n_bars, n_sym), dtype=bool),
        warm_mask=np.ones((n_bars, n_sym), dtype=bool),
        entry_block_mask=np.zeros((n_bars, n_sym), dtype=bool),
        kill_mask=np.zeros((n_bars, n_sym), dtype=bool),
        execution_cost_bps_2d=np.zeros((n_bars, n_sym), dtype=np.float64),
        oi_2d=oi_2d,
        lsr_2d=lsr_2d,
    )


# ── Scenario A: compute_crowding_dampener_mult ─────────────────────────


class TestComputeCrowdingDampenerMult:
    def test_crowded_returns_floor(self) -> None:
        assert compute_crowding_dampener_mult(True, floor_mult=0.30) == 0.30

    def test_not_crowded_returns_one(self) -> None:
        assert compute_crowding_dampener_mult(False, floor_mult=0.30) == 1.0

    def test_requires_floor_mult_argument(self) -> None:
        with pytest.raises(TypeError):
            compute_crowding_dampener_mult(True)  # type: ignore[call-arg]


# ── Scenario B: compute_crowding_persistent_mask_2d ────────────────────


class TestComputeCrowdingPersistentMask2D:
    def test_short_blip_does_not_fire(self) -> None:
        T, N = 20, 2
        oi = np.zeros((T, N), dtype=np.float64)
        lsr = np.zeros((T, N), dtype=np.float64)
        trend = np.ones((T, N), dtype=np.float64)
        oi[5:7, :] = 1.0
        lsr[5:7, :] = 2.0
        mask = compute_crowding_persistent_mask_2d(
            oi, lsr, trend, persistence_bars=3, recovery_cooldown_bars=2,
        )
        assert not mask.any()

    def test_fires_after_persistence_met(self) -> None:
        T, N = 20, 1
        oi = np.zeros((T, N), dtype=np.float64)
        lsr = np.zeros((T, N), dtype=np.float64)
        trend = np.ones((T, N), dtype=np.float64)
        oi[:4, :] = 1.0
        lsr[:4, :] = 2.0
        mask = compute_crowding_persistent_mask_2d(
            oi, lsr, trend, persistence_bars=3, recovery_cooldown_bars=0,
        )
        assert not mask[0, 0]
        assert not mask[1, 0]
        assert not mask[2, 0]
        assert mask[3, 0]
        assert mask[4, 0]

    def test_holds_during_cooldown(self) -> None:
        T, N = 15, 1
        oi = np.zeros((T, N), dtype=np.float64)
        lsr = np.zeros((T, N), dtype=np.float64)
        trend = np.ones((T, N), dtype=np.float64)
        oi[:4, :] = 1.0
        lsr[:4, :] = 2.0
        mask = compute_crowding_persistent_mask_2d(
            oi, lsr, trend, persistence_bars=3, recovery_cooldown_bars=3,
        )
        assert not mask[0, 0]
        assert not mask[1, 0]
        assert not mask[2, 0]
        assert mask[3, 0]
        assert mask[4, 0]
        assert mask[5, 0]
        assert mask[6, 0]
        assert not mask[7, 0]

    def test_columns_independent(self) -> None:
        T, N = 20, 2
        oi = np.zeros((T, N), dtype=np.float64)
        lsr = np.zeros((T, N), dtype=np.float64)
        trend = np.ones((T, N), dtype=np.float64)
        oi[:5, 0] = 1.0
        lsr[:5, 0] = 2.0
        mask = compute_crowding_persistent_mask_2d(
            oi, lsr, trend, persistence_bars=3, recovery_cooldown_bars=0,
        )
        assert mask[:, 0].any()
        assert not mask[:, 1].any()

    def test_rejects_invalid_persistence(self) -> None:
        oi = np.zeros((10, 2), dtype=np.float64)
        lsr = np.zeros((10, 2), dtype=np.float64)
        trend = np.ones((10, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="persistence_bars must be >= 1"):
            compute_crowding_persistent_mask_2d(
                oi, lsr, trend, persistence_bars=0,
            )


# ── Scenario C: compute_positioning_crowding_z_2d ──────────────────────


class TestComputePositioningCrowdingZ2D:
    def test_returns_correct_shape(self) -> None:
        T, N = 300, 2
        oi = np.full((T, N), 1e8, dtype=np.float64)
        lsr = np.full((T, N), 1.5, dtype=np.float64)
        aligned = _make_aligned_fixture(oi_2d=oi, lsr_2d=lsr, n_bars=T, n_sym=N)
        oi_z, lsr_z = compute_positioning_crowding_z_2d(aligned, tf="4h")
        assert oi_z.shape == (T, N)
        assert lsr_z.shape == (T, N)
        assert oi_z.dtype == np.float64
        assert lsr_z.dtype == np.float64

    def test_none_inputs_returns_all_nan(self) -> None:
        T, N = 300, 2
        aligned = _make_aligned_fixture(n_bars=T, n_sym=N)
        oi_z, lsr_z = compute_positioning_crowding_z_2d(aligned, tf="4h")
        assert oi_z.shape == (T, N)
        assert lsr_z.shape == (T, N)
        assert np.isnan(oi_z).all()
        assert np.isnan(lsr_z).all()

    def test_warms_up_before_window(self) -> None:
        T, N = 200, 2
        oi = np.full((T, N), 1e8, dtype=np.float64)
        lsr = np.full((T, N), 1.5, dtype=np.float64)
        oi[:60, :] = 0.0
        lsr[:60, :] = 0.0
        oi[60:90, :] = 2e8
        lsr[60:90, :] = 3.0
        aligned = _make_aligned_fixture(oi_2d=oi, lsr_2d=lsr, n_bars=T, n_sym=N)
        oi_z, lsr_z = compute_positioning_crowding_z_2d(aligned, tf="4h")
        warmup_bars = 42 + 6
        assert np.isnan(oi_z[:warmup_bars]).all()
        assert np.isnan(lsr_z[:warmup_bars]).all()
        assert np.any(np.isfinite(oi_z[warmup_bars:]))
        assert np.any(np.isfinite(lsr_z[warmup_bars:]))

    def test_scales_window_by_timeframe(self) -> None:
        T, N = 200, 2
        oi = np.full((T, N), 1e8, dtype=np.float64)
        lsr = np.full((T, N), 1.5, dtype=np.float64)
        oi[50:100, :] = 2e8
        lsr[50:100, :] = 3.0
        aligned = _make_aligned_fixture(oi_2d=oi, lsr_2d=lsr, n_bars=T, n_sym=N)
        oi_z_4h, _lsr_z_4h = compute_positioning_crowding_z_2d(aligned, tf="4h")
        oi_z_8h, _lsr_z_8h = compute_positioning_crowding_z_2d(aligned, tf="8h")
        first_finite_4h = int(np.where(np.isfinite(oi_z_4h[:, 0]))[0][0])
        first_finite_8h = int(np.where(np.isfinite(oi_z_8h[:, 0]))[0][0])
        ratio = first_finite_4h / max(first_finite_8h, 1)
        assert 1.5 <= ratio <= 2.5
