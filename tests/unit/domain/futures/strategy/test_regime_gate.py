"""Unit tests for regime_gate.apply_regime_gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.regime_gate import apply_regime_gate


def _make_cfg(**kwargs: object) -> StrategyMLConfig:
    """Helper to build StrategyMLConfig with overrides."""
    return StrategyMLConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gate disabled
# ---------------------------------------------------------------------------


def test_gate_disabled_returns_originals() -> None:
    """When regime_gate_enabled=False, original arrays must be returned unchanged."""
    # Arrange
    cfg = _make_cfg(regime_gate_enabled=False)
    al = np.ones((10, 3), dtype=np.float32)
    as_ = np.ones((10, 3), dtype=np.float32)
    dts = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(10)]
    )
    btc = pd.Series(np.linspace(40_000, 50_000, 10), index=pd.to_datetime(dts))

    # Act
    out_l, out_s = apply_regime_gate(al, as_, dts, btc, cfg)

    # Assert — must be the exact same objects (no copy)
    assert out_l is al
    assert out_s is as_


# ---------------------------------------------------------------------------
# Gate enabled — bull / chop
# ---------------------------------------------------------------------------


def test_gate_enabled_bull_full_chop_zero() -> None:
    """Strictly rising BTC → bull bars get scalar=1.0; first trend_window bars unlabeled → scalar=1.0."""
    # Arrange
    n = 100
    close_vals = np.linspace(40_000, 60_000, n)
    dts = pd.date_range("2024-01-01", periods=n, freq="4h")
    btc = pd.Series(close_vals, index=dts)
    al = np.ones((n, 2), dtype=np.float32)
    as_ = np.ones((n, 2), dtype=np.float32)
    cfg = _make_cfg(
        regime_gate_enabled=True,
        regime_exposure_bull=1.0,
        regime_exposure_bear=0.3,
        regime_exposure_chop=0.0,
    )

    # Act
    out_l, out_s = apply_regime_gate(al, as_, dts.to_numpy(), btc, cfg)

    # Assert
    assert out_l.dtype == np.float32
    assert out_s.dtype == np.float32
    # Scaled values must be in [0.0, 1.0] (original was 1.0, scalars ≤ 1.0)
    assert np.all(out_l >= 0.0) and np.all(out_l <= 1.0)
    # First trend_window=30 bars are unlabeled (None → scalar 1.0)
    np.testing.assert_array_almost_equal(out_l[:30], 1.0)


# ---------------------------------------------------------------------------
# Insufficient BTC data → bypass
# ---------------------------------------------------------------------------


def test_gate_insufficient_btc_returns_originals() -> None:
    """Empty BTC series (finite_count < 30) must return originals unchanged."""
    # Arrange
    cfg = _make_cfg(regime_gate_enabled=True)
    al = np.ones((5, 2), dtype=np.float32)
    as_ = np.ones((5, 2), dtype=np.float32)
    dts = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(5)]
    )
    btc: pd.Series = pd.Series(dtype=np.float64)  # type: ignore[type-arg]

    # Act
    out_l, out_s = apply_regime_gate(al, as_, dts, btc, cfg)

    # Assert
    np.testing.assert_array_equal(out_l, al)
    np.testing.assert_array_equal(out_s, as_)


# ---------------------------------------------------------------------------
# Output shape preservation
# ---------------------------------------------------------------------------


def test_gate_output_shape_preserved() -> None:
    """Output arrays must have identical shape [T, N] to inputs."""
    # Arrange
    n, m = 50, 5
    cfg = _make_cfg(regime_gate_enabled=True, regime_exposure_bull=0.8)
    rng = np.random.default_rng(42)
    al = rng.random((n, m)).astype(np.float32)
    as_ = rng.random((n, m)).astype(np.float32)
    dts = pd.date_range("2024-01-01", periods=n, freq="4h").to_numpy()
    btc = pd.Series(
        np.linspace(30_000, 50_000, n),
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )

    # Act
    out_l, out_s = apply_regime_gate(al, as_, dts, btc, cfg)

    # Assert
    assert out_l.shape == (n, m)
    assert out_s.shape == (n, m)
    assert out_l.dtype == np.float32
    assert out_s.dtype == np.float32


# ---------------------------------------------------------------------------
# Bear exposure scalar applied
# ---------------------------------------------------------------------------


def test_gate_bear_scalar_applied() -> None:
    """Declining BTC (bear regime) must scale alpha by regime_exposure_bear."""
    # Arrange — strictly declining → all labeled bars should be bear
    n = 100
    close_vals = np.linspace(60_000, 40_000, n)  # monotone decline
    dts = pd.date_range("2024-01-01", periods=n, freq="4h")
    btc = pd.Series(close_vals, index=dts)
    al = np.full((n, 1), 2.0, dtype=np.float32)
    as_ = np.full((n, 1), 2.0, dtype=np.float32)
    bear_scalar = 0.3
    cfg = _make_cfg(
        regime_gate_enabled=True,
        regime_exposure_bull=1.0,
        regime_exposure_bear=bear_scalar,
        regime_exposure_chop=0.0,
    )

    # Act
    out_l, _ = apply_regime_gate(al, as_, dts.to_numpy(), btc, cfg)

    # Assert — labeled bear bars must be scaled by bear_scalar
    labeled_mask = np.arange(n) >= 30  # first 30 unlabeled (None → 1.0)
    np.testing.assert_array_almost_equal(
        out_l[labeled_mask], 2.0 * bear_scalar, decimal=5
    )
