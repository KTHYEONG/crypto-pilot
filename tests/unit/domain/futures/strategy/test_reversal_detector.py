from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.market_regime import compute_reversal_risk_off_1d


def _close_rise_then_fall() -> NDArray[np.float64]:
    rise = np.linspace(100.0, 110.0, 20, dtype=np.float64)
    fall = np.linspace(110.0, 85.0, 30, dtype=np.float64)
    return np.concatenate([rise, fall]).astype(np.float64)


def _flat_then_decline(
    *,
    flat_n: int = 30,
    decline: tuple[float, ...] = (100.0, 98.0, 95.0, 88.0, 86.0),
) -> NDArray[np.float64]:
    return np.concatenate(
        [
            np.full(flat_n, 100.0, dtype=np.float64),
            np.asarray(decline, dtype=np.float64),
        ],
    ).astype(np.float64)


def test_reversal_detector_flags_decline_early() -> None:
    close = _close_rise_then_fall()
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
    )
    assert risk_off.shape == close.shape
    assert risk_off.dtype == np.bool_
    up_region = risk_off[:20]
    assert not up_region.any(), "no risk-off during uptrend"
    decline_region = risk_off[20:]
    assert decline_region.any(), "risk-off triggers during decline"
    first_true = int(np.argmax(decline_region)) + 20
    min_idx = int(np.argmin(close))
    assert first_true <= min_idx, "first risk-off precedes or equals the lowest close"


def test_reversal_detector_is_causal_shift1() -> None:
    close = np.full(50, 100.0, dtype=np.float64)
    close[-6:-3] = 100.0
    close[-3:] = [97.0, 93.0, 90.0]
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=10, dd_threshold=0.03, mom_fast=3, mom_slow=10,
    )
    assert not risk_off[0], "row0 is always False"
    modified_close = close.copy()
    modified_close[-1] = 85.0
    risk_off_modified = compute_reversal_risk_off_1d(
        modified_close, dd_window=10, dd_threshold=0.03, mom_fast=3, mom_slow=10,
    )
    assert np.array_equal(risk_off[:-1], risk_off_modified[:-1]), "past values unchanged by future price"


def test_reversal_detector_silent_in_uptrend() -> None:
    close = np.linspace(100.0, 200.0, 100, dtype=np.float64)
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=30, dd_threshold=0.05, mom_fast=10, mom_slow=30,
    )
    assert not risk_off.any(), "no risk-off in monotonic uptrend"


def test_reversal_detector_dd_threshold_boundary() -> None:
    close = np.full(60, 100.0, dtype=np.float64)
    peak = np.full(20, 100.0, dtype=np.float64)
    just_below = np.linspace(100.0, 94.01, 8, dtype=np.float64)
    just_above = np.linspace(100.0, 93.99, 8, dtype=np.float64)
    tail = np.full(24, 93.0, dtype=np.float64)
    close = np.concatenate([peak, just_below, just_above, tail])
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=30, dd_threshold=0.06, mom_fast=5, mom_slow=15,
    )
    below_idx = len(peak) + len(just_below) - 1
    above_start = len(peak) + len(just_below)
    assert not risk_off[below_idx], "dd < threshold -> False"
    above_triggered = risk_off[above_start:]
    assert above_triggered.any(), "dd >= threshold -> True at least once"


def test_reversal_detector_requires_negative_momentum() -> None:
    close = np.full(80, 100.0, dtype=np.float64)
    drop = np.linspace(100.0, 80.0, 20, dtype=np.float64)
    recovery = np.linspace(80.0, 95.0, 20, dtype=np.float64)
    close = np.concatenate([close, drop, recovery])
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=30, dd_threshold=0.05, mom_fast=5, mom_slow=20,
    )
    assert risk_off.shape == close.shape
    recovery_region = risk_off[-9:]
    assert not recovery_region.any(), "no risk-off during late recovery (positive momentum despite deep dd)"


def test_reversal_detector_requires_persistent_raw_condition() -> None:
    close = _flat_then_decline(decline=(100.0, 87.0, 99.0, 100.0))
    risk_off = compute_reversal_risk_off_1d(
        close,
        dd_window=20,
        dd_threshold=0.10,
        mom_fast=2,
        mom_slow=8,
        persistence_bars=2,
    )
    assert not risk_off.any(), "single-bar drawdown spike must not trigger with persistence_bars=2"


def test_reversal_detector_flags_sustained_reversal_after_shift() -> None:
    close = _flat_then_decline(decline=(100.0, 96.0, 91.0, 88.0, 86.0, 84.0, 82.0))
    risk_off_legacy = compute_reversal_risk_off_1d(
        close,
        dd_window=20,
        dd_threshold=0.10,
        mom_fast=2,
        mom_slow=8,
        persistence_bars=1,
    )
    legacy_first = int(np.where(risk_off_legacy)[0][0])
    risk_off = compute_reversal_risk_off_1d(
        close,
        dd_window=20,
        dd_threshold=0.10,
        mom_fast=2,
        mom_slow=8,
        persistence_bars=3,
    )
    assert not risk_off[0], "row0 is always False"
    true_idxs = np.where(risk_off)[0]
    assert len(true_idxs) > 0, "must have at least one risk-off bar"
    expected_first = legacy_first + 2
    assert int(true_idxs[0]) == expected_first, (
        f"first risk-off at {true_idxs[0]}, expected {expected_first} "
        f"(legacy first at {legacy_first})"
    )


def test_reversal_detector_persistence_one_matches_legacy_behavior() -> None:
    close = _close_rise_then_fall()
    default_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
    )
    explicit_one_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20, persistence_bars=1,
    )
    assert np.array_equal(default_out, explicit_one_out)


def test_reversal_detector_rejects_invalid_persistence() -> None:
    close = np.full(20, 100.0, dtype=np.float64)
    with pytest.raises(ValueError, match="persistence_bars must be >= 1"):
        compute_reversal_risk_off_1d(
            close, dd_window=5, dd_threshold=0.10, mom_fast=2, mom_slow=8, persistence_bars=0,
        )
