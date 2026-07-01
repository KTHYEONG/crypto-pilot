from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.domain.futures.strategy.market_regime import (
    compute_reversal_risk_off_1d,
    synthetic_crash_defense_verdict,
)


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


# ── L2 Reversal Recovery Cooldown (Scenario 1-3, 6-8) ──────────────


def test_reversal_detector_cooldown_zero_matches_legacy_behavior() -> None:
    """Scenario 1: recovery_cooldown_bars=0 is byte-identical with default (no cooldown)."""
    close = _close_rise_then_fall()
    default_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20, persistence_bars=1,
    )
    explicit_zero_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
        persistence_bars=1, recovery_cooldown_bars=0,
    )
    assert np.array_equal(default_out, explicit_zero_out)


def _close_whipsaw() -> NDArray[np.float64]:
    rise = np.linspace(100.0, 110.0, 30, dtype=np.float64)
    drop = np.linspace(110.0, 85.0, 15, dtype=np.float64)
    bounce = np.linspace(85.0, 92.0, 5, dtype=np.float64)
    redrop = np.linspace(92.0, 78.0, 10, dtype=np.float64)
    return np.concatenate([rise, drop, bounce, redrop]).astype(np.float64)


def test_reversal_detector_cooldown_extends_risk_off_after_raw_clears() -> None:
    """Scenario 2: cooldown extends risk-off after raw clears, never shrinks it."""
    close = _close_whipsaw()
    no_cooldown_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
        persistence_bars=1, recovery_cooldown_bars=0,
    )
    cooldown_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
        persistence_bars=1, recovery_cooldown_bars=5,
    )
    assert cooldown_out.sum() >= no_cooldown_out.sum()
    first_on = int(np.argmax(no_cooldown_out)) if no_cooldown_out.any() else close.shape[0]
    assert np.array_equal(cooldown_out[:first_on], no_cooldown_out[:first_on])


def test_reversal_detector_silent_in_uptrend_with_cooldown() -> None:
    """Scenario 2 boundary: pure uptrend stays silent regardless of cooldown."""
    close = np.linspace(100.0, 200.0, 100, dtype=np.float64)
    for cd in (0, 4, 8):
        risk_off = compute_reversal_risk_off_1d(
            close, dd_window=30, dd_threshold=0.05, mom_fast=10, mom_slow=30,
            persistence_bars=1, recovery_cooldown_bars=cd,
        )
        assert not risk_off.any(), f"no risk-off in uptrend with cd={cd}"


def test_reversal_detector_cooldown_preserves_causal_shift1() -> None:
    """Scenario 3: cooldown state machine does not introduce look-ahead bias."""
    close = np.full(50, 100.0, dtype=np.float64)
    close[-6:-3] = 100.0
    close[-3:] = [97.0, 93.0, 90.0]
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=10, dd_threshold=0.03, mom_fast=3, mom_slow=10,
        persistence_bars=1, recovery_cooldown_bars=5,
    )
    assert not risk_off[0], "row0 is always False"
    modified_close = close.copy()
    modified_close[-1] = 85.0
    risk_off_modified = compute_reversal_risk_off_1d(
        modified_close, dd_window=10, dd_threshold=0.03, mom_fast=3, mom_slow=10,
        persistence_bars=1, recovery_cooldown_bars=5,
    )
    assert np.array_equal(risk_off[:-1], risk_off_modified[:-1])


def _close_raw_flicker() -> NDArray[np.float64]:
    steady = np.full(20, 100.0, dtype=np.float64)
    drop = np.linspace(100.0, 88.0, 10, dtype=np.float64)
    flicker = np.full(15, 90.0, dtype=np.float64)
    flicker[3] = 100.0
    flicker[8] = 99.0
    tail = np.full(10, 88.0, dtype=np.float64)
    return np.concatenate([steady, drop, flicker, tail]).astype(np.float64)


def test_reversal_detector_cooldown_refreshes_on_raw_flicker_during_active_state() -> None:
    """Scenario 6: raw flicker during active state resets off_run counter."""
    close = _close_raw_flicker()
    cd_bars = 8
    risk_off = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=3, mom_slow=10,
        persistence_bars=1, recovery_cooldown_bars=cd_bars,
    )
    true_idxs = np.where(risk_off)[0]
    if true_idxs.shape[0] > 0:
        last_true = int(true_idxs[-1])
        gap = close.shape[0] - 1 - last_true
        assert gap <= cd_bars, (
            f"risk-off should persist near end due to flicker reset, "
            f"last_true={last_true}, gap={gap}, cd_bars={cd_bars}"
        )


def _reconstruct_pre_shift(arr: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Invert the 1-bar shift: raw[i] = arr[i+1] (arr[0] is always False)."""
    t = arr.shape[0]
    out = np.empty(t, dtype=np.bool_)
    out[-1] = False
    out[:-1] = arr[1:]
    return out


def _compute_state_machine_reference(
    raw: NDArray[np.bool_],
    persistent: NDArray[np.bool_],
    cooldown: int,
    *,
    use_raw_for_exit: bool,
) -> NDArray[np.bool_]:
    s_on = False
    off_run = 0
    t = raw.shape[0]
    state = np.zeros(t, dtype=np.bool_)
    for i in range(t):
        if bool(persistent[i]):
            s_on = True
            off_run = 0
        elif s_on:
            exit_sig = raw[i] if use_raw_for_exit else persistent[i]
            off_run = off_run + 1 if not bool(exit_sig) else 0
            if off_run >= max(cooldown, 1) and not bool(exit_sig):
                s_on = False
        state[i] = s_on
    return np.concatenate([[False], state[:-1]]).astype(np.bool_)


def test_reversal_detector_persistence_and_cooldown_combined_uses_raw_not_persistent_for_exit() -> None:
    """Scenario 7: exit counting uses raw, not persistent.

    Extracts raw and persistent signals via shifted outputs of p=1 and p=3 (both cd=0),
    reconstructs the pre-shift arrays, then runs two reference state machines (raw-based
    vs persistent-based exit). Asserts the actual function (p=3, cd=5) matches raw-based
    reference and diverges from persistent-based reference at the expected point.
    """
    close = np.full(100, 100.0, dtype=np.float64)
    close[20:40] = np.linspace(100.0, 83.0, 20, dtype=np.float64)
    close[40:43] = np.linspace(83.0, 96.0, 3, dtype=np.float64)
    close[43:46] = np.linspace(96.0, 87.0, 3, dtype=np.float64)
    close[46:52] = np.linspace(87.0, 96.0, 6, dtype=np.float64)
    close[52:62] = np.linspace(96.0, 81.0, 10, dtype=np.float64)
    close[62:] = np.linspace(81.0, 100.0, 38, dtype=np.float64)

    raw_shifted = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=3, mom_slow=10,
        persistence_bars=1, recovery_cooldown_bars=0,
    )
    persistent_shifted = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=3, mom_slow=10,
        persistence_bars=3, recovery_cooldown_bars=0,
    )
    raw_signal = _reconstruct_pre_shift(raw_shifted)
    persistent_signal = _reconstruct_pre_shift(persistent_shifted)

    actual = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=3, mom_slow=10,
        persistence_bars=3, recovery_cooldown_bars=5,
    )
    expected_raw = _compute_state_machine_reference(
        raw_signal, persistent_signal, 5, use_raw_for_exit=True,
    )
    expected_persistent = _compute_state_machine_reference(
        raw_signal, persistent_signal, 5, use_raw_for_exit=False,
    )

    assert not np.array_equal(expected_raw, expected_persistent), (
        "fixture must produce divergent reference outputs"
    )
    assert np.array_equal(actual, expected_raw), (
        "implementation must match raw-based exit counting, not persistent-based"
    )


def test_reversal_detector_defends_synthetic_crash_shape() -> None:
    """Scenario 8: synthetic crash shape (ATH → sustained decline) triggers risk-off."""
    close = _close_rise_then_fall()
    legacy_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
        persistence_bars=1, recovery_cooldown_bars=8,
    )
    champion_out = compute_reversal_risk_off_1d(
        close, dd_window=20, dd_threshold=0.12, mom_fast=5, mom_slow=20,
        persistence_bars=3, recovery_cooldown_bars=8,
    )
    assert legacy_out.any(), "legacy params must fire on crash shape"
    assert champion_out.any(), "champion params must fire on crash shape"


class TestSyntheticCrashDefenseVerdict:
    """Gate B: synthetic_crash_defense_verdict 시나리오 테스트."""

    def test_fires_on_crash_shape(self) -> None:
        fires, bars = synthetic_crash_defense_verdict(
            dd_window=20, dd_threshold=0.06, mom_fast=5, mom_slow=20,
            persistence_bars=1, recovery_cooldown_bars=8,
        )
        assert fires is True
        assert bars > 0

    def test_does_not_fire_with_unreachable_threshold(self) -> None:
        fires, bars = synthetic_crash_defense_verdict(dd_threshold=0.99)
        assert fires is False
        assert bars == 0

    def test_propagates_invalid_persistence_bars(self) -> None:
        with pytest.raises(ValueError, match="persistence_bars must be >= 1"):
            synthetic_crash_defense_verdict(persistence_bars=0)
