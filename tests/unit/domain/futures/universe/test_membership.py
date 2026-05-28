from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.domain.futures.universe.membership import (
    MembershipMaskBundle,
    build_membership_mask_bundle,
    canonical_symbol,
    inject_membership_masks_into_maps,
)


def test_membership_mask_symbol_canonicalization() -> None:
    timeline = {date(2025, 1, 1): frozenset({"BTCUSDT"})}
    dt = pd.Series(
        pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-01T04:00:00Z", "2025-01-01T08:00:00Z"],
            utc=True,
        )
    )
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTC/USDT",
        timeline=timeline,
        warmup_bars_required=1,
        raw_kill_signal=np.zeros(len(dt), dtype=np.float64),
    )
    assert canonical_symbol("BTC/USDT") == "BTCUSDT"
    assert np.all(bundle.universe_active_mask == 1.0)
    assert np.all(bundle.universe_entry_warm_mask == 1.0)
    assert np.all(bundle.entry_block_mask == 0.0)


def test_membership_kill_and_entry_warm_masks() -> None:
    timeline = {
        date(2025, 1, 1): frozenset({"BTCUSDT"}),
        date(2025, 4, 1): frozenset(),
    }
    dt = pd.Series(
        pd.to_datetime(
            [
                "2025-03-31T20:00:00Z",
                "2025-04-01T00:00:00Z",
                "2025-04-01T04:00:00Z",
            ],
            utc=True,
        )
    )
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTCUSDT",
        timeline=timeline,
        warmup_bars_required=2,
        raw_kill_signal=np.zeros(len(dt), dtype=np.float64),
    )
    np.testing.assert_array_equal(bundle.universe_active_mask, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(bundle.membership_kill_signal, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(bundle.kill_signal, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(bundle.universe_entry_warm_mask, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(bundle.entry_block_mask, np.array([1.0, 1.0, 1.0]))


def test_inference_masks_fallback_to_trading_when_no_inference_timeline() -> None:
    """inference_timeline 미지정 시 inference mask는 trading mask와 동일해야 한다."""
    # Arrange
    timeline = {date(2025, 1, 1): frozenset({"BTCUSDT"})}
    dt = pd.Series(
        pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z", "2025-03-01T00:00:00Z"],
            utc=True,
        )
    )

    # Act
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTCUSDT",
        timeline=timeline,
        warmup_bars_required=1,
    )

    # Assert
    np.testing.assert_array_equal(bundle.inference_active_mask, bundle.universe_active_mask)
    np.testing.assert_array_equal(bundle.inference_entry_warm_mask, bundle.universe_entry_warm_mask)


def test_inference_masks_differ_when_inference_timeline_is_wider() -> None:
    """inference_timeline이 trading_timeline보다 넓으면 inference mask가 더 많이 active해야 한다."""
    # Arrange — trading: Q1만, inference: Q1+Q2
    trading_timeline = {date(2025, 1, 1): frozenset({"BTCUSDT"})}
    inference_timeline = {
        date(2025, 1, 1): frozenset({"BTCUSDT"}),
        date(2025, 4, 1): frozenset({"BTCUSDT"}),
    }
    dt = pd.Series(
        pd.to_datetime(
            [
                "2025-02-01T00:00:00Z",  # Q1 — trading active
                "2025-05-01T00:00:00Z",  # Q2 — trading inactive, inference active
            ],
            utc=True,
        )
    )

    # Act
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTCUSDT",
        timeline=trading_timeline,
        warmup_bars_required=1,
        inference_timeline=inference_timeline,
    )

    # Assert — trading Q2는 0, inference Q2는 1
    np.testing.assert_array_equal(bundle.universe_active_mask, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(bundle.inference_active_mask, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(bundle.inference_entry_warm_mask, np.array([1.0, 1.0]))


def test_inference_masks_warmup_applies_to_inference_timeline() -> None:
    """inference_timeline에도 warmup_bars_required가 적용되어야 한다."""
    # Arrange — inference_timeline에서 Q2에 처음 진입, warmup=2 필요
    trading_timeline = {date(2025, 1, 1): frozenset({"BTCUSDT"})}
    inference_timeline = {date(2025, 4, 1): frozenset({"BTCUSDT"})}
    dt = pd.Series(
        pd.to_datetime(
            [
                "2025-04-01T00:00:00Z",  # idx0: Q2 진입 첫 번째 바 — warmup 미충족
                "2025-04-02T00:00:00Z",  # idx1: Q2 두 번째 바 — warmup 충족
            ],
            utc=True,
        )
    )

    # Act
    bundle = build_membership_mask_bundle(
        datetimes=dt,
        symbol="BTCUSDT",
        timeline=trading_timeline,
        warmup_bars_required=2,
        inference_timeline=inference_timeline,
    )

    # Assert — 첫 바는 warmup 미충족, 두 번째 바는 충족
    np.testing.assert_array_equal(bundle.inference_active_mask, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(bundle.inference_entry_warm_mask, np.array([0.0, 1.0]))


def test_inject_membership_masks_into_maps_adds_inference_columns() -> None:
    """inject_membership_masks_into_maps가 inference_active_mask/inference_entry_warm_mask를 주입해야 한다."""
    # Arrange
    timeline = {date(2025, 1, 1): frozenset({"BTCUSDT"})}
    dt = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"], utc=True)
    frame = pd.DataFrame({"datetime": dt, "close": [100.0, 101.0]})
    data_maps: dict[str, dict[str, object]] = {"BTCUSDT": {"4h": frame.copy()}}
    oos_data_maps: dict[str, dict[str, object]] = {"BTCUSDT": {"4h": frame.copy()}}

    # Act
    inject_membership_masks_into_maps(
        data_maps=data_maps,  # type: ignore[arg-type]
        oos_data_maps=oos_data_maps,  # type: ignore[arg-type]
        symbols=["BTCUSDT"],
        tf="4h",
        timeline=timeline,
        warmup_bars_required=1,
    )

    # Assert
    injected = data_maps["BTCUSDT"]["4h"]
    assert isinstance(injected, pd.DataFrame)
    assert "inference_active_mask" in injected.columns
    assert "inference_entry_warm_mask" in injected.columns
    np.testing.assert_array_equal(injected["inference_active_mask"].to_numpy(), np.array([1.0, 1.0]))
