from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.domain.futures.universe.membership import (
    _normalize_timeline,
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


# ---------------------------------------------------------------------------
# OPT-2 신규 테스트 시나리오
# ---------------------------------------------------------------------------


def _make_dt_series(dates: list[str]) -> pd.Series:
    """UTC datetime Series 생성 헬퍼."""
    return pd.Series(pd.to_datetime(dates, utc=True))


def _make_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"datetime": pd.to_datetime(dates, utc=True), "close": [100.0] * len(dates)})


# S1: inject 경로(pre-normalized) vs build 직접 호출(internal fallback) → 전 필드 동일
def test_opt2_s1_inject_vs_direct_build_results_identical() -> None:
    """inject_membership_masks_into_maps와 build_membership_mask_bundle 직접 호출 결과가 동일해야 한다."""
    # Arrange
    timeline = {
        date(2025, 1, 1): frozenset({"BTCUSDT"}),
        date(2025, 4, 1): frozenset({"BTCUSDT", "ETHUSDT"}),
        date(2025, 7, 1): frozenset({"ETHUSDT"}),
    }
    dates = [
        "2025-01-15T00:00:00Z",
        "2025-02-15T00:00:00Z",
        "2025-04-15T00:00:00Z",
        "2025-07-15T00:00:00Z",
    ]
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    frames_inject = {sym: {"4h": _make_frame(dates)} for sym in symbols}
    oos_inject = {sym: {"4h": _make_frame(dates)} for sym in symbols}

    # Act — inject 경로
    inject_membership_masks_into_maps(
        data_maps=frames_inject,  # type: ignore[arg-type]
        oos_data_maps=oos_inject,  # type: ignore[arg-type]
        symbols=symbols,
        tf="4h",
        timeline=timeline,
        warmup_bars_required=1,
    )

    # Assert — 각 심볼별 build 직접 호출 결과와 비교
    for sym in symbols:
        dt_ser = _make_dt_series(dates)
        expected = build_membership_mask_bundle(
            datetimes=dt_ser,
            symbol=sym,
            timeline=timeline,
            warmup_bars_required=1,
        )
        injected_frame = frames_inject[sym]["4h"]
        assert isinstance(injected_frame, pd.DataFrame)
        np.testing.assert_array_equal(
            injected_frame["universe_active_mask"].to_numpy(),
            expected.universe_active_mask,
            err_msg=f"{sym}: universe_active_mask mismatch",
        )
        np.testing.assert_array_equal(
            injected_frame["universe_entry_warm_mask"].to_numpy(),
            expected.universe_entry_warm_mask,
            err_msg=f"{sym}: universe_entry_warm_mask mismatch",
        )
        np.testing.assert_array_equal(
            injected_frame["entry_block_mask"].to_numpy(),
            expected.entry_block_mask,
            err_msg=f"{sym}: entry_block_mask mismatch",
        )
        np.testing.assert_array_equal(
            injected_frame["inference_active_mask"].to_numpy(),
            expected.inference_active_mask,
            err_msg=f"{sym}: inference_active_mask mismatch",
        )
        np.testing.assert_array_equal(
            injected_frame["inference_entry_warm_mask"].to_numpy(),
            expected.inference_entry_warm_mask,
            err_msg=f"{sym}: inference_entry_warm_mask mismatch",
        )


# S2: 빈 timeline → early return, 어떤 컬럼도 주입되지 않는다
def test_opt2_s2_empty_timeline_returns_early_no_columns_written() -> None:
    """빈 timeline 전달 시 inject는 early return하고 컬럼을 쓰지 않아야 한다."""
    # Arrange
    frame = _make_frame(["2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"])
    data_maps = {"BTCUSDT": {"4h": frame}}
    oos_data_maps: dict[str, dict[str, object]] = {"BTCUSDT": {"4h": frame.copy()}}

    # Act
    inject_membership_masks_into_maps(
        data_maps=data_maps,  # type: ignore[arg-type]
        oos_data_maps=oos_data_maps,  # type: ignore[arg-type]
        symbols=["BTCUSDT"],
        tf="4h",
        timeline={},
        warmup_bars_required=1,
    )

    # Assert — universe_active_mask 미주입 확인
    assert "universe_active_mask" not in data_maps["BTCUSDT"]["4h"].columns


# S3: inference_timeline=None → inference_active_mask == universe_active_mask
def test_opt2_s3_no_inference_timeline_masks_equal_trading_masks() -> None:
    """inference_timeline 미지정 시 inference_active_mask는 universe_active_mask와 동일해야 한다."""
    # Arrange
    timeline = {
        date(2025, 1, 1): frozenset({"BTCUSDT"}),
        date(2025, 4, 1): frozenset(),  # Q2 비활성
    }
    dates = [
        "2025-02-01T00:00:00Z",  # Q1 active
        "2025-05-01T00:00:00Z",  # Q2 inactive
    ]
    frame = _make_frame(dates)
    data_maps = {"BTCUSDT": {"4h": frame}}
    oos_data_maps: dict[str, dict[str, object]] = {"BTCUSDT": {"4h": _make_frame(dates)}}

    # Act
    inject_membership_masks_into_maps(
        data_maps=data_maps,  # type: ignore[arg-type]
        oos_data_maps=oos_data_maps,  # type: ignore[arg-type]
        symbols=["BTCUSDT"],
        tf="4h",
        timeline=timeline,
        warmup_bars_required=1,
        inference_timeline=None,
    )

    # Assert
    result_frame = data_maps["BTCUSDT"]["4h"]
    assert isinstance(result_frame, pd.DataFrame)
    np.testing.assert_array_equal(
        result_frame["inference_active_mask"].to_numpy(),
        result_frame["universe_active_mask"].to_numpy(),
    )
    np.testing.assert_array_equal(
        result_frame["inference_entry_warm_mask"].to_numpy(),
        result_frame["universe_entry_warm_mask"].to_numpy(),
    )


# _normalize_timeline 단위 테스트
def test_normalize_timeline_canonicalizes_keys_and_symbols() -> None:
    """_normalize_timeline은 날짜를 분기 시작일로, 심볼을 canonical 형식으로 정규화해야 한다."""
    # Arrange — 분기 중간 날짜 및 비canonical 심볼
    raw: dict[date, frozenset[str]] = {
        date(2025, 2, 15): frozenset({"btc/usdt", "ETH_USDT"}),
        date(2025, 5, 20): frozenset({"sol/usdt"}),
    }

    # Act
    result = _normalize_timeline(raw)

    # Assert
    assert date(2025, 1, 1) in result  # 2025-02-15 → Q1 시작
    assert date(2025, 4, 1) in result  # 2025-05-20 → Q2 시작
    assert result[date(2025, 1, 1)] == frozenset({"BTCUSDT", "ETHUSDT"})
    assert result[date(2025, 4, 1)] == frozenset({"SOLUSDT"})
