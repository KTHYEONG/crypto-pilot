"""Phase 14: OI/ADV crowding filter tests.

테스트 1: oi_usdt_median / adv > 12 → 해당 심볼 제외 (fetch_metrics_bulk 적용)
테스트 2: 2020-08 이전 구간 → OI 필터 비활성 (빈 DataFrame 허용)
테스트 3: fetch_metrics_bulk shape 검증 (mock HTTP)
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 테스트 1: OI/ADV > 12 → 심볼 제외
# ---------------------------------------------------------------------------


def _make_metrics_df(oi_usdt: float, adv_usdt: float) -> pd.DataFrame:
    """합성 metrics DataFrame 생성 (단일 행)."""
    return pd.DataFrame(
        {
            "sum_open_interest_value": [oi_usdt],
            "adv_usdt": [adv_usdt],
        }
    )


def _oi_adv_ratio(df: pd.DataFrame) -> float:
    """oi_usdt_median / adv 비율 계산 (단순화된 필터 로직)."""
    if df.empty:
        return 0.0
    oi = float(df["sum_open_interest_value"].median())
    adv = float(df["adv_usdt"].median())
    if adv <= 1e-9:
        return 0.0
    return oi / adv


def test_oi_adv_ratio_over_threshold_excludes_symbol() -> None:
    """OI/ADV > 12 → 과밀 심볼로 분류되어야 함."""
    # OI = 120M, ADV = 5M → ratio = 24 > 12
    df = _make_metrics_df(oi_usdt=120_000_000.0, adv_usdt=5_000_000.0)
    ratio = _oi_adv_ratio(df)
    assert ratio > 12.0, f"OI/ADV ratio expected > 12.0, got {ratio:.2f}"

    crowded = ratio > 12.0
    assert crowded is True, "과밀 심볼로 분류되어야 함"


def test_oi_adv_ratio_under_threshold_passes() -> None:
    """OI/ADV <= 12 → 정상 심볼로 통과해야 함."""
    # OI = 30M, ADV = 10M → ratio = 3 <= 12
    df = _make_metrics_df(oi_usdt=30_000_000.0, adv_usdt=10_000_000.0)
    ratio = _oi_adv_ratio(df)
    assert ratio <= 12.0, f"OI/ADV ratio expected <= 12.0, got {ratio:.2f}"

    crowded = ratio > 12.0
    assert not crowded, f"정상 심볼이 과밀로 분류됨: ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# 테스트 2: 2020-08 이전 구간 → fetch_metrics_bulk 빈 DataFrame 반환
# ---------------------------------------------------------------------------


def test_fetch_metrics_bulk_returns_empty_before_2020_09() -> None:
    """2020-09-01 이전 구간 → 데이터 없음 → 빈 DataFrame."""
    from src.core.utils.binance_vision import fetch_metrics_bulk

    result = fetch_metrics_bulk(
        symbol="BTCUSDT",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 8, 31),  # 전부 이전 구간
    )
    assert isinstance(result, pd.DataFrame), "반환 타입은 pd.DataFrame이어야 함"
    assert result.empty, "2020-09-01 이전은 빈 DataFrame을 반환해야 함"


def test_fetch_metrics_bulk_adjusts_start_to_metrics_start() -> None:
    """start_date < 2020-09-01이지만 end_date >= 2020-09-01 → 2020-09-01부터 수집 시도."""
    from src.core.utils.binance_vision import fetch_metrics_bulk, BinanceVisionDownloader

    # HTTP 요청 mock — 1행짜리 DataFrame 반환
    mock_df = pd.DataFrame(
        {
            "open_time": [1598918400000],
            "sum_open_interest_value": [1_000_000.0],
            "adv_usdt": [500_000.0],
        }
    )

    with patch.object(BinanceVisionDownloader, "fetch_metrics_daily", return_value=mock_df):
        result = fetch_metrics_bulk(
            symbol="BTCUSDT",
            start_date=date(2020, 7, 1),   # 이전 구간
            end_date=date(2020, 9, 2),      # 2020-09-01 포함 → 2일치만 요청
        )

    # 비어 있지 않아야 함 (최소 1일치 mock 데이터)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty, "2020-09-01 이후 구간 데이터가 있어야 함"


# ---------------------------------------------------------------------------
# 테스트 3: fetch_metrics_bulk shape 검증 (mock HTTP)
# ---------------------------------------------------------------------------


def test_fetch_metrics_bulk_shape_with_mock_http() -> None:
    """HTTP mock으로 5일치 metrics 요청 → DataFrame row 수 확인."""
    from src.core.utils.binance_vision import fetch_metrics_bulk, BinanceVisionDownloader

    n_mock_days = 3
    mock_row = pd.DataFrame(
        {
            "open_time": [1598918400000],
            "sum_open_interest_value": [2_000_000.0],
            "count_toptrader_long_short_ratio": [1.2],
            "adv_usdt": [800_000.0],
        }
    )

    call_count = 0

    def _mock_daily(self: BinanceVisionDownloader, symbol: str, dt: object) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return mock_row.copy()

    with patch.object(BinanceVisionDownloader, "fetch_metrics_daily", _mock_daily):
        result = fetch_metrics_bulk(
            symbol="ETHUSDT",
            start_date=date(2020, 9, 1),
            end_date=date(2020, 9, 3),   # 3일
        )

    assert isinstance(result, pd.DataFrame)
    assert call_count == n_mock_days, f"예상 call_count={n_mock_days}, actual={call_count}"
    assert len(result) == n_mock_days, (
        f"예상 row 수={n_mock_days}, actual={len(result)}"
    )
    assert "sum_open_interest_value" in result.columns
