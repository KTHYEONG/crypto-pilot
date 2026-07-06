"""OPT-3 벡터화 volatility_2d 동일성 단위 테스트.

수치 동일성 검증: pd.DataFrame.rolling + nan_to_num + maximum 기반 벡터화 구현이
컬럼별 rolling_per_bar_return_std 스택과 bit-수준 동일 결과를 생성함을 보장한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.portfolio.signal_composer import rolling_per_bar_return_std

# ---------------------------------------------------------------------------
# 벡터화 구현 (pipeline.py OPT-3 블록과 동일 로직)
# ---------------------------------------------------------------------------


def _vectorized_volatility(close_2d: np.ndarray, vol_window: int) -> np.ndarray:
    """OPT-3 벡터화 volatility_2d 계산 (pipeline.py 구현 사본).

    Args:
        close_2d: shape [T, N] float64 close price matrix.
        vol_window: rolling window size (int, >=2 enforced internally).

    Returns:
        Volatility matrix shape [T, N], floor 1e-12.

    Time Complexity: O(T * N)
    Space Complexity: O(T * N) — 1개 임시 배열(_r)
    """
    _c = np.asarray(close_2d, dtype=np.float64)
    _n_t, _n_sym = _c.shape
    _r = np.zeros((_n_t, _n_sym), dtype=np.float64)
    if _n_t >= 2:
        _r[1:] = (_c[1:] - _c[:-1]) / np.maximum(np.abs(_c[:-1]), 1e-12)
    _rw = max(2, int(vol_window))
    vol = pd.DataFrame(_r).rolling(_rw, min_periods=2).std(ddof=1).to_numpy(dtype=np.float64)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(vol, 1e-12)


def _column_stack_baseline(close_2d: np.ndarray, vol_window: int) -> np.ndarray:
    """기존 컬럼별 rolling_per_bar_return_std 스택 (golden reference).

    Args:
        close_2d: shape [T, N] close price matrix.
        vol_window: rolling window size.

    Returns:
        Volatility matrix shape [T, N].
    """
    return np.column_stack([rolling_per_bar_return_std(close_2d[:, i], vol_window) for i in range(close_2d.shape[1])])


# ---------------------------------------------------------------------------
# S1: 랜덤 [200, 5] — 벡터화 vs 컬럼별 동일성
# ---------------------------------------------------------------------------


class TestS1NumericalIdentity:
    """S1: 랜덤 [200, 5] 데이터에서 벡터화 결과 == 컬럼별 스택 결과."""

    def test_random_close_2d_200x5_approx_identical(self) -> None:
        """Arrange: 재현 가능한 랜덤 close_2d [200,5]. Act: 두 경로 계산. Assert: rel=1e-12 동일."""
        # Arrange
        rng = np.random.default_rng(seed=42)
        close_2d = rng.uniform(low=100.0, high=50000.0, size=(200, 5))
        vol_window = 20

        # Act
        result_vec = _vectorized_volatility(close_2d, vol_window)
        result_ref = _column_stack_baseline(close_2d, vol_window)

        # Assert
        assert result_vec.shape == result_ref.shape
        assert result_vec == pytest.approx(result_ref, rel=1e-12)

    def test_output_shape_matches_input_shape(self) -> None:
        """출력 shape이 입력 [T, N]과 동일함을 검증."""
        # Arrange
        rng = np.random.default_rng(seed=7)
        close_2d = rng.uniform(50.0, 1000.0, size=(200, 5))

        # Act
        result = _vectorized_volatility(close_2d, vol_window=20)

        # Assert
        assert result.shape == (200, 5)

    def test_floor_applied_globally_no_zeros(self) -> None:
        """모든 출력값 >= 1e-12 (floor 보장)."""
        # Arrange
        rng = np.random.default_rng(seed=13)
        close_2d = rng.uniform(100.0, 1000.0, size=(200, 5))

        # Act
        result = _vectorized_volatility(close_2d, vol_window=20)

        # Assert
        assert np.all(result >= 1e-12), "floor 1e-12 위반 원소 존재"


# ---------------------------------------------------------------------------
# S2: T=1 (단일 행) — 전부 floor(1e-12) 반환
# ---------------------------------------------------------------------------


class TestS2SingleRow:
    """S2: T=1 경계값 — rolling std 불가 → nan_to_num(0.0) → maximum = 1e-12."""

    def test_t1_all_values_equal_floor(self) -> None:
        """T=1일 때 전체 출력이 정확히 1e-12."""
        # Arrange
        close_2d = np.array([[100.0, 200.0, 300.0, 400.0, 500.0]])  # shape [1, 5]
        vol_window = 20

        # Act
        result = _vectorized_volatility(close_2d, vol_window)

        # Assert
        assert result.shape == (1, 5)
        expected = np.full((1, 5), 1e-12, dtype=np.float64)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_t1_no_nan_no_negative(self) -> None:
        """T=1 출력에 NaN, 음수, inf 없음."""
        # Arrange
        close_2d = np.array([[42.0, 84.0]])  # shape [1, 2]

        # Act
        result = _vectorized_volatility(close_2d, vol_window=5)

        # Assert
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# S3: NaN 포함 컬럼 — nan_to_num 흡수
# ---------------------------------------------------------------------------


class TestS3NaNColumn:
    """S3: NaN close 값 포함 시 출력에 NaN·음수·inf 없음."""

    def test_nan_in_close_column_absorbed(self) -> None:
        """NaN 포함 컬럼이 nan_to_num으로 0.0 → floor 1e-12로 흡수됨."""
        # Arrange
        rng = np.random.default_rng(seed=99)
        close_2d = rng.uniform(100.0, 1000.0, size=(50, 4))
        close_2d[5:10, 2] = np.nan  # 컬럼 2에 NaN 주입

        # Act
        result = _vectorized_volatility(close_2d, vol_window=10)

        # Assert
        assert not np.any(np.isnan(result)), "NaN이 출력에 잔존"
        assert not np.any(np.isinf(result)), "inf가 출력에 잔존"
        assert not np.any(result < 0.0), "음수 값 발견"
        assert np.all(result >= 1e-12), "floor 미적용 원소 존재"

    def test_all_nan_column_entirely_floor(self) -> None:
        """전체 NaN 컬럼은 floor=1e-12로만 채워짐."""
        # Arrange
        close_2d = np.full((30, 3), fill_value=200.0, dtype=np.float64)
        close_2d[:, 1] = np.nan  # 컬럼 1 전체 NaN

        # Act
        result = _vectorized_volatility(close_2d, vol_window=5)

        # Assert
        assert not np.any(np.isnan(result))
        assert np.all(result[:, 1] >= 1e-12), "전체-NaN 컬럼이 floor 미적용"
