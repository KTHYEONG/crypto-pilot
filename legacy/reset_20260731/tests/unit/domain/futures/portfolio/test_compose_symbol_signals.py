"""Unit tests for compose_symbol_signals in signal_composer.py.

Coverage targets:
  TI4 — Happy path: 2 symbols, valid=True, raw_mu, beta_btc 검증
  TI5 — QC reject: n_obs < min_obs → valid=False
  TI6 — beta_vs_market_1d=None → beta_btc is None
  TI7 — empty events → empty dict 반환
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.domain.futures.portfolio.signal_composer import compose_symbol_signals
from src.domain.futures.strategy.cs_rank import VOL_FLOOR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_output(
    symbols_list: list[str],
    net_bps_list: list[float],
) -> Any:
    """CandidateModelOutput stub: .events + .expected_net_bps."""
    events = pd.DataFrame({"symbol": symbols_list})
    expected_net_bps = np.array(net_bps_list, dtype=np.float64)
    return SimpleNamespace(events=events, expected_net_bps=expected_net_bps)


def _flat_close_2d(n_bars: int, n_syms: int, price: float = 50_000.0) -> NDArray[np.float64]:
    """상수 종가 행렬 [n_bars, n_syms]."""
    return np.full((n_bars, n_syms), price, dtype=np.float64)


# ---------------------------------------------------------------------------
# TI4: Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    """TI4 — 2 symbols, 충분한 obs, gate 느슨하게(t_stat_floor=0.0)."""

    def test_both_symbols_present(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 1.0, 3.0],
        )
        close_2d = _flat_close_2d(100, 2)
        symbols = ("BTC", "ETH")
        beta = np.array([1.0, 0.8], dtype=np.float64)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=symbols,
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=beta,
        )

        # Assert
        assert "BTC" in result
        assert "ETH" in result

    def test_btc_raw_mu(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 1.0, 3.0],
        )
        close_2d = _flat_close_2d(100, 2)
        beta = np.array([1.0, 0.8], dtype=np.float64)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=beta,
        )

        # Assert — mean(5.0, 7.0) == 6.0
        assert result["BTC"].raw_mu == pytest.approx(6.0)

    def test_eth_raw_mu(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 1.0, 3.0],
        )
        close_2d = _flat_close_2d(100, 2)
        beta = np.array([1.0, 0.8], dtype=np.float64)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=beta,
        )

        # Assert — mean(2.0, 1.0, 3.0) == 2.0
        assert result["ETH"].raw_mu == pytest.approx(2.0)

    def test_volatility_above_floor(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 1.0, 3.0],
        )
        close_2d = _flat_close_2d(100, 2)
        beta = np.array([1.0, 0.8], dtype=np.float64)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=beta,
        )

        # Assert
        assert result["BTC"].volatility >= VOL_FLOOR
        assert result["ETH"].volatility >= VOL_FLOOR

    def test_beta_btc_values(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 1.0, 3.0],
        )
        close_2d = _flat_close_2d(100, 2)
        beta = np.array([1.0, 0.8], dtype=np.float64)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=beta,
        )

        # Assert
        assert result["BTC"].beta_btc == pytest.approx(1.0)
        assert result["ETH"].beta_btc == pytest.approx(0.8)

    def test_valid_true_when_gate_loose(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 1.0, 3.0],
        )
        close_2d = _flat_close_2d(100, 2)
        beta = np.array([1.0, 0.8], dtype=np.float64)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=beta,
        )

        # Assert
        assert result["BTC"].valid is True


# ---------------------------------------------------------------------------
# TI5: QC reject — n_obs < min_obs
# ---------------------------------------------------------------------------


class TestQCReject:
    """TI5 — 1건 obs로 min_obs=5 → valid=False."""

    def test_valid_false_when_insufficient_obs(self) -> None:
        # Arrange
        model_out = _make_model_output(["BTC"], [10.0])
        close_2d = _flat_close_2d(100, 1)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC",),
            tf="4h",
            min_obs=5,
            t_stat_floor=0.0,
        )

        # Assert
        assert "BTC" in result
        assert result["BTC"].valid is False

    def test_n_obs_recorded_correctly(self) -> None:
        # Arrange
        model_out = _make_model_output(["BTC"], [10.0])
        close_2d = _flat_close_2d(100, 1)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC",),
            tf="4h",
            min_obs=5,
            t_stat_floor=0.0,
        )

        # Assert
        assert result["BTC"].n_obs == 1


# ---------------------------------------------------------------------------
# TI6: beta_vs_market_1d=None → beta_btc is None
# ---------------------------------------------------------------------------


class TestNoBeta:
    """TI6 — beta_vs_market_1d=None이면 모든 심볼의 beta_btc is None."""

    def test_beta_btc_is_none_when_no_market_beta(self) -> None:
        # Arrange
        model_out = _make_model_output(
            ["BTC", "BTC", "ETH", "ETH"],
            [5.0, 7.0, 2.0, 4.0],
        )
        close_2d = _flat_close_2d(100, 2)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
            beta_vs_market_1d=None,
        )

        # Assert
        assert all(v.beta_btc is None for v in result.values())


# ---------------------------------------------------------------------------
# TI7: empty events → empty dict
# ---------------------------------------------------------------------------


class TestEmptyEvents:
    """TI7 — events 빈 DataFrame → 빈 dict 반환."""

    def test_empty_events_returns_empty_dict(self) -> None:
        # Arrange
        empty_events = pd.DataFrame(columns=["symbol"])
        model_out = SimpleNamespace(
            events=empty_events,
            expected_net_bps=np.array([], dtype=np.float64),
        )
        close_2d = _flat_close_2d(100, 2)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
        )

        # Assert
        assert result == {}

    def test_missing_symbol_column_returns_empty_dict(self) -> None:
        # Arrange — 'symbol' 컬럼 없음
        bad_events = pd.DataFrame({"price": [100.0]})
        model_out = SimpleNamespace(
            events=bad_events,
            expected_net_bps=np.array([1.0], dtype=np.float64),
        )
        close_2d = _flat_close_2d(100, 2)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
        )

        # Assert
        assert result == {}

    def test_length_mismatch_returns_empty_dict(self) -> None:
        # Arrange — net_bps 길이 != events 길이
        events = pd.DataFrame({"symbol": ["BTC", "ETH"]})
        model_out = SimpleNamespace(
            events=events,
            expected_net_bps=np.array([1.0], dtype=np.float64),  # 길이 불일치
        )
        close_2d = _flat_close_2d(100, 2)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC", "ETH"),
            tf="4h",
            min_obs=2,
            t_stat_floor=0.0,
        )

        # Assert
        assert result == {}

    def test_numpy_last_vol_calculation_correctness(self) -> None:
        # Arrange
        events = pd.DataFrame({"symbol": ["BTC"]})
        model_out = SimpleNamespace(
            events=events,
            expected_net_bps=np.array([10.0], dtype=np.float64),
        )
        # 10개 시점의 close 데이터
        close_2d = np.linspace(100.0, 110.0, 10, dtype=np.float64).reshape(10, 1)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC",),
            tf="4h",
            min_obs=1,
            t_stat_floor=0.0,
        )

        # Assert
        assert "BTC" in result
        assert result["BTC"].volatility > 0.0


class TestConstantSignal:
    """상수 신호 입력 시 t-stat이 폭발하지 않고 0.0으로 반환되는지 검증."""

    def test_constant_net_bps_tstat_is_zero(self) -> None:
        # Arrange - 100개의 관측치 모두 17.056 상수로 구성
        events = pd.DataFrame({"symbol": ["BTC"] * 100})
        model_out = SimpleNamespace(
            events=events,
            expected_net_bps=np.full(100, 17.056, dtype=np.float64),
        )
        close_2d = _flat_close_2d(100, 1)

        # Act
        result = compose_symbol_signals(
            model_output=model_out,
            close_2d=close_2d,
            symbols=("BTC",),
            tf="4h",
            min_obs=4,
            t_stat_floor=0.0,
        )

        # Assert
        assert "BTC" in result
        assert result["BTC"].t_stat == 0.0
