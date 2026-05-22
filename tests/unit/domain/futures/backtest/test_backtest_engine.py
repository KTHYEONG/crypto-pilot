from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.domain.futures.backtest.engine as backtest_engine_mod
from src.core.settings import FUTURES_DATA_DIR
from src.domain.futures.backtest.engine import (
    PortfolioBacktestEngine,
)
from src.domain.futures.backtest.preparation import prepare_backtest_inputs


def test_backtest_engine_multi_symbol_mock() -> None:
    """Tests the PortfolioBacktestEngine with mock data."""
    n_bars = 100
    n_syms = 2
    symbols = ["BTC/USDT", "ETH/USDT"]

    # Mock aligned data
    aligned_data = {
        "close": np.ones((n_bars, n_syms)) * 100.0,
        "high": np.ones((n_bars, n_syms)) * 101.0,
        "low": np.ones((n_bars, n_syms)) * 99.0,
        "open": np.ones((n_bars, n_syms)) * 100.0,
        "atr": np.ones((n_bars, n_syms)) * 2.0,
        "funding_rate_sum": np.zeros((n_bars, n_syms)),
        "kill_signal": np.zeros((n_bars, n_syms)),
        "target_weights": np.ones((n_bars, n_syms)) * 0.3,
        "xs_score_long": np.ones((n_bars, n_syms)) * 0.5,
        "xs_score_short": np.zeros((n_bars, n_syms)),
        "hmm_prob_crisis": np.zeros((n_bars, n_syms)),
        "hmm_modulator_long": np.ones((n_bars, n_syms)),
        "hmm_modulator_short": np.ones((n_bars, n_syms)),
        # Additional required columns for alignment/engine
        "entry_upper": np.zeros((n_bars, n_syms)),
        "entry_lower": np.ones((n_bars, n_syms)) * 999999.0,
        "trend_direction": np.ones((n_bars, n_syms)),
        "strength_filter": np.ones((n_bars, n_syms)),
        "slot_rank_score": np.ones((n_bars, n_syms)),
        "ml_calib_prob": np.zeros((n_bars, n_syms)),
    }

    strategy_params = {
        "K_LONG": 1,
        "K_SHORT": 1,
        "REBALANCE_BARS": 6,
        "ATR_MULT": 3.0,
        "TRAIL_MULT": 3.0,
        "MAX_EXPOSURE_PER_COIN": 1.5,
        "MAX_EXPOSURE": 0.8,
        "LEVERAGE": 5.0,
    }

    engine = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params=strategy_params,
        initial_balance=10000.0
    )
    
    trades_df, equity, final_bal, diag = engine.run()
    
    assert isinstance(trades_df, pd.DataFrame)
    assert isinstance(equity, np.ndarray)
    assert isinstance(final_bal, float)
    assert isinstance(diag, (dict, np.ndarray))
    assert len(equity) == n_bars


def test_aggregate_1h_to_4h_ohlcv_contract() -> None:
    """4 x 1h 봉이 1 x 4h 봉으로 정확히 집계되는지 검증."""
    aligned_data = {
        "open": np.array([[10.0], [11.0], [12.0], [13.0]], dtype=np.float64),
        "high": np.array([[11.0], [12.0], [15.0], [14.0]], dtype=np.float64),
        "low": np.array([[9.0], [8.0], [10.0], [11.0]], dtype=np.float64),
        "close": np.array([[10.5], [11.5], [12.5], [13.5]], dtype=np.float64),
        "volume": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
        "funding_rate_sum": np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64),
        "kill_signal": np.array([[0.0], [0.0], [1.0], [0.0]], dtype=np.float64),
        "atr": np.array([[2.0], [np.nan], [2.2], [np.nan]], dtype=np.float64),
    }
    prepared = prepare_backtest_inputs(
        aligned_data,
        {"TIMEFRAME": "4h", "DATA_TIMEFRAME": "1h"},
    )
    out = prepared.aligned_data
    assert out["open"][0, 0] == pytest.approx(10.0)
    assert out["high"][0, 0] == pytest.approx(15.0)
    assert out["low"][0, 0] == pytest.approx(8.0)
    assert out["close"][0, 0] == pytest.approx(13.5)
    assert out["volume"][0, 0] == pytest.approx(10.0)
    assert out["funding_rate_sum"][0, 0] == pytest.approx(1.0)
    assert out["kill_signal"][0, 0] == pytest.approx(1.0)
    assert out["atr"][0, 0] == pytest.approx(2.2)


def test_backtest_engine_uses_1h_base_for_4h_and_accepts_volume() -> None:
    """TIMEFRAME=4h + DATA_TIMEFRAME=1h에서 집계 후 엔진이 실행되는지 검증."""
    n_bars = 8
    aligned_data = {
        "close": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "high": np.full((n_bars, 1), 101.0, dtype=np.float64),
        "low": np.full((n_bars, 1), 99.0, dtype=np.float64),
        "open": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "atr": np.full((n_bars, 1), 2.0, dtype=np.float64),
        "funding_rate_sum": np.zeros((n_bars, 1), dtype=np.float64),
        "kill_signal": np.zeros((n_bars, 1), dtype=np.float64),
        "volume": np.full((n_bars, 1), 5000.0, dtype=np.float64),
        "target_weights": np.full((n_bars, 1), 0.3, dtype=np.float64),
        "xs_score_long": np.full((n_bars, 1), 0.5, dtype=np.float64),
        "xs_score_short": np.zeros((n_bars, 1), dtype=np.float64),
    }
    params = {
        "TIMEFRAME": "4h",
        "DATA_TIMEFRAME": "1h",
        "REBALANCE_BARS": 1,
        "ATR_MULT": 3.0,
        "TRAIL_MULT": 3.0,
        "MAX_EXPOSURE_PER_COIN": 1.5,
        "MAX_EXPOSURE": 0.8,
        "LEVERAGE": 3.0,
    }
    engine = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=["BTC/USDT"],
        strategy_params=params,
        initial_balance=10000.0,
    )
    trades_df, equity, final_bal, _diag = engine.run()
    assert len(equity) == 2  # 8 x 1h -> 2 x 4h
    assert isinstance(trades_df, pd.DataFrame)
    assert np.isfinite(final_bal)


def test_backtest_engine_passes_volume_2d_to_execution_sim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """volume이 있을 때 execution_sim 인자로 그대로 전달되는지 검증."""
    n_bars = 12
    n_syms = 2
    volume = np.arange(1, n_bars * n_syms + 1, dtype=np.float64).reshape(n_bars, n_syms)
    aligned_data = {
        "close": np.full((n_bars, n_syms), 100.0, dtype=np.float64),
        "high": np.full((n_bars, n_syms), 101.0, dtype=np.float64),
        "low": np.full((n_bars, n_syms), 99.0, dtype=np.float64),
        "open": np.full((n_bars, n_syms), 100.0, dtype=np.float64),
        "atr": np.full((n_bars, n_syms), 2.0, dtype=np.float64),
        "funding_rate_sum": np.zeros((n_bars, n_syms), dtype=np.float64),
        "kill_signal": np.zeros((n_bars, n_syms), dtype=np.float64),
        "target_weights": np.zeros((n_bars, n_syms), dtype=np.float64),
        "volume": volume,
    }
    captured: dict[str, np.ndarray] = {}

    def _fake_exec(*args: object) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        vol_arg = args[23]
        captured["volume_2d"] = np.asarray(vol_arg, dtype=np.float64)
        close_2d = np.asarray(args[0], dtype=np.float64)
        return (
            np.zeros((0, 10), dtype=np.float64),
            1000.0,
            np.zeros(close_2d.shape[0], dtype=np.float64),
            np.zeros(5, dtype=np.int64),
        )

    monkeypatch.setattr(backtest_engine_mod, "backtest_target_weights_numba", _fake_exec)

    engine = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=["BTC/USDT", "ETH/USDT"],
        strategy_params={"TIMEFRAME": "1h", "REBALANCE_BARS": 1},
        initial_balance=1000.0,
    )
    engine.run()
    np.testing.assert_allclose(captured["volume_2d"], volume)


def test_intrabar_1m_mode_smoke_runs_without_src_intrabar_path() -> None:
    """intrabar_1m 플래그 활성 시에도 현행 coarse 경로로 안정 실행."""
    n_bars = 12
    aligned_data = {
        "close": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "high": np.full((n_bars, 1), 101.0, dtype=np.float64),
        "low": np.full((n_bars, 1), 99.0, dtype=np.float64),
        "open": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "atr": np.full((n_bars, 1), 2.0, dtype=np.float64),
        "funding_rate_sum": np.zeros((n_bars, 1), dtype=np.float64),
        "kill_signal": np.zeros((n_bars, 1), dtype=np.float64),
        "target_weights": np.full((n_bars, 1), 0.2, dtype=np.float64),
    }
    engine = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=["BTC/USDT"],
        strategy_params={
            "TIMEFRAME": "1h",
            "REBALANCE_BARS": 1,
            "FUTURES_EXECUTION_MODE": "intrabar_1m",
        },
        initial_balance=1000.0,
    )
    trades_df, equity, final_bal, _diag = engine.run()
    assert isinstance(trades_df, pd.DataFrame)
    assert len(equity) == n_bars
    assert np.isfinite(final_bal)


def test_intrabar_1m_flag_fallback_matches_coarse_outputs() -> None:
    """Src intrabar 경로 미구현 상태에서 intrabar 플래그 결과는 coarse와 동일."""
    n_bars = 16
    aligned_data = {
        "close": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "high": np.full((n_bars, 1), 101.0, dtype=np.float64),
        "low": np.full((n_bars, 1), 99.0, dtype=np.float64),
        "open": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "atr": np.full((n_bars, 1), 2.0, dtype=np.float64),
        "funding_rate_sum": np.zeros((n_bars, 1), dtype=np.float64),
        "kill_signal": np.zeros((n_bars, 1), dtype=np.float64),
        "target_weights": np.where(np.arange(n_bars).reshape(-1, 1) >= 1, 0.3, 0.0),
    }
    common_params = {"TIMEFRAME": "1h", "REBALANCE_BARS": 2}
    engine_coarse = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=["BTC/USDT"],
        strategy_params={**common_params, "FUTURES_EXECUTION_MODE": "coarse"},
        initial_balance=1000.0,
    )
    engine_intrabar = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=["BTC/USDT"],
        strategy_params={**common_params, "FUTURES_EXECUTION_MODE": "intrabar_1m"},
        initial_balance=1000.0,
    )
    trades_coarse, equity_coarse, final_coarse, _ = engine_coarse.run()
    trades_intrabar, equity_intrabar, final_intrabar, _ = engine_intrabar.run()
    assert final_intrabar == pytest.approx(final_coarse, abs=1e-9)
    np.testing.assert_allclose(equity_intrabar, equity_coarse, atol=1e-9, rtol=0.0)
    assert len(trades_intrabar) == len(trades_coarse)


def test_intrabar_1m_window_mapping_basic_contract() -> None:
    """Decision bar -> 1m window 매핑 기본 계약 검증."""
    aligned_data = {
        "close": np.full((3, 1), 100.0, dtype=np.float64),
        "high": np.full((3, 1), 101.0, dtype=np.float64),
        "low": np.full((3, 1), 99.0, dtype=np.float64),
        "open": np.full((3, 1), 100.0, dtype=np.float64),
        "atr": np.full((3, 1), 2.0, dtype=np.float64),
        "dt_index": np.array([60.0, 120.0, 180.0], dtype=np.float64),
        "exec_dt_index_1m": np.array(
            [60.0, 61.0, 62.0, 120.0, 121.0, 180.0, 181.0], dtype=np.float64
        ),
    }
    prepared = prepare_backtest_inputs(
        aligned_data,
        {"TIMEFRAME": "1h", "FUTURES_EXECUTION_MODE": "intrabar_1m"},
    )
    assert prepared.exec_bar_start_1m_idx is not None
    assert prepared.exec_bar_end_1m_idx is not None
    np.testing.assert_array_equal(
        prepared.exec_bar_start_1m_idx, np.array([0, 3, 5], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        prepared.exec_bar_end_1m_idx, np.array([2, 4, 6], dtype=np.int64)
    )


def test_membership_constraints_are_applied_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Membership kill/entry block이 공통 경로에서 적용되는지 검증."""
    n_bars = 6
    aligned_data = {
        "close": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "high": np.full((n_bars, 1), 101.0, dtype=np.float64),
        "low": np.full((n_bars, 1), 99.0, dtype=np.float64),
        "open": np.full((n_bars, 1), 100.0, dtype=np.float64),
        "atr": np.full((n_bars, 1), 2.0, dtype=np.float64),
        "funding_rate_sum": np.zeros((n_bars, 1), dtype=np.float64),
        "kill_signal": np.zeros((n_bars, 1), dtype=np.float64),
        "membership_kill_signal": np.array(
            [[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64
        ),
        "entry_block_mask": np.array(
            [[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64
        ),
        "target_weights": np.full((n_bars, 1), 0.4, dtype=np.float64),
        "symbol_names": np.asarray(["BTCUSDT"], dtype=object),
    }
    captured: dict[str, np.ndarray] = {}

    def _fake_exec(*args: object) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        captured["kill"] = np.asarray(args[5], dtype=np.float64)
        captured["tw"] = np.asarray(args[6], dtype=np.float64)
        close_2d = np.asarray(args[0], dtype=np.float64)
        return (
            np.zeros((0, 10), dtype=np.float64),
            1000.0,
            np.zeros(close_2d.shape[0], dtype=np.float64),
            np.zeros(5, dtype=np.int64),
        )

    monkeypatch.setattr(backtest_engine_mod, "backtest_target_weights_numba", _fake_exec)
    engine = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=["BTC/USDT"],
        strategy_params={"TIMEFRAME": "1h", "REBALANCE_BARS": 1},
        initial_balance=1000.0,
    )
    _trades_df, _equity, _final_bal, diag = engine.run()
    assert captured["kill"][1, 0] == pytest.approx(1.0)
    assert captured["tw"][0, 0] == pytest.approx(0.0)
    assert captured["tw"][1, 0] == pytest.approx(0.0)
    assert captured["tw"][2, 0] == pytest.approx(0.4)
    assert isinstance(diag, np.ndarray)
    assert diag[2] >= 1.0


def test_intrabar_1m_injection_keys_contract_ready_for_execution() -> None:
    """Intrabar 실행에 필요한 1m 주입 키/매핑이 준비되는지 검증."""
    n_decisions = 3
    n_path = 7
    aligned_data = {
        "close": np.full((n_decisions, 1), 100.0, dtype=np.float64),
        "high": np.full((n_decisions, 1), 101.0, dtype=np.float64),
        "low": np.full((n_decisions, 1), 99.0, dtype=np.float64),
        "open": np.full((n_decisions, 1), 100.0, dtype=np.float64),
        "atr": np.full((n_decisions, 1), 2.0, dtype=np.float64),
        "target_weights": np.zeros((n_decisions, 1), dtype=np.float64),
        "kill_signal": np.zeros((n_decisions, 1), dtype=np.float64),
        "dt_index": np.array([60.0, 120.0, 180.0], dtype=np.float64),
        "exec_dt_index_1m": np.array(
            [60.0, 61.0, 62.0, 120.0, 121.0, 180.0, 181.0], dtype=np.float64
        ),
        "exec_open_1m": np.full((n_path, 1), 100.0, dtype=np.float64),
        "exec_high_1m": np.full((n_path, 1), 101.0, dtype=np.float64),
        "exec_low_1m": np.full((n_path, 1), 99.0, dtype=np.float64),
        "exec_close_1m": np.full((n_path, 1), 100.0, dtype=np.float64),
        "exec_volume_1m": np.full((n_path, 1), 50.0, dtype=np.float64),
    }

    prepared = prepare_backtest_inputs(
        aligned_data,
        {"TIMEFRAME": "1h", "FUTURES_EXECUTION_MODE": "intrabar_1m"},
    )
    assert prepared.execution_mode == "intrabar_1m"
    assert prepared.exec_bar_start_1m_idx is not None
    assert prepared.exec_bar_end_1m_idx is not None
    for key in ("exec_open_1m", "exec_high_1m", "exec_low_1m", "exec_close_1m", "exec_volume_1m"):
        assert key in prepared.aligned_data
        assert prepared.aligned_data[key].shape == (n_path, 1)
    assert "exec_bar_start_1m_idx" in prepared.aligned_data
    assert "exec_bar_end_1m_idx" in prepared.aligned_data


@pytest.mark.skipif(not FUTURES_DATA_DIR.exists(), reason="Data directory not found")
def test_backtest_engine_real_data_structure() -> None:
    """Tests the MultiSymbolEngine with real data structure (if available)."""
    files = list(FUTURES_DATA_DIR.glob('*_1h.parquet'))
    if not files:
        pytest.skip("No parquet data files found for test")
        
    df = pd.read_parquet(files[0]).iloc[-200:]
    n_bars = len(df)
    symbols = ["TEST/USDT"]
    
    # Mock the required columns for the engine
    aligned_data = {
        "close": df["close"].to_numpy().reshape(-1, 1),
        "high": df["high"].to_numpy().reshape(-1, 1),
        "low": df["low"].to_numpy().reshape(-1, 1),
        "open": df["open"].to_numpy().reshape(-1, 1),
        "atr": (df["close"] * 0.01).to_numpy().reshape(-1, 1),
        "funding_rate_sum": np.zeros((n_bars, 1)),
        "kill_signal": np.zeros((n_bars, 1)),
        "target_weights": np.ones((n_bars, 1)) * 0.3,
        "xs_score_long": np.ones((n_bars, 1)) * 0.5,
        "xs_score_short": np.zeros((n_bars, 1)),
        "hmm_prob_crisis": np.zeros((n_bars, 1)),
        "hmm_modulator_long": np.ones((n_bars, 1)),
        "hmm_modulator_short": np.ones((n_bars, 1)),
        "entry_upper": np.zeros((n_bars, 1)),
        "entry_lower": np.ones((n_bars, 1)) * 999999.0,
        "trend_direction": np.ones((n_bars, 1)),
        "strength_filter": np.ones((n_bars, 1)),
        "slot_rank_score": np.ones((n_bars, 1)),
        "ml_calib_prob": np.zeros((n_bars, 1)),
    }

    engine = PortfolioBacktestEngine(
        aligned_data=aligned_data,
        symbol_names=symbols,
        strategy_params={"REBALANCE_BARS": 6},
        initial_balance=1000.0
    )
    
    trades_df, _equity, final_bal, _diag = engine.run()
    assert isinstance(trades_df, pd.DataFrame)
    assert final_bal > 0
