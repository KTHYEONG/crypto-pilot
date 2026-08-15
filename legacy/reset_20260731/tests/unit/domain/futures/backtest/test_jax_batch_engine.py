"""Tests for JAX vmap GPU batch backtest engine (hybrid_compilation_opt).

Covers:
  - Scenario 1 (Happy Path): B_batch=4, N_bars=50, S_syms=3
  - Scenario 2 (Edge / LIMIT-02): single-candidate numerical parity with numba
  - Scenario 3 (Error / LIMIT-03): GPU unavailable → JaxBatchUnavailableError
  - Scenario 4 (Integration): active_pipeline dispatch gated by L2_JAX_BATCH_ENABLED
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.backtest.jax_batch_engine import (
    JaxBatchUnavailableError,
    simulate_batch_target_weights_jax,
)


# ===========================================================================
# Scenario 1 — Happy Path
# ===========================================================================


class TestSimulateBatchTargetWeightsJax:
    """Scenario 1: 기본 배치 동작 (B_batch=4, N_bars=50, S_syms=3)."""

    @staticmethod
    def _default_data(b: int = 4, n_bars: int = 50, s_syms: int = 3) -> dict:
        rng = np.random.default_rng(42)
        close = (rng.random((n_bars, s_syms)) + 10.0).astype(np.float32)
        high = close + 0.5
        low = close - 0.5
        open_ = close - 0.1
        return {
            "close": close,
            "high": high,
            "low": low,
            "open": open_,
            "funding_rate": np.zeros((n_bars, s_syms), dtype=np.float32),
            "target_weights_batch": rng.random((b, n_bars, s_syms)).astype(np.float32),
            "initial_balance": 1_000_000.0,
            "maker_fee": 0.0002,
            "taker_fee": 0.0005,
            "slippage_rate": 0.0005,
            "rebalance_bars": 4,
            "max_hold_bars": 96,
            "atr_2d": np.ones((n_bars, s_syms), dtype=np.float32),
            "atr_mult_batch": np.full(b, 2.0, dtype=np.float32),
            "trail_mult_batch": np.full(b, 1.5, dtype=np.float32),
        }

    def test_happy_path_shapes_and_positive_balances(self) -> None:
        b, n_bars, s_syms = 4, 50, 3
        data = self._default_data(b, n_bars, s_syms)

        equity_curves, final_balances, trade_counts = simulate_batch_target_weights_jax(
            close_2d=data["close"],
            high_2d=data["high"],
            low_2d=data["low"],
            open_2d=data["open"],
            funding_rate=data["funding_rate"],
            target_weights_batch=data["target_weights_batch"],
            initial_balance=data["initial_balance"],
            maker_fee=data["maker_fee"],
            taker_fee=data["taker_fee"],
            slippage_rate=data["slippage_rate"],
            rebalance_bars=data["rebalance_bars"],
            max_hold_bars=data["max_hold_bars"],
            atr_2d=data["atr_2d"],
            atr_mult_batch=data["atr_mult_batch"],
            trail_mult_batch=data["trail_mult_batch"],
        )

        assert equity_curves.shape == (b, n_bars)
        assert (final_balances > 0).all()
        assert trade_counts.shape == (b,)
        assert (trade_counts >= 0).all()

    def test_batch_size_1(self) -> None:
        """B_batch=1 경계: 단일 후보도 정상 동작."""
        b, n_bars, s_syms = 1, 20, 2
        data = self._default_data(b, n_bars, s_syms)

        equity_curves, final_balances, trade_counts = simulate_batch_target_weights_jax(
            close_2d=data["close"],
            high_2d=data["high"],
            low_2d=data["low"],
            open_2d=data["open"],
            funding_rate=data["funding_rate"],
            target_weights_batch=data["target_weights_batch"][:1],
            initial_balance=data["initial_balance"],
            maker_fee=data["maker_fee"],
            taker_fee=data["taker_fee"],
            slippage_rate=data["slippage_rate"],
            rebalance_bars=data["rebalance_bars"],
            max_hold_bars=data["max_hold_bars"],
            atr_2d=data["atr_2d"],
            atr_mult_batch=np.full(b, 2.0, dtype=np.float32),
            trail_mult_batch=np.full(b, 1.5, dtype=np.float32),
        )

        assert equity_curves.shape == (1, n_bars)
        assert final_balances[0] > 0
        assert trade_counts[0] >= 0

    def test_different_atr_mult_per_candidate(self) -> None:
        """각 후보가 다른 atr_mult를 가지면 결과가 달라짐."""
        b, n_bars, s_syms = 2, 30, 2
        data = self._default_data(b, n_bars, s_syms)

        atr_mult_batch = np.array([1.0, 5.0], dtype=np.float32)

        _, final_balances, _ = simulate_batch_target_weights_jax(
            close_2d=data["close"],
            high_2d=data["high"],
            low_2d=data["low"],
            open_2d=data["open"],
            funding_rate=data["funding_rate"],
            target_weights_batch=data["target_weights_batch"],
            initial_balance=data["initial_balance"],
            maker_fee=data["maker_fee"],
            taker_fee=data["taker_fee"],
            slippage_rate=data["slippage_rate"],
            rebalance_bars=data["rebalance_bars"],
            max_hold_bars=data["max_hold_bars"],
            atr_2d=data["atr_2d"],
            atr_mult_batch=atr_mult_batch,
            trail_mult_batch=np.full(b, 1.5, dtype=np.float32),
        )

        assert final_balances[0] != pytest.approx(final_balances[1], rel=1e-3)


# ===========================================================================
# Scenario 2 — Numerical Parity with Numba (LIMIT-02)
# ===========================================================================


class TestNumericalParityWithNumba:
    """[LIMIT-02] 단일-후보 결과가 backtest_target_weights_numba와 rtol=1e-6 이내 일치."""

    @staticmethod
    def _make_flat_data(
        n_bars: int = 50,
        n_syms: int = 2,
        price: float = 100.0,
        atr_val: float = 5.0,
    ) -> dict:
        """numaba 테스트와 동일한 flat OHLC + ATR 데이터."""
        o = np.full((n_bars, n_syms), price, dtype=np.float32)
        h = o + atr_val * 0.5
        lo = o - atr_val * 0.5
        c = o.copy()
        return {
            "close": c,
            "high": h,
            "low": lo,
            "open": o,
            "atr": np.full((n_bars, n_syms), atr_val, dtype=np.float32),
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "target_weight": np.full((n_bars, n_syms), 0.3, dtype=np.float32),
        }

    def test_single_candidate_parity_with_numba(self) -> None:
        """B_batch=1 결과가 backtest_target_weights_numba와 rtol=1e-6 일치."""
        n_bars, n_syms = 50, 2
        d = self._make_flat_data(n_bars, n_syms)

        # --- Numba reference ---
        from src.domain.futures.portfolio.execution_sim import backtest_target_weights_numba

        lev = np.ones((n_bars, n_syms), dtype=np.float64)
        ct = backtest_target_weights_numba(
            d["close"].astype(np.float64),
            d["high"].astype(np.float64),
            d["low"].astype(np.float64),
            d["open"].astype(np.float64),
            d["funding"].astype(np.float64),
            np.zeros((n_bars, n_syms), dtype=np.float64),  # kill_signal
            d["target_weight"].astype(np.float64),  # weights
            1_000_000.0,
            lev,
            0.0002,  # maker_fee
            0.0005,  # taker_fee
            0.0005,  # slippage_rate
            4,  # rebalance_bars
            96,  # max_hold_bars
            0.0,  # short_borrow_daily
            4.0,  # bar_hours
            d["atr"].astype(np.float64),
            2.0,  # atr_mult
            1.5,  # trail_mult
            1,  # use_simple_atr_stop
            100,  # max_concurrent
            10.0,  # max_exposure
            100.0,  # max_exp_per_coin
            0.0,  # dd_scaling_threshold
        )
        numba_trades, numba_balance, numba_equity, _ = ct

        # --- JAX batch candidate ---
        equity, final_balance, trade_count = simulate_batch_target_weights_jax(
            close_2d=d["close"],
            high_2d=d["high"],
            low_2d=d["low"],
            open_2d=d["open"],
            funding_rate=d["funding"],
            target_weights_batch=d["target_weight"][np.newaxis, :, :],
            initial_balance=1_000_000.0,
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage_rate=0.0005,
            rebalance_bars=4,
            max_hold_bars=96,
            atr_2d=d["atr"],
            atr_mult_batch=np.full(1, 2.0, dtype=np.float32),
            trail_mult_batch=np.full(1, 1.5, dtype=np.float32),
        )

        # Compare equity curves with rtol=1e-6
        np.testing.assert_allclose(
            equity[0], numba_equity, rtol=1e-6,
            err_msg="equity_curve mismatch between JAX batch and numba",
        )
        assert final_balance[0] == pytest.approx(numba_balance, rel=1e-6)


# ===========================================================================
# Scenario 3 — GPU Unavailable Fallback (LIMIT-03)
# ===========================================================================


class TestGpuUnavailableFallback:
    """[LIMIT-03] GPU 미가용 시 JaxBatchUnavailableError 발생 후 호출부 폴백."""

    def test_gpu_unavailable_raises(self, mocker) -> None:
        """jax.devices 가 RuntimeError를 던지면 JaxBatchUnavailableError로 변환."""
        mocker.patch("jax.devices", side_effect=RuntimeError("no CUDA device"))

        with pytest.raises(JaxBatchUnavailableError, match="no CUDA device"):
            simulate_batch_target_weights_jax(
                close_2d=np.empty((10, 2), dtype=np.float32),
                high_2d=np.empty((10, 2), dtype=np.float32),
                low_2d=np.empty((10, 2), dtype=np.float32),
                open_2d=np.empty((10, 2), dtype=np.float32),
                funding_rate=np.empty((10, 2), dtype=np.float32),
                target_weights_batch=np.empty((1, 10, 2), dtype=np.float32),
                initial_balance=1_000_000.0,
                maker_fee=0.0002,
                taker_fee=0.0005,
                slippage_rate=0.0005,
                rebalance_bars=4,
                max_hold_bars=96,
                atr_2d=np.empty((10, 2), dtype=np.float32),
                atr_mult_batch=np.full(1, 2.0, dtype=np.float32),
                trail_mult_batch=np.full(1, 1.5, dtype=np.float32),
            )

    def test_oom_fallback_raises(self, mocker) -> None:
        """OOM 상황(XLA RuntimeError)도 JaxBatchUnavailableError로 변환."""
        import jax

        real_devices = jax.devices()
        mocker.patch(
            "jax.devices",
            return_value=real_devices,
        )
        mocker.patch(
            "jax.numpy.zeros",
            side_effect=RuntimeError("RESOURCE_EXHAUSTED: Out of memory"),
        )

        with pytest.raises(JaxBatchUnavailableError, match="Out of memory"):
            simulate_batch_target_weights_jax(
                close_2d=np.empty((10, 2), dtype=np.float32),
                high_2d=np.empty((10, 2), dtype=np.float32),
                low_2d=np.empty((10, 2), dtype=np.float32),
                open_2d=np.empty((10, 2), dtype=np.float32),
                funding_rate=np.empty((10, 2), dtype=np.float32),
                target_weights_batch=np.empty((1, 10, 2), dtype=np.float32),
                initial_balance=1_000_000.0,
                maker_fee=0.0002,
                taker_fee=0.0005,
                slippage_rate=0.0005,
                rebalance_bars=4,
                max_hold_bars=96,
                atr_2d=np.empty((10, 2), dtype=np.float32),
                atr_mult_batch=np.full(1, 2.0, dtype=np.float32),
                trail_mult_batch=np.full(1, 1.5, dtype=np.float32),
            )


# ===========================================================================
# Scenario 4 — Integration: active_pipeline dispatch (LIMIT-04)
# ===========================================================================


class TestActivePipelineJaxDispatch:
    """[LIMIT-04] L2_JAX_BATCH_ENABLED flag가 active_pipeline 디스패치를 게이트."""

    def test_flag_false_default(self) -> None:
        """L2_JAX_BATCH_ENABLED=False(default) — config 플래그 기본값 확인."""
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

        assert OPT_FUTURES_CONFIG.get("L2_JAX_BATCH_ENABLED") is False

    def test_jax_engine_referenced_in_dispatch(self) -> None:
        """active_pipeline._run_tiered_l2_study 내 JAX 배치 디스패치 지점이 존재함."""
        import src.application.futures.runner.active_pipeline as ap

        with open(ap.__file__) as f:
            source = f.read()
        # L2_JAX_BATCH_ENABLED flag 확인
        assert "L2_JAX_BATCH_ENABLED" in source
        # L2_JAX_BATCH_MAX_VRAM_GB config 참조
        assert "L2_JAX_BATCH_MAX_VRAM_GB" in source
        # jax_batch_engine import 지점 확인
        assert "from src.domain.futures.backtest import jax_batch_engine as _jbe" in source
