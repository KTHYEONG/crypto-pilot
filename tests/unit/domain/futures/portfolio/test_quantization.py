"""Phase 5: minNotional 양자화 검증.

사양서 §7.6 기준.
"""

from __future__ import annotations

import numpy as np

from src.domain.futures.portfolio.portfolio_constructor import quantize_weights


class TestQuantization:
    """quantize_weights 함수 검증."""

    def test_min_notional_below_threshold_zeroed(self) -> None:
        """MinNotional 미달 주문 → 0.

        equity=10000, price=50000, step_size=0.001, w=0.0001
        notional = 0.0001 * 10000 = 1 USDT < 20 → qty=0
        """
        equity = 10_000.0
        w = np.array([0.0001], dtype=np.float64)
        prices = np.array([50_000.0], dtype=np.float64)
        step_sizes = np.array([0.001], dtype=np.float64)

        result = quantize_weights(w, equity, prices, step_sizes, min_notional=20.0)

        assert result[0] == 0.0, f"minNotional 미달 → 0이어야 함: {result[0]}"

    def test_step_size_quantization_residual(self) -> None:
        """step_size 양자화 잔여 처리.

        w=0.055, equity=1000, price=100, step_size=0.1
        raw_qty = 0.55 → floor → 0.5 → notional=50 USDT → 통과
        """
        equity = 1_000.0
        w = np.array([0.055], dtype=np.float64)
        prices = np.array([100.0], dtype=np.float64)
        step_sizes = np.array([0.1], dtype=np.float64)

        result = quantize_weights(w, equity, prices, step_sizes, min_notional=20.0)

        # 결과 비중 재환산 qty = floor(0.55/0.1)*0.1 = 0.5
        # result = 0.5 * 100 / 1000 = 0.05
        expected_qty = 0.5
        expected_weight = expected_qty * 100.0 / equity
        assert abs(result[0] - expected_weight) < 1e-9, (
            f"step_size 양자화 불일치: {result[0]:.6f} vs {expected_weight:.6f}"
        )

    def test_aum_10k_low_price_symbol_passes(self) -> None:
        """AUM 10k, price=500 USDT → notional=200 → 통과."""
        equity = 10_000.0
        w = np.array([0.02], dtype=np.float64)
        prices = np.array([500.0], dtype=np.float64)
        step_sizes = np.array([0.01], dtype=np.float64)

        result = quantize_weights(w, equity, prices, step_sizes, min_notional=20.0)

        # notional = 0.02 * 10000 = 200 USDT → 통과 (0이 아님)
        assert result[0] > 0.0, f"notional=200 → 통과 ({result[0]})"

    def test_aum_10k_btc_passes_min_notional(self) -> None:
        """AUM 10k, price=50000 USDT, step_size=0.001, w=0.02.

        qty = floor(0.02*10000/50000 / 0.001) * 0.001
            = floor(0.004 / 0.001) * 0.001 = 4 * 0.001 = 0.004
        notional = 0.004 * 50000 = 200 USDT → 통과
        """
        equity = 10_000.0
        w = np.array([0.02], dtype=np.float64)
        prices = np.array([50_000.0], dtype=np.float64)
        step_sizes = np.array([0.001], dtype=np.float64)

        result = quantize_weights(w, equity, prices, step_sizes, min_notional=20.0)

        assert result[0] > 0.0, f"notional=200 → 통과 ({result[0]})"

    def test_multiple_symbols_some_zeroed(self) -> None:
        """여러 심볼: 일부는 minNotional 통과, 일부는 미달."""
        equity = 10_000.0
        w = np.array([0.0001, 0.05, 0.001], dtype=np.float64)
        prices = np.array([50_000.0, 100.0, 1_000.0], dtype=np.float64)
        step_sizes = np.array([0.001, 1.0, 0.1], dtype=np.float64)

        result = quantize_weights(w, equity, prices, step_sizes, min_notional=20.0)

        # 심볼 0: notional = 0.0001 * 10000 = 1 USDT < 20 → 0
        assert result[0] == 0.0, "심볼 0 minNotional 미달 → 0"
        # 심볼 1: notional = 0.05 * 10000 = 500 USDT → 통과
        assert result[1] > 0.0, f"심볼 1 통과 ({result[1]})"

    def test_output_is_weight_not_qty(self) -> None:
        """출력이 수량이 아닌 비중(weight)이어야 함."""
        equity = 10_000.0
        w = np.array([0.10, 0.05], dtype=np.float64)
        prices = np.array([100.0, 200.0], dtype=np.float64)
        step_sizes = np.array([1.0, 0.1], dtype=np.float64)

        result = quantize_weights(w, equity, prices, step_sizes, min_notional=20.0)

        # 비중이므로 |result| ≤ 1.0 (gross ≤ 2.0)
        assert float(np.sum(np.abs(result))) <= 2.0, (
            f"출력이 비중이어야 함 (gross={np.sum(np.abs(result))})"
        )
