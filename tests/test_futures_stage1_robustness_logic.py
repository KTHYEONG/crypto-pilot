import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.execution.opt_main_futures import _eval_combo_task
from src.domain.futures.opt_futures_utils.combination_screener_futures import CombinationScoreFutures

class TestStage1Robustness(unittest.TestCase):
    def setUp(self):
        # 최소한의 데이터 맵 구성
        self.symbols = ["BTC/USDT"]
        self.tf = "4h"
        self.project_root = "."
        self.data_maps = {
            "BTC/USDT": {
                "4h": pd.DataFrame({
                    "datetime": pd.date_range("2024-01-01", periods=300, freq="4h"),
                    "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0
                }),
                "1d": pd.DataFrame({
                    "datetime": pd.date_range("2024-01-01", periods=60, freq="1d"),
                    "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0
                }),
                "is_start_idx_4h": 0,
                "merge_idx_4h": 0
            }
        }
        self.prebuilt = None

    @patch("src.execution.opt_main_futures.objective_futures")
    def test_robustness_scoring_logic(self, mock_obj):
        """
        검증 시나리오:
        - Combo A (Fragile Peak): 최고점은 1.0이나, 대부분의 구간에서 0.0 (불안정)
        - Combo B (Robust Plateau): 최고점은 0.8이나, 전 구간에서 0.7~0.8 (안정적)
        결과: Robust Score는 Combo B가 더 높아야 함.
        """
        
        # Combo A의 Trial 결과 시뮬레이션 (64개)
        # 1개는 1.0, 나머지는 0.05 수준
        fragile_results = [1.0] + [0.05] * 63
        
        # Combo B의 Trial 결과 시뮬레이션 (64개)
        # 전 구간이 0.7~0.8 사이
        robust_results = list(np.linspace(0.7, 0.8, 64))

        # 순차적으로 결과를 반환하도록 설정
        mock_obj.side_effect = fragile_results + robust_results

        # 1. Fragile Combo 평가 (BB_SQUEEZE 사용)
        score_a = _eval_combo_task(
            "BB_SQUEEZE", "EMA_ATR", "inv_vol_parity", 
            self.data_maps, self.symbols, self.tf, self.project_root, self.prebuilt
        )

        # 2. Robust Combo 평가 (동일한 시그널/레짐 사용하나 Mock 결과가 다름)
        score_b = _eval_combo_task(
            "BB_SQUEEZE", "EMA_ATR", "inv_vol_parity", 
            self.data_maps, self.symbols, self.tf, self.project_root, self.prebuilt
        )

        print(f"\n[Test Results]")
        print(f"Combo A (Fragile) - Robust Score: {score_a.p10_gmgr:.4f}")
        print(f"Combo B (Robust)  - Robust Score: {score_b.p10_gmgr:.4f}")

        # Robust Combo의 점수가 더 높아야 함
        self.assertGreater(score_b.p10_gmgr, score_a.p10_gmgr)
        self.assertTrue(score_b.p10_gmgr > 0.5) # 안정적 수익 구간이므로 높은 점수 기대

if __name__ == "__main__":
    unittest.main()
