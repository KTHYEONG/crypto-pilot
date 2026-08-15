"""Scenarios 1-3: compute_risk_severity_code unit tests."""
from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.market_regime import compute_risk_severity_code


class TestComputeRiskSeverityCode:
    def test_compute_risk_severity_code_separates_transition_from_crash(self) -> None:
        """[S1] 모든 구간 calm vol_scale, crisis_active=False → 전부 code==0(calm)."""
        vol_scale = np.full(200, 2.0)
        crisis_active = np.zeros(200, dtype=bool)
        code = compute_risk_severity_code(vol_scale, crisis_active, min_n_eff=20)
        assert np.all(code == 0)

    def test_compute_risk_severity_code_crisis_active_overrides_elevated(self) -> None:
        """[S2] crisis_active=True bar는 vol_scale 무관 무조건 code==2(crash)."""
        vol_scale = np.full(100, 2.0)
        crisis_active = np.zeros(100, dtype=bool)
        crisis_active[70] = True
        code = compute_risk_severity_code(vol_scale, crisis_active, min_n_eff=10)
        assert code[70] == 2
        assert code[69] == 0
        assert np.all(code[:10] == 0)

    def test_compute_risk_severity_code_empty_input_returns_empty(self) -> None:
        """[S3] 빈 배열 입력 시 빈 배열 반환."""
        vol_scale = np.array([], dtype=np.float64)
        crisis_active = np.array([], dtype=bool)
        code = compute_risk_severity_code(vol_scale, crisis_active)
        assert isinstance(code, np.ndarray)
        assert code.size == 0
