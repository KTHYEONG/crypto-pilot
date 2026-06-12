"""Unit tests for tiered_logging.py (§9 Logging Contract).

TI14: format_layer1_table 포맷 검증
TI15: format_system_status with SKIP
TI16: format_window_table 포맷
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.domain.futures.optimization.opt_config import LayeredWindow
from src.domain.futures.strategy.tiered_logging import (
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_system_status,
    format_window_table,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_l1_result(**kwargs: object) -> MagicMock:
    """SWF 기반 Layer1Result mock 생성."""
    r: MagicMock = MagicMock()
    r.pooled_ic = kwargs.get("pooled_ic", 0.05)
    r.pooled_tstat = kwargs.get("pooled_tstat", 2.5)
    r.breadth = kwargs.get("breadth", 0.85)
    r.valid_coverage = kwargs.get("valid_coverage", 0.90)
    r.n_valid = kwargs.get("n_valid", 10)
    r.n_total = kwargs.get("n_total", 12)
    r.n_trade_scope = kwargs.get("n_trade_scope", 12)
    r.gate_passed = kwargs.get("gate_passed", True)
    return r

# ---------------------------------------------------------------------------
# TI14: format_layer1_table
# ---------------------------------------------------------------------------

class TestFormatLayer1Table:
    """TI14: Layer 1 파이프 테이블 포맷 검증 (SWF-K 기준)."""

    def test_basic_pass_contains_required_fields(self) -> None:
        """gate_passed=True일 때 핵심 메트릭(pooled_ic/pooled_tstat)과 PASS 포함 검증."""
        # Arrange
        r1 = _make_l1_result(pooled_ic=0.035, pooled_tstat=2.1, valid_coverage=0.85)

        # Act
        result = format_layer1_table(r1)

        # Assert
        assert "Pooled IC" in result
        assert "PASS" in result
        assert "0.035" in result
        assert "2.10" in result

    def test_blocked_status_when_gate_failed(self) -> None:
        """gate_passed=False일 때 BLOCKED 상태 검증."""
        # Arrange
        r1 = _make_l1_result(
            pooled_ic=-0.010,
            pooled_tstat=0.5,
            breadth=0.20,
            valid_coverage=0.60,
            gate_passed=False,
        )

        # Act
        result = format_layer1_table(r1)

        # Assert
        assert "BLOCKED" in result
        assert "PASS" not in result

    def test_fold_details_appended_when_provided(self) -> None:
        """fold_details 있으면 SWF FOLD DETAILS 테이블 추가 검증."""
        # Arrange
        r1 = _make_l1_result(pooled_ic=0.04, pooled_tstat=2.0)
        folds = [
            {"fold": 1, "ic": 0.042, "breadth": 0.50, "n_valid": 9, "n_events": 120, "pass": True},
            {"fold": 2, "ic": -0.01, "breadth": 0.30, "n_valid": 7, "n_events": 100, "pass": False},
        ]

        # Act
        result = format_layer1_table(r1, fold_details=folds)

        # Assert
        assert "SWF FOLD DETAILS" in result
        assert "FAIL" in result

    def test_per_symbol_top10_appended_when_provided(self) -> None:
        """per_symbol_top10 있으면 PER-SYMBOL AGGREGATE 테이블 추가 검증."""
        # Arrange
        r1 = _make_l1_result()
        symbols = [
            {"symbol": "BTCUSDT", "raw_mu": 0.005, "vol": 0.02, "t_stat": 2.5, "ic": 0.04, "valid": True},
        ]

        # Act
        result = format_layer1_table(r1, per_symbol_top10=symbols)

        # Assert
        assert "PER-SYMBOL AGGREGATE" in result
        assert "BTCUSDT" in result

    def test_valid_coverage_formatted_as_percentage(self) -> None:
        """valid_coverage 85% → '85.0%' 포맷 검증."""
        # Arrange
        r1 = _make_l1_result(valid_coverage=0.85)

        # Act
        result = format_layer1_table(r1)

        # Assert
        assert "85.0%" in result


# ---------------------------------------------------------------------------
# TI15: format_system_status
# ---------------------------------------------------------------------------

class TestFormatSystemStatus:
    """TI15: 시스템 상태 파이프 테이블 검증."""

    def test_l2_l3_none_shows_skip(self) -> None:
        """l2=None, l3=None이면 Layer 2, Layer 3 SKIP 표시 검증."""
        # Arrange
        r1 = SimpleNamespace(gate_passed=True)

        # Act
        result = format_system_status(r1, None, None)

        # Assert
        assert "SKIP" in result
        assert "Layer 2" in result
        assert "Layer 3" in result
        assert "PASS" in result

    def test_l1_pass_shown(self) -> None:
        """l1.gate_passed=True → Layer 1 PASS 검증."""
        # Arrange
        r1 = SimpleNamespace(gate_passed=True)

        # Act
        result = format_system_status(r1, None, None)

        # Assert
        assert "Layer 1" in result
        assert result.count("PASS") >= 1

    def test_l1_blocked_when_gate_failed(self) -> None:
        """l1.gate_passed=False → Layer 1 BLOCKED 검증."""
        # Arrange
        r1 = SimpleNamespace(gate_passed=False, blocker_reason="IC too low")

        # Act
        result = format_system_status(r1, None, None)

        # Assert
        assert "BLOCKED" in result
        assert "IC too low" in result

    def test_all_layers_provided_shows_pass(self) -> None:
        """l1/l2/l3 모두 gate_passed=True → 모두 PASS 검증."""
        # Arrange
        r1 = SimpleNamespace(gate_passed=True)
        r2 = SimpleNamespace(gate_passed=True)
        r3 = SimpleNamespace(gate_passed=True)

        # Act
        result = format_system_status(r1, r2, r3)

        # Assert
        assert result.count("PASS") == 3
        assert "SKIP" not in result
        assert "BLOCKED" not in result


# ---------------------------------------------------------------------------
# TI16: format_window_table
# ---------------------------------------------------------------------------

class TestFormatWindowTable:
    """TI16: WindowTable 파이프 테이블 포맷 검증."""

    @pytest.fixture
    def sample_window(self) -> LayeredWindow:
        """표준 LayeredWindow 픽스처."""
        return LayeredWindow(
            fetch_start=datetime.date(2022, 1, 1),
            l1_start=datetime.date(2023, 1, 1),
            l2_start=datetime.date(2024, 7, 1),
            holdout_start=datetime.date(2025, 7, 1),
            holdout_end=datetime.date(2026, 1, 1),
            regime_floor=datetime.date(2023, 1, 1),
        )

    def test_contains_tiered_header(self, sample_window: LayeredWindow) -> None:
        """TIERED 헤더 포함 검증."""
        # Act
        result = format_window_table(sample_window)

        # Assert
        assert "TIERED" in result

    def test_contains_l1_start_date(self, sample_window: LayeredWindow) -> None:
        """l1_start 날짜 2023-01-01 포함 검증."""
        # Act
        result = format_window_table(sample_window)

        # Assert
        assert "2023-01-01" in result

    def test_contains_regime_floor_segment(self, sample_window: LayeredWindow) -> None:
        """Regime Floor 세그먼트 행 포함 검증."""
        # Act
        result = format_window_table(sample_window)

        # Assert
        assert "Regime Floor" in result

    def test_contains_all_segment_labels(self, sample_window: LayeredWindow) -> None:
        """L1 (SWF), L2 AWF, Holdout 세그먼트 레이블 모두 포함 검증."""
        # Act
        result = format_window_table(sample_window)

        # Assert
        assert "L1 (SWF)" in result
        assert "L2 (AWF)" in result
        assert "Holdout" in result

    def test_holdout_dates_included(self, sample_window: LayeredWindow) -> None:
        """holdout_start와 holdout_end 날짜 포함 검증."""
        # Act
        result = format_window_table(sample_window)

        # Assert
        assert "2025-07-01" in result
        assert "2026-01-01" in result


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestFormatLayer2Table:
    """format_layer2_table 기본 포맷 검증."""

    def test_basic_pass_contains_gate(self) -> None:
        """gate_passed=True → PASS, Top-K 포함 검증."""
        # Arrange
        r2 = SimpleNamespace(
            top_k=5,
            friction_pass_pct=0.75,
            sharpe_hybrid=1.5,
            sharpe_1n=1.0,
            mdd_hybrid=0.12,
            mdd_1n=0.18,
            avg_active_positions=3.5,
            turnover=0.15,
            gate_passed=True,
        )

        # Act
        result = format_layer2_table(r2)

        # Assert
        assert "PASS" in result
        assert "Top-K" in result
        assert "1.50" in result

    def test_awf_folds_appended(self) -> None:
        """awf_folds 있으면 AWF FOLD DETAILS 추가 검증."""
        # Arrange
        r2 = SimpleNamespace(
            top_k=5,
            friction_pass_pct=0.75,
            sharpe_hybrid=1.5,
            sharpe_1n=1.0,
            mdd_hybrid=0.12,
            mdd_1n=0.18,
            avg_active_positions=3.5,
            turnover=0.15,
            gate_passed=True,
        )
        folds = [{"fold": 1, "sharpe": 1.4, "mdd": 0.11, "active_pos": 3.2, "pass": True}]

        # Act
        result = format_layer2_table(r2, awf_folds=folds)

        # Assert
        assert "AWF FOLD DETAILS" in result


class TestFormatLayer3Table:
    """format_layer3_table 기본 포맷 검증."""

    def test_contains_holdout_dates_and_gate(self) -> None:
        """호 기간 날짜와 게이트 상태 포함 검증."""
        # Arrange
        r3 = SimpleNamespace(
            cagr_hybrid=0.45,
            mdd_hybrid=0.15,
            sharpe_hybrid=1.8,
            mar_hybrid=3.0,
            cagr_1n=0.30,
            mdd_1n=0.20,
            sharpe_1n=1.2,
            mar_1n=1.5,
            cagr_vs=0.15,
            mdd_vs=-0.05,
            sharpe_vs=0.6,
            mar_vs=1.5,
            gate_passed=True,
        )

        # Act
        result = format_layer3_table(r3, ho_start="2025-07-01", ho_end="2026-01-01")

        # Assert
        assert "2025-07-01" in result
        assert "2026-01-01" in result
        assert "PASS" in result
        assert "L1+L2 Hybrid" in result
        assert "1/N Baseline" in result


# ---------------------------------------------------------------------------
# S11 / S12: SWF 전환 문자열 검증
# ---------------------------------------------------------------------------

class TestFormatLayer1TableSwfStrings:
    """S11/S12: CPCV→SWF 교체 및 fold_pass_ratio 메인 테이블 미포함 검증."""

    def test_format_layer1_table_no_cpcv_string(self) -> None:
        """S11: 출력에 'CPCV' 없고 'SWF', 'Pooled IC' 포함, 'Mean IC (fold)' 없음."""
        # Arrange
        r = _make_l1_result()
        fold_details = [
            {"fold": 1, "ic": 0.04, "breadth": 0.9, "n_valid": 10, "n_events": 1000, "pass": True},
        ]

        # Act
        result = format_layer1_table(r, fold_details=fold_details)

        # Assert
        assert "CPCV" not in result
        assert "SWF" in result
        assert "Pooled IC" in result
        assert "Mean IC (fold)" not in result

    def test_format_layer1_table_no_fold_pass_ratio_in_main(self) -> None:
        """S12: 메인 테이블 섹션에 'Fold Pass Ratio' 없고, 'NW HAC' 행 존재."""
        # Arrange
        r = _make_l1_result()

        # Act
        result = format_layer1_table(r)
        # SWF FOLD DETAILS 이전 부분만 추출 (fold_details 미전달이므로 전체가 메인)
        main_section = (
            result.split("[SWF FOLD DETAILS]")[0]
            if "[SWF FOLD DETAILS]" in result
            else result
        )

        # Assert
        assert "Fold Pass Ratio" not in main_section
        assert "NW HAC" in result
