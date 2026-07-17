"""Unit tests for tiered_logging.py (§9 Logging Contract).

TI14: format_layer1_table 포맷 검증
TI15: format_system_status with SKIP
TI16: format_window_table 포맷
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.optimization.opt_config import LayeredWindow
from src.domain.futures.strategy.candidate_contracts import (
    Layer1FoldReadiness,
    Layer1GateCheck,
    Layer1GateReport,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_logging import (
    _format_top_symbol_contributions,
    evaluation_window_bottleneck_verdict,
    format_layer1_deployment_registry_table,
    format_layer1_gate_table,
    format_layer1_outer_fold_table,
    format_layer1_table,
    format_layer2_table,
    format_layer3_table,
    format_layer_universe_audit_table,
    format_system_status,
    format_window_table,
    render_family_rejection_funnel,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_l1_result(**kwargs: object) -> MagicMock:
    """SWF 기반 Layer1Result mock 생성."""
    r: MagicMock = MagicMock()
    r.breadth = kwargs.get("breadth", 0.85)
    r.n_valid = kwargs.get("n_valid", 10)
    r.n_total = kwargs.get("n_total", 12)
    r.n_trade_scope = kwargs.get("n_trade_scope", 12)
    r.gate_passed = kwargs.get("gate_passed", True)
    r.cs_ic_mean = kwargs.get("cs_ic_mean", 0.05)
    r.cs_ic_tstat = kwargs.get("cs_ic_tstat", 2.5)
    r.cs_ic_fold_pass_ratio = kwargs.get("cs_ic_fold_pass_ratio", 0.85)
    r.decile_lift_bps = kwargs.get("decile_lift_bps", 3.0)
    r.strategy_panel = kwargs.get("strategy_panel", ())
    r.n_valid_strategies = kwargs.get("n_valid_strategies", 5)
    r.panel_diversity = kwargs.get("panel_diversity", 0.4)
    return r


# ---------------------------------------------------------------------------
# TI14: format_layer1_table
# ---------------------------------------------------------------------------


class TestFormatLayer1Table:
    """TI14: Layer 1 파이프 테이블 포맷 검증 (SWF-K 기준)."""

    def test_basic_pass_contains_required_fields(self) -> None:
        """gate_passed=True일 때 핵심 메트릭(CS IC/panel)과 PASS 포함 검증."""
        # Arrange
        r1 = _make_l1_result(cs_ic_mean=0.035, cs_ic_tstat=2.1, cs_ic_fold_pass_ratio=0.85)

        # Act
        result = format_layer1_table(r1)

        # Assert
        assert "CS IC Mean" in result
        assert "PASS" in result
        assert "0.035" in result
        assert "2.10" in result

    def test_blocked_status_when_gate_failed(self) -> None:
        """gate_passed=False일 때 BLOCKED 상태 검증."""
        # Arrange
        r1 = _make_l1_result(
            cs_ic_mean=-0.010,
            cs_ic_tstat=0.5,
            breadth=0.20,
            cs_ic_fold_pass_ratio=0.20,
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
        r1 = _make_l1_result(cs_ic_mean=0.04, cs_ic_tstat=2.0)
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
        """cs_ic_fold_pass_ratio 85% → '85.0%' 포맷 검증."""
        # Arrange
        r1 = _make_l1_result(cs_ic_fold_pass_ratio=0.85)

        # Act
        result = format_layer1_table(r1)

        # Assert
        assert "85.0%" in result


def test_format_layer1_gate_table_uses_explicit_checks() -> None:
    report = Layer1GateReport(
        checks=(
            Layer1GateCheck("sym_count", 2.0, 6.0, "ge", False, "2.000"),
            Layer1GateCheck("probe_bps", 0.8, 0.0, "gt", True, None),
        ),
        passed=False,
        blockers=("sym_count:2.000",),
    )

    result = format_layer1_gate_table(report)

    assert "Symbol-Breadth" in result
    assert "❌" in result
    assert "BLOCKER" in result
    assert "BLOCKED" in result
    assert "1/2 Passed" in result


def test_format_layer1_outer_fold_table_shows_ready_symbol_count() -> None:
    reports = (
        Layer1FoldReadiness(
            fold_id=10,
            registry_source_end_idx=100,
            outer_oos_start_idx=110,
            outer_oos_end_idx=140,
            ready_symbols=("BTCUSDT", "ETHUSDT"),
            valid_opportunity_timestamp_count=4,
            opportunity_ic=0.12,
            opportunity_ic_series=(0.1, 0.14),
            probe_bps=1.5,
            probe_gross_edge_series_bps=(1.0, 2.0),
            passed=True,
            blockers=(),
        ),
    )

    result = format_layer1_outer_fold_table(reports)

    assert "LAYER 1 OUTER FOLD READINESS" in result
    assert "2 symbols loaded" in result
    assert "READY" in result
    assert "✅" in result
    assert "Fold #10" in result


def test_format_layer1_outer_fold_table_uses_calendar_periods_and_symbol_preview() -> None:
    reports = (
        Layer1FoldReadiness(
            fold_id=1,
            registry_source_end_idx=3,
            outer_oos_start_idx=3,
            outer_oos_end_idx=6,
            ready_symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            valid_opportunity_timestamp_count=7,
            probe_bps=1.2,
            passed=True,
        ),
    )
    datetimes = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-02"),
            np.datetime64("2025-01-03"),
            np.datetime64("2025-01-04"),
            np.datetime64("2025-01-05"),
            np.datetime64("2025-01-06"),
        ]
    )

    result = format_layer1_outer_fold_table(reports, datetimes=datetimes, max_symbols=2)

    assert "FitEnd: 2025-01-03" in result
    assert "OOS: 2025-01-04 ~ 2025-01-06" in result
    assert "BTCUSDT, ETHUSDT, +1 more" in result


def test_format_layer_universe_audit_table_renders_rows() -> None:
    audit = SimpleNamespace(
        layer="L2",
        start_idx=10,
        end_idx=20,
        start_date="2025-01-01",
        end_date="2025-01-10",
        symbol_count=12,
        active_symbol_count_min=3,
        active_symbol_count_median=7.0,
        active_symbol_count_max=10,
        entry_block_count=4,
        kill_count=2,
        symbols=("BTCUSDT", "ETHUSDT"),
        warnings=("low_active_tail",),
    )

    result = format_layer_universe_audit_table((audit,))

    assert "LAYER UNIVERSE AUDIT" in result
    assert "L2" in result
    assert "low_active_tail" in result


def test_format_layer1_deployment_registry_table_lists_strategy_rows() -> None:
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=4.0,
        mean_incremental_bps=1.5,
        bootstrap_tstat_incremental=2.1,
        p_value=0.02,
        q_value=0.04,
        positive_fold_ratio=0.75,
        n_obs=20,
        effective_n=15.0,
        n_folds=3,
        reliability=0.9,
        qualified=True,
        rejection_reasons=(),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (evidence,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="deployment",
    )

    result = format_layer1_deployment_registry_table(registry)

    assert "L1 FINAL PROMOTION SUMMARY" in result
    assert "trend (fast) [bull]" in result
    assert "BTCUSDT" in result


# ---------------------------------------------------------------------------
# TI-REG: format_layer1_deployment_registry_table 확장 테스트 (spec S1~S5)
# ---------------------------------------------------------------------------


def _make_evidence(
    symbol: str,
    strategy_id: str,
    q_value: float,
    hard_eligible: bool = True,
    structural_reasons: tuple[str, ...] = (),
) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(symbol, strategy_id, "all"),
        mean_gross_bps=5.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.5,
        p_value=0.02,
        q_value=q_value,
        positive_fold_ratio=0.75,
        n_obs=20,
        effective_n=15.0,
        n_folds=3,
        hard_eligible=hard_eligible,
        structural_reasons=structural_reasons,
        quality_weight=0.8 if hard_eligible else 0.0,
    )


def _make_registry(*evidence_items: SymbolStrategyEvidence) -> QualifiedSignalRegistry:
    by_symbol: dict[str, tuple[SymbolStrategyEvidence, ...]] = {}
    for ev in evidence_items:
        sym = ev.key.symbol
        by_symbol.setdefault(sym, ())
        by_symbol[sym] = by_symbol[sym] + (ev,)
    return QualifiedSignalRegistry(
        by_symbol=by_symbol,
        ready_symbols=tuple(by_symbol.keys()),
        trade_scope_count=len(by_symbol),
        registry_version="test",
    )


class TestDeploymentRegistryTablePassFail:
    """S1~S5: PASS/FAIL 분리, 라벨, maturity, 하위호환 검증."""

    def test_s1_pass_full_fail_summary(self) -> None:
        """S1 (Happy): 2 PASS + 3 FAIL all_evidence → PASS 전체 + [NOT PROMOTED] 1줄."""
        # Arrange
        pass_ev1 = _make_evidence("BTCUSDT", "trend:fast", q_value=0.20)
        pass_ev2 = _make_evidence("ETHUSDT", "mom:slow", q_value=0.60)
        fail_ev1 = _make_evidence(
            "SOLUSDT", "rsi:6", q_value=0.90, hard_eligible=False, structural_reasons=("insufficient_folds",)
        )
        fail_ev2 = _make_evidence(
            "DOGEUSDT", "rsi:6", q_value=0.95, hard_eligible=False, structural_reasons=("insufficient_folds",)
        )
        fail_ev3 = _make_evidence(
            "XRPUSDT", "bb:20", q_value=0.99, hard_eligible=False, structural_reasons=("no_incremental_edge",)
        )
        registry = _make_registry(pass_ev1, pass_ev2)
        all_ev = (pass_ev1, pass_ev2, fail_ev1, fail_ev2, fail_ev3)

        # Act
        result = format_layer1_deployment_registry_table(registry, all_evidence=all_ev)

        # Assert
        assert "[NOT PROMOTED] 3 pairs" in result
        assert "insufficient_foldsx2" in result
        assert "no_incremental_edgex1" in result
        assert "BTCUSDT" in result
        assert "ETHUSDT" in result

    def test_s2_backward_compat_no_fail_section(self) -> None:
        """S2 (하위호환): all_evidence=() → FAIL 섹션 미출력."""
        # Arrange
        ev = _make_evidence("BTCUSDT", "trend:fast", q_value=0.10)
        registry = _make_registry(ev)

        # Act
        result = format_layer1_deployment_registry_table(registry)

        # Assert
        assert "[NOT PROMOTED]" not in result
        assert "L1 FINAL PROMOTION SUMMARY" in result

    def test_s4_label_no_rejected_keyword(self) -> None:
        """S4 (라벨): q=0.10/0.50/0.80 모두 [L2-PASS], REJECTED 문자열 없음."""
        # Arrange
        ev_hi = _make_evidence("BTCUSDT", "trend:fast", q_value=0.10)
        ev_mid = _make_evidence("ETHUSDT", "mom:slow", q_value=0.50)
        ev_lo = _make_evidence("SOLUSDT", "rsi:6", q_value=0.80)
        registry = _make_registry(ev_hi, ev_mid, ev_lo)

        # Act
        result = format_layer1_deployment_registry_table(registry)

        # Assert
        assert "REJECTED" not in result
        assert "WATCH" not in result
        assert "PROMOTED" not in result
        # New format: Q:hi/mid/lo removed, continuous metrics used instead
        assert "[L2-PASS]" not in result
        assert "LCB(bps)" in result
        assert "CONV" in result
        assert "FOLDS" in result

    def test_s5_empty_registry_with_fail_evidence(self) -> None:
        """S5 (Empty registry): 빈 registry + all_evidence 3 fail → 미전달 메시지 + FAIL 요약."""
        # Arrange
        empty_registry = QualifiedSignalRegistry(
            by_symbol={},
            ready_symbols=(),
            trade_scope_count=0,
            registry_version="test",
        )
        fail_evs = tuple(
            _make_evidence(
                f"SYM{i}USDT", "rsi:6", q_value=0.99, hard_eligible=False, structural_reasons=("insufficient_folds",)
            )
            for i in range(3)
        )

        # Act
        result = format_layer1_deployment_registry_table(empty_registry, all_evidence=fail_evs)

        # Assert
        assert "No variants promoted to Layer 2" in result
        assert "[NOT PROMOTED] 3 pairs" in result


# ---------------------------------------------------------------------------
# TI-MAT: format_layer1_outer_fold_table maturity censoring 노출
# ---------------------------------------------------------------------------


class TestOuterFoldTableMaturityDisplay:
    """S3: dropped_by_maturity_count > 0 → [censored: N] 노출."""

    def test_s3_censored_count_shown_when_nonzero(self) -> None:
        """dropped_by_maturity_count=5 → Quality 라인에 [censored: 5] 표시."""
        # Arrange
        report = Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=100,
            outer_oos_start_idx=80,
            outer_oos_end_idx=100,
            ready_symbols=("BTCUSDT", "ETHUSDT"),
            matched_event_count=50,
            probe_bps=42.5,
            probe_lcb_bps=30.0,
            passed=True,
            dropped_by_maturity_count=5,
        )

        # Act
        result = format_layer1_outer_fold_table((report,))

        # Assert
        assert "[censored: 5]" in result
        assert "42.50 bps" in result

    def test_s3_no_censored_label_when_zero(self) -> None:
        """dropped_by_maturity_count=0 → [censored:] 미노출."""
        # Arrange
        report = Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=100,
            outer_oos_start_idx=80,
            outer_oos_end_idx=100,
            ready_symbols=("BTCUSDT",),
            matched_event_count=30,
            probe_bps=55.0,
            probe_lcb_bps=40.0,
            passed=True,
            dropped_by_maturity_count=0,
        )

        # Act
        result = format_layer1_outer_fold_table((report,))

        # Assert
        assert "[censored:" not in result


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

    def test_l3_error_is_reported_separately(self) -> None:
        """l3 error 상태는 L3_ERROR로 구분 표시."""
        r1 = SimpleNamespace(gate_passed=True)
        r2 = SimpleNamespace(gate_passed=True)
        r3 = SimpleNamespace(gate_passed=False, blocker_reason="empty_holdout_window", status="L3_ERROR")

        result = format_system_status(r1, r2, r3)

        assert "L3_ERROR" in result
        assert "empty_holdout_window" in result


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


def _make_l2_ns(**kwargs: object) -> SimpleNamespace:
    """format_layer2_table용 SimpleNamespace 팩토리 (신규 필드 포함)."""
    defaults = {
        "sharpe_hybrid": 1.5,
        "sharpe_baseline": 1.0,
        "mdd_hybrid": 0.12,
        "mdd_baseline": 0.18,
        "cagr_hybrid": 0.40,
        "cagr_baseline": 0.20,
        "mar_hybrid": 3.3,
        "mar_baseline": 1.1,
        "fold_pass_ratio": 0.75,
        "turnover": 0.15,
        "friction_pass_pct": 0.75,
        "gate_passed": True,
        "blocker_reason": "",
        "psr_hybrid": 0.92,
        "dsr_hybrid": 0.85,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestFormatLayer2Table:
    """format_layer2_table 기본 포맷 검증."""

    def test_basic_pass_contains_gate(self) -> None:
        """gate_passed=True → PASS, Sharpe/CAGR 값 포함 검증."""
        # Arrange
        r2 = _make_l2_ns()

        # Act
        result = format_layer2_table(r2)

        # Assert
        assert "PASS" in result
        assert "Sharpe" in result
        assert "1.500" in result
        assert "CAGR" in result
        assert "PSR" in result
        assert "0.920" in result
        assert "DSR" in result
        assert "0.850" in result

    def test_blocked_shows_blocker_reason(self) -> None:
        """gate_passed=False + blocker_reason → 로그에 reason 포함."""
        # Arrange
        r2 = _make_l2_ns(gate_passed=False, blocker_reason="cagr", cagr_hybrid=-0.05)

        # Act
        result = format_layer2_table(r2)

        # Assert
        assert "BLOCKED" in result
        assert "cagr" in result

    def test_friction_shown_as_gate(self) -> None:
        """friction_pass_pct는 Robustness 게이트에 영향을 줌을 검증."""
        # Arrange
        r2 = _make_l2_ns(friction_pass_pct=0.30)

        # Act
        result = format_layer2_table(r2)

        # Assert
        assert "Robust" in result
        assert "❌" in result  # friction_pass_pct=0.30 < 0.50

    def test_uplift_gate_shown_as_additive(self) -> None:
        """Sharpe Uplift 임계가 config 값 (>=+0.05) 으로 표기됨 검증."""
        # Arrange
        r2 = _make_l2_ns(sharpe_baseline=0.5)

        # Act
        result = format_layer2_table(r2)

        # Assert — 기본 config.l2_min_sharpe_uplift=0.05
        assert ">=+0.05" in result

    def test_awf_folds_appended(self) -> None:
        """awf_folds 있으면 fold 테이블 추가 검증."""
        # Arrange
        r2 = _make_l2_ns()
        folds = [{"fold": 1, "sharpe": 1.4, "mdd": 0.11, "pass": True, "period": "2025-01-01 ~ 2025-03-31"}]

        # Act
        result = format_layer2_table(r2, awf_folds=folds)

        # Assert
        assert "Fold" in result
        assert "PASS" in result
        assert "Period: 2025-01-01 ~ 2025-03-31" in result

    def test_renders_l2_regime_trend_efficiency_diagnostic(self) -> None:
        """[L2-REGIME] 라인에 fit/cal ER mean/corr 정확한 substring 렌더링."""
        r2 = _make_l2_ns(mean_trend_efficiency=0.218, trend_efficiency_corr=-0.016)

        result = format_layer2_table(r2)

        assert "[L2-REGIME]" in result
        assert "mean=0.218" in result
        assert "corr=-0.016" in result

    def test_awf_folds_render_trend_efficiency_column(self) -> None:
        """Fold detail 라인에 per-fold ER 컬럼이 렌더링된다."""
        r2 = _make_l2_ns()
        folds = [
            {
                "fold": 1,
                "sharpe": 1.4,
                "mdd": 0.11,
                "cagr": 0.32,
                "pass": True,
                "period": "2025-01-01 ~ 2025-03-31",
                "trend_efficiency": 0.305,
            }
        ]

        result = format_layer2_table(r2, awf_folds=folds)

        assert "ER: 0.305" in result

    def test_awf_folds_render_selected_symbols(self) -> None:
        r2 = _make_l2_ns()
        folds = [
            {
                "fold": 1,
                "sharpe": 1.4,
                "mdd": 0.11,
                "cagr": 0.32,
                "pass": True,
                "period": "2025-01-01 ~ 2025-03-31",
                "symbols": ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            }
        ]

        result = format_layer2_table(r2, awf_folds=folds)

        assert "Symbols: 3 [BTCUSDT, ETHUSDT, SOLUSDT]" in result

    def test_nan_fold_shown_safely(self) -> None:
        """fold sharpe=nan → 'nan' 문자열로 안전 렌더링."""
        # Arrange
        r2 = _make_l2_ns()
        folds = [{"fold": 1, "sharpe": float("nan"), "mdd": float("nan"), "pass": False}]

        # Act
        result = format_layer2_table(r2, awf_folds=folds)

        # Assert
        assert "nan" in result

    def test_format_layer2_table_uses_ew_bench_header(self) -> None:
        """출력에 Gate 임계값이나 PnL 등이 표기되는지 검증."""
        # Arrange
        r2 = _make_l2_ns()

        # Act
        result = format_layer2_table(r2)

        # Assert
        assert "CAGR" in result
        assert "PnL" in result

    def test_format_layer2_table_mar_na_when_cagr_negative(self) -> None:
        """cagr_hybrid < 0이면 MAR 셀에 'n/a(loss)' 표기 검증."""
        # Arrange
        r2 = _make_l2_ns(cagr_hybrid=-0.10, mar_hybrid=-2.0)

        # Act
        result = format_layer2_table(r2)

        # Assert
        assert "n/a(loss)" in result

    def test_format_layer2_table_renders_evaluation_period_in_header(self) -> None:
        """L2 scorecard header에 실제 평가 기간이 표시되어야 한다."""
        r2 = _make_l2_ns()

        result = format_layer2_table(
            r2,
            evaluation_start="2025-10-01",
            evaluation_end="2026-03-31",
        )

        assert "[LAYER 2 PORTFOLIO SCORECARD] (2025-10-01 ~ 2026-03-31)" in result

    def test_format_layer2_table_psr_gate_shown(self) -> None:
        """PSR 게이트가 표시되고 psr_hybrid=0.50 < 0.90이면 ❌ 상태 검증."""
        # Arrange
        r2 = _make_l2_ns(psr_hybrid=0.50, dsr_hybrid=0.80)

        # Act
        result = format_layer2_table(r2)

        # Assert — PSR is the blocker gate
        assert "Integrity" in result
        assert "PSR" in result
        assert "❌" in result  # psr_hybrid=0.50 < config.l2_min_psr=0.90
        assert "DSR" in result
        assert "(diag)" in result

    def test_format_layer2_table_renders_long_short_pnl_split(self) -> None:
        """Scenario 6: [L2-LONGSHORT] 라인에 long/short realized price가 정확히 렌더링."""
        r2 = _make_l2_ns(realized_price_long=-0.020, realized_price_short=0.045)

        result = format_layer2_table(r2)

        assert "[L2-LONGSHORT]" in result
        assert "long=-0.0200" in result
        assert "short=+0.0450" in result

    def test_format_layer2_table_renders_per_symbol_long_short_top_lines(self) -> None:
        """Scenario 11: [L2-LONGSHORT-TOP] 라인에 top symbol contributions 렌더링."""
        r2 = _make_l2_ns(
            realized_price_long_by_symbol=(("ARUSDT", -0.021),),
            realized_price_short_by_symbol=(("BTCUSDT", 0.012),),
        )

        result = format_layer2_table(r2)

        assert "[L2-LONGSHORT-TOP]" in result
        assert "ARUSDT(-0.0210)" in result
        assert "BTCUSDT(+0.0120)" in result

    def test_format_layer2_table_uses_config_uplift_threshold_not_hardcoded(self) -> None:
        """config.l2_min_sharpe_uplift=0.05로 통과하는 uplift (0.10)이 ✅로 표시됨."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2AllocationConfig,
        )

        cfg = Layer2AllocationConfig(l2_min_sharpe_uplift=0.05)
        r2 = _make_l2_ns(sharpe_hybrid=1.2, sharpe_baseline=1.1)

        result = format_layer2_table(r2, config=cfg)

        assert ">=+0.05" in result
        assert "✅ [Uplift" in result


class TestEvaluationWindowBottleneckVerdict:
    """Gate A: evaluation_window_bottleneck_verdict 시나리오 테스트."""

    def test_covered_when_stress_fold_present(self) -> None:
        folds = [
            {"mdd": 0.20, "cagr": -0.05},
            {"mdd": 0.05, "cagr": 0.30},
        ]
        covered, detail = evaluation_window_bottleneck_verdict(folds)
        assert covered is True
        assert "fold=0" in detail
        assert "mdd=0.200" in detail

    def test_not_covered_when_only_bull_drawdown_fold(self) -> None:
        folds = [
            {"mdd": 0.1747, "cagr": 0.3395},
        ]
        covered, detail = evaluation_window_bottleneck_verdict(folds)
        assert covered is False
        assert detail == "no_bottleneck_caliber_fold_in_window"

    def test_skips_nan_folds(self) -> None:
        folds = [
            {"mdd": float("nan"), "cagr": -0.05},
            {"mdd": 0.25, "cagr": float("nan")},
            {"mdd": 0.30, "cagr": -0.10},
        ]
        covered, _ = evaluation_window_bottleneck_verdict(folds)
        assert covered is True

    def test_empty_folds_returns_not_covered(self) -> None:
        covered, detail = evaluation_window_bottleneck_verdict([])
        assert covered is False
        assert detail == "no_bottleneck_caliber_fold_in_window"

    def test_banner_appended_when_not_covered(self) -> None:
        r2 = _make_l2_ns()
        folds = [{"fold": 1, "mdd": 0.05, "cagr": 0.10, "sharpe": 1.0, "pass": True}]
        result = format_layer2_table(r2, awf_folds=folds)
        assert "NO-CRISIS-WINDOW" in result

    def test_no_banner_when_covered(self) -> None:
        r2 = _make_l2_ns()
        folds = [{"fold": 1, "mdd": 0.20, "cagr": -0.05, "sharpe": 1.0, "pass": True}]
        result = format_layer2_table(r2, awf_folds=folds)
        assert "NO-CRISIS-WINDOW" not in result


class TestFormatLayer3Table:
    """format_layer3_table 기본 포맷 검증."""

    def test_uses_actual_layer3result_field_names(self) -> None:
        """실제 Layer3Result 필드명(cagr/mdd/sharpe/...) 기준으로 렌더링."""
        r3 = SimpleNamespace(
            cagr=0.45,
            mdd=0.15,
            sharpe=1.8,
            mar=3.0,
            cagr_baseline=0.30,
            mdd_baseline=0.20,
            sharpe_baseline=1.2,
            mar_baseline=1.5,
            gate_passed=True,
            sortino=1.6,
            cvar95=0.03,
            n_trades=24,
        )

        result = format_layer3_table(r3, ho_start="2025-07-01", ho_end="2026-01-01")

        assert "2025-07-01" in result
        assert "2026-01-01" in result
        assert "HOLDOUT VALIDATION SCORECARD" in result
        assert "GROWTH" in result
        assert "DEPLOY-READY" in result
        assert "✅" in result
        assert "45.0%" in result
        assert "1.800" in result
        assert "CVaR95" in result
        assert "Calmar" not in result

    def test_blocked_shows_blocker_reason(self) -> None:
        """gate_passed=False + blocker_reason이면 BLOCKED 요약 표시."""
        r3 = SimpleNamespace(
            cagr=0.08,
            mdd=0.18,
            sharpe=0.9,
            mar=0.44,
            cagr_baseline=0.10,
            mdd_baseline=0.16,
            sharpe_baseline=1.1,
            mar_baseline=0.63,
            gate_passed=False,
            blocker_reason="growth_lcb",
            sortino=-0.2,
            cvar95=0.02,
            n_trades=5,
        )

        result = format_layer3_table(r3, ho_start="2025-07-01", ho_end="2026-01-01")

        assert ">> FINAL RESULT : ❌ BLOCKED (Reason: growth_lcb)" in result

    def test_error_renders_summary_instead_of_metric_table(self) -> None:
        """error 상태면 메트릭 표 대신 짧은 error summary 출력."""
        r3 = SimpleNamespace(
            gate_passed=False,
            blocker_reason="no_holdout_signals",
            status="ERROR",
        )

        result = format_layer3_table(r3, ho_start="2025-07-01", ho_end="2026-01-01")

        assert "Error Summary" in result
        assert "no_holdout_signals" in result
        assert "CAGR" not in result
        assert ">> FINAL RESULT : ❌ ERROR (no_holdout_signals)" in result

    def test_renders_new_compounding_and_sanity_fields(self) -> None:
        """S10: total_return/equity_multiple/n_trades 신규 필드와 DEPLOY-READY 라벨 렌더링."""
        # Arrange
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer3Result

        r3 = Layer3Result(
            cagr=0.35,
            mdd=0.12,
            sharpe=1.6,
            mar=2.9,
            cagr_baseline=0.20,
            mdd_baseline=0.18,
            sharpe_baseline=1.1,
            mar_baseline=1.1,
            gate_passed=True,
            blocker_reason="",
            total_return=0.17,
            equity_multiple=1.17,
            sortino=1.4,
            sortino_baseline=0.9,
            n_trades=58,
            cvar95=0.03,
            avg_gross_exposure=0.45,
        )

        # Act
        result = format_layer3_table(r3, holdout_start="2025-10-01", holdout_end="2026-03-31")

        # Assert
        assert "+17.0%" in result
        assert "1.17" in result
        assert "58" in result
        assert "DEPLOY-READY" in result

    def test_layer3_efficiency_and_risk_thresholds_render(self) -> None:
        """L3는 Sharpe/Sortino/CVaR/MDD threshold를 명시해야 한다."""
        r3 = SimpleNamespace(
            cagr=0.12,
            mdd=0.22,
            sharpe=-0.1,
            mar=0.55,
            cagr_baseline=0.08,
            mdd_baseline=0.24,
            sharpe_baseline=0.2,
            mar_baseline=0.33,
            gate_passed=False,
            blocker_reason="sharpe_abs",
            total_return=0.05,
            equity_multiple=1.05,
            sortino=-0.3,
            cvar95=0.07,
            avg_gross_exposure=0.4,
            n_trades=12,
            min_trades=10,
            max_mdd_abs=0.20,
            min_sharpe=0.0,
            min_sortino=0.0,
            max_cvar95=0.06,
        )

        result = format_layer3_table(r3, holdout_start="2025-10-01", holdout_end="2026-03-31")

        assert "Sharpe: -0.100 (>=0.000)" in result
        assert "Sortino: -0.300 (>=0.000)" in result
        assert "CVaR95: 7.0% (<= 6.0%)" in result
        assert "MDD: 22.0% (<= 20.0%)" in result


# ---------------------------------------------------------------------------
# S11 / S12: SWF 전환 문자열 검증
# ---------------------------------------------------------------------------


class TestFormatLayer1TableSwfStrings:
    """S11/S12: CPCV→SWF 교체 및 fold_pass_ratio 메인 테이블 미포함 검증."""

    def test_format_layer1_table_no_cpcv_string(self) -> None:
        """S11: 출력에 'CPCV' 없고 'SWF', 'CS IC' 포함, 'Mean IC (fold)' 없음."""
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
        assert "CS IC Mean" in result
        assert "Mean IC (fold)" not in result

    def test_format_layer1_table_no_fold_pass_ratio_in_main(self) -> None:
        """S12: 메인 테이블 섹션에 구식 'Fold Pass Ratio' 없고, 새 CS 행 존재."""
        # Arrange
        r = _make_l1_result()

        # Act
        result = format_layer1_table(r)
        # SWF FOLD DETAILS 이전 부분만 추출 (fold_details 미전달이므로 전체가 메인)
        main_section = result.split("[SWF FOLD DETAILS]")[0] if "[SWF FOLD DETAILS]" in result else result

        # Assert
        assert "Fold Pass Ratio" not in main_section
        assert "CS Fold Pass%" in result


# ---------------------------------------------------------------------------
# render_family_rejection_funnel (docs/specs/l1-nontrend-diversification-measure-first.md C2)
# ---------------------------------------------------------------------------


def _fake_ev(symbol: str, strategy_id: str, reasons: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        key=SignalSourceKey(
            symbol=symbol,
            strategy_id=strategy_id,
            activation_context="all",
        ),
        structural_reasons=reasons,
    )


class TestRenderFamilyRejectionFunnel:
    def test_groups_by_family(self) -> None:
        failed = [
            _fake_ev("AUSDT", "xs_oi_skew:v1", ("quality_weight_zero",)),
            _fake_ev("BUSDT", "xs_oi_skew:v1", ("quality_weight_zero",)),
            _fake_ev("CUSDT", "trend_ma:ma1", ("no_incremental_edge",)),
        ]

        lines = render_family_rejection_funnel(failed)

        joined = "\n".join(lines)
        assert "xs_oi_skew" in joined
        assert "total=2" in joined
        assert "quality_weight_zerox2" in joined
        assert "trend_ma" in joined
        assert "no_incremental_edgex1" in joined

    def test_empty_returns_empty(self) -> None:
        assert render_family_rejection_funnel([]) == []

    def test_handles_missing_colon(self) -> None:
        failed = [_fake_ev("AUSDT", "orphan", ("quality_weight_zero",))]

        lines = render_family_rejection_funnel(failed)

        assert any("orphan" in line for line in lines)

    def test_empty_reasons_fallback(self) -> None:
        failed = [_fake_ev("AUSDT", "trend_ma:ma1", ())]

        lines = render_family_rejection_funnel(failed)

        assert any("quality_weight_zerox1" in line for line in lines)


class TestFormatLayer3TableRegimeDiagnostics:
    """Scenario 4: format_layer3_table renders regime and trend efficiency diagnostic."""

    def test_renders_regime_and_trend_efficiency_diagnostic(self) -> None:
        """New diagnostic block for regime mix and Kaufman ER renders exact substrings."""
        from types import SimpleNamespace

        r3 = SimpleNamespace(
            gate_passed=True,
            regime_bull_pct=10.0,
            regime_bear_pct=25.0,
            regime_crisis_pct=65.0,
            mean_trend_efficiency=0.18,
            trend_efficiency_corr=-0.31,
        )

        result = format_layer3_table(r3, holdout_start="2025-07-01", holdout_end="2026-01-01")

        assert "bull=10.0%" in result
        assert "bear=25.0%" in result
        assert "crisis=65.0%" in result
        assert "mean=0.180" in result
        assert "corr=-0.310" in result

    def test_format_layer3_table_renders_long_short_pnl_split(self) -> None:
        """Scenario 5: [L3-LONGSHORT] 라인에 long/short price 및 bar 수가 정확히 렌더링."""
        from types import SimpleNamespace

        r3 = SimpleNamespace(
            gate_passed=True,
            mean_trend_efficiency=0.0,
            realized_price_long=-0.085,
            realized_price_short=0.012,
            bars_long=38,
            bars_short=9,
        )

        result = format_layer3_table(r3, holdout_start="2025-07-01", holdout_end="2026-01-01")

        assert "[L3-LONGSHORT]" in result
        assert "long=-0.0850" in result
        assert "short=+0.0120" in result
        assert "long=38 short=9" in result

    def test_format_layer3_table_renders_per_symbol_long_short_top_lines(self) -> None:
        """Scenario 10: [L3-LONGSHORT-TOP] 라인에 top symbol contributions 렌더링."""
        from types import SimpleNamespace

        r3 = SimpleNamespace(
            gate_passed=True,
            mean_trend_efficiency=0.0,
            realized_price_long_by_symbol=(("ARUSDT", -0.021), ("ZRXUSDT", -0.018)),
            realized_price_short_by_symbol=(("BTCUSDT", 0.012),),
        )

        result = format_layer3_table(r3, holdout_start="2025-07-01", holdout_end="2026-01-01")

        assert "[L3-LONGSHORT-TOP]" in result
        assert "ARUSDT(-0.0210)" in result
        assert "ZRXUSDT(-0.0180)" in result
        assert "BTCUSDT(+0.0120)" in result


class TestFormatTopSymbolContributions:
    """_format_top_symbol_contributions 정렬 및 truncation."""

    def test_format_top_symbol_contributions_sorts_ascending_for_losers(self) -> None:
        """Scenario 7: ascending=True → 가장 큰 손실(가장 작은 값) 순."""
        pairs = (("A", -0.01), ("B", -0.05), ("C", 0.02), ("D", -0.03))
        result = _format_top_symbol_contributions(pairs, top_n=2, ascending=True)
        assert result == "B(-0.0500) D(-0.0300)"

    def test_format_top_symbol_contributions_sorts_descending_for_winners(self) -> None:
        """Scenario 8: ascending=False → 가장 큰 이익(가장 큰 값) 순."""
        pairs = (("A", -0.01), ("B", -0.05), ("C", 0.02), ("D", -0.03))
        result = _format_top_symbol_contributions(pairs, top_n=2, ascending=False)
        assert result == "C(+0.0200) A(-0.0100)"

    def test_format_top_symbol_contributions_empty_returns_na(self) -> None:
        """Scenario 9: empty pairs → 'n/a'."""
        result = _format_top_symbol_contributions((), ascending=True)
        assert result == "n/a"


class TestFormatMajorSymbolDiagLine:
    """_format_major_symbol_diag_line 포맷 및 예외 처리."""

    def test_format_major_symbol_diag_line_success(self) -> None:
        """Scenario 1: 정상 입력 → 포맷 일치."""
        from src.domain.futures.strategy.tiered_logging import (
            _format_major_symbol_diag_line,
        )
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSignalSizingSummary,
        )

        summary = MajorSymbolSignalSizingSummary(
            symbol="BTCUSDT",
            n_obs=4,
            mu_bullish_pct=0.5,
            weight_long_pct=0.75,
            stale_long_pct=0.25,
            regime_cap_engaged_pct=0.25,
            mean_regime_risk_mult_when_long=0.93333,
        )
        result = _format_major_symbol_diag_line(summary)
        assert result == ("BTCUSDT: mu_bull=50.0% w_long=75.0% stale_long=25.0% cap_engaged=25.0% avg_mult=0.933")

    def test_format_major_symbol_diag_line_zero_obs_no_crash(self) -> None:
        """Scenario 2: n_obs=0 → 크래시 없음."""
        from src.domain.futures.strategy.tiered_logging import (
            _format_major_symbol_diag_line,
        )
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MajorSymbolSignalSizingSummary,
        )

        summary = MajorSymbolSignalSizingSummary(
            symbol="ETHUSDT",
            n_obs=0,
            mu_bullish_pct=0.0,
            weight_long_pct=0.0,
            stale_long_pct=0.0,
            regime_cap_engaged_pct=0.0,
            mean_regime_risk_mult_when_long=0.0,
        )
        result = _format_major_symbol_diag_line(summary)
        assert result == ("ETHUSDT: mu_bull=0.0% w_long=0.0% stale_long=0.0% cap_engaged=0.0% avg_mult=0.000")


class TestLayerTableOmitsMajorDiag:
    """format_layer2_table / format_layer3_table — major_diag 속성 없을 때 생략."""

    def test_format_layer2_table_omits_major_diag_lines_when_attribute_absent(self) -> None:
        """Scenario 3a: major_symbol_diag 속성 없는 최소 result → [L2-MAJOR-DIAG] 라인 없음."""
        r = SimpleNamespace(
            sharpe_hybrid=1.5,
            sharpe_baseline=1.0,
            mdd_hybrid=0.20,
            mdd_baseline=0.25,
            cagr_hybrid=0.40,
            mar_hybrid=2.0,
            fold_pass_ratio=0.8,
            turnover=5.0,
            friction_pass_pct=0.75,
            gate_passed=True,
            blocker_reason="",
            psr_hybrid=0.9,
            dsr_hybrid=0.8,
            mean_trend_efficiency=0.3,
            trend_efficiency_corr=0.1,
            realized_price_long=0.0,
            realized_price_short=0.0,
            cvar_95_hybrid=0.04,
            sortino_hybrid=2.0,
            terminal_multiple=1.5,
            total_pnl_pct=0.5,
            trade_count=50,
            risk_utilization=0.8,
        )
        result = format_layer2_table(r)
        assert "[L2-MAJOR-DIAG]" not in result

    def test_format_layer3_table_omits_major_diag_lines_when_attribute_absent(self) -> None:
        """Scenario 3b: major_symbol_diag 속성 없는 최소 L3 result → [L3-MAJOR-DIAG] 라인 없음."""
        r = SimpleNamespace(
            cagr=0.40,
            mdd=0.20,
            sharpe=1.5,
            mar=2.0,
            cagr_baseline=0.30,
            mdd_baseline=0.25,
            sharpe_baseline=1.0,
            mar_baseline=1.2,
            gate_passed=True,
            blocker_reason="",
            total_return=0.5,
            equity_multiple=1.5,
            sortino=2.0,
            sortino_baseline=1.5,
            n_trades=50,
            cvar95=0.04,
            avg_gross_exposure=1.2,
            mean_trend_efficiency=0.3,
            trend_efficiency_corr=0.1,
        )
        result = format_layer3_table(r)
        assert "[L3-MAJOR-DIAG]" not in result
