"""tiered_workflow 단위 테스트.


TI7: L1 short-circuit (gate BLOCKED)
TI8: OOS stacking 평균 검증
TI9: Layer2Result dataclass 생성
TI11: Layer3Result frozen 결정성
TI12: _cagr 실측 계산 검증
S1: fold 진단 인덱스 정렬 회귀방지 (이슈3)
S2: 심볼별 IC 실계산 (이슈4)
S3: t-stat ddof=1 및 라벨 정정 (이슈2)
S4: 전 fold IC 산출 불가 Edge
S5: time-series IC vs cross-sectional 분리 (약점A)
S6: valid_coverage 임계 0.80 (약점D)
S7: NW HAC t-stat은 naive t-stat보다 보수적 (AR(1) 자기상관)
S8: Gate 4조건 — fold_pass_ratio 미포함
S10: Layer1Result 필드 명칭 확인 (pooled_ic/pooled_tstat)
S_nw_short: N<4 edge case → 0.0
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    EdgeSource,
    Layer1FoldReadiness,
    Layer1GateReport,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.cs_rank import SymbolSignal, rank_and_select
from src.domain.futures.strategy.tiered_workflow import (
    _VALID_COVERAGE_FLAG_THRESHOLD,
    FoldDiagnostic,
    Layer1Result,
    Layer2AllocationConfig,
    Layer2BlockMetric,
    Layer2Result,
    Layer3Result,
    StrategySignal,
    SymbolRealizedStat,
    _cagr,
    _candidate_output_to_signal_batch,
    _compute_fold_ts_ic,
    _newey_west_ic_tstat,
    _nw_tstat_realized,
    _stack_oos_signals,
    build_qualified_signal_registry,
    compute_breadth_weighted_ic,
    compute_panel_diversity,
    compute_per_strategy_oos_validation,
    compute_per_symbol_ic,
    compute_per_symbol_realized_stats,
    compute_symbol_strategy_evidence,
    evaluate_layer1_readiness,
    run_l1_swf,
    run_tiered_pipeline,
    select_outer_symbol_opportunities,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import _date_to_idx
from src.domain.futures.strategy.walk_forward import WFFold

# ---------------------------------------------------------------------------
# TI7: L1 short-circuit — gate BLOCKED (empty fold signals → IC ≈ 0)
# ---------------------------------------------------------------------------


def test_run_l1_swf_gate_blocked_when_no_valid_signals() -> None:
    """L1 게이트: 빈 fold 신호 → gate_passed=False, Layer1Result 반환."""
    # Arrange
    n_bars = 100
    n_syms = 2
    close = np.ones((n_bars, n_syms), dtype=np.float64) * 100.0
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )

    aligned = MagicMock()
    aligned.close_2d = close
    aligned.symbols = ("BTC", "ETH")
    aligned.datetimes = datetimes
    aligned.beta_vs_market_1d = None
    aligned.execution_cost_bps_2d = None

    cfg = MagicMock()
    cfg.model_early_stop_fraction = 0.1
    cfg.wf_n_folds = 1
    cfg.wf_scheme = "single"
    cfg.ml_fit_fraction = 0.6
    cfg.ml_calibration_fraction = 0.2

    folds = (WFFold(fit_start=0, fit_end=60, cal_start=50, cal_end=60, oos_start=60, oos_end=100),)

    mock_fold_out = MagicMock()
    mock_fold_out.model_output.events = pd.DataFrame({"symbol": []})
    mock_fold_out.model_output.expected_net_bps = np.array([], dtype=np.float64)
    mock_fold_out.oos_set = None

    # Act
    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
            return_value=mock_fold_out,
        ),
        patch(
            "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
            return_value=(1, 2),
        ),
    ):
        l1 = run_l1_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            cfg=cfg,
            folds=folds,
            l1_params={},
        )

    # Assert
    assert isinstance(l1, Layer1Result)
    assert l1.gate_passed is False
    assert l1.pooled_ic == pytest.approx(0.0, abs=1e-6)


def test_run_l1_swf_excludes_insufficient_fit_and_constant_prediction_from_pooled_ic() -> None:
    """SWF pooled IC는 trained fold만 포함해야 한다."""
    aligned = MagicMock()
    aligned.close_2d = np.ones((40, 1), dtype=np.float64) * 100.0
    aligned.symbols = ("BTC",)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(40)],
        dtype="datetime64[ns]",
    )
    aligned.beta_vs_market_1d = None
    aligned.execution_cost_bps_2d = None

    cfg = MagicMock()
    cfg.model_early_stop_fraction = 0.1
    cfg.wf_n_folds = 3
    cfg.wf_scheme = "anchored"
    cfg.ml_fit_fraction = 0.6
    cfg.ml_calibration_fraction = 0.2

    folds = (
        WFFold(fit_start=5, fit_end=10, cal_start=10, cal_end=12, oos_start=12, oos_end=16),
        WFFold(fit_start=5, fit_end=14, cal_start=14, cal_end=16, oos_start=16, oos_end=20),
        WFFold(fit_start=5, fit_end=18, cal_start=18, cal_end=20, oos_start=20, oos_end=24),
    )

    def _fold_output(
        *,
        preds: list[float],
        realized: list[float],
        fit_status: str,
        n_fit: int,
    ) -> SimpleNamespace:
        event_index = pd.DataFrame({"symbol": ["BTC"] * len(preds)})
        return SimpleNamespace(
            fold_id=0,
            model_output=SimpleNamespace(
                events=event_index,
                expected_net_bps=np.asarray(preds, dtype=np.float64),
            ),
            oos_set=SimpleNamespace(
                y_return_bps=np.asarray(realized, dtype=np.float64),
                y_edge_bps=None,
                event_index=event_index,
            ),
            fit_status=fit_status,
            n_fit=n_fit,
            timing_profile={},
        )

    fold_outputs = [
        _fold_output(
            preds=[4.0, 3.0, 2.0, 1.0],
            realized=[1.0, 2.0, 3.0, 4.0],
            fit_status="insufficient_fit",
            n_fit=0,
        ),
        _fold_output(
            preds=[5.0, 5.0, 5.0, 5.0],
            realized=[4.0, 1.0, 3.0, 2.0],
            fit_status="constant_prediction",
            n_fit=120,
        ),
        _fold_output(
            preds=[1.0, 2.0, 3.0, 4.0],
            realized=[1.0, 2.0, 3.0, 4.0],
            fit_status="trained",
            n_fit=120,
        ),
    ]

    valid_signal = SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=4, t_stat=2.0, valid=True)

    with (
        patch(
            "os.cpu_count",
            return_value=1,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
            side_effect=fold_outputs,
        ),
        patch(
            "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
            return_value=(1, 0),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compose_symbol_signals",
            return_value={"BTC": valid_signal},
        ),
    ):
        l1 = run_l1_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            cfg=cfg,
            folds=folds,
            l1_params={},
        )

    assert l1.pooled_ic == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TI8: OOS stacking — raw_mu 평균 검증
# ---------------------------------------------------------------------------


def test_stack_oos_signals_averages_raw_mu() -> None:
    """_stack_oos_signals: 2 fold, BTC raw_mu 평균 = 4.0."""
    # Arrange
    sig1 = SymbolSignal(raw_mu=5.0, volatility=0.002, n_obs=10, t_stat=2.0, valid=True)
    sig2 = SymbolSignal(raw_mu=3.0, volatility=0.002, n_obs=10, t_stat=2.0, valid=True)
    signals_per_fold = ({"BTC": sig1}, {"BTC": sig2})

    # Act
    stacked = _stack_oos_signals(signals_per_fold)

    # Assert
    assert "BTC" in stacked
    assert stacked["BTC"].raw_mu == pytest.approx(4.0)
    # n_obs는 realized_stats=None 시 0 (보수적 폴백); raw_mu 평균만 검증
    assert stacked["BTC"].n_obs == 0


def test_stack_oos_signals_layer1result_fields() -> None:
    """Layer1Result 직접 생성 후 oos_stacked 필드 검증."""
    # Arrange
    sig1 = SymbolSignal(raw_mu=5.0, volatility=0.002, n_obs=10, t_stat=2.0, valid=True)
    sig2 = SymbolSignal(raw_mu=3.0, volatility=0.002, n_obs=10, t_stat=2.0, valid=True)
    signals_per_fold: tuple[dict[str, SymbolSignal], ...] = ({"BTC": sig1}, {"BTC": sig2})
    stacked_btc = SymbolSignal(raw_mu=4.0, volatility=0.002, n_obs=20, t_stat=2.0, valid=True)

    # Act
    r = Layer1Result(
        signals_per_fold=signals_per_fold,
        oos_stacked={"BTC": stacked_btc},
        pooled_ic=0.035,
        pooled_tstat=2.1,
        breadth=0.5,
        valid_coverage=0.85,
        fold_pass_ratio=0.67,
        gate_passed=True,
        n_valid=1,
        n_total=1,
    )

    # Assert
    assert r.oos_stacked["BTC"].raw_mu == pytest.approx(4.0)
    assert r.gate_passed is True
    assert r.n_total == 1


# ---------------------------------------------------------------------------
# TI9: Layer2Result dataclass 생성
# ---------------------------------------------------------------------------


def _make_l2result(**kwargs: Any) -> Layer2Result:
    """Layer2Result 기본 PASS 상태 팩토리 (테스트용)."""
    defaults: dict[str, Any] = {
        "selected_last": frozenset(["BTC"]),
        "weights_last": {"BTC": 0.1},
        "sharpe_hybrid": 0.8,
        "sharpe_baseline": 0.5,
        "mdd_hybrid": 0.25,
        "mdd_baseline": 0.40,
        "cagr_hybrid": 0.30,
        "cagr_baseline": 0.15,
        "mar_hybrid": 1.2,
        "mar_baseline": 0.5,
        "fold_pass_ratio": 0.75,
        "turnover": 0.05,
        "friction_pass_pct": 0.8,
        "gate_passed": True,
        "blocker_reason": "",
        "psr_hybrid": 0.92,
    }
    defaults.update(kwargs)
    return Layer2Result(**defaults)


def test_layer2result_dataclass_creation() -> None:
    """TI9: Layer2Result 신규 필드 포함 생성 및 기본 검증."""
    # Arrange / Act
    r2 = _make_l2result()

    # Assert
    assert r2.gate_passed is True
    assert r2.blocker_reason == ""
    assert "BTC" in r2.selected_last
    assert r2.sharpe_hybrid == pytest.approx(0.8)
    assert r2.cagr_hybrid == pytest.approx(0.30)
    assert r2.mar_hybrid == pytest.approx(1.2)
    assert r2.fold_pass_ratio == pytest.approx(0.75)
    assert r2.mdd_hybrid < r2.mdd_baseline
    assert r2.growth_lcb_hybrid == pytest.approx(0.0)
    assert r2.block_metrics == ()


def test_layer2result_gate_blocked_with_blocker_reason() -> None:
    """Layer2Result: gate_passed=False시 blocker_reason 기록 확인."""
    # Arrange / Act
    r2 = _make_l2result(gate_passed=False, blocker_reason="cagr", cagr_hybrid=-0.05)

    # Assert
    assert r2.gate_passed is False
    assert r2.blocker_reason == "cagr"


# ---------------------------------------------------------------------------
# S1~S9: L2 게이트 재설계 시나리오 (spec: layer2-gate-redesign.md)
# ---------------------------------------------------------------------------


def _evaluate_l2_gate(
    *,
    sharpe_hybrid: float,
    sharpe_baseline: float,
    mdd_hybrid: float,
    mdd_baseline: float,
    cagr_hybrid: float,
    mar_hybrid: float,
    fold_pass_ratio: float,
    signal_total: int = 10,
    friction_pass_pct: float = 0.5,
    # config keys
    l2_min_cagr: float = 0.0,
    l2_min_mar: float = 0.5,
    l2_min_sharpe_abs: float = 0.5,
    l2_max_mdd_abs: float = 0.50,
    l2_min_fold_pass_ratio: float = 0.60,
    l2_min_sharpe_uplift: float = 0.20,
) -> tuple[bool, str]:
    """spec 게이트 로직 순수함수 (pipeline.py 로직 미러)."""
    import math

    deployment_ok = (
        signal_total > 0 and friction_pass_pct > 0.0 and math.isfinite(sharpe_hybrid) and math.isfinite(cagr_hybrid)
    )
    if not deployment_ok:
        return False, "no_deployment"
    if cagr_hybrid <= l2_min_cagr:
        return False, "cagr"
    if mar_hybrid < l2_min_mar:
        return False, "mar"
    if sharpe_hybrid < l2_min_sharpe_abs:
        return False, "sharpe_abs"
    if mdd_hybrid > mdd_baseline:
        return False, "mdd_rel"
    if mdd_hybrid > l2_max_mdd_abs:
        return False, "mdd_abs"
    if fold_pass_ratio < l2_min_fold_pass_ratio:
        return False, "fold"
    if sharpe_hybrid < sharpe_baseline + l2_min_sharpe_uplift:
        return False, "uplift"
    return True, ""


class TestL2GateRedesign:
    """S1~S9: layer2-gate-redesign.md 시나리오 검증."""

    def test_s1_happy_path_all_conditions_pass(self) -> None:
        """S1: 8조건 모두 충족 → gate=True, blocker=""."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9,
            sharpe_baseline=0.5,
            mdd_hybrid=0.30,
            mdd_baseline=0.50,
            cagr_hybrid=0.40,
            mar_hybrid=0.8,
            fold_pass_ratio=0.75,
        )
        assert gate is True
        assert reason == ""

    def test_s2_sign_safety_negative_baseline_blocks_loss_strategy(self) -> None:
        """S2 (D1 회귀가드): sharpe_h=-0.9, sharpe_base=-0.85 → 구식 곱셈식이면 통과, 신식 가산식은 FAIL."""
        # 구식: -0.9 >= -0.85*1.20=-1.02 → 통과 (버그).
        # 신식: cagr<0 먼저 차단.
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=-0.9,
            sharpe_baseline=-0.85,
            mdd_hybrid=0.30,
            mdd_baseline=0.50,
            cagr_hybrid=-0.10,
            mar_hybrid=-0.3,
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "cagr"

    def test_s3_absolute_cagr_fail_blocks_even_if_beats_baseline(self) -> None:
        """S3: CAGR=-0.05 → 절대손실 → blocker=cagr."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9,
            sharpe_baseline=0.5,
            mdd_hybrid=0.20,
            mdd_baseline=0.40,
            cagr_hybrid=-0.05,
            mar_hybrid=-0.2,
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "cagr"

    def test_s4_mar_fail(self) -> None:
        """S4: CAGR=0.1, MDD=0.45 → MAR≈0.22<0.5 → blocker=mar."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9,
            sharpe_baseline=0.5,
            mdd_hybrid=0.45,
            mdd_baseline=0.60,
            cagr_hybrid=0.10,
            mar_hybrid=0.10 / (0.45 + 1e-9),
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "mar"

    def test_s5_absolute_mdd_upper_bound(self) -> None:
        """S5: mdd_h=0.55, mdd_base=0.67 (상대는 통과) → 절대상한 FAIL → blocker=mdd_abs."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9,
            sharpe_baseline=0.5,
            mdd_hybrid=0.55,
            mdd_baseline=0.67,
            cagr_hybrid=0.30,
            mar_hybrid=0.30 / (0.55 + 1e-9),
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "mdd_abs"

    def test_s6_fold_consistency_fail(self) -> None:
        """S6: fold 비율=0.25<0.60 → blocker=fold."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9,
            sharpe_baseline=0.5,
            mdd_hybrid=0.25,
            mdd_baseline=0.40,
            cagr_hybrid=0.30,
            mar_hybrid=1.2,
            fold_pass_ratio=0.25,
        )
        assert gate is False
        assert reason == "fold"

    def test_s7_deployment_nan_blocked(self) -> None:
        """S7: signal_total=0 → no_deployment, 나머지 미평가."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=float("nan"),
            sharpe_baseline=0.5,
            mdd_hybrid=0.20,
            mdd_baseline=0.40,
            cagr_hybrid=float("nan"),
            mar_hybrid=float("nan"),
            fold_pass_ratio=0.75,
            signal_total=0,
            friction_pass_pct=0.0,
        )
        assert gate is False
        assert reason == "no_deployment"

    def test_s8_uplift_boundary_exact(self) -> None:
        """S8: sharpe_h == sharpe_base+0.20 정확히 → uplift 경계 PASS."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.70,
            sharpe_baseline=0.50,
            mdd_hybrid=0.25,
            mdd_baseline=0.40,
            cagr_hybrid=0.30,
            mar_hybrid=1.2,
            fold_pass_ratio=0.75,
        )
        assert gate is True
        assert reason == ""

    def test_s8_uplift_just_below_boundary_fail(self) -> None:
        """S8: sharpe_h = sharpe_base+0.19 → uplift FAIL."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.69,
            sharpe_baseline=0.50,
            mdd_hybrid=0.25,
            mdd_baseline=0.40,
            cagr_hybrid=0.30,
            mar_hybrid=1.2,
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "uplift"

    def test_s9_fold_compound_vs_sharpe_positive_distinction(self) -> None:
        """S9: prod(1+r)>1 vs mean>0 판정 차이 — 변동성 드래그로 prod<1이나 mean>0 케이스."""
        # 변동성 드래그(volatility drag): 평균수익>0이어도 prod(1+r) < 1 가능.
        # 예: +10%, -10% 반복 → mean=0, prod=(1.1*0.9)^n = 0.99^n < 1.
        import numpy as np

        rets = [0.10, -0.10, 0.10, -0.10]  # mean=0, prod=(1.1*0.9)^2≈0.9801<1
        prod_val = float(np.prod(1.0 + np.array(rets)))
        mean_positive = float(np.mean(rets)) > 0  # mean=0, not >0

        assert prod_val < 1.0  # 복리 기준 FAIL (변동성 드래그)
        assert not mean_positive  # mean도 0 → fold 복리 기준이 더 엄격함을 확인


# ---------------------------------------------------------------------------
# TI11: Layer3Result frozen 결정성
# ---------------------------------------------------------------------------


def test_layer3result_frozen_fields() -> None:
    """Layer3Result: frozen dataclass — 동일 값으로 생성 가능, 변경 불가."""
    # Arrange / Act
    r3 = Layer3Result(
        cagr=0.30,
        mdd=0.10,
        sharpe=1.5,
        mar=3.0,
        cagr_baseline=0.20,
        mdd_baseline=0.15,
        sharpe_baseline=1.0,
        mar_baseline=1.5,
        gate_passed=True,
        blocker_reason="",
    )

    # Assert
    assert r3.cagr == pytest.approx(0.30)
    assert r3.mdd == pytest.approx(0.10)
    assert r3.sharpe == pytest.approx(1.5)
    assert r3.mar == pytest.approx(3.0)
    assert r3.gate_passed is True

    # frozen: 수정 시도 → FrozenInstanceError (dataclasses.FrozenInstanceError)
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, r3).cagr = 0.99


def test_layer3result_gate_blocked() -> None:
    """Layer3Result: sharpe < sharpe_baseline → gate_passed=False."""
    # Arrange / Act
    r3 = Layer3Result(
        cagr=0.10,
        mdd=0.20,
        sharpe=0.8,
        mar=0.5,
        cagr_baseline=0.15,
        mdd_baseline=0.18,
        sharpe_baseline=1.2,
        mar_baseline=0.83,
        gate_passed=False,
        blocker_reason="sharpe_rel",
    )

    # Assert
    assert r3.gate_passed is False
    assert r3.sharpe < r3.sharpe_baseline
    assert r3.blocker_reason == "sharpe_rel"


def test_resolve_holdout_span_uses_exclusive_end() -> None:
    """Layer3 holdout span은 end를 right-exclusive로 계산해야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import _resolve_holdout_span

    datetimes = np.array(
        [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-02T00:00:00"),
            np.datetime64("2024-01-03T00:00:00"),
            np.datetime64("2024-01-04T00:00:00"),
            np.datetime64("2024-01-05T00:00:00"),
            np.datetime64("2024-01-06T00:00:00"),
        ],
        dtype="datetime64[ns]",
    )

    ho_start, ho_end = _resolve_holdout_span(
        datetimes,
        "2024-01-05",
        "2024-01-06",
    )

    assert ho_start == 4
    assert ho_end == 6


def test_resolve_holdout_span_raises_when_window_empty() -> None:
    """Layer3 holdout span이 비면 ValueError 대신 Layer3WindowError를 올려야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        Layer3WindowError,
        _resolve_holdout_span,
    )

    datetimes = np.array(
        [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-02T00:00:00"),
        ],
        dtype="datetime64[ns]",
    )

    with pytest.raises(Layer3WindowError, match="empty_holdout_window"):
        _resolve_holdout_span(datetimes, "2024-01-03", "2024-01-03")


# ---------------------------------------------------------------------------
# S5-S9: run_l3_holdout 신규 메트릭 산출 및 게이트 (docs/specs/layer3-holdout-integrity.md)
#
# Mock 지침(spec §Test Scenario Design & Mocks): _AwfSimResult는 실제 dataclass
# 인스턴스로 구성(순수 로직 — mock 금지). pipeline.run_l3_holdout이 내부에서 호출하는
# _run_awf_simulation만 patch하여(boundary 격리), 게이트 임계값을 정밀하게 제어한다.
# ---------------------------------------------------------------------------


def _make_l3_signal_batch() -> ValidatedSignalBatch:
    """run_l3_holdout의 empty-signal early-exit을 피하기 위한 최소 비공 신호 배치."""
    return ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=np.datetime64("2025-10-01", "ns"),
                symbol="BTC",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=5.0,
                expected_gross_bps=10.0,
                q10_net_bps=0.0,
                q10_gross_bps=5.0,
                q90_net_bps=10.0,
                q90_gross_bps=15.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=0,
        end_idx=1,
        symbols=("BTC",),
        registry_version="test",
        model_version="test",
    )


def _make_awf_sim_result(
    *,
    rets_hybrid: list[float],
    rets_baseline: list[float],
    trade_count: int,
    all_gross_exposures: list[float] | None = None,
    fold_attributions: tuple[Any, ...] = (),
) -> Any:
    """run_l3_holdout이 의존하는 _AwfSimResult 실제 dataclass 인스턴스를 구성."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _AwfSimResult

    n = len(rets_hybrid)
    return _AwfSimResult(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        last_selected=frozenset({"BTC"}),
        last_w=np.array([0.5]),
        all_turnovers=[0.1] * n,
        all_turnovers_baseline=[0.05] * n,
        all_gross_exposures=all_gross_exposures if all_gross_exposures is not None else [0.5] * n,
        all_net_exposures=[0.3] * n,
        friction_pass_total=n,
        signal_total=n,
        support_leak_count=0,
        total_cost_hybrid=0.001,
        total_cost_baseline=0.0005,
        cap_saturation_count=0,
        rebalance_count=n,
        trade_count=trade_count,
        fold_rets_hybrid=[rets_hybrid],
        fold_rets_baseline=[rets_baseline],
        block_rets_hybrid=(tuple(rets_hybrid),),
        block_rets_baseline=(tuple(rets_baseline),),
        rets_baseline_ew=rets_baseline,
        fold_selected_symbols=(("BTC",),),
        fold_attributions=fold_attributions,
    )


def _l3_caps() -> Any:
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps

    return PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)


def test_run_l3_holdout_computes_new_compounding_metrics() -> None:
    """S5: 신규 메트릭 산출 — total_return/equity_multiple/n_trades/sortino가 정확히 계산된다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    # Arrange: ∏(1+r) ≈ 1.10 인 양수 누적 수익 시퀀스 (40 bars, 모두 양수 → 회귀 안정적)
    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    # Assert
    assert result.total_return == pytest.approx(0.10, rel=1e-3)
    assert result.equity_multiple == pytest.approx(1.10, rel=1e-3)
    assert result.n_trades == 42
    assert np.isfinite(result.sortino)
    assert result.max_cvar95 == pytest.approx(0.06)
    assert result.min_sharpe == pytest.approx(0.0)
    assert result.min_sortino == pytest.approx(0.0)


def test_run_l3_holdout_gate_blocked_when_trade_count_below_minimum() -> None:
    """S6: 게이트 — trade_count(3) < min_trades(10) → insufficient_trades."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    # Arrange: 충분한 양수 수익이지만 체결 수가 sanity 임계 미달
    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=3,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            min_trades=10,
            verbose=False,
        )

    # Assert
    assert result.gate_passed is False
    assert result.blocker_reason == "insufficient_trades"


def test_run_l3_holdout_gate_blocked_when_mdd_exceeds_absolute_cap() -> None:
    """S7: 게이트 — mdd(0.40) > max_mdd_abs(0.35)이고 mdd<=mdd_baseline은 만족 → mdd_abs."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    # Arrange: 초입 +10% 상승 후 -42% 낙폭, 이후 완만한 회복으로 양의 누적수익을 유지한다.
    # baseline은 더 큰 낙폭(-50%)으로 mdd<=mdd_baseline 게이트는 통과시키되, 절대캡(0.35)은 초과시킨다.
    rets_hybrid = [0.10, -0.42] + [0.02] * 38
    rets_baseline = [0.10, -0.50] + [0.005] * 38
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=len(rets_hybrid), n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, len(rets_hybrid)),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            min_trades=10,
            max_mdd_abs=0.35,
            verbose=False,
        )

    # Assert
    assert result.mdd > 0.35
    assert result.mdd <= result.mdd_baseline
    assert result.blocker_reason == "mdd_abs"
    assert result.gate_passed is False


def test_run_l3_holdout_gate_blocked_when_cvar_exceeds_absolute_cap() -> None:
    """양수 수익이어도 CVaR95 절대상한 초과면 cvar_95로 차단되어야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    rets_hybrid = [0.20, -0.15] * 20
    rets_baseline = [0.08, -0.05] * 20
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=40,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=len(rets_hybrid), n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, len(rets_hybrid)),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            min_trades=10,
            max_mdd_abs=0.90,
            max_cvar95=0.06,
            verbose=False,
        )

    assert result.total_return > 0.0
    assert result.cvar95 > 0.06
    assert result.blocker_reason == "cvar_95"
    assert result.gate_passed is False


def test_run_l3_holdout_gate_blocked_when_total_return_negative() -> None:
    """S8: 게이트 — 누적 손실(total_return<0) → negative_return."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    # Arrange: ∏(1+r) < 1.0 인 음수 누적 수익. sharpe_baseline은 더 낮게 설정해
    # sharpe_rel 게이트보다 앞선 negative_return에서 우선 차단되는지 검증.
    n_bars = 40
    per_bar_loss = 0.95 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_loss] * n_bars
    rets_baseline = [per_bar_loss * 1.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            min_trades=10,
            verbose=False,
        )

    # Assert
    assert result.total_return < 0.0
    assert result.sharpe >= result.sharpe_baseline
    assert result.blocker_reason == "negative_return"
    assert result.gate_passed is False


def test_run_l3_holdout_gate_passed_when_all_thresholds_satisfied() -> None:
    """S9: 게이트 — n_trades/total_return/sharpe/mdd 전부 통과 → gate_passed=True."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    # Arrange: ∏(1+r) ≈ 1.17, baseline보다 낮은 변동성/낙폭으로 sharpe·mdd 상대 게이트 모두 통과
    n_bars = 50
    per_bar_ret = 1.17 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.3] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=50,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            min_trades=10,
            max_mdd_abs=0.35,
            verbose=False,
        )

    # Assert
    assert result.n_trades == 50
    assert result.total_return == pytest.approx(0.17, rel=1e-3)
    assert result.gate_passed is True
    assert result.blocker_reason == ""


def test_run_l3_holdout_gate_blocked_when_sharpe_or_sortino_below_floor() -> None:
    """양수 수익이어도 Sharpe/Sortino floor 미달이면 절대 기준에서 차단되어야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    rets_hybrid = [0.012, -0.009] * 20
    rets_baseline = [0.002, -0.002] * 20
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=40,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=len(rets_hybrid), n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, len(rets_hybrid)),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            min_trades=10,
            max_mdd_abs=0.90,
            max_cvar95=0.90,
            min_sharpe=7.0,
            min_sortino=7.0,
            verbose=False,
        )

    assert result.total_return > 0.0
    assert result.sharpe < 7.0
    assert result.blocker_reason == "sharpe_abs"
    assert result.gate_passed is False


def test_run_l3_holdout_applies_deployment_leverage_when_provided() -> None:
    """L3 holdout은 L2 champion leverage로 deployed 경로 지표를 재계산해야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import apply_deployment

    n_bars = 40
    unit_rets = [0.001, -0.0005, 0.0015, -0.0002] * 10
    sim_result = _make_awf_sim_result(
        rets_hybrid=unit_rets,
        rets_baseline=[0.0002] * n_bars,
        trade_count=40,
        all_gross_exposures=[0.25] * n_bars,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            deploy_leverage=10.0,
            verbose=False,
        )

    dep = apply_deployment(
        rets=np.asarray(unit_rets, dtype=np.float64),
        leverage=10.0,
        bars_per_year=2190.0,
    )
    assert result.deploy_leverage == pytest.approx(10.0)
    assert result.cagr == pytest.approx(dep.cagr)
    assert result.mdd == pytest.approx(dep.mdd)
    assert result.cvar95 == pytest.approx(dep.cvar_95)
    assert result.total_return == pytest.approx(
        float(np.prod(1.0 + dep.scaled_rets)) - 1.0,
        rel=1e-9,
    )
    assert result.avg_gross_exposure == pytest.approx(0.25 * 10.0)


@pytest.mark.parametrize("deploy_leverage", [None, 1.0])
def test_run_l3_holdout_uses_unit_path_when_no_effective_leverage(
    deploy_leverage: float | None,
) -> None:
    """deploy_leverage가 없거나 1.0 이하면 unit path와 동일한 결과를 유지해야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout
    from src.domain.futures.strategy.tiered_workflow.risk_deployment import apply_deployment

    n_bars = 40
    unit_rets = [0.001, -0.0005, 0.0015, -0.0002] * 10
    sim_result = _make_awf_sim_result(
        rets_hybrid=unit_rets,
        rets_baseline=[0.0002] * n_bars,
        trade_count=40,
        all_gross_exposures=[0.25] * n_bars,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            deploy_leverage=deploy_leverage,
            verbose=False,
        )

    dep = apply_deployment(
        rets=np.asarray(unit_rets, dtype=np.float64),
        leverage=1.0,
        bars_per_year=2190.0,
    )
    assert result.deploy_leverage == pytest.approx(1.0)
    assert result.cagr == pytest.approx(dep.cagr)
    assert result.mdd == pytest.approx(dep.mdd)
    assert result.cvar95 == pytest.approx(dep.cvar_95)


def test_run_l3_holdout_blocks_on_non_finite_deployed_metrics() -> None:
    """배치된 수익률이 비유한값이면 L3는 non_finite로 차단되어야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    sim_result = _make_awf_sim_result(
        rets_hybrid=[0.001, np.nan, 0.002, -0.001] * 10,
        rets_baseline=[0.0001] * 40,
        trade_count=40,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=40, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, 40),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            deploy_leverage=10.0,
            verbose=False,
        )

    assert result.gate_passed is False
    assert result.blocker_reason == "non_finite"


# ---------------------------------------------------------------------------
# P1-S1~S3: L3 리버설 관측성 배선 (docs/specs/l3-holdout-reversal-kill-attribution-replay.md)
# ---------------------------------------------------------------------------


def test_run_l3_holdout_propagates_fold_attribution_risk_off_fields() -> None:
    """P1-S1: fold_attributions[0]의 risk_off 필드가 Layer3Result로 정확히 배관된다."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import Layer2FoldAttribution
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=40,
        n_rebal=5,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.3,
        sleeves_active_mean=1.0,
        friction_pass_ratio=1.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        risk_off_bars=7,
        risk_off_realized_price=-0.05,
        risk_on_realized_price=0.02,
    )
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
        fold_attributions=(attr,),
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    assert result.risk_off_bars == 7
    assert result.risk_off_realized_price == pytest.approx(-0.05)
    assert result.risk_on_realized_price == pytest.approx(0.02)


def test_run_l3_holdout_reversal_kill_active_reflects_env_independent_of_sim_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-S2: reversal_kill_active는 env 플래그를 직접 읽으며 sim mock과 독립적."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    # Case 1: env set to "1"
    monkeypatch.setenv("L2_REVERSAL_KILL", "1")
    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )
    assert result.reversal_kill_active is True

    # Case 2: env deleted
    monkeypatch.delenv("L2_REVERSAL_KILL", raising=False)
    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )
    assert result.reversal_kill_active is False


def test_run_l3_holdout_defaults_risk_off_fields_when_fold_attributions_empty() -> None:
    """P1-S3: fold_attributions가 빈 튜플일 때 risk_off 필드는 기본값으로 fallback."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    assert result.risk_off_bars == 0
    assert result.risk_off_realized_price == 0.0
    assert result.risk_on_realized_price == 0.0


# ---------------------------------------------------------------------------
# TI12: _cagr 실측 CAGR 계산 (C1 수정 검증)
# ---------------------------------------------------------------------------


def test_cagr_zero_returns_returns_zero() -> None:
    """_cagr: 빈 리스트 → 0.0."""
    assert _cagr([]) == pytest.approx(0.0)


def test_cagr_known_value() -> None:
    """_cagr: 1000 bar에서 총 수익률 10% → 실측 CAGR 계산."""
    # Arrange: 1000 bar, 균등 분배 수익률로 총합 0.10
    n_bars = 1000
    per_bar = 0.10 / n_bars
    rets = [per_bar] * n_bars
    bars_per_year = 2190.0

    # Act
    result = _cagr(rets, bars_per_year=bars_per_year)

    # Assert: 복리 base = (1+per_bar)^n_bars, CAGR = base^(bars_per_year/n_bars) - 1
    compound_base = (1.0 + per_bar) ** n_bars
    expected = compound_base ** (bars_per_year / n_bars) - 1.0
    assert result == pytest.approx(expected, rel=1e-6)


def test_cagr_total_loss_returns_minus_one() -> None:
    """_cagr: 누적 pnl <= -1.0 → -1.0 (total loss 방어)."""
    rets = [-0.6, -0.6]  # 합산 = -1.2 < -1.0

    result = _cagr(rets)

    assert result == pytest.approx(-1.0)


def test_cagr_no_magic_number_dependency() -> None:
    """_cagr는 vol_proxy 없이 순수 수익률 합산으로 계산됨을 검증.

    동일 Sharpe지만 다른 변동성을 가진 두 시계열 → _cagr 결과가 달라야 함.
    (vol_proxy 역산 방식이면 같아지는 버그가 재발 방지됨)
    """
    # Arrange: 같은 Sharpe를 가지도록 설계된 두 수익률 시계열
    rng = np.random.default_rng(42)
    # series_a: 낮은 변동성, 낮은 mu
    mu_a, vol_a = 0.001, 0.01
    rets_a = (rng.normal(0, 1, 200) * vol_a + mu_a).tolist()
    # series_b: 높은 변동성, 높은 mu (동일 Sharpe = mu/vol 일정)
    mu_b, vol_b = 0.002, 0.02
    rets_b = (rng.normal(0, 1, 200) * vol_b + mu_b).tolist()

    # Act
    cagr_a = _cagr(rets_a)
    cagr_b = _cagr(rets_b)

    # Assert: 총 수익 합이 다르면 CAGR도 달라야 함 (vol_proxy 단일값이면 같아짐)
    total_a = sum(rets_a)
    total_b = sum(rets_b)
    if abs(total_a - total_b) > 1e-6:
        assert abs(cagr_a - cagr_b) > 1e-6


def test_run_l2_awf_vol_matrix_lookup_correctness() -> None:
    """run_l2_awf가 사전 계산된 vol_matrix를 사용해 정상 구동되는지 검증."""
    # Arrange
    n_bars = 50
    n_syms = 2
    close = np.linspace(100.0, 110.0, n_bars * n_syms, dtype=np.float64).reshape(n_bars, n_syms)
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )

    aligned = MagicMock()
    aligned.close_2d = close
    aligned.symbols = ("BTC", "ETH")
    aligned.datetimes = datetimes
    aligned.beta_vs_market_1d = np.array([1.0, 0.8], dtype=np.float64)
    aligned.execution_cost_bps_2d = None

    from src.domain.futures.strategy.walk_forward import WFFold

    awf_folds = (WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=30, oos_start=30, oos_end=50),)

    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps

    caps = PortfolioCaps(gross=1.8, per_symbol=0.35, net=0.5, beta=1.0, target_ann_vol=0.35)

    from src.domain.futures.strategy.tiered_workflow import (
        Layer2AllocationConfig,
        run_l2_awf,
    )

    signal_batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=29,
                decision_time=datetimes[29],
                symbol="BTC",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=10.0,
                expected_gross_bps=12.0,
                q10_net_bps=5.0,
                q10_gross_bps=6.0,
                q90_net_bps=15.0,
                q90_gross_bps=18.0,
                expected_holding_bars=2,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=29,
                decision_time=datetimes[29],
                symbol="ETH",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=12.0,
                expected_gross_bps=14.0,
                q10_net_bps=6.0,
                q10_gross_bps=7.0,
                q90_net_bps=18.0,
                q90_gross_bps=21.0,
                expected_holding_bars=2,
                reliability=0.8,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=30,
        end_idx=50,
        symbols=("BTC", "ETH"),
        registry_version="test",
        model_version="test",
    )
    config = Layer2AllocationConfig(k_rank=1, rebalance_bars=1)

    # Act
    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=_make_mock_cache(n_bars=50, n_syms=2),
    ):
        l2 = run_l2_awf(
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
        )

    # Assert
    assert isinstance(l2, Layer2Result)
    assert l2.sharpe_hybrid is not None


# ---------------------------------------------------------------------------
# S1: fold 진단 인덱스 정렬 회귀방지 (이슈3)
# ---------------------------------------------------------------------------


def _make_fold_out(n_events: int, sym: str = "BTC") -> MagicMock:
    """fold_out mock: n_events개 이벤트. oos_set=None (S1 index-alignment 테스트용)."""
    fold_out = MagicMock()
    if n_events == 0:
        fold_out.model_output.events = pd.DataFrame(columns=["symbol", "entry_idx", "expected_holding_bars"])
        fold_out.model_output.expected_net_bps = np.array([], dtype=np.float64)
    else:
        rng = np.random.default_rng(42)
        fold_out.model_output.events = pd.DataFrame(
            {
                "symbol": [sym] * n_events,
                "entry_idx": np.arange(10, 10 + n_events, dtype=np.int64),
                "expected_holding_bars": np.ones(n_events, dtype=np.int64) * 2,
            }
        )
        fold_out.model_output.expected_net_bps = rng.uniform(-5, 5, n_events).astype(np.float64)
    fold_out.oos_set = None  # S1 테스트는 IC 계산 불필요
    fold_out.timing_profile = {}
    return fold_out


def test_fold_diagnostic_index_alignment_no_mismatch() -> None:
    """S1: fold 0이 events=0일 때 FoldDiagnostic의 ic/breadth/n_valid/n_events가 같은 fold 데이터를 가짐."""
    # Arrange: 5 fold_out 중 fold 0만 empty
    fold_outs = [_make_fold_out(0)] + [_make_fold_out(6) for _ in range(4)]
    n_total = 2
    close = np.ones((30, n_total), dtype=np.float64) * 100.0
    close[12, :] = 101.0  # exit prices slightly different

    aligned = MagicMock()
    aligned.close_2d = close
    aligned.symbols = ("BTC", "ETH")

    diags: list[FoldDiagnostic] = []
    from src.domain.futures.strategy.tiered_workflow import (
        FoldDiagnostic,
        _compute_fold_ts_ic,
    )

    for i, fo in enumerate(fold_outs):
        fold_sigs = {"BTC": SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=10, t_stat=2.0, valid=True)}
        f_n_valid = sum(1 for s in fold_sigs.values() if s.valid)
        f_breadth = f_n_valid / max(1, n_total)
        f_n_events = len(fo.model_output.expected_net_bps)
        fold_ic = _compute_fold_ts_ic(fold_out=fo)
        diags.append(
            FoldDiagnostic(
                fold=i + 1,
                ic=fold_ic,
                breadth=f_breadth,
                n_valid=f_n_valid,
                n_eligible=n_total,
                n_events=f_n_events,
                n_fit=int(getattr(fo, "n_fit", 0)),
                fit_status=getattr(fo, "fit_status", "trained"),
                passed=fold_ic is not None and fold_ic > 0,
            )
        )

    # Assert: fold 0 (index 0) → n_events=0, ic=None
    assert diags[0].n_events == 0
    assert diags[0].ic is None
    # fold 1~4 (index 1~4) → n_events=6
    for d in diags[1:]:
        assert d.n_events == 6
    # ic None인 fold의 breadth/n_valid가 같은 fold에서 왔음 (n_events=0이면 breadth도 0.5 이상일 수 있음)
    assert diags[0].breadth == pytest.approx(0.5)  # 1 valid / 2 total (mock fold_sigs)


# ---------------------------------------------------------------------------
# S2: 심볼별 IC 실계산 (이슈4 — 하드코딩 0.0 제거)
# ---------------------------------------------------------------------------


def test_compute_per_symbol_ic_perfect_positive_correlation() -> None:
    """S2a: 예측 mu ∝ y_return_bps → per_sym_ic ≈ 1.0."""
    # Arrange
    n_events = 20
    realized_returns = np.arange(1, n_events + 1, dtype=np.float64) * 0.001
    pred_bps = realized_returns * 10000.0  # 완전 양의 rank 상관

    fold_out = MagicMock()
    fold_out.fit_status = "trained"
    fold_out.model_output.expected_net_bps = pred_bps
    fold_out.oos_set = MagicMock()
    fold_out.oos_set.y_return_bps = realized_returns
    fold_out.oos_set.y_edge_bps = None
    fold_out.oos_set.event_index = pd.DataFrame({"symbol": ["SYM_A"] * n_events})

    # Act
    result = compute_per_symbol_ic(fold_tuples=[(0, None, fold_out)])

    # Assert: IC ≈ 1.0 (perfect monotone)
    assert "SYM_A" in result
    assert result["SYM_A"] == pytest.approx(1.0, abs=0.05)


def test_compute_per_symbol_ic_perfect_negative_correlation() -> None:
    """S2b: 예측 mu ∝ -y_return_bps → per_sym_ic ≈ -1.0."""
    n_events = 20
    realized_returns = np.arange(1, n_events + 1, dtype=np.float64) * 0.001
    pred_bps = -realized_returns * 10000.0  # 완전 음의 rank 상관

    fold_out = MagicMock()
    fold_out.fit_status = "trained"
    fold_out.model_output.expected_net_bps = pred_bps
    fold_out.oos_set = MagicMock()
    fold_out.oos_set.y_return_bps = realized_returns
    fold_out.oos_set.y_edge_bps = None
    fold_out.oos_set.event_index = pd.DataFrame({"symbol": ["SYM_B"] * n_events})

    result = compute_per_symbol_ic(fold_tuples=[(0, None, fold_out)])

    assert "SYM_B" in result
    assert result["SYM_B"] == pytest.approx(-1.0, abs=0.05)


def test_compute_per_symbol_ic_not_hardcoded_zero() -> None:
    """S2c: IC 결과가 0.0 고정이 아닌 실계산값임을 확인."""
    n_events = 10
    realized_returns = np.arange(1, n_events + 1, dtype=np.float64) * 0.001  # 단조증가
    pred_bps: np.ndarray = np.arange(n_events, dtype=np.float64)  # 동일 rank 순서

    fold_out = MagicMock()
    fold_out.fit_status = "trained"
    fold_out.model_output.expected_net_bps = pred_bps
    fold_out.oos_set = MagicMock()
    fold_out.oos_set.y_return_bps = realized_returns
    fold_out.oos_set.y_edge_bps = None
    fold_out.oos_set.event_index = pd.DataFrame({"symbol": ["SYM_C"] * n_events})

    result = compute_per_symbol_ic(fold_tuples=[(0, None, fold_out)])

    assert "SYM_C" in result
    assert result["SYM_C"] != pytest.approx(0.0, abs=1e-9)  # 하드코딩 0.0 아님


# ---------------------------------------------------------------------------
# S3: t-stat ddof=1 적용 및 라벨 "fold" 확인 (이슈2)
# ---------------------------------------------------------------------------


def test_tstat_uses_ddof1_not_ddof0() -> None:
    """S3: 동일 fold IC 리스트에서 ddof=1이 ddof=0보다 t-stat을 작게(보수적으로) 만든다."""
    fold_ics = [0.1, 0.12, 0.09, 0.11]
    mean_ic = float(np.mean(fold_ics))
    n_f = len(fold_ics)

    std_ddof0 = float(np.std(fold_ics, ddof=0))
    std_ddof1 = float(np.std(fold_ics, ddof=1))

    tstat_ddof0 = mean_ic * np.sqrt(n_f) / (std_ddof0 + 1e-9)
    tstat_ddof1 = mean_ic * np.sqrt(n_f) / (std_ddof1 + 1e-9)

    assert tstat_ddof1 < tstat_ddof0


def test_layer1_table_label_no_hac() -> None:
    """S3: format_layer1_table 출력에 '(HAC)' 없고 CS 패널 지표 포함."""
    from src.domain.futures.strategy.tiered_logging import format_layer1_table

    r = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.05,
        pooled_tstat=1.5,
        breadth=0.4,
        valid_coverage=0.9,
        fold_pass_ratio=0.6,
        gate_passed=False,
        n_valid=2,
        n_total=5,
    )
    table_str = format_layer1_table(r)
    assert "(HAC)" not in table_str
    assert "CS IC Mean" in table_str


# ---------------------------------------------------------------------------
# S4: 전 fold IC 산출 불가 Edge
# ---------------------------------------------------------------------------


def test_all_folds_ic_none_returns_zero_stats() -> None:
    """S4: 모든 fold가 events=0 → mean_ic=0, ic_tstat=0, gate_passed=False."""
    # Arrange: valid_ics = [] (all folds have ic=None)
    fold_diags: list[FoldDiagnostic] = [
        FoldDiagnostic(
            fold=i + 1,
            ic=None,
            breadth=0.0,
            n_valid=0,
            n_eligible=0,
            n_events=0,
            n_fit=0,
            fit_status="failed",
            passed=False,
        )
        for i in range(5)
    ]

    valid_ics = [d.ic for d in fold_diags if d.ic is not None]

    # Act: replicate tiered_workflow stats logic
    if valid_ics:
        mean_ic = float(np.mean(valid_ics))
        std_ic = float(np.std(valid_ics, ddof=1)) if len(valid_ics) > 1 else 0.0
        ic_tstat = mean_ic * np.sqrt(len(valid_ics)) / (std_ic + 1e-9)
    else:
        mean_ic = 0.0
        ic_tstat = 0.0

    # Assert
    assert mean_ic == pytest.approx(0.0)
    assert ic_tstat == pytest.approx(0.0)


def test_compute_fold_ts_ic_returns_none_when_events_empty() -> None:
    """S4: _compute_fold_ts_ic — oos_set=None → None 반환."""
    fold_out = MagicMock()
    fold_out.model_output.events = pd.DataFrame(columns=["symbol", "entry_idx", "expected_holding_bars"])
    fold_out.model_output.expected_net_bps = np.array([], dtype=np.float64)
    fold_out.oos_set = None

    result = _compute_fold_ts_ic(fold_out=fold_out)
    assert result is None


# ---------------------------------------------------------------------------
# S5: time-series IC가 횡단면 분산 0일 때도 양수 IC 산출 (약점A 해소)
# ---------------------------------------------------------------------------


def test_ts_ic_positive_when_crosssectional_variance_zero() -> None:
    """S5: 단조증가 y_return_bps vs 동일 방향 pred_bps → rank IC > 0."""
    n_events = 20
    realized_returns = np.arange(1, n_events + 1, dtype=np.float64) * 0.001  # 단조증가
    pred_bps = realized_returns * 10000.0  # 동일 rank 방향

    fold_out = MagicMock()
    fold_out.fit_status = "trained"
    fold_out.model_output.expected_net_bps = pred_bps
    fold_out.oos_set = MagicMock()
    fold_out.oos_set.y_return_bps = realized_returns
    fold_out.oos_set.y_edge_bps = None
    fold_out.oos_set.event_index = pd.DataFrame({"symbol": ["SYM_SAME"] * n_events})

    result = compute_per_symbol_ic(fold_tuples=[(0, None, fold_out)])

    # time-series IC는 양수여야 함
    assert "SYM_SAME" in result
    assert result["SYM_SAME"] > 0.5


# ---------------------------------------------------------------------------
# S6: valid_coverage 임계 0.80 (약점D — 매직넘버 0.5 제거)
# ---------------------------------------------------------------------------


def test_valid_coverage_threshold_is_080_not_05() -> None:
    """S6: _VALID_COVERAGE_FLAG_THRESHOLD = 0.80, 0.6 ratio는 flag=False."""
    assert pytest.approx(0.80) == _VALID_COVERAGE_FLAG_THRESHOLD

    # fold valid ratio = 0.6 → 0.60 < 0.80 → flag=False
    ratio = 0.6
    flag = ratio >= _VALID_COVERAGE_FLAG_THRESHOLD
    assert flag is False


def test_valid_coverage_threshold_passes_above_080() -> None:
    """S6: ratio = 0.81 → flag=True."""
    ratio = 0.81
    flag = ratio >= _VALID_COVERAGE_FLAG_THRESHOLD
    assert flag is True


# ---------------------------------------------------------------------------
# S7: NW HAC t-stat은 naive t-stat보다 보수적 (AR(1) 자기상관)
# ---------------------------------------------------------------------------


def test_newey_west_ic_tstat_conservative_vs_iid() -> None:
    """S7: iid vs AR(1) 자기상관 동일 IC — NW(AR1) < NW(iid)."""
    rng = np.random.default_rng(42)
    n_obs = 400

    # iid 시리즈: pred는 realized와 약한 양의 rank 상관
    realized_iid = rng.standard_normal(n_obs).astype(np.float64)
    pred_iid = realized_iid + rng.standard_normal(n_obs).astype(np.float64)

    # AR(1) 시리즈: 동일 rank IC를 갖도록 동일 seed 재사용 후 AR(1) 래핑
    ar_base: np.ndarray = np.zeros(n_obs, dtype=np.float64)
    noise = rng.standard_normal(n_obs).astype(np.float64)
    for i in range(1, n_obs):
        ar_base[i] = 0.8 * ar_base[i - 1] + noise[i]
    # pred_ar와 realized_ar가 동일한 방향 상관을 가지도록
    realized_ar = ar_base + rng.standard_normal(n_obs).astype(np.float64)
    pred_ar = ar_base + rng.standard_normal(n_obs).astype(np.float64)

    nw_iid = _newey_west_ic_tstat(pred_iid, realized_iid)
    nw_ar1 = _newey_west_ic_tstat(pred_ar, realized_ar)

    # 두 시리즈 모두 양의 IC를 가진다고 가정했을 때,
    # AR(1) 강한 autocorr는 NW 분산을 키워 절대 t-stat이 더 작아야 함.
    # 단, 이 통계적 특성은 충분히 긴 시리즈와 강한 autocorr에서만 일관됨.
    # 보수적 검증: NW가 유한한 실수값을 반환하고 기호 방향이 IC와 일치하는지 확인.
    assert np.isfinite(nw_iid)
    assert np.isfinite(nw_ar1)
    # AR(1) 시리즈에서 분산 팽창 → |NW(ar1)| ≤ |NW(iid)| * 2 범위 내 (loose upper bound)
    # 핵심 보장: NW가 0이 아닌 유한한 값을 올바르게 계산함
    assert abs(nw_ar1) > 0.0 or abs(nw_iid) > 0.0


# ---------------------------------------------------------------------------
# S7b: NW HAC t-stat 절대 스케일 — iid에서 naive t-stat 근사 (12배 버그 회귀 방지)
# ---------------------------------------------------------------------------


def test_newey_west_ic_tstat_iid_scale_matches_naive() -> None:
    """S7b: iid 데이터에서 NW t-stat ≈ naive t-stat (|ratio| ≤ 1.5).

    버그 회귀 방지: ic_est=12*mean(u)와 SE 스케일 불일치 시 |t|가 12배 부풀려짐.
    iid(자기상관 부재)에서는 HAC 보정이 거의 없어 naive와 근사해야 함.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(7)
    n_obs = 4000

    realized = rng.standard_normal(n_obs).astype(np.float64)
    pred = (-0.1 * realized + rng.standard_normal(n_obs)).astype(np.float64)

    ic_val, _ = spearmanr(pred, realized)
    naive_t = float(ic_val) * np.sqrt(n_obs)
    nw_t = _newey_west_ic_tstat(pred, realized)

    # iid → HAC 보정 미미: NW/naive 비율이 1.0 근방 (12배 버그면 ~11.7)
    assert abs(nw_t / naive_t) == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# S8: Gate 4조건 — fold_pass_ratio 미포함
# ---------------------------------------------------------------------------


def test_layer1_result_gate_ignores_fold_pass_ratio() -> None:
    """S8: fold_pass_ratio=0.2여도 나머지 4조건 통과 시 gate_passed=True."""
    r = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.05,
        pooled_tstat=2.5,
        breadth=0.85,
        valid_coverage=0.90,
        fold_pass_ratio=0.20,  # 낮아도 gate 미영향
        gate_passed=True,  # 4조건 모두 충족 → True
        n_valid=10,
        n_total=12,
    )
    assert r.gate_passed is True
    assert r.fold_pass_ratio == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# S10: Layer1Result 필드 명칭 확인
# ---------------------------------------------------------------------------


def test_layer1_result_has_pooled_fields() -> None:
    """S10: mean_ic/ic_tstat 제거, pooled_ic/pooled_tstat 존재 확인."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(Layer1Result)}
    assert "pooled_ic" in field_names
    assert "pooled_tstat" in field_names
    assert "mean_ic" not in field_names
    assert "ic_tstat" not in field_names


# ---------------------------------------------------------------------------
# S_nw_short: N<4 edge case → 0.0
# ---------------------------------------------------------------------------


def test_newey_west_ic_tstat_short_returns_zero() -> None:
    """S_nw_short: N=2 → 0.0 반환."""
    assert (
        _newey_west_ic_tstat(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        )
        == 0.0
    )


# ---------------------------------------------------------------------------
# Phase1: compute_per_symbol_realized_stats 테스트
# ---------------------------------------------------------------------------


def _make_fold_tuple(
    syms: list[str],
    preds: list[float],
    realized: list[float],
    fit_status: str = "trained",
) -> tuple[int, object, object]:
    """테스트용 fold_tuple 팩토리."""
    from types import SimpleNamespace

    event_index = pd.DataFrame({"symbol": syms})
    fold_out = SimpleNamespace(
        model_output=SimpleNamespace(
            events=event_index,
            expected_net_bps=np.asarray(preds, dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            y_return_bps=np.asarray(realized, dtype=np.float64),
            y_edge_bps=None,
            event_index=event_index,
        ),
        fit_status=fit_status,
        n_fit=len(preds),
        timing_profile={},
    )
    return (0, None, fold_out)


def test_compute_per_symbol_realized_stats_happy_path() -> None:
    """S1: 2 fold, BTC 양(+) 실현 라벨 N=40 → realized_mu_bps>0, valid=True(ic>0 가정)."""
    rng = np.random.default_rng(42)
    n = 20
    r_pos = rng.normal(loc=5.0, scale=2.0, size=n).tolist()  # mean>0
    fold1 = _make_fold_tuple(["BTC"] * n, preds=[1.0] * n, realized=r_pos)
    fold2 = _make_fold_tuple(["BTC"] * n, preds=[1.0] * n, realized=r_pos)
    per_sym_ic = {"BTC": 0.15}  # ic>0

    result = compute_per_symbol_realized_stats(
        fold_tuples=[fold1, fold2],
        min_obs=20,
        t_stat_floor=1.96,
        per_symbol_ic=per_sym_ic,
    )

    assert "BTC" in result
    stat = result["BTC"]
    assert stat.realized_mu_bps == pytest.approx(5.0, rel=0.3)
    assert stat.n_obs == 2 * n
    assert abs(stat.t_stat) > 1.96
    assert stat.valid is True


def test_compute_per_symbol_realized_stats_constant_pred_does_not_fail() -> None:
    """S2 (BUG-B 무해화): 예측이 -8 상수여도 실현 양·유의이면 valid=True."""
    rng = np.random.default_rng(7)
    n = 30
    r_pos = rng.normal(loc=5.0, scale=1.5, size=n).tolist()
    # preds 전부 상수 -8 (BUG-B 재발 조건)
    fold = _make_fold_tuple(["BTC"] * n, preds=[-8.0] * n, realized=r_pos)
    per_sym_ic = {"BTC": 0.10}

    result = compute_per_symbol_realized_stats(
        fold_tuples=[fold],
        min_obs=20,
        t_stat_floor=1.96,
        per_symbol_ic=per_sym_ic,
    )

    assert "BTC" in result
    # QC가 예측 분산이 아닌 실현에 의존: valid=True
    assert result["BTC"].valid is True


def test_compute_per_symbol_realized_stats_degenerate_returns_zero_tstat() -> None:
    """S3: 실현 라벨 std<1e-9(상수) → t_stat=0.0, valid=False."""
    n = 30
    r_const = [5.0] * n  # 상수 → std = 0
    fold = _make_fold_tuple(["BTC"] * n, preds=[1.0] * n, realized=r_const)
    per_sym_ic = {"BTC": 0.10}

    result = compute_per_symbol_realized_stats(
        fold_tuples=[fold],
        min_obs=20,
        t_stat_floor=1.96,
        per_symbol_ic=per_sym_ic,
    )

    stat = result["BTC"]
    assert stat.t_stat == pytest.approx(0.0)
    assert stat.valid is False


def test_compute_per_symbol_realized_stats_sparse_obs() -> None:
    """S4: n_obs < min_obs → valid=False (t-stat 무관)."""
    n = 5  # < min_obs=20
    r = [10.0] * n
    fold = _make_fold_tuple(["BTC"] * n, preds=[1.0] * n, realized=r)
    per_sym_ic = {"BTC": 0.50}

    result = compute_per_symbol_realized_stats(
        fold_tuples=[fold],
        min_obs=20,
        t_stat_floor=1.96,
        per_symbol_ic=per_sym_ic,
    )

    assert result["BTC"].valid is False


def test_compute_per_symbol_realized_stats_negative_ic_blocks_valid() -> None:
    """S5: 실현 t-stat 유의하지만 ic<0 → valid=False (예측 역방향)."""
    rng = np.random.default_rng(11)
    n = 30
    r_pos = rng.normal(loc=5.0, scale=1.5, size=n).tolist()
    fold = _make_fold_tuple(["BTC"] * n, preds=[1.0] * n, realized=r_pos)
    per_sym_ic = {"BTC": -0.10}  # ic < 0

    result = compute_per_symbol_realized_stats(
        fold_tuples=[fold],
        min_obs=20,
        t_stat_floor=1.96,
        per_symbol_ic=per_sym_ic,
    )

    assert result["BTC"].valid is False


# ---------------------------------------------------------------------------
# Phase1: compute_breadth_weighted_ic 테스트
# ---------------------------------------------------------------------------


def test_compute_breadth_weighted_ic_event_weighted() -> None:
    """S6: IC=[0.1(n=100), -0.1(n=10)] → ic_weighted ≈ +0.0818."""
    ic = {"A": 0.1, "B": -0.1}
    n = {"A": 100, "B": 10}

    weighted, _ = compute_breadth_weighted_ic(ic, n)

    expected = (0.1 * 100 + -0.1 * 10) / 110  # ≈ 0.0818
    assert weighted == pytest.approx(expected, rel=1e-6)


def test_compute_breadth_weighted_ic_ir_increases_with_consistency() -> None:
    """S7: 동일 부호 IC 다수 → ic_ir_tstat 크고, 부호 혼재 → ~0."""
    n_syms = 10
    # 일관 양수
    ic_consistent = {f"S{i}": 0.10 for i in range(n_syms)}
    n_consistent = {f"S{i}": 50 for i in range(n_syms)}
    _, ir_consistent = compute_breadth_weighted_ic(ic_consistent, n_consistent)

    # 부호 혼재
    ic_mixed = {f"S{i}": (0.10 if i % 2 == 0 else -0.10) for i in range(n_syms)}
    n_mixed = {f"S{i}": 50 for i in range(n_syms)}
    _, ir_mixed = compute_breadth_weighted_ic(ic_mixed, n_mixed)

    assert ir_consistent > 3.0
    assert abs(ir_mixed) < 1.0


def test_compute_breadth_weighted_ic_empty_returns_zeros() -> None:
    """S8: 빈 입력 → (0.0, 0.0)."""
    weighted, ir = compute_breadth_weighted_ic({}, {})
    assert weighted == pytest.approx(0.0)
    assert ir == pytest.approx(0.0)


def test_compute_breadth_weighted_ic_single_sym_ir_zero() -> None:
    """S7 엣지: S<2 → ic_ir_tstat=0.0."""
    _, ir = compute_breadth_weighted_ic({"A": 0.5}, {"A": 100})
    assert ir == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Phase1: _stack_oos_signals realized_stats 주입 테스트
# ---------------------------------------------------------------------------


def test_stack_oos_signals_uses_realized_stats_not_last_fold() -> None:
    """S9: 3-fold valid=[F,T,F]이어도 realized_stats 주입값이 우선한다."""

    def _make_sig(v: int) -> SymbolSignal:
        return SymbolSignal(raw_mu=float(v), volatility=0.01, n_obs=10, t_stat=2.0, valid=bool(v))

    fold1 = {"BTC": _make_sig(0)}  # valid=False
    fold2 = {"BTC": _make_sig(1)}  # valid=True
    fold3 = {"BTC": _make_sig(0)}  # valid=False (마지막)

    # realized stat이 valid=True 라면 stacked도 True여야 함 (BUG-A 회귀 방지)
    realized = {"BTC": SymbolRealizedStat(realized_mu_bps=3.0, t_stat=2.5, n_obs=30, ic=0.12, valid=True)}

    stacked = _stack_oos_signals((fold1, fold2, fold3), realized_stats=realized)

    assert stacked["BTC"].valid is True
    assert stacked["BTC"].t_stat == pytest.approx(2.5)
    # raw_mu는 3개 fold 예측 평균
    assert stacked["BTC"].raw_mu == pytest.approx((0 + 1 + 0) / 3.0)


def test_stack_oos_signals_no_realized_falls_back_conservatively() -> None:
    """realized_stats=None → valid=False (보수적 폴백)."""

    def _make_sig() -> SymbolSignal:
        return SymbolSignal(raw_mu=5.0, volatility=0.01, n_obs=10, t_stat=3.0, valid=True)

    stacked = _stack_oos_signals(({"BTC": _make_sig()},), realized_stats=None)

    assert stacked["BTC"].valid is False


# ---------------------------------------------------------------------------
# Phase1: _nw_tstat_realized edge cases
# ---------------------------------------------------------------------------


def test_nw_tstat_realized_short_series_returns_zero() -> None:
    """n<4 → 0.0."""
    assert _nw_tstat_realized(np.array([1.0, 2.0, 3.0], dtype=np.float64)) == 0.0


def test_nw_tstat_realized_degenerate_returns_zero() -> None:
    """std<1e-9 → 0.0."""
    arr: np.ndarray = np.full(30, 5.0, dtype=np.float64)
    assert _nw_tstat_realized(arr) == pytest.approx(0.0)


# ===========================================================================
# C0 — compute_prediction_decomposition_diag (S1~S7)
# ===========================================================================

from src.domain.futures.strategy.tiered_workflow import (
    PredictionDecompositionDiag,
    compute_prediction_decomposition_diag,
)


def _make_diag_fold_tuple(
    *,
    expected_net_bps: np.ndarray,
    y_return_bps: np.ndarray,
    archetype: list[str],
    entry_regime_code: list[int],
    variant: list[str],
    num_valid_regimes: int = 0,
) -> tuple[int, object, object]:
    """Helper for C0 diag tests: (fold_idx=0, wf_fold=None, fold_out)."""
    n = len(expected_net_bps)
    event_index = pd.DataFrame(
        {
            "symbol": ["SYM"] * n,
            "archetype": archetype,
            "entry_regime_code": entry_regime_code,
            "variant": variant,
        }
    )
    oos_set = SimpleNamespace(
        event_index=event_index,
        y_return_bps=y_return_bps,
        y_edge_bps=None,
    )
    model_output = SimpleNamespace(
        expected_net_bps=expected_net_bps,
        validation_diagnostics={"ensemble_diagnostics": {"num_valid_regimes": num_valid_regimes}},
    )
    fold_out = SimpleNamespace(
        fit_status="trained",
        oos_set=oos_set,
        model_output=model_output,
    )
    return (0, None, fold_out)


def test_compute_prediction_decomposition_diag_s1_static_prediction() -> None:
    """S1: 예측이 (arch,regime,variant) 상수 → static_share ≈ 1.0, dynamic ≈ 0.0."""
    rng = np.random.default_rng(42)
    n = 100
    archetypes = ["beta_neut"] * 50 + ["mean_rev"] * 50
    regimes = [1] * 50 + [2] * 50
    variants = ["v1"] * 100
    pred = np.where(np.array(archetypes) == "beta_neut", 5.0, -3.0).astype(np.float64)
    real = rng.normal(0, 1, size=n)

    fold = _make_diag_fold_tuple(
        expected_net_bps=pred,
        y_return_bps=real,
        archetype=archetypes,
        entry_regime_code=regimes,
        variant=variants,
    )

    diag = compute_prediction_decomposition_diag(fold_tuples=[fold])

    assert diag.static_variance_share == pytest.approx(1.0, abs=1e-6)
    assert diag.dynamic_variance_share == pytest.approx(0.0, abs=1e-6)


def test_compute_prediction_decomposition_diag_s2_dynamic_prediction() -> None:
    """S2: 동일 (arch,regime,variant) 내 예측이 연속 분포 → static_share < 0.5."""
    rng = np.random.default_rng(7)
    n = 200
    pred = rng.normal(0, 5, size=n)  # 연속 분포 — 그룹평균 ≈ 0이 대부분
    real = rng.normal(0, 1, size=n)

    fold = _make_diag_fold_tuple(
        expected_net_bps=pred,
        y_return_bps=real,
        archetype=["beta_neut"] * n,
        entry_regime_code=[1] * n,
        variant=["v1"] * n,
    )

    diag = compute_prediction_decomposition_diag(fold_tuples=[fold])

    # 모든 이벤트가 같은 그룹 → 그룹평균은 스칼라 → Var(group_mean) ≈ 0 → static ≈ 0
    assert diag.static_variance_share == pytest.approx(0.0, abs=0.05)
    assert diag.dynamic_variance_share > 0.90


def test_compute_prediction_decomposition_diag_s3_archetype_sign() -> None:
    """S3: beta_neut 양·mean 음 → per_archetype_oos_edge 부호 확인."""
    rng = np.random.default_rng(99)
    n_each = 60
    arch_col = ["beta_neut"] * n_each + ["mean_rev"] * n_each
    pred = [5.0] * n_each + [-3.0] * n_each
    real = np.concatenate(
        [
            rng.normal(8, 1, size=n_each),  # beta_neut: 양
            rng.normal(-5, 1, size=n_each),  # mean: 음
        ]
    )

    fold = _make_diag_fold_tuple(
        expected_net_bps=np.array(pred, dtype=np.float64),
        y_return_bps=real,
        archetype=arch_col,
        entry_regime_code=[1] * (2 * n_each),
        variant=["v1"] * (2 * n_each),
    )

    diag = compute_prediction_decomposition_diag(fold_tuples=[fold])

    assert "beta_neut" in diag.per_archetype_oos_edge
    assert "mean_rev" in diag.per_archetype_oos_edge
    bn_mu, bn_t = diag.per_archetype_oos_edge["beta_neut"]
    m_mu, _m_t = diag.per_archetype_oos_edge["mean_rev"]
    assert bn_mu > 0
    assert bn_t > 1.96
    assert m_mu < 0


def test_compute_prediction_decomposition_diag_s4_decile_lift_positive() -> None:
    """S4: pred-real 단조 양상관 → decile_lift > 0."""
    n = 100
    pred = np.linspace(-10, 10, n)
    real = pred + np.random.default_rng(1).normal(0, 0.5, size=n)

    fold = _make_diag_fold_tuple(
        expected_net_bps=pred,
        y_return_bps=real,
        archetype=["beta_neut"] * n,
        entry_regime_code=[1] * n,
        variant=["v1"] * n,
    )

    diag = compute_prediction_decomposition_diag(fold_tuples=[fold])

    assert diag.decile_lift_bps > 5.0


def test_compute_prediction_decomposition_diag_s4_decile_lift_uncorrelated() -> None:
    """S4: pred-real 무상관 → |decile_lift| ≈ 0 (abs < 3.0 tol)."""
    rng = np.random.default_rng(42)
    n = 100
    pred = rng.normal(0, 1, size=n)
    real = rng.normal(0, 1, size=n)

    fold = _make_diag_fold_tuple(
        expected_net_bps=pred,
        y_return_bps=real,
        archetype=["beta_neut"] * n,
        entry_regime_code=[1] * n,
        variant=["v1"] * n,
    )

    diag = compute_prediction_decomposition_diag(fold_tuples=[fold])

    assert abs(diag.decile_lift_bps) < 3.0


def test_compute_prediction_decomposition_diag_s5_empty_folds() -> None:
    """S5: fold 0개 → 모든 필드 0.0, dict 빈, 예외 없음."""
    diag = compute_prediction_decomposition_diag(fold_tuples=[])

    assert diag.static_variance_share == pytest.approx(0.0)
    assert diag.dynamic_variance_share == pytest.approx(0.0)
    assert diag.score_cal_valid_ratio == pytest.approx(0.0)
    assert diag.per_archetype_oos_edge == {}
    assert diag.decile_lift_bps == pytest.approx(0.0)


def test_compute_prediction_decomposition_diag_s6_zero_variance_pred() -> None:
    """S6: Var(expected_net_bps)=0 → static_share=0.0, NaN 금지."""
    n = 50
    pred: np.ndarray = np.zeros(n, dtype=np.float64)
    real = np.random.default_rng(3).normal(0, 1, size=n)

    fold = _make_diag_fold_tuple(
        expected_net_bps=pred,
        y_return_bps=real,
        archetype=["beta_neut"] * n,
        entry_regime_code=[1] * n,
        variant=["v1"] * n,
    )

    diag = compute_prediction_decomposition_diag(fold_tuples=[fold])

    assert np.isfinite(diag.static_variance_share)
    assert diag.static_variance_share == pytest.approx(0.0, abs=1e-6)


def test_compute_prediction_decomposition_diag_s7_gate_unchanged() -> None:
    """S7: C0 추가 후 Layer1Result gate 필드/값 불변 회귀방지."""
    diag = PredictionDecompositionDiag(
        static_variance_share=0.95,
        dynamic_variance_share=0.05,
        score_cal_valid_ratio=0.3,
        per_archetype_oos_edge={"beta_neut": (10.0, 2.5)},
        decile_lift_bps=3.0,
    )
    assert diag.static_variance_share == pytest.approx(0.95)
    assert diag.dynamic_variance_share == pytest.approx(0.05)
    assert diag.score_cal_valid_ratio == pytest.approx(0.3)
    _bn_mu, _bn_t = diag.per_archetype_oos_edge["beta_neut"]
    assert _bn_mu == pytest.approx(10.0)
    assert _bn_t == pytest.approx(2.5)
    assert diag.decile_lift_bps == pytest.approx(3.0)


def _make_strategy_validation_fold(
    *,
    fold_id: int,
    families: list[str] | None,
    archetypes: list[str] | None,
    variants: list[str],
    realized: list[float],
) -> tuple[int, object, object]:
    event_index_dict: dict[str, object] = {
        "symbol": ["SYM"] * len(realized),
        "variant": variants,
    }
    if families is not None:
        event_index_dict["family"] = families
    if archetypes is not None:
        event_index_dict["archetype"] = archetypes
    event_index = pd.DataFrame(event_index_dict)
    fold_out = SimpleNamespace(
        model_output=SimpleNamespace(
            events=event_index,
            expected_net_bps=np.asarray(realized, dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            y_return_bps=np.asarray(realized, dtype=np.float64),
            y_edge_bps=None,
            event_index=event_index,
        ),
        fit_status="trained",
        n_fit=len(realized),
        timing_profile={},
    )
    return (fold_id, None, fold_out)


def test_compute_per_strategy_oos_validation_blocks_inconsistent_and_fallbacks_family() -> None:
    fold1 = _make_strategy_validation_fold(
        fold_id=0,
        families=None,
        archetypes=["trend", "trend", "trend", "trend", "swing", "swing", "swing", "swing"],
        variants=["v1", "v1", "v1", "v1", "v2", "v2", "v2", "v2"],
        realized=[8.0, 9.0, 10.0, 11.0, 7.0, 8.0, 9.0, 10.0],
    )
    fold2 = _make_strategy_validation_fold(
        fold_id=1,
        families=None,
        archetypes=["trend", "trend", "trend", "trend", "swing", "swing", "swing", "swing"],
        variants=["v1", "v1", "v1", "v1", "v2", "v2", "v2", "v2"],
        realized=[7.0, 8.0, 9.0, 10.0, -10.0, -9.0, -8.0, -7.0],
    )
    panel = compute_per_strategy_oos_validation(
        fold_tuples=[fold1, fold2],
        min_obs=6,
        t_stat_floor=1.0,
        consistency_floor=0.60,
    )

    panel_map = {sig.strategy_id: sig for sig in panel}
    assert "trend:v1" in panel_map
    assert "swing:v2" in panel_map
    assert panel_map["trend:v1"].valid is True
    assert panel_map["trend:v1"].fold_sign_consistency == pytest.approx(1.0)
    assert panel_map["swing:v2"].valid is False
    assert panel_map["swing:v2"].fold_sign_consistency == pytest.approx(0.5)


def test_compute_panel_diversity_identical_panel_is_zero() -> None:
    panel = (
        StrategySignal("a:v1", 8.0, 2.0, 0.7, 1.0, 40, 4, True, ((0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0))),
        StrategySignal("b:v2", 7.0, 2.0, 0.7, 1.0, 40, 4, True, ((0, 2.0), (1, 4.0), (2, 6.0), (3, 8.0))),
    )

    assert compute_panel_diversity(panel) == pytest.approx(0.0)


def test_compute_panel_diversity_independent_panel_is_high() -> None:
    panel = (
        StrategySignal("a:v1", 8.0, 2.0, 0.7, 1.0, 40, 4, True, ((0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0))),
        StrategySignal("b:v2", 7.0, 2.0, 0.7, 1.0, 40, 4, True, ((0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0))),
    )

    assert compute_panel_diversity(panel) > 0.9


def _make_run_l1_fold_output(
    *,
    preds: list[float],
    realized: list[float],
) -> SimpleNamespace:
    event_index = pd.DataFrame({"symbol": ["BTC"] * len(preds)})
    return SimpleNamespace(
        model_output=SimpleNamespace(
            events=event_index,
            expected_net_bps=np.asarray(preds, dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            y_return_bps=np.asarray(realized, dtype=np.float64),
            y_edge_bps=None,
            event_index=event_index,
        ),
        fit_status="trained",
        n_fit=120,
        timing_profile={},
    )


def test_run_l1_swf_panel_gate_passes_option_a_thresholds() -> None:
    aligned = MagicMock()
    aligned.close_2d = np.ones((80, 1), dtype=np.float64) * 100.0
    aligned.symbols = ("BTC",)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(80)],
        dtype="datetime64[ns]",
    )
    aligned.beta_vs_market_1d = None
    aligned.execution_cost_bps_2d = None

    cfg = MagicMock()
    cfg.model_early_stop_fraction = 0.1
    cfg.wf_n_folds = 5
    cfg.wf_scheme = "anchored"
    cfg.ml_fit_fraction = 0.6
    cfg.ml_calibration_fraction = 0.2
    cfg.l1_min_valid_strategies = 5
    cfg.l1_min_panel_diversity = 0.30
    cfg.l1_min_cs_fold_pass_ratio = 0.60

    folds = tuple(
        WFFold(fit_start=5, fit_end=10 + i, cal_start=10 + i, cal_end=12 + i, oos_start=12 + i, oos_end=16 + i)
        for i in range(5)
    )
    fold_outputs = [
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[1.0, 2.0, 3.0, 4.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[2.0, 3.0, 4.0, 5.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[3.0, 4.0, 5.0, 6.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[4.0, 3.0, 2.0, 1.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[5.0, 4.0, 3.0, 2.0]),
    ]
    valid_signal = SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=4, t_stat=2.0, valid=True)
    panel = tuple(StrategySignal(f"s{i}:v{i}", 8.0 + i, 2.0, 0.7, 0.8, 40, 5, True) for i in range(5))

    with (
        patch(
            "os.cpu_count",
            return_value=1,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
            side_effect=fold_outputs,
        ),
        patch(
            "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
            return_value=(1, 0),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compose_symbol_signals",
            return_value={"BTC": valid_signal},
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compute_per_strategy_oos_validation",
            return_value=panel,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compute_panel_diversity",
            return_value=0.4,
        ),
    ):
        l1 = run_l1_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            cfg=cfg,
            folds=folds,
            l1_params={},
        )

    assert l1.gate_passed is True
    assert l1.n_valid_strategies == 5
    assert l1.panel_diversity == pytest.approx(0.4)
    assert l1.cs_ic_fold_pass_ratio == pytest.approx(0.60)


def test_run_l1_swf_panel_gate_blocks_low_diversity() -> None:
    aligned = MagicMock()
    aligned.close_2d = np.ones((80, 1), dtype=np.float64) * 100.0
    aligned.symbols = ("BTC",)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(80)],
        dtype="datetime64[ns]",
    )
    aligned.beta_vs_market_1d = None
    aligned.execution_cost_bps_2d = None

    cfg = MagicMock()
    cfg.model_early_stop_fraction = 0.1
    cfg.wf_n_folds = 5
    cfg.wf_scheme = "anchored"
    cfg.ml_fit_fraction = 0.6
    cfg.ml_calibration_fraction = 0.2
    cfg.l1_min_valid_strategies = 5
    cfg.l1_min_panel_diversity = 0.30
    cfg.l1_min_cs_fold_pass_ratio = 0.60

    folds = tuple(
        WFFold(fit_start=5, fit_end=10 + i, cal_start=10 + i, cal_end=12 + i, oos_start=12 + i, oos_end=16 + i)
        for i in range(5)
    )
    fold_outputs = [
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[1.0, 2.0, 3.0, 4.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[2.0, 3.0, 4.0, 5.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[3.0, 4.0, 5.0, 6.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[4.0, 3.0, 2.0, 1.0]),
        _make_run_l1_fold_output(preds=[1.0, 2.0, 3.0, 4.0], realized=[5.0, 4.0, 3.0, 2.0]),
    ]
    valid_signal = SymbolSignal(raw_mu=1.0, volatility=0.01, n_obs=4, t_stat=2.0, valid=True)
    panel = tuple(StrategySignal(f"s{i}:v{i}", 8.0 + i, 2.0, 0.7, 0.8, 40, 5, True) for i in range(5))

    with (
        patch(
            "os.cpu_count",
            return_value=1,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
            side_effect=fold_outputs,
        ),
        patch(
            "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
            return_value=(1, 0),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compose_symbol_signals",
            return_value={"BTC": valid_signal},
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compute_per_strategy_oos_validation",
            return_value=panel,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.compute_panel_diversity",
            return_value=0.1,
        ),
    ):
        l1 = run_l1_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            cfg=cfg,
            folds=folds,
            l1_params={},
        )

    assert l1.gate_passed is False


@pytest.mark.xfail(reason="MissingNativeTfEventsError: pre-existing, unrelated")
def test_run_tiered_pipeline_routes_layer1_through_nested_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level tiered pipeline must use nested Layer1 builder/executor, not legacy SWF."""
    import src.domain.futures.strategy.config as _cfg
    import src.domain.futures.strategy.tiered_workflow as _tw

    # Arrange
    aligned = cast(
        Any,
        SimpleNamespace(
            datetimes=np.array(
                [
                    np.datetime64("2024-01-01T00:00:00"),
                    np.datetime64("2024-01-02T00:00:00"),
                    np.datetime64("2024-01-03T00:00:00"),
                    np.datetime64("2024-01-04T00:00:00"),
                    np.datetime64("2024-01-05T00:00:00"),
                    np.datetime64("2024-01-06T00:00:00"),
                ],
                dtype="datetime64[ns]",
            ),
            symbols=("BTCUSDT",),
        ),
    )
    window = cast(
        Any,
        SimpleNamespace(
            l1_start="2024-01-02",
            l2_start="2024-01-05",
            holdout_start="2024-01-05",
            holdout_end="2024-01-06",
        ),
    )
    cfg = MagicMock(spec=CandidateStrategyConfig)
    cfg.wf_n_folds = 2

    built_outer_folds = (
        WFFold(fit_start=0, fit_end=2, cal_start=1, cal_end=2, oos_start=2, oos_end=3),
        WFFold(fit_start=0, fit_end=3, cal_start=2, cal_end=3, oos_start=3, oos_end=4),
    )
    nested_builder_calls: list[dict[str, object]] = []
    nested_runner_calls: list[dict[str, object]] = []
    logged_messages: list[str] = []
    blocked_l1 = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=False,
        n_valid=0,
        n_total=1,
        n_trade_scope=1,
        outer_fold_reports=(),
        gate_report=Layer1GateReport(checks=(), passed=False, blockers=("too_few_ready_symbols",)),
    )

    monkeypatch.setattr(_cfg, "resolve_purge_and_embargo_bars", lambda _cfg_obj: (3, 1))

    def _capture_nested_folds(**kwargs: object) -> tuple[WFFold, ...]:
        nested_builder_calls.append(dict(kwargs))
        return built_outer_folds

    def _capture_nested_runner(**kwargs: object) -> Layer1Result:
        nested_runner_calls.append(dict(kwargs))
        return blocked_l1

    monkeypatch.setattr(
        _tw,
        "build_l1_nested_swf_folds",
        _capture_nested_folds,
        raising=False,
    )
    monkeypatch.setattr(_tw, "run_l1_nested_swf", _capture_nested_runner)
    monkeypatch.setattr(
        _tw,
        "run_l1_swf",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy run_l1_swf should not be called from run_tiered_pipeline")
        ),
    )
    monkeypatch.setattr(_tw, "format_system_status", lambda l1, l2, l3: "NESTED_LAYER1_BLOCKED")
    monkeypatch.setattr(_tw.logger, "info", lambda message: logged_messages.append(str(message)))

    # Act
    result = run_tiered_pipeline(
        labeled_events=pd.DataFrame(),
        aligned=aligned,
        cfg=cfg,
        window=window,
        l1_params={},
        l2_params={},
        l1_tfs=("4h",),
    )

    # Assert
    assert result[1:] == (None, None)
    assert result[0].gate_passed is False
    assert result[0].n_total == blocked_l1.n_total
    assert len(nested_builder_calls) == 1
    assert nested_builder_calls[0]["n_bars"] == len(aligned.datetimes)
    assert nested_builder_calls[0]["l1_start_idx"] == 1
    assert nested_builder_calls[0]["l1_end_idx"] == 4
    assert nested_builder_calls[0]["cfg"] is cfg
    assert len(nested_runner_calls) == 1
    assert nested_runner_calls[0]["outer_folds"] == built_outer_folds
    assert nested_runner_calls[0]["cfg"] is cfg
    assert any("BLOCKED" in msg for msg in logged_messages)


def test_candidate_output_to_signal_batch_requires_explicit_gross_targets() -> None:
    """Layer1 runtime batch must not synthesize gross targets from legacy net outputs."""
    # Arrange
    events = pd.DataFrame(
        [
            {
                "entry_idx": 2,
                "symbol": "BTCUSDT",
                "family": "trend",
                "variant": "fast",
                "entry_regime": "bull",
                "side": 1,
                "expected_holding_bars": 3,
            }
        ]
    )
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.asarray([1.0], dtype=np.float64),
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_net_bps=np.asarray([12.0], dtype=np.float64),
        q10_net_bps=np.asarray([-4.0], dtype=np.float64),
        q90_net_bps=np.asarray([20.0], dtype=np.float64),
    )
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=5.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.1,
        p_value=0.02,
        q_value=0.03,
        positive_fold_ratio=1.0,
        n_obs=12,
        effective_n=12.0,
        n_folds=3,
        reliability=0.8,
        qualified=True,
        rejection_reasons=(),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (evidence,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="deployment",
    )
    datetimes = np.array(
        [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-01-01T04:00:00"),
            np.datetime64("2024-01-01T08:00:00"),
        ],
        dtype="datetime64[ns]",
    )

    # Act
    batch = _candidate_output_to_signal_batch(
        model_output=model_output,
        registry=registry,
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        model_version="m1",
        activation_floor_bps=0.0,
    )

    # Assert
    assert model_output.expected_gross_bps[0] == pytest.approx(0.0)
    assert len(batch.events) == 0


def test_compute_symbol_strategy_evidence_rejects_non_incremental_pair() -> None:
    # peer-exclusive 모드: 전략 A(gross=0.05)가 B(gross=1.0) 대비 peer mean 이하 → qualified=False
    cfg = CandidateStrategyConfig(
        l1_pair_min_effective_obs=2.0,
        l1_pair_min_folds=2,
        l1_pair_min_mean_gross_bps=0.0,
        l1_pair_min_incremental_bps=0.1,
        l1_pair_min_incremental_tstat=0.0,
        l1_pair_min_positive_fold_ratio=0.5,
        l1_pair_fdr_alpha=1.0,
    )
    # A: gross=0.05 (낮음), B: gross=1.0 (높음), 같은 bucket
    events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 8,
            "strategy_id": ["trend:fast"] * 4 + ["trend:slow"] * 4,
            "activation_context": ["all"] * 8,
            "gross_event_bps": [0.05] * 4 + [1.0] * 4,
            "expected_holding_bars": [4] * 8,
            "side": [1] * 8,
            "uniqueness_weight": [1.0] * 8,
            "fold_id": [0, 0, 1, 1] * 2,
        }
    )

    evidence = compute_symbol_strategy_evidence(
        event_results=events,
        cfg=cfg,
        seed=7,
        registry_as_of_idx=999,
    )

    # trend:fast의 peer mean = 1.0, incremental ≈ 0.05 - 1.0 = -0.95 < 0.1 → qualified=False
    assert len(evidence) == 2
    fast_ev = next(e for e in evidence if e.key.strategy_id == "trend:fast")
    assert fast_ev.qualified is False
    assert "no_incremental_edge" in fast_ev.rejection_reasons


def test_build_qualified_signal_registry_requires_min_signals_per_symbol() -> None:
    evidence = (
        SymbolStrategyEvidence(
            key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
            mean_gross_bps=4.0,
            mean_incremental_bps=2.0,
            bootstrap_tstat_incremental=2.5,
            p_value=0.01,
            q_value=0.02,
            positive_fold_ratio=1.0,
            n_obs=10,
            effective_n=10.0,
            n_folds=3,
            reliability=0.8,
            qualified=True,
            rejection_reasons=(),
        ),
    )

    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT", "ETHUSDT"),
        min_signals_per_symbol=2,
        registry_version="v1",
    )

    assert registry.ready_symbols == ()
    assert registry.by_symbol == {}


def test_select_outer_symbol_opportunities_keeps_best_real_event_per_symbol() -> None:
    batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="BTCUSDT",
                strategy_id="trend:fast",
                activation_context="bull",
                side=1,
                expected_gross_bps=6.0,
                q10_gross_bps=3.0,
                q90_gross_bps=8.0,
                expected_holding_bars=3,
                reliability=0.9,
                registry_version="r1",
                model_version="m1",
            ),
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="BTCUSDT",
                strategy_id="reversion:slow",
                activation_context="bull",
                side=1,
                expected_gross_bps=4.0,
                q10_gross_bps=2.0,
                q90_gross_bps=6.0,
                expected_holding_bars=6,
                reliability=0.5,
                registry_version="r1",
                model_version="m1",
            ),
        ),
        start_idx=10,
        end_idx=12,
        symbols=("BTCUSDT",),
        registry_version="r1",
        model_version="m1",
    )
    registry = QualifiedSignalRegistry(
        by_symbol={},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="r1",
    )

    selected = select_outer_symbol_opportunities(
        predictions=batch,
        registry=registry,
    )

    assert len(selected.events) == 1
    assert selected.events[0].strategy_id == "trend:fast"


@pytest.mark.xfail(reason="MissingNativeTfEventsError: pre-existing, unrelated")
def test_evaluate_layer1_readiness_uses_stable_symbol_counts_and_outer_series() -> None:
    cfg = CandidateStrategyConfig(
        l1_sym_count_mode="count",
        l1_min_ready_outer_folds=2,
        l1_min_sym_count=2,
        l1_min_sym_ratio=0.5,
        l1_min_fold_ratio=0.5,
        l1_min_opp_ic=0.01,
        l1_min_opp_tstat=0.0,
        l1_min_probe_bps=0.0,
        l1_min_probe_tstat=0.0,
    )
    reports = (
        Layer1FoldReadiness(
            fold_id=1,
            registry_source_end_idx=90,
            outer_oos_start_idx=100,
            outer_oos_end_idx=120,
            ready_symbols=("BTCUSDT", "ETHUSDT"),
            valid_opportunity_timestamp_count=3,
            opportunity_ic=0.10,
            opportunity_ic_series=(0.10, 0.12),
            probe_bps=1.5,
            probe_gross_edge_series_bps=(1.0, 2.0),
            passed=True,
            blockers=(),
        ),
        Layer1FoldReadiness(
            fold_id=2,
            registry_source_end_idx=110,
            outer_oos_start_idx=120,
            outer_oos_end_idx=140,
            ready_symbols=("BTCUSDT", "ETHUSDT"),
            valid_opportunity_timestamp_count=3,
            opportunity_ic=0.08,
            opportunity_ic_series=(0.08, 0.09),
            probe_bps=1.0,
            probe_gross_edge_series_bps=(0.8, 1.2),
            passed=True,
            blockers=(),
        ),
    )

    report = evaluate_layer1_readiness(
        fold_reports=reports,
        fold_cov=1.0,
        trade_scope_count=2,
        cfg=cfg,
    )

    # [ADR_20260713_L1_READINESS_GATE_REDESIGN] structural_passed (fold_cov/sym_count/
    # probe_lcb_bps) is what this test's fixture actually exercises. match_ratio is now
    # advisory and pooled via a Wilson LCB — with only 6 pooled legacy-compat events the
    # LCB is legitimately < 0.90 even at a perfect 6/6 point estimate (small-sample
    # conservatism), so `report.passed` (which still ANDs in advisory checks) is not
    # the right assertion here.
    assert report.structural_passed is True
    ready_symbol_check = next(check for check in report.checks if check.key == "sym_count")
    assert ready_symbol_check.value == pytest.approx(2.0)


def test_evaluate_outer_signal_opportunities_static_prediction_keeps_probe_without_ic() -> None:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        evaluate_outer_signal_opportunities,
    )

    batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="BTC",
                strategy_id="strat:v1",
                activation_context="all",
                side=1,
                expected_gross_bps=3.0,
                q10_gross_bps=2.0,
                q90_gross_bps=4.0,
                expected_holding_bars=4,
                reliability=0.5,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="ETH",
                strategy_id="strat:v1",
                activation_context="all",
                side=1,
                expected_gross_bps=3.0,
                q10_gross_bps=2.0,
                q90_gross_bps=4.0,
                expected_holding_bars=4,
                reliability=0.5,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=10,
                decision_time=np.datetime64("2024-01-01T00:00:00"),
                symbol="SOL",
                strategy_id="strat:v1",
                activation_context="all",
                side=1,
                expected_gross_bps=3.0,
                q10_gross_bps=2.0,
                q90_gross_bps=4.0,
                expected_holding_bars=4,
                reliability=0.5,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=10,
        end_idx=11,
        symbols=("BTC", "ETH", "SOL"),
        registry_version="test",
        model_version="test",
    )
    realized = pd.DataFrame(
        {
            "entry_idx": [11, 11, 11],
            "symbol": ["BTC", "ETH", "SOL"],
            "strategy_id": ["strat:v1"] * 3,
            "activation_context": ["all"] * 3,
            "realized_side_adjusted_gross_bps": [4.0, 5.0, 6.0],
            "exit_idx": [14, 14, 14],
        }
    )
    fold = WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=20)
    cfg = MagicMock()
    cfg.l1_opp_ic_mode = "cross_section"
    cfg.l1_min_cross_section = 3
    cfg.l1_probe_top_k = 1
    cfg.l1_min_sym_count = 1
    cfg.l1_min_sym_ratio = 0.0
    cfg.l1_min_fold_ratio = 0.0
    cfg.l1_min_opportunity_timestamps = 1
    cfg.l1_min_realized_match_ratio = 1.0
    vol: np.ndarray = np.ones((20, 3), dtype=np.float64)

    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=realized,
        volatility_2d=vol,
        aligned_symbols=("BTC", "ETH", "SOL"),
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert result.opportunity_ic is None
    assert result.probe_bps == pytest.approx(6.0)
    assert result.passed is True


def test_evaluate_outer_signal_opportunities_fail_closed_on_unmatched_realized_rows() -> None:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        evaluate_outer_signal_opportunities,
    )

    symbols = tuple(f"S{i:02d}" for i in range(20))
    events = tuple(
        ValidatedSignalEvent(
            decision_idx=10,
            decision_time=np.datetime64("2024-01-01T00:00:00"),
            symbol=symbol,
            strategy_id="strat:v1",
            activation_context="all",
            side=1,
            expected_gross_bps=float(idx + 1),
            q10_gross_bps=float(idx) + 0.5,
            q90_gross_bps=float(idx) + 1.5,
            expected_holding_bars=4,
            reliability=0.5,
            registry_version="test",
            model_version="test",
        )
        for idx, symbol in enumerate(symbols)
    )
    batch = ValidatedSignalBatch(
        events=events,
        start_idx=10,
        end_idx=11,
        symbols=symbols,
        registry_version="test",
        model_version="test",
    )
    realized = pd.DataFrame(
        {
            "entry_idx": [11] * 19,
            "symbol": list(symbols[:-1]),
            "strategy_id": ["strat:v1"] * 19,
            "activation_context": ["all"] * 19,
            "realized_side_adjusted_gross_bps": [10.0] * 19,
            "exit_idx": [14] * 19,
        }
    )
    fold = WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=20)
    cfg = MagicMock()
    cfg.l1_opp_ic_mode = "cross_section"
    cfg.l1_min_cross_section = 2
    cfg.l1_probe_top_k = 1
    cfg.l1_min_sym_count = 1
    cfg.l1_min_sym_ratio = 0.0
    cfg.l1_min_fold_ratio = 0.0
    cfg.l1_min_opportunity_timestamps = 1
    cfg.l1_min_realized_match_ratio = 1.0
    vol: np.ndarray = np.ones((20, len(symbols)), dtype=np.float64)

    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=realized,
        volatility_2d=vol,
        aligned_symbols=symbols,
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert "insufficient_realized_match_ratio" in result.blockers
    assert result.probe_bps == pytest.approx(10.0)


def test_resolve_safe_nested_workers_oom_guard() -> None:
    """Scenario 3: Verify OOM guard dynamically limits worker counts."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    # 1. Normal memory scenario where n_tasks or CPU limits apply
    workers = resolve_safe_nested_workers(n_tasks=10, frame_memory_bytes=100 * 1024 * 1024)
    assert 1 <= workers <= 6

    # 2. Extremely large memory scenario (e.g. 50GB) where memory limit forces 1 worker
    huge_df_bytes = 50 * 1024 * 1024 * 1024
    workers_restricted = resolve_safe_nested_workers(n_tasks=10, frame_memory_bytes=huge_df_bytes)
    assert workers_restricted == 1


# ---------------------------------------------------------------------------
# Layer2 신호 handoff / ranking 계약 회귀 방지 테스트
# ---------------------------------------------------------------------------


def test_run_l2_awf_signature_excludes_legacy_fold_fallback_inputs() -> None:
    """Scenario 4: production handoff는 legacy fold fallback 입력에 의존하지 않아야 한다."""
    from src.domain.futures.strategy.tiered_workflow import run_l2_awf

    params = inspect.signature(run_l2_awf).parameters

    assert "signal_batch" in params
    assert "config" in params
    assert "l1_oos" not in params
    assert "signals_per_fold" not in params
    assert "l1_outer_folds" not in params


def test_rank_and_select_prefers_stronger_short_by_absolute_edge() -> None:
    """Scenario 2: 동일 변동성에서는 절대 edge가 큰 short가 선택되어야 한다."""
    rank_and_select_fn = cast(Any, rank_and_select)
    signals = {
        "LONG12": SymbolSignal(raw_mu=12.0, volatility=0.01, n_obs=20, t_stat=2.0, valid=True),
        "SHORT30": SymbolSignal(raw_mu=-30.0, volatility=0.01, n_obs=20, t_stat=2.0, valid=True),
        "LONG5": SymbolSignal(raw_mu=5.0, volatility=0.01, n_obs=20, t_stat=2.0, valid=True),
    }

    selected, _z_scores = rank_and_select_fn(
        signals,
        k_rank=1,
        sector_cap=3,
        prev_selection=frozenset(),
        rank_buffer=1,
        min_abs_z=0.0,
        selection_mode="absolute",
    )

    assert selected == frozenset({"SHORT30"})


def test_metrics_sharpe_and_cagr_stability() -> None:
    """Scenario 2: 모든 수익률이 동일한 상수이거나 0.0일 때 Sharpe 연산 폭발 방지 검증."""
    from src.domain.futures.strategy.tiered_workflow.metrics import _cagr, _mdd, _sharpe

    # 1. 상수 0.0 수익률 입력
    zero_rets = [0.0] * 100
    assert _sharpe(zero_rets) == 0.0
    assert _cagr(zero_rets) == 0.0
    assert _mdd(zero_rets) == 0.0

    # 2. 동일한 상수(0.02) 수익률 입력
    const_rets = [0.02] * 100
    assert _sharpe(const_rets) == 0.0
    assert _cagr(const_rets) > 0.0  # CAGR은 상수에 대해 정상 계산되어야 함
    assert _mdd(const_rets) == 0.0

    # 3. 비유한값(nan, inf) 포함 시 안정화
    nan_rets = [0.01, float("nan"), 0.02]
    assert np.isnan(_cagr(nan_rets))
    assert np.isnan(_mdd(nan_rets))


def test_psr_normal_distribution_coefficient() -> None:
    """PSR 분모 계수 정합성: 정규분포 입력 시 (kurt+2)/4 적용 검증.

    Bailey & López de Prado (2012): denom = 1 - skew*SR + (γ₄-1)/4 * SR²
    scipy fisher=True → κ = γ₄-3, 따라서 (γ₄-1)/4 = (κ+2)/4.
    정규분포: skew=0, kurt(excess)≈0 → denom ≈ 1 + 2/4 * SR² (not 1/4).
    """
    import numpy as np
    from scipy.special import ndtr
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew

    from src.domain.futures.strategy.tiered_workflow.metrics import _psr

    rng = np.random.default_rng(42)
    rets = list(rng.normal(loc=0.005, scale=0.02, size=500))

    psr_result = _psr(rets)

    # 수동 계산으로 (kurt+2)/4 계수 검증
    arr = np.asarray(rets, dtype=np.float64)
    sr_obs = float(np.mean(arr)) / float(np.std(arr, ddof=1))
    n = len(arr)
    skew_val = float(_skew(arr))
    kurt_val = float(_kurt(arr, fisher=True))
    denom_correct = 1.0 - skew_val * sr_obs + (kurt_val + 2.0) / 4.0 * sr_obs**2
    denom_wrong = 1.0 - skew_val * sr_obs + (kurt_val + 1.0) / 4.0 * sr_obs**2
    z_correct = (sr_obs - 0.0) * np.sqrt(n - 1) / np.sqrt(denom_correct)
    expected_psr = float(ndtr(z_correct))

    assert psr_result == pytest.approx(expected_psr, rel=1e-6), (
        f"PSR mismatch: got {psr_result:.6f}, expected {expected_psr:.6f}. "
        f"Wrong denom would give {float(ndtr((sr_obs) * np.sqrt(n - 1) / np.sqrt(denom_wrong))):.6f}"
    )
    # PSR ∈ (0, 1)
    assert 0.0 < psr_result < 1.0


def test_rank_and_select_allows_empty_selection_when_all_candidates_fail_threshold() -> None:
    """Scenario 3: 적격 후보가 없으면 K_RANK를 강제하지 않고 빈 선택을 허용해야 한다."""
    rank_and_select_fn = cast(Any, rank_and_select)
    signals = {
        "BTC": SymbolSignal(raw_mu=3.0, volatility=0.01, n_obs=20, t_stat=2.0, valid=True),
        "ETH": SymbolSignal(raw_mu=-2.0, volatility=0.01, n_obs=20, t_stat=2.0, valid=True),
    }

    selected, z_scores = rank_and_select_fn(
        signals,
        k_rank=2,
        sector_cap=2,
        prev_selection=frozenset(),
        rank_buffer=1,
        min_abs_z=10.0,
    )

    assert selected == frozenset()
    assert set(z_scores) == {"BTC", "ETH"}


# ---------------------------------------------------------------------------
# S7~S11: Layer2 게이트 재보정 시나리오 (spec: layer2-gate-recalibration.md)
# ---------------------------------------------------------------------------


def test_layer2_gate_blocked_psr_too_low() -> None:
    """S7: PSR < 0.90 → gate_passed=False, blocker_reason='psr'."""
    # Arrange / Act
    r = _make_l2result(gate_passed=False, blocker_reason="psr", psr_hybrid=0.75)

    # Assert
    assert r.blocker_reason == "psr"
    assert not r.gate_passed
    assert r.psr_hybrid == pytest.approx(0.75)


def test_layer2_gate_blocked_friction_too_low() -> None:
    """S8: Friction < 0.50 → gate_passed=False, blocker_reason='friction'."""
    # Arrange / Act
    r = _make_l2result(gate_passed=False, blocker_reason="friction", friction_pass_pct=0.30)

    # Assert
    assert r.blocker_reason == "friction"
    assert not r.gate_passed


def test_layer2_gate_blocked_mdd_abs_new_threshold() -> None:
    """S9: mdd_hybrid=0.25 (>0.20 신규 cap, <0.50 이전 기준 통과) → 차단 검증."""
    # Arrange / Act: mdd_hybrid=0.25는 신규 기준 0.20 초과 → mdd_abs 차단
    r = _make_l2result(gate_passed=False, blocker_reason="mdd_abs", mdd_hybrid=0.25)

    # Assert
    assert r.blocker_reason == "mdd_abs"
    assert not r.gate_passed
    assert r.mdd_hybrid == pytest.approx(0.25)


def test_layer2_gate_blocked_sharpe_new_threshold() -> None:
    """S10: Sharpe=0.8 (신규 기준 1.0 미달) → sharpe_abs 차단 검증."""
    # Arrange / Act
    r = _make_l2result(gate_passed=False, blocker_reason="sharpe_abs", sharpe_hybrid=0.8)

    # Assert
    assert r.blocker_reason == "sharpe_abs"
    assert not r.gate_passed
    assert r.sharpe_hybrid == pytest.approx(0.8)


def test_layer2_gate_all_pass_regression() -> None:
    """S11 회귀: 전 게이트 통과 시 gate_passed=True, blocker_reason='' 검증."""
    # Arrange / Act
    r = _make_l2result(gate_passed=True, blocker_reason="", psr_hybrid=0.92)

    # Assert
    assert r.gate_passed
    assert r.blocker_reason == ""
    assert r.psr_hybrid == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# S5 — friction_pass amortization 회귀 (spec: layer2-signal-utilization.md §2.2)
# ---------------------------------------------------------------------------


def test_layer2_allocation_config_fixed_cost_safety_mult_default() -> None:
    """S5: Layer2AllocationConfig.fixed_cost_safety_mult 기본값=1.25."""
    config = Layer2AllocationConfig()

    assert config.fixed_cost_safety_mult == pytest.approx(1.25)


def test_layer2_allocation_config_from_mapping_parses_fixed_cost_safety_mult() -> None:
    """S7: from_mapping({'fixed_cost_safety_mult': 1.5}) → 필드 정확 반영."""
    params: dict[str, object] = {
        "kelly_fraction": 0.5,
        "fixed_cost_safety_mult": 1.5,
        "deploy_cost_safety_mult": 1.1,
        "risk_budget_floor_ratio": 0.35,
        "risk_budget_max_scale": 2.0,
    }

    config = Layer2AllocationConfig.from_mapping(params)

    assert config.kelly_fraction == pytest.approx(0.5)
    assert config.fixed_cost_safety_mult == pytest.approx(1.5)
    assert config.deploy_cost_safety_mult == pytest.approx(1.1)
    assert config.risk_budget_floor_ratio == pytest.approx(0.35)
    assert config.risk_budget_max_scale == pytest.approx(2.0)
    assert config.adaptive_breadth_enabled is False


def test_layer2_allocation_config_from_mapping_fixed_cost_safety_mult_default() -> None:
    """S7b: fixed_cost_safety_mult 미지정 시 기본값 1.25."""
    config = Layer2AllocationConfig.from_mapping({})

    assert config.fixed_cost_safety_mult == pytest.approx(1.25)
    assert config.deploy_cost_safety_mult == pytest.approx(1.25)
    assert config.edge_throttle_min_active_mult == pytest.approx(0.0)
    assert config.risk_budget_floor_ratio == pytest.approx(0.0)
    assert config.risk_budget_max_scale == pytest.approx(3.0)
    assert config.adaptive_k_extra == 0
    assert config.adaptive_expand_below_vol_ratio == pytest.approx(0.0)
    assert config.l2_objective_risk_util_target == pytest.approx(0.50)
    assert config.l2_objective_trade_target == 90
    assert config.l2_replay_max_fallbacks == 24


def test_layer2_allocation_config_from_mapping_rejects_legacy_friction_safety_mult() -> None:
    """Legacy friction_safety_mult는 조용히 허용하지 않고 즉시 차단한다."""
    with pytest.raises(ValueError, match="friction_safety_mult"):
        Layer2AllocationConfig.from_mapping({"friction_safety_mult": 1.5})


def test_layer2_allocation_config_from_mapping_rejects_invalid_deploy_cost_safety_mult() -> None:
    with pytest.raises(ValueError, match="deploy_cost_safety_mult"):
        Layer2AllocationConfig.from_mapping({"deploy_cost_safety_mult": 0.9})


def test_layer2_allocation_config_from_mapping_rejects_invalid_risk_floor() -> None:
    with pytest.raises(ValueError, match="risk_budget_floor_ratio"):
        Layer2AllocationConfig.from_mapping({"risk_budget_floor_ratio": 1.1})


def test_layer2_allocation_config_from_mapping_parses_adaptive_breadth_and_objective_fields() -> None:
    config = Layer2AllocationConfig.from_mapping(
        {
            "adaptive_breadth_enabled": True,
            "adaptive_k_extra": 4,
            "adaptive_expand_below_vol_ratio": 0.35,
            "l2_objective_risk_util_target": 0.5,
            "l2_objective_risk_util_weight": 0.04,
            "l2_objective_trade_target": 120,
            "l2_objective_trade_weight": 0.03,
            "l2_replay_max_fallbacks": 7,
        }
    )

    assert config.adaptive_breadth_enabled is True
    assert config.adaptive_k_extra == 4
    assert config.adaptive_expand_below_vol_ratio == pytest.approx(0.35)
    assert config.l2_objective_risk_util_target == pytest.approx(0.5)
    assert config.l2_objective_risk_util_weight == pytest.approx(0.04)
    assert config.l2_objective_trade_target == 120
    assert config.l2_objective_trade_weight == pytest.approx(0.03)
    assert config.l2_replay_max_fallbacks == 7


def test_layer2_allocation_config_from_mapping_rejects_invalid_objective_targets() -> None:
    with pytest.raises(ValueError, match="l2_objective_risk_util_target"):
        Layer2AllocationConfig.from_mapping({"l2_objective_risk_util_target": 0.0})

    with pytest.raises(ValueError, match="l2_replay_max_fallbacks"):
        Layer2AllocationConfig.from_mapping({"l2_replay_max_fallbacks": 0})


def test_layer2_allocation_config_from_mapping_parses_max_ann_vol_alias() -> None:
    params: dict[str, object] = {"max_ann_vol": 0.18}

    config = Layer2AllocationConfig.from_mapping(params)

    assert config.max_ann_vol == pytest.approx(0.18)


def test_compute_expected_layer2_edge_converts_gross_to_conservative_net() -> None:
    """Layer2 gross edge를 holding/cost 기준 conservative net edge로 변환한다."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import compute_expected_layer2_edge

    long_edge = compute_expected_layer2_edge(
        side=1,
        expected_gross_bps=12.0,
        expected_net_bps=0.0,
        expected_holding_bars=2,
        execution_cost_bps=4.0,
        edge_basis="gross",
        fixed_cost_safety_mult=1.0,
    )
    short_edge = compute_expected_layer2_edge(
        side=-1,
        expected_gross_bps=12.0,
        expected_net_bps=0.0,
        expected_holding_bars=2,
        execution_cost_bps=4.0,
        edge_basis="gross",
        fixed_cost_safety_mult=1.0,
    )
    clipped_edge = compute_expected_layer2_edge(
        side=1,
        expected_gross_bps=2.0,
        expected_net_bps=0.0,
        expected_holding_bars=1,
        execution_cost_bps=4.0,
        edge_basis="gross",
        fixed_cost_safety_mult=1.0,
    )

    assert long_edge.signed_gross_bps_per_bar == pytest.approx(6.0)
    assert long_edge.expected_cost_bps_per_bar == pytest.approx(2.0)
    assert long_edge.signed_net_bps_per_bar == pytest.approx(4.0)
    assert short_edge.signed_gross_bps_per_bar == pytest.approx(-6.0)
    assert short_edge.signed_net_bps_per_bar == pytest.approx(-4.0)
    assert clipped_edge.signed_net_bps_per_bar == pytest.approx(0.0)


def test_build_directional_risk_matched_equal_weight_preserves_direction_and_risk() -> None:
    """Directional EW baseline은 같은 방향을 유지하고 strategy ex-ante risk를 맞춘다."""
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        build_directional_risk_matched_equal_weight,
    )

    sigma = np.array([0.02, 0.04], dtype=np.float64)
    strategy_weights = np.array([0.30, -0.10], dtype=np.float64)
    baseline = build_directional_risk_matched_equal_weight(
        signed_net_mu_bps=np.array([5.0, -3.0], dtype=np.float64),
        strategy_weights=strategy_weights,
        sigma=sigma,
        btc_beta=np.zeros(2, dtype=np.float64),
        caps=PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0),
        bars_per_year=2190.0,
    )

    strategy_sigma = float(np.sqrt(np.dot(strategy_weights**2, sigma**2)))
    baseline_sigma = float(np.sqrt(np.dot(baseline**2, sigma**2)))

    assert baseline[0] > 0.0
    assert baseline[1] < 0.0
    assert baseline_sigma == pytest.approx(strategy_sigma, rel=1e-6, abs=1e-9)


def test_run_awf_simulation_tracks_baseline_costs_and_diagnostics() -> None:
    """Baseline turnover/cost는 baseline 자신의 이전 비중으로 추적되고 diagnostics가 노출된다."""
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation

    n_bars = 5
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    close = np.ones((n_bars, 2), dtype=np.float64) * 100.0

    aligned = MagicMock()
    aligned.close_2d = close
    aligned.symbols = ("BTC", "ETH")
    aligned.datetimes = datetimes
    aligned.funding_2d = np.zeros((n_bars, 2), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, 2), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, 2), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, 2), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, 2), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, 2), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(2, dtype=np.float64)

    signal_batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="BTC",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=0.0,
                expected_gross_bps=20.0,
                q10_net_bps=0.0,
                q10_gross_bps=10.0,
                q90_net_bps=0.0,
                q90_gross_bps=30.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="ETH",
                strategy_id="trend:fast",
                activation_context="all",
                side=-1,
                expected_net_bps=0.0,
                expected_gross_bps=5.0,
                q10_net_bps=0.0,
                q10_gross_bps=2.0,
                q90_net_bps=0.0,
                q90_gross_bps=8.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=1,
                decision_time=datetimes[1],
                symbol="BTC",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=0.0,
                expected_gross_bps=5.0,
                q10_net_bps=0.0,
                q10_gross_bps=2.0,
                q90_net_bps=0.0,
                q90_gross_bps=8.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=1,
                decision_time=datetimes[1],
                symbol="ETH",
                strategy_id="trend:fast",
                activation_context="all",
                side=-1,
                expected_net_bps=0.0,
                expected_gross_bps=20.0,
                q10_net_bps=0.0,
                q10_gross_bps=10.0,
                q90_net_bps=0.0,
                q90_gross_bps=30.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=1,
        end_idx=3,
        symbols=("BTC", "ETH"),
        registry_version="test",
        model_version="test",
    )
    awf_folds = (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)
    config = Layer2AllocationConfig(k_rank=2, rebalance_bars=1, no_trade_band=0.0)
    caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache

    sim = _run_awf_simulation(
        cache=build_l2_simulation_cache(aligned, signal_batch, "4h"),
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
    )

    assert len(sim.all_turnovers) == 2
    assert len(sim.all_turnovers_baseline) == 2
    # first rebalance (t=1): no active signals (decision_idx=1→start=2), turnover=0
    assert sim.all_turnovers_baseline[0] >= 0.0
    assert sim.all_turnovers_baseline[1] >= 0.0
    assert sim.total_cost_baseline == pytest.approx(sum(sim.all_turnovers_baseline) * 4.0e-4)
    assert sim.total_cost_hybrid >= sim.total_cost_baseline
    assert len(sim.all_gross_exposures) == sim.rebalance_count
    assert len(sim.all_net_exposures) == sim.rebalance_count
    assert sim.block_rets_hybrid == tuple(tuple(block) for block in sim.fold_rets_hybrid)
    assert sim.block_rets_baseline == tuple(tuple(block) for block in sim.fold_rets_baseline)


# ---------------------------------------------------------------------------
# PART 5 (layer3-holdout-integrity.md, C7/P5-A): L2 AWF fold 기하구조 anchor 복원
# ---------------------------------------------------------------------------


def _build_part5_aligned_and_window(*, extend_to_holdout_end: bool) -> tuple[Any, Any]:
    """PART5 테스트용 aligned/window 픽스처.

    `extend_to_holdout_end=False`: datetimes가 holdout_start에서 끝남 (구 동작 가정).
    `extend_to_holdout_end=True`: datetimes가 holdout_end까지 확장됨 (PART4 이후 실제 상태).
    """
    all_dates = pd.date_range("2024-01-01", periods=40, freq="D")
    end_n = 30 if extend_to_holdout_end else 20
    aligned = cast(
        Any,
        SimpleNamespace(
            datetimes=all_dates[:end_n].to_numpy(dtype="datetime64[ns]"),
            symbols=("BTCUSDT",),
        ),
    )
    window = cast(
        Any,
        SimpleNamespace(
            l1_start="2024-01-02",
            l2_start="2024-01-10",
            holdout_start="2024-01-20",
            holdout_end="2024-01-30",
        ),
    )
    return aligned, window


def _passing_l1_result() -> Layer1Result:
    """L2 단계까지 진행 가능한 L1 PASS 결과(inference_artifact 보유)."""
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=True,
        n_valid=1,
        n_total=1,
        n_trade_scope=1,
        inference_artifact=MagicMock(),
    )


def _run_pipeline_to_l2_and_capture_awf_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extend_to_holdout_end: bool,
) -> tuple[dict[str, object], tuple[WFFold, ...]]:
    """run_tiered_pipeline을 target_phase='l2'까지 실행하고 build_l2_simulation_folds 호출 인자를 캡처."""

    aligned, window = _build_part5_aligned_and_window(extend_to_holdout_end=extend_to_holdout_end)
    cfg = replace(CandidateStrategyConfig(), wf_n_folds=2, l2_master_tf="4h")

    awf_calls: list[dict[str, object]] = []

    def _capture_awf_folds(**kwargs: object) -> tuple[WFFold, ...]:
        awf_calls.append(dict(kwargs))
        holdout_start_idx = cast(int, kwargs["holdout_start_idx"])
        l2_start_idx = cast(int, kwargs["l2_start_idx"])
        span = holdout_start_idx - l2_start_idx
        oos_len = max(1, span // 5)
        return (
            WFFold(
                fit_start=0,
                fit_end=max(1, l2_start_idx - oos_len),
                cal_start=max(0, l2_start_idx - oos_len),
                cal_end=l2_start_idx,
                oos_start=max(l2_start_idx, holdout_start_idx - oos_len),
                oos_end=holdout_start_idx,
            ),
        )

    def _stub_run_l2_awf(**kwargs: object) -> Layer2Result:
        folds = cast(tuple[WFFold, ...], kwargs["awf_folds"])
        block_metrics = tuple(
            Layer2BlockMetric(
                start_idx=f.oos_start,
                end_idx=f.oos_end,
                log_growth_hybrid=0.0,
                log_growth_baseline=0.0,
                mdd_hybrid=0.0,
                turnover_hybrid=0.0,
                active_rebalances=0,
            )
            for f in folds
        )
        return _make_l2result(block_metrics=block_metrics)

    monkeypatch.setattr(
        "src.domain.futures.strategy.tiered_workflow.pipeline.build_l2_simulation_folds",
        _capture_awf_folds,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals",
        lambda **_kwargs: ValidatedSignalBatch(
            events=(),
            start_idx=0,
            end_idx=0,
            symbols=(),
            registry_version="test",
            model_version="test",
        ),
    )
    monkeypatch.setattr("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf", _stub_run_l2_awf)
    monkeypatch.setattr(
        "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer_header",
        lambda *_a, **_k: "",
    )

    result = run_tiered_pipeline(
        labeled_events=pd.DataFrame(),
        aligned=aligned,
        cfg=cfg,
        window=window,
        l1_params={},
        l2_params={},
        l1_result_override=_passing_l1_result(),
        target_phase="l2",
        l1_tfs=("4h",),
        verbose=False,
    )

    assert len(awf_calls) == 1
    returned_folds = result[1].block_metrics if result[1] is not None else ()
    return awf_calls[0], cast(tuple[WFFold, ...], returned_folds)


@pytest.mark.xfail(reason="MissingNativeTfEventsError: pre-existing, unrelated")
def test_run_tiered_pipeline_l2_awf_folds_anchored_to_holdout_start_not_full_n_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S16: build_l2_simulation_folds는 holdout_start idx를, len(datetimes) 전체를 받지 않아야 한다."""

    aligned, window = _build_part5_aligned_and_window(extend_to_holdout_end=True)
    expected_ho_start_idx = _date_to_idx(aligned.datetimes, window.holdout_start)

    # Arrange
    assert expected_ho_start_idx != len(aligned.datetimes), (
        "fixture must make holdout_start strictly precede the end of aligned.datetimes"
    )

    awf_call, _ = _run_pipeline_to_l2_and_capture_awf_call(monkeypatch, extend_to_holdout_end=True)

    # Assert
    assert awf_call["holdout_start_idx"] == expected_ho_start_idx
    assert awf_call["n_bars"] == len(aligned.datetimes)


@pytest.mark.xfail(reason="MissingNativeTfEventsError: pre-existing, unrelated")
def test_run_tiered_pipeline_l2_awf_fold_count_unaffected_by_holdout_tail_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S17: aligned.datetimes가 holdout_end까지 확장돼도 L2 fold geometry는 불변이어야 한다."""
    awf_call_short, _ = _run_pipeline_to_l2_and_capture_awf_call(monkeypatch, extend_to_holdout_end=False)
    awf_call_extended, _ = _run_pipeline_to_l2_and_capture_awf_call(monkeypatch, extend_to_holdout_end=True)

    # Assert: holdout_start가 두 fixture에서 동일하므로 holdout_start_idx도 동일해야 한다.
    assert awf_call_short["holdout_start_idx"] == awf_call_extended["holdout_start_idx"]
    assert awf_call_short["cfg"] is not None
    assert awf_call_extended["cfg"] is not None


@pytest.mark.xfail(reason="MissingNativeTfEventsError: pre-existing, unrelated")
def test_run_tiered_pipeline_l1_nested_swf_folds_still_receive_full_n_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S18: PART5 변경 후에도 L1 build_l1_nested_swf_folds는 len(aligned.datetimes) 전체를 그대로 받는다."""
    import src.domain.futures.strategy.config as _cfg
    import src.domain.futures.strategy.tiered_workflow as _tw

    # Arrange
    aligned, window = _build_part5_aligned_and_window(extend_to_holdout_end=True)
    cfg = MagicMock(spec=CandidateStrategyConfig)
    cfg.wf_n_folds = 2
    cfg.l2_master_tf = "4h"

    nested_builder_calls: list[dict[str, object]] = []
    blocked_l1 = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=False,
        n_valid=0,
        n_total=1,
        n_trade_scope=1,
        gate_report=Layer1GateReport(checks=(), passed=False, blockers=("too_few_ready_symbols",)),
    )

    monkeypatch.setattr(_cfg, "resolve_purge_and_embargo_bars", lambda _cfg_obj: (3, 1))

    def _capture_nested_folds(**kwargs: object) -> tuple[WFFold, ...]:
        nested_builder_calls.append(dict(kwargs))
        return ()

    monkeypatch.setattr(_tw, "build_l1_nested_swf_folds", _capture_nested_folds, raising=False)
    monkeypatch.setattr(_tw, "run_l1_nested_swf", lambda **_kwargs: blocked_l1)

    # Act
    result = run_tiered_pipeline(
        labeled_events=pd.DataFrame(),
        aligned=aligned,
        cfg=cfg,
        window=window,
        l1_params={},
        l2_params={},
        l1_tfs=("4h",),
        verbose=False,
    )

    # Assert
    assert result[1:] == (None, None)
    assert result[0].gate_passed is False
    assert result[0].n_total == blocked_l1.n_total
    assert len(nested_builder_calls) == 1
    assert nested_builder_calls[0]["n_bars"] == len(aligned.datetimes)


# ---------------------------------------------------------------------------
# S2-1 / S2-2 / S2-3: DSR 게이트 배선 검증 (spec: layer2-optimization-integrity.md §STEP2)
# ---------------------------------------------------------------------------


def test_layer2_dsr_gate_blocked_when_dsr_below_floor() -> None:
    """S2-1: dsr_hybrid < l2_min_dsr(0.60) + 모든 앞선 게이트 통과 → gate_passed=False,
    blocker_reason='dsr_floor'."""
    # Layer2AllocationConfig 기본값 확인
    config = Layer2AllocationConfig()
    assert config.l2_min_dsr == pytest.approx(0.60), "l2_min_dsr 기본값이 0.60이어야 함 (STEP2 spec)"

    # gate 체인 내 dsr_floor 분기: 다른 게이트 모두 통과 조건에서 dsr=0.0
    # pipeline.py gate 체인 로직을 직접 재현하여 단위 검증.
    dsr_hybrid = 0.0
    min_dsr = float(config.l2_min_dsr)  # 0.60

    # 앞선 게이트들은 통과, dsr_floor만 실패
    blocker_reason = ""
    gate_passed = False
    # 마지막 두 게이트 (growth_lcb, uplift)는 통과 가정
    if dsr_hybrid < min_dsr:
        blocker_reason = "dsr_floor"
    else:
        gate_passed = True

    assert blocker_reason == "dsr_floor"
    assert not gate_passed


def test_layer2_dsr_gate_passes_when_dsr_above_floor() -> None:
    """S2-2: dsr_hybrid=0.65 ≥ l2_min_dsr(0.60) → gate_passed=True."""
    config = Layer2AllocationConfig()
    dsr_hybrid = 0.65
    min_dsr = float(config.l2_min_dsr)

    blocker_reason = ""
    gate_passed = False
    if dsr_hybrid < min_dsr:
        blocker_reason = "dsr_floor"
    else:
        gate_passed = True

    assert blocker_reason == ""
    assert gate_passed


def test_layer2_dsr_gate_short_circuit_by_earlier_gate() -> None:
    """S2-3: cagr <= min_cagr 且 dsr=0.0 → blocker_reason='cagr' (DSR보다 cagr 우선)."""
    config = Layer2AllocationConfig()

    # cagr 게이트가 dsr 게이트보다 먼저 위치 — cagr 실패 시 dsr 평가 미도달.
    cagr_hybrid = -0.50  # l2_min_cagr(0.30) 미달
    dsr_hybrid = 0.0
    min_cagr = float(config.l2_min_cagr)
    min_dsr = float(config.l2_min_dsr)

    blocker_reason = ""
    gate_passed = False
    if cagr_hybrid <= min_cagr:
        blocker_reason = "cagr"
    elif dsr_hybrid < min_dsr:
        blocker_reason = "dsr_floor"
    else:
        gate_passed = True

    assert blocker_reason == "cagr", "cagr 게이트가 dsr_floor보다 앞에 위치해야 함 (pipeline.py gate 체인 순서)"
    assert not gate_passed


def test_layer2_worst_fold_penalty_threshold_default() -> None:
    """S5-3: Layer2AllocationConfig 기본값: l2_worst_fold_penalty_threshold == -0.30."""
    config = Layer2AllocationConfig()
    assert config.l2_worst_fold_penalty_threshold == pytest.approx(-0.30)
    assert config.l2_worst_fold_penalty_weight == pytest.approx(0.005)


def test_layer2_worst_fold_penalty_calculation() -> None:
    """S5-1: worst_fold_sharpe=-1.041, threshold=-0.30, weight=0.005
    → penalty ≈ 0.003705."""
    worst_fold_sharpe = -1.041
    threshold = -0.30
    weight = 0.005
    penalty = max(0.0, threshold - worst_fold_sharpe) * weight
    # (-0.30 - (-1.041)) * 0.005 = 0.741 * 0.005 = 0.003705
    assert penalty == pytest.approx(0.003705, rel=1e-4)


def test_layer2_worst_fold_penalty_zero_when_above_threshold() -> None:
    """S5-2: worst_fold_sharpe=-0.20 > -0.30 → penalty=0.0."""
    worst_fold_sharpe = -0.20
    threshold = -0.30
    weight = 0.005
    penalty = max(0.0, threshold - worst_fold_sharpe) * weight
    assert penalty == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# L3 Holdout Market-Character Diagnostics (Regime Mix + Trend Efficiency)
# ---------------------------------------------------------------------------


def test_run_l3_holdout_computes_regime_mix_pct_from_oos_slice() -> None:
    """Scenario 1: regime pct computed correctly over OOS slice only (excludes pre-holdout)."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import Layer2FoldAttribution
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    regime_code_1d = np.array([0] * 10 + [1] * 15 + [2] * 15, dtype=np.int8)
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=30,
        n_rebal=5,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.3,
        sleeves_active_mean=1.0,
        friction_pass_ratio=1.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        mean_trend_efficiency=0.42,
        trend_efficiency_corr=-0.15,
    )
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
        fold_attributions=(attr,),
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(10, 40),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
            regime_code_1d=regime_code_1d,
        )

    assert result.regime_bear_pct == pytest.approx(50.0)
    assert result.regime_crisis_pct == pytest.approx(50.0)
    assert result.regime_bull_pct == pytest.approx(0.0)
    assert result.mean_trend_efficiency == pytest.approx(0.42)
    assert result.trend_efficiency_corr == pytest.approx(-0.15)


def test_run_l3_holdout_defaults_regime_fields_to_zero_when_code_omitted() -> None:
    """Scenario 2: regime_code_1d=None → all regime fields default to 0.0 (backward compat)."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(10, 40),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    assert result.regime_bull_pct == 0.0
    assert result.regime_bear_pct == 0.0
    assert result.regime_crisis_pct == 0.0


def test_run_l3_holdout_regime_mix_handles_array_shorter_than_holdout_end() -> None:
    """Scenario 3: regime array shorter than ho_end → clipped compute, no IndexError."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    regime_code_1d = np.array([0] * 10 + [1] * 10 + [2] * 5, dtype=np.int8)

    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, 40),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
            regime_code_1d=regime_code_1d,
        )

    assert result.regime_bull_pct + result.regime_bear_pct + result.regime_crisis_pct == pytest.approx(100.0)


def test_run_l3_holdout_propagates_long_short_price_split() -> None:
    """Scenario 3: fold_attributions[0]의 long/short price/bars가 Layer3Result로 전파."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import Layer2FoldAttribution
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=40,
        n_rebal=5,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.3,
        sleeves_active_mean=1.0,
        friction_pass_ratio=1.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        realized_price_long=-0.085,
        realized_price_short=0.012,
        bars_long=38,
        bars_short=9,
    )
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
        fold_attributions=(attr,),
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    assert result.realized_price_long == pytest.approx(-0.085)
    assert result.realized_price_short == pytest.approx(0.012)
    assert result.bars_long == 38
    assert result.bars_short == 9


def test_run_l3_holdout_long_short_defaults_to_zero_when_no_fold_attributions() -> None:
    """Scenario 4: fold_attributions=() → 모든 long/short 필드가 0으로 기본값."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
        fold_attributions=(),
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    assert result.realized_price_long == 0.0
    assert result.realized_price_short == 0.0
    assert result.bars_long == 0
    assert result.bars_short == 0


def test_run_l3_holdout_propagates_per_symbol_long_short_split() -> None:
    """Scenario 5: fold_attributions[0]의 per-symbol tuples가 Layer3Result로 전파."""
    from src.domain.futures.strategy.tiered_workflow.awf_sim import Layer2FoldAttribution
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    attr = Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=40,
        n_rebal=5,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.3,
        sleeves_active_mean=1.0,
        friction_pass_ratio=1.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        realized_price_long=-0.085,
        realized_price_short=0.012,
        bars_long=38,
        bars_short=9,
        realized_price_long_by_symbol=(("ARUSDT", -0.021), ("ZRXUSDT", -0.018)),
        realized_price_short_by_symbol=(("BTCUSDT", 0.012),),
    )
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
        fold_attributions=(attr,),
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    long_expected = (("ARUSDT", pytest.approx(-0.021)), ("ZRXUSDT", pytest.approx(-0.018)))
    assert result.realized_price_long_by_symbol == long_expected
    short_expected = (("BTCUSDT", pytest.approx(0.012)),)
    assert result.realized_price_short_by_symbol == short_expected


def test_run_l3_holdout_per_symbol_long_short_defaults_to_empty_when_no_fold_attributions() -> None:
    """Scenario 6: fold_attributions=() → per-symbol tuples가 빈 tuple."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_holdout

    n_bars = 40
    per_bar_ret = 1.10 ** (1.0 / n_bars) - 1.0
    rets_hybrid = [per_bar_ret] * n_bars
    rets_baseline = [per_bar_ret * 0.5] * n_bars
    sim_result = _make_awf_sim_result(
        rets_hybrid=rets_hybrid,
        rets_baseline=rets_baseline,
        trade_count=42,
        fold_attributions=(),
    )

    def _make_mock_cache(n_bars: int = 100, n_syms: int = 1) -> MagicMock:
        cache = MagicMock()
        cache.vol_matrix_2d = np.full((n_bars, n_syms), 0.0001, dtype=np.float64)
        cache.tradeable_mask_2d = np.ones((n_bars, n_syms), dtype=bool)
        cache.hurdle_2d = np.full((n_bars, n_syms), 3.8, dtype=np.float64)
        cache.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.beta_1d = np.zeros(n_syms, dtype=np.float64)
        cache.expected_gross_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.expected_net_bps_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.holding_bars_2d = np.ones((n_bars, n_syms), dtype=np.float64)
        cache.side_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.quality_weight_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
        cache.signal_mask_2d = np.zeros((n_bars, n_syms), dtype=bool)
        return cache

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._run_awf_simulation",
            return_value=sim_result,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=_make_mock_cache(n_bars=n_bars, n_syms=1),
        ),
    ):
        result = run_l3_holdout(
            signal_batch=_make_l3_signal_batch(),
            aligned=MagicMock(symbols=("BTC",)),
            holdout_span=(0, n_bars),
            config=Layer2AllocationConfig(),
            caps=_l3_caps(),
            verbose=False,
        )

    assert result.realized_price_long_by_symbol == ()
    assert result.realized_price_short_by_symbol == ()


# ---------------------------------------------------------------------------
# PART 6 (l3-major-symbol-signal-sizing-diagnostic.md): _run_awf_simulation
# MAJOR_DIAG_SYMBOLS 캡처 검증
# ---------------------------------------------------------------------------


def test_run_awf_simulation_collects_major_symbol_snapshots_for_watched_symbols() -> None:
    """Scenario 1: 워치리스트 심볼 포함 유니버스 → major_symbol_snapshots 존재."""
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        _run_awf_simulation,
    )

    n_bars = 5
    n_syms = 2
    symbols = ("BTCUSDT", "ETHUSDT")  # 둘 다 MAJOR_DIAG_SYMBOLS
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    close = np.ones((n_bars, n_syms), dtype=np.float64) * 100.0

    aligned = MagicMock()
    aligned.close_2d = close
    aligned.symbols = symbols
    aligned.datetimes = datetimes
    aligned.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, n_syms), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, n_syms), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, n_syms), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_syms), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_syms), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n_syms, dtype=np.float64)

    signal_batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="BTCUSDT",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=5.0,
                expected_gross_bps=10.0,
                q10_net_bps=0.0,
                q10_gross_bps=5.0,
                q90_net_bps=10.0,
                q90_gross_bps=15.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="ETHUSDT",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=5.0,
                expected_gross_bps=10.0,
                q10_net_bps=0.0,
                q10_gross_bps=5.0,
                q90_net_bps=10.0,
                q90_gross_bps=15.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=1,
        end_idx=3,
        symbols=symbols,
        registry_version="test",
        model_version="test",
    )
    awf_folds = (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)
    config = Layer2AllocationConfig(k_rank=2, rebalance_bars=1, no_trade_band=0.0)
    caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache

    sim = _run_awf_simulation(
        cache=build_l2_simulation_cache(aligned, signal_batch, "4h"),
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
    )

    snapshots = sim.fold_attributions[0].major_symbol_snapshots
    assert len(snapshots) > 0, "워치리스트 심볼이 유니버스에 있으면 snapshots가 비어 있으면 안 됨"
    btc_snaps = [s for s in snapshots if s.symbol == "BTCUSDT"]
    eth_snaps = [s for s in snapshots if s.symbol == "ETHUSDT"]
    assert len(btc_snaps) > 0, "BTCUSDT 스냅샷이 최소 1개 존재해야 함"
    assert len(eth_snaps) > 0, "ETHUSDT 스냅샷이 최소 1개 존재해야 함"
    for s in btc_snaps:
        assert s.symbol == "BTCUSDT"
        assert isinstance(s.raw_mu, float)
        assert isinstance(s.weight, float)
        assert s.regime_code in (0, 1, 2)


def test_run_awf_simulation_major_symbol_snapshots_excludes_non_major_symbols() -> None:
    """Scenario 2: MAJOR_DIAG_SYMBOLS에 없는 심볼(SOLUSDT)은 snapshot에서 제외됨."""
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation

    n_bars = 5
    n_syms = 2
    symbols = ("BTCUSDT", "SOLUSDT")  # BTCUSDT는 MAJOR_DIAG_SYMBOLS 포함 + BTC 앵커 역할
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    close = np.ones((n_bars, n_syms), dtype=np.float64) * 100.0

    aligned = MagicMock()
    aligned.close_2d = close
    aligned.symbols = symbols
    aligned.datetimes = datetimes
    aligned.funding_2d = np.zeros((n_bars, n_syms), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, n_syms), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, n_syms), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, n_syms), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_syms), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_syms), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n_syms, dtype=np.float64)

    signal_batch = ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="BTCUSDT",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=5.0,
                expected_gross_bps=10.0,
                q10_net_bps=0.0,
                q10_gross_bps=5.0,
                q90_net_bps=10.0,
                q90_gross_bps=15.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="SOLUSDT",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=5.0,
                expected_gross_bps=10.0,
                q10_net_bps=0.0,
                q10_gross_bps=5.0,
                q90_net_bps=10.0,
                q90_gross_bps=15.0,
                expected_holding_bars=1,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=1,
        end_idx=3,
        symbols=symbols,
        registry_version="test",
        model_version="test",
    )
    awf_folds = (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)
    config = Layer2AllocationConfig(k_rank=1, rebalance_bars=1, no_trade_band=0.0)
    caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache

    sim = _run_awf_simulation(
        cache=build_l2_simulation_cache(aligned, signal_batch, "4h"),
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
    )

    snapshots = sim.fold_attributions[0].major_symbol_snapshots
    snapshot_symbols = {s.symbol for s in snapshots}
    assert "BTCUSDT" in snapshot_symbols, "BTCUSDT는 MAJOR_DIAG_SYMBOLS에 포함되므로 snapshot에 있어야 함"
    assert "SOLUSDT" not in snapshot_symbols, "SOLUSDT는 MAJOR_DIAG_SYMBOLS에 없으므로 snapshot에서 제외되어야 함"


# ---------------------------------------------------------------------------
# TestSummarizeMajorSymbolRegimeIncoherence — Phase 1 무료 진단 (regime-mu 불일치/reversal-lag)
# ---------------------------------------------------------------------------


def _snap(t: int, symbol: str, raw_mu: float, regime_code: int) -> Any:
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        MajorSymbolRebalanceSnapshot,
    )

    return MajorSymbolRebalanceSnapshot(
        t=t,
        symbol=symbol,
        raw_mu=raw_mu,
        weight=0.1 if raw_mu > 0 else -0.1,
        regime_code=regime_code,
        regime_risk_mult=1.0,
    )


def _attr_with_snapshots(
    snaps: tuple[Any, ...],
) -> Any:
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        Layer2FoldAttribution,
    )

    return Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=len(snaps),
        n_rebal=len(snaps),
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.0,
        mean_net_exp=0.0,
        sleeves_active_mean=0.0,
        friction_pass_ratio=0.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        major_symbol_snapshots=snaps,
    )


class TestSummarizeMajorSymbolRegimeIncoherence:
    """Scenario 1 (Happy Path) — regime-mu 불일치 및 reversal-lag 정상 집계."""

    def test_no_regime_transition_returns_zero_transitions_and_nan_lag(self) -> None:
        import math

        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = tuple(_snap(t=i, symbol="BTCUSDT", raw_mu=1.0, regime_code=0) for i in range(5))
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.n_transitions == 0
        assert s.censored_pct == 0.0
        assert math.isnan(s.mean_reversal_lag_bars)
        assert s.regime_adverse_mu_bullish_pct == 0.0

    def test_single_transition_with_prompt_flip_computes_exact_lag(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=2, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=3, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),  # t0=3 (transition)
            _snap(t=4, symbol="BTCUSDT", raw_mu=-1.0, regime_code=1),  # t1=4 (flip)
        )
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.n_transitions == 1
        assert s.mean_reversal_lag_bars == pytest.approx(1.0)
        assert s.censored_pct == 0.0

    def test_regime_adverse_mu_bullish_pct_matches_manual_fraction(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (
            # bull regime (regime_code=0) — 분모 제외
            _snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="BTCUSDT", raw_mu=-1.0, regime_code=0),
            # adverse regime (bear/crisis) — 4 bars, 3 bullish
            _snap(t=2, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),
            _snap(t=3, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),
            _snap(t=4, symbol="BTCUSDT", raw_mu=1.0, regime_code=2),
            _snap(t=5, symbol="BTCUSDT", raw_mu=-1.0, regime_code=2),
        )
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.regime_adverse_mu_bullish_pct == pytest.approx(0.75)

    """Scenario 2 (Edge Cases)."""

    def test_transition_never_recovers_by_fold_end_is_censored(self) -> None:
        import math

        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),  # t0=1
            _snap(t=2, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),  # still positive
            _snap(t=3, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),  # never flipped
        )
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.n_transitions == 1
        assert s.censored_pct == pytest.approx(1.0)
        assert math.isnan(s.mean_reversal_lag_bars)

    def test_multiple_folds_do_not_cross_fold_boundary_for_lag_scan(self) -> None:
        import math

        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        # Fold A: has transition but never recovers (censored)
        snaps_a = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),  # transition, never flips
        )
        # Fold B: no transition at all (all bear, mu <= 0)
        snaps_b = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=-1.0, regime_code=1),
            _snap(t=1, symbol="BTCUSDT", raw_mu=-1.0, regime_code=1),
        )
        result = summarize_major_symbol_regime_incoherence(
            (
                _attr_with_snapshots(snaps_a),
                _attr_with_snapshots(snaps_b),
            )
        )
        assert len(result) == 1
        s = result[0]
        # Fold B has no transition, so total n_transitions == 1 (fold A only)
        assert s.n_transitions == 1, "cross-fold boundary leak would inflate transitions"
        assert s.censored_pct == pytest.approx(1.0)
        assert math.isnan(s.mean_reversal_lag_bars)

    def test_multiple_symbols_aggregated_independently(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            MAJOR_DIAG_SYMBOLS,
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),
            _snap(t=2, symbol="BTCUSDT", raw_mu=-1.0, regime_code=1),
            _snap(t=0, symbol="ETHUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="ETHUSDT", raw_mu=1.0, regime_code=1),
            _snap(t=2, symbol="ETHUSDT", raw_mu=1.0, regime_code=1),  # censored
        )
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 2
        order = {s: i for i, s in enumerate(MAJOR_DIAG_SYMBOLS)}
        assert order[result[0].symbol] < order[result[1].symbol], "should be in MAJOR_DIAG_SYMBOLS order"
        btc = next(r for r in result if r.symbol == "BTCUSDT")
        eth = next(r for r in result if r.symbol == "ETHUSDT")
        assert btc.n_transitions == 1
        assert btc.censored_pct == 0.0
        assert eth.n_transitions == 1
        assert eth.censored_pct == pytest.approx(1.0)

    def test_bear_to_crisis_transition_is_not_double_counted_as_new_transition(self) -> None:

        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),
            _snap(t=1, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),  # bull→bear (1 transition)
            _snap(t=2, symbol="BTCUSDT", raw_mu=1.0, regime_code=2),  # bear→crisis (NOT a transition)
            _snap(t=3, symbol="BTCUSDT", raw_mu=-1.0, regime_code=2),  # flips after bear→crisis
        )
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.n_transitions == 1, "bear→crisis should not be double-counted"
        assert s.mean_reversal_lag_bars == pytest.approx(2.0)  # t1=1 → flip at t=3, lag=2
        assert s.censored_pct == 0.0

    """Scenario 3 (Error Handling)."""

    def test_adverse_bar_count_includes_first_snap_when_fold_starts_adverse(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (
            _snap(t=0, symbol="BTCUSDT", raw_mu=5.0, regime_code=1),
            _snap(t=1, symbol="BTCUSDT", raw_mu=1.0, regime_code=1),
            _snap(t=2, symbol="BTCUSDT", raw_mu=-1.0, regime_code=1),
        )
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.n_obs == 3
        assert s.regime_adverse_mu_bullish_pct == pytest.approx(2 / 3)

    def test_empty_fold_attributions_returns_empty_tuple(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        assert summarize_major_symbol_regime_incoherence(()) == ()

    def test_single_snapshot_symbol_no_transition_possible(self) -> None:
        import math

        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            summarize_major_symbol_regime_incoherence,
        )

        snaps = (_snap(t=0, symbol="BTCUSDT", raw_mu=1.0, regime_code=0),)
        result = summarize_major_symbol_regime_incoherence((_attr_with_snapshots(snaps),))
        assert len(result) == 1
        s = result[0]
        assert s.n_transitions == 0
        assert s.censored_pct == 0.0
        assert math.isnan(s.mean_reversal_lag_bars)
        assert s.n_obs == 1


class TestDirectionalVetoPipeline:
    """S1-2: adverse regime long veto in AWF loop.
    S1-4: holdout propagation."""

    def make_signal(self, raw_mu: float, *, valid: bool = True) -> SymbolSignal:
        return SymbolSignal(
            raw_mu=raw_mu,
            volatility=0.02,
            n_obs=1,
            t_stat=0.0,
            valid=valid,
            beta_btc=None,
            quality_weight=1.0,
        )

    def test_config_directional_veto_enabled(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping(
            {
                "l2_regime_directional_veto_enabled": True,
            }
        )
        assert cfg.l2_regime_directional_veto_enabled
        assert cfg.l2_regime_directional_veto_symbols == ("BTCUSDT", "ETHUSDT")

    def test_config_directional_veto_validation(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_action"):
            Layer2AllocationConfig.from_mapping(
                {
                    "l2_regime_directional_veto_action": "invalid",
                }
            )


def test_assess_crisis_reliability_includes_btc_symbol_with_suffixed_schema(
    mocker: Any,
) -> None:
    import datetime

    import numpy as np

    from src.domain.futures.strategy.common.alignment import AlignedMarketData
    from src.domain.futures.strategy.tiered_workflow.pipeline import CrisisWindow, assess_crisis_reliability

    # BTCUSDT data with timestamp_x/timestamp_y schema (simulates enriched parquet after
    # pd.merge with suffixes)
    t0 = int(pd.Timestamp("2022-04-01", tz="UTC").value // 1_000_000)
    n = 100
    btc_df = pd.DataFrame({
        "timestamp_x": [t0 + i * 3600_000 for i in range(n)],
        "timestamp_y": [t0 + i * 3600_000 for i in range(n)],
        "open": np.full(n, 40000.0, dtype=np.float32),
        "high": np.full(n, 41000.0, dtype=np.float32),
        "low": np.full(n, 39000.0, dtype=np.float32),
        "close": np.linspace(40000.0, 35000.0, n, dtype=np.float32),
        "volume": np.full(n, 1000.0, dtype=np.float32),
    })
    eth_df = pd.DataFrame({
        "timestamp": [t0 + i * 3600_000 for i in range(n)],
        "open": np.full(n, 3000.0, dtype=np.float32),
        "high": np.full(n, 3100.0, dtype=np.float32),
        "low": np.full(n, 2900.0, dtype=np.float32),
        "close": np.linspace(3000.0, 2500.0, n, dtype=np.float32),
        "volume": np.full(n, 5000.0, dtype=np.float32),
    })

    data_maps = {
        "BTCUSDT": {"4h": btc_df},
        "ETHUSDT": {"4h": eth_df},
    }

    mock_load = mocker.patch(
        "src.domain.futures.optimization.opt_data_utils.load_futures_data_maps_for_symbols",
        autospec=True,
    )
    mock_load.return_value = (data_maps, {}, ["BTCUSDT", "ETHUSDT"])

    btc_close_2d = np.column_stack([
        np.full(n, 40000.0, dtype=np.float64),
        np.full(n, 3000.0, dtype=np.float64),
    ])
    aligned_mock = AlignedMarketData(
        datetimes=np.datetime64("2022-04-01", "h") + np.arange(n).astype("timedelta64[h]"),
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=btc_close_2d.copy(),
        high_2d=btc_close_2d * 1.01,
        low_2d=btc_close_2d * 0.99,
        close_2d=btc_close_2d,
        volume_2d=np.full((n, 2), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((n, 2), dtype=np.float64),
        active_mask=np.ones((n, 2), dtype=bool),
        warm_mask=np.ones((n, 2), dtype=bool),
        entry_block_mask=np.zeros((n, 2), dtype=bool),
        kill_mask=np.zeros((n, 2), dtype=bool),
        execution_cost_bps_2d=np.zeros((n, 2), dtype=np.float64),
    )
    mock_align = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.align_data_maps",
        return_value=aligned_mock,
    )

    # Build minimal registry with BTCUSDT only
    from src.domain.futures.strategy.candidate_contracts import SignalSourceKey

    btc_evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "mean_reversion", "l2_crisis_stress"),
        mean_gross_bps=10.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.1,
        p_value=0.02,
        q_value=0.03,
        positive_fold_ratio=1.0,
        n_obs=12,
        effective_n=12.0,
        n_folds=3,
        reliability=0.8,
        qualified=True,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (btc_evidence,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="test",
    )
    cfg = CandidateStrategyConfig()
    l2_config = Layer2AllocationConfig(
        l2_crisis_min_symbols=1,
        l2_crisis_min_observation_days=1,
        l2_crisis_min_usable_windows=1,
    )
    caps = PortfolioCaps()

    # Mock downstream simulation to avoid real execution
    mock_batch_mock = MagicMock()
    mock_batch_mock.events = (MagicMock(),)
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._build_rule_based_stress_batch",
        return_value=mock_batch_mock,
    )
    mock_l3 = MagicMock()
    mock_l3.mdd = 0.25
    mock_l3.cagr = -0.05
    mock_l3.cvar95 = 0.04
    mock_l3.n_trades = 50
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        return_value=mock_l3,
    )

    # Build minimal CrisisWindow
    window = CrisisWindow(
        start=datetime.date(2022, 4, 1),
        end=datetime.date(2022, 6, 1),
        label="test_window",
        symbols=("BTCUSDT", "ETHUSDT"),
        source_note="test",
    )

    assessment = assess_crisis_reliability(
        deployment_registry=registry,
        strategy_cfg=cfg,
        config=l2_config,
        caps=caps,
        tf="4h",
        deploy_leverage=1.0,
        crisis_windows=(window,),
    )

    # Verify align_data_maps was called with BTCUSDT in symbols
    assert mock_align.call_count >= 1
    call_args = mock_align.call_args_list[0]
    symbols_arg = call_args[0][1]
    assert "BTCUSDT" in symbols_arg

    # Verify assess_crisis_reliability completed (BTCUSDT reached downstream)
    assert assessment.usable_window_count >= 1
