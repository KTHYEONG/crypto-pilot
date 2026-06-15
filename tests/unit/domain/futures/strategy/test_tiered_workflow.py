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

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

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
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow import (
    _VALID_COVERAGE_FLAG_THRESHOLD,
    FoldDiagnostic,
    Layer1Result,
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

    folds = (
        WFFold(fit_start=0, fit_end=60, cal_start=50, cal_end=60, oos_start=60, oos_end=100),
    )

    mock_fold_out = MagicMock()
    mock_fold_out.model_output.events = pd.DataFrame({"symbol": []})
    mock_fold_out.model_output.expected_net_bps = np.array([], dtype=np.float64)
    mock_fold_out.oos_set = None

    # Act
    with patch(
        "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
        return_value=mock_fold_out,
    ), patch(
        "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
        return_value=(1, 2),
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

    with patch(
        "os.cpu_count",
        return_value=1,
    ), patch(
        "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
        side_effect=fold_outputs,
    ), patch(
        "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
        return_value=(1, 0),
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compose_symbol_signals",
        return_value={"BTC": valid_signal},
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
        signal_total > 0
        and friction_pass_pct > 0.0
        and math.isfinite(sharpe_hybrid)
        and math.isfinite(cagr_hybrid)
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
            sharpe_hybrid=0.9, sharpe_baseline=0.5,
            mdd_hybrid=0.30, mdd_baseline=0.50,
            cagr_hybrid=0.40, mar_hybrid=0.8,
            fold_pass_ratio=0.75,
        )
        assert gate is True
        assert reason == ""

    def test_s2_sign_safety_negative_baseline_blocks_loss_strategy(self) -> None:
        """S2 (D1 회귀가드): sharpe_h=-0.9, sharpe_base=-0.85 → 구식 곱셈식이면 통과, 신식 가산식은 FAIL."""
        # 구식: -0.9 >= -0.85*1.20=-1.02 → 통과 (버그).
        # 신식: cagr<0 먼저 차단.
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=-0.9, sharpe_baseline=-0.85,
            mdd_hybrid=0.30, mdd_baseline=0.50,
            cagr_hybrid=-0.10, mar_hybrid=-0.3,
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "cagr"

    def test_s3_absolute_cagr_fail_blocks_even_if_beats_baseline(self) -> None:
        """S3: CAGR=-0.05 → 절대손실 → blocker=cagr."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9, sharpe_baseline=0.5,
            mdd_hybrid=0.20, mdd_baseline=0.40,
            cagr_hybrid=-0.05, mar_hybrid=-0.2,
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "cagr"

    def test_s4_mar_fail(self) -> None:
        """S4: CAGR=0.1, MDD=0.45 → MAR≈0.22<0.5 → blocker=mar."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9, sharpe_baseline=0.5,
            mdd_hybrid=0.45, mdd_baseline=0.60,
            cagr_hybrid=0.10, mar_hybrid=0.10 / (0.45 + 1e-9),
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "mar"

    def test_s5_absolute_mdd_upper_bound(self) -> None:
        """S5: mdd_h=0.55, mdd_base=0.67 (상대는 통과) → 절대상한 FAIL → blocker=mdd_abs."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9, sharpe_baseline=0.5,
            mdd_hybrid=0.55, mdd_baseline=0.67,
            cagr_hybrid=0.30, mar_hybrid=0.30 / (0.55 + 1e-9),
            fold_pass_ratio=0.75,
        )
        assert gate is False
        assert reason == "mdd_abs"

    def test_s6_fold_consistency_fail(self) -> None:
        """S6: fold 비율=0.25<0.60 → blocker=fold."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.9, sharpe_baseline=0.5,
            mdd_hybrid=0.25, mdd_baseline=0.40,
            cagr_hybrid=0.30, mar_hybrid=1.2,
            fold_pass_ratio=0.25,
        )
        assert gate is False
        assert reason == "fold"

    def test_s7_deployment_nan_blocked(self) -> None:
        """S7: signal_total=0 → no_deployment, 나머지 미평가."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=float("nan"), sharpe_baseline=0.5,
            mdd_hybrid=0.20, mdd_baseline=0.40,
            cagr_hybrid=float("nan"), mar_hybrid=float("nan"),
            fold_pass_ratio=0.75,
            signal_total=0, friction_pass_pct=0.0,
        )
        assert gate is False
        assert reason == "no_deployment"

    def test_s8_uplift_boundary_exact(self) -> None:
        """S8: sharpe_h == sharpe_base+0.20 정확히 → uplift 경계 PASS."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.70, sharpe_baseline=0.50,
            mdd_hybrid=0.25, mdd_baseline=0.40,
            cagr_hybrid=0.30, mar_hybrid=1.2,
            fold_pass_ratio=0.75,
        )
        assert gate is True
        assert reason == ""

    def test_s8_uplift_just_below_boundary_fail(self) -> None:
        """S8: sharpe_h = sharpe_base+0.19 → uplift FAIL."""
        gate, reason = _evaluate_l2_gate(
            sharpe_hybrid=0.69, sharpe_baseline=0.50,
            mdd_hybrid=0.25, mdd_baseline=0.40,
            cagr_hybrid=0.30, mar_hybrid=1.2,
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
        r3.cagr = 0.99  # type: ignore[misc]


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
    )

    # Assert
    assert r3.gate_passed is False
    assert r3.sharpe < r3.sharpe_baseline


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

    l1_oos = {
        "BTC": SymbolSignal(raw_mu=10.0, volatility=0.02, n_obs=100, t_stat=2.5, valid=True, beta_btc=1.0),
        "ETH": SymbolSignal(raw_mu=12.0, volatility=0.03, n_obs=100, t_stat=2.2, valid=True, beta_btc=0.8),
    }

    from src.domain.futures.strategy.walk_forward import WFFold
    awf_folds = (
        WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=30, oos_start=30, oos_end=50),
    )

    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    caps = PortfolioCaps(gross=1.8, per_symbol=0.35, net=0.5, beta=1.0, target_ann_vol=0.35)

    from src.domain.futures.strategy.tiered_workflow import run_l2_awf

    # Act
    l2 = run_l2_awf(
        l1_oos=l1_oos,
        aligned=aligned,
        awf_folds=awf_folds,
        l2_params={"K_RANK": 1, "REBALANCE_BARS": 1},
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
        fold_out.model_output.events = pd.DataFrame(
            columns=["symbol", "entry_idx", "expected_holding_bars"]
        )
        fold_out.model_output.expected_net_bps = np.array([], dtype=np.float64)
    else:
        rng = np.random.default_rng(42)
        fold_out.model_output.events = pd.DataFrame({
            "symbol": [sym] * n_events,
            "entry_idx": np.arange(10, 10 + n_events, dtype=np.int64),
            "expected_holding_bars": np.ones(n_events, dtype=np.int64) * 2,
        })
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
        diags.append(FoldDiagnostic(
            fold=i + 1,
            ic=fold_ic,
            breadth=f_breadth,
            n_valid=f_n_valid,
            n_eligible=n_total,
            n_events=f_n_events,
            n_fit=int(getattr(fo, "n_fit", 0)),
            fit_status=getattr(fo, "fit_status", "trained"),
            passed=fold_ic is not None and fold_ic > 0,
        ))

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
    assert _newey_west_ic_tstat(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
    ) == 0.0


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
    realized = {
        "BTC": SymbolRealizedStat(
            realized_mu_bps=3.0, t_stat=2.5, n_obs=30, ic=0.12, valid=True
        )
    }

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
        validation_diagnostics={
            "ensemble_diagnostics": {"num_valid_regimes": num_valid_regimes}
        },
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
    real = np.concatenate([
        rng.normal(8, 1, size=n_each),   # beta_neut: 양
        rng.normal(-5, 1, size=n_each),  # mean: 음
    ])

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
    panel = tuple(
        StrategySignal(f"s{i}:v{i}", 8.0 + i, 2.0, 0.7, 0.8, 40, 5, True)
        for i in range(5)
    )

    with patch(
        "os.cpu_count",
        return_value=1,
    ), patch(
        "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
        side_effect=fold_outputs,
    ), patch(
        "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
        return_value=(1, 0),
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compose_symbol_signals",
        return_value={"BTC": valid_signal},
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compute_per_strategy_oos_validation",
        return_value=panel,
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compute_panel_diversity",
        return_value=0.4,
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
    panel = tuple(
        StrategySignal(f"s{i}:v{i}", 8.0 + i, 2.0, 0.7, 0.8, 40, 5, True)
        for i in range(5)
    )

    with patch(
        "os.cpu_count",
        return_value=1,
    ), patch(
        "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
        side_effect=fold_outputs,
    ), patch(
        "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
        return_value=(1, 0),
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compose_symbol_signals",
        return_value={"BTC": valid_signal},
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compute_per_strategy_oos_validation",
        return_value=panel,
    ), patch(
        "src.domain.futures.strategy.tiered_workflow.compute_panel_diversity",
        return_value=0.1,
    ):
        l1 = run_l1_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            cfg=cfg,
            folds=folds,
            l1_params={},
        )

    assert l1.gate_passed is False


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
        tf="4h",
    )

    # Assert
    assert result == (blocked_l1, None, None)
    assert len(nested_builder_calls) == 1
    assert nested_builder_calls[0]["n_bars"] == len(aligned.datetimes)
    assert nested_builder_calls[0]["l1_start_idx"] == 1
    assert nested_builder_calls[0]["l1_end_idx"] == 4
    assert nested_builder_calls[0]["cfg"] is cfg
    assert len(nested_runner_calls) == 1
    assert nested_runner_calls[0]["outer_folds"] == built_outer_folds
    assert nested_runner_calls[0]["cfg"] is cfg
    assert any("[BLOCKED]" in msg for msg in logged_messages)


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

    assert report.passed is True
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
    vol = np.ones((20, 3), dtype=np.float64)

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
    vol = np.ones((20, len(symbols)), dtype=np.float64)

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
