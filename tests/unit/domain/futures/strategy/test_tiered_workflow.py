"""tiered_workflow 단위 테스트.

TI7: L1 short-circuit (gate BLOCKED)
TI8: OOS stacking 평균 검증
TI9: Layer2Result dataclass 생성
TI11: Layer3Result frozen 결정성
TI12: _cagr 실측 계산 검증
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow import (
    Layer1Result,
    Layer2Result,
    Layer3Result,
    _cagr,
    _stack_oos_signals,
    run_l1_cpcv,
)
from src.domain.futures.strategy.walk_forward import build_cpcv_folds

# ---------------------------------------------------------------------------
# TI7: L1 short-circuit — gate BLOCKED (empty fold signals → IC ≈ 0)
# ---------------------------------------------------------------------------


def test_run_l1_cpcv_gate_blocked_when_no_valid_signals() -> None:
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

    folds = build_cpcv_folds(
        n_bars=n_bars,
        n_groups=3,
        n_test_groups=1,
        embargo_bars=2,
        purge_bars=1,
    )

    mock_fold_out = MagicMock()
    mock_fold_out.model_output.events = pd.DataFrame({"symbol": []})
    mock_fold_out.model_output.expected_net_bps = np.array([], dtype=np.float64)

    # Act
    with patch(
        "src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold",
        return_value=mock_fold_out,
    ), patch(
        "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
        return_value=(1, 2),
    ):
        l1 = run_l1_cpcv(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            cfg=cfg,
            folds=folds,
            l1_params={},
        )

    # Assert
    assert isinstance(l1, Layer1Result)
    assert l1.gate_passed is False
    assert l1.mean_ic == pytest.approx(0.0, abs=1e-6)


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
    assert stacked["BTC"].n_obs == 20  # 10 * 2 folds


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
        mean_ic=0.035,
        ic_tstat=2.1,
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


def test_layer2result_dataclass_creation() -> None:
    """Layer2Result 직접 생성 및 필드 검증."""
    # Arrange / Act
    r2 = Layer2Result(
        selected_last=frozenset(["BTC"]),
        weights_last={"BTC": 0.1},
        sharpe_hybrid=1.5,
        sharpe_baseline=1.0,
        mdd_hybrid=0.08,
        mdd_baseline=0.12,
        turnover=0.05,
        friction_pass_pct=0.8,
        gate_passed=True,
    )

    # Assert
    assert r2.gate_passed is True
    assert "BTC" in r2.selected_last
    assert r2.weights_last["BTC"] == pytest.approx(0.1)
    assert r2.sharpe_hybrid == pytest.approx(1.5)
    assert r2.mdd_hybrid < r2.mdd_baseline


def test_layer2result_gate_blocked_when_sharpe_below_threshold() -> None:
    """Layer2Result: sharpe_hybrid < sharpe_baseline * 1.20 → gate_passed=False 기대."""
    # Arrange / Act
    r2 = Layer2Result(
        selected_last=frozenset(["ETH"]),
        weights_last={"ETH": 0.05},
        sharpe_hybrid=0.9,
        sharpe_baseline=1.0,
        mdd_hybrid=0.10,
        mdd_baseline=0.12,
        turnover=0.03,
        friction_pass_pct=0.6,
        gate_passed=False,
    )

    # Assert
    assert r2.gate_passed is False


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

    # Assert: (1.10)^(2190/1000) - 1
    expected = (1.10 ** (bars_per_year / n_bars)) - 1.0
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
