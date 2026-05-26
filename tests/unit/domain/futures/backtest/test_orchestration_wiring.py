"""Phase 12: Orchestration wiring tests.

테스트 1: objective_ml_phase_d 실행 후 trial.user_attrs에 IS_ROBUST_SCORE 키 존재 확인
          (compute_v3_score가 실제로 호출됨을 간접 검증)
테스트 2: PurgeBarsRegistry.validate() — 미등록 시 RuntimeError 전파
테스트 3: project_all_caps 적용 — gross_cap 초과 weight 입력 → gross ≤ 3.0 확인
테스트 4: evaluate_sequential_promotion_gate → PromotionGateResult 타입 반환 확인
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 테스트 1: compute_v3_score 호출 여부 (IS_ROBUST_SCORE attr 확인)
# ---------------------------------------------------------------------------


def _make_minimal_ctx(n_bars: int = 60) -> Any:
    """합성 데이터로 구성한 최소 MLPhaseDContext."""
    from src.domain.futures.optimization.optimizer import MLPhaseDContext

    rng = np.random.default_rng(0)
    close = 100.0 + rng.standard_normal(n_bars).cumsum()
    xs = rng.standard_normal(n_bars)

    df_mock: Any = MagicMock()
    df_mock.__len__ = lambda self: n_bars
    df_mock.empty = False

    # aligned dict — backtest_engine이 기대하는 키 구성
    aligned: dict[str, Any] = {
        "close": close.reshape(-1, 1),
        "xs_score_long": np.clip(xs, 0, None).reshape(-1, 1),
        "xs_score_short": np.clip(-xs, 0, None).reshape(-1, 1),
        "hmm_prob_crisis": np.zeros((n_bars, 1)),
        "hmm_prob_chop": np.zeros((n_bars, 1)),
        "target_weights": np.zeros((n_bars, 1)),
        "volume": np.ones((n_bars, 1)) * 1e6,
        "open": close.reshape(-1, 1),
        "high": (close + 1).reshape(-1, 1),
        "low": (close - 1).reshape(-1, 1),
        # ATR — True Range 간이 계산 (high - low)
        "atr": np.ones((n_bars, 1)) * 2.0,
        # funding_rate (선택): 없으면 0으로 처리됨
        "funding_rate": np.zeros((n_bars, 1)),
    }
    leg_slice: dict[str, Any] = {
        "data": aligned,
        "leg_range": (0, n_bars),
    }

    ctx = MLPhaseDContext(
        data_maps={},
        symbols=["BTCUSDT"],
        tf="1h",
        seed=0,
        effective_total_trials=10,
        run_id="test_run",
    )
    ctx.awf_leg_slices = [leg_slice]
    ctx.multi_alignment_info = {
        "eff_ref_len": n_bars,
        "alignment_offsets": {},
    }
    ctx.estimated_b = 1.05
    return ctx


def test_v3_score_key_set_in_trial_attrs() -> None:
    """IS_ROBUST_SCORE attr → compute_v3_score 호출 경로 검증."""
    from src.domain.futures.optimization.optimizer import (
        _base_engine_params,
        _evaluate_awf_phase_d_aggregate,
    )

    ctx = _make_minimal_ctx(n_bars=80)
    params = _base_engine_params({}, "1h")

    # trial=None 경로로 실행 (단위 테스트 — Optuna 스터디 불필요)
    _result, diag = _evaluate_awf_phase_d_aggregate(ctx, params, trial=None)

    # robust_val 키가 diag에 존재해야 함
    assert "robust_val" in diag, "compute_v3_score 결과인 robust_val 키가 diag에 없음"
    assert np.isfinite(float(diag["robust_val"])), "robust_val이 유한하지 않음"


# ---------------------------------------------------------------------------
# 테스트 2: PurgeBarsRegistry.validate() — 미등록 시 RuntimeError
# ---------------------------------------------------------------------------


def test_purge_bars_registry_validate_raises_when_empty() -> None:
    """빈 레지스트리에서 validate() → RuntimeError."""
    from src.domain.futures.validation.gates import PurgeBarsRegistry

    registry = PurgeBarsRegistry()
    with pytest.raises(RuntimeError, match="No modules registered purge_bars"):
        registry.validate()


def test_purge_bars_registry_validate_passes_when_registered() -> None:
    """등록 후 validate() → 정상 통과."""
    from src.domain.futures.validation.gates import (
        ModulePurgeBarsMeta,
        PurgeBarsRegistry,
    )

    registry = PurgeBarsRegistry()
    registry.register(
        ModulePurgeBarsMeta(
            module_name="LabelModule",
            purge_bars=24,
            reason="label_horizon",
        )
    )
    # should not raise
    registry.validate()
    assert registry.get_boundary_purge_bars() == 24


def test_objective_propagates_registry_runtime_error() -> None:
    """ctx.registry 설정 후 objective_ml_phase_d 진입 → validate() RuntimeError 전파."""
    import optuna

    from src.domain.futures.optimization.optimizer import (
        objective_ml_phase_d,
    )
    from src.domain.futures.validation.gates import PurgeBarsRegistry

    ctx = _make_minimal_ctx(n_bars=80)
    # 미등록 레지스트리를 주입
    ctx.registry = PurgeBarsRegistry()

    study = optuna.create_study(direction="minimize")

    def _objective(trial: optuna.Trial) -> float:
        objective_ml_phase_d(trial, ctx)
        return 0.0

    with pytest.raises(RuntimeError, match="No modules registered purge_bars"):
        study.optimize(_objective, n_trials=1, catch=())


# ---------------------------------------------------------------------------
# 테스트 3: project_all_caps — gross_cap 초과 weight 입력 → gross ≤ 3.0
# ---------------------------------------------------------------------------


def test_project_all_caps_enforces_gross_cap() -> None:
    """gross_cap=3.0 초과 weight 입력 → L1 norm ≤ 3.0."""
    from src.domain.futures.portfolio.portfolio_constructor import (
        PortfolioCaps,
        project_all_caps,
    )

    rng = np.random.default_rng(42)
    w = rng.uniform(0.5, 1.5, size=10)  # gross ≈ 10 → 3 초과
    btc_beta = np.zeros(10)
    sigma_port = 0.01
    bars_per_year = 365 * 24

    caps = PortfolioCaps(gross=3.0, per_symbol=1.0, target_ann_vol=0.20)
    w_proj = project_all_caps(w, btc_beta, sigma_port, bars_per_year, caps)

    gross_out = float(np.sum(np.abs(w_proj)))
    assert gross_out <= 3.0 + 1e-6, f"gross_cap 위반: {gross_out:.4f} > 3.0"


# ---------------------------------------------------------------------------
# 테스트 4: evaluate_sequential_promotion_gate → PromotionGateResult 반환
# ---------------------------------------------------------------------------


def test_evaluate_sequential_promotion_gate_returns_result() -> None:
    """evaluate_sequential_promotion_gate → PromotionGateResult 타입 확인."""
    from src.domain.futures.validation.champion_registry import (
        ChampionMetricsV3,
        PromotionGateResult,
        evaluate_sequential_promotion_gate,
    )

    candidate = ChampionMetricsV3(
        atomic_oos_pass_ratio=0.80,
        capacity_ceiling_usdt=300_000.0,
        median_log_growth=0.05,
        worst_block_mdd=0.10,
        absolute_decay_bps_yr=-5.0,
        dsr=0.60,
    )

    # wf_result mock: passed=True
    wf_mock = MagicMock()
    wf_mock.passed = True

    # atomic_result mock: pass_ratio >= 0.70
    atomic_mock = MagicMock()
    atomic_mock.pass_ratio = 0.75

    # dual_decay mock: passed=True
    dual_mock = MagicMock()
    dual_mock.passed = True

    capacity_results = {50_000: True, 100_000: True, 250_000: True}

    result = evaluate_sequential_promotion_gate(
        candidate=candidate,
        champion=None,
        wf_result=wf_mock,
        dual_decay=dual_mock,
        atomic_result=atomic_mock,
        capacity_results=capacity_results,
        intrabar_tw=1.10,
        intrabar_mdd=0.15,
    )

    assert isinstance(result, PromotionGateResult), (
        f"반환 타입이 PromotionGateResult가 아님: {type(result)}"
    )
    assert result.passed is True, f"gate_failures={result.gate_failures}"
    assert result.promoted_to_champion is True


def test_strategy_compose_wiring_produces_target_weights_with_policy_contract() -> None:
    from src.domain.futures.optimization.objectives import _compose_strategy_scores_inplace
    from src.domain.futures.portfolio.portfolio_constructor import precompute_rebalance_weights
    from src.domain.futures.portfolio.portfolio_optimizer import PortfolioPolicyInputs

    n_bars, n_syms = 8, 2
    close = np.column_stack(
        [
            np.linspace(100.0, 105.0, num=n_bars),
            np.linspace(101.0, 106.0, num=n_bars),
        ]
    ).astype(np.float64)
    alpha_long = np.full((n_bars, n_syms), 0.004, dtype=np.float64)
    alpha_short = np.zeros((n_bars, n_syms), dtype=np.float64)
    execution_cost_bps = np.full((n_bars, n_syms), 1.0, dtype=np.float64)
    sigma_3d = np.repeat(np.eye(n_syms, dtype=np.float64)[None, :, :] * 2e-4, n_bars, axis=0)
    beta_2d = np.ones((n_bars, n_syms), dtype=np.float64)

    aligned: dict[str, Any] = {
        "alpha_long": alpha_long,
        "alpha_short": alpha_short,
        "execution_cost_bps_2d": execution_cost_bps,
    }
    params: dict[str, Any] = {
        "BETA_ALPHA": 1.0,
        "EV_HURDLE_BPS": 5.0,
    }
    _compose_strategy_scores_inplace(aligned, params)

    policy_inputs = PortfolioPolicyInputs(
        mu_long_2d=np.asarray(aligned["mu_long_2d"], dtype=np.float64),
        mu_short_2d=np.asarray(aligned["mu_short_2d"], dtype=np.float64),
        risk_sigma_3d=sigma_3d,
        risk_beta_2d=beta_2d,
        cost_fraction_2d=np.asarray(aligned["mu_long_2d"], dtype=np.float64) * 0.0,
        cost_bps_2d=execution_cost_bps,
        cost_source=str(aligned["_strategy_cost_snapshot_meta"]["execution_cost_bps_source"]),
    )

    w = precompute_rebalance_weights(
        close,
        np.asarray(aligned["xs_score_long"], dtype=np.float64),
        np.asarray(aligned["xs_score_short"], dtype=np.float64),
        rebalance_bars=1,
        lookback=5,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=0.8,
        per_symbol_cap=0.3,
        sigma_3d=sigma_3d,
        btc_beta_2d=beta_2d,
        policy_inputs=policy_inputs,
    )

    assert w.shape == (n_bars, n_syms)
    assert float(np.sum(np.abs(w[-1]))) > 0.0
    assert "_strategy_cost_snapshot_meta" in aligned
