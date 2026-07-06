"""SSOT: evaluate_l2_trial ↔ run_l2_awf 통일 계약 검증.

Scenarios:
  S1 — deploy_leverage_override 적용
  S2 — None 보존 (내부 calibrate)
  S3 — override ≤ 1.0 무시
  S4 — run_l2_awf vs evaluate_l2_trial parity 불변식 (핵심)
  S5 — champion L* 경로 일치
  S6 — adapter 필드 매핑
  S7 — 빈 folds fallback
  S8 — gate 판정 동등성 (회귀 방지 스냅샷)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.futures.optimization.workflow import evaluate_l2_trial
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,
    Layer2AllocationConfig,
    Layer2BlockMetric,
    Layer2DeployableScore,
    Layer2GateEvaluation,
    Layer2TrialEvaluation,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    _layer2_result_from_trial_eval,
    run_l2_awf,
)
from src.domain.futures.strategy.walk_forward import WFFold

# ── helpers ──────────────────────────────────────────────────────────


def _make_gate(promotion_passed: bool = True, blocker: str = "") -> Layer2GateEvaluation:
    return Layer2GateEvaluation(
        optuna_constraint_values=(),
        promotion_passed=promotion_passed,
        promotion_blocker=blocker,
        promotion_constraint_values=(),
    )


def _dummy_metrics(n: int = 2) -> tuple[Layer2BlockMetric, ...]:
    return tuple(
        Layer2BlockMetric(
            start_idx=i * 10,
            end_idx=(i + 1) * 10,
            log_growth_hybrid=0.01,
            log_growth_baseline=0.005,
            mdd_hybrid=0.02,
            turnover_hybrid=0.3,
            active_rebalances=5,
        )
        for i in range(n)
    )


def _make_fake_sim(*, l_star: float = 1.0) -> MagicMock:
    """_AwfSimResult 호환 mock."""
    sim = MagicMock()
    sim.rets_hybrid = [0.001] * 100
    sim.rets_baseline = [0.0005] * 100
    sim.rets_baseline_ew = [0.0004] * 100
    sim.fit_rets_hybrid = [0.002] * 50
    sim.last_selected = frozenset({"BTC", "ETH"})
    sim.last_w = np.array([0.6, 0.4], dtype=np.float64)
    sim.all_turnovers = [0.3] * 10
    sim.all_gross_exposures = [0.8] * 10
    sim.all_net_exposures = [0.5] * 10
    sim.friction_pass_total = 90
    sim.signal_total = 100
    sim.support_leak_count = 0
    sim.total_cost_hybrid = 0.0005
    sim.cap_saturation_count = 2
    sim.rebalance_count = 10
    sim.trade_count = 50
    sim.fold_rets_hybrid = [[0.002] * 30, [0.001] * 30]
    sim.block_rets_hybrid = (tuple([0.001] * 10), tuple([0.001] * 10))
    sim.block_rets_baseline = (tuple([0.0005] * 10), tuple([0.0005] * 10))
    sim.fold_selected_symbols = (("BTC", "ETH"), ("BTC",))
    sim.fold_attributions = ()
    return sim


def _make_minimal_eval(
    *,
    cagr_hybrid: float = 0.35,
    mdd_hybrid: float = 0.15,
    fold_pass_ratio: float = 0.75,
    trade_count: int = 50,
    deploy_leverage: float = 2.5,
    deploy_binding: str = "mdd",
    gate: Layer2GateEvaluation | None = None,
    **overrides: Any,
) -> Layer2TrialEvaluation:
    """Minimal Layer2TrialEvaluation for adapter/parity tests."""
    base: dict[str, Any] = {
        "objective_value": 1.0,
        "constraint_values": (),
        "cagr_hybrid": cagr_hybrid,
        "cagr_baseline": 0.15,
        "growth_lcb_hybrid": 0.10,
        "growth_lcb_baseline": 0.05,
        "sharpe_hac_hybrid": 1.5,
        "sharpe_hac_baseline": 0.8,
        "psr_hybrid": 0.6,
        "mdd_hybrid": mdd_hybrid,
        "cvar_95_hybrid": 0.12,
        "fold_pass_ratio": fold_pass_ratio,
        "break_even_pass_pct": 0.85,
        "average_gross_exposure": 0.8,
        "cap_saturation_ratio": 0.2,
        "total_cost_bps": 5.0,
        "block_metrics": _dummy_metrics(2),
        "returns_hybrid": tuple([0.001] * 100),
        "returns_baseline": tuple([0.0005] * 100),
        "sharpe_hybrid": 1.8,
        "sharpe_hac_baseline_ew": 0.9,
        "sortino_hybrid": 2.0,
        "trade_count": trade_count,
        "risk_utilization": 0.6,
        "deployment_objective_bonus": 0.0,
        "worst_fold_sharpe": 0.5,
        "gate": gate or _make_gate(True, ""),
        "fit_returns_hybrid": (),
        "deploy_leverage": deploy_leverage,
        "deploy_binding": deploy_binding,
        "recent_fold_passed": True,
        "recent_fold_sharpe": 1.2,
        "recent_fold_cagr": 0.30,
        "recent_fold_mdd": 0.10,
        "latest_to_median_cagr": 1.0,
        "fold_deployed_cagrs": (0.3, 0.25),
        "fold_selected_symbols": (("BTC", "ETH"), ("BTC",)),
        "worst_fold_cagr": 0.25,
        "positive_block_delta_ratio": 0.6,
        "bucket_reliability_mean": 0.7,
        "entry_spike_penalty": 0.0,
        "deployable_score": Layer2DeployableScore(
            cagr=0.35,
            sortino=2.0,
            sharpe=1.8,
            calmar=2.33,
            mdd=0.15,
            fold_pass_ratio=0.75,
            score=0.8,
            worst_fold_cagr=0.25,
            positive_block_delta_ratio=0.6,
            cost_drag=0.02,
            bucket_reliability_mean=0.7,
            entry_spike_penalty=0.0,
        ),
        "last_selected_symbols": ("BTC", "ETH"),
        "last_weights": (0.6, 0.4),
        "all_turnovers": (0.3, 0.25, 0.35),
        "rebalance_count": 10,
        "all_net_exposures": (0.5, 0.55),
        "rets_baseline_ew": tuple([0.0004] * 100),
    }
    base.update(overrides)
    return Layer2TrialEvaluation(**base)


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def base_config() -> Layer2AllocationConfig:
    return Layer2AllocationConfig(
        k_rank=3,
        rebalance_bars=3,
        kelly_fraction=0.25,
        l2_min_cagr=0.30,
        l2_min_mar=1.0,
        l2_max_mdd_abs=0.31,
        l2_max_cvar_95=0.12,
        l2_deploy_mdd_margin=0.10,
        l2_deploy_cvar_margin=0.10,
        l2_deploy_l_hard_cap=10.0,
        l2_max_exchange_leverage=20.0,
        l2_growth_lcb_z=1.645,
        l2_min_dsr=0.0,
    )


def _make_folds(n: int = 2) -> tuple[WFFold, ...]:
    return tuple(
        WFFold(
            fit_start=i * 10,
            fit_end=(i + 1) * 10,
            cal_start=i * 10,
            cal_end=(i + 1) * 10,
            oos_start=(i + 1) * 10,
            oos_end=(i + 2) * 10,
        )
        for i in range(n)
    )


# ── S1: override 적용 ────────────────────────────────────────────────


class TestEvaluateL2TrialOverride:
    """C1: deploy_leverage_override 파라미터 검증."""

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_uses_leverage_override(
        self,
        mock_audit: MagicMock,
        mock_fold_diag: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_sim: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S1: override=3.5 → deploy_leverage==3.5, deploy_binding=='champion'."""
        mock_sim.return_value = _make_fake_sim()
        mock_fold_diag.return_value = MagicMock(
            fold_pass_ratio=0.75,
            fold_deployed_cagrs=[0.3, 0.25],
            recent_fold_passed=True,
            recent_fold_sharpe=1.2,
            recent_fold_cagr=0.3,
            recent_fold_mdd=0.1,
            latest_to_median_cagr=1.0,
            fold_selected_symbols=[("BTC", "ETH"), ("BTC",)],
        )
        mock_audit.return_value = MagicMock(warnings=())
        mock_gate.return_value = MagicMock(
            optuna_constraint_values=(),
            promotion_passed=True,
            promotion_blocker="",
        )

        eval = evaluate_l2_trial(
            cache=MagicMock(spec=L2SimulationCache),
            signal_batch=MagicMock(),
            aligned=MagicMock(symbols=("BTC", "ETH"), datetimes=np.array(["2024-01-01"], dtype="datetime64[ns]")),
            awf_folds=_make_folds(2),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            deploy_leverage_override=3.5,
        )

        assert eval.deploy_leverage == pytest.approx(3.5, rel=1e-12)
        assert eval.deploy_binding == "champion"
        mock_calibrate.assert_not_called()

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_none_override_preserves_calibration(
        self,
        mock_audit: MagicMock,
        mock_fold_diag: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_sim: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S2: override=None → 내부 calibrate 경로 실행."""
        mock_sim.return_value = _make_fake_sim()
        mock_calibrate.return_value = (2.0, "mdd", 0.12)
        mock_fold_diag.return_value = MagicMock(
            fold_pass_ratio=0.75,
            fold_deployed_cagrs=[0.3, 0.25],
            recent_fold_passed=True,
            recent_fold_sharpe=1.2,
            recent_fold_cagr=0.3,
            recent_fold_mdd=0.1,
            latest_to_median_cagr=1.0,
            fold_selected_symbols=[("BTC", "ETH"), ("BTC",)],
        )
        mock_audit.return_value = MagicMock(warnings=())
        mock_gate.return_value = MagicMock(
            optuna_constraint_values=(),
            promotion_passed=True,
            promotion_blocker="",
        )

        eval = evaluate_l2_trial(
            cache=MagicMock(spec=L2SimulationCache),
            signal_batch=MagicMock(),
            aligned=MagicMock(symbols=("BTC", "ETH"), datetimes=np.array(["2024-01-01"], dtype="datetime64[ns]")),
            awf_folds=_make_folds(2),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            deploy_leverage_override=None,
        )

        mock_calibrate.assert_called_once()
        # deploy_leverage should be from calibrate (mock returns 2.0)
        assert eval.deploy_leverage == pytest.approx(2.0, rel=1e-12)

    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim._run_awf_simulation")
    @patch("src.domain.futures.strategy.tiered_workflow.l2_gate.evaluate_layer2_gate")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.calibrate_deployment_leverage")
    @patch("src.domain.futures.strategy.tiered_workflow.risk_deployment.compute_layer2_fold_diagnostics")
    @patch("src.domain.futures.optimization.workflow.build_layer_universe_audit")
    def test_ignores_subunit_override(
        self,
        mock_audit: MagicMock,
        mock_fold_diag: MagicMock,
        mock_calibrate: MagicMock,
        mock_gate: MagicMock,
        mock_sim: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S3: override=0.8 (≤1.0) → 무시하고 내부 calibrate."""
        mock_sim.return_value = _make_fake_sim()
        mock_calibrate.return_value = (1.5, "mdd", 0.08)
        mock_fold_diag.return_value = MagicMock(
            fold_pass_ratio=0.75,
            fold_deployed_cagrs=[0.3, 0.25],
            recent_fold_passed=True,
            recent_fold_sharpe=1.2,
            recent_fold_cagr=0.3,
            recent_fold_mdd=0.1,
            latest_to_median_cagr=1.0,
            fold_selected_symbols=[("BTC", "ETH"), ("BTC",)],
        )
        mock_audit.return_value = MagicMock(warnings=())
        mock_gate.return_value = MagicMock(
            optuna_constraint_values=(),
            promotion_passed=True,
            promotion_blocker="",
        )

        eval = evaluate_l2_trial(
            cache=MagicMock(spec=L2SimulationCache),
            signal_batch=MagicMock(),
            aligned=MagicMock(symbols=("BTC", "ETH"), datetimes=np.array(["2024-01-01"], dtype="datetime64[ns]")),
            awf_folds=_make_folds(2),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            deploy_leverage_override=0.8,
        )

        mock_calibrate.assert_called_once()
        assert eval.deploy_leverage == pytest.approx(1.5, rel=1e-12)


# ── S4: parity 불변식 (핵심) ─────────────────────────────────────────


class TestRunL2AwfParity:
    """C2/C3: run_l2_awf가 evaluate_l2_trial에 위임 후 parity 유지."""

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache")
    def test_run_l2_awf_matches_evaluate_l2_trial_exactly(
        self,
        mock_build_cache: MagicMock,
        mock_eval: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S4: 동일 입력으로 run_l2_awf와 evaluate_l2_trial의 공통 지표가 일치."""
        mock_build_cache.return_value = MagicMock()
        common_eval = _make_minimal_eval(
            gate=_make_gate(True, ""),
        )
        mock_eval.return_value = common_eval

        result = run_l2_awf(
            signal_batch=MagicMock(),
            aligned=MagicMock(
                symbols=("BTC", "ETH"),
                datetimes=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            ),
            awf_folds=_make_folds(2),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            verbose=False,
            deploy_leverage=None,
        )

        assert result.cagr_hybrid == pytest.approx(common_eval.cagr_hybrid, rel=1e-12)
        assert result.mdd_hybrid == pytest.approx(common_eval.mdd_hybrid, rel=1e-12)
        assert result.trade_count == common_eval.trade_count
        assert result.fold_pass_ratio == pytest.approx(common_eval.fold_pass_ratio, rel=1e-12)
        assert result.sharpe_hybrid == pytest.approx(common_eval.sharpe_hybrid, rel=1e-12)
        assert result.sharpe_hac_hybrid == pytest.approx(common_eval.sharpe_hac_hybrid, rel=1e-12)
        assert result.sortino_hybrid == pytest.approx(common_eval.sortino_hybrid, rel=1e-12)
        assert result.psr_hybrid == pytest.approx(common_eval.psr_hybrid, rel=1e-12)
        assert result.risk_utilization == pytest.approx(common_eval.risk_utilization, rel=1e-12)
        assert result.deploy_leverage == pytest.approx(common_eval.deploy_leverage, rel=1e-12)
        assert result.recent_fold_passed == common_eval.recent_fold_passed
        assert result.recent_fold_sharpe == pytest.approx(common_eval.recent_fold_sharpe, rel=1e-12)
        assert result.recent_fold_cagr == pytest.approx(common_eval.recent_fold_cagr, rel=1e-12)
        assert result.recent_fold_mdd == pytest.approx(common_eval.recent_fold_mdd, rel=1e-12)

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache")
    def test_run_l2_awf_champion_leverage_matches_trial_override(
        self,
        mock_build_cache: MagicMock,
        mock_eval: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S5: deploy_leverage=3.9 전달 시 run_l2_awf의 L*가 eval에 전파."""
        mock_build_cache.return_value = MagicMock()
        mock_eval.return_value = _make_minimal_eval(
            deploy_leverage=3.9,
            deploy_binding="champion",
            gate=_make_gate(True, ""),
        )

        result = run_l2_awf(
            signal_batch=MagicMock(),
            aligned=MagicMock(
                symbols=("BTC", "ETH"),
                datetimes=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            ),
            awf_folds=_make_folds(2),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            verbose=False,
            deploy_leverage=3.9,
        )

        mock_eval.assert_called_once()
        _, kwargs = mock_eval.call_args
        assert kwargs["deploy_leverage_override"] == 3.9
        assert result.deploy_leverage == pytest.approx(3.9, rel=1e-12)

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache")
    def test_run_l2_awf_empty_folds_fallback(
        self,
        mock_build_cache: MagicMock,
        mock_eval: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S7: 빈 folds → crash 없이 fallback 처리."""
        mock_build_cache.return_value = MagicMock()
        mock_eval.return_value = _make_minimal_eval(
            gate=_make_gate(False, "no_folds"),
        )

        result = run_l2_awf(
            signal_batch=MagicMock(),
            aligned=MagicMock(
                symbols=("BTC", "ETH"),
                datetimes=np.array(["2024-01-01"], dtype="datetime64[ns]"),
            ),
            awf_folds=(),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            verbose=False,
            deploy_leverage=None,
        )
        assert result is not None

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache")
    def test_run_l2_awf_gate_verdict_unchanged(
        self,
        mock_build_cache: MagicMock,
        mock_eval: MagicMock,
        base_config: Layer2AllocationConfig,
    ) -> None:
        """S8: 위임 후 gate_passed/blocker_reason이 eval.gate와 동일."""
        mock_build_cache.return_value = MagicMock()
        mock_eval.return_value = _make_minimal_eval(
            gate=_make_gate(False, "low_cagr"),
        )

        result = run_l2_awf(
            signal_batch=MagicMock(),
            aligned=MagicMock(
                symbols=("BTC", "ETH"),
                datetimes=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            ),
            awf_folds=_make_folds(2),
            config=base_config,
            caps=MagicMock(),
            tf="4h",
            verbose=False,
            deploy_leverage=None,
        )

        assert result.gate_passed is False
        assert result.blocker_reason == "low_cagr"


# ── S6: adapter 필드 매핑 ────────────────────────────────────────────


class TestLayer2ResultAdapter:
    """C3: _layer2_result_from_trial_eval 매핑 정확성."""

    def test_adapter_maps_common_fields(self) -> None:
        """S6: 공통 16+ 필드가 eval과 동일, 배포 전용은 extras 반영."""
        gate = _make_gate(True, "")
        eval = _make_minimal_eval(gate=gate)
        extras = {
            "sharpe_baseline": 0.9,
            "mdd_baseline": 0.10,
            "cagr_baseline": 0.15,
            "turnover": 0.3,
            "average_net_exposure": 0.5,
            "n_rebalances": 10,
            "dsr_hybrid": 0.6,
            "terminal_multiple": 1.25,
        }

        result = _layer2_result_from_trial_eval(
            eval,
            gate_passed=True,
            blocker_reason="",
            extras=extras,
        )

        # 공통 필드 1:1 복사 검증
        assert result.sharpe_hybrid == pytest.approx(eval.sharpe_hybrid)
        assert result.mdd_hybrid == pytest.approx(eval.mdd_hybrid)
        assert result.cagr_hybrid == pytest.approx(eval.cagr_hybrid)
        assert result.fold_pass_ratio == pytest.approx(eval.fold_pass_ratio)
        assert result.sharpe_hac_hybrid == pytest.approx(eval.sharpe_hac_hybrid)
        assert result.sharpe_hac_baseline == pytest.approx(eval.sharpe_hac_baseline)
        assert result.psr_hybrid == pytest.approx(eval.psr_hybrid)
        assert result.growth_lcb_hybrid == pytest.approx(eval.growth_lcb_hybrid)
        assert result.growth_lcb_baseline == pytest.approx(eval.growth_lcb_baseline)
        assert result.sortino_hybrid == pytest.approx(eval.sortino_hybrid)
        assert result.trade_count == eval.trade_count
        assert result.risk_utilization == pytest.approx(eval.risk_utilization)
        assert result.deploy_leverage == pytest.approx(eval.deploy_leverage)
        assert result.average_gross_exposure == pytest.approx(eval.average_gross_exposure)
        assert result.cap_saturation_ratio == pytest.approx(eval.cap_saturation_ratio)
        assert result.total_cost_bps == pytest.approx(eval.total_cost_bps)
        assert result.cvar_95_hybrid == pytest.approx(eval.cvar_95_hybrid)
        assert result.recent_fold_passed == eval.recent_fold_passed
        assert result.recent_fold_sharpe == pytest.approx(eval.recent_fold_sharpe)
        assert result.recent_fold_cagr == pytest.approx(eval.recent_fold_cagr)
        assert result.recent_fold_mdd == pytest.approx(eval.recent_fold_mdd)
        assert result.friction_pass_pct == pytest.approx(eval.break_even_pass_pct)

        # 배포 전용 extras 반영 검증
        assert result.sharpe_baseline == pytest.approx(extras["sharpe_baseline"])
        assert result.mdd_baseline == pytest.approx(extras["mdd_baseline"])
        assert result.cagr_baseline == pytest.approx(extras["cagr_baseline"])
        assert result.turnover == pytest.approx(extras["turnover"])
        assert result.average_net_exposure == pytest.approx(extras["average_net_exposure"])
        assert result.n_rebalances == extras["n_rebalances"]
        assert result.dsr_hybrid == pytest.approx(extras["dsr_hybrid"])
        assert result.terminal_multiple == pytest.approx(extras["terminal_multiple"])
        assert result.total_pnl_pct == pytest.approx(extras["terminal_multiple"] - 1.0)

        # gate
        assert result.gate_passed is True
        assert result.blocker_reason == ""
        assert result.allocation_policy == "diagonal_kelly"

    def test_adapter_weights_last_mapping(self) -> None:
        """S6b: last_selected_symbols/weights_last 매핑."""
        eval = _make_minimal_eval(
            last_selected_symbols=("BTC", "ETH"),
            last_weights=(0.6, 0.4),
        )
        extras = {
            "sharpe_baseline": 0.9,
            "mdd_baseline": 0.1,
            "cagr_baseline": 0.15,
            "turnover": 0.3,
            "average_net_exposure": 0.5,
            "n_rebalances": 10,
            "dsr_hybrid": 0.6,
            "terminal_multiple": 1.25,
        }

        result = _layer2_result_from_trial_eval(
            eval,
            gate_passed=True,
            blocker_reason="",
            extras=extras,
        )

        assert result.selected_last == frozenset({"BTC", "ETH"})
        assert result.weights_last["BTC"] == pytest.approx(0.6)
        assert result.weights_last["ETH"] == pytest.approx(0.4)
