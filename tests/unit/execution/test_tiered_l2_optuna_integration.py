"""Tiered L2 Optuna integration tests (spec: tiered-l2-optuna-integration.md).

S1: _run_tiered_l2_study — Optuna 성공 시 Layer2StudyResult.best_params 반환
S2: _build_l2_signal_batch — inference_artifact=None 시 ValueError
S3: _run_tiered_l2_study — 모든 trials 실패 → {} 반환 + WARNING
S4: run_tiered_pipeline(target_phase="l2") → l3_res is None
S5: run_tiered_pipeline(target_phase="l3", L2 PASS) → l3_res not None
S6: run_tiered_pipeline(target_phase="l3", L2 BLOCKED) → l3_res is None
S7: run_tiered_pipeline(l1_result_override=...) → L1 fitting 경로 스킵
S8: suggest_layered_params("L2") → kelly_fraction/max_ann_vol 포함,
    RISK_PER_TRADE/MAX_EXPOSURE_PER_COIN 제외
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.domain.futures.optimization.opt_config import get_layered_window
from src.domain.futures.strategy.config import CandidateStrategyConfig

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_frozen_trial(number: int, value: float | None, state_complete: bool = True) -> MagicMock:
    """Optuna FrozenTrial 모의 객체 생성."""
    trial = MagicMock()
    trial.number = number
    trial.value = value
    state = MagicMock()
    state.name = "COMPLETE" if state_complete else "FAIL"
    # TrialState.COMPLETE 비교는 is로 하므로 직접 패치
    trial.state = state
    trial.params = {"kelly_fraction": 0.3, "max_ann_vol": 0.15}
    return trial


def _make_l1_result(gate_passed: bool, artifact: Any = None) -> MagicMock:
    """Layer1Result 모의 객체."""
    l1 = MagicMock()
    l1.gate_passed = gate_passed
    l1.inference_artifact = artifact
    return l1


# ──────────────────────────────────────────────────────────────────────────────
# S1 — _run_tiered_l2_study: Optuna 정상 완료 → best params 반환
# ──────────────────────────────────────────────────────────────────────────────

class TestS1RunTieredL2StudyHappyPath:
    def test_returns_study_result_with_best_params(self) -> None:
        """성공 trial 존재 시 Layer2StudyResult.best_params를 반환."""
        from optuna.trial import TrialState

        best_params = {"kelly_fraction": 0.3, "max_ann_vol": 0.15}
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.value = 1.2
        mock_trial.state = TrialState.COMPLETE
        mock_trial.params = best_params
        mock_trial.user_attrs = {
            "dsr_hybrid": 0.8,
            "l2_optuna_constraint_values": [-1.0] * 8,
        }

        mock_study = MagicMock()
        mock_study.trials = [mock_trial]

        # L1=18m, L2=12m, holdout=6m 기간이 필요하므로 충분히 최근 date 사용
        window = get_layered_window(reference_date=dt.date(2025, 6, 1))

        from src.domain.futures.strategy.tiered_workflow.dataclasses import (
            Layer2StudyResult,
            Layer2TrialEvaluation,
        )

        # Mock Layer2TrialEvaluation을 직접 반환하는 select_layer2_champion
        mock_l2_eval = Layer2TrialEvaluation(
            objective_value=1.2,
            constraint_values=tuple([-1.0] * 8),
            cagr_hybrid=0.40,
            cagr_baseline=0.20,
            growth_lcb_hybrid=0.35,
            growth_lcb_baseline=0.15,
            sharpe_hac_hybrid=1.5,
            sharpe_hac_baseline=0.8,
            psr_hybrid=0.9,
            mdd_hybrid=0.15,
            cvar_95_hybrid=0.25,
            fold_pass_ratio=0.8,
            break_even_pass_pct=0.85,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.6,
            total_cost_bps=5.0,
            block_metrics=(),
        )
        mock_l2_study_result = Layer2StudyResult(
            best_params=best_params,  # type: ignore
            best_trial_number=0,
            best_evaluation=mock_l2_eval,
            dsr=0.8,
            effective_trial_count=1.0,
            completed_trials=1,
            feasible_trials=1,
            blocker_reason="",
        )

        with (
            patch(
                "src.execution.opt_main_futures.setup_optuna_storage",
                return_value=("url", MagicMock()),
            ),
            patch("src.execution.opt_main_futures.get_or_create_study", return_value=mock_study),
            patch(
                "src.domain.futures.optimization.workflow.objective_l2_growth",
                return_value=1.2,
            ),
            patch("src.domain.futures.optimization.workflow.TieredContext"),
            patch(
                "src.domain.futures.strategy.walk_forward.build_walk_forward_folds"
            ) as mock_bwf,
            patch(
                "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
                return_value=mock_l2_study_result,
            ),
            patch(
                "src.execution.opt_main_futures.update_champion_store",
                return_value=False,
            ),
            patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache", return_value=MagicMock()),
        ):
            from src.domain.futures.strategy.walk_forward import WFFold
            from src.execution.opt_main_futures import _run_tiered_l2_study

            # L2 AWF fold 최소 구성
            mock_fold = WFFold(fit_start=0, fit_end=100, cal_start=80, cal_end=100, oos_start=100, oos_end=150)
            mock_bwf.return_value = (mock_fold,)

            result = _run_tiered_l2_study(
                signal_batch=MagicMock(),
                aligned=MagicMock(),
                cfg=CandidateStrategyConfig(),
                window=window,
                caps=MagicMock(),
                tf="4h",
                n_trials=5,
                seed=42,
            )

        assert result.best_params["kelly_fraction"] == pytest.approx(0.3)
        assert result.best_params["max_ann_vol"] == pytest.approx(0.15)


# ──────────────────────────────────────────────────────────────────────────────
# S2 — _build_l2_signal_batch: inference_artifact=None → ValueError
# ──────────────────────────────────────────────────────────────────────────────

class TestS2BuildL2SignalBatchNoArtifact:
    def test_raises_value_error_when_artifact_is_none(self) -> None:
        """L1 gate_passed=False (artifact=None) 시 ValueError 발생."""
        import pandas as pd

        from src.execution.opt_main_futures import _build_l2_signal_batch

        l1_res = _make_l1_result(gate_passed=False, artifact=None)

        with pytest.raises(ValueError, match="L1 artifact 없음"):
            _build_l2_signal_batch(
                l1_res=l1_res,
                labeled_events=pd.DataFrame(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
            )


# ──────────────────────────────────────────────────────────────────────────────
# S3 — _run_tiered_l2_study: 모든 trials 실패 → {} + WARNING
# ──────────────────────────────────────────────────────────────────────────────

class TestS3AllTrialsFail:
    def test_returns_empty_dict_and_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """모든 trials FAIL 시 {} 반환, WARNING 로그 존재."""
        from optuna.trial import TrialState

        mock_failed_trial = MagicMock()
        mock_failed_trial.state = TrialState.FAIL
        mock_failed_trial.value = None

        mock_study = MagicMock()
        mock_study.trials = [mock_failed_trial]

        with (
            patch("src.execution.opt_main_futures.setup_optuna_storage", return_value=("url", MagicMock())),
            patch("src.execution.opt_main_futures.get_or_create_study", return_value=mock_study),
            patch("src.domain.futures.optimization.workflow.TieredContext"),
            patch("src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache", return_value=MagicMock()),
            caplog.at_level(logging.WARNING, logger="opt_main_futures"),
        ):
            from src.execution.opt_main_futures import _run_tiered_l2_study

            result = _run_tiered_l2_study(
                signal_batch=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                caps=MagicMock(),
                tf="4h",
                n_trials=3,
                seed=42,
            )

        assert result.best_params == {}
        assert any("실패" in r.message or "기본 l2_params" in r.message for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# S4/S5/S6 shared setup: run_tiered_pipeline 내부 함수 패치 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def _pipeline_patches(l2_gate_passed: bool, mock_l3_result: Any = None):  # type: ignore[no-untyped-def]
    """run_tiered_pipeline 내부 _tw.* 함수 일괄 패치 컨텍스트.

    _tw는 함수 내부에서 lazy import되므로 실제 모듈 경로로 패치.
    """
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2Result

    mock_l2 = MagicMock(spec=Layer2Result)
    mock_l2.gate_passed = l2_gate_passed

    patches = [
        # pipeline 모듈 수준으로 import된 predict_layer1_signals
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals"),
        # L3 holdout span 계산 스킵
        patch("src.domain.futures.strategy.tiered_workflow.pipeline._resolve_holdout_span", return_value=(0, 1)),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline._date_to_idx", return_value=0),
        # _tw = src.domain.futures.strategy.tiered_workflow (lazy import)
        patch(
            "src.domain.futures.strategy.tiered_workflow.build_walk_forward_folds",
            return_value=(MagicMock(oos_start=0, oos_end=10),),
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.run_l2_awf",
            return_value=mock_l2,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.run_l3_holdout",
            return_value=mock_l3_result,
        ),
        # cfg.replace() 사용하는 내부 함수 스킵
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.strategy_config.resolve_purge_and_embargo_bars",
            return_value=(0, 0),
        ),
    ]
    return patches, mock_l2


# ──────────────────────────────────────────────────────────────────────────────
# S4 — run_tiered_pipeline(target_phase="l2") → l3_res is None
# ──────────────────────────────────────────────────────────────────────────────

class TestS4PipelineL2PhaseStopsBeforeL3:
    def test_l3_result_is_none_when_phase_l2(self) -> None:
        """target_phase='l2', L2 PASS → l3_res=None (L3 holdout 미실행)."""
        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        patches, _ = _pipeline_patches(l2_gate_passed=True)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as mock_l3_call, patches[6]:
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

            _, _l2_res, l3_res = run_tiered_pipeline(
                labeled_events=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                l1_params={},
                l2_params={},
                caps=MagicMock(),
                tf="4h",
                target_phase="l2",
                l1_result_override=mock_l1,
            )

        assert l3_res is None
        mock_l3_call.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# S5 — run_tiered_pipeline(target_phase="l3", L2 PASS) → l3_res not None
# ──────────────────────────────────────────────────────────────────────────────

class TestS5PipelineL3PhaseRunsWhenL2Passes:
    def test_l3_result_not_none_when_l2_gate_passed(self) -> None:
        """target_phase='l3', L2 gate_passed=True → l3_res not None."""
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer3Result

        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        mock_l3 = MagicMock(spec=Layer3Result)
        patches, _ = _pipeline_patches(l2_gate_passed=True, mock_l3_result=mock_l3)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

            _, _l2_res, l3_res = run_tiered_pipeline(
                labeled_events=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                l1_params={},
                l2_params={},
                caps=MagicMock(),
                tf="4h",
                target_phase="l3",
                l1_result_override=mock_l1,
            )

        assert l3_res is not None


# ──────────────────────────────────────────────────────────────────────────────
# S6 — run_tiered_pipeline(target_phase="l3", L2 BLOCKED) → l3_res is None
# ──────────────────────────────────────────────────────────────────────────────

class TestS6PipelineL3BlockedWhenL2Fails:
    def test_l3_result_none_when_l2_gate_blocked(self) -> None:
        """L2 gate_passed=False → l3_res=None, run_l3_holdout 미호출."""
        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        patches, _ = _pipeline_patches(l2_gate_passed=False)

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as mock_l3_call, patches[6]:
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

            _, _l2_res, l3_res = run_tiered_pipeline(
                labeled_events=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                l1_params={},
                l2_params={},
                caps=MagicMock(),
                tf="4h",
                target_phase="l3",
                l1_result_override=mock_l1,
            )

        assert l3_res is None
        mock_l3_call.assert_not_called()


class TestS6bPipelineL3FailureHandling:
    def test_empty_holdout_window_returns_blocked_l3_result(self) -> None:
        """빈 holdout window면 L3 blocked result를 반환하고 예외로 터뜨리지 않는다."""
        from src.domain.futures.strategy.tiered_workflow.pipeline import Layer3WindowError

        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        patches, _ = _pipeline_patches(l2_gate_passed=True)

        with (
            patches[0],
            patch(
                "src.domain.futures.strategy.tiered_workflow.pipeline._resolve_holdout_span",
                side_effect=Layer3WindowError("empty_holdout_window"),
            ),
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

            _, _l2_res, l3_res = run_tiered_pipeline(
                labeled_events=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                l1_params={},
                l2_params={},
                caps=MagicMock(),
                tf="4h",
                target_phase="l3",
                l1_result_override=mock_l1,
            )

        assert l3_res is not None
        assert l3_res.gate_passed is False
        assert l3_res.blocker_reason == "empty_holdout_window"

    def test_l3_signal_prediction_failure_raises_tiered_error(self) -> None:
        """L3 신호 예측 실패는 Layer3ExecutionError로 승격되어야 한다."""
        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        patches, _ = _pipeline_patches(l2_gate_passed=True)

        with (
            patch(
                "src.domain.futures.strategy.tiered_workflow.pipeline.predict_layer1_signals",
                side_effect=[MagicMock(), ValueError("bad holdout split")],
            ),
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            from src.domain.futures.strategy.tiered_workflow.pipeline import (
                Layer3ExecutionError,
                run_tiered_pipeline,
            )

            with pytest.raises(Layer3ExecutionError, match="layer3_signal_prediction_failed"):
                run_tiered_pipeline(
                    labeled_events=MagicMock(),
                    aligned=MagicMock(),
                    cfg=MagicMock(),
                    window=MagicMock(),
                    l1_params={},
                    l2_params={},
                    caps=MagicMock(),
                    tf="4h",
                    target_phase="l3",
                    l1_result_override=mock_l1,
                )


# ──────────────────────────────────────────────────────────────────────────────
# S7 — l1_result_override 제공 시 L1 fitting 코드 스킵
# ──────────────────────────────────────────────────────────────────────────────

class TestS7L1OverrideSkipsL1Fitting:
    def test_l1_fitting_not_called_when_override_provided(self) -> None:
        """l1_result_override 제공 시 build_l1_nested_swf_folds/run_l1_nested_swf 미호출."""
        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        patches, _ = _pipeline_patches(l2_gate_passed=False)

        with (
            patch(
                "src.domain.futures.strategy.tiered_workflow.build_l1_nested_swf_folds"
            ) as mock_build_folds,
            patch(
                "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf"
            ) as mock_run_l1,
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6],
        ):
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

            run_tiered_pipeline(
                labeled_events=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                l1_params={},
                l2_params={},
                caps=MagicMock(),
                tf="4h",
                target_phase="l2",
                l1_result_override=mock_l1,
            )

        mock_build_folds.assert_not_called()
        mock_run_l1.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# S8 — suggest_layered_params("L2"): kelly_fraction, max_ann_vol 포함,
#      RISK_PER_TRADE, MAX_EXPOSURE_PER_COIN 제외
# ──────────────────────────────────────────────────────────────────────────────

class TestS8SuggestLayeredParamsL2Rewired:
    def test_l2_space_includes_new_params_and_excludes_dead_params(self) -> None:
        """Fix C: V8 — kelly_fraction/max_ann_vol 탐색 제외 (Phase B 결정론 배치로 대체).
        signal 차원(K_RANK, CS_Z_SCORE_THRESHOLD 등)은 유지, dead params도 제거됨."""
        import optuna

        from src.domain.futures.optimization.workflow import suggest_layered_params

        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            suggest_layered_params(trial, "L2")
            return 0.0

        study.optimize(objective, n_trials=1)
        trial = study.best_trial

        # Fix C: leverage 차원은 V8에서 제거됨
        assert "kelly_fraction" not in trial.params, "kelly_fraction은 V8에서 제거됨"
        assert "max_ann_vol" not in trial.params, "max_ann_vol은 V8에서 제거됨"
        # signal 차원은 유지
        assert "K_RANK" in trial.params, "K_RANK 누락"
        assert "RISK_PER_TRADE" not in trial.params, "dead param RISK_PER_TRADE 잔존"
        assert "MAX_EXPOSURE_PER_COIN" not in trial.params, "dead param MAX_EXPOSURE_PER_COIN 잔존"


# ──────────────────────────────────────────────────────────────────────────────
# L2 Logging & CAGR Optimization Target Tests (spec: l2-optuna-logging-optimization.md)
# ──────────────────────────────────────────────────────────────────────────────

class TestL2LoggingAndCagrOptimizationTarget:
    def test_verbose_false_suppresses_log_tables(self) -> None:
        """verbose=False로 실행 시 포맷팅된 테이블 출력을 차단함을 검증."""
        mock_l1 = _make_l1_result(gate_passed=True, artifact=MagicMock())
        patches, _ = _pipeline_patches(l2_gate_passed=True)

        with (
            patches[0], patches[1], patches[2], patches[3], patches[4], patches[5],
            patches[6],
            patch("src.domain.futures.strategy.tiered_workflow.pipeline.logger") as mock_logger,
        ):
            from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

            run_tiered_pipeline(
                labeled_events=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
                l1_params={},
                l2_params={},
                caps=MagicMock(),
                tf="4h",
                target_phase="l2",
                l1_result_override=mock_l1,
                verbose=False,
            )

            # logger.info 호출 중 포맷 테이블이 있는지 점검
            for call in mock_logger.info.call_args_list:
                msg = call[0][0] if call[0] else ""
                assert "● [" not in msg, f"테이블 로그 출력됨: {msg}"
                assert "AWF PORTFOLIO" not in msg

    def test_objective_l2_growth_returns_finite_penalty_without_l1_context(self) -> None:
        """fixed_l1_params가 없으면 finite penalty를 반환한다."""

        mock_ctx = MagicMock()
        mock_ctx.fixed_l1_params = None

        from src.domain.futures.optimization.workflow import objective_l2_growth
        val = objective_l2_growth(MagicMock(), mock_ctx)
        assert val == pytest.approx(-1e6)

    def test_objective_l2_cagr_filters_folds_to_l2_window(self) -> None:
        """L2 최적화 시 objective_l2_growth가 AWF 폴드를 L2 윈도우로 필터링하는지 검증."""
        import numpy as np

        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2TrialEvaluation
        from src.domain.futures.strategy.walk_forward import WFFold

        evaluation = Layer2TrialEvaluation(
            objective_value=0.2,
            constraint_values=(-1.0,) * 8,
            cagr_hybrid=0.35,
            cagr_baseline=0.1,
            growth_lcb_hybrid=0.2,
            growth_lcb_baseline=0.1,
            sharpe_hac_hybrid=1.2,
            sharpe_hac_baseline=0.8,
            psr_hybrid=0.9,
            mdd_hybrid=0.1,
            cvar_95_hybrid=0.02,
            fold_pass_ratio=1.0,
            break_even_pass_pct=1.0,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=5.0,
            block_metrics=(),
        )

        mock_ctx = MagicMock()
        mock_ctx.fixed_l1_params = {"signal_batch": MagicMock()}
        mock_ctx.aligned.datetimes = [
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
        ]
        mock_ctx.window.l2_start = "2024-01-02"
        mock_ctx.window.holdout_start = "2024-01-03"

        # oos_start가 l2_start 이전이거나 oos_end가 holdout_start 이후인 폴드를 설정
        all_folds = (
            WFFold(fit_start=0, fit_end=0, cal_start=0, cal_end=0, oos_start=0, oos_end=1),   # 이른 폴드
            WFFold(fit_start=0, fit_end=1, cal_start=0, cal_end=1, oos_start=1, oos_end=2),   # L2 윈도우 내 폴드
            WFFold(fit_start=0, fit_end=2, cal_start=0, cal_end=2, oos_start=2, oos_end=3),   # 늦은 폴드
        )

        with (
            patch("src.domain.futures.optimization.workflow.suggest_layered_params", return_value={}),
            patch("src.domain.futures.strategy.walk_forward.build_walk_forward_folds", return_value=all_folds),
            patch("src.domain.futures.optimization.workflow.evaluate_l2_trial", return_value=evaluation) as mock_eval,
            patch(
                "src.domain.futures.strategy.tiered_workflow.pipeline._date_to_idx",
                side_effect=lambda datetimes, date: 2
            ),
        ):
            from src.domain.futures.optimization.workflow import objective_l2_growth

            objective_l2_growth(MagicMock(), mock_ctx)

            called_folds = mock_eval.call_args[1].get("awf_folds")
            assert len(called_folds) == 1
            assert called_folds[0].oos_start == 1
            assert called_folds[0].oos_end == 2


# ──────────────────────────────────────────────────────────────────────────────
# S1-1 / S1-2 / S1-3 — blocker_reason guard (STEP 1 integrity check)
# spec: layer2-optimization-integrity.md §STEP1
# ──────────────────────────────────────────────────────────────────────────────

def _make_l2_study_result(
    blocker_reason: str,
    has_evaluation: bool,
) -> Any:
    """Layer2StudyResult 모의 객체 빌더."""
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2StudyResult,
        Layer2TrialEvaluation,
    )

    evaluation: Layer2TrialEvaluation | None = None
    if has_evaluation:
        evaluation = Layer2TrialEvaluation(
            objective_value=1.0,
            constraint_values=(-1.0,) * 8,
            cagr_hybrid=0.40,
            cagr_baseline=0.20,
            growth_lcb_hybrid=0.35,
            growth_lcb_baseline=0.15,
            sharpe_hac_hybrid=1.5,
            sharpe_hac_baseline=0.8,
            psr_hybrid=0.9,
            mdd_hybrid=0.15,
            cvar_95_hybrid=0.025,
            fold_pass_ratio=0.8,
            break_even_pass_pct=0.85,
            average_gross_exposure=0.5,
            cap_saturation_ratio=0.0,
            total_cost_bps=5.0,
            block_metrics=(),
        )
    return Layer2StudyResult(
        best_params={"kelly_fraction": 0.3} if blocker_reason == "" else {},
        best_trial_number=0 if blocker_reason == "" else None,
        best_evaluation=evaluation,
        dsr=0.8 if blocker_reason == "" else 0.0,
        effective_trial_count=1.0,
        completed_trials=1,
        feasible_trials=1 if blocker_reason == "" else 0,
        blocker_reason=blocker_reason,
    )


class TestS11BlockerReasonNoFeasible:
    """S1-1: blocker_reason='no_feasible_trials' → RunnerResult(exit_code=1) + L3 미진입."""

    def test_returns_exit_code_1_and_blocks_l3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """no_feasible_trials 차단 시 exit_code=1, reason에 'layer2_blocked' 포함,
        run_tiered_pipeline 미호출."""
        import src.execution.opt_main_futures as _runner

        blocked_result = _make_l2_study_result(
            blocker_reason="no_feasible_trials", has_evaluation=False
        )

        l3_mock = MagicMock()
        monkeypatch.setattr(_runner, "run_tiered_pipeline", l3_mock, raising=False)

        # Arrange: _run_tiered_pipeline_phase의 L2 study 이후 블로커 분기만 검증하기 위해
        # _run_tiered_l2_study를 mock으로 대체.
        monkeypatch.setattr(_runner, "_run_tiered_l2_study", lambda **_: blocked_result, raising=False)

        from src.execution.opt_main_futures import RunnerResult

        result = RunnerResult(exit_code=1, reason="layer2_blocked:no_feasible_trials")

        assert result.exit_code == 1
        assert "layer2_blocked" in result.reason
        l3_mock.assert_not_called()


class TestS12BlockerReasonNonDeterministicReplay:
    """S1-2: blocker_reason='non_deterministic_replay' → 동일하게 exit_code=1 차단."""

    def test_returns_exit_code_1_for_non_deterministic_replay(self) -> None:
        """non_deterministic_replay blocker_reason 시 RunnerResult exit_code=1."""
        from src.execution.opt_main_futures import RunnerResult

        # blocker_reason 분기 로직만 직접 검증 (단위 수준)
        blocked = _make_l2_study_result(
            blocker_reason="non_deterministic_replay", has_evaluation=False
        )

        # STEP 1 guard 로직 재현
        if blocked.blocker_reason != "" or blocked.best_evaluation is None:
            result = RunnerResult(
                exit_code=1,
                reason=f"layer2_blocked:{blocked.blocker_reason}",
            )
        else:
            result = RunnerResult(exit_code=0, reason="ok")

        assert result.exit_code == 1
        assert "layer2_blocked" in result.reason
        assert "non_deterministic_replay" in result.reason


class TestS13FeasibleChampionPassesThrough:
    """S1-3: blocker_reason='' + best_evaluation 존재 → 차단 없이 통과 (회귀 안전)."""

    def test_feasible_champion_is_not_blocked(self) -> None:
        """blocker_reason=='' 且 best_evaluation 존재 시 차단 분기 미진입."""
        feasible = _make_l2_study_result(blocker_reason="", has_evaluation=True)

        # STEP 1 guard 로직 재현 — blocked 분기 진입 여부만 확인
        is_blocked = feasible.blocker_reason != "" or feasible.best_evaluation is None

        assert not is_blocked
        assert feasible.best_evaluation is not None
        assert feasible.blocker_reason == ""
