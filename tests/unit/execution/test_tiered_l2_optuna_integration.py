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

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
        mock_trial.user_attrs = {"dsr_hybrid": 0.8, "l2_constraint_values": [-1.0] * 9}

        mock_study = MagicMock()
        mock_study.trials = [mock_trial]

        with (
            patch("src.execution.opt_main_futures.setup_optuna_storage", return_value=("url", MagicMock())),
            patch("optuna.create_study", return_value=mock_study),
            patch("src.domain.futures.optimization.workflow.objective_l2_growth", return_value=1.2),
            patch("src.domain.futures.optimization.workflow.TieredContext"),
        ):
            from src.execution.opt_main_futures import _run_tiered_l2_study

            result = _run_tiered_l2_study(
                signal_batch=MagicMock(),
                aligned=MagicMock(),
                cfg=MagicMock(),
                window=MagicMock(),
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
            patch("optuna.create_study", return_value=mock_study),
            patch("src.domain.futures.optimization.workflow.TieredContext"),
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
        """suggest_layered_params('L2')가 kelly_fraction/max_ann_vol를 포함
        하고 RISK_PER_TRADE/MAX_EXPOSURE_PER_COIN을 제외함."""
        import optuna

        from src.domain.futures.optimization.workflow import suggest_layered_params

        study = optuna.create_study(direction="maximize")

        def objective(trial: optuna.Trial) -> float:
            suggest_layered_params(trial, "L2")
            return 0.0

        study.optimize(objective, n_trials=1)
        trial = study.best_trial

        assert "kelly_fraction" in trial.params, "kelly_fraction 누락"
        assert "max_ann_vol" in trial.params, "max_ann_vol 누락"
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
            constraint_values=(-1.0,) * 9,
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
