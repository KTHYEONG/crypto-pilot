"""Spec: futures-l2-reversal-economic-replay, Scenario 4-8."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from src.execution import opt_main_futures


def _attr(
    *,
    fold_idx: int,
    risk_off_bars: int,
    risk_off_price: float,
    risk_on_price: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        fold_idx=fold_idx,
        risk_off_bars=risk_off_bars,
        risk_off_realized_price=risk_off_price,
        risk_on_realized_price=risk_on_price,
    )


def _eval(
    *,
    cagr: float,
    mdd: float,
    fold_cagrs: tuple[float | None, ...],
    fold_mdds: tuple[float | None, ...],
    risk_off: tuple[int, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        cagr_hybrid=cagr,
        mdd_hybrid=mdd,
        trade_count=10,
        deploy_leverage=4.0,
        fold_deployed_cagrs=fold_cagrs,
        fold_deployed_mdds=fold_mdds,
        fold_attributions=tuple(
            _attr(
                fold_idx=i,
                risk_off_bars=value,
                risk_off_price=0.01 * value,
                risk_on_price=0.02,
            )
            for i, value in enumerate(risk_off)
        ),
        last_selected_symbols=("BTCUSDT", "ETHUSDT"),
    )


class TestFoldMetricsFromL2Evaluation:
    """Scenario 4: Fold metric extraction handles missing fields."""

    def test_fold_metrics_from_l2_evaluation_handles_missing_fold_fields(self) -> None:
        evaluation = _eval(
            cagr=0.1, mdd=0.2, fold_cagrs=(0.1, None), fold_mdds=(0.05,), risk_off=(3,),
        )
        metrics = opt_main_futures._fold_metrics_from_l2_evaluation(
            variant_name="test", evaluation=evaluation,
        )
        assert len(metrics) == 2
        assert metrics[0].cagr == 0.1
        assert metrics[0].mdd == 0.05
        assert metrics[0].risk_off_bars == 3
        assert metrics[1].cagr is None
        assert metrics[1].mdd is None
        assert metrics[1].risk_off_bars == 0


class TestAdoptionVerdict:
    """Scenario 5-6: Adoption verdict logic."""

    def _make_result(
        self,
        *,
        variant: str,
        cagr: float,
        fold_metrics: tuple,
        selection_parity: bool = True,
    ) -> opt_main_futures.L2ReversalReplayResult:
        return opt_main_futures.L2ReversalReplayResult(
            variant=variant,
            cagr=cagr,
            mdd=0.3,
            trade_count=10,
            deploy_leverage=4.0,
            selection_parity=selection_parity,
            metric_parity=True,
            fold_metrics=fold_metrics,
            adoption_passed=False,
            blocker_reason="",
        )

    def _fold_metric(self, variant: str, idx: int, cagr: float | None) -> opt_main_futures.L2ReversalReplayFoldMetric:
        return opt_main_futures.L2ReversalReplayFoldMetric(
            variant=variant,
            fold_idx=idx,
            cagr=cagr,
            mdd=0.05,
            risk_off_bars=5 if idx == 0 else 0,
            risk_off_realized_price=0.05,
            risk_on_realized_price=0.02,
        )

    def test_reversal_replay_adoption_verdict_accepts_balanced_candidate(self) -> None:
        baseline = self._make_result(
            variant="baseline_off", cagr=0.015,
            fold_metrics=(
                self._fold_metric("baseline_off", 0, -0.274),
                self._fold_metric("baseline_off", 1, 0.05),
                self._fold_metric("baseline_off", 2, 0.04),
            ),
        )
        legacy = self._make_result(
            variant="legacy_006_p1", cagr=0.018,
            fold_metrics=(
                self._fold_metric("legacy_006_p1", 0, -0.216),
                self._fold_metric("legacy_006_p1", 1, -0.005),
                self._fold_metric("legacy_006_p1", 2, 0.017),
            ),
        )
        candidate = self._make_result(
            variant="balanced_010_p2", cagr=0.025,
            fold_metrics=(
                self._fold_metric("balanced_010_p2", 0, -0.230),
                self._fold_metric("balanced_010_p2", 1, 0.045),
                self._fold_metric("balanced_010_p2", 2, 0.035),
            ),
        )
        passed, reason = opt_main_futures._reversal_replay_adoption_verdict(
            baseline=baseline,
            legacy=legacy,
            candidate=candidate,
        )
        assert passed is True
        assert reason == ""

    def test_reversal_replay_adoption_verdict_blocks_non_bottleneck_damage(self) -> None:
        baseline = self._make_result(
            variant="baseline_off", cagr=0.015,
            fold_metrics=(
                self._fold_metric("baseline_off", 0, -0.274),
                self._fold_metric("baseline_off", 1, 0.05),
                self._fold_metric("baseline_off", 2, 0.04),
            ),
        )
        legacy = self._make_result(
            variant="legacy_006_p1", cagr=0.018,
            fold_metrics=(
                self._fold_metric("legacy_006_p1", 0, -0.216),
                self._fold_metric("legacy_006_p1", 1, -0.005),
                self._fold_metric("legacy_006_p1", 2, 0.017),
            ),
        )
        candidate = self._make_result(
            variant="balanced_010_p2", cagr=0.025,
            fold_metrics=(
                self._fold_metric("balanced_010_p2", 0, -0.230),
                self._fold_metric("balanced_010_p2", 1, 0.03),
                self._fold_metric("balanced_010_p2", 2, 0.035),
            ),
        )
        passed, reason = opt_main_futures._reversal_replay_adoption_verdict(
            baseline=baseline,
            legacy=legacy,
            candidate=candidate,
        )
        assert passed is False
        assert reason == "non_bottleneck_damage"


class TestTemporaryReversalEnv:
    """Scenario 7 helper: env scope test."""

    def test_temporary_reversal_env_restores_original(self) -> None:
        os.environ["L2_REVERSAL_KILL"] = "original"
        os.environ["L2_REVERSAL_DD_THRESHOLD"] = "0.99"
        os.environ["L2_REVERSAL_PERSISTENCE_BARS"] = "5"
        variant = opt_main_futures.L2ReversalReplayVariant(
            name="test", enabled=True, dd_threshold=0.10, persistence_bars=2,
        )
        with opt_main_futures._temporary_reversal_env(variant):
            assert os.environ.get("L2_REVERSAL_KILL") == "1"
            assert os.environ.get("L2_REVERSAL_DD_THRESHOLD") == "0.1"
            assert os.environ.get("L2_REVERSAL_PERSISTENCE_BARS") == "2"
        assert os.environ.get("L2_REVERSAL_KILL") == "original"
        assert os.environ.get("L2_REVERSAL_DD_THRESHOLD") == "0.99"
        assert os.environ.get("L2_REVERSAL_PERSISTENCE_BARS") == "5"
        del os.environ["L2_REVERSAL_KILL"]
        del os.environ["L2_REVERSAL_DD_THRESHOLD"]
        del os.environ["L2_REVERSAL_PERSISTENCE_BARS"]


class TestRunL2ReversalEconomicReplay:
    """Scenario 7-8: Replay integration."""

    @patch("src.domain.futures.optimization.workflow.evaluate_l2_trial")
    @patch("src.domain.futures.strategy.tiered_workflow.replay_parity.assert_selection_replay_parity")
    def test_run_l2_reversal_economic_replay_scopes_env_and_checks_metric_parity(
        self,
        mock_parity: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        captured_envs: list[dict[str, str | None]] = []

        def _side_effect(**kwargs: Any) -> SimpleNamespace:
            captured_envs.append({
                "L2_REVERSAL_KILL": os.environ.get("L2_REVERSAL_KILL"),
                "L2_REVERSAL_DD_THRESHOLD": os.environ.get("L2_REVERSAL_DD_THRESHOLD"),
                "L2_REVERSAL_PERSISTENCE_BARS": os.environ.get("L2_REVERSAL_PERSISTENCE_BARS"),
            })
            return SimpleNamespace(
                cagr_hybrid=0.02,
                mdd_hybrid=0.15,
                trade_count=10,
                deploy_leverage=4.0,
                fold_deployed_cagrs=(0.01,),
                fold_deployed_mdds=(0.1,),
                fold_attributions=(),
                last_selected_symbols=("BTCUSDT", "ETHUSDT"),
            )

        mock_eval.side_effect = _side_effect
        mock_parity.return_value = True

        results = opt_main_futures._run_l2_reversal_economic_replay(
            signal_batch=MagicMock(),
            aligned=SimpleNamespace(symbols=("BTCUSDT",), close_2d=MagicMock()),
            awf_folds=(MagicMock(), MagicMock()),
            base_l2_params={"k_rank": 3},
            caps=MagicMock(),
            tf="1h",
            deploy_leverage=4.0,
            prebuilt_cache=MagicMock(),
            reference_evaluation=SimpleNamespace(
                deploy_leverage=4.0,
                last_selected_symbols=("BTCUSDT", "ETHUSDT"),
            ),
        )
        assert len(results) == 5
        assert len(captured_envs) == 5
        first_call = captured_envs[0]
        variant_calls = captured_envs[1:]
        assert first_call["L2_REVERSAL_KILL"] is None or first_call["L2_REVERSAL_KILL"] == ""
        thresholds = [c["L2_REVERSAL_DD_THRESHOLD"] for c in variant_calls]
        assert thresholds == ["0.06", "0.1", "0.1", "0.12"]
        baseline_result = results[0]
        assert baseline_result.metric_parity is True
        for variant_result in results[1:]:
            assert variant_result.metric_parity is False
        baseline = results[0]
        assert baseline.variant == "baseline_off"
        assert baseline.blocker_reason == "baseline"
        assert baseline.adoption_passed is False
        assert os.environ.get("L2_REVERSAL_KILL") is None
