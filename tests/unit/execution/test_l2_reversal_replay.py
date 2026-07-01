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


def _make_result(
    *,
    variant: str,
    cagr: float,
    fold_metrics: tuple[opt_main_futures.L2ReversalReplayFoldMetric, ...],
    selection_parity: bool = True,
    dd_threshold: float = 0.06,
    persistence_bars: int = 1,
    recovery_cooldown_bars: int = 0,
) -> opt_main_futures.L2ReversalReplayResult:
    return opt_main_futures.L2ReversalReplayResult(
        variant=variant,
        dd_threshold=dd_threshold,
        persistence_bars=persistence_bars,
        recovery_cooldown_bars=recovery_cooldown_bars,
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


def _fold_metric(variant: str, idx: int, cagr: float | None) -> opt_main_futures.L2ReversalReplayFoldMetric:
    return opt_main_futures.L2ReversalReplayFoldMetric(
        variant=variant,
        fold_idx=idx,
        cagr=cagr,
        mdd=0.05,
        risk_off_bars=5 if idx == 0 else 0,
        risk_off_realized_price=0.05,
        risk_on_realized_price=0.02,
    )


class TestAdoptionVerdict:
    """Scenario 5-6: Adoption verdict logic."""

    def test_reversal_replay_adoption_verdict_accepts_balanced_candidate(self) -> None:
        baseline = _make_result(
            variant="baseline_off", cagr=0.015,
            fold_metrics=(
                _fold_metric("baseline_off", 0, -0.274),
                _fold_metric("baseline_off", 1, 0.05),
                _fold_metric("baseline_off", 2, 0.04),
            ),
        )
        legacy = _make_result(
            variant="legacy_006_p1", cagr=0.018,
            fold_metrics=(
                _fold_metric("legacy_006_p1", 0, -0.216),
                _fold_metric("legacy_006_p1", 1, -0.005),
                _fold_metric("legacy_006_p1", 2, 0.017),
            ),
        )
        candidate = _make_result(
            variant="balanced_010_p2", cagr=0.025,
            fold_metrics=(
                _fold_metric("balanced_010_p2", 0, -0.230),
                _fold_metric("balanced_010_p2", 1, 0.045),
                _fold_metric("balanced_010_p2", 2, 0.035),
            ),
        )
        passed, reason = opt_main_futures._reversal_replay_adoption_verdict(
            baseline=baseline,
            legacy=legacy,
            candidate=candidate,
        )
        assert passed is True
        assert reason == "adopted_no_stress_in_window"

    def test_reversal_replay_adoption_verdict_blocks_non_stress_damage(self) -> None:
        baseline = _make_result(
            variant="baseline_off", cagr=0.015,
            fold_metrics=(
                _fold_metric("baseline_off", 0, -0.274),
                _fold_metric("baseline_off", 1, 0.05),
                _fold_metric("baseline_off", 2, 0.04),
            ),
        )
        legacy = _make_result(
            variant="legacy_006_p1", cagr=0.018,
            fold_metrics=(
                _fold_metric("legacy_006_p1", 0, -0.216),
                _fold_metric("legacy_006_p1", 1, -0.005),
                _fold_metric("legacy_006_p1", 2, 0.017),
            ),
        )
        candidate = _make_result(
            variant="balanced_010_p2", cagr=0.025,
            fold_metrics=(
                _fold_metric("balanced_010_p2", 0, -0.230),
                _fold_metric("balanced_010_p2", 1, 0.03),
                _fold_metric("balanced_010_p2", 2, 0.035),
            ),
        )
        passed, reason = opt_main_futures._reversal_replay_adoption_verdict(
            baseline=baseline,
            legacy=legacy,
            candidate=candidate,
        )
        assert passed is False
        assert reason == "non_stress_damage"


class TestTemporaryReversalEnv:
    """Scenario 7 helper: env scope test."""

    def test_temporary_reversal_env_restores_original(self) -> None:
        os.environ["L2_REVERSAL_KILL"] = "original"
        os.environ["L2_REVERSAL_DD_THRESHOLD"] = "0.99"
        os.environ["L2_REVERSAL_PERSISTENCE_BARS"] = "5"
        os.environ["L2_REVERSAL_RECOVERY_COOLDOWN"] = "99"
        variant = opt_main_futures.L2ReversalReplayVariant(
            name="test", enabled=True, dd_threshold=0.10, persistence_bars=2, recovery_cooldown_bars=4,
        )
        with opt_main_futures._temporary_reversal_env(variant):
            assert os.environ.get("L2_REVERSAL_KILL") == "1"
            assert os.environ.get("L2_REVERSAL_DD_THRESHOLD") == "0.1"
            assert os.environ.get("L2_REVERSAL_PERSISTENCE_BARS") == "2"
            assert os.environ.get("L2_REVERSAL_RECOVERY_COOLDOWN") == "4"
        assert os.environ.get("L2_REVERSAL_KILL") == "original"
        assert os.environ.get("L2_REVERSAL_DD_THRESHOLD") == "0.99"
        assert os.environ.get("L2_REVERSAL_PERSISTENCE_BARS") == "5"
        assert os.environ.get("L2_REVERSAL_RECOVERY_COOLDOWN") == "99"
        del os.environ["L2_REVERSAL_KILL"]
        del os.environ["L2_REVERSAL_DD_THRESHOLD"]
        del os.environ["L2_REVERSAL_PERSISTENCE_BARS"]
        del os.environ["L2_REVERSAL_RECOVERY_COOLDOWN"]


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
                "L2_REVERSAL_RECOVERY_COOLDOWN": os.environ.get("L2_REVERSAL_RECOVERY_COOLDOWN"),
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
        assert len(results) == 8
        assert len(captured_envs) == 8
        first_call = captured_envs[0]
        variant_calls = captured_envs[1:]
        assert first_call["L2_REVERSAL_KILL"] is None or first_call["L2_REVERSAL_KILL"] == ""
        thresholds = [c["L2_REVERSAL_DD_THRESHOLD"] for c in variant_calls]
        assert thresholds == ["0.06", "0.1", "0.1", "0.12", "0.06", "0.06", "0.12"]
        cooldowns = [c["L2_REVERSAL_RECOVERY_COOLDOWN"] for c in variant_calls]
        assert cooldowns == ["0", "0", "0", "0", "4", "8", "8"]
        baseline_result = results[0]
        assert baseline_result.metric_parity is True
        for variant_result in results[1:]:
            assert variant_result.metric_parity is False
        baseline = results[0]
        assert baseline.variant == "baseline_off"
        assert baseline.blocker_reason == "baseline"
        assert baseline.adoption_passed is False
        assert os.environ.get("L2_REVERSAL_KILL") is None


# ── L2 Reversal Recovery Cooldown (Scenario 5, 9) ──────────────


class TestReversalCooldownScenarios:
    """Scenarios 5, 9 for recovery cooldown spec."""

    def test_l2_reversal_replay_variants_default_cooldown_zero_backward_compat(self) -> None:
        """Scenario 5: existing variants have recovery_cooldown_bars=0 default."""
        variants = opt_main_futures._l2_reversal_replay_variants()
        legacy_names = {"baseline_off", "legacy_006_p1", "balanced_010_p2", "balanced_010_p3", "current_012_p3"}
        for v in variants:
            if v.name in legacy_names:
                assert v.recovery_cooldown_bars == 0, f"{v.name}: cooldown must be 0"

    def test_reversal_replay_adoption_verdict_skips_stress_check_when_no_fold_exceeds_mdd_threshold(
        self,
    ) -> None:
        """Scenario 9: no-stress window automatically skips stress checks."""
        baseline = _make_result(
            variant="baseline_off", cagr=0.015,
            fold_metrics=(
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="baseline_off", fold_idx=0, cagr=0.01, mdd=0.1152,
                    risk_off_bars=0, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="baseline_off", fold_idx=1, cagr=0.02, mdd=0.1748,
                    risk_off_bars=0, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="baseline_off", fold_idx=2, cagr=0.025, mdd=0.1336,
                    risk_off_bars=0, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
            ),
        )
        legacy = _make_result(
            variant="legacy_006_p1", cagr=0.018,
            fold_metrics=(
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="legacy_006_p1", fold_idx=0, cagr=0.005, mdd=0.1152,
                    risk_off_bars=10, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="legacy_006_p1", fold_idx=1, cagr=0.025, mdd=0.1748,
                    risk_off_bars=5, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="legacy_006_p1", fold_idx=2, cagr=0.03, mdd=0.1336,
                    risk_off_bars=0, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
            ),
        )
        candidate = _make_result(
            variant="balanced_010_p2", cagr=0.022,
            fold_metrics=(
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="balanced_010_p2", fold_idx=0, cagr=0.008, mdd=0.1152,
                    risk_off_bars=12, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="balanced_010_p2", fold_idx=1, cagr=0.028, mdd=0.1748,
                    risk_off_bars=8, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
                opt_main_futures.L2ReversalReplayFoldMetric(
                    variant="balanced_010_p2", fold_idx=2, cagr=0.032, mdd=0.1336,
                    risk_off_bars=0, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
                ),
            ),
        )
        _, reason = opt_main_futures._reversal_replay_adoption_verdict(
            baseline=baseline,
            legacy=legacy,
            candidate=candidate,
            stress_mdd_threshold=0.15,
        )
        assert reason != "legacy_no_improvement", (
            "no-stress window must not spuriously block with legacy_no_improvement"
        )
