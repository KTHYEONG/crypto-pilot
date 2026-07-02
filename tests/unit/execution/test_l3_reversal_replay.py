"""Spec: l3-holdout-reversal-kill-attribution-replay, Scenario P2-S5~S6."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    ReversalEpisode,
    _extract_risk_off_episodes,
)


def test_run_l3_reversal_economic_replay_scopes_env_per_variant() -> None:
    """P2-S5: replay 하네스가 8개 variant 각각에 올바른 env를 스코핑한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_reversal_economic_replay

    captured_envs: list[dict[str, str | None]] = []

    def _side_effect(**kwargs: Any) -> SimpleNamespace:
        captured_envs.append({
            "L2_REVERSAL_KILL": os.environ.get("L2_REVERSAL_KILL"),
            "L2_REVERSAL_DD_THRESHOLD": os.environ.get("L2_REVERSAL_DD_THRESHOLD"),
            "L2_REVERSAL_PERSISTENCE_BARS": os.environ.get("L2_REVERSAL_PERSISTENCE_BARS"),
            "L2_REVERSAL_RECOVERY_COOLDOWN": os.environ.get("L2_REVERSAL_RECOVERY_COOLDOWN"),
        })
        return SimpleNamespace(
            cagr=0.02,
            mdd=0.15,
            sharpe=0.5,
            mar=0.02 / 0.15,
            cagr_baseline=0.01,
            mdd_baseline=0.12,
            sharpe_baseline=0.3,
            mar_baseline=0.01 / 0.12,
            gate_passed=True,
            blocker_reason="",
            total_return=0.02,
            equity_multiple=1.02,
            sortino=0.6,
            sortino_baseline=0.4,
            n_trades=20,
            cvar95=0.03,
            avg_gross_exposure=0.5,
            deploy_leverage=1.0,
            min_trades=10,
            max_mdd_abs=0.35,
            min_sharpe=0.0,
            min_sortino=0.0,
            max_cvar95=0.06,
            risk_off_bars=0,
            risk_off_realized_price=0.0,
            risk_on_realized_price=0.0,
            reversal_kill_active=False,
            risk_off_episodes=(),
        )

    with patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        side_effect=_side_effect,
    ):
        results = run_l3_reversal_economic_replay(
            signal_batch=MagicMock(),
            aligned=MagicMock(),
            holdout_span=(0, 40),
            config=MagicMock(),
            caps=MagicMock(),
            tf="4h",
            deploy_leverage=None,
        )

    assert len(results) == 8
    assert len(captured_envs) == 8

    first_call = captured_envs[0]
    variant_calls = captured_envs[1:]

    assert first_call["L2_REVERSAL_KILL"] is None or first_call["L2_REVERSAL_KILL"] == ""

    thresholds = [c["L2_REVERSAL_DD_THRESHOLD"] for c in variant_calls]
    assert thresholds == ["0.06", "0.1", "0.1", "0.12", "0.06", "0.06", "0.12"]

    persistence = [c["L2_REVERSAL_PERSISTENCE_BARS"] for c in variant_calls]
    assert persistence == ["1", "2", "3", "3", "1", "1", "3"]

    cooldowns = [c["L2_REVERSAL_RECOVERY_COOLDOWN"] for c in variant_calls]
    assert cooldowns == ["0", "0", "0", "0", "4", "8", "8"]

    baseline = results[0]
    assert baseline.variant == "baseline_off"


def test_run_l3_reversal_economic_replay_restores_env_after_completion() -> None:
    """P2-S6: 함수 종료 후 env가 원상복구된다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l3_reversal_economic_replay

    pre_env = os.environ.get("L2_REVERSAL_KILL")

    with patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        return_value=SimpleNamespace(
            cagr=0.02, mdd=0.15, sharpe=0.5, mar=0.02 / 0.15,
            cagr_baseline=0.01, mdd_baseline=0.12, sharpe_baseline=0.3, mar_baseline=0.01 / 0.12,
            gate_passed=True, blocker_reason="",
            total_return=0.02, equity_multiple=1.02, sortino=0.6, sortino_baseline=0.4,
            n_trades=20, cvar95=0.03, avg_gross_exposure=0.5, deploy_leverage=1.0,
            min_trades=10, max_mdd_abs=0.35, min_sharpe=0.0, min_sortino=0.0, max_cvar95=0.06,
            risk_off_bars=0, risk_off_realized_price=0.0, risk_on_realized_price=0.0,
            reversal_kill_active=False, risk_off_episodes=(),
        ),
    ):
        run_l3_reversal_economic_replay(
            signal_batch=MagicMock(),
            aligned=MagicMock(),
            holdout_span=(0, 40),
            config=MagicMock(),
            caps=MagicMock(),
            tf="4h",
            deploy_leverage=None,
        )

    post_env = os.environ.get("L2_REVERSAL_KILL")
    assert post_env == pre_env


def test_fold_attribution_extracts_risk_off_episodes_from_contiguous_runs() -> None:
    """[F,F,T,T,T,F,F,T,F] should yield episodes at (2,5) and (7,8)."""
    mask = np.array([False, False, True, True, True, False, False, True, False], dtype=bool)
    prices = np.array([0.0, 0.0, 0.01, 0.02, 0.03, 0.0, 0.0, 0.04, 0.0], dtype=np.float64)

    episodes = _extract_risk_off_episodes(mask, prices)

    assert len(episodes) == 2
    assert episodes[0] == ReversalEpisode(start_idx=2, end_idx=5, realized_price=0.06)
    assert episodes[1] == ReversalEpisode(start_idx=7, end_idx=8, realized_price=0.04)


def test_fold_attribution_closes_open_episode_at_fold_end() -> None:
    """[F,T,T,T] should close open episode at end with end_idx == len(mask)."""
    mask = np.array([False, True, True, True], dtype=bool)
    prices = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)

    episodes = _extract_risk_off_episodes(mask, prices)

    assert len(episodes) == 1
    assert episodes[0] == ReversalEpisode(start_idx=1, end_idx=4, realized_price=0.06)
