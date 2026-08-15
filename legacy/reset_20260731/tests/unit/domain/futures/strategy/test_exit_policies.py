from __future__ import annotations

from src.domain.futures.strategy.exit_policies import build_exit_policies_for_panel


def test_build_exit_policies_for_trend_volatile_adds_fail_fast() -> None:
    policies = build_exit_policies_for_panel(
        archetype="trend",
        regime_name="bull_volatile",
        base_expected_holding_bars=12,
        base_min_holding_bars=4,
        max_policies=2,
    )

    assert [policy.policy_id for policy in policies] == ["trend_grind", "trend_fast_fail"]
    assert policies[1].expected_holding_bars <= 8
    assert policies[1].min_holding_bars <= 3


def test_build_exit_policies_for_mean_rev_uses_snapback() -> None:
    policies = build_exit_policies_for_panel(
        archetype="mean_rev",
        regime_name="transition",
        base_expected_holding_bars=10,
        base_min_holding_bars=3,
        max_policies=2,
    )

    assert len(policies) == 1
    assert policies[0].policy_id == "snapback"
    assert policies[0].stop_atr_mult == 0.90
    assert policies[0].take_profit_atr_mult == 1.60


def test_build_exit_policies_fallback_uses_panel_scalars() -> None:
    policies = build_exit_policies_for_panel(
        archetype="unknown_archetype",
        regime_name="transition",
        base_expected_holding_bars=10,
        base_min_holding_bars=3,
        max_policies=1,
        fallback_stop_atr_mult=2.4,
        fallback_take_profit_atr_mult=4.8,
    )

    assert len(policies) == 1
    assert policies[0].policy_id == "legacy"
    assert policies[0].stop_atr_mult == 2.4
    assert policies[0].take_profit_atr_mult == 4.8
