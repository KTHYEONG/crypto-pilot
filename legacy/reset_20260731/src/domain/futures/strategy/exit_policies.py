from __future__ import annotations

from src.domain.futures.strategy.candidate_contracts import SignalExitPolicy


def _policy(
    *,
    policy_id: str,
    archetype: str,
    stop_atr_mult: float,
    take_profit_atr_mult: float,
    expected_holding_bars: int,
    min_holding_bars: int,
    description: str,
) -> SignalExitPolicy:
    expected = max(1, int(expected_holding_bars))
    minimum = min(expected, max(1, int(min_holding_bars)))
    if stop_atr_mult <= 0.0 or take_profit_atr_mult <= 0.0:
        raise ValueError("exit policy ATR multipliers must be positive")
    return SignalExitPolicy(
        policy_id=policy_id,
        archetype=archetype,  # type: ignore[arg-type]
        stop_atr_mult=float(stop_atr_mult),
        take_profit_atr_mult=float(take_profit_atr_mult),
        expected_holding_bars=expected,
        min_holding_bars=minimum,
        description=description,
    )


def build_exit_policies_for_panel(
    *,
    archetype: str,
    regime_name: str,
    base_expected_holding_bars: int,
    base_min_holding_bars: int,
    max_policies: int,
    fallback_stop_atr_mult: float | None = None,
    fallback_take_profit_atr_mult: float | None = None,
) -> tuple[SignalExitPolicy, ...]:
    base_hold = max(1, int(base_expected_holding_bars))
    base_min = min(base_hold, max(1, int(base_min_holding_bars)))
    policies: list[SignalExitPolicy] = []

    if archetype == "trend":
        policies.append(
            _policy(
                policy_id="trend_grind",
                archetype=archetype,
                stop_atr_mult=1.25,
                take_profit_atr_mult=3.50,
                expected_holding_bars=base_hold,
                min_holding_bars=base_min,
                description="Trend continuation in stable regime.",
            )
        )
        if regime_name in {"bull_volatile", "bear_volatile", "crash"}:
            policies.append(
                _policy(
                    policy_id="trend_fast_fail",
                    archetype=archetype,
                    stop_atr_mult=0.90,
                    take_profit_atr_mult=2.25,
                    expected_holding_bars=min(base_hold, 8),
                    min_holding_bars=min(base_min, 3),
                    description="Trend continuation with tighter fail-fast exit.",
                )
            )
    elif archetype == "ts_mom":
        policies.append(
            _policy(
                policy_id="momentum_follow",
                archetype=archetype,
                stop_atr_mult=1.25,
                take_profit_atr_mult=3.00,
                expected_holding_bars=base_hold,
                min_holding_bars=base_min,
                description="Momentum follow-through exit.",
            )
        )
    elif archetype in {"mean_rev", "beta_neut"}:
        policies.append(
            _policy(
                policy_id="snapback",
                archetype=archetype,
                stop_atr_mult=0.90,
                take_profit_atr_mult=1.60,
                expected_holding_bars=min(base_hold, 6),
                min_holding_bars=2,
                description="Fast mean reversion snapback exit.",
            )
        )
    elif archetype in {"flow_rev", "unwind"}:
        policies.append(
            _policy(
                policy_id="flow_exhaustion",
                archetype=archetype,
                stop_atr_mult=0.75,
                take_profit_atr_mult=1.50,
                expected_holding_bars=min(base_hold, 4),
                min_holding_bars=1,
                description="Short-lived flow exhaustion exit.",
            )
        )
    elif archetype == "carry_rev":
        policies.append(
            _policy(
                policy_id="carry_decay",
                archetype=archetype,
                stop_atr_mult=1.50,
                take_profit_atr_mult=2.00,
                expected_holding_bars=min(base_hold, 24),
                min_holding_bars=max(4, min(base_hold, base_min)),
                description="Carry normalization exit.",
            )
        )
    elif archetype == "xs_alpha":
        policies.append(
            _policy(
                policy_id="xs_neutral",
                archetype=archetype,
                stop_atr_mult=1.25,
                take_profit_atr_mult=2.25,
                expected_holding_bars=base_hold,
                min_holding_bars=base_min,
                description="Cross-sectional market-neutral hold.",
            )
        )

    if not policies:
        policies.append(
            _policy(
                policy_id="default",
                archetype="mean_rev",
                stop_atr_mult=(
                    float(fallback_stop_atr_mult)
                    if fallback_stop_atr_mult is not None and fallback_stop_atr_mult > 0.0
                    else 1.00
                ),
                take_profit_atr_mult=(
                    float(fallback_take_profit_atr_mult)
                    if fallback_take_profit_atr_mult is not None and fallback_take_profit_atr_mult > 0.0
                    else 2.00
                ),
                expected_holding_bars=base_hold,
                min_holding_bars=base_min,
                description="Fallback legacy policy.",
            )
        )

    return tuple(policies[: max(1, int(max_policies))])
