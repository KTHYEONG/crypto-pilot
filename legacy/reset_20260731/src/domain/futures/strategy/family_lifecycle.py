"""Signal family x timeframe retirement registry.

[ADR_20260706_L0_SIGNAL_FAMILY_DIVERSITY][ADR_20260710_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN]
"""

from __future__ import annotations

FAMILY_TF_RETIREMENT: frozenset[tuple[str, str]] = frozenset(
    {
        ("residual_reversion", "4h"),
        ("funding_extreme_reversal", "4h"),
        # Removed from ALL_SIGNAL_FAMILIES/candidate_families entirely (2026-07-07):
        # rank_ic indistinguishable from noise at 4h, real-run confirmed
        # ([ADR_20260707_L0_ALPHA_EFFECTIVENESS_REDESIGN] follow-up).
        ("flow_exhaustion_reversal", "4h"),
        ("funding_carry", "4h"),
        ("funding_flow_unwind", "4h"),
        ("funding_term_structure_carry", "4h"),
        ("positioning_unwind", "4h"),
        ("xs_carry", "4h"),
        ("flow_trend_continuation", "4h"),
        # --- l0_signal_yield_improvement (run 4h_1783474978) ---
        ("ichimoku_trend", "12h"),  # cost_drag=20.53, gross≈0.75bps — 방향성 정보 사실상 0
        ("vol_breakout", "4h"),  # gross=-27.70bps — 비용 이전에 gross 자체가 확정적으로 음수
        ("residual_momentum_xs", "4h"),  # cost_drag=3.77, nw_tstat=-9.58 — 강한 확신의 음의 엣지
        ("xs_residual_rebalance", "4h"),  # cost_drag=5.14, nw_tstat=-6.40 — 상동
        # --- l0_signal_breadth_diversity_redesign (Fix 3 — LIMIT-11) ---
        ("liquidity_vacuum_breakout", "1h"),
        ("liquidity_vacuum_breakout", "2h"),
        ("liquidity_vacuum_breakout", "4h"),
        ("liquidity_vacuum_breakout", "6h"),
        ("liquidity_vacuum_breakout", "8h"),
        ("liquidity_vacuum_breakout", "12h"),
        # [ADR_20260710_L0_ENTRY_EXIT_SIGNAL_EFFECTIVENESS_REDESIGN] Fix 4 [LIMIT-14]
        ("sparse_breakout_retest_v2", "1h"),
        ("sparse_breakout_retest_v2", "2h"),
        ("sparse_breakout_retest_v2", "4h"),
        ("sparse_breakout_retest_v2", "6h"),
        ("sparse_breakout_retest_v2", "8h"),
        ("sparse_breakout_retest_v2", "12h"),
        ("sparse_breakout_retest_liquidity", "1h"),
        ("sparse_breakout_retest_liquidity", "2h"),
        ("sparse_breakout_retest_liquidity", "4h"),
        ("sparse_breakout_retest_liquidity", "6h"),
        ("sparse_breakout_retest_liquidity", "8h"),
        ("sparse_breakout_retest_liquidity", "12h"),
    }
)


RETIRED_FAMILIES: frozenset[str] = frozenset(
    {
        # [ADR_20260713_L0_L1_ASSET_GROWTH_RESTRUCTURE] durable-zero: 0% gate pass rate
        # across all evaluated TFs (run 4h_1783923826, pooled n=298), consistent with prior
        # economic-replay rejection cycles. Removed from candidate_families/_DEFAULT_PER_TF_FAMILIES
        # (all TFs) but signal-generation code in signals/rules.py / strategy/rule_signals.py is kept
        # intentionally for direct unit-test coverage of the underlying indicator logic.
        "xs_momentum",
        "xs_flow",
        "xs_oi_skew",
        "funding_flow_carry",
        "lsr_oi_regime_filter",
        "supertrend",
        "ichimoku_trend",
        "carry_net_of_funding",
        "liquidity_participation_breakout",
        "btc_neutral_residual_reversal",
        "price_band_reversion",
        "funding_flow_exhaustion_sparse",
        "oi_lsr_unwind",
        "vol_contraction_breakout",
    }
)


def is_family_tf_retired(family: str, tf: str) -> bool:
    """Return True when (family, tf) already failed a full EconomicReplay cycle.

    Retired pairs must not re-enter EconomicReplay without new evidence
    (e.g. a different tf not yet tested). Callers MUST NOT bypass this
    for the same (family, tf); a different tf for the same family is allowed.
    """
    return (family, tf) in FAMILY_TF_RETIREMENT


def resolve_retired_families_for_tf(tf: str) -> tuple[str, ...]:
    """Return all families retired for the given tf, per FAMILY_TF_RETIREMENT.

    Consumed by recipe-catalog construction to actually exclude retired
    (family, tf) pairs — FAMILY_TF_RETIREMENT was previously never wired
    into any call site. [ADR_20260708_L0_SIGNAL_YIELD_IMPROVEMENT]
    """
    return tuple(family for family, retired_tf in FAMILY_TF_RETIREMENT if retired_tf == tf)
