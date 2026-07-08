"""Signal family x timeframe retirement registry. [ADR_20260706_L0_SIGNAL_FAMILY_DIVERSITY]"""

from __future__ import annotations

FAMILY_TF_RETIREMENT: frozenset[tuple[str, str]] = frozenset({
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
    ("ichimoku_trend", "12h"),          # cost_drag=20.53, gross≈0.75bps — 방향성 정보 사실상 0
    ("vol_breakout", "4h"),             # gross=-27.70bps — 비용 이전에 gross 자체가 확정적으로 음수
    ("residual_momentum_xs", "4h"),     # cost_drag=3.77, nw_tstat=-9.58 — 강한 확신의 음의 엣지
    ("xs_residual_rebalance", "4h"),    # cost_drag=5.14, nw_tstat=-6.40 — 상동
})


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
