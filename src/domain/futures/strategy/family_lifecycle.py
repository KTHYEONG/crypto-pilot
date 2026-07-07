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
})


def is_family_tf_retired(family: str, tf: str) -> bool:
    """Return True when (family, tf) already failed a full EconomicReplay cycle.

    Retired pairs must not re-enter EconomicReplay without new evidence
    (e.g. a different tf not yet tested). Callers MUST NOT bypass this
    for the same (family, tf); a different tf for the same family is allowed.
    """
    return (family, tf) in FAMILY_TF_RETIREMENT
