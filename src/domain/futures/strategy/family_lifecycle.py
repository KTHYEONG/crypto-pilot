"""Signal family x timeframe retirement registry. [ADR_20260706_L0_SIGNAL_FAMILY_DIVERSITY]"""

from __future__ import annotations

FAMILY_TF_RETIREMENT: frozenset[tuple[str, str]] = frozenset({
    ("residual_reversion", "4h"),
    ("funding_extreme_reversal", "4h"),
})


def is_family_tf_retired(family: str, tf: str) -> bool:
    """Return True when (family, tf) already failed a full EconomicReplay cycle.

    Retired pairs must not re-enter EconomicReplay without new evidence
    (e.g. a different tf not yet tested). Callers MUST NOT bypass this
    for the same (family, tf); a different tf for the same family is allowed.
    """
    return (family, tf) in FAMILY_TF_RETIREMENT
