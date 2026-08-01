"""Pre-registered return sources that may never enter expert evaluation.

Previously rejected hypotheses must never be re-labelled as new experts and
re-enter promotion by renaming; infrastructure that merely registers them is
never a promotion.
"""

from __future__ import annotations

FORBIDDEN_RETURN_SOURCES: frozenset[str] = frozenset({
    "donchian_multi_symbol_diversification",
    "bollinger_mean_reversion",
    "cross_sectional_momentum",
    "cash_carry",
    "taker_flow",
    "funding_signed_directional",
})
