"""Weak prior penalty for mismatched SIGNAL_TYPE x EXIT_FAMILY (tmp.md)."""

from __future__ import annotations

_TRENDISH: frozenset[str] = frozenset(
    {"SUPERTREND", "FRAMA_EVR", "ADX_BREAKOUT", "OBV_MA", "RS_MOMENTUM"}
)
_MEANREV: frozenset[str] = frozenset({"RSI2_PULLBACK", "STOCHRSI_CROSS", "KC_PULLBACK", "VIX_FIX"})
_HYBRID: frozenset[str] = frozenset({"BB_SQUEEZE", "MACD_HIST_DIV"})


def exit_family_prior_penalty(signal_type: str, exit_family: str) -> float:
    """
    Return a small objective subtraction (positive = penalize) for awkward pairings.
    Not a hard gate.
    """
    s = str(signal_type).upper()
    f = str(exit_family).upper()
    if s in _TRENDISH and f == "FAST_REALIZE":
        return 0.12
    if s in _MEANREV and f == "TREND_HOLD":
        return 0.12
    if s in _HYBRID and f == "FAST_REALIZE":
        return 0.08
    return 0.0
