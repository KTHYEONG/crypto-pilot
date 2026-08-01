"""Source-controlled registry of the eighteen frozen technical candidates.

The registry is the single place the fixed candidate matrix lives: exactly two
directional candidates per family and nine frozen families, all with
catalog-supplied indicator configurations. A candidate resolves only by its
exact ``return_source``; an unknown or retired identity fails closed.
"""

from __future__ import annotations

from src.research.technical_experts.contracts import TechnicalCandidate

TECHNICAL_CANDIDATES: tuple[TechnicalCandidate, ...] = (
    # ema_alignment: EMA 20/50/200 alignment and price reclaim in the slow trend.
    TechnicalCandidate(
        "technical_ema_alignment_long_v1", "technical_ema_alignment_long_v1",
        "ema_alignment", "LONG", {"fast": 20, "mid": 50, "slow": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_ema_alignment_short_v1", "technical_ema_alignment_short_v1",
        "ema_alignment", "SHORT", {"fast": 20, "mid": 50, "slow": 200}, 201,
    ),
    # macd_histogram_regime: MACD(12,26,9) histogram zero-cross inside the 200 EMA regime.
    TechnicalCandidate(
        "technical_macd_histogram_regime_long_v1", "technical_macd_histogram_regime_long_v1",
        "macd_histogram_regime", "LONG",
        {"fast": 12, "slow": 26, "signal": 9, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_macd_histogram_regime_short_v1", "technical_macd_histogram_regime_short_v1",
        "macd_histogram_regime", "SHORT",
        {"fast": 12, "slow": 26, "signal": 9, "regime": 200}, 201,
    ),
    # adx_di_regime: ADX(14) >= 25 with a directional +DI/-DI crossover.
    TechnicalCandidate(
        "technical_adx_di_regime_long_v1", "technical_adx_di_regime_long_v1",
        "adx_di_regime", "LONG", {"period": 14}, 30,
    ),
    TechnicalCandidate(
        "technical_adx_di_regime_short_v1", "technical_adx_di_regime_short_v1",
        "adx_di_regime", "SHORT", {"period": 14}, 30,
    ),
    # ichimoku_cloud: price beyond the current, non-forward 52-bar cloud.
    TechnicalCandidate(
        "technical_ichimoku_cloud_long_v1", "technical_ichimoku_cloud_long_v1",
        "ichimoku_cloud", "LONG", {"tenkan": 9, "kijun": 26, "span": 52}, 53,
    ),
    TechnicalCandidate(
        "technical_ichimoku_cloud_short_v1", "technical_ichimoku_cloud_short_v1",
        "ichimoku_cloud", "SHORT", {"tenkan": 9, "kijun": 26, "span": 52}, 53,
    ),
    # bb_squeeze_breakout: bandwidth below the prior 120-bar 20th percentile then
    # a close outside the band in the 200 EMA direction.
    TechnicalCandidate(
        "technical_bb_squeeze_breakout_long_v1", "technical_bb_squeeze_breakout_long_v1",
        "bb_squeeze_breakout", "LONG",
        {"period": 20, "mult": 2.0, "squeeze_window": 120, "squeeze_percentile": 0.2, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_bb_squeeze_breakout_short_v1", "technical_bb_squeeze_breakout_short_v1",
        "bb_squeeze_breakout", "SHORT",
        {"period": 20, "mult": 2.0, "squeeze_window": 120, "squeeze_percentile": 0.2, "regime": 200}, 201,
    ),
    # rsi_trend_pullback: RSI(14) recrosses 40/60 only in the 200 EMA trend direction.
    TechnicalCandidate(
        "technical_rsi_trend_pullback_long_v1", "technical_rsi_trend_pullback_long_v1",
        "rsi_trend_pullback", "LONG", {"period": 14, "lower": 40.0, "upper": 60.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_rsi_trend_pullback_short_v1", "technical_rsi_trend_pullback_short_v1",
        "rsi_trend_pullback", "SHORT", {"period": 14, "lower": 40.0, "upper": 60.0, "regime": 200}, 201,
    ),
    # stochastic_trend_pullback: Stoch(14,3,3) %K/%D cross out of 30/70 in the trend direction.
    TechnicalCandidate(
        "technical_stochastic_trend_pullback_long_v1", "technical_stochastic_trend_pullback_long_v1",
        "stochastic_trend_pullback", "LONG",
        {"k_period": 14, "d_period": 3, "smooth": 3, "lower": 30.0, "upper": 70.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_stochastic_trend_pullback_short_v1", "technical_stochastic_trend_pullback_short_v1",
        "stochastic_trend_pullback", "SHORT",
        {"k_period": 14, "d_period": 3, "smooth": 3, "lower": 30.0, "upper": 70.0, "regime": 200}, 201,
    ),
    # cci_trend_pullback: CCI(20) recrosses -100/+100 only in the 200 EMA trend direction.
    TechnicalCandidate(
        "technical_cci_trend_pullback_long_v1", "technical_cci_trend_pullback_long_v1",
        "cci_trend_pullback", "LONG", {"period": 20, "lower": -100.0, "upper": 100.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_cci_trend_pullback_short_v1", "technical_cci_trend_pullback_short_v1",
        "cci_trend_pullback", "SHORT", {"period": 20, "lower": -100.0, "upper": 100.0, "regime": 200}, 201,
    ),
    # mfi_trend_pullback: MFI(14) recrosses 20/80 only in the 200 EMA trend direction.
    TechnicalCandidate(
        "technical_mfi_trend_pullback_long_v1", "technical_mfi_trend_pullback_long_v1",
        "mfi_trend_pullback", "LONG", {"period": 14, "lower": 20.0, "upper": 80.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_mfi_trend_pullback_short_v1", "technical_mfi_trend_pullback_short_v1",
        "mfi_trend_pullback", "SHORT", {"period": 14, "lower": 20.0, "upper": 80.0, "regime": 200}, 201,
    ),
)

_FROZEN_FAMILIES = {
    "ema_alignment",
    "macd_histogram_regime",
    "adx_di_regime",
    "ichimoku_cloud",
    "bb_squeeze_breakout",
    "rsi_trend_pullback",
    "stochastic_trend_pullback",
    "cci_trend_pullback",
    "mfi_trend_pullback",
}


def resolve_technical_candidate(return_source: str) -> TechnicalCandidate:
    """Resolve one frozen candidate by its exact return source.

    An unknown or retired return source raises ``ValueError``; aliases are never
    mapped, so a rejected identity cannot be re-entered under another name.
    """
    for candidate in TECHNICAL_CANDIDATES:
        if candidate.return_source == return_source:
            return candidate
    raise ValueError(f"unknown or retired technical return source '{return_source}'")


def _check_contract() -> None:
    """Executable assertions locking the frozen 18-candidate registry surface."""
    ids = [c.candidate_id for c in TECHNICAL_CANDIDATES]
    sources = [c.return_source for c in TECHNICAL_CANDIDATES]
    assert len(TECHNICAL_CANDIDATES) == 18
    assert len(set(ids)) == 18
    assert len(set(sources)) == 18
    assert {c.family for c in TECHNICAL_CANDIDATES} == _FROZEN_FAMILIES
    assert {c.side for c in TECHNICAL_CANDIDATES} == {"LONG", "SHORT"}
    for family in _FROZEN_FAMILIES:
        sides = {c.side for c in TECHNICAL_CANDIDATES if c.family == family}
        assert sides == {"LONG", "SHORT"}
    assert (
        resolve_technical_candidate("technical_macd_histogram_regime_long_v1").family
        == "macd_histogram_regime"
    )


_check_contract()
