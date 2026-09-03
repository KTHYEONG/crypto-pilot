"""Source-controlled registry of the eighteen frozen technical candidates.

The registry is the single place the fixed candidate matrix lives: exactly two
directional candidates per family and nine frozen families, all with
catalog-supplied indicator configurations. A candidate resolves only by its
exact ``return_source``; an unknown or retired identity fails closed.
"""

from __future__ import annotations

import dataclasses

from src.market_data.storage.loaders import timeframe_scale_factor
from src.research.technical_experts.contracts import TechnicalCandidate

TECHNICAL_CANDIDATES: tuple[TechnicalCandidate, ...] = (
    # ema_alignment: EMA 20/50/200 alignment and price reclaim in the slow trend.
    TechnicalCandidate(
        "technical_ema_alignment_long", "technical_ema_alignment_long",
        "ema_alignment", "LONG", {"fast": 20, "mid": 50, "slow": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_ema_alignment_short", "technical_ema_alignment_short",
        "ema_alignment", "SHORT", {"fast": 20, "mid": 50, "slow": 200}, 201,
    ),
    # macd_histogram_regime: MACD(12,26,9) histogram zero-cross inside the 200 EMA regime.
    TechnicalCandidate(
        "technical_macd_histogram_regime_long", "technical_macd_histogram_regime_long",
        "macd_histogram_regime", "LONG",
        {"fast": 12, "slow": 26, "signal": 9, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_macd_histogram_regime_short", "technical_macd_histogram_regime_short",
        "macd_histogram_regime", "SHORT",
        {"fast": 12, "slow": 26, "signal": 9, "regime": 200}, 201,
    ),
    # adx_di_regime: ADX(14) >= 25 with a directional +DI/-DI crossover.
    TechnicalCandidate(
        "technical_adx_di_regime_long", "technical_adx_di_regime_long",
        "adx_di_regime", "LONG", {"period": 14}, 30,
    ),
    TechnicalCandidate(
        "technical_adx_di_regime_short", "technical_adx_di_regime_short",
        "adx_di_regime", "SHORT", {"period": 14}, 30,
    ),
    # ichimoku_cloud: price beyond the current, non-forward 52-bar cloud.
    TechnicalCandidate(
        "technical_ichimoku_cloud_long", "technical_ichimoku_cloud_long",
        "ichimoku_cloud", "LONG", {"tenkan": 9, "kijun": 26, "span": 52}, 53,
    ),
    TechnicalCandidate(
        "technical_ichimoku_cloud_short", "technical_ichimoku_cloud_short",
        "ichimoku_cloud", "SHORT", {"tenkan": 9, "kijun": 26, "span": 52}, 53,
    ),
    # bb_squeeze_breakout: bandwidth below the prior 120-bar 20th percentile then
    # a close outside the band in the 200 EMA direction.
    TechnicalCandidate(
        "technical_bb_squeeze_breakout_long", "technical_bb_squeeze_breakout_long",
        "bb_squeeze_breakout", "LONG",
        {"period": 20, "mult": 2.0, "squeeze_window": 120, "squeeze_percentile": 0.2, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_bb_squeeze_breakout_short", "technical_bb_squeeze_breakout_short",
        "bb_squeeze_breakout", "SHORT",
        {"period": 20, "mult": 2.0, "squeeze_window": 120, "squeeze_percentile": 0.2, "regime": 200}, 201,
    ),
    # rsi_trend_pullback: RSI(14) recrosses 40/60 only in the 200 EMA trend direction.
    TechnicalCandidate(
        "technical_rsi_trend_pullback_long", "technical_rsi_trend_pullback_long",
        "rsi_trend_pullback", "LONG", {"period": 14, "lower": 40.0, "upper": 60.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_rsi_trend_pullback_short", "technical_rsi_trend_pullback_short",
        "rsi_trend_pullback", "SHORT", {"period": 14, "lower": 40.0, "upper": 60.0, "regime": 200}, 201,
    ),
    # stochastic_trend_pullback: Stoch(14,3,3) %K/%D cross out of 30/70 in the trend direction.
    TechnicalCandidate(
        "technical_stochastic_trend_pullback_long", "technical_stochastic_trend_pullback_long",
        "stochastic_trend_pullback", "LONG",
        {"k_period": 14, "d_period": 3, "smooth": 3, "lower": 30.0, "upper": 70.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_stochastic_trend_pullback_short", "technical_stochastic_trend_pullback_short",
        "stochastic_trend_pullback", "SHORT",
        {"k_period": 14, "d_period": 3, "smooth": 3, "lower": 30.0, "upper": 70.0, "regime": 200}, 201,
    ),
    # cci_trend_pullback: CCI(20) recrosses -100/+100 only in the 200 EMA trend direction.
    TechnicalCandidate(
        "technical_cci_trend_pullback_long", "technical_cci_trend_pullback_long",
        "cci_trend_pullback", "LONG", {"period": 20, "lower": -100.0, "upper": 100.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_cci_trend_pullback_short", "technical_cci_trend_pullback_short",
        "cci_trend_pullback", "SHORT", {"period": 20, "lower": -100.0, "upper": 100.0, "regime": 200}, 201,
    ),
    # mfi_trend_pullback: MFI(14) recrosses 20/80 only in the 200 EMA trend direction.
    TechnicalCandidate(
        "technical_mfi_trend_pullback_long", "technical_mfi_trend_pullback_long",
        "mfi_trend_pullback", "LONG", {"period": 14, "lower": 20.0, "upper": 80.0, "regime": 200}, 201,
    ),
    TechnicalCandidate(
        "technical_mfi_trend_pullback_short", "technical_mfi_trend_pullback_short",
        "mfi_trend_pullback", "SHORT", {"period": 14, "lower": 20.0, "upper": 80.0, "regime": 200}, 201,
    ),
)

_CANDIDATES_BY_SOURCE: dict[str, TechnicalCandidate] = {
    candidate.return_source: candidate for candidate in TECHNICAL_CANDIDATES
}

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

# Frozen families currently admitted as technical candidates. ``supertrend``,
# ``parabolic_sar`` and ``keltner_channel_breakout`` are implemented in
# signals.py but are NOT yet admitted here: they must clear the same
# LibraryAdmissionConfig gate as the existing families before a catalog entry
# (both sides) is added -- no bar is lowered for "proven" strategies.
TECHNICAL_EXPERT_FAMILIES: frozenset[str] = frozenset(_FROZEN_FAMILIES)

_BASELINE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
_BASELINE_TIMEFRAME = "4h"


def _build_admitted_family_matrix() -> dict[str, dict[tuple[str, str], bool]]:
    """Build the family x symbol x timeframe admission pass/fail matrix.

    The matrix is the evidence gate for catalog membership: a family must show
    at least one admission pass across all measured ``(symbol, timeframe)``
    cells or it is pruned from the catalog. It is populated from the §1
    timeframe census (``docs/results/timeframe-census.md``) when available;
    pending that census run it records the measured baseline evidence from
    ``docs/results/rolling-res.md`` (the ``technical-5symbol-rolling`` profile
    at ``4h``, whose eighteen long/short candidates are the current admitted
    universe). Non-measured cells are ``False`` and never fabricated.
    """
    matrix: dict[str, dict[tuple[str, str], bool]] = {}
    for family in TECHNICAL_EXPERT_FAMILIES:
        matrix[family] = {
            (symbol, _BASELINE_TIMEFRAME): True
            for symbol in _BASELINE_SYMBOLS
        }
    return matrix


ADMITTED_FAMILY_MATRIX: dict[str, dict[tuple[str, str], bool]] = _build_admitted_family_matrix()


def resolve_technical_candidate(
    return_source: str, *, timeframe: str = "4h",
) -> TechnicalCandidate:
    """Resolve one frozen candidate by its exact return source.

    An unknown or retired return source raises ``ValueError``; aliases are never
    mapped, so a rejected identity cannot be re-entered under another name. The
    catalog's fixed bar counts are 4h-reference values; at any other
    ``timeframe`` every int-valued config entry and ``min_history_bars`` are
    rescaled (float thresholds pass through) so the same calendar window is
    preserved. ``TECHNICAL_CANDIDATES`` stays the frozen 4h-reference registry.
    """
    candidate = _CANDIDATES_BY_SOURCE.get(return_source)
    if candidate is None:
        raise ValueError(f"unknown or retired technical return source '{return_source}'")
    scale = timeframe_scale_factor(timeframe)
    config = {
        key: max(1, round(value * scale)) if isinstance(value, int) else value
        for key, value in candidate.config.items()
    }
    return dataclasses.replace(
        candidate,
        config=config,
        min_history_bars=max(1, round(candidate.min_history_bars * scale)),
    )


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
        resolve_technical_candidate("technical_macd_histogram_regime_long").family
        == "macd_histogram_regime"
    )
    assert set(TECHNICAL_EXPERT_FAMILIES) == _FROZEN_FAMILIES
    for family in TECHNICAL_EXPERT_FAMILIES:
        assert family in ADMITTED_FAMILY_MATRIX
        assert any(ADMITTED_FAMILY_MATRIX[family].values())


_check_contract()
