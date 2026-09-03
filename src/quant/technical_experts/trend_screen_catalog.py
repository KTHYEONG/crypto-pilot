"""Pre-registered 30-identity trend-screen catalog and its frozen profile.

The baseline-gate screen is a separate research universe from the eighteen
production technical candidates: it pre-registers exactly 15 named trend
families times LONG/SHORT (30 identities) on exactly the 15-symbol futures
universe, all evaluated under the same causal execution rules. This catalog is
research evidence only -- none of these identities is admitted to the production
``TECHNICAL_CANDIDATES`` registry and the screen never registers a candidate.
The family configurations are implementation constants, never CLI arguments.
"""

from __future__ import annotations

import pandas as pd

from src.quant.evaluation.policy import HOLDOUT_CUTOFF
from src.quant.technical_experts.contracts import TechnicalCandidate

TREND_SCREEN_PROFILE_ID = "baseline_gate_performance_v1"

TREND_SCREEN_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "NEARUSDT",
)

# Discovery 2022-04-01..2023-12-31; qualification 2024-01-01..2025-12-31.
# The 23:59:59 boundaries keep the final bar of each window (matching the
# shared HOLDOUT_CUTOFF convention for inclusive slicing).
TREND_SCREEN_DISCOVERY_START = pd.Timestamp("2022-04-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
QUALIFICATION_START = pd.Timestamp("2024-01-01", tz="UTC")
QUALIFICATION_END = HOLDOUT_CUTOFF

TREND_SCREEN_FAMILIES: tuple[str, ...] = (
    "ema_alignment",
    "macd_histogram_regime",
    "adx_di_regime",
    "ichimoku_cloud",
    "bb_squeeze_breakout",
    "supertrend",
    "parabolic_sar",
    "keltner_channel_breakout",
    "donchian_breakout",
    "chandelier_trend",
    "aroon_trend",
    "vortex_trend",
    "hull_moving_average",
    "regression_slope",
    "atr_volatility_breakout",
)

_FAMILY_CONFIGS: dict[str, dict[str, int | float]] = {
    "ema_alignment": {"fast": 20, "mid": 50, "slow": 200},
    "macd_histogram_regime": {"fast": 12, "slow": 26, "signal": 9, "regime": 200},
    "adx_di_regime": {"period": 14},
    "ichimoku_cloud": {"tenkan": 9, "kijun": 26, "span": 52},
    "bb_squeeze_breakout": {
        "period": 20, "mult": 2.0, "squeeze_window": 120,
        "squeeze_percentile": 0.2, "regime": 200,
    },
    "supertrend": {"period": 10, "mult": 3.0, "regime": 200},
    "parabolic_sar": {"step": 0.02, "max_step": 0.2, "regime": 200},
    "keltner_channel_breakout": {"period": 20, "mult": 2.0, "regime": 200},
    "donchian_breakout": {"entry": 55, "exit": 20, "regime": 200},
    "chandelier_trend": {"period": 22, "mult": 3.0, "regime": 200},
    "aroon_trend": {"period": 25, "regime": 200},
    "vortex_trend": {"period": 14, "regime": 200},
    "hull_moving_average": {"period": 55, "regime": 200},
    "regression_slope": {"period": 63, "regime": 200},
    "atr_volatility_breakout": {"period": 20, "mult": 1.5, "regime": 200},
}

# A screen identity needs enough warm-up to form its trend decision; the ADX and
# Ichimoku families need less, the remaining trend families all need the 200-bar
# EMA regime.
_FAMILY_MIN_BARS: dict[str, int] = {
    "ema_alignment": 201,
    "macd_histogram_regime": 201,
    "adx_di_regime": 30,
    "ichimoku_cloud": 53,
    "bb_squeeze_breakout": 201,
    "supertrend": 201,
    "parabolic_sar": 201,
    "keltner_channel_breakout": 201,
    "donchian_breakout": 201,
    "chandelier_trend": 201,
    "aroon_trend": 201,
    "vortex_trend": 201,
    "hull_moving_average": 201,
    "regression_slope": 201,
    "atr_volatility_breakout": 201,
}


def _build_trend_screen_candidates() -> tuple[TechnicalCandidate, ...]:
    candidates: list[TechnicalCandidate] = []
    for family in TREND_SCREEN_FAMILIES:
        for side in ("LONG", "SHORT"):
            return_source = f"technical_{family}_{side.lower()}"
            candidates.append(TechnicalCandidate(
                return_source,
                return_source,
                family,
                side,
                _FAMILY_CONFIGS[family],
                _FAMILY_MIN_BARS[family],
            ))
    return tuple(candidates)


TREND_SCREEN_CANDIDATES: tuple[TechnicalCandidate, ...] = _build_trend_screen_candidates()


def _check_contract() -> None:
    """Executable assertions locking the 30-identity screen catalog surface."""
    from src.quant.technical_experts.catalog import TECHNICAL_CANDIDATES

    ids = [candidate.candidate_id for candidate in TREND_SCREEN_CANDIDATES]
    sources = [candidate.return_source for candidate in TREND_SCREEN_CANDIDATES]
    assert len(TREND_SCREEN_CANDIDATES) == 30
    assert len(set(ids)) == 30
    assert len(set(sources)) == 30
    assert {candidate.family for candidate in TREND_SCREEN_CANDIDATES} == set(
        TREND_SCREEN_FAMILIES
    )
    for family in TREND_SCREEN_FAMILIES:
        sides = {
            candidate.side
            for candidate in TREND_SCREEN_CANDIDATES
            if candidate.family == family
        }
        assert sides == {"LONG", "SHORT"}
    assert len(TREND_SCREEN_SYMBOLS) == 15
    assert len(set(TREND_SCREEN_SYMBOLS)) == 15
    # The screen is separate research evidence: importing the screen catalog
    # never mutates the production 18-candidate registry.
    assert len(TECHNICAL_CANDIDATES) == 18
    assert DISCOVERY_END < QUALIFICATION_START
    assert pd.Timestamp("2026-01-01", tz="UTC") > QUALIFICATION_END


_check_contract()
