from __future__ import annotations

import pandas as pd

from src.quant.technical_experts.catalog import (
    TECHNICAL_CANDIDATES,
    TECHNICAL_EXPERT_FAMILIES,
)
from src.quant.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    QUALIFICATION_END,
    QUALIFICATION_START,
    TREND_SCREEN_DISCOVERY_START,
    TREND_SCREEN_CANDIDATES,
    TREND_SCREEN_FAMILIES,
    TREND_SCREEN_PROFILE_ID,
    TREND_SCREEN_SYMBOLS,
)


class TestTrendScreenCatalog:
    def test_exactly_30_unique_identities(self) -> None:
        ids = [c.candidate_id for c in TREND_SCREEN_CANDIDATES]
        sources = [c.return_source for c in TREND_SCREEN_CANDIDATES]
        assert len(TREND_SCREEN_CANDIDATES) == 30
        assert len(set(ids)) == 30
        assert len(set(sources)) == 30

    def test_exactly_15_exact_symbols(self) -> None:
        assert len(TREND_SCREEN_SYMBOLS) == 15
        assert len(set(TREND_SCREEN_SYMBOLS)) == 15
        assert TREND_SCREEN_SYMBOLS == (
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
            "DOGEUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT",
            "BCHUSDT", "DOTUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT",
        )

    def test_fifteen_families_each_with_long_and_short(self) -> None:
        assert len(TREND_SCREEN_FAMILIES) == 15
        assert set(TREND_SCREEN_FAMILIES) == {c.family for c in TREND_SCREEN_CANDIDATES}
        for family in TREND_SCREEN_FAMILIES:
            sides = {c.side for c in TREND_SCREEN_CANDIDATES if c.family == family}
            assert sides == {"LONG", "SHORT"}, family

    def test_no_production_catalog_mutation(self) -> None:
        # Importing the screen catalog must never change the frozen production
        # 18-candidate registry or the admitted family set.
        assert len(TECHNICAL_CANDIDATES) == 18
        assert {
            "ema_alignment",
            "macd_histogram_regime",
            "adx_di_regime",
            "ichimoku_cloud",
            "bb_squeeze_breakout",
            "rsi_trend_pullback",
            "stochastic_trend_pullback",
            "cci_trend_pullback",
            "mfi_trend_pullback",
        } == TECHNICAL_EXPERT_FAMILIES
        assert TECHNICAL_EXPERT_FAMILIES.isdisjoint(
            {"donchian_breakout", "chandelier_trend", "aroon_trend", "vortex_trend",
             "hull_moving_average", "regression_slope", "atr_volatility_breakout"}
        )

    def test_profile_windows_are_chronological_and_sealed(self) -> None:
        assert TREND_SCREEN_PROFILE_ID == "baseline_gate_performance_v1"
        assert TREND_SCREEN_DISCOVERY_START < DISCOVERY_END < QUALIFICATION_START < QUALIFICATION_END
        assert QUALIFICATION_END.normalize() + pd.Timedelta(days=1) > QUALIFICATION_END
        assert QUALIFICATION_END.year == 2025

    def test_qualification_end_is_shared_holdout_cutoff(self) -> None:
        # SCENARIO_MHS_GAP_HARDENING_05: QUALIFICATION_END must be the very
        # object as policy.HOLDOUT_CUTOFF -- a regression guard against the two
        # independently-typed literals silently diverging again.
        from src.quant.evaluation.policy import HOLDOUT_CUTOFF

        assert QUALIFICATION_END is HOLDOUT_CUTOFF
        assert pd.Timestamp("2025-12-31 23:59:59", tz="UTC") == QUALIFICATION_END
