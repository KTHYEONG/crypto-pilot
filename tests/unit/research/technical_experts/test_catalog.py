from __future__ import annotations

import pytest

from src.research.technical_experts.catalog import (
    TECHNICAL_CANDIDATES,
    resolve_technical_candidate,
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


class TestTechnicalCatalog:
    def test_registry_is_exact_and_fail_closed(self) -> None:
        assert len(TECHNICAL_CANDIDATES) == 18
        assert {c.family for c in TECHNICAL_CANDIDATES} == _FROZEN_FAMILIES
        assert {c.side for c in TECHNICAL_CANDIDATES} == {"LONG", "SHORT"}
        assert (
            resolve_technical_candidate("technical_macd_histogram_regime_long_v1").family
            == "macd_histogram_regime"
        )

    def test_all_candidates_have_unique_source_controlled_identities(self) -> None:
        ids = [c.candidate_id for c in TECHNICAL_CANDIDATES]
        sources = [c.return_source for c in TECHNICAL_CANDIDATES]
        assert len(set(ids)) == 18
        assert len(set(sources)) == 18
        for candidate in TECHNICAL_CANDIDATES:
            assert candidate.return_source == f"technical_{candidate.family}_{candidate.side.lower()}_v1"

    def test_unknown_or_retired_source_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown or retired"):
            resolve_technical_candidate("technical_open_interest_deleveraging_v1")
        with pytest.raises(ValueError, match="unknown or retired"):
            resolve_technical_candidate("technical_macd_histogram_regime_long_v2")
        with pytest.raises(ValueError, match="unknown or retired"):
            resolve_technical_candidate("bollinger_mean_reversion")

    def test_rejected_anti_pattern_sources_are_not_registered(self) -> None:
        sources = {c.return_source for c in TECHNICAL_CANDIDATES}
        for rejected in (
            "open_interest_deleveraging_v1",
            "naive_bollinger_mean_reversion_long_v1",
            "funding_signed_directional_v1",
            "taker_flow_v1",
            "cash_carry_v1",
        ):
            assert rejected not in sources
