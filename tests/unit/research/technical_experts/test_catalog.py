from __future__ import annotations

import pytest

from src.market_data.storage.loaders import timeframe_scale_factor
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
            resolve_technical_candidate("technical_macd_histogram_regime_long").family
            == "macd_histogram_regime"
        )

    def test_all_candidates_have_unique_source_controlled_identities(self) -> None:
        ids = [c.candidate_id for c in TECHNICAL_CANDIDATES]
        sources = [c.return_source for c in TECHNICAL_CANDIDATES]
        assert len(set(ids)) == 18
        assert len(set(sources)) == 18
        for candidate in TECHNICAL_CANDIDATES:
            assert candidate.return_source == f"technical_{candidate.family}_{candidate.side.lower()}"

    def test_unknown_or_retired_source_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown or retired"):
            resolve_technical_candidate("technical_open_interest_deleveraging")
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

    def test_default_4h_reproduces_baseline_for_every_candidate(self) -> None:
        # TIS-01: scale_factor is exactly 1.0 at the 4h reference, so the
        # explicit 4h resolution is byte-identical to the default path for all
        # 18 identities - zero regression on the existing 4h-only behavior.
        for candidate in TECHNICAL_CANDIDATES:
            default = resolve_technical_candidate(candidate.return_source)
            explicit_4h = resolve_technical_candidate(
                candidate.return_source, timeframe="4h",
            )
            assert explicit_4h == default
            assert explicit_4h.config == default.config
            assert explicit_4h.min_history_bars == default.min_history_bars

    def test_1d_scales_down_bar_counts_and_preserves_float_thresholds(self) -> None:
        # TIS-02: at 1d the 4h/1d ratio (1/6) shrinks every int-valued config
        # entry and min_history_bars; float thresholds pass through untouched.
        scaled = resolve_technical_candidate(
            "technical_ema_alignment_long", timeframe="1d",
        )
        assert scaled.config == {
            "fast": round(20 / 6),
            "mid": round(50 / 6),
            "slow": round(200 / 6),
        }
        assert scaled.min_history_bars == max(1, round(201 / 6))

        rsi = resolve_technical_candidate(
            "technical_rsi_trend_pullback_long", timeframe="1d",
        )
        assert rsi.config["period"] == max(1, round(14 / 6))
        assert rsi.config["regime"] == max(1, round(200 / 6))
        assert rsi.config["lower"] == 40.0
        assert rsi.config["upper"] == 60.0

        bb = resolve_technical_candidate(
            "technical_bb_squeeze_breakout_long", timeframe="1d",
        )
        assert bb.config["mult"] == 2.0
        assert bb.config["squeeze_percentile"] == 0.2
        assert bb.config["squeeze_window"] == max(1, round(120 / 6))

    def test_1d_scaling_uses_the_shared_scale_factor_and_floor_of_1(self) -> None:
        # TIS-02: every int-valued config entry and min_history_bars are scaled
        # by the shared 4h/1d factor with a floor of 1; a bar count that rounds
        # to zero (e.g. stochastic d_period 3 * 1/6 = 0.5) never collapses.
        factor = timeframe_scale_factor("1d")
        assert factor == 1 / 6
        for candidate in TECHNICAL_CANDIDATES:
            default = resolve_technical_candidate(candidate.return_source)
            scaled = resolve_technical_candidate(
                candidate.return_source, timeframe="1d",
            )
            for key, value in default.config.items():
                if isinstance(value, int):
                    assert scaled.config[key] == max(1, round(value * factor))
                else:
                    assert scaled.config[key] == value
            assert scaled.min_history_bars == max(1, round(default.min_history_bars * factor))
        stochastic = resolve_technical_candidate(
            "technical_stochastic_trend_pullback_long", timeframe="1d",
        )
        assert stochastic.config["d_period"] == 1
        assert stochastic.config["smooth"] == 1

    def test_unknown_source_fails_closed_at_any_timeframe(self) -> None:
        with pytest.raises(ValueError, match="unknown or retired"):
            resolve_technical_candidate(
                "technical_open_interest_deleveraging", timeframe="1d",
            )
