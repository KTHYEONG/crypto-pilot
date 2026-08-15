from __future__ import annotations

import pytest

from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
from src.domain.futures.compound.contracts import AlphaCandidateState


class TestBuildMultiscaleAlphaCatalog:
    def test_catalog_has_exact_explicit_recipe_contract(self) -> None:
        catalog = build_multiscale_alpha_catalog()
        assert len(catalog) == 12

        ids = {r.recipe_id for r in catalog}
        assert len(ids) == 12
        assert "ts_trend_4h_h24" in ids
        assert "ts_trend_12h_h72" in ids
        assert "ts_trend_1d_h168" in ids
        assert "xs_resmom_4h_h24" in ids
        assert "xs_resmom_12h_h72" in ids
        assert "breakout_4h_h24" in ids
        assert "breakout_12h_h72" in ids
        assert "carry_funding_event_h8" in ids
        assert "basis_reversion_1h_h8" in ids
        assert "flow_imbalance_15m_h1" in ids
        assert "flow_oi_confirm_1h_h4" in ids
        assert "liquidity_exhaustion_15m_h1" in ids

    def test_no_recipe_starts_active(self) -> None:
        catalog = build_multiscale_alpha_catalog()
        for r in catalog:
            assert r.initial_state != AlphaCandidateState.ACTIVE

    def test_all_recipes_have_supported_timeframes(self) -> None:
        catalog = build_multiscale_alpha_catalog()
        supported = {"15m", "1h", "4h", "12h", "1d", "funding_event"}
        for r in catalog:
            assert r.native_timeframe in supported

    def test_raises_on_duplicate_id(self) -> None:
        from unittest.mock import patch

        with patch(
            "src.domain.futures.compound.alpha_catalog._MULTISCALE_RECIPES",
            (
                {"recipe_id": "dup", "family": "trend", "native_timeframe": "4h",
                 "lookback_hours": (72,), "horizon_hours": 24,
                 "required_fields": ("close",), "initial_state": AlphaCandidateState.CORE_CANDIDATE,
                 "max_half_life_hours": 12.0},
                {"recipe_id": "dup", "family": "trend", "native_timeframe": "12h",
                 "lookback_hours": (168,), "horizon_hours": 72,
                 "required_fields": ("close",), "initial_state": AlphaCandidateState.CORE_CANDIDATE,
                 "max_half_life_hours": 36.0},
            ),
        ), pytest.raises(ValueError, match="duplicate"):
            build_multiscale_alpha_catalog()
