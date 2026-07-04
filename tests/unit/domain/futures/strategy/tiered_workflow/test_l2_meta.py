from __future__ import annotations

from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    _build_bucket_reliability,
    _parse_meta_group_ids,
)


def test_regime_bucket_reliability_allows_consistent_fit_cal() -> None:
    reliability = _build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=18.0,
        cal_edge_bps=12.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    assert reliability.sign_consistent is True
    assert reliability.action == "allow"
    assert reliability.reliability >= 0.55


def test_regime_bucket_reliability_pools_sign_flip() -> None:
    reliability = _build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=18.0,
        cal_edge_bps=-4.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    assert reliability.sign_consistent is False
    assert reliability.action == "pool"


# ── L3: Adaptive Regime-Reliability ──────────────────────────────────────

import pytest

from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    bear_edge_per_bar_bps,
    compute_regime_reliability_multiplier,
)


class TestRegimeReliabilityMultiplier:

    def test_reliability_mult_negative_edge_downweights(self) -> None:
        result = compute_regime_reliability_multiplier(
            [-30.0, -25.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert result == pytest.approx(0.2)

    def test_reliability_mult_positive_edge_keeps_full(self) -> None:
        result = compute_regime_reliability_multiplier(
            [150.0, 140.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert result == pytest.approx(1.0)

    def test_reliability_mult_linear_ramp_midpoint(self) -> None:
        result = compute_regime_reliability_multiplier(
            [-5.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        expected = 0.2 + 0.8 * (-5.0 - (-10.0)) / (0.0 - (-10.0))
        assert result == pytest.approx(expected)

    def test_reliability_mult_empty_returns_neutral(self) -> None:
        result = compute_regime_reliability_multiplier([])
        assert result == 1.0

    def test_reliability_mult_clamped_to_floor_bounds(self) -> None:
        low = compute_regime_reliability_multiplier(
            [-1000.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert low == pytest.approx(0.2)

        high = compute_regime_reliability_multiplier(
            [1000.0],
            floor=0.2,
            neg_edge_at_floor_bps=-10.0,
            pos_edge_at_full_bps=0.0,
        )
        assert high == pytest.approx(1.0)

    def test_reliability_mult_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            compute_regime_reliability_multiplier([], floor=0.0)
        with pytest.raises(ValueError, match="floor"):
            compute_regime_reliability_multiplier([], floor=1.5)
        with pytest.raises(ValueError, match="pos_edge_at_full_bps"):
            compute_regime_reliability_multiplier(
                [], floor=0.2, pos_edge_at_full_bps=-10.0, neg_edge_at_floor_bps=0.0,
            )


class TestBearEdgePerBarBps:

    def test_bear_edge_per_bar_bps(self) -> None:
        assert bear_edge_per_bar_bps(0.015, 100) == pytest.approx(1.5)
        assert bear_edge_per_bar_bps(-0.30, 200) == pytest.approx(-15.0)
        assert bear_edge_per_bar_bps(5.0, 0) == 0.0


# ── _parse_meta_group_ids ────────────────────────────────────────────


class TestParseMetaGroupIds:
    """All error paths are covered by defensive fallback (no pytest.raises needed)."""

    # Scenario 1: Happy Path
    def test_parse_meta_group_ids_splits_family_and_tf_from_canonical_format(
        self,
    ) -> None:
        result = _parse_meta_group_ids("trend_ma:ema_12_72_4h")
        assert result == ("trend_ma", "4h")

    # Scenario 1b: Family literal containing hour pattern
    def test_parse_meta_group_ids_preserves_family_literal_containing_hour_pattern(
        self,
    ) -> None:
        result = _parse_meta_group_ids("macd_4h:base_variant_1h")
        assert result == ("macd_4h", "1h")

    # Scenario 2a: Variant without tf suffix
    def test_parse_meta_group_ids_defaults_tf_unknown_when_variant_has_no_suffix(
        self,
    ) -> None:
        result = _parse_meta_group_ids("trend_ma:ema_12_72")
        assert result == ("trend_ma", "unknown")

    # Scenario 2b: Legacy no-colon format (regression guard)
    def test_parse_meta_group_ids_preserves_legacy_no_colon_format_correctly(
        self,
    ) -> None:
        result = _parse_meta_group_ids("trend_ma_4h")
        assert result == ("trend_ma", "4h")

    # Scenario 2c: Empty string
    def test_parse_meta_group_ids_returns_unknown_pair_for_empty_string(
        self,
    ) -> None:
        result = _parse_meta_group_ids("")
        assert result == ("unknown", "unknown")
