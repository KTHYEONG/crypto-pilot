from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.research.portfolio.growth_router import (
    ContextState,
    build_rolling_segments,
    causal_router,
    compute_context_features,
    context_state_for,
    enough_deployment_folds,
)


def _dates(n: int, start: str = "2022-01-01") -> list[pd.Timestamp]:
    return [
        pd.Timestamp(start, tz="UTC") + pd.DateOffset(months=i) for i in range(n)
    ]


class TestBuildRollingSegments:
    def test_three_month_windows_with_prior_discovery(self) -> None:
        dates = _dates(36)
        segments = build_rolling_segments(dates)
        assert len(segments) == 11
        first = segments[0]
        assert first.deployment_dates == tuple(dates[3:6])
        assert first.discovery_dates == tuple(dates[0:3])
        assert all(d < first.deployment_dates[0] for d in first.discovery_dates)
        for left, right in itertools.pairwise(segments):
            assert left.deployment_dates[-1] < right.deployment_dates[0]

    def test_trailing_partial_window_is_dropped(self) -> None:
        dates = _dates(36 + 2)
        segments = build_rolling_segments(dates)
        assert len(segments) == 11
        assert segments[-1].deployment_dates[-1] <= dates[-1]

    def test_discovery_never_overlaps_deployment(self) -> None:
        segments = build_rolling_segments(_dates(36))
        for segment in segments:
            assert max(segment.discovery_dates) < min(segment.deployment_dates)

    def test_rejects_non_positive_horizons(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            build_rolling_segments(_dates(12), discovery_months=0)


class TestEnoughDeploymentFolds:
    def test_requires_three_six_month_folds(self) -> None:
        assert enough_deployment_folds(build_rolling_segments(_dates(36))) is True
        assert enough_deployment_folds(build_rolling_segments(_dates(12 + 11))) is False

    def test_fails_closed_on_empty(self) -> None:
        assert enough_deployment_folds([]) is False


class TestContextState:
    def test_context_state_for_partitions_features(self) -> None:
        state = context_state_for(
            (0.001, 0.02, 0.9), vol_threshold=0.01, breadth_threshold=0.5,
        )
        assert state == ContextState(market_ret_up=True, high_vol=True, wide_breadth=True)
        low = context_state_for(
            (-0.001, 0.005, 0.4), vol_threshold=0.01, breadth_threshold=0.5,
        )
        assert low == ContextState(market_ret_up=False, high_vol=False, wide_breadth=False)

    def test_context_state_for_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            context_state_for((float("nan"), 0.01, 0.5), vol_threshold=0.01, breadth_threshold=0.5)

    def test_compute_context_features_trailing_window(self) -> None:
        market = np.arange(400, dtype=np.float64)
        breadth = np.full(400, 0.8)
        mean_ret, vol, mean_breadth = compute_context_features(market, breadth, end_idx=300)
        assert mean_ret == pytest.approx(np.mean(market[300 - 180 : 300]))
        assert vol == pytest.approx(np.std(market[300 - 180 : 300]))
        assert mean_breadth == pytest.approx(0.8)

    def test_compute_context_features_fails_closed_on_insufficient_samples(self) -> None:
        market = np.arange(10, dtype=np.float64)
        breadth = np.full(10, 0.8)
        out = compute_context_features(market, breadth, end_idx=10)
        assert all(np.isnan(value) for value in out)


class TestCausalRouter:
    def test_selects_highest_positive_context_lcb(self) -> None:
        assert causal_router({"funding": 0.03, "taker": 0.12, "trend": 0.05}) == "taker"

    def test_cash_when_no_positive_pair(self) -> None:
        assert causal_router({"a": -0.1, "b": 0.0, "c": float("nan")}) is None

    def test_cash_on_empty_evidence(self) -> None:
        assert causal_router({}) is None

    def test_ignores_non_finite(self) -> None:
        assert causal_router({"a": float("inf"), "b": 0.1}) == "b"

    def test_tiebreak_is_lexicographic_and_deterministic(self) -> None:
        assert causal_router({"z_sleeve": 0.1, "a_sleeve": 0.1}) == "a_sleeve"
        assert causal_router({"z_sleeve": 0.1, "a_sleeve": 0.1}) == causal_router(
            {"z_sleeve": 0.1, "a_sleeve": 0.1},
        )
