from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.sleeve_blend.weights import (
    _cap_symbol_weights_np,
    _causal_weight_series,
    compute_causal_risk_weights,
    component_labels,
    symbol_of_component,
)

_BARS_PER_DAY = 6


def _returns_frame(
    n_days: int,
    vols: dict[str, float],
    seed: int = 7,
) -> pd.DataFrame:
    """Deterministic per-component 4h return history spanning ``n_days``."""
    n = n_days * _BARS_PER_DAY
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for component, vol in vols.items():
        data[component] = rng.normal(0.0, vol, n)
    return pd.DataFrame(data, index=idx)


def _as_of(frame: pd.DataFrame, days_in: int = 45) -> pd.Timestamp:
    return frame.index[days_in * _BARS_PER_DAY]


def _symbol_sums(weights: pd.Series) -> pd.Series:
    return weights.groupby(weights.index.map(symbol_of_component)).sum()


class TestSymbolOfComponent:
    def test_labels_and_parsing(self) -> None:
        assert component_labels("BTCUSDT") == ("BTCUSDT:long", "BTCUSDT:short")
        assert symbol_of_component("BTCUSDT:long") == "BTCUSDT"
        assert symbol_of_component("BTCUSDT:short") == "BTCUSDT"

    def test_malformed_label_raises(self) -> None:
        with pytest.raises(ValueError, match="malformed component label"):
            symbol_of_component("BTCUSDT")


class TestComputeCausalRiskWeights:
    def test_zero_cap_input_leaves_budget_unallocated(self) -> None:
        result = _cap_symbol_weights_np(
            np.zeros(2), np.asarray([0, 1]), 2, max_symbol_weight=0.25,
        )
        assert np.array_equal(result, np.zeros(2))

    def test_causal_inverse_vol_weights_respect_symbol_cap(self) -> None:
        # SC-SGV2-05: five symbols with disparate prior vols, one carrying both
        # a long and a short component; the frozen contract demands weights sum
        # to one and every symbol aggregate be at most 0.25 after capping.
        frame = _returns_frame(60, vols={
            "A:long": 0.005, "A:short": 0.005, "B:long": 0.006,
            "B:short": 0.006, "C:long": 0.008, "D:long": 0.010, "E:long": 0.012,
        })
        active = tuple(frame.columns)
        as_of = _as_of(frame)
        weights = compute_causal_risk_weights(frame, active, as_of=as_of)

        assert weights.index.tolist() == list(active)
        assert abs(float(weights.sum()) - 1.0) < 1e-12
        assert float(_symbol_sums(weights).max()) <= 0.25 + 1e-12

        # the lowest-volatility symbol (A) carries the highest pre-cap
        # inverse-vol signal and its long+short aggregate absorbs the cap.
        assert _symbol_sums(weights)["A"] == pytest.approx(0.25, abs=1e-9)
        assert _symbol_sums(weights)["E"] < _symbol_sums(weights)["D"] < _symbol_sums(weights)["C"]

    def test_infeasible_cap_never_exceeds_cap(self) -> None:
        # With only two symbols the 0.25 aggregate cap cannot hold one unit of
        # weight; the cap must still never push a symbol over 0.25 and the
        # leftover budget is left unallocated (cash) instead.
        frame = _returns_frame(60, vols={"A:long": 0.003, "A:short": 0.003, "B:long": 0.05})
        weights = compute_causal_risk_weights(
            frame, ("A:long", "A:short", "B:long"), _as_of(frame),
        )
        assert float(_symbol_sums(weights).max()) <= 0.25 + 1e-12
        assert float(weights.sum()) < 1.0

    def test_as_of_bar_and_future_returns_are_not_used(self) -> None:
        # Returns strictly before as_of only: injecting a huge spike exactly at
        # or after as_of must not change the computed weights.
        frame = _returns_frame(60, vols={"A:long": 0.005, "B:long": 0.020})
        as_of = _as_of(frame)
        baseline = compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of)

        spike = frame.copy()
        spike.loc[as_of, "A:long"] = 1.0
        spike.loc[as_of + pd.Timedelta(hours=4), "A:long"] = 1.0
        after = compute_causal_risk_weights(spike, tuple(spike.columns), as_of=as_of)
        assert baseline.equals(after)

    def test_insufficient_history_returns_all_zero(self) -> None:
        # SC-SGV2-06: a completed history of only 20 days is not a full month,
        # so every component weight is zero (the candidate stays in cash).
        frame = _returns_frame(20, vols={"A:long": 0.005, "B:long": 0.020})
        as_of = _as_of(frame, days_in=19)
        weights = compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of)
        assert (weights == 0.0).all()
        assert float(weights.sum()) == 0.0

    def test_no_history_before_as_of_returns_all_zero(self) -> None:
        frame = _returns_frame(60, vols={"A:long": 0.005, "B:long": 0.020})
        as_of = frame.index[0]
        weights = compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of)
        assert (weights == 0.0).all()

    def test_all_zero_volatility_returns_all_zero(self) -> None:
        frame = _returns_frame(60, vols={"A:long": 0.005, "B:long": 0.020})
        as_of = _as_of(frame)
        frame.loc[frame.index < as_of, :] = 0.0
        weights = compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of)
        assert (weights == 0.0).all()

    def test_non_finite_or_zero_volatility_component_gets_zero(self) -> None:
        # A component with no variation in its completed window has zero
        # volatility and must receive zero weight while the remaining symbols
        # are renormalized to sum to one.
        frame = _returns_frame(
            60, vols={
                "A:long": 0.005, "A:short": 0.005, "B:long": 0.020,
                "C:long": 0.020, "D:long": 0.020,
            },
        )
        as_of = _as_of(frame)
        frame.loc[frame.index < as_of, "A:short"] = 0.0
        weights = compute_causal_risk_weights(
            frame, ("A:long", "A:short", "B:long", "C:long", "D:long"), as_of=as_of,
        )
        assert float(weights["A:short"]) == 0.0
        assert abs(float(weights.sum()) - 1.0) < 1e-12
        assert float(weights["A:long"]) == pytest.approx(0.25, abs=1e-9)

    def test_vectorized_series_matches_contract_function(self) -> None:
        # the sleeve portfolio uses the vectorized weight series; it must be
        # row-wise identical to the per-bar contract function.
        frame = _returns_frame(60, vols={
            "A:long": 0.005, "A:short": 0.005, "B:long": 0.006,
            "B:short": 0.006, "C:long": 0.008, "D:long": 0.010, "E:long": 0.012,
        })
        active = tuple(frame.columns)
        series = _causal_weight_series(frame, active, history_days=30, max_symbol_weight=0.25)
        for i in range(len(frame)):
            expected = compute_causal_risk_weights(frame, active, as_of=frame.index[i])
            assert series.iloc[i].index.equals(expected.index)
            assert np.allclose(
                series.iloc[i].to_numpy(), expected.to_numpy(), rtol=1e-6, atol=1e-12,
            ), f"row {i} disagrees"

    def test_validation(self) -> None:
        frame = _returns_frame(60, vols={"A:long": 0.005, "B:long": 0.020})
        as_of = _as_of(frame)
        with pytest.raises(ValueError, match="active_components must be non-empty"):
            compute_causal_risk_weights(frame, (), as_of=as_of)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_causal_risk_weights(
                pd.DataFrame({"A:long": [0.01, 0.02]}), ("A:long",), as_of=as_of,
            )
        with pytest.raises(ValueError, match="missing from returns"):
            compute_causal_risk_weights(frame, ("A:long", "ZZZ:short"), as_of=as_of)
        with pytest.raises(ValueError, match="history_days"):
            compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of, history_days=0)
        with pytest.raises(ValueError, match="max_symbol_weight"):
            compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of, max_symbol_weight=1.5)
        with pytest.raises(ValueError, match="tz-aware"):
            compute_causal_risk_weights(frame, tuple(frame.columns), as_of=as_of.replace(tzinfo=None))
        with pytest.raises(ValueError, match=r"pd\.Timestamp"):
            compute_causal_risk_weights(frame, tuple(frame.columns), as_of="2024-02-15")  # type: ignore[arg-type]
        naive_frame = frame.copy()
        naive_frame.index = naive_frame.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-naive"):
            compute_causal_risk_weights(naive_frame, tuple(frame.columns), as_of=as_of)
        with pytest.raises(ValueError, match="monotonic"):
            compute_causal_risk_weights(frame.iloc[::-1], tuple(frame.columns), as_of=as_of)
