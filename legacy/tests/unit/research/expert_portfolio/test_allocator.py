from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.expert_portfolio.allocator import (
    _causal_lcb_weight_series,
    causal_block_aware_inflation,
    compute_causal_lcb_weights,
)
from src.research.expert_portfolio.models import ExpertDefinition, ExpertPortfolioSpec


def _expert(
    expert_id: str,
    family: str,
    symbols: tuple[str, ...],
    code_hash: str = "hash",
) -> ExpertDefinition:
    return ExpertDefinition(expert_id, "return_source", family, symbols, "run_backtest", code_hash)


def _panel(
    n: int,
    experts: list[ExpertDefinition],
    mean: float = 0.001,
    vol: float = 0.005,
    seed: int = 7,
) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for expert in experts:
        data[expert.expert_id] = rng.normal(mean, vol, n)
    return pd.DataFrame(data, index=idx)


def _cash_weights(spec: ExpertPortfolioSpec) -> pd.Series:
    return pd.Series(
        {e.expert_id: 0.0 for e in spec.experts} | {"CASH": 1.0},
        dtype=np.float64,
    )


def _weight_columns(spec: ExpertPortfolioSpec) -> list[str]:
    return [e.expert_id for e in spec.experts] + ["CASH"]


class TestNonPositiveLcbReturnsCashOnly:
    def test_insufficient_history_returns_cash_only(self) -> None:
        spec = ExpertPortfolioSpec(
            experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f1", ("S2",))),
            min_history_bars=30,
        )
        panel = _panel(20, list(spec.experts), mean=0.001)
        weights = compute_causal_lcb_weights(
            panel, spec, as_of=panel.index[19], previous_weights=_cash_weights(spec),
        )
        assert float(weights.drop("CASH").sum()) == 0.0
        assert float(weights["CASH"]) == pytest.approx(1.0)


def test_non_positive_lcb_returns_cash_only() -> None:
    # EP-02: no eligible positive lower-confidence evidence can produce risky
    # exposure; the target is exact cash-only.
    spec = ExpertPortfolioSpec(
        experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f1", ("S2",))),
        min_history_bars=20,
    )
    panel = _panel(181, list(spec.experts), mean=-0.002)
    weights = compute_causal_lcb_weights(
        panel, spec, as_of=panel.index[180], previous_weights=_cash_weights(spec),
    )
    assert float(weights.drop("CASH").sum()) == 0.0
    assert float(weights["CASH"]) == pytest.approx(1.0)


class TestNoLookahead:
    def test_future_returns_do_not_change_decision_weights(self) -> None:
        # EP-01: mutating the current or any future bar's return must not change
        # the target decided at bar t (only completed history is causal).
        spec = ExpertPortfolioSpec(
            experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f2", ("S2",))),
            min_history_bars=20,
        )
        panel = _panel(181, list(spec.experts), mean=0.001)
        series = _causal_lcb_weight_series(panel, spec)

        spike = panel.copy()
        spike.loc[panel.index[175:], "e1"] = 1.0
        after = _causal_lcb_weight_series(spike, spec)
        for t in range(0, 175):
            assert np.allclose(
                after.iloc[t].to_numpy(), series.iloc[t].to_numpy(),
                rtol=1e-9, atol=1e-12,
            ), f"row {t} changed after future mutation"

    def test_vectorized_series_matches_contract_function(self) -> None:
        # the master backtest uses the vectorized series; it must be row-wise
        # identical to the per-bar contract function.
        spec = ExpertPortfolioSpec(
            experts=(
                _expert("e1", "f1", ("S1",)),
                _expert("e2", "f1", ("S1",)),
                _expert("e3", "f2", ("S2",)),
            ),
            family_exposure_limit=0.5,
            symbol_exposure_limit=0.5,
            min_history_bars=20,
        )
        panel = _panel(181, list(spec.experts), mean=0.001)
        series = _causal_lcb_weight_series(panel, spec)
        previous = _cash_weights(spec)
        for t in range(120, 181):
            row = compute_causal_lcb_weights(
                panel, spec, as_of=panel.index[t], previous_weights=previous,
            )
            assert row.index.tolist() == _weight_columns(spec)
            assert np.allclose(row.to_numpy(), series.iloc[t].to_numpy(), rtol=1e-9, atol=1e-12)


class TestConstrainedAllocation:
    def test_family_and_symbol_budgets_are_honoured(self) -> None:
        # EP-03: three families with an overlapping underlying symbol share the
        # exposure budget; every risky weight is finite/non-negative and no
        # family or symbol aggregate exceeds its pre-registered limit.
        spec = ExpertPortfolioSpec(
            experts=(
                _expert("e1", "f1", ("S1",)),
                _expert("e2", "f1", ("S1",)),
                _expert("e3", "f2", ("S1", "S2")),
                _expert("e4", "f3", ("S2",)),
            ),
            family_exposure_limit=0.5,
            symbol_exposure_limit=0.5,
            gross_exposure=1.0,
            min_history_bars=20,
        )
        panel = _panel(181, list(spec.experts), mean=0.001, seed=3)
        weights = compute_causal_lcb_weights(
            panel, spec, as_of=panel.index[180], previous_weights=_cash_weights(spec),
        )
        risky = weights.drop("CASH")
        assert np.isfinite(risky.to_numpy()).all()
        assert (risky >= 0.0).all()
        assert float(risky.sum()) <= spec.gross_exposure + 1e-12
        for family in {"f1", "f2", "f3"}:
            members = [e.expert_id for e in spec.experts if e.family == family]
            assert float(risky[members].sum()) <= spec.family_exposure_limit + 1e-12
        for symbol in {"S1", "S2"}:
            members = [e.expert_id for e in spec.experts if symbol in e.symbols]
            assert float(risky[members].sum()) <= spec.symbol_exposure_limit + 1e-12
        assert float(weights["CASH"]) == pytest.approx(
            spec.gross_exposure - float(risky.sum()), abs=1e-9,
        )

    def test_weak_evidence_is_never_scaled_up(self) -> None:
        # a raw allocation below gross exposure must not be leveraged up to the
        # budget; the surplus stays in cash.
        spec = ExpertPortfolioSpec(
            experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f2", ("S2",))),
            min_history_bars=20,
        )
        panel = _panel(181, list(spec.experts), mean=0.001, seed=11)
        weights = compute_causal_lcb_weights(
            panel, spec, as_of=panel.index[180], previous_weights=_cash_weights(spec),
        )
        total = float(weights.drop("CASH").sum())
        assert 0.0 < total <= 1.0
        assert float(weights["CASH"]) == pytest.approx(1.0 - total, abs=1e-9)


class TestBlockAwareInflation:
    def test_white_noise_inflates_to_about_one(self) -> None:
        # the Bartlett/Newey-West factor for independent returns stays near one
        # (it has a known small downward bias of a few percent); averaging over
        # many independent draws removes the per-draw sampling noise.
        values: list[float] = []
        for seed in range(40):
            rets = np.random.default_rng(seed).normal(0.0, 0.005, 300)
            values.append(float(causal_block_aware_inflation(rets)[250]))
        assert 0.70 <= float(np.mean(values)) <= 1.10

    def test_autocorrelated_series_inflates_above_one(self) -> None:
        # a random-walk-like series has strong positive autocorrelation, so the
        # block-aware variance factor must be materially above one.
        rng = np.random.default_rng(6)
        rets = np.cumsum(rng.normal(0.0, 0.005, 300))
        inflation = causal_block_aware_inflation(rets)
        assert float(inflation[250]) > 1.0
        assert float(inflation[50]) > 1.0


class TestValidation:
    def test_malformed_panel_raises(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1", "f1", ("S1",)),))
        as_of = pd.Timestamp("2024-01-01", tz="UTC")
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_causal_lcb_weights(
                pd.DataFrame({"e1": [0.01, 0.02]}), spec,
                as_of=as_of, previous_weights=_cash_weights(spec),
            )
        panel = _panel(50, list(spec.experts))
        with pytest.raises(ValueError, match="monotonic"):
            compute_causal_lcb_weights(
                panel.iloc[::-1], spec, as_of=panel.index[30], previous_weights=_cash_weights(spec),
            )
        dup = panel.copy()
        dup.index = [panel.index[0], *list(panel.index[:-1])]
        with pytest.raises(ValueError, match="duplicate"):
            compute_causal_lcb_weights(dup, spec, as_of=panel.index[30], previous_weights=_cash_weights(spec))

    def test_invalid_previous_weights_raises(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1", "f1", ("S1",)),))
        panel = _panel(50, list(spec.experts))
        with pytest.raises(ValueError, match="aligned"):
            compute_causal_lcb_weights(
                panel, spec, as_of=panel.index[30],
                previous_weights=pd.Series({"e1": 0.0, "WRONG": 1.0}),
            )
        with pytest.raises(ValueError, match="finite"):
            compute_causal_lcb_weights(
                panel, spec, as_of=panel.index[30],
                previous_weights=pd.Series({"e1": np.nan, "CASH": 1.0}),
            )
        with pytest.raises(ValueError, match="non-negative"):
            compute_causal_lcb_weights(
                panel, spec, as_of=panel.index[30],
                previous_weights=pd.Series({"e1": -0.1, "CASH": 1.0}),
            )

    def test_missing_expert_column_raises(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f2", ("S2",))))
        panel = _panel(50, list(spec.experts)).drop(columns=["e2"])
        with pytest.raises(ValueError, match="missing from"):
            compute_causal_lcb_weights(panel, spec, as_of=panel.index[30], previous_weights=_cash_weights(spec))

    def test_invalid_as_of_raises(self) -> None:
        spec = ExpertPortfolioSpec(experts=(_expert("e1", "f1", ("S1",)),))
        panel = _panel(50, list(spec.experts))
        with pytest.raises(ValueError, match="as_of"):
            compute_causal_lcb_weights(
                panel, spec, as_of=panel.index[30].replace(tzinfo=None),
                previous_weights=_cash_weights(spec),
            )
        with pytest.raises(ValueError, match="not in the component_returns index"):
            compute_causal_lcb_weights(
                panel, spec, as_of=panel.index[-1] + pd.Timedelta(hours=4),
                previous_weights=_cash_weights(spec),
            )

    def test_nan_history_fails_closed_to_cash(self) -> None:
        # EP-05: an expert with invalid (non-finite) completed data receives
        # zero weight for every later decision bar while a healthy expert with
        # the same drift keeps allocating.
        spec = ExpertPortfolioSpec(
            experts=(_expert("e1", "f1", ("S1",)), _expert("e2", "f2", ("S2",))),
            min_history_bars=20,
        )
        panel = _panel(100, list(spec.experts), mean=0.002)
        panel.loc[panel.index[50], "e1"] = np.nan
        series = _causal_lcb_weight_series(panel, spec)
        assert float(series.loc[panel.index[80], "e1"]) == 0.0
        assert float(series.loc[panel.index[80], "e2"]) > 0.0
