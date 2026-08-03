from __future__ import annotations

import pandas as pd
import pytest

from src.research.sleeve_blend.contracts import (
    DirectionalSleeveSpec,
    FixedSleevePortfolioSpec,
    FIXED_DIRECTIONAL_SYMBOLS,
)


def test_fixed_sleeve_portfolio_spec_defaults_are_frozen() -> None:
    spec = FixedSleevePortfolioSpec(symbols=("BTCUSDT", "ETHUSDT"))
    assert spec.mdd_budget_fraction == 0.85
    with pytest.raises(AttributeError):
        spec.mdd_budget_fraction = 0.5  # type: ignore[misc]


def test_fixed_sleeve_portfolio_spec_rejects_single_symbol() -> None:
    with pytest.raises(ValueError, match="at least 2 symbols"):
        FixedSleevePortfolioSpec(symbols=("BTCUSDT",))


def test_fixed_sleeve_portfolio_spec_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        FixedSleevePortfolioSpec(symbols=("BTCUSDT", "BTCUSDT"))


def test_fixed_sleeve_portfolio_spec_rejects_out_of_range_budget() -> None:
    with pytest.raises(ValueError, match="mdd_budget_fraction"):
        FixedSleevePortfolioSpec(symbols=("A", "B"), mdd_budget_fraction=0.0)
    with pytest.raises(ValueError, match="mdd_budget_fraction"):
        FixedSleevePortfolioSpec(symbols=("A", "B"), mdd_budget_fraction=1.0)
    with pytest.raises(ValueError, match="mdd_budget_fraction"):
        FixedSleevePortfolioSpec(symbols=("A", "B"), mdd_budget_fraction=1.5)


def test_pbgt_01_core5_universe_accepts_only_exact_declared_membership() -> None:
    """PBGT-01: core5_v1 accepts only its exact declared order; arbitrary,
    duplicate, reordered, or substituted tuples raise ValueError."""
    from src.research.sleeve_blend.contracts import BlendUniverseSpec

    spec = BlendUniverseSpec()
    assert spec.universe_id == "core5_v1"
    assert spec.symbols == ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")

    BlendUniverseSpec(
        universe_id="core5_v1",
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
    )
    for bad in (
        (),
        ("BTCUSDT",),
        ("BTCUSDT", "BTCUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        ("ETHUSDT", "BTCUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "SOLUSDT"),
    ):
        with pytest.raises(ValueError, match="exact declared core5_v1 order"):
            BlendUniverseSpec(universe_id="core5_v1", symbols=bad)
    with pytest.raises(ValueError, match="unknown universe_id"):
        BlendUniverseSpec(universe_id="core9_v9")


def test_pbgt_01_tournament_request_validates_qualification_interval() -> None:
    """An empty qualification interval is a contract violation."""
    from src.research.sleeve_blend.contracts import PortfolioBlendTournamentRequest

    with pytest.raises(ValueError, match="qualification_interval must not be empty"):
        PortfolioBlendTournamentRequest(qualification_interval="")
    with pytest.raises(ValueError, match="Invalid frequency"):
        PortfolioBlendTournamentRequest(qualification_interval="not-a-frequency")
    with pytest.raises(ValueError, match="initial_equity"):
        PortfolioBlendTournamentRequest(initial_equity=0.0)
    with pytest.raises(ValueError, match="discovery_end must be tz-aware UTC"):
        PortfolioBlendTournamentRequest(
            discovery_end=pd.Timestamp("2024-12-31 23:59:59"),
        )


def test_causal_leverage_spec_contract_ranges() -> None:
    from src.research.sleeve_blend.contracts import CausalLeverageSpec

    spec = CausalLeverageSpec()
    assert spec.lookback_days == 365
    assert spec.risk_budget_fraction == 0.85
    assert spec.max_gross_leverage == 3.0
    with pytest.raises(ValueError, match="lookback_days"):
        CausalLeverageSpec(lookback_days=0)
    with pytest.raises(ValueError, match="risk_budget_fraction"):
        CausalLeverageSpec(risk_budget_fraction=0.0)
    with pytest.raises(ValueError, match="risk_budget_fraction"):
        CausalLeverageSpec(risk_budget_fraction=1.0)
    with pytest.raises(ValueError, match="max_gross_leverage"):
        CausalLeverageSpec(max_gross_leverage=0.5)


def test_directional_sleeve_spec_freezes_symbols_and_limits_settings() -> None:
    spec = DirectionalSleeveSpec()
    assert spec.symbols == FIXED_DIRECTIONAL_SYMBOLS
    assert spec.history_days == 30
    assert spec.max_symbol_weight == 0.25

    with pytest.raises(ValueError, match="fixed directional set"):
        DirectionalSleeveSpec(symbols=("BTCUSDT", "ETHUSDT"))
    with pytest.raises(ValueError, match="history_days"):
        DirectionalSleeveSpec(history_days=0)
    with pytest.raises(ValueError, match="max_symbol_weight"):
        DirectionalSleeveSpec(max_symbol_weight=0.0)


def test_backtest_facade_preserves_canonical_object_identity() -> None:
    """The backtest facade re-exports the canonical module objects."""
    from src.research.sleeve_blend import backtest as facade
    from src.research.sleeve_blend import directional, fixed, weights

    assert facade.run_fixed_sleeve_portfolio is fixed.run_fixed_sleeve_portfolio
    assert facade.run_fixed_sleeve_portfolio_calibrated is (
        fixed.run_fixed_sleeve_portfolio_calibrated
    )
    assert facade.run_fixed_sleeve_portfolio_with_leverage is (
        fixed.run_fixed_sleeve_portfolio_with_leverage
    )
    assert facade.compute_causal_risk_weights is weights.compute_causal_risk_weights
    assert facade.component_labels is weights.component_labels
    assert facade.symbol_of_component is weights.symbol_of_component
    assert facade.run_directional_sleeve_portfolio is (
        directional.run_directional_sleeve_portfolio
    )
    assert facade.run_directional_sleeve_portfolio_with_weights is (
        directional.run_directional_sleeve_portfolio_with_weights
    )
    assert facade.run_directional_sleeve_portfolio_fixed_weights is (
        directional.run_directional_sleeve_portfolio_fixed_weights
    )
