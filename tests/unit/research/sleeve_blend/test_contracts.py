from __future__ import annotations

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
