from __future__ import annotations

import pytest

from src.research.sleeve_blend.contracts import FixedSleevePortfolioSpec


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
