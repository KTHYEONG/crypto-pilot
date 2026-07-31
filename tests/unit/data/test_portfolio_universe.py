from __future__ import annotations

import pandas as pd
import pytest

from src.core.types import PortfolioSpec
from src.data.portfolio_universe import select_liquid_universe


def _frame(quote_vol: float, n: int = 4000, start: str = "2023-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "quote_vol": float(quote_vol),
    }, index=idx)


@pytest.fixture
def six_liquid_frames() -> dict[str, pd.DataFrame]:
    vols = {"A": 600.0, "B": 500.0, "C": 400.0, "D": 300.0, "E": 200.0, "F": 100.0}
    return {symbol: _frame(vol) for symbol, vol in vols.items()}


class TestSelectLiquidUniverse:
    def test_selects_top_five_by_pre_asof_quote_volume(
        self, six_liquid_frames: dict[str, pd.DataFrame],
    ) -> None:
        # SC-PORT-01: deterministic top-five by trailing liquidity, symbol tie-break.
        as_of = pd.Timestamp("2024-01-01", tz="UTC")
        result = select_liquid_universe(six_liquid_frames, as_of, PortfolioSpec())
        assert result == ("A", "B", "C", "D", "E")
        assert select_liquid_universe(six_liquid_frames, as_of, PortfolioSpec()) == result

    def test_future_quote_volume_does_not_change_selection(
        self, six_liquid_frames: dict[str, pd.DataFrame],
    ) -> None:
        # SC-PORT-01: future volume must never leak into a historical decision.
        as_of = pd.Timestamp("2024-01-01", tz="UTC")
        spiked = six_liquid_frames["F"].copy()
        spiked.loc[spiked.index > as_of, "quote_vol"] = 1e12
        frames = {**six_liquid_frames, "F": spiked}
        assert select_liquid_universe(frames, as_of, PortfolioSpec()) == ("A", "B", "C", "D", "E")

    def test_returns_fewer_than_five_when_universe_is_smaller(
        self, six_liquid_frames: dict[str, pd.DataFrame],
    ) -> None:
        frames = {s: six_liquid_frames[s] for s in ("A", "B", "C")}
        result = select_liquid_universe(frames, pd.Timestamp("2024-01-01", tz="UTC"), PortfolioSpec())
        assert result == ("A", "B", "C")

    def test_gapped_symbol_is_excluded(self, six_liquid_frames: dict[str, pd.DataFrame]) -> None:
        # SC-PORT-02: a bar gap disqualifies the symbol for entry.
        gapped = six_liquid_frames["B"].drop(six_liquid_frames["B"].index[1000])
        frames = {**six_liquid_frames, "B": gapped}
        result = select_liquid_universe(frames, pd.Timestamp("2024-01-01", tz="UTC"), PortfolioSpec())
        assert "B" not in result
        assert result == ("A", "C", "D", "E", "F")

    def test_malformed_or_insufficient_history_symbols_are_excluded(
        self, six_liquid_frames: dict[str, pd.DataFrame],
    ) -> None:
        frames = dict(six_liquid_frames)
        frames["G"] = six_liquid_frames["A"].tz_localize(None)
        frames["H"] = six_liquid_frames["B"].drop(columns=["quote_vol"])
        frames["I"] = _frame(1e9, start="2024-06-01")
        result = select_liquid_universe(frames, pd.Timestamp("2024-01-01", tz="UTC"), PortfolioSpec())
        assert result == ("A", "B", "C", "D", "E")

    def test_naive_as_of_raises(self, six_liquid_frames: dict[str, pd.DataFrame]) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            select_liquid_universe(six_liquid_frames, pd.Timestamp("2024-01-01"), PortfolioSpec())

    def test_invalid_portfolio_spec_rejected(self) -> None:
        with pytest.raises(ValueError, match="universe_size"):
            PortfolioSpec(universe_size=4, max_positions=5)
        with pytest.raises(ValueError, match="liquidity_lookback_days"):
            PortfolioSpec(liquidity_lookback_days=0)
