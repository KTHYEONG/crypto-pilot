from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.mhs.panel import build_uniform_grid, liquid_half_eligibility, load_base_panel, partition_symbols
from src.research.universe.pit_universe import symbol_partition


def _write_symbols(directory: Path, symbols: list[str], periods: int) -> pd.DatetimeIndex:
    ts = pd.date_range("2021-01-01", periods=periods, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    for i, sym in enumerate(symbols):
        prices = pd.Series(100.0 + i, index=ts).cumsum() / 100.0
        pd.DataFrame(
            {
                "timestamp": epoch,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "quote_vol": [1000.0 + i] * periods,
            },
        ).to_parquet(directory / f"{sym}.parquet")
    return ts


class TestBuildUniformGrid:
    def test_inclusive_utc_grid(self) -> None:
        grid = build_uniform_grid(
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2021-01-02", tz="UTC"),
            "1h",
        )
        assert len(grid) == 25
        assert str(grid.tz) == "UTC"
        assert grid[0] == pd.Timestamp("2021-01-01", tz="UTC")
        assert grid[-1] == pd.Timestamp("2021-01-02", tz="UTC")

    def test_fails_closed_on_tz_naive(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            build_uniform_grid(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02"), "1h")

    def test_fails_closed_on_empty_range(self) -> None:
        with pytest.raises(ValueError, match="must be < end"):
            build_uniform_grid(
                pd.Timestamp("2021-01-02", tz="UTC"),
                pd.Timestamp("2021-01-01", tz="UTC"),
                "1h",
            )


class TestPartitionSymbols:
    """MHS-03-PARTITION-DISJOINT: dev/holdout are disjoint and match symbol_partition."""

    SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")

    def test_dev_holdout_disjoint_and_total(self) -> None:
        dev = partition_symbols(self.SYMBOLS, "dev")
        hold = partition_symbols(self.SYMBOLS, "holdout")
        assert partition_symbols(self.SYMBOLS, "all") == list(self.SYMBOLS)
        assert sorted(dev + hold) == sorted(self.SYMBOLS)
        assert set(dev).isdisjoint(hold)
        assert all(symbol_partition(s) == "dev" for s in dev)
        assert all(symbol_partition(s) == "holdout" for s in hold)

    def test_unknown_partition_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="unknown partition"):
            partition_symbols(self.SYMBOLS, "train")

    def test_order_preserving(self) -> None:
        dev = partition_symbols(self.SYMBOLS, "dev")
        assert dev == [s for s in self.SYMBOLS if symbol_partition(s) == "dev"]


class TestLoadBasePanel:
    """MHS-02-PANEL-NO-SURVIVORSHIP: delisted symbols keep NaN after their last bar."""

    def test_keeps_symbol_that_stops_mid_window(self, tmp_path: Path) -> None:
        directory = tmp_path / "1h"
        directory.mkdir(parents=True)
        symbols = ["AAAUSDT", "BBBUSDT"]
        ts = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        pd.DataFrame(
            {"timestamp": epoch, "close": [1.0, 2.0, 3.0, 4.0], "quote_vol": [10.0] * 4},
        ).to_parquet(directory / "AAAUSDT.parquet")
        # BBBUSDT stops after bar 1 (delisted mid-window).
        pd.DataFrame(
            {"timestamp": epoch[:2], "close": [4.0, 3.0], "quote_vol": [10.0, 10.0]},
        ).to_parquet(directory / "BBBUSDT.parquet")

        panel = load_base_panel(
            root=str(tmp_path),
            interval="1h",
            columns=("close", "quote_vol"),
            start=ts[0],
            end=ts[-1],
            partition="all",
            min_bars=1,
        )
        assert set(panel) == {"close", "quote_vol"}
        assert panel["close"].shape == (4, 2)
        assert list(panel["close"].columns) == ["AAAUSDT", "BBBUSDT"]
        assert panel["close"].index.equals(ts)
        assert panel["close"]["AAAUSDT"].tolist() == [1.0, 2.0, 3.0, 4.0]
        # BBBUSDT is NOT forward-filled after its last bar.
        assert pd.isna(panel["close"].loc[ts[2], "BBBUSDT"])
        assert pd.isna(panel["close"].loc[ts[3], "BBBUSDT"])

    def test_no_symbol_survives_raises(self, tmp_path: Path) -> None:
        directory = tmp_path / "1h"
        directory.mkdir(parents=True)
        ts = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        pd.DataFrame({"timestamp": epoch, "close": [1.0] * 4, "quote_vol": [10.0] * 4}).to_parquet(
            directory / "ZZZUSDT.parquet",
        )
        with pytest.raises(ValueError, match="no symbol survived"):
            load_base_panel(
                root=str(tmp_path), interval="1h", columns=("close", "quote_vol"),
                start=ts[0], end=ts[-1], partition="dev", min_bars=4,
            )

    def test_drops_symbol_below_min_bars(self, tmp_path: Path) -> None:
        directory = tmp_path / "1h"
        directory.mkdir(parents=True)
        ts = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        for sym in ("AAAUSDT", "BBBUSDT"):
            pd.DataFrame({"timestamp": epoch, "close": [1.0] * 4, "quote_vol": [10.0] * 4}).to_parquet(
                directory / f"{sym}.parquet",
            )
        panel = load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close",),
            start=ts[0], end=ts[-1], partition="all", min_bars=4,
        )
        assert list(panel["close"].columns) == ["AAAUSDT", "BBBUSDT"]
        with pytest.raises(ValueError, match="no symbol survived"):
            load_base_panel(
                root=str(tmp_path), interval="1h", columns=("close",),
                start=ts[0], end=ts[-1], partition="all", min_bars=10,
            )


class TestLiquidHalfEligibility:
    """MHS-02-PANEL-NO-SURVIVORSHIP: eligibility is PIT and needs a full trailing window."""

    def test_pit_trailing_median_eligibility(self) -> None:
        quote = pd.DataFrame(
            {
                "A": [1.0, 1.0, 1.0, 1.0],
                "B": [10.0, 10.0, 10.0, 10.0],
                "C": [float("nan"), float("nan"), 100.0, 100.0],
            },
        )
        eligible = liquid_half_eligibility(quote, lookback_bars=2, min_history_bars=2)
        assert eligible.iloc[0].eq(False).all()
        assert bool(eligible.loc[1, "B"])
        assert not bool(eligible.loc[1, "A"])
        assert not bool(eligible.loc[1, "C"])
        assert bool(eligible.loc[3, "C"])

    def test_future_volume_cannot_change_earlier_eligibility(self) -> None:
        quote = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0], "B": [10.0, 10.0, 10.0, 10.0]})
        before = liquid_half_eligibility(quote, lookback_bars=2, min_history_bars=2)
        quote.loc[3, "B"] = 0.0
        after = liquid_half_eligibility(quote, lookback_bars=2, min_history_bars=2)
        assert before.iloc[1].equals(after.iloc[1])

    def test_invalid_window_fails_closed(self) -> None:
        quote = pd.DataFrame({"A": [1.0, 1.0]})
        with pytest.raises(ValueError, match="lookback_bars"):
            liquid_half_eligibility(quote, lookback_bars=1, min_history_bars=2)


def test_synthetic_panel_writes_epoch_via_timedelta(tmp_path: Path) -> None:
    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = _write_symbols(directory, ["AAAUSDT", "BBBUSDT"], 4)
    panel = load_base_panel(
        root=str(tmp_path), interval="1h", columns=("close", "quote_vol"),
        start=ts[0], end=ts[-1], partition="all", min_bars=1,
    )
    assert os.path.exists(directory / "AAAUSDT.parquet")
