from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.mhs.panel import build_uniform_grid, liquid_half_eligibility, load_base_panel, partition_symbols
from src.research.universe.pit_universe import symbol_partition


class TestBasePanelProjection:
    """MHS-PROJECTION-PUSHDOWN: the inclusive [start, end] predicate is pushed
    into the pyarrow read so out-of-range rows are never materialized, while
    grid, PIT partition, duplicate-last, and min_bars results stay identical."""

    def _write_long(self, tmp_path: Path, n: int = 3000) -> pd.DatetimeIndex:
        directory = tmp_path / "1h"
        directory.mkdir(parents=True)
        ts = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
        epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        for sym in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
            prices = pd.Series(100.0, index=ts).cumsum() / 100.0
            pd.DataFrame(
                {"timestamp": epoch, "close": prices, "quote_vol": [1000.0] * n},
            ).to_parquet(directory / f"{sym}.parquet")
        return ts

    def test_window_projection_matches_full_load_slice(self, tmp_path: Path) -> None:
        ts = self._write_long(tmp_path)
        start = ts[500]
        end = ts[1200]
        window = load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "quote_vol"),
            start=start, end=end, partition="all", min_bars=1,
        )
        full = load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1,
        )
        sliced = {c: full[c].loc[start:end] for c in ("close", "quote_vol")}
        assert set(window) == {"close", "quote_vol"}
        assert window["close"].index.equals(build_uniform_grid(start, end, "1h"))
        for c in ("close", "quote_vol"):
            assert list(window[c].columns) == list(sliced[c].columns)
            assert window[c].notna().sum().eq(sliced[c].notna().sum()).all()
            assert window[c].loc[start:end].reindex(sliced[c].index).equals(sliced[c])

    def test_predicate_is_pushed_into_parquet_read(self, tmp_path: Path, monkeypatch) -> None:
        import pyarrow.parquet as pq

        ts = self._write_long(tmp_path)
        start = ts[100]
        end = ts[900]
        captured: list[list] = []

        real_read = pq.read_table

        def fake_read(path, columns=None, filters=None):
            captured.append(filters)
            return real_read(path, columns=columns, filters=filters)

        monkeypatch.setattr(pq, "read_table", fake_read)
        load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "quote_vol"),
            start=start, end=end, partition="all", min_bars=1,
        )
        assert captured
        start_ms = int(start.value // 1_000_000)
        end_ms = int(end.value // 1_000_000)
        for filters in captured:
            assert filters is not None
            assert filters == [[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]]

    def test_inclusive_bounds_and_duplicate_last_preserved(self, tmp_path: Path) -> None:
        directory = tmp_path / "1h"
        directory.mkdir(parents=True)
        ts = pd.date_range("2021-01-01", periods=6, freq="1h", tz="UTC")
        epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        # duplicate last bar for BBBUSDT: only the final duplicate survives.
        pd.DataFrame(
            {
                "timestamp": [*list(epoch), int(epoch[-1])],
                "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0],
                "quote_vol": [10.0] * 7,
            },
        ).to_parquet(directory / "AAAUSDT.parquet")
        panel = load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close",),
            start=ts[0], end=ts[-1], partition="all", min_bars=1,
        )
        assert panel["close"]["AAAUSDT"].loc[ts[-1]] == 99.0
        assert panel["close"]["AAAUSDT"].loc[ts[0]] == 1.0

    def test_min_bars_rule_applies_to_projected_rows(self, tmp_path: Path) -> None:
        directory = tmp_path / "1h"
        directory.mkdir(parents=True)
        ts = pd.date_range("2021-01-01", periods=50, freq="1h", tz="UTC")
        epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
        for sym in ("AAAUSDT", "BBBUSDT"):
            pd.DataFrame(
                {"timestamp": epoch, "close": [1.0] * 50, "quote_vol": [10.0] * 50},
            ).to_parquet(directory / f"{sym}.parquet")
        window = load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close",),
            start=ts[10], end=ts[40], partition="all", min_bars=25,
        )
        assert list(window["close"].columns) == ["AAAUSDT", "BBBUSDT"]
        with pytest.raises(ValueError, match="no symbol survived"):
            load_base_panel(
                root=str(tmp_path), interval="1h", columns=("close",),
                start=ts[10], end=ts[40], partition="all", min_bars=32,
            )


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


class TestFillMarkParityMask:
    """SCENARIO_MHS_FILL_MARK_PARITY_01: fill_mark_parity_mask correctness."""

    def test_divergent_cells_false(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
        cols = ["A", "B", "C"]
        fill = pd.DataFrame(
            {"A": [1.0, 1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.19, 1.19], "C": [1.0, 1.0, 1.0, 1.0]},
            index=idx, columns=cols,
        )
        mark = pd.DataFrame(
            {"A": [1.0, 1.0, 1.0, 1.0], "B": [1.0, 1.0, 0.0835, 0.0835], "C": [1.0, 1.0, 1.0, 1.0]},
            index=idx, columns=cols,
        )
        mask = fill_mark_parity_mask(fill, mark)
        assert mask.dtypes.eq(bool).all()
        # B rows 2-3 are divergent: |log(1.19/0.0835)| = |log(14.25)| ~ 2.657 >> 0.0488
        assert mask.loc[idx[2], "B"] == False  # noqa: E712
        assert mask.loc[idx[3], "B"] == False  # noqa: E712
        # All other cells are True
        assert mask.loc[idx[0], "B"] == True  # noqa: E712
        assert mask.loc[idx[1], "B"] == True  # noqa: E712
        assert mask["A"].all()
        assert mask["C"].all()

    def test_identical_panels_all_true(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        data = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0]}, index=idx)
        mask = fill_mark_parity_mask(data, data.copy())
        assert mask.all().all()

    def test_nan_mark_true_fail_open(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        fill = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
        mark = pd.DataFrame({"A": [float("nan"), 1.0]}, index=idx)
        mask = fill_mark_parity_mask(fill, mark)
        assert mask["A"].all()

    def test_zero_mark_true_fail_open(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        fill = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
        mark = pd.DataFrame({"A": [0.0, 1.0]}, index=idx)
        mask = fill_mark_parity_mask(fill, mark)
        assert mask["A"].all()

    def test_negative_mark_true_fail_open(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        fill = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
        mark = pd.DataFrame({"A": [-1.0, 1.0]}, index=idx)
        mask = fill_mark_parity_mask(fill, mark)
        assert mask["A"].all()

    def test_max_log_divergence_zero_raises(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        data = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
        with pytest.raises(ValueError, match="max_log_divergence"):
            fill_mark_parity_mask(data, data.copy(), max_log_divergence=0.0)

    def test_max_log_divergence_negative_raises(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
        data = pd.DataFrame({"A": [1.0, 2.0]}, index=idx)
        with pytest.raises(ValueError, match="max_log_divergence"):
            fill_mark_parity_mask(data, data.copy(), max_log_divergence=-0.1)

    def test_index_mismatch_raises(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx_a = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        idx_b = pd.date_range("2021-01-02", periods=3, freq="1h", tz="UTC")
        fill = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx_a)
        mark = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx_b)
        with pytest.raises(ValueError, match="index"):
            fill_mark_parity_mask(fill, mark)

    def test_column_mismatch_raises(self) -> None:
        from src.mhs.panel import fill_mark_parity_mask

        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        fill = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
        mark = pd.DataFrame({"B": [1.0, 2.0, 3.0]}, index=idx)
        with pytest.raises(ValueError, match="column"):
            fill_mark_parity_mask(fill, mark)
