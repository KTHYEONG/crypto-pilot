from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.mhs.panel import build_uniform_grid, liquid_half_eligibility, load_base_panel, partition_symbols
from src.quant.universe.pit_universe import symbol_partition


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


def test_load_base_panel_pit_min_history_admits_eligible_short_history_symbol(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel
    from src.mhs.params import PANEL_MIN_HISTORY_BARS

    assert PANEL_MIN_HISTORY_BARS == 720
    start = pd.Timestamp("2024-01-01", tz="UTC")
    root = tmp_path / "ohlcv"
    (root / "1h").mkdir(parents=True)
    for symbol, n_bars in (("SYMAUSDT", 800), ("SYMBUSDT", 700)):
        idx = pd.date_range(start, periods=n_bars, freq="1h", tz="UTC")
        frame = pd.DataFrame({"timestamp": np.array([ts.value // 1_000_000 for ts in idx], dtype="int64"), "close": np.linspace(1.0, 2.0, n_bars)})
        frame.to_parquet(root / "1h" / f"{symbol}.parquet", index=False)
    end = start + pd.Timedelta(hours=799)
    panel = load_base_panel(str(root), "1h", ("close",), start, end, partition="all", min_bars=PANEL_MIN_HISTORY_BARS)
    assert list(panel["close"].columns) == ["SYMAUSDT"]


def test_load_panel_stage_uses_pit_min_history_bars(monkeypatch) -> None:
    import types
    import pandas as pd
    import pytest
    import src.mhs.pipeline.stages.panel as stage
    from src.mhs.params import PANEL_MIN_HISTORY_BARS
    from src.mhs.pipeline.config import MhsRunConfig

    captured: dict[str, object] = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-load")

    monkeypatch.setattr(stage, "load_base_panel", fake_load)
    ctx = types.SimpleNamespace(
        config=MhsRunConfig(data_root="/nonexistent-root", log_run=False),
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-03-01", tz="UTC"),
    )
    with pytest.raises(RuntimeError, match="stop-after-load"):
        stage.load_panel(ctx, None)
    assert captured["min_bars"] == PANEL_MIN_HISTORY_BARS


def test_load_feature_panels_uses_pit_min_history_bars(monkeypatch) -> None:
    import pandas as pd
    import src.mhs.evaluation.diagnostics as diagnostics
    from src.mhs.params import PANEL_MIN_HISTORY_BARS

    grid = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    captured: dict[str, object] = {}

    def fake_load(root, interval, columns, start, end, partition="dev", min_bars=0):
        captured["min_bars"] = min_bars
        return {c: pd.DataFrame(1.0, index=grid, columns=["AAAUSDT"]) for c in columns}

    monkeypatch.setattr(diagnostics, "_available_panel_columns", lambda root, requested: ("close",))
    monkeypatch.setattr(diagnostics, "load_base_panel", fake_load)
    panels = diagnostics._load_feature_panels("/root", grid[0], grid[-1], grid, ["AAAUSDT"], columns=("close",))
    assert captured["min_bars"] == PANEL_MIN_HISTORY_BARS
    assert list(panels["close"].columns) == ["AAAUSDT"]


def test_load_base_panel_ignores_legacy_temp_artifacts(tmp_path) -> None:
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    frame = pd.DataFrame({"timestamp": epoch, "close": [1.0, 2.0, 3.0, 4.0], "quote_vol": [10.0] * 4})
    frame.to_parquet(directory / "AAAUSDT.parquet")
    frame.to_parquet(directory / "AAAUSDT.tmp.parquet")

    panel = load_base_panel(
        root=str(tmp_path), interval="1h", columns=("close", "quote_vol"),
        start=ts[0], end=ts[-1], partition="all", min_bars=1,
    )

    assert list(panel["close"].columns) == ["AAAUSDT"]


# --- auto appended from contract: signal_input_quarantine ---


def test_panel_quarantine_records_unique_symbols() -> None:
    from src.mhs.panel import PanelQuarantine, QuarantineRecord

    quarantine = PanelQuarantine(protected=frozenset({"BTCUSDT"}))
    quarantine.add("AAAUSDT", "unreadable:ArrowInvalid")
    quarantine.add("AAAUSDT", "decision_bar_missing")
    quarantine.add("BBBUSDT", "funding_unreadable:ValueError")

    assert quarantine.records == [
        QuarantineRecord(symbol="AAAUSDT", reason="unreadable:ArrowInvalid"),
        QuarantineRecord(symbol="BBBUSDT", reason="funding_unreadable:ValueError"),
    ]
    assert quarantine.symbols == frozenset({"AAAUSDT", "BBBUSDT"})


def test_panel_quarantine_protected_symbol_fails_closed() -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.panel import PanelQuarantine

    quarantine = PanelQuarantine(protected=frozenset({"BTCUSDT", "ETHUSDT"}))

    with pytest.raises(DataIntegrityError, match="protected symbol ETHUSDT"):
        quarantine.add("ETHUSDT", "decision_bar_missing")

    assert quarantine.records == []


def test_panel_quarantine_enforce_limit_is_min_of_count_and_fraction() -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.panel import PanelQuarantine

    large = PanelQuarantine(protected=frozenset())
    for i in range(5):
        large.add(f"S{i}USDT", "decision_bar_missing")
    large.enforce_limit(522)
    large.add("S5USDT", "decision_bar_missing")
    with pytest.raises(DataIntegrityError, match="exceeds limit 5"):
        large.enforce_limit(522)

    small = PanelQuarantine(protected=frozenset())
    small.add("AUSDT", "decision_bar_missing")
    small.enforce_limit(50)
    small.add("BUSDT", "decision_bar_missing")
    with pytest.raises(DataIntegrityError, match="exceeds limit 1"):
        small.enforce_limit(50)


def test_load_base_panel_without_quarantine_keeps_fail_closed_on_corrupt_file(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )
    import pyarrow as pa
    import pytest

    _write("AAAUSDT", slice(None))
    (directory / "BBBUSDT.parquet").write_bytes(b"not a parquet")

    with pytest.raises((OSError, pa.ArrowException)):
        _load(None)


def test_load_base_panel_quarantines_corrupt_file_in_survivor_scan(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )

    from src.mhs.panel import PanelQuarantine

    _write("AAAUSDT", slice(None))
    (directory / "BBBUSDT.parquet").write_bytes(b"not a parquet")
    quarantine = PanelQuarantine(protected=frozenset())

    panel = _load(quarantine)

    assert list(panel["close"].columns) == ["AAAUSDT"]
    assert [(r.symbol, r.reason) for r in quarantine.records] == [("BBBUSDT", "unreadable:ArrowInvalid")]


def test_load_base_panel_quarantines_missing_column_in_column_read(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )

    from src.mhs.panel import PanelQuarantine

    _write("AAAUSDT", slice(None))
    _write("BBBUSDT", slice(None), columns=("close", "open"))
    _write("CCCUSDT", slice(None))
    quarantine = PanelQuarantine(protected=frozenset())

    panel = _load(quarantine)

    for field in ("close", "open", "quote_vol"):
        assert list(panel[field].columns) == ["AAAUSDT", "CCCUSDT"]
    assert np.allclose(panel["close"]["CCCUSDT"].to_numpy(), np.linspace(1.0, 2.0, 100))
    assert [(r.symbol, r.reason) for r in quarantine.records] == [("BBBUSDT", "unreadable:ArrowInvalid")]


def test_load_base_panel_quarantines_symbol_missing_recent_decision_bar(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )

    from src.mhs.panel import PanelQuarantine

    _write("AAAUSDT", slice(None))
    _write("BBBUSDT", slice(0, 99))
    _write("DEADUSDT", slice(0, 10))
    quarantine = PanelQuarantine(protected=frozenset())

    panel = _load(quarantine)

    assert list(panel["close"].columns) == ["AAAUSDT", "DEADUSDT"]
    assert pd.isna(panel["close"].loc[ts[-1], "DEADUSDT"])
    assert [(r.symbol, r.reason) for r in quarantine.records] == [("BBBUSDT", "decision_bar_missing")]


def test_load_base_panel_quarantine_over_limit_fails_closed(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )

    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.panel import PanelQuarantine

    _write("AAAUSDT", slice(None))
    (directory / "BBBUSDT.parquet").write_bytes(b"not a parquet")
    (directory / "CCCUSDT.parquet").write_bytes(b"not a parquet")

    with pytest.raises(DataIntegrityError, match="exceeds limit 1"):
        _load(PanelQuarantine(protected=frozenset()))


def test_load_base_panel_protected_symbol_unreadable_fails_closed(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )

    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.panel import PanelQuarantine

    _write("AAAUSDT", slice(None))
    (directory / "BBBUSDT.parquet").write_bytes(b"not a parquet")

    with pytest.raises(DataIntegrityError, match="protected symbol BBBUSDT"):
        _load(PanelQuarantine(protected=frozenset({"BBBUSDT"})))


def test_load_base_panel_without_quarantine_keeps_fail_closed_on_missing_column(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )
    import pyarrow as pa
    import pytest

    _write("AAAUSDT", slice(None), columns=("close", "open"))

    with pytest.raises(pa.ArrowException):
        _load(None)


def test_load_base_panel_quarantine_dropping_every_survivor_raises_no_survivor(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = pd.date_range("2026-09-01", periods=100, freq="1h", tz="UTC")
    epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    def _write(symbol: str, rows: slice, columns: tuple[str, ...] = ("close", "open", "quote_vol")) -> None:
        frame = pd.DataFrame({"timestamp": epoch[rows]})
        for column in columns:
            frame[column] = np.linspace(1.0, 2.0, len(frame))
        frame.to_parquet(directory / f"{symbol}.parquet", index=False)

    def _load(quarantine):
        return load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close", "open", "quote_vol"),
            start=ts[0], end=ts[-1], partition="all", min_bars=1, quarantine=quarantine,
        )
    import pytest
    from src.mhs.panel import PanelQuarantine

    _write("AAAUSDT", slice(None), columns=("close", "open"))
    quarantine = PanelQuarantine(protected=frozenset())

    with pytest.raises(ValueError, match="no symbol survived the panel filters"):
        _load(quarantine)

    assert [(r.symbol, r.reason) for r in quarantine.records] == [("AAAUSDT", "unreadable:ArrowInvalid")]


def test_load_base_panel_legacy_policy_keeps_zombie_bars(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    def _write_symbol(path, flat_flags, start="2024-01-01"):
        n = len(flat_flags)
        ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
        flat = np.asarray(flat_flags, dtype=bool)
        close = 100.0 + np.arange(n, dtype="float64")
        pd.DataFrame({
            "timestamp": (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
            "open": close,
            "high": np.where(flat, close, close + 1.0),
            "low": np.where(flat, close, close - 1.0),
            "close": close,
            "volume": np.where(flat, 0.0, 5.0),
            "quote_vol": np.where(flat, 0.0, 500.0),
        }).to_parquet(path, index=False)
        return ts
    from src.mhs.panel import DATA_POLICY_LEGACY, load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = _write_symbol(directory / "AAAUSDT.parquet", [False] * 10 + [True] * 40)

    panel = load_base_panel(
        root=str(tmp_path), interval="1h", columns=("close",), start=ts[0], end=ts[-1],
        partition="all", min_bars=1, data_policy=DATA_POLICY_LEGACY,
    )

    assert panel["close"]["AAAUSDT"].notna().all()


def test_load_base_panel_zombie_policy_masks_only_long_flat_runs(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    def _write_symbol(path, flat_flags, start="2024-01-01"):
        n = len(flat_flags)
        ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
        flat = np.asarray(flat_flags, dtype=bool)
        close = 100.0 + np.arange(n, dtype="float64")
        pd.DataFrame({
            "timestamp": (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
            "open": close,
            "high": np.where(flat, close, close + 1.0),
            "low": np.where(flat, close, close - 1.0),
            "close": close,
            "volume": np.where(flat, 0.0, 5.0),
            "quote_vol": np.where(flat, 0.0, 500.0),
        }).to_parquet(path, index=False)
        return ts
    from src.mhs.panel import DATA_POLICY_ZOMBIE_MASK_V1, ZOMBIE_FLAT_RUN_BARS, load_base_panel

    assert ZOMBIE_FLAT_RUN_BARS == 24
    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    flags = [False] * 10 + [True] + [False] * 9 + [True] * 40
    ts = _write_symbol(directory / "AAAUSDT.parquet", flags)

    panel = load_base_panel(
        root=str(tmp_path), interval="1h", columns=("close", "quote_vol"), start=ts[0], end=ts[-1],
        partition="all", min_bars=1, data_policy=DATA_POLICY_ZOMBIE_MASK_V1,
    )

    close = panel["close"]["AAAUSDT"]
    assert close.iloc[:43].notna().all()
    assert close.iloc[43:].isna().all()
    assert panel["quote_vol"]["AAAUSDT"].iloc[43:].isna().all()
    assert close.iloc[10] == 110.0


def test_load_base_panel_zombie_policy_uses_pre_window_bars_and_drops_fully_masked_symbol(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    def _write_symbol(path, flat_flags, start="2024-01-01"):
        n = len(flat_flags)
        ts = pd.date_range(start, periods=n, freq="1h", tz="UTC")
        flat = np.asarray(flat_flags, dtype=bool)
        close = 100.0 + np.arange(n, dtype="float64")
        pd.DataFrame({
            "timestamp": (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
            "open": close,
            "high": np.where(flat, close, close + 1.0),
            "low": np.where(flat, close, close - 1.0),
            "close": close,
            "volume": np.where(flat, 0.0, 5.0),
            "quote_vol": np.where(flat, 0.0, 500.0),
        }).to_parquet(path, index=False)
        return ts
    from src.mhs.panel import DATA_POLICY_ZOMBIE_MASK_V1, load_base_panel

    directory = tmp_path / "1h"
    directory.mkdir(parents=True)
    ts = _write_symbol(directory / "AAAUSDT.parquet", [False] * 10 + [True] * 50)
    _write_symbol(directory / "BBBUSDT.parquet", [False] * 60)

    panel = load_base_panel(
        root=str(tmp_path), interval="1h", columns=("close",), start=ts[50], end=ts[59],
        partition="all", min_bars=1, data_policy=DATA_POLICY_ZOMBIE_MASK_V1,
    )

    assert list(panel["close"].columns) == ["BBBUSDT"]


def test_load_base_panel_rejects_unknown_data_policy(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.mhs.panel import load_base_panel

    with pytest.raises(ValueError, match="unknown data_policy"):
        load_base_panel(
            root=str(tmp_path), interval="1h", columns=("close",),
            start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-01-02", tz="UTC"),
            partition="all", min_bars=1, data_policy="zombie_mask_v9",
        )


def test_zombie_masked_timestamps_requires_volume_high_low(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.panel import zombie_masked_timestamps

    path = tmp_path / "AAAUSDT.parquet"
    pd.DataFrame({"timestamp": [0, 3_600_000], "close": [1.0, 2.0]}).to_parquet(path, index=False)

    with pytest.raises(DataIntegrityError, match="zombie mask requires"):
        zombie_masked_timestamps(str(path), 0, 3_600_000, 3_600_000)


def test_zombie_masked_timestamps_empty_range_and_duplicate_keep_last(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.panel import zombie_masked_timestamps

    hour = 3_600_000
    path = tmp_path / "AAAUSDT.parquet"
    n = 30
    ts = [i * hour for i in range(n)] + [5 * hour]
    volume = [0.0] * n + [7.0]
    high = [1.0] * n + [2.0]
    low = [1.0] * (n + 1)
    pd.DataFrame({"timestamp": ts, "volume": volume, "high": high, "low": low}).to_parquet(path, index=False)

    empty = zombie_masked_timestamps(str(path), 1000 * hour, 1001 * hour, hour)
    assert empty.dtype == np.int64
    assert empty.size == 0

    masked = zombie_masked_timestamps(str(path), 0, (n - 1) * hour, hour, 24)
    assert masked.tolist() == [i * hour for i in range(29, n)]

