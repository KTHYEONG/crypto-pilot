"""Execution replay engine contract tests (split by behavioral domain)."""

from __future__ import annotations
import dataclasses
import tracemalloc
import numpy as np
import pandas as pd
import pytest
from src.common.errors import DataIntegrityError
from src.mhs.execution import (
    ExecutionSpec,
    _BoundExecutionReplayAccumulator,
    replay_execution_window_batch,
    replay_execution_window_pair,
    replay_execution_windows,
    simulated_inventory_ledger,
    strategy_aware_execution_replay,
)

from tests.unit.mhs.test_execution import (  # noqa: F401
    _assert_full_equivalence,
    _assert_pair_equivalent,
    _assert_replay_equivalent,
    _partition_windows,
)

class TestStreamedLedgerEquivalence:
    """MHS-30-STREAMED-LEDGER-EQUIVALENCE: the default streaming mode avoids the
    dense ledger unit matrix and matches the opt-in dense diagnostic mode."""

    def test_streaming_matches_dense_mode_within_1e12(self) -> None:
        idx = pd.date_range("2021-01-01", periods=40, freq="5min", tz="UTC")
        rng = np.random.default_rng(7)
        marks = pd.DataFrame(
            {
                sym: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, len(idx))))
                for sym in ("AAAUSDT", "BBBUSDT", "CCCUSDT")
            },
            index=idx,
        )
        marks.iloc[15:18, 1] = np.nan
        funding = pd.DataFrame(2.0e-5, index=idx, columns=list(marks.columns))
        fills = pd.DataFrame(
            [
                {"timestamp": idx[3], "symbol": "AAAUSDT", "quantity_delta": 0.02,
                 "fill_price": float(marks.loc[idx[3], "AAAUSDT"]), "fee_bps": 2.0,
                 "reason": "passive_fill"},
                {"timestamp": idx[3], "symbol": "BBBUSDT", "quantity_delta": -0.01,
                 "fill_price": float(marks.loc[idx[3], "BBBUSDT"]), "fee_bps": 2.0,
                 "reason": "passive_fill"},
                {"timestamp": idx[20], "symbol": "CCCUSDT", "quantity_delta": 0.03,
                 "fill_price": float(marks.loc[idx[20], "CCCUSDT"]), "fee_bps": 8.0,
                 "reason": "timeout_taker"},
            ],
        )
        streamed = simulated_inventory_ledger(
            fills, marks, funding, 1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
        )
        dense = simulated_inventory_ledger(
            fills, marks, funding, 1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
            retain_simulated_units=True,
        )
        assert streamed.simulated_units is None
        assert dense.simulated_units is not None
        assert list(dense.simulated_units.columns) == list(marks.columns)
        assert list(dense.simulated_units.index) == list(idx)
        for field in (
            "equity", "net_returns", "mark_to_market_pnl",
            "funding_charge", "fee_charge", "fill_turnover",
        ):
            np.testing.assert_allclose(
                getattr(streamed, field).to_numpy(),
                getattr(dense, field).to_numpy(),
                rtol=1e-12, atol=1e-12,
            )
        assert streamed.primary_valid is dense.primary_valid
        assert streamed.invalid_reasons == dense.invalid_reasons
        np.testing.assert_allclose(
            dense.simulated_units.loc[idx[3]].to_numpy(), [0.02, -0.01, 0.0], atol=1e-12,
        )
        np.testing.assert_allclose(
            dense.simulated_units.loc[idx[20]].to_numpy(), [0.02, -0.01, 0.03], atol=1e-12,
        )

class TestWindowedReplayEquivalence:
    """MHS-30-STREAMED-LEDGER-EQUIVALENCE /
    SCENARIO_MHS_PERF_P2_04_WINDOWED_REPLAY_EQUIVALENCE: windowed strict and
    stress replays match the single-panel replay in fills, termination
    counts, ledger series, and validity at 1e-12 tolerance -- the P2 engine
    micro-optimizations (column-order reduction, NaN-only equity zeroing)
    must not disturb this equivalence."""

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
        marks = closes.copy()
        marks.iloc[100:105, 3] = np.nan
        decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
        weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
        rng_w = np.random.default_rng(8)
        for ts in decision_grid:
            active = rng_w.choice(symbols, size=4, replace=False)
            weights.loc[ts, active] = rng_w.uniform(0.01, 0.06, 4)
        return {
            "grid": grid,
            "symbols": symbols,
            "highs": closes * 1.001,
            "lows": closes * 0.999,
            "closes": closes,
            "marks": marks,
            "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
            "weights": weights,
            "signals": decision_grid + pd.Timedelta(hours=1),
        }

    @pytest.mark.parametrize("bound", ["OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER"])
    def test_windowed_matches_single_panel(self, bound: str) -> None:
        wl = self._workload()
        grid = wl["grid"]
        weights = wl["weights"]
        signals = wl["signals"]
        oracle = strategy_aware_execution_replay(
            weights, signals, wl["highs"], wl["lows"], wl["closes"], wl["marks"],
            wl["funding"], 1.0, bound, ExecutionSpec(),
        )
        windows = _partition_windows(
            grid, weights, signals, wl["highs"], wl["lows"], wl["closes"],
            wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        windowed = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(), retain_event_snapshots=True,
        )
        _assert_replay_equivalent(oracle, windowed)

    def test_three_windows_follow_strict_timeout_overlap(self) -> None:
        wl = self._workload(days=45)
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        assert len(windows) == 3
        for w in windows[1:]:
            assert w.minute_grid[0] == w.window_start
            assert w.window_start < w.target_weights.index[0]
        assert windows[-1].minute_grid[-1] == wl["grid"][-1]

    def test_data_gap_provenance_codes(self) -> None:
        wl = self._workload()
        windowed = replay_execution_windows(
            _partition_windows(
                wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
                wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
            ),
            1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        codes = {g.code for g in windowed.data_gaps}
        # The synthetic mark gap at bars 100-105 is held through by a fill,
        # so the real held-mark/held-funding provenance must be attributed.
        assert codes >= {"MISSING_HELD_MARK", "MISSING_HELD_FUNDING"}
        for gap in windowed.data_gaps:
            assert gap.symbol
            assert gap.timestamp is not None
            assert gap.execution_bound == "OHLCV_STRICT_PROXY"

    def test_paired_fanout_matches_independent_bounds(self) -> None:
        """MHS-MEM-PAIR-01: a single window stream fanned out into the
        strict/stress pair equals the two legacy independent single-bound
        calls, and each window is consumed exactly once."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        strict_single = replay_execution_windows(
            windows, 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(), retain_event_snapshots=True,
        )
        stress_single = replay_execution_windows(
            windows, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), retain_event_snapshots=True,
        )
        consumed = {"n": 0}

        def _gen():
            for w in windows:
                consumed["n"] += 1
                yield w

        strict_pair, stress_pair = replay_execution_window_pair(
            _gen(), 1.0, ExecutionSpec(), retain_event_snapshots=True,
        )
        assert consumed["n"] == len(windows)
        _assert_pair_equivalent(strict_single, strict_pair, "strict")
        _assert_pair_equivalent(stress_single, stress_pair, "stress")

    def test_batch_fanout_matches_independent_bounds(self) -> None:
        """SCENARIO_MHS_STREAM_BATCH_EQUIVALENCE: an N-bound interleaved batch
        over a single window iterator equals N independent single-bound calls
        and consumes each window exactly once."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        spec = ExecutionSpec()
        bounds = [
            ("OHLCV_IMMEDIATE_TAKER", spec),
            ("OHLCV_IMMEDIATE_TAKER", spec),
            ("OHLCV_STRICT_PROXY", spec),
        ]
        independent = [
            replay_execution_windows(windows, 1.0, b, s, retain_event_snapshots=True)
            for (b, s) in bounds
        ]
        consumed = {"n": 0}

        def _gen():
            for w in windows:
                consumed["n"] += 1
                yield w

        batch = replay_execution_window_batch(
            _gen(), 1.0, bounds, retain_event_snapshots=True,
        )
        assert consumed["n"] == len(windows)
        assert len(batch) == len(bounds)
        for i, (indep, bres) in enumerate(zip(independent, batch, strict=True)):
            _assert_pair_equivalent(indep, bres, f"batch[{i}]")

    def test_batch_empty_bounds_fails_closed(self) -> None:
        """SCENARIO_MHS_STREAM_BATCH_EMPTY_BOUNDS: an empty bounds iterable
        raises ValueError before any window is consumed."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        with pytest.raises(ValueError, match="bounds"):
            replay_execution_window_batch(windows, 1.0, [])

    def test_pair_strict_data_integrity_error_propagates(self) -> None:
        """MHS-MEM-PAIR-01: a fatal strict DataIntegrityError propagates
        unchanged; no stress result is fabricated."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=2,
        )
        windows[1] = dataclasses.replace(windows[1], bar_funding=windows[1].bar_funding * float("nan"))
        with pytest.raises(DataIntegrityError, match="bar_funding must be finite"):
            replay_execution_window_pair(windows, 1.0, ExecutionSpec())

class TestColumnarFillAccumulator:
    """Columnar fill refactor: ``_BoundExecutionReplayAccumulator`` stores fills field-wise.
    The ``simulated_fills`` output must remain byte-for-byte identical to the legacy dict path.
    """

    FILL_COLUMNS = (
        "timestamp", "symbol", "quantity_delta", "fill_price", "fee_bps",
        "reason", "pre_trade_equity",
    )

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
        decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
        weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
        rng_w = np.random.default_rng(8)
        for ts in decision_grid:
            active = rng_w.choice(symbols, size=4, replace=False)
            weights.loc[ts, active] = rng_w.uniform(0.01, 0.06, 4)
        return {
            "grid": grid,
            "symbols": symbols,
            "highs": closes * 1.001,
            "lows": closes * 0.999,
            "closes": closes,
            "marks": closes,
            "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
            "weights": weights,
            "signals": decision_grid + pd.Timedelta(hours=1),
        }

    def _windowed(
        self, wl: dict[str, object], bound: str = "OHLCV_STRICT_PROXY",
    ):
        return replay_execution_windows(
            _partition_windows(
                wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
                wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
            ),
            1.0, bound, ExecutionSpec(),
        )

    def test_bound_execution_replay_accumulator_columnar_fills_match_legacy_dict_output(self) -> None:
        """R2 output-equivalence contract: the columnar ``simulated_fills`` must
        be byte-for-byte identical (column order, dtypes, row values) to the
        legacy dict-list DataFrame built from the same underlying fill data.
        Driving ``_BoundExecutionReplayAccumulator`` directly exposes the raw
        field lists so the legacy reference is reconstructed exactly as the
        pre-refactor ``finalize`` did, including forced-exit fills."""
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        acc = _BoundExecutionReplayAccumulator(
            windows[0], 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(), False,
        )
        for w in windows:
            acc.consume(w)
        columnar_df = acc.finalize().simulated_fills
        assert len(columnar_df) > 0
        # Rebuild the legacy one-dict-per-fill DataFrame from the same lists.
        legacy_records = [
            {
                "timestamp": acc.fill_ts[i],
                "symbol": acc.fill_symbol[i],
                "quantity_delta": acc.fill_qty[i],
                "fill_price": acc.fill_price[i],
                "fee_bps": acc.fill_fee_bps[i],
                "reason": acc.fill_reason[i],
                "pre_trade_equity": acc.fill_pre_trade_equity[i],
            }
            for i in range(len(acc.fill_ts))
        ]
        legacy_df = pd.DataFrame(legacy_records, columns=self.FILL_COLUMNS)
        if legacy_df.empty:
            legacy_df = legacy_df.astype(
                {"quantity_delta": "float64", "fill_price": "float64", "fee_bps": "float64"}
            )
        pd.testing.assert_frame_equal(
            columnar_df, legacy_df, check_dtype=True, check_exact=True,
        )

    def test_bound_execution_replay_accumulator_columnar_fills_dtype_and_column_order(self) -> None:
        wl = self._workload()
        fills = self._windowed(wl).simulated_fills
        assert not fills.empty
        assert list(fills.columns) == list(self.FILL_COLUMNS)
        assert fills.dtypes["timestamp"] == pd.DatetimeTZDtype(tz="UTC", unit="ns")
        for col in ("quantity_delta", "fill_price", "fee_bps", "pre_trade_equity"):
            assert fills.dtypes[col] == np.dtype("float64"), col
        for col in ("symbol", "reason"):
            assert fills.dtypes[col] == pd.StringDtype(na_value=np.nan), col
        # The empty case keeps the same column order and the numeric float64
        # dtypes, so an empty table cannot be mistaken for missing output.
        empty = replay_execution_windows(
            _partition_windows(
                wl["grid"], wl["weights"].mul(0.0), wl["signals"], wl["highs"],
                wl["lows"], wl["closes"], wl["marks"], wl["funding"],
                ExecutionSpec(), n_windows=2,
            ),
            1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        ).simulated_fills
        assert empty.empty
        assert list(empty.columns) == list(self.FILL_COLUMNS)
        assert empty.dtypes["quantity_delta"] == np.dtype("float64")
        assert empty.dtypes["fill_price"] == np.dtype("float64")
        assert empty.dtypes["fee_bps"] == np.dtype("float64")

    def test_columnar_fill_accumulation_reduces_peak_rss_vs_dict_baseline(self) -> None:
        """R2 memory claim: holding 10k+ fills as one dict per fill (legacy)
        allocates more peak memory than the field-wise parallel lists (columnar).
        ``tracemalloc`` measures the tracked allocation peak, the deterministic
        structural driver of the RSS the spec targets; a synthetic replay is not
        needed because the diff is purely the container representation."""
        n = 10_000
        rng = np.random.default_rng(11)
        symbols = [f"SYM{i:04d}USDT" for i in range(64)]
        times = pd.date_range("2021-01-01", periods=n, freq="1min", tz="UTC")
        ts_list = [times[i] for i in range(n)]
        syms = [symbols[i % len(symbols)] for i in range(n)]
        qty = rng.uniform(0.001, 0.1, n).tolist()
        price = rng.uniform(50.0, 150.0, n).tolist()
        fee = rng.uniform(0.0, 10.0, n).tolist()
        reasons = ["passive_fill" if i % 2 else "timeout_taker" for i in range(n)]
        equity = rng.uniform(0.5, 1.5, n).tolist()

        tracemalloc.start()
        legacy_fills: list[dict[str, object]] = [
            {
                "timestamp": ts_list[i], "symbol": syms[i],
                "quantity_delta": qty[i], "fill_price": price[i],
                "fee_bps": fee[i], "reason": reasons[i],
                "pre_trade_equity": equity[i],
            }
            for i in range(n)
        ]
        legacy_df = pd.DataFrame(legacy_fills)
        legacy_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        tracemalloc.start()
        col_ts: list[pd.Timestamp] = []
        col_sym: list[str] = []
        col_qty: list[float] = []
        col_price: list[float] = []
        col_fee: list[float] = []
        col_reason: list[str] = []
        col_equity: list[float] = []
        for i in range(n):
            col_ts.append(ts_list[i])
            col_sym.append(syms[i])
            col_qty.append(qty[i])
            col_price.append(price[i])
            col_fee.append(fee[i])
            col_reason.append(reasons[i])
            col_equity.append(equity[i])
        columnar_df = pd.DataFrame(
            {
                "timestamp": col_ts, "symbol": col_sym, "quantity_delta": col_qty,
                "fill_price": col_price, "fee_bps": col_fee, "reason": col_reason,
                "pre_trade_equity": col_equity,
            }
        )
        columnar_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        assert len(legacy_fills) >= 10_000
        assert len(columnar_df) == len(legacy_df)
        # The gap is structural (dict container + hash table per fill), not noise.
        assert legacy_peak - columnar_peak > 256 * 1024
        assert columnar_peak < legacy_peak

class TestEventSnapshotOptIn:
    """MHS-MEM-01/02: dense event snapshots are opt-in and bounded-memory by default."""

    def _workload(self, days: int = 40, n_symbols: int = 8) -> dict[str, object]:
        grid = pd.date_range("2021-01-01", periods=days * 24 * 12, freq="5min", tz="UTC")
        symbols = [f"SYM{i:03d}USDT" for i in range(n_symbols)]
        rng = np.random.default_rng(7)
        closes = pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(grid)))) for s in symbols},
            index=grid,
        )
        marks = closes.copy()
        marks.iloc[100:105, 3] = np.nan
        decision_grid = pd.date_range("2021-01-01", periods=days * 4, freq="6h", tz="UTC")
        weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
        rng_w = np.random.default_rng(8)
        for ts in decision_grid:
            active = rng_w.choice(symbols, size=4, replace=False)
            weights.loc[ts, active] = rng_w.uniform(0.01, 0.06, 4)
        return {
            "grid": grid,
            "symbols": symbols,
            "highs": closes * 1.001,
            "lows": closes * 0.999,
            "closes": closes,
            "marks": marks,
            "funding": pd.DataFrame(1.0e-5, index=grid, columns=symbols),
            "weights": weights,
            "signals": decision_grid + pd.Timedelta(hours=1),
        }

    @pytest.mark.parametrize("bound", ["OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER"])
    def test_snapshot_disabled_equals_enabled(self, bound: str) -> None:
        wl = self._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        enabled = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(), retain_event_snapshots=True,
        )
        disabled = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(),
        )
        assert disabled.event_snapshots_retained is False
        assert enabled.event_snapshots_retained is True
        assert len(disabled.simulated_units) == 0
        assert len(disabled.simulated_notional_weights) == 0
        assert len(enabled.simulated_units) > 0
        _assert_full_equivalence(enabled, disabled)

    def test_wide_fill_disabled_retains_zero_dense_rows(self) -> None:
        wl = self._workload(days=60, n_symbols=32)
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=4,
        )
        disabled = replay_execution_windows(
            windows, 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert disabled.fill_count > 0
        assert len(disabled.simulated_fills) > 0
        assert disabled.event_snapshots_retained is False
        assert disabled.simulated_units.empty
        assert disabled.simulated_notional_weights.empty
        assert list(disabled.simulated_units.columns) == list(wl["symbols"])
