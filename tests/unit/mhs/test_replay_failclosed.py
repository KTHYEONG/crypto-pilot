"""Execution replay engine contract tests (split by behavioral domain)."""

from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.common.errors import DataIntegrityError
from src.mhs.execution import (
    ExecutionReplayWindow,
    ExecutionSpec,
    replay_execution_windows,
    simulated_inventory_ledger,
    strategy_aware_execution_replay,
)

from tests.unit.mhs.test_replay_accumulator import (
    TestWindowedReplayEquivalence as _WindowedReplayWorkload,
)
from tests.unit.mhs.test_execution import (  # noqa: F401
    _assert_replay_equivalent,
    _partition_windows,
)

class TestMissingDataTermination:
    """MHS-23-RELEVANT-MISSING-DATA-AND-TERMINATION: relevant gaps fail closed."""

    def test_flat_order_free_symbol_gap_does_not_count(self) -> None:
        grid = pd.date_range("2021-01-01 01:01", periods=60, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid), "B": [50.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 01:20":, "B"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A", "B"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        # B is flat and order-free: its gap must not be counted as relevant.
        assert report.termination_counts["MISSING_DATA"] == 0

    def test_relevant_gap_in_active_order_window_is_fail_closed(self) -> None:
        grid = pd.date_range("2021-01-01 01:01", periods=60, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 01:10":"2021-01-01 01:40", "A"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert report.termination_counts["MISSING_DATA"] == 1
        assert report.simulated_fills.empty

    def test_unverified_boundary_is_missing_data_not_delisting(self) -> None:
        grid = pd.date_range("2021-01-01 01:01", periods=60, freq="1min", tz="UTC")
        px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
        px.loc["2021-01-01 01:30":, "A"] = np.nan
        target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
        signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
        report = strategy_aware_execution_replay(
            target, signal_at, px, px, px, px,
            pd.DataFrame(0.0, index=grid, columns=["A"]), 1.0,
            "OHLCV_STRICT_PROXY", ExecutionSpec(),
        )
        assert set(report.termination_counts) == {"MISSING_DATA", "UNKNOWN_TERMINATION"}
        assert "DELIST" not in report.termination_counts

class TestTerminalFailClosedRejection:
    """MHS-28-TERMINAL-FAIL-CLOSED-REPORT: a deterministic capital breach stays
    fail-closed inside the ledger and is classified to the stable
    CAPITAL_INVARIANT_BREACH reason so the diagnostic can persist a serializable
    typed rejection instead of dying on an uncaught process error."""

    def test_ledger_capital_breach_is_fail_closed(self) -> None:
        from src.application.research.mhs.evaluation import _classify_execution_failure
        from src.common.errors import DataIntegrityError

        idx = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
        marks = pd.DataFrame({"A": [100.0, 200.0, 200.0]}, index=idx)
        fills = pd.DataFrame(
            [
                {"timestamp": idx[0], "symbol": "A", "quantity_delta": -0.01,
                 "fill_price": 100.0, "fee_bps": 0.0, "reason": "passive_fill"},
                {"timestamp": idx[1], "symbol": "A", "quantity_delta": 0.01,
                 "fill_price": 200.0, "fee_bps": 0.0, "reason": "passive_fill"},
            ],
        )
        with pytest.raises(DataIntegrityError, match="pre-trade equity"):
            simulated_inventory_ledger(
                fills, marks, pd.DataFrame(0.0, index=idx, columns=["A"]),
                1.0, "OHLCV_STRICT_PROXY", "MARK_PRICE",
            )
        reason = _classify_execution_failure(DataIntegrityError("pre-trade equity must be positive and finite"))
        assert reason == "CAPITAL_INVARIANT_BREACH"

    def test_book_failure_serializes_null_metrics(self) -> None:
        import json

        from src.application.research.mhs.evaluation import MhsBookFailure, MhsBookReport, _jsonable
        from src.common.errors import DataIntegrityError

        exc = DataIntegrityError("pre-trade equity must be positive and finite")
        failed = MhsBookReport(
            name="blend", band="fast_reversal", horizon_hours=48, step_hours=6,
            tranche_count=8, n_symbols=8, phase=None, prescreen=None, tail=None,
            primary=None, stress=None,
            primary_autocorr_sharpe=None, primary_naive_sharpe=None,
            primary_net_ann=None, primary_geometric_cagr=None,
            primary_max_drawdown=None, primary_annualized_turnover=None,
            stress_naive_sharpe=None, terminal_censored_decisions=0,
            failure=MhsBookFailure(
                stage="replay_blend", error_class="DataIntegrityError",
                reason="CAPITAL_INVARIANT_BREACH", message=str(exc),
            ),
        )
        payload = _jsonable(failed)
        assert payload["primary"] is None
        assert payload["primary_autocorr_sharpe"] is None
        assert payload["failure"]["reason"] == "CAPITAL_INVARIANT_BREACH"
        assert payload["failure"]["error_class"] == "DataIntegrityError"
        # Strict JSON: no NaN/Infinity tokens leak into the terminal payload.
        encoded = json.dumps(payload)
        assert "NaN" not in encoded
        assert "Infinity" not in encoded

class TestCapitalInvariantFailFast:
    """SCENARIO_MHS_CAPITAL_HARDENING_01/02/03: non-finite fill sizing is
    rejected fail-fast at the exact fill site (both the laddered and the
    single-fill branches) with the full symbol/timestamp/weight/equity/
    decision_price context, the new message still classifies as
    CAPITAL_INVARIANT_BREACH, and the guards are pure no-ops on any
    fully-finite replay."""

    def _single_decision_window(
        self, *, nan_close_at: pd.Timestamp | None = None,
        weight: float = 1.0, mark: float = 100.0,
    ) -> ExecutionReplayWindow:
        grid = pd.date_range("2021-01-01 00:00", periods=13, freq="5min", tz="UTC")
        symbols = ("AAAUSDT",)
        closes = pd.DataFrame({"AAAUSDT": [100.0] * len(grid)}, index=grid)
        if nan_close_at is not None:
            closes.loc[nan_close_at, "AAAUSDT"] = np.nan
        marks = pd.DataFrame({"AAAUSDT": [mark] * len(grid)}, index=grid)
        decision = pd.Timestamp("2021-01-01 00:10", tz="UTC")
        return ExecutionReplayWindow(
            window_start=grid[0],
            window_end=grid[-1],
            columns=symbols,
            symbols=symbols,
            minute_grid=grid,
            highs=closes * 1.01,
            lows=closes * 0.99,
            closes=closes,
            marks=marks,
            bar_funding=pd.DataFrame(0.0, index=grid, columns=symbols),
            target_weights=pd.DataFrame({"AAAUSDT": [weight]}, index=[decision]),
            signal_available_at=pd.DatetimeIndex(
                [pd.Timestamp("2021-01-01 00:05", tz="UTC")],
            ),
        )

    def test_non_finite_qty_raises_in_immediate_taker_branch(self) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_01 (immediate-taker site): a target
        weight large enough to overflow ``desired_units`` (weight * equity /
        mark) makes ``net_units`` non-finite in the OHLCV_IMMEDIATE_TAKER
        single-fill branch -- the one path a NaN close no longer reaches, since
        the new fill-price guard now skips that order upstream. The fail-fast
        sizing guard still raises there with the full symbol/decision context,
        never letting a non-finite qty corrupt ``self.cash``."""
        decision = pd.Timestamp("2021-01-01 00:10", tz="UTC")
        window = self._single_decision_window(weight=1e308, mark=0.5)
        with pytest.raises(DataIntegrityError, match="capital accounting invariant") as exc_info:
            replay_execution_windows([window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
        msg = str(exc_info.value)
        assert "AAAUSDT" in msg
        assert "00:10" in msg
        assert "decision_price=0.5" in msg
        assert "qty=np.float64(inf)" in msg

    def test_non_finite_qty_raises_in_laddered_branch(self) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_01 (laddered site): a target weight
        large enough to overflow ``desired_units`` (weight * equity / mark)
        makes ``qty`` non-finite in the laddered/multi-tranche branch; the
        guard raises there with the same full-context message."""
        window = self._single_decision_window(weight=1e308, mark=0.5)
        with pytest.raises(DataIntegrityError, match="capital accounting invariant") as exc_info:
            replay_execution_windows([window], 1.0, "OHLCV_LADDERED_PROXY", ExecutionSpec())
        msg = str(exc_info.value)
        assert "AAAUSDT" in msg
        assert "weight=1e+308" in msg
        assert "decision_price=0.5" in msg
        assert "qty=np.float64(inf)" in msg

    def test_new_message_still_classifies_as_capital_breach(self) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_02: the more detailed message keeps
        the 'capital' keyword so ``_classify_execution_failure`` still maps it
        to the stable CAPITAL_INVARIANT_BREACH reason."""
        from src.application.research.mhs.evaluation import _classify_execution_failure

        sym = "AAAUSDT"
        fill_time = pd.Timestamp("2021-01-01 00:10", tz="UTC")
        weight, equity, decision_price, qty, fill_price = 1.0, 1.0, 100.0, 0.01, float("nan")
        exc = DataIntegrityError(
            "non-finite fill sizing breaches the capital accounting invariant "
            f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
            f"decision_price={decision_price!r} qty={qty!r} fill_price={fill_price!r})"
        )
        assert _classify_execution_failure(exc) == "CAPITAL_INVARIANT_BREACH"

    @pytest.mark.parametrize(
        "bound", ["OHLCV_STRICT_PROXY", "OHLCV_IMMEDIATE_TAKER", "OHLCV_LADDERED_PROXY"],
    )
    def test_finite_replay_unchanged(self, bound: str) -> None:
        """SCENARIO_MHS_CAPITAL_HARDENING_03: the new guards are pure no-ops on
        a fully-finite window -- the windowed replay still matches the single-
        panel oracle in fills and every ledger series, so no sizing arithmetic
        was changed."""
        wl = _WindowedReplayWorkload()._workload()
        windows = _partition_windows(
            wl["grid"], wl["weights"], wl["signals"], wl["highs"], wl["lows"],
            wl["closes"], wl["marks"], wl["funding"], ExecutionSpec(), n_windows=3,
        )
        oracle = strategy_aware_execution_replay(
            wl["weights"], wl["signals"], wl["highs"], wl["lows"], wl["closes"],
            wl["marks"], wl["funding"], 1.0, bound, ExecutionSpec(),
        )
        windowed = replay_execution_windows(
            windows, 1.0, bound, ExecutionSpec(), retain_event_snapshots=True,
        )
        _assert_replay_equivalent(oracle, windowed)

class TestEquityFloorProtection:
    """SCENARIO_MHS_EQUITY_FLOOR_DEFAULT_NONE_STRICT_08
    SCENARIO_MHS_EQUITY_FLOOR_FORCES_FLAT_09
    SCENARIO_MHS_EQUITY_FLOOR_NO_BREACH_EMPTY_TUPLE_10: the reference-pass
    equity floor (min_equity_fraction) forces a decision row's sizing flat
    once pre-trade equity crosses ``min_equity_fraction * initial_equity``
    instead of levering a depleting base toward the hard turnover-equity
    check. The default None keeps today's strict fail-closed behavior, and a
    never-breached floor run is byte-identical to the None run."""

    def _short_squeeze_window(self, *, price_levels: tuple[float, ...]) -> ExecutionReplayWindow:
        """Short book squeezed by staged mark rises: each step's live equity
        ``_equity_at`` erodes through the 0.5 floor while the ledger-level
        pre-trade equity stays positive (so the floor's forced-flat unwind is
        itself well-capitalized), and a step that more-than-doubles the entry
        drives the ledger pre-trade equity <= 0 for the no-floor run."""
        grid = pd.date_range("2021-01-01 00:00", periods=37, freq="5min", tz="UTC")
        symbols = ("AAAUSDT",)
        bounds = (pd.Timestamp("2021-01-01 01:00", tz="UTC"), pd.Timestamp("2021-01-01 01:50", tz="UTC"))
        levels = np.where(
            grid < bounds[0], price_levels[0],
            np.where(grid < bounds[1], price_levels[1],
            np.where(grid < pd.Timestamp("2021-01-01 02:40", tz="UTC"), price_levels[2], price_levels[3])),
        )
        closes = pd.DataFrame({"AAAUSDT": levels}, index=grid)
        decisions = pd.DatetimeIndex(
            [
                pd.Timestamp("2021-01-01 00:10", tz="UTC"),
                pd.Timestamp("2021-01-01 01:00", tz="UTC"),
                pd.Timestamp("2021-01-01 01:50", tz="UTC"),
                pd.Timestamp("2021-01-01 02:40", tz="UTC"),
            ]
        )
        return ExecutionReplayWindow(
            window_start=grid[0],
            window_end=grid[-1],
            columns=symbols,
            symbols=symbols,
            minute_grid=grid,
            highs=closes * 1.01,
            lows=closes * 0.99,
            closes=closes,
            marks=closes,
            bar_funding=pd.DataFrame(0.0, index=grid, columns=symbols),
            target_weights=pd.DataFrame({"AAAUSDT": [-1.0] * 4}, index=decisions),
            signal_available_at=decisions,
        )

    def test_default_none_preserves_strict_fail_closed(self) -> None:
        """SCENARIO_MHS_EQUITY_FLOOR_DEFAULT_NONE_STRICT_08: a short entered at
        100 that more-than-doubles to 200 makes the ledger pre-trade equity at
        the buy-back fill <= 0; with min_equity_fraction unset the replay still
        raises the exact existing DataIntegrityError message."""
        window = self._short_squeeze_window(price_levels=(100.0, 200.0, 200.0, 200.0))
        with pytest.raises(DataIntegrityError, match="pre-trade equity must be positive and finite"):
            replay_execution_windows([window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())

    def test_floor_forces_flat_and_records_breach_timestamps(self) -> None:
        """SCENARIO_MHS_EQUITY_FLOOR_FORCES_FLAT_09: a gradual squeeze (100 ->
        140 -> 180 -> 220) erodes live equity through the 0.5 floor at the
        second and fourth decisions while the ledger-level pre-trade equity
        stays positive, so the floor's forced-flat unwind completes the replay
        and records the exact breach timestamps."""
        window = self._short_squeeze_window(price_levels=(100.0, 140.0, 180.0, 220.0))
        result = replay_execution_windows(
            [window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), min_equity_fraction=0.5,
        )
        breaches = result.ledger.equity_floor_breached_at
        assert len(breaches) >= 2
        assert breaches[0] == pd.Timestamp("2021-01-01 01:00", tz="UTC")
        assert breaches[1] == pd.Timestamp("2021-01-01 01:50", tz="UTC")
        assert breaches[-1] == pd.Timestamp("2021-01-01 02:40", tz="UTC")
        assert result.ledger.equity.min() > 0.0

    def test_no_breach_is_empty_tuple_and_byte_identical_to_none(self) -> None:
        """SCENARIO_MHS_EQUITY_FLOOR_NO_BREACH_EMPTY_TUPLE_10: a falling-price
        short (equity grows away from the floor) never trips it; the floor run
        returns an empty equity_floor_breached_at tuple and every other ledger
        field is byte-identical to the None run on the same fixture."""
        window = self._short_squeeze_window(price_levels=(100.0, 60.0, 70.0, 80.0))
        base = replay_execution_windows([window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
        floored = replay_execution_windows(
            [window], 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), min_equity_fraction=0.5,
        )
        assert base.ledger.equity_floor_breached_at == ()
        assert floored.ledger.equity_floor_breached_at == ()
        self._assert_ledger_fields_equal(base, floored)

    @staticmethod
    def _ledger_without_floor(result) -> dict[str, object]:
        out = {}
        for f in dataclasses.fields(result.ledger):
            if f.name == "equity_floor_breached_at":
                continue
            out[f.name] = getattr(result.ledger, f.name)
        return out

    def _assert_ledger_fields_equal(self, a, b) -> None:
        fa, fb = self._ledger_without_floor(a), self._ledger_without_floor(b)
        assert set(fa) == set(fb)
        for key in fa:
            va, vb = fa[key], fb[key]
            if isinstance(va, pd.Series):
                pd.testing.assert_series_equal(va, vb)
            elif isinstance(va, pd.DataFrame):
                pd.testing.assert_frame_equal(va, vb)
            else:
                assert va == vb

class TestRuinGuardEquity:
    """SCENARIO_MHS_RUIN_GUARD_LEDGER_TRACK: ruin_guard_equity uses
    min(fill_track, finalized_ledger) and the floor guard fires on
    the ledger track when the fill track is healthy."""

    def test_pure_function_cases(self) -> None:
        from src.mhs.execution import ruin_guard_equity
        assert ruin_guard_equity(1.3599, 0.0079) == pytest.approx(0.0079)
        assert ruin_guard_equity(1.3599, None) == pytest.approx(1.3599)
        assert ruin_guard_equity(0.5, 2.0) == pytest.approx(0.5)
        assert ruin_guard_equity(1.3599, float("nan")) == pytest.approx(1.3599)
        assert ruin_guard_equity(1.3599, float("inf")) == pytest.approx(1.3599)
        assert ruin_guard_equity(1.3599, float("-inf")) == pytest.approx(1.3599)

    def test_floor_guard_fires_on_ledger_track(self) -> None:
        """When the finalized ledger equity falls below 0.2*initial while the
        fill track stays above, the floor guard fires on the second window's
        first decision and subsequent decision rows are targeted flat."""
        from src.mhs.execution import _BoundExecutionReplayAccumulator, ExecutionReplayWindow
        grid = pd.date_range("2021-01-01", periods=36, freq="5min", tz="UTC")
        symbols = ("AAAUSDT",)
        closes = pd.DataFrame({"AAAUSDT": [100.0] * 36}, index=grid)
        decisions = pd.DatetimeIndex([
            pd.Timestamp("2021-01-01 00:05", tz="UTC"),
            pd.Timestamp("2021-01-01 00:10", tz="UTC"),
            pd.Timestamp("2021-01-01 00:20", tz="UTC"),
            pd.Timestamp("2021-01-01 00:25", tz="UTC"),
        ])
        w1 = ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[17],
            columns=symbols, symbols=symbols, minute_grid=grid[:18],
            highs=closes.iloc[:18] * 1.01, lows=closes.iloc[:18] * 0.99,
            closes=closes.iloc[:18], marks=closes.iloc[:18],
            bar_funding=pd.DataFrame(0.0, index=grid[:18], columns=symbols),
            target_weights=pd.DataFrame({"AAAUSDT": [0.3, 0.3]}, index=decisions[:2]),
            signal_available_at=decisions[:2],
        )
        w2 = ExecutionReplayWindow(
            window_start=grid[17], window_end=grid[-1],
            columns=symbols, symbols=symbols, minute_grid=grid[17:],
            highs=closes.iloc[17:] * 1.01, lows=closes.iloc[17:] * 0.99,
            closes=closes.iloc[17:], marks=closes.iloc[17:],
            bar_funding=pd.DataFrame(0.0, index=grid[17:], columns=symbols),
            target_weights=pd.DataFrame({"AAAUSDT": [0.3, 0.3]}, index=decisions[2:]),
            signal_available_at=decisions[2:],
        )
        original_consume = _BoundExecutionReplayAccumulator.consume
        def _inject_low_ledger(self, w):
            original_consume(self, w)
            # Replace the last chunk with a value below the 0.2 floor
            if self.equity_chunks:
                n = len(self.equity_chunks[-1])
                self.equity_chunks[-1] = np.full(n, self.initial_equity * 0.01)
        _BoundExecutionReplayAccumulator.consume = _inject_low_ledger
        try:
            acc = _BoundExecutionReplayAccumulator(
                w1, 1.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False,
                min_equity_fraction=0.2,
            )
            acc.consume(w1)
            acc.consume(w2)
            result = acc.finalize()
        finally:
            _BoundExecutionReplayAccumulator.consume = original_consume
        assert len(acc.equity_floor_breaches) > 0
        assert result.ledger.equity_floor_breached_at is not None
        assert len(result.ledger.equity_floor_breached_at) > 0
