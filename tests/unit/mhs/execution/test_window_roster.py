"""Roster invariant: every yielded window covers the inventory any bound can carry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.mhs.execution.window_stream as ws
from src.common.errors import DataIntegrityError
from src.mhs.execution.accounting import QTY_EPS
from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
from src.mhs.execution.window_stream import (
    _estimate_mhs_execution_allocation as _real_estimate,
)
from src.mhs.types import ExecutionSpec

SYMBOLS = ("AUSDT", "BUSDT", "CUSDT")
START = pd.Timestamp("2022-01-01", tz="UTC")


def _write_lake(root, days, zero_vol=None) -> None:
    lake = root / "3m"
    lake.mkdir(parents=True, exist_ok=True)
    labels = pd.date_range(START, START + pd.Timedelta(days=days), freq="3min", tz="UTC")
    ms = (labels - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    for k, sym in enumerate(SYMBOLS):
        close = 100.0 + k + 0.01 * np.sin(np.arange(len(labels)) / 50.0)
        qv = np.full(len(labels), 1000.0)
        if zero_vol and sym in zero_vol:
            a, b = zero_vol[sym]
            qv[(labels >= a) & (labels <= b)] = 0.0
        pd.DataFrame(
            {
                "timestamp": ms.to_numpy(dtype="int64"),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "quote_vol": qv,
            }
        ).to_parquet(lake / f"{sym}.parquet")


def _targets(decisions, rows):
    frame = pd.DataFrame(0.0, index=decisions, columns=list(SYMBOLS))
    for i, row in enumerate(rows):
        for sym, value in row.items():
            frame.iloc[i, SYMBOLS.index(sym)] = value
    return frame


def _patch_budget(monkeypatch, planned) -> None:
    monkeypatch.setattr(
        ws,
        "plan_mhs_execution_bars",
        lambda **k: max(k["minimum_bars"], min(planned, k["requested_bars"])),
    )
    monkeypatch.setattr(ws, "assert_mhs_allocation_budget", lambda **k: None)


def _patch_generous_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        ws, "plan_mhs_execution_bars", lambda **k: k["requested_bars"]
    )
    monkeypatch.setattr(ws, "assert_mhs_allocation_budget", lambda **k: None)


def _drive(targets, signals, end, root, *, mode, budget):
    """Consume the stream with a live accumulator, checking held ⊆ roster."""
    spec = ExecutionSpec()
    funding = {
        s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
        for s in SYMBOLS
    }
    holder: dict = {}

    def _live():
        return holder["acc"].required_symbols() if "acc" in holder else frozenset()

    req = _live if mode == "required" else None
    kw = {} if budget is None else {"budget_bytes": 1, "reserve_bytes": None}
    windows = []
    gen = ws._iter_mhs_execution_windows(
        targets, signals, str(root), "3m", START, end, funding, spec,
        required_symbols=req, **kw,
    )
    for w in gen:
        acc = holder.get("acc")
        if acc is None:
            acc = holder["acc"] = _BoundExecutionReplayAccumulator(
                w, 1000.0, "OHLCV_IMMEDIATE_TAKER", spec, False
            )
        held = {acc.columns[i] for i in np.flatnonzero(np.abs(acc.units_arr) >= QTY_EPS)}
        assert held <= set(w.symbols), f"held {sorted(held)} outside roster {list(w.symbols)}"
        assert list(w.symbols) == sorted(w.symbols, key=list(w.columns).index)
        windows.append(w)
        acc.consume(w)
    return windows, holder["acc"].finalize()


def _s1():
    end = START + pd.Timedelta(days=1)
    dec = pd.date_range(START, periods=4, freq="4h", tz="UTC")
    targets = _targets(
        dec,
        [
            {"AUSDT": 0.1, "BUSDT": 0.1},
            {"AUSDT": np.nan, "BUSDT": 0.1},
            {"AUSDT": np.nan, "BUSDT": 0.1},
            {"AUSDT": np.nan, "BUSDT": 0.1},
        ],
    )
    return targets, dec + pd.Timedelta(hours=1), end


def _s2_targets():
    dec = pd.date_range(START, periods=4, freq="4h", tz="UTC")
    targets = _targets(
        dec,
        [
            {"AUSDT": 0.1, "BUSDT": 0.1},
            {"BUSDT": 0.1},
            {"BUSDT": 0.1},
            {"BUSDT": 0.1},
        ],
    )
    return targets, dec + pd.Timedelta(hours=1), dec


def test_nan_hold_keeps_symbol_in_gap_pieces(tmp_path, monkeypatch) -> None:
    """S1: NaN-held A stays rostered through budgeted gap pieces in both modes."""
    _patch_budget(monkeypatch, 40)
    targets, signals, end = _s1()
    _write_lake(tmp_path, 1)
    for mode in ("fallback", "required"):
        windows, result = _drive(targets, signals, end, tmp_path, mode=mode, budget=1)
        assert result.ledger.primary_valid
        assert len(windows) == 13
        assert all("AUSDT" in w.symbols for w in windows)


def test_blocked_exit_stays_covered(tmp_path, monkeypatch) -> None:
    """S2: A exit blocked by zero volume stays covered; genuine gap fails the ledger."""
    _patch_budget(monkeypatch, 40)
    targets, signals, dec = _s2_targets()
    end = START + pd.Timedelta(days=1)
    _write_lake(tmp_path, 1, zero_vol={"AUSDT": (dec[1], dec[1] + pd.Timedelta(hours=2))})
    windows, result = _drive(targets, signals, end, tmp_path, mode="fallback", budget=1)
    first = next(i for i, w in enumerate(windows) if "AUSDT" in w.symbols)
    assert all("AUSDT" in w.symbols for w in windows[first:])
    assert not result.ledger.primary_valid
    assert "KNOWN_ZERO_VOLUME" in {g.code for g in result.data_gaps}


def test_tail_pieces_carry_held_symbol(tmp_path, monkeypatch) -> None:
    """S4: budgeted tail pieces keep NaN-held A and the replay finalizes valid."""
    _patch_budget(monkeypatch, 40)
    end = START + pd.Timedelta(days=1)
    _write_lake(tmp_path, 1)
    dec = pd.date_range(START, periods=2, freq="2h", tz="UTC")
    targets = _targets(
        dec,
        [{"AUSDT": 0.1, "BUSDT": 0.1}, {"AUSDT": np.nan, "BUSDT": 0.1}],
    )
    windows, result = _drive(
        targets, dec + pd.Timedelta(hours=1), end, tmp_path, mode="fallback", budget=1
    )
    tails = [w for w in windows if len(w.target_weights) == 0]
    assert tails
    assert all("AUSDT" in w.symbols for w in tails)
    assert result.ledger.primary_valid


def test_unbudgeted_partitions_carry_nan_hold(tmp_path) -> None:
    """S3: all three 31-day partitions roster NaN-held A without a budget."""
    end = START + pd.Timedelta(days=70)
    _write_lake(tmp_path, 70)
    dec = pd.date_range(START, periods=68, freq="1D", tz="UTC")
    rows = [{"AUSDT": 0.1, "BUSDT": 0.1}] + [{"AUSDT": np.nan, "BUSDT": 0.1}] * 67
    windows, result = _drive(
        _targets(dec, rows), dec + pd.Timedelta(hours=1), end, tmp_path,
        mode="fallback", budget=None,
    )
    assert len({w.logical_partition for w in windows}) == 3
    assert all("AUSDT" in w.symbols for w in windows)
    assert result.ledger.primary_valid


def test_single_budgeted_piece_reuses_planning_snapshot(tmp_path, monkeypatch) -> None:
    """B2/B3: one live-callback call per partition; piece roster equals planning roster."""
    _patch_generous_budget(monkeypatch)
    end = START + pd.Timedelta(days=70)
    _write_lake(tmp_path, 70)
    dec = pd.date_range(START, periods=68, freq="1D", tz="UTC")
    rows = [{"AUSDT": 0.1, "BUSDT": 0.1}] + [{"AUSDT": np.nan, "BUSDT": 0.1}] * 67
    targets = _targets(dec, rows)
    signals = dec + pd.Timedelta(hours=1)
    funding = {
        s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
        for s in SYMBOLS
    }
    calls = []
    widths: list[int] = []

    def _counting():
        calls.append(1)
        return frozenset({"CUSDT"})

    def _recording(*, n_symbols, n_columns, bound_count, _widths=widths):
        _widths.append(n_symbols)
        return _real_estimate(
            n_symbols=n_symbols, n_columns=n_columns, bound_count=bound_count
        )

    monkeypatch.setattr(ws, "_estimate_mhs_execution_allocation", _recording)
    windows = list(
        ws._iter_mhs_execution_windows(
            targets, signals, str(tmp_path), "3m", START, end, funding, ExecutionSpec(),
            required_symbols=_counting, budget_bytes=1, reserve_bytes=None,
        )
    )
    assert len({w.logical_partition for w in windows}) == 3
    assert len(calls) == 3
    assert [list(w.symbols) for w in windows] == [
        ["AUSDT", "BUSDT", "CUSDT"],
        ["AUSDT", "BUSDT", "CUSDT"],
        ["BUSDT", "CUSDT"],
    ]
    assert widths == [3, 3, 3, 3, 2, 2]


def test_planning_roster_bounds_every_piece(tmp_path, monkeypatch) -> None:
    """B2: the planning width covers every piece roster in its partition, both modes."""
    _patch_budget(monkeypatch, 40)
    _write_lake(tmp_path, 1)
    for targets, signals, end in (_s1(), (*_s2_targets()[:2], START + pd.Timedelta(days=1))):
        for mode in ("fallback", "required"):
            widths: list[int] = []

            def _recording(*, n_symbols, n_columns, bound_count, _widths=widths):
                _widths.append(n_symbols)
                return _real_estimate(
                    n_symbols=n_symbols, n_columns=n_columns, bound_count=bound_count
                )

            monkeypatch.setattr(ws, "_estimate_mhs_execution_allocation", _recording)

            def _req():
                return frozenset({"CUSDT"})

            req = _req if mode == "required" else None
            spec = ExecutionSpec()
            funding = {
                s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
                for s in SYMBOLS
            }
            windows = list(
                ws._iter_mhs_execution_windows(
                    targets, signals, str(tmp_path), "3m", START, end, funding, spec,
                    required_symbols=req, budget_bytes=1, reserve_bytes=None,
                )
            )
            assert widths
            assert windows
            assert len(widths) == 1 + len(windows)
            assert all(len(w.symbols) <= widths[0] for w in windows)


def test_union_rule_is_uniform_across_branches(tmp_path, monkeypatch) -> None:
    """One union rule for gap, decision and tail pieces in required mode."""
    _patch_budget(monkeypatch, 40)
    targets, signals, end = _s1()
    _write_lake(tmp_path, 1)
    spec = ExecutionSpec()
    funding = {
        s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
        for s in SYMBOLS
    }
    columns = list(targets.columns)
    windows = list(
        ws._iter_mhs_execution_windows(
            targets, signals, str(tmp_path), "3m", START, end, funding, spec,
            required_symbols=lambda: frozenset({"CUSDT"}),
            budget_bytes=1, reserve_bytes=None,
        )
    )
    kinds = set()
    prev: set[str] = set()
    for w in windows:
        if len(w.target_weights):
            kinds.add("decision")
            piece_active = set(
                w.target_weights.columns[
                    (w.target_weights.notna() & w.target_weights.ne(0.0)).any(axis=0)
                ]
            )
            expected = piece_active | prev | {"CUSDT"}
            prev = set(piece_active)
        else:
            kinds.add("gap/tail")
            expected = prev | {"CUSDT"}
        assert list(w.symbols) == [s for s in columns if s in expected]
    assert kinds == {"decision", "gap/tail"}


@pytest.mark.parametrize("budget", ["unbudgeted", "single", "split"])
def test_unknown_required_symbol_fails_closed_once(tmp_path, monkeypatch, budget) -> None:
    """A callback naming an unknown symbol fails closed on the first piece."""
    if budget == "single":
        _patch_generous_budget(monkeypatch)
    elif budget == "split":
        _patch_budget(monkeypatch, 40)
    targets, signals, end = _s1()
    _write_lake(tmp_path, 1)
    spec = ExecutionSpec()
    funding = {
        s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
        for s in SYMBOLS
    }
    kw = {}
    if budget != "unbudgeted":
        kw = {"budget_bytes": 1, "reserve_bytes": None}
    gen = ws._iter_mhs_execution_windows(
        targets, signals, str(tmp_path), "3m", START, end, funding, spec,
        required_symbols=lambda: frozenset({"ZZZUSDT"}), **kw,
    )
    with pytest.raises(DataIntegrityError, match="not in canonical columns"):
        list(gen)


def test_fallback_stream_is_consumer_independent(tmp_path, monkeypatch) -> None:
    """Fallback output is a pure function of the target path for every bound."""
    _patch_budget(monkeypatch, 40)
    from src.mhs.execution import replay_execution_window_batch
    from src.mhs.evaluation.windows import _rescaled_windows

    targets, signals, dec = _s2_targets()
    end = START + pd.Timedelta(days=1)
    _write_lake(tmp_path, 1)
    spec = ExecutionSpec()

    def _generate():
        funding = {
            s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
            for s in SYMBOLS
        }
        return list(
            ws._iter_mhs_execution_windows(
                targets, signals, str(tmp_path), "3m", START, end, funding, spec,
                budget_bytes=1, reserve_bytes=None,
            )
        )

    first, second = _generate(), _generate()
    assert len(first) == len(second) > 0
    for a, b in zip(first, second, strict=True):
        assert list(a.symbols) == list(b.symbols)
        assert a.minute_grid.equals(b.minute_grid)
        assert a.logical_partition == b.logical_partition
        assert a.window_start == b.window_start
        assert a.window_end == b.window_end

    bounds = [("OHLCV_IMMEDIATE_TAKER", spec), ("OHLCV_STRICT_PROXY", spec)]
    assert len(replay_execution_window_batch(iter(second), 1000.0, bounds)) == 2
    scale = pd.Series(0.5, index=targets.index)
    rescaled = list(_rescaled_windows(iter(_generate()), scale))
    assert len(replay_execution_window_batch(iter(rescaled), 1000.0, bounds)) == 2


def test_never_targeted_columns_stay_out(tmp_path, monkeypatch) -> None:
    """A column with only 0/NaN targets never enters any fallback roster."""
    _patch_budget(monkeypatch, 40)
    end = START + pd.Timedelta(days=1)
    _write_lake(tmp_path, 1)
    dec = pd.date_range(START, periods=4, freq="4h", tz="UTC")
    targets = _targets(
        dec,
        [
            {"AUSDT": 0.1, "BUSDT": 0.1, "CUSDT": np.nan},
            {"AUSDT": 0.1, "BUSDT": 0.1, "CUSDT": 0.0},
            {"AUSDT": 0.1, "BUSDT": 0.1, "CUSDT": np.nan},
            {"BUSDT": 0.1, "CUSDT": 0.0},
        ],
    )
    spec = ExecutionSpec()
    funding = {
        s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC"))
        for s in SYMBOLS
    }
    windows = list(
        ws._iter_mhs_execution_windows(
            targets, dec + pd.Timedelta(hours=1), str(tmp_path), "3m",
            START, end, funding, spec, budget_bytes=1, reserve_bytes=None,
        )
    )
    assert windows
    assert all("CUSDT" not in w.symbols for w in windows)


def test_first_active_ordinals_marks_never_targeted_at_end() -> None:
    """Helper contract: first finite-nonzero ordinal per column, else n_rows."""
    idx = pd.date_range("2022-01-01", periods=3, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {"AUSDT": [0.0, 0.5, np.nan], "BUSDT": [np.nan, np.nan, np.nan]},
        index=idx,
    )
    np.testing.assert_array_equal(
        ws._first_active_ordinals(frame), np.array([1, 3], dtype=np.int64)
    )
    assert list(ws._piece_roster(("AUSDT", "BUSDT"), piece_active=frozenset({"BUSDT"}), previous_active=frozenset({"AUSDT"}), requirement=frozenset())) == ["AUSDT", "BUSDT"]


def test_settled_symbol_leaves_target_only_roster(tmp_path, monkeypatch) -> None:
    """Target-only mode drops a delivered symbol with exact-zero later targets."""
    import pandas as pd
    from src.mhs.instrument_settlements import InstrumentSettlementRecord, assemble_instrument_settlement_registry
    _patch_budget(monkeypatch, 200)
    dec = pd.date_range(START, periods=4, freq="12h", tz="UTC")
    targets = _targets(dec, [{"AUSDT": 0.5}, {"AUSDT": 0.0}, {"AUSDT": 0.0}, {"BUSDT": 0.2}])
    delivery = dec[1].floor("3min")
    last_trade = delivery - pd.Timedelta(minutes=3)
    end = START + pd.Timedelta(days=3)
    _write_lake(tmp_path, 3, zero_vol={"AUSDT": (last_trade, delivery)})
    announced = last_trade - pd.Timedelta(days=7)
    record = InstrumentSettlementRecord(symbol="AUSDT", event_id=f"AUSDT:{int(delivery.value // 1_000_000)}", announced_at=announced, announcement_source="proxy_lead", announcement_evidence="", last_trade_at=last_trade, delivery_at=delivery, settlement_price=100.0, price_source="twap30_proxy", price_evidence="lake", fee_bps=5.0, evidence_digest="sha256:x", verified_at=delivery)
    registry = assemble_instrument_settlement_registry([record], [])
    spec = ExecutionSpec()
    funding = {s: pd.Series(0.0, index=pd.date_range(START, end, freq="3min", tz="UTC")) for s in SYMBOLS}
    windows = list(ws._iter_mhs_execution_windows(targets, dec + pd.Timedelta(hours=1), str(tmp_path), "3m", START, end, funding, spec, settlement_registry=registry))
    assert windows
    assert any("AUSDT" in w.symbols for w in windows[:1])
    assert all("AUSDT" not in w.symbols for w in windows[1:])
    from src.mhs.execution import replay_execution_windows
    result = replay_execution_windows(iter(windows), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    assert result.ledger.primary_valid
