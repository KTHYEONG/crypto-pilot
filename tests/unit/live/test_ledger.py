"""내부 포지션 원장: LedgerState 로드/저장(원자적)/체결 반영 및 reconcile_or_halt 연동."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.live.account import AccountSnapshot, reconcile_or_halt
from src.live.ledger import (
    LedgerState,
    default_ledger_path,
    load_ledger,
    save_ledger,
)
from src.live.planner import OrderIntent


def _intent(symbol: str, side: str, qty: str) -> OrderIntent:
    return OrderIntent(
        symbol=symbol, side=side, quantity=Decimal(qty), reduce_only=False,
        target_qty=Decimal(qty), current_qty=Decimal(0), client_order_prefix="run1",
        leg_index=0, decision_price=Decimal("100"),
    )


def test_load_ledger_missing_file_returns_empty(tmp_path: Path) -> None:
    state = load_ledger(tmp_path / "nope.json")
    assert state.positions == {}
    assert state.equity_high_water_mark == 0


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    state = LedgerState(
        positions={"BTCUSDT": Decimal("1.5"), "ETHUSDT": Decimal("-2")},
        equity_high_water_mark=Decimal("2500"),
    )
    save_ledger(path, state)
    loaded = load_ledger(path)
    assert loaded == state


def test_load_ledger_promotes_legacy_flat_layout(tmp_path: Path) -> None:
    """레거시 평면 dict({symbol: qty})는 positions 로 읽고 hwm=0 으로 승격한다."""
    path = tmp_path / "legacy_ledger.json"
    path.write_text('{"BTCUSDT": "1.5", "ETHUSDT": "-2"}', encoding="utf-8")
    loaded = load_ledger(path)
    assert loaded == LedgerState(
        positions={"BTCUSDT": Decimal("1.5"), "ETHUSDT": Decimal("-2")},
        equity_high_water_mark=Decimal("0"),
    )


def test_save_ledger_drops_zero_positions_and_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    save_ledger(
        path,
        LedgerState(
            positions={"BTCUSDT": Decimal("0"), "ETHUSDT": Decimal("3")},
            equity_high_water_mark=Decimal("100"),
        ),
    )
    assert load_ledger(path) == LedgerState(
        positions={"ETHUSDT": Decimal("3")}, equity_high_water_mark=Decimal("100")
    )
    # 원자적 기록: 임시 파일 잔존물이 없어야 한다.
    assert list(tmp_path.iterdir()) == [path]


def test_load_ledger_corrupt_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_ledger(path)


def test_default_ledger_path_under_data_state() -> None:
    path = default_ledger_path()
    assert path.parts[-2:] == ("state", "live_position_ledger.json")


def test_reconcile_uses_loaded_ledger_not_hardcoded_empty(tmp_path: Path) -> None:
    """§1.4 I-RECONCILE-FIRST: 원장에 기록된 포지션이 스냅샷과 다르면 breach여야 한다."""
    path = tmp_path / "ledger.json"
    save_ledger(path, LedgerState(positions={"BTCUSDT": Decimal("1")}))
    ledger_positions = load_ledger(path).positions
    snapshot = AccountSnapshot(
        taken_at=__import__("pandas").Timestamp.now(tz="UTC"),
        wallet_balance=Decimal("1000"), available_balance=Decimal("1000"),
        total_maint_margin=Decimal("0"), unrealized_pnl=Decimal("0"),
        positions={"BTCUSDT": Decimal("0")},
        dual_side_position=False, multi_assets_margin=False,
    )
    from src.live.errors import ReconciliationBreach

    with pytest.raises(ReconciliationBreach):
        reconcile_or_halt(snapshot, ledger_positions, qty_tolerance_fraction=0.001)


def test_SCENARIO_LIVE_43_LEDGER_CASH_ROUND_TRIPS_AND_STAYS_BACKWARD_COMPATIBLE(tmp_path: Path) -> None:
    """SCENARIO_LIVE_43: cash_usdt round-trips; None omits the key entirely
    (byte-compatible with pre-cash ledgers); legacy flat layout yields None."""
    path = tmp_path / "ledger.json"
    state = LedgerState(
        positions={"AAAUSDT": Decimal("0.4")},
        equity_high_water_mark=Decimal("2000"),
        cash_usdt=Decimal("1234.5"),
    )
    save_ledger(path, state)
    assert load_ledger(path) == state

    none_path = tmp_path / "ledger_none.json"
    save_ledger(none_path, LedgerState(positions={}, equity_high_water_mark=Decimal("0"), cash_usdt=None))
    raw = json.loads(none_path.read_text(encoding="utf-8"))
    assert "cash_usdt" not in raw
    assert load_ledger(none_path).cash_usdt is None

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text('{"BTCUSDT": "1.5"}', encoding="utf-8")
    assert load_ledger(legacy_path).cash_usdt is None


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_43_LEDGER_CASH_ROUND_TRIPS_AND_STAYS_BACKWARD_COMPATIBLE",
    "SCENARIO_LIVE_44_FILL_CASH_FLOW_SIGN_CONVENTION",
)

def test_ledger_roundtrip_funding_accrued_through(tmp_path) -> None:
    import json
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import LedgerState, load_ledger, save_ledger

    path = tmp_path / "ledger.json"
    state = LedgerState(positions={"AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("1900"), funding_accrued_through=pd.Timestamp("2026-09-01 01:03", tz="UTC"))
    save_ledger(path, state)
    assert load_ledger(path) == state
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("funding_accrued_through")
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_ledger(path).funding_accrued_through is None


def test_ledger_corrupt_numerics_fail_closed(tmp_path) -> None:
    import json
    from src.common.errors import DataIntegrityError
    from src.live.ledger import load_ledger

    base = {"positions": {"AAAUSDT": "1"}, "equity_high_water_mark": "2000", "cash_usdt": "1900"}
    cases = [
        {**base, "equity_high_water_mark": "abc"},
        {**base, "cash_usdt": "abc"},
        {**base, "funding_accrued_through": "not-a-time"},
        {"positions": {"AAAUSDT": "abc"}, "equity_high_water_mark": "2000"},
    ]
    for i, payload in enumerate(cases):
        path = tmp_path / f"bad_{i}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(DataIntegrityError):
            load_ledger(path)


def test_ledger_round_trips_last_executed_decision_time(tmp_path) -> None:
    import json
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import LedgerState, load_ledger, save_ledger

    path = tmp_path / "ledger.json"
    executed = pd.Timestamp("2026-09-14 00:00Z")
    save_ledger(path, LedgerState(positions={"AAAUSDT": Decimal("1.5")}, last_executed_decision_time=executed))

    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_ledger(path)

    assert raw["last_executed_decision_time"] == "2026-09-14T00:00:00+00:00"
    assert loaded.last_executed_decision_time == executed
    assert loaded.positions == {"AAAUSDT": Decimal("1.5")}


def test_ledger_legacy_file_without_last_executed_loads_none(tmp_path) -> None:
    import json
    from src.live.ledger import load_ledger, save_ledger

    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"positions": {"AAAUSDT": "2"}, "equity_high_water_mark": "2000"}), encoding="utf-8")

    loaded = load_ledger(path)
    save_ledger(path, loaded)

    assert loaded.last_executed_decision_time is None
    assert "last_executed_decision_time" not in json.loads(path.read_text(encoding="utf-8"))


def test_ledger_rejects_invalid_last_executed_decision_time(tmp_path) -> None:
    import json
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.ledger import load_ledger

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"positions": {}, "last_executed_decision_time": "not-a-time"}), encoding="utf-8")
    naive = tmp_path / "naive.json"
    naive.write_text(json.dumps({"positions": {}, "last_executed_decision_time": "2026-09-14T00:00:00"}), encoding="utf-8")

    with pytest.raises(DataIntegrityError):
        load_ledger(bad)
    with pytest.raises(DataIntegrityError):
        load_ledger(naive)



def test_ledger_roundtrip_funding_watermarks_and_position_history(tmp_path) -> None:
    import json
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import LedgerState, PositionSnapshot, load_ledger, save_ledger

    path = tmp_path / "ledger.json"
    state = LedgerState(
        positions={"AUSDT": Decimal("2")},
        equity_high_water_mark=Decimal("2000"),
        cash_usdt=Decimal("1900"),
        funding_watermarks={"AUSDT": pd.Timestamp("2026-09-01 08:00:00.004", tz="UTC")},
        position_history=(
            PositionSnapshot(effective_from=pd.Timestamp("2026-08-31 01:26", tz="UTC"), positions={"AUSDT": Decimal("1")}),
            PositionSnapshot(effective_from=pd.Timestamp("2026-09-01 01:26", tz="UTC"), positions={"AUSDT": Decimal("2")}),
        ),
    )

    save_ledger(path, state)

    assert load_ledger(path) == state
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("funding_watermarks")
    raw.pop("position_history")
    path.write_text(json.dumps(raw), encoding="utf-8")
    legacy = load_ledger(path)
    assert legacy.funding_watermarks == {}
    assert legacy.position_history == ()


def test_load_ledger_rejects_malformed_watermarks_and_history(tmp_path) -> None:
    import json
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.ledger import load_ledger

    base = {"positions": {"AUSDT": "1"}, "equity_high_water_mark": "2000", "cash_usdt": "1900"}
    cases = [
        ({"funding_watermarks": ["AUSDT"]}, "funding_watermarks"),
        ({"funding_watermarks": {"AUSDT": "2026-09-01 08:00"}}, "funding_watermarks"),
        ({"funding_watermarks": {"AUSDT": "not-a-time"}}, "funding_watermarks"),
        ({"position_history": {"effective_from": "2026-09-01T00:00:00+00:00"}}, "position_history"),
        ({"position_history": [{"positions": {"AUSDT": "1"}}]}, "position_history"),
        ({"position_history": [{"effective_from": "2026-09-01T00:00:00+00:00", "positions": ["AUSDT"]}]}, "position_history"),
        ({"position_history": [{"effective_from": "2026-09-01T00:00:00+00:00", "positions": {"AUSDT": "x"}}]}, "position_history"),
    ]
    for extra, name in cases:
        path = tmp_path / f"case_{name}_{len(str(extra))}.json"
        path.write_text(json.dumps({**base, **extra}), encoding="utf-8")
        with pytest.raises(DataIntegrityError, match=name):
            load_ledger(path)


def test_append_position_snapshot_dedupes_filters_zero_and_bounds() -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    from src.live.ledger import (
        PositionSnapshot,
        append_position_snapshot,
    )
    from src.live.ledger import POSITION_HISTORY_MAX

    t = pd.Timestamp("2026-09-01 01:26", tz="UTC")
    first = append_position_snapshot((), t, {"AUSDT": Decimal("1"), "BUSDT": Decimal("0")})
    assert first == (PositionSnapshot(effective_from=t, positions={"AUSDT": Decimal("1")}),)
    assert append_position_snapshot(first, t + pd.Timedelta(days=1), {"AUSDT": Decimal("1")}) == first

    history = first
    for day in range(1, 7):
        history = append_position_snapshot(history, t + pd.Timedelta(days=day), {"AUSDT": Decimal(day + 1)})
    assert POSITION_HISTORY_MAX == 4
    assert len(history) == 4
    assert history[-1].positions == {"AUSDT": Decimal("7")}
    assert [snap.effective_from for snap in history] == sorted(snap.effective_from for snap in history)

    with pytest.raises(ValueError, match="tz-aware"):
        append_position_snapshot((), pd.Timestamp("2026-09-01 01:26"), {"AUSDT": Decimal("1")})


def test_position_at_uses_snapshot_effective_strictly_before_epoch() -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.ledger import (
        append_position_snapshot,
        position_at,
    )

    t0 = pd.Timestamp("2026-09-01 01:26", tz="UTC")
    t1 = pd.Timestamp("2026-09-02 01:26", tz="UTC")
    history = append_position_snapshot(append_position_snapshot((), t0, {"AUSDT": Decimal("2")}), t1, {"BUSDT": Decimal("-1")})

    assert position_at(history, "AUSDT", pd.Timestamp("2026-09-02 00:00", tz="UTC")) == Decimal("2")
    assert position_at(history, "AUSDT", t1) == Decimal("2")
    assert position_at(history, "AUSDT", pd.Timestamp("2026-09-02 04:00", tz="UTC")) == Decimal("0")
    assert position_at(history, "BUSDT", pd.Timestamp("2026-09-02 04:00", tz="UTC")) == Decimal("-1")
    with pytest.raises(DataIntegrityError, match="predates position history"):
        position_at(history, "AUSDT", pd.Timestamp("2026-09-01 00:00", tz="UTC"))


def test_accrue_funding_by_watermark_late_data_matches_full_information() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import (
        accrue_funding_by_watermark,
        append_position_snapshot,
    )

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    t1 = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    t2 = pd.Timestamp("2026-09-03 01:30", tz="UTC")
    a_epochs = pd.date_range("2026-09-01 04:00", "2026-09-03 00:00", freq="4h", tz="UTC")
    b_epochs = pd.date_range("2026-09-01 08:00", "2026-09-03 00:00", freq="8h", tz="UTC")
    funding = {
        "AUSDT": pd.Series([round(0.0001 * (i + 1), 4) for i in range(len(a_epochs))], index=a_epochs),
        "BUSDT": pd.Series([-0.0002] * len(b_epochs), index=b_epochs),
    }
    bars = pd.date_range("2026-09-01 00:00", "2026-09-03 01:00", freq="1h", tz="UTC")
    marks = {
        "AUSDT": pd.Series([100.0 + i for i in range(len(bars))], index=bars),
        "BUSDT": pd.Series([50.0] * len(bars), index=bars),
    }
    day1 = append_position_snapshot((), t0, {"AUSDT": Decimal("2"), "BUSDT": Decimal("-1")})
    day2 = append_position_snapshot(day1, t1, {"AUSDT": Decimal("1"), "BUSDT": Decimal("-1")})

    full = accrue_funding_by_watermark(day2, {}, funding, marks, t2)

    cut = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    outage = {symbol: series[series.index <= cut] for symbol, series in funding.items()}
    first = accrue_funding_by_watermark(day1, {}, outage, marks, t1)
    second = accrue_funding_by_watermark(day2, first.watermarks, funding, marks, t2)

    assert first.watermarks == {"AUSDT": cut, "BUSDT": cut}
    assert first.lag_by_symbol == {"AUSDT": t1 - cut, "BUSDT": t1 - cut}
    assert first.cash_delta + second.cash_delta == full.cash_delta
    assert full.watermarks == {"AUSDT": pd.Timestamp("2026-09-03 00:00", tz="UTC"), "BUSDT": pd.Timestamp("2026-09-03 00:00", tz="UTC")}
    expected = Decimal(0)
    for epoch, rate in funding["AUSDT"].items():
        qty = Decimal("2") if epoch < t1 else Decimal("1")
        expected += -(Decimal(str(rate)) * qty * Decimal(str(marks["AUSDT"].loc[epoch])))
    for rate in funding["BUSDT"]:
        expected += -(Decimal(str(rate)) * Decimal("-1") * Decimal("50.0"))
    assert full.cash_delta == expected


def test_accrue_funding_by_watermark_is_idempotent_on_rerun() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import (
        accrue_funding_by_watermark,
        append_position_snapshot,
    )

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    t1 = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    t2 = pd.Timestamp("2026-09-03 01:30", tz="UTC")
    a_epochs = pd.date_range("2026-09-01 04:00", "2026-09-03 00:00", freq="4h", tz="UTC")
    b_epochs = pd.date_range("2026-09-01 08:00", "2026-09-03 00:00", freq="8h", tz="UTC")
    funding = {
        "AUSDT": pd.Series([round(0.0001 * (i + 1), 4) for i in range(len(a_epochs))], index=a_epochs),
        "BUSDT": pd.Series([-0.0002] * len(b_epochs), index=b_epochs),
    }
    bars = pd.date_range("2026-09-01 00:00", "2026-09-03 01:00", freq="1h", tz="UTC")
    marks = {
        "AUSDT": pd.Series([100.0 + i for i in range(len(bars))], index=bars),
        "BUSDT": pd.Series([50.0] * len(bars), index=bars),
    }
    day1 = append_position_snapshot((), t0, {"AUSDT": Decimal("2"), "BUSDT": Decimal("-1")})
    day2 = append_position_snapshot(day1, t1, {"AUSDT": Decimal("1"), "BUSDT": Decimal("-1")})

    once = accrue_funding_by_watermark(day2, {}, funding, marks, t2)
    again = accrue_funding_by_watermark(day2, once.watermarks, funding, marks, t2)

    assert once.cash_delta != 0
    assert again.cash_delta == Decimal(0)
    assert again.watermarks == once.watermarks


def test_accrue_funding_by_watermark_stops_at_epoch_without_mark() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import (
        accrue_funding_by_watermark,
        append_position_snapshot,
    )

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-01 17:00", tz="UTC")
    history = append_position_snapshot((), t0, {"AUSDT": Decimal("1")})
    e1 = pd.Timestamp("2026-09-01 08:00:00.004", tz="UTC")
    e2 = pd.Timestamp("2026-09-01 16:00:00.002", tz="UTC")
    funding = {"AUSDT": pd.Series([0.001, 0.002], index=pd.DatetimeIndex([e1, e2]))}
    marks = {"AUSDT": pd.Series([200.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-01 08:00", tz="UTC")]))}

    result = accrue_funding_by_watermark(history, {}, funding, marks, now)

    assert result.cash_delta == Decimal("-0.2")
    assert result.watermarks == {"AUSDT": e1}
    assert result.lag_by_symbol == {"AUSDT": now - e1}


def test_accrue_funding_by_watermark_closed_symbol_stops_at_delivery() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import (
        accrue_funding_by_watermark,
        append_position_snapshot,
    )

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    delivery = pd.Timestamp("2026-09-01 12:00", tz="UTC")
    history = append_position_snapshot((), t0, {"AUSDT": Decimal("1")})
    epochs = pd.DatetimeIndex(pd.to_datetime(["2026-09-01 08:00", "2026-09-01 16:00", "2026-09-02 00:00"], utc=True))
    funding = {"AUSDT": pd.Series([0.001, 0.5, 0.5], index=epochs)}
    marks = {"AUSDT": pd.Series([100.0, 100.0, 100.0], index=epochs)}

    result = accrue_funding_by_watermark(history, {}, funding, marks, now, closed_at={"AUSDT": delivery})

    assert result.cash_delta == Decimal("-0.1")
    assert result.watermarks == {"AUSDT": pd.Timestamp("2026-09-01 08:00", tz="UTC")}
    assert result.lag_by_symbol == {}


def test_accrue_funding_by_watermark_released_symbol_owes_until_settled() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import (
        accrue_funding_by_watermark,
        append_position_snapshot,
    )

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    t1 = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-02 05:00", tz="UTC")
    history = append_position_snapshot(append_position_snapshot((), t0, {"AUSDT": Decimal("1")}), t1, {})
    e1 = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    e2 = pd.Timestamp("2026-09-02 00:00", tz="UTC")
    e3 = pd.Timestamp("2026-09-02 04:00", tz="UTC")
    marks = {"AUSDT": pd.Series([100.0, 100.0], index=pd.DatetimeIndex([e1, e2]))}

    owed = accrue_funding_by_watermark(history, {"AUSDT": e1}, {"AUSDT": pd.Series([0.001], index=pd.DatetimeIndex([e1]))}, marks, now)
    assert owed.cash_delta == Decimal(0)
    assert owed.watermarks == {"AUSDT": e1}
    assert owed.lag_by_symbol == {"AUSDT": now - e1}

    settled = accrue_funding_by_watermark(history, {"AUSDT": e1}, {"AUSDT": pd.Series([0.001, 0.002, 0.003], index=pd.DatetimeIndex([e1, e2, e3]))}, marks, now)
    assert settled.cash_delta == Decimal("-0.2")
    assert settled.watermarks == {}
    assert settled.lag_by_symbol == {}

    never_held = accrue_funding_by_watermark(history, {"ZUSDT": e1}, {}, {}, now)
    assert never_held.watermarks == {}


def test_accrue_funding_by_watermark_infers_interval_and_seeds_from_holding_start() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import (
        accrue_funding_by_watermark,
        append_position_snapshot,
    )

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-01 09:00", tz="UTC")
    history = append_position_snapshot((), t0, {"AUSDT": Decimal("1"), "BUSDT": Decimal("1"), "CUSDT": Decimal("1")})
    four = pd.DatetimeIndex(pd.to_datetime(["2026-09-01 00:00:00.003", "2026-09-01 04:00", "2026-09-01 08:00:00.001"], utc=True, format="ISO8601"))
    funding = {
        "AUSDT": pd.Series([0.9, 0.001, 0.001], index=four),
        "BUSDT": pd.Series([0.001], index=pd.DatetimeIndex([pd.Timestamp("2026-09-01 08:00", tz="UTC")])),
    }
    bars = pd.DatetimeIndex(pd.to_datetime(["2026-09-01 00:00", "2026-09-01 04:00", "2026-09-01 08:00"], utc=True))
    marks = {"AUSDT": pd.Series([10.0, 10.0, 10.0], index=bars), "BUSDT": pd.Series([10.0, 10.0, 10.0], index=bars)}

    result = accrue_funding_by_watermark(history, {}, funding, marks, now)

    assert result.cash_delta == Decimal("-0.030")
    assert result.interval_by_symbol == {"AUSDT": pd.Timedelta(hours=4), "BUSDT": pd.Timedelta(hours=8), "CUSDT": pd.Timedelta(hours=8)}
    assert result.watermarks["CUSDT"] == t0
    assert result.lag_by_symbol["CUSDT"] == now - t0


def test_accrue_funding_by_watermark_new_symbol_seeds_from_reacquisition() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import accrue_funding_by_watermark, append_position_snapshot

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    t1 = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-02 09:00", tz="UTC")
    history = append_position_snapshot(
        append_position_snapshot((), t0, {"BUSDT": Decimal("1")}),
        t1,
        {"AUSDT": Decimal("2"), "BUSDT": Decimal("1")},
    )
    epoch = pd.Timestamp("2026-09-02 04:00", tz="UTC")
    funding = {"AUSDT": pd.Series([0.001], index=pd.DatetimeIndex([epoch]))}
    marks = {"AUSDT": pd.Series([100.0], index=pd.DatetimeIndex([epoch]))}

    result = accrue_funding_by_watermark(history, {}, funding, marks, now)

    assert result.cash_delta == Decimal("-0.2")
    assert result.watermarks["AUSDT"] == epoch


def test_accrue_funding_by_watermark_skips_zero_holding_without_watermark() -> None:
    from decimal import Decimal
    import pandas as pd
    from src.live.ledger import PositionSnapshot, accrue_funding_by_watermark

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-02 09:00", tz="UTC")
    history = (PositionSnapshot(effective_from=t0, positions={"AUSDT": Decimal(0)}),)

    result = accrue_funding_by_watermark(history, {}, {}, {}, now)

    assert result.cash_delta == Decimal(0)
    assert result.watermarks == {}
    assert result.lag_by_symbol == {}


def test_accrue_funding_events_sum_exactly_to_cash_delta() -> None:
    """Two held symbols across three funding epochs: event amounts sum to cash_delta exactly."""
    from decimal import Decimal

    import pandas as pd

    from src.live.ledger import accrue_funding_by_watermark, append_position_snapshot

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    epochs = pd.DatetimeIndex(
        ["2026-09-01 08:00", "2026-09-01 16:00", "2026-09-02 00:00"], tz="UTC"
    )
    funding = {
        "AUSDT": pd.Series([0.0001, 0.0002, 0.0003], index=epochs),
        "BUSDT": pd.Series([-0.0002, -0.0002, -0.0002], index=epochs),
    }
    closes = {
        "AUSDT": pd.Series([100.0, 101.0, 102.0], index=epochs),
        "BUSDT": pd.Series([50.0, 50.0, 50.0], index=epochs),
    }
    history = append_position_snapshot((), t0, {"AUSDT": Decimal("2"), "BUSDT": Decimal("-1")})

    result = accrue_funding_by_watermark(history, {}, funding, closes, now)

    assert len(result.events) == 6
    assert {(e.symbol, e.epoch) for e in result.events} == {
        (s, e) for s in ("AUSDT", "BUSDT") for e in epochs
    }
    assert [(e.symbol, e.epoch) for e in result.events] == sorted(
        [(e.symbol, e.epoch) for e in result.events]
    )
    assert sum((e.amount for e in result.events), Decimal(0)) == result.cash_delta
    assert all(e.price_source == "trade_close_1h" for e in result.events)


def test_accrue_funding_events_stop_where_watermark_stops_on_missing_price() -> None:
    """Second epoch hourly close missing: only the first epoch produces an event."""
    from decimal import Decimal

    import pandas as pd

    from src.live.ledger import accrue_funding_by_watermark, append_position_snapshot

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-01 17:00", tz="UTC")
    e1 = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    e2 = pd.Timestamp("2026-09-01 16:00", tz="UTC")
    history = append_position_snapshot((), t0, {"AUSDT": Decimal("1")})
    funding = {"AUSDT": pd.Series([0.001, 0.002], index=pd.DatetimeIndex([e1, e2]))}
    closes = {"AUSDT": pd.Series([200.0], index=pd.DatetimeIndex([e1]))}

    result = accrue_funding_by_watermark(history, {}, funding, closes, now)

    assert [(e.symbol, e.epoch) for e in result.events] == [("AUSDT", e1)]
    assert result.cash_delta == Decimal("-0.2")
    assert result.watermarks == {"AUSDT": e1}


def test_accrue_funding_zero_quantity_epoch_yields_no_event_but_advances() -> None:
    """Zero position at an epoch adds no event, yet the watermark advances past it."""
    from decimal import Decimal

    import pandas as pd

    from src.live.ledger import accrue_funding_by_watermark, append_position_snapshot

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    t1 = pd.Timestamp("2026-09-02 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-03 01:30", tz="UTC")
    e1 = pd.Timestamp("2026-09-01 08:00", tz="UTC")
    e2 = pd.Timestamp("2026-09-02 08:00", tz="UTC")
    history = append_position_snapshot(
        append_position_snapshot((), t0, {}), t1, {"AUSDT": Decimal("1")}
    )
    funding = {"AUSDT": pd.Series([0.001, 0.002], index=pd.DatetimeIndex([e1, e2]))}
    closes = {"AUSDT": pd.Series([100.0, 100.0], index=pd.DatetimeIndex([e1, e2]))}

    result = accrue_funding_by_watermark(history, {"AUSDT": t0}, funding, closes, now)

    assert [(e.symbol, e.epoch) for e in result.events] == [("AUSDT", e2)]
    assert result.cash_delta == Decimal("-0.2")
    assert result.watermarks == {"AUSDT": e2}


def test_accrue_funding_legacy_outputs_unchanged_for_fixed_inputs() -> None:
    """Pre-change outputs (cash, watermarks, lag, interval) are pinned as literals."""
    from decimal import Decimal

    import pandas as pd

    from src.live.ledger import accrue_funding_by_watermark, append_position_snapshot

    t0 = pd.Timestamp("2026-09-01 01:30", tz="UTC")
    now = pd.Timestamp("2026-09-01 17:00", tz="UTC")
    e1 = pd.Timestamp("2026-09-01 08:00:00.004", tz="UTC")
    e2 = pd.Timestamp("2026-09-01 16:00:00.002", tz="UTC")
    history = append_position_snapshot((), t0, {"AUSDT": Decimal("1")})
    funding = {"AUSDT": pd.Series([0.001, 0.002], index=pd.DatetimeIndex([e1, e2]))}
    marks = {"AUSDT": pd.Series([200.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-01 08:00", tz="UTC")]))}

    result = accrue_funding_by_watermark(history, {}, funding, marks, now)

    assert result.cash_delta == Decimal("-0.2")
    assert result.watermarks == {"AUSDT": e1}
    assert result.lag_by_symbol == {"AUSDT": now - e1}
    assert result.interval_by_symbol == {"AUSDT": pd.Timedelta(hours=8)}


def test_ledger_roundtrips_funding_backfill_markers(tmp_path) -> None:
    import json
    from decimal import Decimal

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.ledger import LedgerState, load_ledger, save_ledger

    path = tmp_path / "ledger.json"
    started = pd.Timestamp("2026-09-15 01:05Z")
    backfilled = pd.Timestamp("2026-09-15 01:05Z")

    save_ledger(path, LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1"), funding_accrual_started_at=started, funding_backfilled_through=backfilled))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["funding_accrual_started_at"] == started.isoformat()
    assert raw["funding_backfilled_through"] == backfilled.isoformat()
    reloaded = load_ledger(path)
    assert reloaded.funding_accrual_started_at == started
    assert reloaded.funding_backfilled_through == backfilled

    save_ledger(path, LedgerState(positions={"AAAUSDT": Decimal("1")}))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "funding_accrual_started_at" not in raw
    assert "funding_backfilled_through" not in raw
    assert load_ledger(path).funding_backfilled_through is None

    path.write_text(json.dumps({"positions": {}, "funding_backfilled_through": "2026-09-15 01:05"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_ledger(path)


def _journal_fill(fill_seq, side, qty, price, *, fee_bps=0.0, symbol="AAAUSDT", kind="execution", filled_at=None):
    """Minimal journal-fill stand-in carrying the fields commit_journal_fills folds."""
    from types import SimpleNamespace

    import pandas as pd

    return SimpleNamespace(
        fill_seq=fill_seq,
        kind=kind,
        attempt_seq=0,
        symbol=symbol,
        side=side,
        quantity=Decimal(str(qty)),
        price=Decimal(str(price)),
        fee_bps=float(fee_bps),
        liquidity="taker",
        reason="timeout_taker",
        filled_at=filled_at or pd.Timestamp("2026-09-20 00:00", tz="UTC"),
        client_order_id=f"cid-{fill_seq}",
        leg_index=0,
        cumulative_executed_qty=None,
        simulated=True,
    )


def test_commit_journal_fills_replay_is_idempotent(tmp_path: Path) -> None:
    """Same journal range applied twice yields the same positions and watermark."""
    from src.live.ledger import commit_journal_fills

    path = tmp_path / "ledger.json"
    fills = [
        _journal_fill(0, "BUY", "1", "100"),
        _journal_fill(1, "BUY", "1", "100"),
        _journal_fill(2, "SELL", "0.5", "100"),
    ]
    once = commit_journal_fills(
        path, LedgerState(), fills, equity=None, track_cash=False,
        starting_capital=Decimal("0"),
    )
    assert once.positions == {"AAAUSDT": Decimal("1.5")}
    assert once.journal_applied_fill_seq == 2
    twice = commit_journal_fills(
        path, once, fills, equity=None, track_cash=False,
        starting_capital=Decimal("0"),
    )
    assert twice.positions == {"AAAUSDT": Decimal("1.5")}
    assert twice.journal_applied_fill_seq == 2
    assert load_ledger(path).positions == {"AAAUSDT": Decimal("1.5")}


def test_commit_journal_fills_gap_fails_closed(tmp_path: Path) -> None:
    """Missing fill_seq relative to the watermark raises and leaves the file unchanged."""
    from src.live.ledger import commit_journal_fills

    path = tmp_path / "ledger.json"
    base = LedgerState(
        positions={"AAAUSDT": Decimal("1")}, journal_applied_fill_seq=0,
    )
    save_ledger(path, base)
    with pytest.raises(DataIntegrityError):
        commit_journal_fills(
            path, base, [_journal_fill(2, "BUY", "1", "100"), _journal_fill(3, "BUY", "1", "100")],
            equity=None, track_cash=False, starting_capital=Decimal("0"),
        )
    assert load_ledger(path) == base


def test_commit_journal_fills_conserves_paper_cash(tmp_path: Path) -> None:
    """BUY 2 @100 (5bps) + SELL 1 @110 (2bps) moves cash by the exact signed notional minus fees."""
    from src.live.ledger import commit_journal_fills

    path = tmp_path / "ledger.json"
    base = LedgerState(cash_usdt=Decimal("1000"))
    fills = [
        _journal_fill(0, "BUY", "2", "100", fee_bps=5.0),
        _journal_fill(1, "SELL", "1", "110", fee_bps=2.0),
    ]
    result = commit_journal_fills(
        path, base, fills, equity=None, track_cash=True,
        starting_capital=Decimal("0"),
    )
    assert result.cash_usdt == Decimal("909.878")
    assert result.positions == {"AAAUSDT": Decimal("1")}


def test_commit_journal_fills_operator_resync_skips_cash(tmp_path: Path) -> None:
    """operator_resync fills move positions only, never cash."""
    from src.live.ledger import commit_journal_fills

    path = tmp_path / "ledger.json"
    base = LedgerState(cash_usdt=Decimal("1000"))
    result = commit_journal_fills(
        path, base, [_journal_fill(0, "BUY", "3", "100", kind="operator_resync")],
        equity=None, track_cash=True, starting_capital=Decimal("0"),
    )
    assert result.positions == {"AAAUSDT": Decimal("3")}
    assert result.cash_usdt == Decimal("1000")


def test_commit_journal_fills_retains_history_needed_by_funding(tmp_path: Path) -> None:
    """Snapshots older than POSITION_HISTORY_MAX survive when funding still needs them."""
    import pandas as pd

    from src.live.ledger import PositionSnapshot, commit_journal_fills

    path = tmp_path / "ledger.json"
    t0 = pd.Timestamp("2026-09-01 01:00", tz="UTC")
    history = tuple(
        PositionSnapshot(
            effective_from=t0 + pd.Timedelta(days=day),
            positions={"AAAUSDT": Decimal(day + 1)},
        )
        for day in range(6)
    )
    base = LedgerState(
        positions={"AAAUSDT": Decimal("6")},
        funding_watermarks={"AAAUSDT": t0},
        position_history=history,
        journal_applied_fill_seq=5,
    )
    result = commit_journal_fills(
        path, base,
        [_journal_fill(6, "BUY", "1", "100", filled_at=t0 + pd.Timedelta(days=6))],
        equity=None, track_cash=False, starting_capital=Decimal("0"),
    )
    assert result.journal_applied_fill_seq == 6
    assert any(snap.effective_from == t0 for snap in result.position_history)


def test_mark_fills_recorded_never_exceeds_applied(tmp_path: Path) -> None:
    """Advancing the recorded watermark above applied fails without touching state."""
    from src.live.ledger import mark_fills_recorded

    path = tmp_path / "ledger.json"
    base = LedgerState(
        positions={"AAAUSDT": Decimal("1")}, journal_applied_fill_seq=3,
    )
    with pytest.raises(ValueError, match="above"):
        mark_fills_recorded(path, base, 5)
    assert not path.exists()
    advanced = mark_fills_recorded(path, base, 2)
    assert advanced.journal_recorded_fill_seq == 2
    assert advanced.journal_applied_fill_seq == 3
    assert load_ledger(path).journal_recorded_fill_seq == 2
    with pytest.raises(ValueError, match="backwards"):
        mark_fills_recorded(path, advanced, 1)
    path.unlink()
    assert mark_fills_recorded(path, advanced, 2) is advanced
    assert not path.exists()  # no-op advance performs no write




def test_ledger_derisk_flag_round_trip_and_guards(tmp_path) -> None:
    """De-risk flag persists, unions reasons, and validates on load."""
    import pytest

    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.live.ledger import clear_derisk, enter_derisk, load_ledger, save_ledger
    from src.live.ledger import LedgerState

    path = tmp_path / "ledger_derisk.json"
    save_ledger(path, LedgerState(positions={"AAAUSDT": Decimal("1")}))
    state = load_ledger(path)
    assert state.derisk_since is None
    assert state.derisk_reasons == ()

    entered = enter_derisk(path, state, reasons=("foreign_open_orders",), now=pd.Timestamp("2026-09-14T00:00:00Z"))
    assert entered.derisk_since == pd.Timestamp("2026-09-14T00:00:00Z")
    reloaded = load_ledger(path)
    assert reloaded.derisk_since == entered.derisk_since
    assert reloaded.derisk_reasons == ("foreign_open_orders",)

    again = enter_derisk(path, reloaded, reasons=("reconciliation_breach", "foreign_open_orders"), now=pd.Timestamp("2026-09-15T00:00:00Z"))
    assert again.derisk_since == entered.derisk_since
    assert again.derisk_reasons == ("foreign_open_orders", "reconciliation_breach")

    cleared = clear_derisk(path, again)
    assert cleared.derisk_since is None
    assert load_ledger(path).derisk_since is None

    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["derisk_reasons"] = ["reconciliation_breach"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="without derisk_since"):
        load_ledger(path)
    raw["derisk_reasons"] = "nope"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="must be a list"):
        load_ledger(path)


def test_ledger_watermark_guards_fail_closed(tmp_path) -> None:
    """Non-integer watermarks and inverted journals fail closed."""
    import json

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.ledger import LedgerState, load_ledger, save_ledger

    path = tmp_path / "ledger_wm.json"
    save_ledger(path, LedgerState(positions={}))
    raw = json.loads(path.read_text(encoding="utf-8"))
    for bad in (True, "x", -2):
        raw["journal_applied_fill_seq"] = bad
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(DataIntegrityError, match="journal_applied_fill_seq"):
            load_ledger(path)
    raw["journal_applied_fill_seq"] = 1
    raw["journal_recorded_fill_seq"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="exceeds"):
        load_ledger(path)


def test_commit_journal_fills_rejects_unknown_side(tmp_path) -> None:
    """A fill with an unknown side fails closed."""
    from types import SimpleNamespace

    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.ledger import LedgerState, commit_journal_fills

    path = tmp_path / "ledger_side.json"
    fill = SimpleNamespace(fill_seq=0, side="HOLD", symbol="AAAUSDT", quantity=Decimal("1"), price=Decimal("1"), fee_bps=0.0)
    with pytest.raises(DataIntegrityError, match="unknown side"):
        commit_journal_fills(path, LedgerState(), [fill], equity=None, track_cash=False, starting_capital=Decimal("0"))


def test_save_ledger_tolerates_unfsyncable_directory(tmp_path, monkeypatch) -> None:
    """An OSError on directory fsync still leaves a valid ledger."""
    from src.live.ledger import LedgerState, load_ledger, save_ledger

    path = tmp_path / "ledger_dir.json"
    monkeypatch.setattr("os.open", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    save_ledger(path, LedgerState(positions={"AAAUSDT": Decimal("2")}))
    assert load_ledger(path).positions == {"AAAUSDT": Decimal("2")}


def test_enter_derisk_rejects_naive_now(tmp_path) -> None:
    """De-risk entry requires a tz-aware timestamp."""
    import pandas as pd
    import pytest

    from src.live.ledger import LedgerState, enter_derisk

    with pytest.raises(ValueError, match="tz-aware"):
        enter_derisk(tmp_path / "x.json", LedgerState(), reasons=("a",), now=pd.Timestamp("2026-09-14 00:00"))
