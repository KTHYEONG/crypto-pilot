"""내부 포지션 원장: LedgerState 로드/저장(원자적)/체결 반영 및 reconcile_or_halt 연동."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.live.account import AccountSnapshot, reconcile_or_halt
from src.live.executor import ExecutionOutcome
from src.live.ledger import (
    LedgerState,
    apply_outcomes,
    compute_fill_cash_flow,
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


def test_apply_outcomes_buy_and_sell_signed_correctly() -> None:
    intents = [_intent("BTCUSDT", "BUY", "1"), _intent("ETHUSDT", "SELL", "2")]
    outcomes = [
        ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED"),
        ExecutionOutcome(symbol="ETHUSDT", filled_qty=Decimal("2"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("50"), chases=0, status="FILLED"),
    ]
    updated = apply_outcomes({}, intents, outcomes)
    assert updated == {"BTCUSDT": Decimal("1"), "ETHUSDT": Decimal("-2")}


def test_apply_outcomes_accumulates_onto_existing_position() -> None:
    intents = [_intent("BTCUSDT", "BUY", "0.5")]
    outcomes = [
        ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("0.5"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED"),
    ]
    updated = apply_outcomes({"BTCUSDT": Decimal("1")}, intents, outcomes)
    assert updated == {"BTCUSDT": Decimal("1.5")}


def test_apply_outcomes_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        apply_outcomes({}, [_intent("BTCUSDT", "BUY", "1")], [])


def test_apply_outcomes_symbol_mismatch_raises() -> None:
    intents = [_intent("BTCUSDT", "BUY", "1")]
    outcomes = [
        ExecutionOutcome(symbol="ETHUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=None, chases=0, status="FILLED"),
    ]
    with pytest.raises(ValueError, match="mismatch"):
        apply_outcomes({}, intents, outcomes)


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


def test_SCENARIO_LIVE_44_FILL_CASH_FLOW_SIGN_CONVENTION() -> None:
    """SCENARIO_LIVE_44: BUY decreases cash, SELL increases it; unfilled/None
    price contributes exactly 0; two intents sum algebraically."""
    buy = _intent("AAAUSDT", "BUY", "2")
    buy_outcome = ExecutionOutcome(
        symbol="AAAUSDT", filled_qty=Decimal("2"), unfilled_qty=Decimal("0"),
        avg_fill_price=Decimal("100"), chases=0, status="FILLED",
    )
    assert compute_fill_cash_flow([buy], [buy_outcome]) == Decimal("-200.1")  # fee 5bps

    sell = _intent("AAAUSDT", "SELL", "2")
    sell_outcome = ExecutionOutcome(
        symbol="AAAUSDT", filled_qty=Decimal("2"), unfilled_qty=Decimal("0"),
        avg_fill_price=Decimal("100"), chases=0, status="FILLED",
    )
    assert compute_fill_cash_flow([sell], [sell_outcome]) == Decimal("199.9")  # fee 5bps

    zero_fill = ExecutionOutcome(
        symbol="AAAUSDT", filled_qty=Decimal("0"), unfilled_qty=Decimal("2"),
        avg_fill_price=Decimal("100"), chases=0, status="RESIDUAL",
    )
    assert compute_fill_cash_flow([buy], [zero_fill]) == Decimal("0")
    no_price = ExecutionOutcome(
        symbol="AAAUSDT", filled_qty=Decimal("2"), unfilled_qty=Decimal("0"),
        avg_fill_price=None, chases=0, status="SHADOW",
    )
    assert compute_fill_cash_flow([buy], [no_price]) == Decimal("0")

    combined = compute_fill_cash_flow(
        [buy, _intent("BBBUSDT", "SELL", "2")],
        [
            buy_outcome,
            ExecutionOutcome(
                symbol="BBBUSDT", filled_qty=Decimal("2"), unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            ),
        ],
    )
    assert combined == Decimal("-200.1") + Decimal("199.9") == Decimal("-0.2")


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_43_LEDGER_CASH_ROUND_TRIPS_AND_STAYS_BACKWARD_COMPATIBLE",
    "SCENARIO_LIVE_44_FILL_CASH_FLOW_SIGN_CONVENTION",
)

def test_SCENARIO_PARITY_05_fee_accounted_cashflow():
    """SCENARIO_PARITY_05-fee-accounted-cashflow"""
    from decimal import Decimal
    from src.live.planner import OrderIntent
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import compute_fill_cash_flow
    intent_buy = OrderIntent(symbol="AAAUSDT", side="BUY", quantity=Decimal("1.0"), reduce_only=False, target_qty=Decimal("1.0"), current_qty=Decimal("0"), client_order_prefix="run1", leg_index=0, decision_price=Decimal("100"))
    # maker 2bps
    outcome_maker = ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("1.0"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.0"), chases=0, status="FILLED", fills=((Decimal("1.0"), Decimal("100.0"), 2.0, "maker_fill", "maker"),), maker_qty=Decimal("1.0"), taker_qty=Decimal("0"))
    assert compute_fill_cash_flow([intent_buy], [outcome_maker]) == Decimal("-100.02")
    # taker 5bps
    outcome_taker = ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("1.0"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.0"), chases=0, status="FILLED", fills=((Decimal("1.0"), Decimal("100.0"), 5.0, "timeout_taker", "taker"),), maker_qty=Decimal("0"), taker_qty=Decimal("1.0"))
    assert compute_fill_cash_flow([intent_buy], [outcome_taker]) == Decimal("-100.05")
    # SELL maker
    intent_sell = OrderIntent(symbol="AAAUSDT", side="SELL", quantity=Decimal("1.0"), reduce_only=False, target_qty=Decimal("0"), current_qty=Decimal("1.0"), client_order_prefix="run1", leg_index=0, decision_price=Decimal("100"))
    outcome_sell_maker = ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("1.0"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.0"), chases=0, status="FILLED", fills=((Decimal("1.0"), Decimal("100.0"), 2.0, "maker_fill", "maker"),), maker_qty=Decimal("1.0"), taker_qty=Decimal("0"))
    assert compute_fill_cash_flow([intent_sell], [outcome_sell_maker]) == Decimal("99.98")


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


