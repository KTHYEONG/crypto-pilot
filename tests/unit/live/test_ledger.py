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
