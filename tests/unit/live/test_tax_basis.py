"""Moving-average cost-basis fold invariants (probe a / a'' / f regressions)."""

from __future__ import annotations

from dataclasses import replace
from decimal import ROUND_HALF_EVEN, Decimal, Inexact, Rounded, localcontext

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.tax_basis import (
    AVERAGE_ENTRY_QUANTUM,
    FLAT_POSITION,
    AverageCostPosition,
    apply_fill,
    apply_settlement_close,
    fold_symbol,
    require_moving_average,
)
from src.live.tax_ledger import TaxRecord

_T0 = pd.Timestamp("2027-01-01 00:00:00", tz="UTC")


def _trade(rid: str, side: str, qty: str, price: str, seq: int, symbol: str = "BTCUSDT") -> TaxRecord:
    return TaxRecord(
        record_id=rid,
        kind="TRADE",
        event_time=_T0 + pd.Timedelta(seconds=seq),
        symbol=symbol,
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        quote_qty=Decimal(qty) * Decimal(price),
        fee=Decimal("0"),
        fee_asset="USDT",
        realized_pnl=Decimal("0"),
        income_asset="USDT",
        is_maker=False,
        venue_id=seq,
        source="simulated",
        mode="paper",
    )


def _settlement(rid: str, amount: str, seq: int, symbol: str = "BTCUSDT") -> TaxRecord:
    return TaxRecord(
        record_id=rid,
        kind="REALIZED_PNL",
        event_time=_T0 + pd.Timedelta(seconds=seq),
        symbol=symbol,
        side="",
        quantity=Decimal("0"),
        price=Decimal("0"),
        quote_qty=Decimal("0"),
        fee=Decimal("0"),
        fee_asset="",
        realized_pnl=Decimal(amount),
        income_asset="USDT",
        is_maker=False,
        venue_id=seq,
        source="venue",
        mode="live_testnet",
        income_type="DELIVERED_SETTELMENT",
    )


def test_same_direction_fills_re_average() -> None:
    steps = fold_symbol([_trade("S1", "BUY", "1", "100", 1), _trade("S2", "BUY", "1", "200", 2)])
    assert [s.realized_pnl for s in steps] == [Decimal("0"), Decimal("0")]
    assert steps[-1].position_after.quantity == Decimal("2")
    assert steps[-1].position_after.avg_entry == Decimal("150")


def test_partial_close_keeps_basis() -> None:
    steps = fold_symbol([
        _trade("S1", "BUY", "1", "100", 1),
        _trade("S2", "BUY", "1", "200", 2),
        _trade("S3", "SELL", "1", "300", 3),
    ])
    last = steps[-1]
    assert last.realized_pnl == Decimal("150")
    assert last.closed_direction == "long"
    assert last.position_after.quantity == Decimal("1")
    assert last.position_after.avg_entry == Decimal("150")
    assert last.position_after.opened_at == _T0 + pd.Timedelta(seconds=1)


def test_short_round_trip_realizes_on_close() -> None:
    steps = fold_symbol([_trade("S1", "SELL", "1", "100", 1), _trade("S2", "BUY", "1", "90", 2)])
    close = steps[-1]
    assert close.realized_pnl == Decimal("10")
    assert close.closed_direction == "short"
    assert close.entry_notional_closed == Decimal("100")
    assert close.exit_notional_closed == Decimal("90")
    assert close.position_after == FLAT_POSITION


def test_flip_closes_then_opens_remainder() -> None:
    steps = fold_symbol([
        _trade("S1", "BUY", "1", "100", 1),
        _trade("S2", "SELL", "2", "110", 2),
        _trade("S3", "BUY", "1", "105", 3),
    ])
    assert steps[1].realized_pnl == Decimal("10")
    assert steps[1].closed_direction == "long"
    assert steps[1].position_after.quantity == Decimal("-1")
    assert steps[1].position_after.avg_entry == Decimal("110")
    assert steps[1].position_after.opened_at == _T0 + pd.Timedelta(seconds=2)
    assert steps[2].realized_pnl == Decimal("5")
    assert steps[2].closed_direction == "short"
    assert steps[-1].position_after == FLAT_POSITION
    assert steps[-1].position_after.avg_entry == Decimal("0")


def test_flat_resets_basis() -> None:
    steps = fold_symbol([
        _trade("S1", "BUY", "1", "90", 1),
        _trade("S2", "SELL", "1", "100", 2),
        _trade("S3", "BUY", "1", "50", 3),
    ])
    assert steps[1].position_after == FLAT_POSITION
    assert steps[-1].position_after.avg_entry == Decimal("50")
    assert steps[-1].position_after.quantity == Decimal("1")


def test_per_fill_conservation_over_random_sequence() -> None:
    records = [
        _trade(
            f"S{i}", "BUY" if (i * 37 + 11) % 3 else "SELL",
            ["1", "2", "0.5", "3"][i % 4],
            str(100 + (i * 37) % 400) + ".25",
            i,
        )
        for i in range(1, 201)
    ]
    steps = fold_symbol(records)
    total_realized = Decimal("0")
    total_signed = Decimal("0")
    for step in steps:
        rec = next(r for r in records if r.record_id == step.record_id)
        signed = rec.quantity if rec.side == "BUY" else -rec.quantity
        assert step.closed_quantity + step.opened_quantity == rec.quantity
        assert step.position_after.quantity == step.position_before.quantity + signed
        flat = step.position_after.quantity == 0
        assert flat == (step.position_after.avg_entry == 0)
        assert flat == (step.position_after.opened_at is None)
        if step.closed_direction == "long":
            assert step.realized_pnl == step.exit_notional_closed - step.entry_notional_closed
            total_signed += step.exit_notional_closed - step.entry_notional_closed
        elif step.closed_direction == "short":
            assert step.realized_pnl == step.entry_notional_closed - step.exit_notional_closed
            total_signed += step.entry_notional_closed - step.exit_notional_closed
        else:
            assert step.closed_quantity == 0
            assert step.realized_pnl == 0
        total_realized += step.realized_pnl
    assert total_realized == total_signed


def test_single_rounding_point_at_re_average() -> None:
    steps = fold_symbol([_trade("S1", "BUY", "1", "1", 1), _trade("S2", "BUY", "2", "2", 2)])
    expected = (Decimal(5) / Decimal(3)).quantize(AVERAGE_ENTRY_QUANTUM, rounding=ROUND_HALF_EVEN)
    assert steps[-1].position_after.avg_entry == expected
    (close,) = fold_symbol([
        _trade("S1", "BUY", "1", "1", 1),
        _trade("S2", "BUY", "2", "2", 2),
        _trade("S3", "SELL", "3", "2", 3),
    ])[-1:]
    assert close.realized_pnl == (Decimal("2") - expected) * 3


def test_exact_decimal_accumulation() -> None:
    steps = fold_symbol([_trade(f"S{i}", "BUY", "0.1", "0.1", i) for i in range(1, 11)])
    assert steps[-1].position_after.quantity == Decimal("1.0")
    assert steps[-1].position_after.avg_entry == Decimal("0.1")


def test_causal_prefix_invariance() -> None:
    base = [
        _trade(f"S{i}", "BUY" if i % 3 else "SELL", "1", str(100 + i), i)
        for i in range(1, 11)
    ]
    first = fold_symbol(base)
    perturbed = base + [
        _trade(f"T{i}", "BUY", "1", str(1000 + i * 13), 100 + i) for i in range(5)
    ]
    second = fold_symbol(perturbed)
    assert [s.realized_pnl for s in second[:10]] == [s.realized_pnl for s in first]
    assert [s.position_after for s in second[:10]] == [s.position_after for s in first]


def test_input_order_independence() -> None:
    records = [
        _trade(f"S{i}", "BUY" if (i * 29 + 7) % 3 else "SELL", ["1", "2"][(i * 17) % 2], str(90 + i), i)
        for i in range(1, 21)
    ]
    shuffled = list(reversed(records))
    assert fold_symbol(shuffled) == fold_symbol(records)


def test_delivery_settlement_closes_position() -> None:
    steps = fold_symbol([
        _trade("S1", "BUY", "2", "100", 1),
        _settlement("V1", "12.5", 2),
    ])
    close = steps[-1]
    assert close.realized_pnl == Decimal("12.5")
    assert close.entry_notional_closed == Decimal("200")
    assert close.exit_notional_closed == Decimal("212.5")
    assert close.position_after == FLAT_POSITION


def test_settlement_on_flat_position_fails_closed() -> None:
    with pytest.raises(DataIntegrityError):
        fold_symbol([_settlement("V1", "12.5", 1)])


def test_only_moving_average() -> None:
    require_moving_average("moving_average")
    with pytest.raises(ValueError, match="unsupported cost basis"):
        require_moving_average("fifo")


@pytest.mark.parametrize("kind", ["FUNDING_FEE", "COMMISSION", "TRANSFER", "UNCLASSIFIED", "REALIZED_PNL"])
def test_mixed_symbols_or_foreign_kinds_rejected(kind: str) -> None:
    foreign = TaxRecord(
        record_id="F1",
        kind=kind,
        event_time=_T0 + pd.Timedelta(seconds=9),
        symbol="BTCUSDT",
        side="",
        quantity=Decimal("0"),
        price=Decimal("0"),
        quote_qty=Decimal("0"),
        fee=Decimal("0"),
        fee_asset="",
        realized_pnl=Decimal("1"),
        income_asset="USDT",
        is_maker=False,
        venue_id=9,
        source="venue" if kind != "FUNDING_FEE" else "simulated",
        mode="live_testnet" if kind != "FUNDING_FEE" else "paper",
        income_type="FUNDING_FEE" if kind != "FUNDING_FEE" else "",
    )
    with pytest.raises(ValueError, match="fold_symbol"):
        fold_symbol([_trade("S1", "BUY", "1", "100", 1), foreign])
    with pytest.raises(ValueError, match="fold_symbol"):
        fold_symbol([
            _trade("S1", "BUY", "1", "100", 1, symbol="BTCUSDT"),
            _trade("S2", "BUY", "1", "100", 2, symbol="ETHUSDT"),
        ])


def test_apply_fill_rejects_non_trade() -> None:
    with pytest.raises(ValueError, match="TRADE"):
        apply_fill(FLAT_POSITION, _settlement("V1", "1", 1))


def test_apply_settlement_close_rejects_non_settlement() -> None:
    with pytest.raises(ValueError, match="REALIZED_PNL"):
        apply_settlement_close(
            AverageCostPosition(Decimal("1"), Decimal("100"), _T0),
            _trade("S1", "BUY", "1", "100", 1),
        )


def test_flat_position_constant() -> None:
    assert AverageCostPosition(Decimal(0), Decimal(0), None) == FLAT_POSITION


def test_delivery_settlement_closes_short_position() -> None:
    steps = fold_symbol([
        _trade("S1", "SELL", "2", "100", 1),
        _settlement("V1", "12.5", 2),
    ])
    close = steps[-1]
    assert close.closed_direction == "short"
    assert close.realized_pnl == Decimal("12.5")
    assert close.entry_notional_closed == Decimal("200")
    assert close.exit_notional_closed == Decimal("187.5")
    assert close.position_after == FLAT_POSITION


def test_duplicate_record_id_in_fold_fails_closed() -> None:
    with pytest.raises(DataIntegrityError, match="duplicate record_id"):
        fold_symbol([_trade("S1", "BUY", "1", "100", 1), _trade("S1", "BUY", "1", "100", 2)])


@pytest.mark.parametrize("precision", [6, 28, 60])
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_same_price_round_trip_is_exact_under_any_caller_context(precision: int, side: str) -> None:
    opposite = "SELL" if side == "BUY" else "BUY"
    price = "12345678901234567890.123456789"
    records = [_trade("open", side, "1", price, 1), _trade("close", opposite, "1", price, 2)]

    with localcontext() as caller:
        caller.prec = precision
        caller.traps[Inexact] = True
        caller.traps[Rounded] = True
        caller.clear_flags()
        steps = fold_symbol(records)
        assert caller.prec == precision
        assert caller.traps[Inexact]
        assert caller.traps[Rounded]
        assert not any(caller.flags.values())

    assert steps[-1].realized_pnl == Decimal(0)
    assert steps[-1].entry_notional_closed == Decimal(price)
    assert steps[-1].exit_notional_closed == Decimal(price)
    assert steps[-1].position_after == FLAT_POSITION


@pytest.mark.parametrize("precision", [6, 28, 60])
def test_reaveraging_and_partial_close_are_caller_context_independent(precision: int) -> None:
    records = [
        _trade("open", "SELL", "1", "1", 1),
        _trade("add", "SELL", "2", "2", 2),
        _trade("partial", "BUY", "1", "2", 3),
        _trade("close", "BUY", "2", "2", 4),
    ]
    with localcontext() as caller:
        caller.prec = precision
        caller.traps[Inexact] = True
        caller.traps[Rounded] = True
        steps = fold_symbol(records)

    assert steps[1].position_after.avg_entry == Decimal("1.666666666666666667")
    assert steps[2].realized_pnl == Decimal("-0.333333333333333333")
    assert steps[3].realized_pnl == Decimal("-0.666666666666666666")
    assert steps[2].position_after.quantity == Decimal("-2")
    assert steps[3].position_after == FLAT_POSITION


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_precision_exhaustion_raises_instead_of_rounding_quantity(side: str) -> None:
    record = replace(_trade("fill", side, "1", "1", 1), quantity=Decimal("1." + "1234567890" * 7))
    with localcontext() as caller:
        caller.prec = 6
        caller.traps[Inexact] = False
        with pytest.raises(Inexact):
            apply_fill(FLAT_POSITION, record)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("price", ["1E-20", "5E-19"])
def test_reaveraging_that_erases_positive_basis_fails_closed(side: str, price: str) -> None:
    opened = apply_fill(FLAT_POSITION, _trade("open", side, "1", price, 1)).position_after
    with pytest.raises(DataIntegrityError, match="avg_entry rounds to zero"):
        apply_fill(opened, _trade("add", side, "1", price, 2))
    assert opened.avg_entry == Decimal(price)
    assert opened.quantity == (Decimal(1) if side == "BUY" else Decimal(-1))


@pytest.mark.parametrize(
    ("first_price", "second_price", "expected"),
    [
        ("1", "1.000000000000000001", "1.000000000000000000"),
        ("1.000000000000000001", "1.000000000000000002", "1.000000000000000002"),
        ("1E-18", "1E-18", "1E-18"),
    ],
    ids=["half_even_down", "half_even_up", "minimum_nonzero_basis"],
)
def test_reaveraging_half_even_ties_and_quantum_boundary(
    first_price: str, second_price: str, expected: str,
) -> None:
    steps = fold_symbol([
        _trade("open", "BUY", "1", first_price, 1),
        _trade("add", "BUY", "1", second_price, 2),
    ])
    assert steps[-1].position_after.avg_entry == Decimal(expected)
    assert steps[-1].position_after.quantity == Decimal(2)


def test_short_settlement_negation_preserves_full_amount_under_low_precision() -> None:
    record = _settlement("settlement", "12345678901234567890.123456789", 2)
    initial = AverageCostPosition(Decimal(-1), Decimal("22345678901234567890.123456789"), _T0)
    with localcontext() as caller:
        caller.prec = 6
        caller.traps[Inexact] = True
        step = apply_settlement_close(initial, record)
    assert step.entry_notional_closed == initial.avg_entry
    assert step.exit_notional_closed == Decimal("10000000000000000000")
    assert step.realized_pnl == record.realized_pnl
    assert step.position_after == FLAT_POSITION
