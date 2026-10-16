"""Reconciler R1..R6 invariants."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.live.testnet_drill.reconcile import ReconciliationEvidence, evaluate_reconciliation
from tests.unit.live.testnet_drill._stubs import make_config, patch_drill_root


@pytest.fixture
def drill_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return patch_drill_root(monkeypatch, tmp_path)


def _base(**overrides) -> ReconciliationEvidence:
    base = {
        "symbols": ("BTCUSDT",),
        "journal_positions": (("BTCUSDT", Decimal("1.0")),),
        "venue_positions": (("BTCUSDT", Decimal("1.0")),),
        "journal_observed": (("oid-1", Decimal("0.4")),),
        "venue_executed": (("oid-1", Decimal("0.4")),),
        "journal_trade_qty": (("oid-1", Decimal("0.4")),),
        "journal_trade_vwap": (("oid-1", Decimal("100")),),
        "journal_trade_fee": (("oid-1", Decimal("0.01")),),
        "journal_maker_qty": (),
        "venue_trade_qty": (("oid-1", Decimal("0.4")),),
        "venue_trade_vwap": (("oid-1", Decimal("100")),),
        "venue_trade_commission": (("oid-1", Decimal("0.01")),),
        "venue_maker_count": (),
        "journal_maker_count": (),
        "wallet_start": Decimal("1000"),
        "wallet_end": Decimal("1000"),
        "income_sums": (),
        "open_orders": (),
        "fill_keys": (("oid-1", Decimal("0.4")),),
    }
    base.update(overrides)
    return ReconciliationEvidence(**base)


def _by_name(results) -> dict:
    return {item.name: item for item in results}


def test_position_parity_detects_drift() -> None:
    evidence = _base(
        journal_positions=(("BTCUSDT", Decimal("1.0")),),
        venue_positions=(("BTCUSDT", Decimal("0.9")),),
    )
    results = _by_name(evaluate_reconciliation(evidence, make_config()))
    assert not results["position_parity"].passed
    assert "BTCUSDT" in results["position_parity"].detail
    assert "1.0" in results["position_parity"].detail


def test_watermark_parity_names_order() -> None:
    evidence = _base(
        journal_observed=(("oid-9", Decimal("0.4")),),
        venue_executed=(("oid-9", Decimal("0.5")),),
    )
    results = _by_name(evaluate_reconciliation(evidence, make_config()))
    assert not results["order_watermark_parity"].passed
    assert "oid-9" in results["order_watermark_parity"].detail


def test_trade_parity_tolerance_and_fee() -> None:
    passing = _base()
    assert _by_name(evaluate_reconciliation(passing, make_config()))["trade_parity"].passed
    failing = _base(venue_trade_commission=(("oid-1", Decimal("5.0")),))
    assert not _by_name(evaluate_reconciliation(failing, make_config()))["trade_parity"].passed


def test_estimated_recovery_fee_may_exceed_but_not_undercut_venue() -> None:
    over = _base(journal_trade_fee=(("oid-1", Decimal("0.5")),), estimated_fee_orders=("oid-1",))
    assert _by_name(evaluate_reconciliation(over, make_config()))["trade_parity"].passed
    under = _base(
        journal_trade_fee=(("oid-1", Decimal("0.001")),),
        venue_trade_commission=(("oid-1", Decimal("0.5")),),
        estimated_fee_orders=("oid-1",),
    )
    result = _by_name(evaluate_reconciliation(under, make_config()))["trade_parity"]
    assert not result.passed
    assert "undercuts" in result.detail
    exact_required = _base(journal_trade_fee=(("oid-1", Decimal("0.5")),))
    assert not _by_name(evaluate_reconciliation(exact_required, make_config()))["trade_parity"].passed


def test_journal_fold_marks_recovery_fee_as_estimated() -> None:
    from types import SimpleNamespace

    from src.live.testnet_drill.reconcile import _journal_fold

    def fill(kind: str, order: str) -> SimpleNamespace:
        return SimpleNamespace(
            kind=kind, symbol="BTCUSDT", side="BUY", quantity=Decimal("1"), price=Decimal("100"),
            fee_bps=4.0, liquidity="taker", client_order_id=order,
        )

    positions, trades = _journal_fold([fill("execution", "a"), fill("recovered", "b"), fill("orphan_settlement", "c")])
    assert positions == {"BTCUSDT": Decimal("3")}
    assert trades.estimated == {"b", "c"}


def test_maker_mismatch_reported_not_failed() -> None:
    evidence = _base(
        journal_maker_count=(("oid-1", 1),),
        venue_maker_count=(("oid-1", 0),),
    )
    results = _by_name(evaluate_reconciliation(evidence, make_config()))
    assert results["trade_parity"].passed
    assert "maker" in results["trade_parity"].detail


def test_wallet_parity() -> None:
    passing = _base(
        wallet_start=Decimal("1000"),
        wallet_end=Decimal("1005"),
        income_sums=(("REALIZED_PNL", Decimal("6")), ("COMMISSION", Decimal("-2")), ("FUNDING_FEE", Decimal("1"))),
    )
    assert _by_name(evaluate_reconciliation(passing, make_config()))["wallet_parity"].passed
    failing = _base(
        wallet_start=Decimal("1000"),
        wallet_end=Decimal("1020"),
        income_sums=(("REALIZED_PNL", Decimal("6")), ("COMMISSION", Decimal("-2")), ("FUNDING_FEE", Decimal("1"))),
    )
    assert not _by_name(evaluate_reconciliation(failing, make_config()))["wallet_parity"].passed


def test_duplicate_fill_detection() -> None:
    evidence = _base(fill_keys=(("oid-1", Decimal("0.4")), ("oid-1", Decimal("0.4"))))
    assert not _by_name(evaluate_reconciliation(evidence, make_config()))["no_duplicate_fills"].passed


def test_decimal_exactness_at_tenth_place() -> None:
    evidence = _base(
        journal_positions=(("BTCUSDT", Decimal("1.0000000001")),),
        venue_positions=(("BTCUSDT", Decimal("1.0")),),
    )
    assert _by_name(evaluate_reconciliation(evidence, make_config()))["no_duplicate_fills"].passed is not False
    assert not _by_name(evaluate_reconciliation(evidence, make_config()))["position_parity"].passed


def test_reconciler_read_only(drill_root) -> None:
    from src.live.testnet_drill.reconcile import reconcile_session
    from src.live.testnet_drill.session import open_drill_session
    from tests.unit.live.testnet_drill._stubs import StatefulVenue, make_settings

    venue = StatefulVenue(["BTCUSDT"])
    session = open_drill_session(make_settings(), make_config(), run_id="20261016a", inner=venue)
    journal_path = session.journal.path
    before = journal_path.read_bytes() if journal_path.exists() else b""
    original = session.client.request

    def request(method, path, params=None, *, signed=False):
        if path == "/fapi/v1/userTrades":
            return []
        return original(method, path, params, signed=signed)

    session.client.request = request
    evidence = reconcile_session(session, since_ms=0, settle_wait_s=0.0)
    assert evidence.symbols == ("BTCUSDT",)
    after = journal_path.read_bytes() if journal_path.exists() else b""
    assert before == after
    for record in session.transport.records:
        assert record.method == "GET"
    session.transport.close()
    session.audit.close()


def test_trade_qty_mismatch_fails() -> None:
    evidence = _base(
        journal_trade_qty=(("oid-1", Decimal("0.4")),),
        venue_trade_qty=(("oid-1", Decimal("0.2")),),
    )
    results = _by_name(evaluate_reconciliation(evidence, make_config()))
    assert not results["trade_parity"].passed
    assert "qty" in results["trade_parity"].detail


def test_trade_vwap_mismatch_fails() -> None:
    evidence = _base(
        journal_trade_vwap=(("oid-1", Decimal("100")),),
        venue_trade_vwap=(("oid-1", Decimal("120")),),
    )
    results = _by_name(evaluate_reconciliation(evidence, make_config()))
    assert not results["trade_parity"].passed
    assert "vwap" in results["trade_parity"].detail


def test_missing_wallet_baseline_does_not_pass() -> None:
    result = _by_name(evaluate_reconciliation(_base(wallet_start=None), make_config()))
    assert not result["wallet_parity"].passed


def test_duplicate_watermarks_compare_decimal_values() -> None:
    result = _by_name(
        evaluate_reconciliation(
            _base(
                fill_keys=(
                    ("oid-1", Decimal("0.40")),
                    ("oid-1", Decimal("0.4")),
                )
            ),
            make_config(),
        )
    )
    assert not result["no_duplicate_fills"].passed


def test_read_failure_never_becomes_empty_success() -> None:
    evidence = _base(read_errors=("TransientReadError",))
    assert not any(item.passed for item in evaluate_reconciliation(evidence, make_config()))


def test_reconcile_numeric_order_id_and_wallet_baseline(drill_root) -> None:
    import dataclasses

    import pandas as pd

    from src.live.testnet_drill.reconcile import reconcile_session
    from src.live.testnet_drill.session import open_drill_session
    from tests.unit.live.testnet_drill._stubs import StatefulVenue, make_settings

    venue = StatefulVenue(["BTCUSDT"], positions={"BTCUSDT": "0.4"}, wallet="1001")
    session = open_drill_session(make_settings(), make_config(), run_id="populated", inner=venue)
    session = dataclasses.replace(session, starting_wallet_balance=Decimal("1000"))
    journal = session.journal
    journal.record_submit(
        "oid-1", "BTCUSDT", 0, attempt_seq=0, side="BUY", quantity=Decimal("0.4"), reduce_only=False, leg_index=0
    )
    journal.record_fill(
        kind="execution",
        attempt_seq=0,
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.4"),
        price=Decimal("100"),
        fee_bps=5.0,
        liquidity="maker",
        reason="maker_fill",
        filled_at=pd.Timestamp.now(tz="UTC"),
        client_order_id="oid-1",
        leg_index=0,
        cumulative_executed_qty=Decimal("0.4"),
        simulated=False,
    )
    original = session.client.request

    def request(method, path, params=None, *, signed=False):
        if path == "/fapi/v1/order":
            return {"orderId": 12345, "executedQty": "0.4"}
        if path == "/fapi/v1/userTrades":
            return [
                {
                    "id": 7,
                    "orderId": 12345,
                    "time": 1,
                    "qty": "0.4",
                    "price": "100",
                    "commission": "0.02",
                    "commissionAsset": "USDT",
                    "maker": True,
                }
            ]
        if path == "/fapi/v1/income":
            return [{"tranId": 8, "incomeType": "REALIZED_PNL", "asset": "USDT", "income": "1"}]
        return original(method, path, params, signed=signed)

    session.client.request = request
    before = journal.path.read_bytes()
    evidence = reconcile_session(session, since_ms=0, settle_wait_s=0)
    assert dict(evidence.venue_trade_qty) == {"oid-1": Decimal("0.4")}
    assert evidence.wallet_start == Decimal("1000")
    assert all(item.passed for item in evaluate_reconciliation(evidence, make_config()))
    assert journal.path.read_bytes() == before
    assert all(record.method == "GET" for record in session.transport.records)
    session.transport.close()
    session.audit.close()


@pytest.mark.parametrize("field", ["journal_positions", "venue_trade_qty", "journal_trade_fee"])
def test_non_finite_evidence_fails(field) -> None:
    assert not any(
        item.passed for item in evaluate_reconciliation(_base(**{field: (("BTCUSDT", Decimal("NaN")),)}), make_config())
    )


def test_qty_tolerance_does_not_relax_order_watermark() -> None:
    evidence = _base(venue_executed=(("oid-1", Decimal("0.4000000001")),))
    assert not _by_name(evaluate_reconciliation(evidence, make_config(qty_tolerance=Decimal("1"))))[
        "order_watermark_parity"
    ].passed


def test_qty_tolerance_does_not_relax_trade_parity() -> None:
    evidence = _base(venue_trade_qty=(("oid-1", Decimal("0.4000000001")),))
    assert not _by_name(evaluate_reconciliation(evidence, make_config(qty_tolerance=Decimal("1"))))[
        "trade_parity"
    ].passed


def test_missing_vwap_fails() -> None:
    assert not _by_name(evaluate_reconciliation(_base(venue_trade_vwap=()), make_config()))["trade_parity"].passed


def test_missing_order_read_fails_even_when_watermark_zero() -> None:
    evidence = _base(journal_observed=(("oid-1", Decimal(0)),), venue_executed=())
    assert not _by_name(evaluate_reconciliation(evidence, make_config()))["order_watermark_parity"].passed


@pytest.fixture
def venue_session(tmp_path):
    from types import SimpleNamespace

    from src.live.order_journal import OrderJournal

    class Client:
        def __init__(self):
            self.risk = [{"symbol": "BTCUSDT", "positionAmt": "0"}]
            self.orders = {"one": {"orderId": 100, "executedQty": "0"}}
            self.trades = [[]]
            self.incomes = [[]]
            self.wallet = {"totalWalletBalance": "1000"}
            self.calls = []

        def position_risk(self):
            return self.risk

        def query_order(self, symbol, client_id):
            return self.orders[client_id]

        def request(self, method, path, params, *, signed):
            self.calls.append((method, path, params))
            if path.endswith("userTrades"):
                return self.trades.pop(0) if len(self.trades) > 1 else self.trades[0]
            return self.incomes.pop(0) if len(self.incomes) > 1 else self.incomes[0]

        def account(self):
            return self.wallet

        def open_orders(self):
            return []

    journal = OrderJournal(tmp_path / "orders.jsonl")
    journal.record_submit(
        "one", "BTCUSDT", 0, attempt_seq=0, side="BUY", quantity=Decimal("1"), reduce_only=False, leg_index=0
    )
    return SimpleNamespace(
        config=make_config(), journal=journal, client=Client(), starting_wallet_balance=Decimal("1000")
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "object rows"),
        (["bad"], "object rows"),
        ([{"symbol": "BTCUSDT", "positionAmt": "NaN"}], "non-finite"),
        ([{"symbol": "BTCUSDT", "positionAmt": "0", "positionSide": "LONG"}], "hedge"),
        ([], "missing symbols"),
    ],
)
def test_invalid_position_truth_is_not_ignored(venue_session, payload, message):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.client.risk = payload
    with pytest.raises(DataIntegrityError, match=message):
        _snapshot(venue_session, 0, 100)


@pytest.mark.parametrize(
    ("payload", "message"), [(None, "invalid response"), ({"orderId": 100, "executedQty": "Infinity"}, "non-finite")]
)
def test_invalid_order_truth_fails(venue_session, payload, message):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.client.orders["one"] = payload
    with pytest.raises(DataIntegrityError, match=message):
        _snapshot(venue_session, 0, 100)


def test_order_ids_must_have_unique_identity(venue_session):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.journal.record_submit(
        "two", "BTCUSDT", 1, attempt_seq=0, side="BUY", quantity=Decimal("1"), reduce_only=False, leg_index=0
    )
    venue_session.client.orders["two"] = venue_session.client.orders["one"]
    with pytest.raises(DataIntegrityError, match="multiple client ids"):
        _snapshot(venue_session, 0, 100)


def _trade(**overrides):
    return {
        "id": 1,
        "orderId": 100,
        "time": 1,
        "qty": "1",
        "price": "100",
        "commissionAsset": "USDT",
        "commission": "0",
        "maker": False,
        **overrides,
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_trade(), _trade()], "duplicate"),
        ([_trade(commissionAsset="BNB")], "commission asset"),
        ([_trade(maker="false")], "maker flag"),
        ([_trade(qty="0")], "non-positive"),
    ],
)
def test_invalid_trade_truth_fails(venue_session, rows, message):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.client.trades = [rows]
    with pytest.raises(DataIntegrityError, match=message):
        _snapshot(venue_session, 0, 100)


def test_full_trade_and_income_pages_are_read(venue_session):
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.client.risk.append({"symbol": "ETHUSDT", "positionAmt": "1"})
    venue_session.client.trades = [[_trade(id=i, orderId=999) for i in range(1000)], [_trade(id=1000, time=101)]]
    venue_session.client.incomes = [
        [{"tranId": i, "incomeType": "COMMISSION", "asset": "USDT", "income": "-0.001"} for i in range(1000)],
        [],
    ]
    evidence = _snapshot(venue_session, 0, 100)
    assert dict(evidence.income_sums) == {"COMMISSION": Decimal("-1.000")}
    assert evidence.venue_trade_qty == ()
    assert venue_session.client.calls[1][2]["fromId"] == 1000
    assert venue_session.client.calls[-1][2]["page"] == 2


def test_non_advancing_trade_cursor_fails(venue_session):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.client.trades = [
        [_trade(id=2000 + i, orderId=999) for i in range(1000)],
        [_trade(id=i, orderId=999) for i in range(1000)],
    ]
    with pytest.raises(DataIntegrityError, match="cursor"):
        _snapshot(venue_session, 0, 100)


def test_duplicate_income_is_not_double_counted(venue_session):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    row = {"tranId": 1, "incomeType": "COMMISSION", "asset": "USDT", "income": "-1"}
    venue_session.client.incomes = [[row, row]]
    with pytest.raises(DataIntegrityError, match="duplicate"):
        _snapshot(venue_session, 0, 100)


def test_invalid_account_truth_fails(venue_session):
    from src.common.errors import DataIntegrityError
    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.client.wallet = []
    with pytest.raises(DataIntegrityError, match="account"):
        _snapshot(venue_session, 0, 100)


def test_journaled_adjustments_move_position_fold_but_not_trade_parity(venue_session):
    import pandas as pd

    from src.live.testnet_drill.reconcile import _snapshot

    venue_session.journal.record_fill(
        kind="operator_resync",
        attempt_seq=None,
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee_bps=0,
        liquidity="taker",
        reason="operator_resync",
        filled_at=pd.Timestamp.now(tz="UTC"),
        client_order_id=None,
        leg_index=0,
        cumulative_executed_qty=None,
        simulated=False,
    )
    evidence = _snapshot(venue_session, 0, 100)
    assert evidence.journal_positions == (("BTCUSDT", Decimal("1")),)
    assert evidence.journal_trade_qty == ()


@pytest.mark.parametrize("wait", [-1, float("nan"), float("inf")])
def test_invalid_settlement_wait(venue_session, wait):
    from src.live.testnet_drill.reconcile import reconcile_session

    with pytest.raises(ValueError, match="settle_wait_s"):
        reconcile_session(venue_session, since_ms=0, settle_wait_s=wait)


def test_successful_empty_trade_read_is_retried_until_lag_settles(venue_session, monkeypatch):
    from src.live.testnet_drill import reconcile

    venue_session.client.trades = [[], [_trade()]]
    venue_session.journal.record_fill(
        kind="execution",
        attempt_seq=0,
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee_bps=0,
        liquidity="taker",
        reason="timeout_taker",
        filled_at=__import__("pandas").Timestamp.now(tz="UTC"),
        client_order_id="one",
        leg_index=0,
        cumulative_executed_qty=Decimal("1"),
        simulated=False,
    )
    venue_session.client.orders["one"]["executedQty"] = "1"
    venue_session.client.risk[0]["positionAmt"] = "1"
    clock = [0.0]
    monkeypatch.setattr(reconcile.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(reconcile.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    evidence = reconcile.reconcile_session(venue_session, since_ms=0, settle_wait_s=1)
    assert dict(evidence.venue_trade_qty)["one"] == Decimal("1")
    assert clock[0] > 0


def test_transient_read_exhaustion_is_explicit(venue_session, monkeypatch):
    from src.live.errors import TransientReadError
    from src.live.testnet_drill import reconcile

    def unavailable():
        raise TransientReadError("down", path="positionRisk", http_status=503, code=None, attempts=1)

    monkeypatch.setattr(venue_session.client, "position_risk", unavailable)
    evidence = reconcile.reconcile_session(venue_session, since_ms=0, settle_wait_s=0)
    assert evidence.read_errors == ("TransientReadError",)
    assert not any(item.passed for item in reconcile.evaluate_reconciliation(evidence, make_config()))


@pytest.mark.parametrize(
    ("code", "terminal", "raises"),
    [(-2013, "NOT_PLACED", False), (-2013, "OBSOLETE", False), (-2013, "CANCELED", True), (-2013, None, True),
     (-1000, "NOT_PLACED", True)],
)
def test_unknown_order_is_zero_only_for_unplaced_terminals(venue_session, code, terminal, raises):
    from src.live.errors import VenueError
    from src.live.testnet_drill.reconcile import _snapshot

    def missing(symbol, client_id):
        raise VenueError("unknown order", code=code, http_status=400, path="/fapi/v1/order", payload_digest="")

    venue_session.client.query_order = missing
    if terminal is not None:
        venue_session.journal.record_terminal("one", terminal)
    if raises:
        with pytest.raises(VenueError):
            _snapshot(venue_session, 0, 100)
    else:
        assert dict(_snapshot(venue_session, 0, 100).venue_executed) == {"one": Decimal(0)}
