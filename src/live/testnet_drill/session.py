"""One isolated drill run: venue client, journal, audit, filters, workspace."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.errors import LiveTradingError
from src.live.filters import SymbolFilters, parse_exchange_filters
from src.live.order_journal import OrderJournal
from src.live.planner import CLIENT_ORDER_NAMESPACE, symbol_tag
from src.live.rest import BinanceFuturesRestClient, KeepAliveTransport
from src.live.settings import LiveSettings
from src.live.testnet_drill.config import DrillConfig
from src.live.testnet_drill.faults import FaultAction, FaultInjectingTransport, FaultRule, WireRecord
from src.live.testnet_drill.guard import _PRODUCTION_PATH_FIELDS, DrillRefused, assert_drill_allowed, drill_scratch_root

logger = logging.getLogger("TestnetDrill")

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,15}")
_CLIENT_ID_RE = re.compile(r"^[.A-Za-z0-9_-]{1,36}$")
_WALLET_STATE_NAME = "last_wallet.json"


class SlateNotClean(LiveTradingError):  # noqa: N818 - spec pins the name
    """Clean slate could not flatten the drill account within the retry budget."""

    def __init__(self, message: str, *, residuals: Mapping[str, str]) -> None:
        super().__init__(message)
        self.residuals = dict(residuals)


class DirtySessionError(LiveTradingError):
    """Venue configuration could not be restored; stop all subsequent scenarios."""


@dataclass(frozen=True, slots=True)
class SlateReport:
    """Starting wallet and margin setup observed while flattening the drill account."""

    starting_wallet_balance: Decimal
    leverage: tuple[tuple[str, int], ...]
    margin_type: tuple[tuple[str, str], ...]
    testnet_reset_suspected: bool


@dataclass(frozen=True, slots=True)
class DrillSession:
    """One isolated drill run bound to a workspace under the drill scratch root."""

    settings: LiveSettings
    config: DrillConfig
    run_id: str
    workspace: Path
    client_order_prefix: str
    client: BinanceFuturesRestClient
    journal: OrderJournal
    audit: AuditLog
    filters: Mapping[str, SymbolFilters]
    transport: FaultInjectingTransport
    opened_at: float
    starting_wallet_balance: Decimal | None = None
    counterparties: list[DrillSession] = field(default_factory=list, repr=False, compare=False)

    @property
    def aux_client(self) -> BinanceFuturesRestClient | None:
        """Client of the independently credentialed counterparty account."""
        return self.counterparties[0].client if self.counterparties else None

    def next_client_order_id(self, symbol: str) -> str:
        """Allocate the next drill-namespaced client order id for a symbol."""
        candidate = f"{self.client_order_prefix}-{symbol_tag(symbol)[:6]}-{uuid4().hex[:16]}"
        if _CLIENT_ID_RE.fullmatch(candidate) is None:
            raise DrillRefused("drill refused: client_order_id")
        return candidate

    def guarded_new_order(self, params: Mapping[str, Any], *, raw: bool = False) -> Any:
        """Send one drill order after prefix and notional-cap checks."""
        order = dict(params)
        symbol = str(order.get("symbol", ""))
        if symbol not in self.filters or symbol not in self.config.symbols:
            raise DrillRefused("drill refused: symbol")
        raw_id = order.get("newClientOrderId")
        if raw_id is None:
            order["newClientOrderId"] = self.next_client_order_id(symbol)
        elif _CLIENT_ID_RE.fullmatch(str(raw_id)) is None or not (
            str(raw_id).startswith(self.client_order_prefix + "-")
            or (
                raw
                and re.match(r"^\d{8}-", str(raw_id))
                and any(s.client_order_id == str(raw_id) for s in self.journal.submits())
            )
        ):
            raise DrillRefused("drill refused: client_order_id")
        if order.get("type") == "MARKET":
            order_for_cap = {**order, "price": str(_market_price(self.client, symbol))}
        else:
            order_for_cap = order
        notional = _order_notional(order_for_cap)
        if notional is None:
            raise DrillRefused("drill refused: notional")
        if notional > self.config.max_probe_notional_usdt:
            raise DrillRefused("drill refused: max_notional")
        return self.client.new_order(order)


def _order_notional(order: Mapping[str, Any]) -> Decimal | None:
    try:
        if order.get("quantity") is None or order.get("price") is None:
            return None
        quantity = Decimal(str(order["quantity"]))
        price = Decimal(str(order["price"]))
        if not quantity.is_finite() or not price.is_finite() or quantity <= 0 or price <= 0:
            return None
        return quantity * price
    except (InvalidOperation, ValueError, TypeError):
        return None


def _market_price(client: BinanceFuturesRestClient, symbol: str) -> Decimal:
    quote = client.book_ticker(symbol)
    prices = [Decimal(str(quote[key])) for key in ("bidPrice", "askPrice")]
    if any(not price.is_finite() or price <= 0 for price in prices):
        raise DrillRefused("drill refused: market_price")
    return max(prices)


def open_drill_session(
    settings: LiveSettings,
    config: DrillConfig,
    *,
    run_id: str,
    rules: Sequence[FaultRule] = (),
    clock: Callable[[], float] = time.time,
    inner: Any | None = None,
    workspace: Path | None = None,
) -> DrillSession:
    """Open an isolated drill session against the testnet venue."""
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise DrillRefused("drill refused: run_id")
    workspace = workspace if workspace is not None else drill_scratch_root() / run_id
    assert_drill_allowed(settings, workspace)
    if multiprocessing.parent_process() is None and any(
        rule.action in (FaultAction.KILL_BEFORE_DELIVERY, FaultAction.KILL_AFTER_DELIVERY, FaultAction.KILL_ON_RESPONSE)
        for rule in rules
    ):
        raise DrillRefused("drill refused: parent_kill")
    isolated = settings.model_copy(
        update={
            "record_run_id": None,
            **{
                name: str(workspace / filename)
                for name, filename in {
                    "ledger_path": "position_ledger.json",
                    "order_journal_path": "order_journal.jsonl",
                    "tax_ledger_dir": "tax_ledger",
                    "heartbeat_path": "heartbeat.json",
                    "fills_dir": "fills",
                    "execution_quality_dir": "execution_quality",
                    "portfolio_state_dir": "portfolio_state",
                    "microstructure_dir": "microstructure",
                    "alert_outbox_path": "alert_outbox.json",
                }.items()
            },
        }
    )
    assert_drill_allowed(isolated, workspace)
    if any(
        not (workspace / name).resolve().is_relative_to(workspace.resolve())
        for name in ("audit.jsonl", "order_journal.jsonl", "last_wallet.json", "wire.jsonl")
    ):
        raise DrillRefused("drill refused: session_paths")
    workspace.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(workspace / "audit.jsonl")
    journal = OrderJournal(workspace / "order_journal.jsonl")
    transport = FaultInjectingTransport(
        inner if inner is not None else KeepAliveTransport(settings.order_base_url),
        rules,
        base_url=settings.order_base_url,
        sleep=time.sleep,
        kill=os._exit,
        on_record=lambda record: _persist_wire(workspace / "wire.jsonl", record),
    )
    client = BinanceFuturesRestClient(
        settings.order_base_url,
        settings.order_api_key,
        settings.order_api_secret,
        settings.mode,
        audit,
        recv_window_ms=config.recv_window_ms,
        session=transport,
    )
    try:
        client.sync_server_time()
        filters = parse_exchange_filters(client.exchange_info())
        missing = [symbol for symbol in config.symbols if symbol not in filters]
        if missing:
            raise DrillRefused("drill refused: symbols")
    except BaseException:
        with contextlib.suppress(Exception):
            transport.close()
        with contextlib.suppress(Exception):
            audit.close()
        raise
    suffix = hashlib.sha256(run_id.encode()).hexdigest()[:6]
    return DrillSession(
        settings=isolated,
        config=config,
        run_id=run_id,
        workspace=workspace,
        client_order_prefix=f"{CLIENT_ORDER_NAMESPACE}drl{suffix}",
        client=client,
        journal=journal,
        audit=audit,
        filters=filters,
        transport=transport,
        opened_at=clock(),
    )


def open_counterparty_session(session: DrillSession, *, inner: Any | None = None) -> DrillSession:
    """Open a flat, separately credentialed testnet account with isolated state."""
    if session.counterparties:
        return session.counterparties[0]
    settings = session.settings
    key, secret = settings.order_api_key2, settings.order_api_secret2
    if key is None or secret is None or not key.get_secret_value() or not secret.get_secret_value():
        raise DrillRefused("drill refused: counterparty_credentials")
    other_keys = (settings.order_api_key, settings.api_key)
    if any(other is not None and key.get_secret_value() == other.get_secret_value() for other in other_keys):
        raise DrillRefused("drill refused: counterparty_key_separation")
    child_settings = settings.model_copy(
        update={
            "order_api_key": key,
            "order_api_secret": secret,
            "api_key": None,
            "api_secret": None,
            **dict.fromkeys(_PRODUCTION_PATH_FIELDS),
        }
    )
    aux = open_drill_session(
        child_settings, session.config, run_id=session.run_id, workspace=session.workspace / "counterparty", inner=inner
    )
    aux = replace(aux, client_order_prefix="aux" + session.client_order_prefix.removeprefix(CLIENT_ORDER_NAMESPACE))
    try:
        if _read_positions(aux.client) or aux.client.open_orders():
            raise DrillRefused("drill refused: counterparty_not_flat")
    except BaseException:
        try:
            aux.transport.close()
        finally:
            aux.audit.close()
        raise
    session.counterparties.append(aux)
    return aux


def _persist_wire(path: Path, record: WireRecord) -> None:
    payload = asdict(record)
    payload["response_body"] = record.response_body.hex()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_positions(client: BinanceFuturesRestClient) -> dict[str, Decimal]:
    risk = client.request("GET", "/fapi/v2/positionRisk", signed=True)
    if not isinstance(risk, list):
        raise DataIntegrityError("positionRisk endpoint returned an unexpected schema")
    positions: dict[str, Decimal] = {}
    for entry in risk:
        if not isinstance(entry, dict) or "symbol" not in entry or "positionAmt" not in entry:
            raise DataIntegrityError("positionRisk row missing symbol/positionAmt")
        qty = Decimal(str(entry["positionAmt"]))
        if not qty.is_finite() or str(entry.get("positionSide", "BOTH")) != "BOTH":
            raise DrillRefused("drill refused: position_mode")
        if qty != 0:
            positions[str(entry["symbol"])] = qty
    return positions


def _margin_setup(
    account: Mapping[str, Any], symbols: Sequence[str]
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]]:
    rows = account.get("positions")
    by_symbol: dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, list):
        for entry in rows:
            if isinstance(entry, dict) and "symbol" in entry:
                by_symbol[str(entry["symbol"])] = entry
    leverage: list[tuple[str, int]] = []
    margin_type: list[tuple[str, str]] = []
    for symbol in symbols:
        entry = by_symbol.get(symbol, {})
        try:
            leverage.append((symbol, int(str(entry.get("leverage", 0)))))
        except (TypeError, ValueError):
            leverage.append((symbol, 0))
        raw_margin = entry.get("marginType", entry.get("isolated", "unknown"))
        if isinstance(raw_margin, bool):
            raw_margin = "isolated" if raw_margin else "cross"
        margin_type.append((symbol, str(raw_margin)))
    return tuple(leverage), tuple(margin_type)


def _flatten_positions(session: DrillSession, positions: Mapping[str, Decimal]) -> None:
    for symbol in sorted(positions):
        qty = positions[symbol]
        if qty == 0:
            continue
        filters = session.filters[symbol]
        price = _market_price(session.client, symbol)
        cap_qty = (session.config.max_probe_notional_usdt / price / filters.step_size).to_integral_value(
            rounding=ROUND_DOWN
        ) * filters.step_size
        quantity = min(abs(qty), cap_qty, filters.max_qty)
        if quantity <= 0:
            raise DrillRefused("drill refused: flatten_cap")
        order = {
            "symbol": symbol,
            "side": "SELL" if qty > 0 else "BUY",
            "type": "MARKET",
            "quantity": str(quantity),
            "reduceOnly": "true",
        }
        if session.starting_wallet_balance is not None:
            from src.live.testnet_drill.execution import settle_journal, submit_probe

            submit_probe(session, order)
            settle_journal(session)
        else:
            session.guarded_new_order(order)


def _wallet_state_path(session: DrillSession) -> Path:
    return session.workspace / _WALLET_STATE_NAME


def _testnet_reset_suspected(session: DrillSession, wallet: Decimal) -> bool:
    path = _wallet_state_path(session)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        previous = Decimal(str(raw["balance"]))
        ended_at_ms = int(raw["ended_at_ms"])
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation):
        return False
    income = session.client.income(start_time_ms=ended_at_ms, end_time_ms=int(session.opened_at * 1000))
    if len(income) >= 1000:
        return False
    explained = sum((Decimal(str(row["income"])) for row in income if row.get("asset") == "USDT"), Decimal(0))
    return abs(wallet - previous - explained) > session.config.fee_abs_tolerance_usdt


def prepare_clean_slate(session: DrillSession) -> SlateReport:
    """Cancel every open order and flatten every position; raise if not flat."""
    account = session.client.request("GET", "/fapi/v2/account", signed=True)
    if not isinstance(account, dict) or "totalWalletBalance" not in account:
        raise DataIntegrityError("account endpoint returned an unexpected schema")
    wallet = Decimal(str(account["totalWalletBalance"]))
    leverage, margin_type = _margin_setup(account, list(session.config.symbols))
    residuals: dict[str, str] = {}
    for _ in range(session.config.slate_max_attempts):
        positions = _read_positions(session.client)
        open_orders = session.client.open_orders()
        symbols = {*positions, *(str(o.get("symbol", "")) for o in open_orders if isinstance(o, dict))}
        if symbols - set(session.config.symbols):
            raise DrillRefused("drill refused: out_of_scope_account_state")
        for symbol in sorted(s for s in symbols if s):
            session.client.request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        _flatten_positions(session, _read_positions(session.client))
        positions = _read_positions(session.client)
        open_orders = session.client.open_orders()
        residuals = {symbol: str(qty) for symbol, qty in sorted(positions.items())}
        if open_orders:
            residuals["open_orders"] = str(len(open_orders))
        if not residuals:
            break
    if residuals:
        raise SlateNotClean(f"drill slate not clean: {', '.join(sorted(residuals))}", residuals=residuals)
    return SlateReport(
        starting_wallet_balance=wallet,
        leverage=leverage,
        margin_type=margin_type,
        testnet_reset_suspected=_testnet_reset_suspected(session, wallet),
    )


def teardown_session(session: DrillSession, *, close_handles: bool = True) -> None:
    """Cancel own open orders, flatten, close handles; never raises on cleanup failure."""
    try:
        for aux in session.counterparties:
            teardown_session(aux, close_handles=close_handles)
        try:
            for order in session.client.open_orders():
                if not isinstance(order, dict):
                    continue
                if (
                    str(order.get("clientOrderId", "")).startswith(session.client_order_prefix + "-")
                    and str(order.get("symbol", "")) in session.config.symbols
                ):
                    session.client.cancel_order(str(order.get("symbol", "")), str(order.get("clientOrderId", "")))
        except Exception as exc:
            with contextlib.suppress(Exception):
                session.audit.record("teardown_issue", stage="cancel", error=type(exc).__name__)
        try:
            residuals: dict[str, Decimal] = {}
            if session.starting_wallet_balance is not None:
                from src.live.testnet_drill.execution import settle_journal

                settle_journal(session)
            for _ in range(session.config.slate_max_attempts):
                _flatten_positions(session, _read_positions(session.client))
                residuals = _read_positions(session.client)
                if not residuals:
                    break
            if residuals:
                raise SlateNotClean("drill teardown not flat", residuals={s: str(q) for s, q in residuals.items()})
            account = session.client.account()
            _wallet_state_path(session).write_text(
                json.dumps(
                    {
                        "balance": str(account["totalWalletBalance"]),
                        "run_id": session.run_id,
                        "ended_at_ms": int(time.time() * 1000),
                    }
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                session.audit.record("teardown_issue", stage="flatten", error=type(exc).__name__)
    finally:
        if close_handles:
            with contextlib.suppress(Exception):
                session.transport.close()
            with contextlib.suppress(Exception):
                session.audit.close()


def close_session_handles(session: DrillSession) -> None:
    """Close all account transports and audit streams after final reconciliation."""
    try:
        for aux in session.counterparties:
            close_session_handles(aux)
    finally:
        try:
            session.transport.close()
        finally:
            session.audit.close()
