"""Yearly tax summary: KST-year facts from the Decimal tax ledger (moving average only)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, Inexact, InvalidOperation, localcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.durable_io import durable_write_text
from src.common.errors import DataIntegrityError
from src.live.settings import ExecutionMode
from src.live.tax_basis import (
    COST_BASIS_MOVING_AVERAGE,
    FLAT_POSITION,
    TAX_DECIMAL_PRECISION,
    AverageCostPosition,
    FillRealization,
    fold_symbol,
    require_moving_average,
)
from src.live.tax_ledger import read_tax_ledger
from src.live.tax_schema import (
    BOUNDARY_MARK_NOT_SUPPLIED,
    DELIVERY_SETTLEMENT_INCOME_TYPE,
    BoundaryMark,
    TaxCoverage,
    TaxRecord,
    tax_event_sort_key,
)

if TYPE_CHECKING:
    from src.live.settings import LiveSettings

TAX_SUMMARY_SCHEMA_VERSION: Final[int] = 1
TAX_SUMMARY_SOURCES: Final[Mapping[str, frozenset[str]]] = {
    "venue": frozenset({"venue"}),
    "simulated": frozenset({"simulated", "delisting_settlement"}),
}
SYMBOL_DECIMAL_BUCKETS: Final[tuple[str, ...]] = ("buy_quantity", "sell_quantity", "buy_notional", "sell_notional", "closed_quantity", "entry_notional_closed", "exit_notional_closed", "trading_pnl_long", "trading_pnl_short", "trading_pnl_unattributed", "trading_pnl", "delivery_settlement_pnl", "trade_fees", "funding", "net_pnl")
TAX_SUMMARY_KEYS: Final[tuple[str, ...]] = ("schema_version", "year", "timezone", "period_start", "period_end", "regime_start", "mode", "source", "cost_basis", "settlement_asset", "per_symbol", "totals", "opening_inventory", "closing_inventory", "other_income", "transfers", "reconciliation")
RECONCILIATION_STATUSES: Final[frozenset[str]] = frozenset({"reconciled", "incomplete", "not_applicable"})
COVERAGE_ISSUE_CODES: Final[frozenset[str]] = frozenset({"coverage_unknown", "genesis_missing", "genesis_not_flat", "period_precedes_genesis", "income_coverage_unknown", "income_coverage_late", "income_gap", "collection_behind_period_end", "pre_genesis_inventory_discarded"})
MISMATCH_ISSUE_CODES: Final[frozenset[str]] = frozenset({"fold_pnl_mismatch", "income_pnl_mismatch", "commission_mismatch"})

@dataclass(frozen=True, slots=True)
class TaxSummaryConfig:
    """Typed parameters of the yearly tax summary; LiveSettings is the single source of defaults.

    timezone: IANA zone whose calendar year is the tax year (Korea: Asia/Seoul).
    regime_start: first local date of the virtual-asset taxation regime; positions opened before
        its local midnight are flagged ``opened_before_regime`` for deemed-acquisition review.
    settlement_asset: asset of P&L, funding and fee totals (USDT for USDT-M). Amounts in any other
        asset are reported per asset, never summed into it.
    reconcile_abs_tolerance / reconcile_per_fill_tolerance: per symbol and period, two amounts
        reconcile when |difference| <= abs + per_fill * n, n being the number of fills the compared
        amounts aggregate (venue amounts are rounded per fill).

    Raises:
        ValueError: unknown timezone, empty settlement_asset, or a tolerance that is not a finite,
            non-negative Decimal.
    """

    timezone: str
    regime_start: date
    settlement_asset: str
    reconcile_abs_tolerance: Decimal
    reconcile_per_fill_tolerance: Decimal

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {self.timezone!r}") from exc
        if not isinstance(self.settlement_asset, str) or not self.settlement_asset:
            raise ValueError("settlement_asset must be non-empty")
        if self.settlement_asset != self.settlement_asset.upper() or any(
            ch.isspace() for ch in self.settlement_asset
        ):
            raise ValueError(f"settlement_asset must be upper-case without whitespace: {self.settlement_asset!r}")
        for name in ("reconcile_abs_tolerance", "reconcile_per_fill_tolerance"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative Decimal")

    @classmethod
    def from_settings(cls, settings: LiveSettings) -> TaxSummaryConfig:
        return cls(
            timezone=settings.tax_timezone,
            regime_start=settings.tax_regime_start,
            settlement_asset=settings.tax_settlement_asset,
            reconcile_abs_tolerance=settings.tax_reconcile_abs_tolerance,
            reconcile_per_fill_tolerance=settings.tax_reconcile_per_fill_tolerance,
        )

def tax_year_bounds(year: int, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC instants [start, end) of calendar ``year`` in ``timezone``.

    Raises:
        ValueError: ``year`` is not an int (bool rejected) or ``timezone`` is unknown.
    """
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(f"year must be an int: {year!r}")
    try:
        zone = ZoneInfo(timezone)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown timezone: {timezone!r}") from exc
    start = datetime(year, 1, 1, tzinfo=zone).astimezone(ZoneInfo("UTC"))
    end = datetime(year + 1, 1, 1, tzinfo=zone).astimezone(ZoneInfo("UTC"))
    return (pd.Timestamp(start), pd.Timestamp(end))

def regime_boundary_utc(config: TaxSummaryConfig) -> pd.Timestamp:
    """Instant at which pre-regime inventory is valued; the 1h bar ending here supplies the OHLCV boundary mark."""
    zone = ZoneInfo(config.timezone)
    local_midnight = datetime(
        config.regime_start.year, config.regime_start.month, config.regime_start.day, tzinfo=zone
    )
    return pd.Timestamp(local_midnight.astimezone(ZoneInfo("UTC")))

def tax_source_for_mode(mode: ExecutionMode) -> Literal["venue", "simulated"]:
    """Ledger source written by an execution mode: suppressed modes (SHADOW, PAPER) write simulated records; live modes collect venue records."""
    if mode in (ExecutionMode.SHADOW, ExecutionMode.PAPER):
        return "simulated"
    return "venue"

def pre_regime_inventory_symbols(summary: Mapping[str, Any]) -> frozenset[str]:
    out: set[str] = set()
    for section in ("opening_inventory", "closing_inventory"):
        inventory = summary.get(section, {})
        if isinstance(inventory, Mapping):
            for symbol, entry in inventory.items():
                if isinstance(entry, Mapping) and entry.get("opened_before_regime") is True:
                    out.add(str(symbol))
    return frozenset(out)

def _zero_symbol_entry() -> dict[str, Any]:
    entry: dict[str, Any] = {key: Decimal(0) for key in SYMBOL_DECIMAL_BUCKETS}
    entry["n_fills"] = 0
    entry["other_asset_fees"] = {}
    return entry

def _check_ledger_purity(records: Sequence[TaxRecord], source: str, settlement_asset: str) -> str | None:
    allowed = TAX_SUMMARY_SOURCES[source]
    modes: set[str] = set()
    for record in records:
        if record.source not in allowed:
            raise DataIntegrityError(f"record {record.record_id!r} source {record.source!r} outside family {source!r}")
        modes.add(record.mode)
        if record.kind in ("TRADE", "REALIZED_PNL", "FUNDING_FEE") and record.income_asset != settlement_asset:
            raise DataIntegrityError(f"record {record.record_id!r} asset {record.income_asset!r} is not the settlement asset {settlement_asset!r}")
        if source == "simulated" and record.kind not in ("TRADE", "FUNDING_FEE"):
            raise DataIntegrityError(f"simulated record {record.record_id!r} kind {record.kind!r} is not TRADE/FUNDING_FEE")
    if len(modes) > 1:
        raise DataIntegrityError(f"ledger spans more than one mode: {sorted(modes)}")
    return next(iter(modes)) if modes else None

def _is_delivery(record: TaxRecord) -> bool:
    return record.kind == "REALIZED_PNL" and record.income_type == DELIVERY_SETTLEMENT_INCOME_TYPE

def _fold_all(records: Sequence[TaxRecord], coverage: TaxCoverage | None, period_end: pd.Timestamp) -> tuple[dict[str, tuple[FillRealization, ...]], dict[str, TaxRecord], set[str]]:
    by_symbol: dict[str, list[TaxRecord]] = {}
    for record in sorted(records, key=tax_event_sort_key):
        if record.event_time < period_end and (record.kind == "TRADE" or _is_delivery(record)):
            by_symbol.setdefault(record.symbol, []).append(record)
    realizations: dict[str, tuple[FillRealization, ...]] = {}
    record_by_id: dict[str, TaxRecord] = {}
    pre_genesis_symbols: set[str] = set()
    genesis_at = coverage.genesis_at if coverage is not None else None
    flat_genesis = coverage is not None and coverage.genesis_at is not None and not any(v != 0 for v in coverage.genesis_positions.values())
    for symbol, rows in by_symbol.items():
        for record in rows:
            record_by_id[record.record_id] = record
        if flat_genesis and genesis_at is not None and genesis_at < period_end:
            pre = [r for r in rows if r.event_time < genesis_at]
            post = [r for r in rows if r.event_time >= genesis_at]
            pre_steps = fold_symbol(pre, initial=FLAT_POSITION)
            if pre_steps and pre_steps[-1].position_after.quantity != 0:
                pre_genesis_symbols.add(symbol)
            realizations[symbol] = pre_steps + fold_symbol(post, initial=FLAT_POSITION)
        else:
            realizations[symbol] = fold_symbol(rows, initial=FLAT_POSITION) if rows else ()
    return realizations, record_by_id, pre_genesis_symbols


def _position_at(steps: tuple[FillRealization, ...], cutoff: pd.Timestamp, coverage: TaxCoverage | None, period_end: pd.Timestamp) -> AverageCostPosition:
    position: AverageCostPosition = FLAT_POSITION
    reset = coverage.genesis_at if coverage is not None and not coverage.genesis_positions else None
    if reset is not None and reset >= period_end:
        reset = None
    for step in steps:
        if step.event_time < cutoff:
            if reset is None or cutoff < reset or step.event_time >= reset:
                position = step.position_after
        else:
            break
    return position

def _inventory_entry(position: AverageCostPosition, boundary: pd.Timestamp, marks: Mapping[str, BoundaryMark] | None, symbol: str) -> dict[str, Any] | None:
    if position.quantity == 0 or position.opened_at is None:
        return None
    base: dict[str, Any] = {"quantity": position.quantity, "avg_entry": position.avg_entry, "opened_at": position.opened_at}
    if position.opened_at >= boundary:
        base.update({"opened_before_regime": False, "boundary_mark": None, "boundary_mark_source": None, "boundary_mark_unavailable_reason": None})
        return base
    base["opened_before_regime"] = True
    if marks is not None and symbol in marks:
        mark = marks[symbol]
        base.update({"boundary_mark": mark.price, "boundary_mark_source": mark.source, "boundary_mark_unavailable_reason": mark.unavailable_reason})
        return base
    base.update({"boundary_mark": None, "boundary_mark_source": None, "boundary_mark_unavailable_reason": BOUNDARY_MARK_NOT_SUPPLIED})
    return base

def _coverage_issue_list(coverage: TaxCoverage | None, period_start: pd.Timestamp, period_end: pd.Timestamp, pre_genesis_symbols: set[str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    def _add(code: str, symbol: str | None = None) -> None:
        issues.append({"code": code, "symbol": symbol, "detail": code if symbol is None else f"{code}:{symbol}"})

    def _key(item: dict[str, Any]) -> tuple[str, str]:
        return (item["code"], item["symbol"] or "")
    if coverage is None:
        _add("coverage_unknown")
        return sorted(issues, key=_key)
    if coverage.genesis_at is None:
        _add("genesis_missing")
    genesis_at = coverage.genesis_at
    if any(v != 0 for v in coverage.genesis_positions.values()):
        _add("genesis_not_flat")
    if genesis_at is not None and genesis_at > period_start:
        _add("period_precedes_genesis")
    if coverage.income_covered_from is None:
        _add("income_coverage_unknown")
    elif genesis_at is not None and coverage.income_covered_from > genesis_at:
        _add("income_coverage_late")
    if genesis_at is not None and any(not (g_end < genesis_at or g_start >= period_end) for g_start, g_end in coverage.income_gaps):
        _add("income_gap")
    if coverage.collected_through is None or coverage.collected_through < period_end:
        _add("collection_behind_period_end")
    if genesis_at is not None and pre_genesis_symbols and period_start < genesis_at:
        for symbol in sorted(pre_genesis_symbols):
            _add("pre_genesis_inventory_discarded", symbol)
    return sorted(issues, key=_key)

def _tolerance(config: TaxSummaryConfig, n: int, diff: Decimal) -> bool:
    limit = config.reconcile_abs_tolerance + config.reconcile_per_fill_tolerance * n
    return abs(diff) <= limit

def build_tax_year_summary(
    records: Sequence[TaxRecord],
    year: int,
    *,
    source: Literal["venue", "simulated"],
    config: TaxSummaryConfig,
    cost_basis: str = COST_BASIS_MOVING_AVERAGE,
    coverage: TaxCoverage | None = None,
    boundary_marks: Mapping[str, BoundaryMark] | None = None,
) -> dict[str, Any]:
    """Summarize one tax year of a ledger as jurisdiction-agnostic facts.

    The whole ledger (inception to date) is validated and folded per symbol with the signed
    moving-average basis; only the output is restricted to [period_start, period_end) of ``year``
    in ``config.timezone``. Positions open at a boundary appear as opening/closing inventory, so
    carry-in basis is never lost. No tax rate, deduction or income classification is computed: the
    output preserves the facts any classification needs.

    P&L sources:
        venue: per-fill TRADE.realized_pnl (exchange average-entry, one-way mode) is the trading
            P&L; venue delivery settlements (DELIVERED_SETTELMENT income) close the fold position
            and add their amount. The fold's realized P&L, income REALIZED_PNL and income
            COMMISSION are reconciliation inputs only, never added to P&L.
        simulated: the fold's realized P&L is the trading P&L.
    FUNDING_FEE amounts form the ``funding`` bucket. TRANSFER rows are reported under
    ``transfers`` and never enter P&L. Every other venue income type is itemized under
    ``other_income`` by raw income type and asset, outside P&L.

    Completeness (venue): the period is ``reconciled`` only if ``coverage`` proves a flat genesis at
    or before period_start, contiguous income collection from genesis through period_end and no
    income gap; otherwise ``incomplete`` with coverage issue codes, and reconciliation differences
    are reported as issues instead of raised. Simulated summaries are ``not_applicable``.

    Args:
        records: Validated records of one ledger directory (``read_tax_ledger`` output).
        year: Tax year in ``config.timezone``.
        source: Ledger family to summarize; every record must belong to it.
        config: Typed summary parameters.
        cost_basis: Must be COST_BASIS_MOVING_AVERAGE.
        coverage: Venue collection coverage; must be None for ``source="simulated"``.
        boundary_marks: Optional per-symbol regime-boundary reference prices with provenance
            (operator-supplied, or the close of the 1h bar ending at the local regime boundary),
            attached verbatim to inventory entries opened before the regime. The summary never
            chooses between it and avg_entry.

    Returns:
        A dict with exactly TAX_SUMMARY_KEYS (see Core Invariants for nested schema).

    Raises:
        ValueError: unsupported cost_basis or source; coverage given for a simulated summary; a
            boundary mark that names a symbol with no pre-regime inventory entry in this summary.
        DataIntegrityError: records outside the source family; more than one distinct mode; a
            TRADE, REALIZED_PNL or FUNDING_FEE amount not in the settlement asset; a simulated
            record of a kind other than TRADE/FUNDING_FEE; a fold error; or a reconciliation
            difference beyond tolerance while coverage is complete.
    """
    require_moving_average(cost_basis)
    if source not in TAX_SUMMARY_SOURCES:
        raise ValueError(f"unsupported source: {source!r}")
    if source == "simulated" and coverage is not None:
        raise ValueError("coverage must be None for a simulated summary")
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(f"year must be an int: {year!r}")
    with localcontext() as ctx:
        ctx.prec = TAX_DECIMAL_PRECISION
        ctx.traps[Inexact] = True
        ctx.traps[InvalidOperation] = True
        return _build(year, records, source=source, config=config, coverage=coverage, marks=boundary_marks)

def _build(
    year: int,
    records: Sequence[TaxRecord],
    *,
    source: str,
    config: TaxSummaryConfig,
    coverage: TaxCoverage | None,
    marks: Mapping[str, BoundaryMark] | None,
) -> dict[str, Any]:
    ordered = sorted(records, key=tax_event_sort_key)
    mode = _check_ledger_purity(ordered, source, config.settlement_asset)
    period_start, period_end = tax_year_bounds(year, config.timezone)
    boundary = regime_boundary_utc(config)
    realizations, record_by_id, pre_genesis = _fold_all(ordered, coverage, period_end)
    opening: dict[str, Any] = {}
    closing: dict[str, Any] = {}
    for symbol, steps in realizations.items():
        open_pos = _position_at(steps, period_start, coverage, period_end)
        close_pos = _position_at(steps, period_end, coverage, period_end)
        open_entry = _inventory_entry(open_pos, boundary, marks, symbol)
        close_entry = _inventory_entry(close_pos, boundary, marks, symbol)
        if open_entry is not None:
            opening[symbol] = open_entry
        if close_entry is not None:
            closing[symbol] = close_entry
    if marks:
        known = {
            symbol for inventory in (opening, closing) for symbol, entry in inventory.items()
            if entry["opened_before_regime"]
        }
        for symbol in marks:
            if symbol not in known:
                raise ValueError(f"boundary mark for symbol without pre-regime inventory: {symbol!r}")
    per_symbol, transfers, other_income = _buckets(
        ordered, realizations, record_by_id, period_start, period_end,
        source=source, settlement=config.settlement_asset,
    )
    for symbol in list(opening) + list(closing):
        if symbol not in per_symbol:
            per_symbol[symbol] = _zero_symbol_entry()
    totals = _zero_symbol_entry()
    for entry in per_symbol.values():
        for key in SYMBOL_DECIMAL_BUCKETS:
            totals[key] = totals[key] + entry[key]
        totals["n_fills"] = totals["n_fills"] + entry["n_fills"]
        for asset, amount in entry["other_asset_fees"].items():
            totals["other_asset_fees"][asset] = totals["other_asset_fees"].get(asset, Decimal(0)) + amount
    for entry in per_symbol.values():
        assert entry["trading_pnl"] == entry["trading_pnl_long"] + entry["trading_pnl_short"] + entry["trading_pnl_unattributed"]
        assert entry["net_pnl"] == entry["trading_pnl"] + entry["funding"] - entry["trade_fees"]
    reconciliation = _reconcile(
        ordered, realizations, record_by_id, period_start, period_end,
        source=source, config=config, coverage=coverage, pre_genesis=pre_genesis,
    )
    return {
        "schema_version": TAX_SUMMARY_SCHEMA_VERSION,
        "year": year,
        "timezone": config.timezone,
        "period_start": period_start,
        "period_end": period_end,
        "regime_start": config.regime_start,
        "mode": mode,
        "source": source,
        "cost_basis": COST_BASIS_MOVING_AVERAGE,
        "settlement_asset": config.settlement_asset,
        "per_symbol": per_symbol,
        "totals": totals,
        "opening_inventory": opening,
        "closing_inventory": closing,
        "other_income": other_income,
        "transfers": transfers,
        "reconciliation": reconciliation,
    }

def _buckets(
    ordered: Sequence[TaxRecord],
    realizations: Mapping[str, tuple[FillRealization, ...]],
    record_by_id: Mapping[str, TaxRecord],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    *,
    source: str,
    settlement: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_symbol: dict[str, Any] = {}
    transfers: dict[str, dict[str, Decimal]] = {}
    other_income: dict[str, dict[str, Decimal]] = {}

    def _entry(symbol: str) -> dict[str, Any]:
        if symbol not in per_symbol:
            per_symbol[symbol] = _zero_symbol_entry()
        result: dict[str, Any] = per_symbol[symbol]
        return result

    for record in ordered:
        in_period = period_start <= record.event_time < period_end
        if not in_period:
            continue
        if record.kind == "TRADE":
            _add_trade_buckets(_entry(record.symbol), record, settlement)
        elif record.kind == "FUNDING_FEE":
            _entry(record.symbol)["funding"] += record.realized_pnl
        elif record.kind == "TRANSFER":
            transfers.setdefault(record.income_type, {}).setdefault(record.income_asset, Decimal(0))
            transfers[record.income_type][record.income_asset] += record.realized_pnl
        elif record.kind == "UNCLASSIFIED":
            other_income.setdefault(record.income_type, {}).setdefault(record.income_asset, Decimal(0))
            other_income[record.income_type][record.income_asset] += record.realized_pnl
    for symbol, steps in realizations.items():
        for step in steps:
            if not (period_start <= step.event_time < period_end):
                continue
            _add_realization_buckets(_entry(symbol), step, record_by_id[step.record_id], source)
    for entry in per_symbol.values():
        entry["trading_pnl"] = (
            entry["trading_pnl_long"] + entry["trading_pnl_short"] + entry["trading_pnl_unattributed"]
        )
        entry["net_pnl"] = entry["trading_pnl"] + entry["funding"] - entry["trade_fees"]
    return per_symbol, transfers, other_income


def _add_trade_buckets(entry: dict[str, Any], record: TaxRecord, settlement: str) -> None:
    entry["n_fills"] += 1
    side = record.side.lower()
    entry[f"{side}_quantity"] += record.quantity
    entry[f"{side}_notional"] += record.quote_qty
    if record.fee != 0:
        if record.fee_asset == settlement:
            entry["trade_fees"] += record.fee
        else:
            fees = entry["other_asset_fees"]
            fees[record.fee_asset] = fees.get(record.fee_asset, Decimal(0)) + record.fee


def _add_realization_buckets(entry: dict[str, Any], step: FillRealization, record: TaxRecord, source: str) -> None:
    for key in ("closed_quantity", "entry_notional_closed", "exit_notional_closed"):
        entry[key] += getattr(step, key)
    amount = record.realized_pnl if source == "venue" else step.realized_pnl
    direction = step.closed_direction or "unattributed"
    entry[f"trading_pnl_{direction}"] += amount
    if _is_delivery(record):
        entry["delivery_settlement_pnl"] += record.realized_pnl

def _reconcile(
    ordered: Sequence[TaxRecord],
    realizations: Mapping[str, tuple[FillRealization, ...]],
    record_by_id: Mapping[str, TaxRecord],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    *,
    source: str,
    config: TaxSummaryConfig,
    coverage: TaxCoverage | None,
    pre_genesis: set[str],
) -> dict[str, Any]:
    if source == "simulated":
        return {"status": "not_applicable", "issues": [], "coverage": None, "per_symbol": {}}
    issues = _coverage_issue_list(coverage, period_start, period_end, pre_genesis)
    has_coverage_issue = len(issues) > 0
    symbols, fold_sums, venue_sums, income_sums, fee_by_asset, comm_by_asset, n_close, fills_with_fee_asset = _recon_inputs(ordered, realizations, record_by_id, period_start, period_end)
    per_detail: dict[str, Any] = {}
    for symbol in sorted(symbols):
        per_detail[symbol] = {"fold_trading_pnl": fold_sums.get(symbol, Decimal(0)), "venue_trading_pnl": venue_sums.get(symbol, Decimal(0)), "income_realized_pnl": income_sums.get(symbol, Decimal(0)), "trade_fees_by_asset": dict(fee_by_asset.get(symbol, {})), "income_commission_by_asset": dict(comm_by_asset.get(symbol, {}))}
    mismatches = _recon_mismatches(symbols, fold_sums, venue_sums, income_sums, fee_by_asset, comm_by_asset, n_close, fills_with_fee_asset, config)
    if has_coverage_issue:
        issues = sorted(issues + mismatches, key=lambda i: (i["code"], i["symbol"] or ""))
        status = "incomplete"
    elif mismatches:
        details = "; ".join(f"{m['code']}:{m['symbol']}:{m['detail']}" for m in mismatches)
        raise DataIntegrityError(f"tax reconciliation failed: {details}")
    else:
        status = "reconciled"
    coverage_dict: dict[str, Any] | None = {"genesis_at": coverage.genesis_at, "income_covered_from": coverage.income_covered_from, "collected_through": coverage.collected_through, "income_gaps": list(coverage.income_gaps)} if coverage is not None else None
    return {"status": status, "issues": issues, "coverage": coverage_dict, "per_symbol": per_detail}


def _recon_inputs(ordered: Sequence[TaxRecord], realizations: Mapping[str, tuple[FillRealization, ...]], record_by_id: Mapping[str, TaxRecord], period_start: pd.Timestamp, period_end: pd.Timestamp) -> tuple[set[str], dict[str, Decimal], dict[str, Decimal], dict[str, Decimal], dict[str, dict[str, Decimal]], dict[str, dict[str, Decimal]], dict[str, int], dict[str, dict[str, int]]]:
    fold_sums: dict[str, Decimal] = {}
    venue_sums: dict[str, Decimal] = {}
    income_sums: dict[str, Decimal] = {}
    fee_by_asset: dict[str, dict[str, Decimal]] = {}
    comm_by_asset: dict[str, dict[str, Decimal]] = {}
    n_close: dict[str, int] = {}
    fills_with_fee_asset: dict[str, dict[str, int]] = {}
    symbols: set[str] = set()
    for record in ordered:
        if not (period_start <= record.event_time < period_end):
            continue
        if record.kind == "TRADE":
            symbols.add(record.symbol)
            venue_sums[record.symbol] = venue_sums.get(record.symbol, Decimal(0)) + record.realized_pnl
            asset_map = fee_by_asset.setdefault(record.symbol, {})
            asset_map[record.fee_asset] = asset_map.get(record.fee_asset, Decimal(0)) + record.fee
            count_map = fills_with_fee_asset.setdefault(record.symbol, {})
            count_map[record.fee_asset] = count_map.get(record.fee_asset, 0) + 1
        elif record.kind in ("REALIZED_PNL", "COMMISSION"):
            symbols.add(record.symbol)
            if record.kind == "COMMISSION":
                asset_map = comm_by_asset.setdefault(record.symbol, {})
                asset_map[record.income_asset] = asset_map.get(record.income_asset, Decimal(0)) + record.realized_pnl
    for record in ordered:
        if period_start <= record.event_time < period_end and record.kind == "REALIZED_PNL" and not _is_delivery(record):
            income_sums[record.symbol] = income_sums.get(record.symbol, Decimal(0)) + record.realized_pnl
    for symbol, steps in realizations.items():
        total = Decimal(0)
        count = 0
        for step in steps:
            if not (period_start <= step.event_time < period_end):
                continue
            linked_step = record_by_id.get(step.record_id)
            if linked_step is None or linked_step.kind != "TRADE":
                continue
            total += step.realized_pnl
            if step.closed_quantity > 0 or linked_step.realized_pnl != 0:
                count += 1
        if symbol in symbols:
            fold_sums[symbol] = total
            n_close[symbol] = count
    return symbols, fold_sums, venue_sums, income_sums, fee_by_asset, comm_by_asset, n_close, fills_with_fee_asset


def _recon_mismatches(symbols: set[str], fold_sums: Mapping[str, Decimal], venue_sums: Mapping[str, Decimal], income_sums: Mapping[str, Decimal], fee_by_asset: Mapping[str, Mapping[str, Decimal]], comm_by_asset: Mapping[str, Mapping[str, Decimal]], n_close: Mapping[str, int], fills_with_fee_asset: Mapping[str, Mapping[str, int]], config: TaxSummaryConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        n = n_close.get(symbol, 0)
        fold_total = fold_sums.get(symbol, Decimal(0))
        venue_total = venue_sums.get(symbol, Decimal(0))
        income_total = income_sums.get(symbol, Decimal(0))
        if not _tolerance(config, n, fold_total - venue_total):
            out.append({"code": "fold_pnl_mismatch", "symbol": symbol, "detail": f"{symbol} fold={fold_total} venue={venue_total}"})
        if not _tolerance(config, n, income_total - venue_total):
            out.append({"code": "income_pnl_mismatch", "symbol": symbol, "detail": f"{symbol} income={income_total} venue={venue_total}"})
        for asset in sorted(set(fee_by_asset.get(symbol, {})) | set(comm_by_asset.get(symbol, {}))):
            combined = fee_by_asset.get(symbol, {}).get(asset, Decimal(0)) + comm_by_asset.get(symbol, {}).get(asset, Decimal(0))
            if not _tolerance(config, fills_with_fee_asset.get(symbol, {}).get(asset, 0), combined):
                out.append({"code": "commission_mismatch", "symbol": symbol, "detail": f"{symbol}:{asset} diff={combined}"})
    return sorted(out, key=lambda i: (i["code"], i["symbol"] or ""))


def summarize_tax_year(
    year: int,
    ledger_dir: Path | str,
    *,
    source: Literal["venue", "simulated"],
    config: TaxSummaryConfig,
    cost_basis: str = COST_BASIS_MOVING_AVERAGE,
    coverage: TaxCoverage | None = None,
    boundary_marks: Mapping[str, BoundaryMark] | None = None,
    derive_boundary_marks: Callable[[frozenset[str], pd.Timestamp], Mapping[str, BoundaryMark]] | None = None,
) -> dict[str, Any]:
    """Load ``ledger_dir`` with ``read_tax_ledger`` and return ``build_tax_year_summary``.

    ``ledger_dir`` and ``source`` are required: callers resolve them from LiveSettings so a summary
    can never silently read another run's (or an empty default) directory.

    When ``derive_boundary_marks`` is given, the summary is built once without marks, the
    callable is asked for the marks of ``pre_regime_inventory_symbols`` at
    ``regime_boundary_utc(config)``, and the summary is rebuilt with them; it is not called when
    no pre-regime inventory exists.

    Raises:
        TaxLedgerCorruptError: the ledger fails validation.
        ValueError: both ``boundary_marks`` and ``derive_boundary_marks`` given; the derived
            mapping names a symbol outside the requested set; or as ``build_tax_year_summary``.
        DataIntegrityError: as ``build_tax_year_summary``.
    """
    if boundary_marks is not None and derive_boundary_marks is not None:
        raise ValueError("pass boundary_marks or derive_boundary_marks, not both")
    records = read_tax_ledger(ledger_dir)
    if derive_boundary_marks is None:
        return build_tax_year_summary(
            records, year, source=source, config=config, cost_basis=cost_basis,
            coverage=coverage, boundary_marks=boundary_marks,
        )
    first = build_tax_year_summary(
        records, year, source=source, config=config, cost_basis=cost_basis,
        coverage=coverage, boundary_marks=None,
    )
    needed = pre_regime_inventory_symbols(first)
    if not needed:
        return first
    derived = derive_boundary_marks(needed, regime_boundary_utc(config))
    if set(derived) != set(needed):
        raise ValueError(f"derived marks must name exactly {sorted(needed)}")
    return build_tax_year_summary(
        records, year, source=source, config=config, cost_basis=cost_basis,
        coverage=coverage, boundary_marks=dict(derived),
    )

def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, datetime):
        raise TypeError("summary timestamps must be tz-aware pd.Timestamp values")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")

def tax_summary_to_json(summary: Mapping[str, Any]) -> str:
    """Serialize a summary losslessly: Decimal -> fixed-point string (``format(value, "f")``), pd.Timestamp -> ISO-8601 UTC, date -> ISO date; keys sorted; non-ASCII kept.

    Raises:
        TypeError: a value of any other non-JSON type (never stringified implicitly).
    """
    return json.dumps(_to_jsonable(dict(summary)), sort_keys=True, ensure_ascii=False)

def write_tax_summary(summary: Mapping[str, Any], path: Path) -> Path:
    """Write ``tax_summary_to_json(summary)`` crash-safely via ``durable_write_text`` (temp file fsynced, os.replace, parent directory fsynced) and return ``path``. Parent directories are created."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    durable_write_text(target, tax_summary_to_json(summary))
    return target
