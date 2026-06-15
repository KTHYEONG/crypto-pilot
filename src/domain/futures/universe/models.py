"""Universe models and data structures for futures universe selection."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.settings import FUTURES_DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = FUTURES_DATA_DIR / "universe_ledger.db"
LEVERAGED_TOKEN_PATTERNS = ("UP", "DOWN", "BULL", "BEAR")
_SQLITE_LEDGER_SUFFIXES = (".db", ".sqlite", ".sqlite3", "")
_PARQUET_LEDGER_SUFFIXES = (".parquet", ".pq")


class RejectCode(StrEnum):
    """Canonical reject reason codes across stages."""

    NOT_TRADING = "NOT_TRADING"
    NOT_LISTED = "NOT_LISTED"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"
    LOW_COVERAGE = "LOW_COVERAGE"
    TOO_MANY_ZERO_VOLUME_BARS = "TOO_MANY_ZERO_VOLUME_BARS"
    EXCESSIVE_GAPS = "EXCESSIVE_GAPS"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    HIGH_EXECUTION_COST = "HIGH_EXECUTION_COST"
    FUNDING_ANOMALY = "FUNDING_ANOMALY"
    BASIS_ANOMALY = "BASIS_ANOMALY"
    RISK_EVENT_OVERRIDE = "RISK_EVENT_OVERRIDE"
    LISTING_TOO_YOUNG = "LISTING_TOO_YOUNG"
    VOL_BAND_VIOLATION = "VOL_BAND_VIOLATION"
    RANKED_OUT = "RANKED_OUT"


class EventType(StrEnum):
    """Manual risk-event categories."""

    SCHEDULED_UNLOCK = "SCHEDULED_UNLOCK"
    EXCHANGE_HALT = "EXCHANGE_HALT"
    REGULATORY = "REGULATORY"
    SECURITY_INCIDENT = "SECURITY_INCIDENT"


@dataclass(frozen=True, slots=True)
class LedgerRow:
    """Daily append-only universe ledger row."""

    symbol: str
    date: str
    knowledge_date: str
    is_listed: bool
    is_trading: bool
    status: str
    first_kline_date: str
    delist_date: str | None = None
    delist_announcement: str | None = None
    adv_usdt_median: float = 0.0
    adv_usdt_mean: float = 0.0
    has_kline: bool = False
    has_funding: bool = False
    n_bar_gaps: int = 0
    max_gap_bars: int = 0
    frozen_bars: int = 0
    last_60d_coverage: float = 0.0
    n_zero_volume_bars_60d: int = 0
    funding_rate_8h: float = 0.0
    listing_age_days: int = 0
    vol_30d: float = 0.0
    risk_event_override: str | None = None
    updated_at_utc: str = ""
    # New fields for pipeline compatibility
    tf: str = "4h"
    contract_type: str = "PERPETUAL"
    quote_asset: str = "USDT"
    margin_asset: str = "USDT"
    is_coverage: bool = True
    n_is_bars: int = 0
    expected_is_bars: int = 0
    amihud_30d: float = 0.0
    screening_clip_usdt: float = 10000.0
    taker_fee_bps: float = 5.0
    half_spread_bps: float = 1.0
    impact_bps: float = 0.0
    tick_cost_bps: float = 0.0
    tick_size: float = 0.0
    mark_price: float = 0.0
    funding_zscore: float = 0.0
    contract_multiplier: float = 1.0
    has_nan: bool = False
    has_inf: bool = False
    has_timestamp_issues: bool = False


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Input data lock record for reproducibility."""

    symbol: str
    period: str
    source: str
    sha256: str
    is_final: bool
    updated_at_utc: str
    tf: str = ""
    url: str = ""
    bytes: int = 0
    fetched_at_utc: str = ""


@dataclass(frozen=True, slots=True)
class ManualEventRow:
    """Manual risk-event input with PIT-safe knowledge-date."""

    symbol: str
    event_type: EventType
    event_date: str
    knowledge_date: str
    severity: str
    action: str
    source_url: str
    recorded_at_utc: str


@dataclass(frozen=True, slots=True)
class SymbolMeta:
    """Per-symbol metadata carried by a universe snapshot."""

    symbol: str
    role: str
    adv_usdt: float
    execution_cost_bps: float
    funding_carry_8h: float
    beta_vs_market: float
    cluster_id: int
    tradeable_rank: int
    basis_annualized_mean: float | None
    basis_vol: float | None
    capacity_clip_usdt_list: tuple[float, ...]
    cluster_size: float = 1.0
    anchor_cluster_member: float = 0.0
    vol_30d: float = 0.0
    friction_score: float = 0.0
    alpha_capacity_score: float = 0.0
    diversification_score: float = 0.0
    tradeable_score: float = 0.0


@dataclass(frozen=True, slots=True)
class UniverseRunManifest:
    """Persistent manifest for one deterministic universe build."""

    as_of: str
    tf: str
    schema_version: int
    run_id: str
    config_hash: str
    data_manifest_hash: str
    generated_at_utc: str
    ledger_confidence: str
    basket_ref: tuple[str, ...]
    basket_weights: tuple[float, ...]
    n_stage0: int
    n_stage1_pass: int
    n_stage2_pass: int
    n_stage3_pass: int
    n_stage4_pass: int
    n_stage5_pass: int
    n_stage6_selected: int


@dataclass(frozen=True, slots=True)
class FilterReport:
    """Audit report for stage-level pass/fail reasons and metrics."""

    symbol: str
    stage0_pass: bool
    stage1_reason: RejectCode | None
    stage1_metrics: dict[str, float]
    stage2_reason: RejectCode | None
    stage2_metrics: dict[str, float]
    stage3_reason: RejectCode | None
    stage3_metrics: dict[str, float]
    stage4_reason: RejectCode | None
    stage4_metrics: dict[str, float]
    stage5_reason: RejectCode | None
    stage5_metrics: dict[str, float]
    stage6_reason: RejectCode | None
    stage6_metrics: dict[str, float]
    final_rank: int | None
    final_cluster_id: int | None
    audit_trail: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    """Frozen snapshot for replayable universe membership."""

    as_of: str
    tf: str
    schema_version: int
    config_hash: str
    data_manifest_hash: str
    basket_ref: tuple[str, ...]
    basket_weights: tuple[float, ...]
    selected: tuple[SymbolMeta, ...]
    rejected: dict[str, FilterReport]
    generated_at_utc: str
    ledger_confidence: str
    n_stage0: int
    n_stage1_pass: int
    n_stage2_pass: int
    n_stage3_pass: int
    n_stage4_pass: int
    n_stage5_pass: int
    n_stage6_selected: int
    # Current-quarter Stage6 selected symbols for candidate ML training.
    training_panel: tuple[str, ...] = field(default_factory=tuple)
    # Historical quarterly Stage6 union for candidate ML loading.
    inference_panel: tuple[str, ...] = field(default_factory=tuple)
    # Current-quarter Stage6 selected symbols for candidate ML inference.
    live_inference_panel: tuple[str, ...] = field(default_factory=tuple)
    # Historical quarterly Stage6 union for trading membership.
    historical_trading_panel: tuple[str, ...] = field(default_factory=tuple)
    # SSOT: quarterly Stage6 members for candidate ML membership masks.
    # dict key = quarter_start date (isoformat str로 직렬화), value = tuple of symbols (sorted)
    inference_panel_quarter_membership: dict[date, tuple[str, ...]] = field(
        default_factory=dict
    )
    # Current-quarter Stage5 survivors retained for audit and research only.
    stage5_research_panel: tuple[str, ...] = field(default_factory=tuple)


def _to_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def query_ledger_as_of(
    ledger: pd.DataFrame,
    *,
    as_of: str | date,
    tf: str,
    symbols: Iterable[str] | None = None,
    enforce_eligibility: bool = True,
) -> pd.DataFrame:
    """Filter ledger rows with PIT-safe constraints.

    Args:
        ledger: Source ledger rows.
        as_of: Evaluation date.
        tf: Timeframe (e.g., 4h, 1h).
        symbols: Optional symbol whitelist.
        enforce_eligibility: Enforce Stage0 eligibility ``is_listed & is_trading``.

    Returns:
        PIT-filtered rows where ``knowledge_date <= as_of`` and ``date <= as_of``.

    """
    if ledger.empty:
        return ledger.copy()

    as_of_date = _to_date(as_of)
    out = ledger.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.date
    out["knowledge_date"] = pd.to_datetime(
        out["knowledge_date"], utc=True, errors="coerce"
    ).dt.date
    mask = (out["tf"] == tf) & (out["date"] <= as_of_date) & (out["knowledge_date"] <= as_of_date)
    if symbols is not None:
        symbol_set = set(symbols)
        mask &= out["symbol"].isin(symbol_set)
    if enforce_eligibility:
        is_listed = (
            out.get("is_listed", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        )
        is_trading = (
            out.get("is_trading", pd.Series(True, index=out.index)).fillna(False).astype(bool)
        )
        mask &= is_listed & is_trading
    out = out.loc[mask]
    out = out.sort_values(["symbol", "date", "knowledge_date"])
    return out


def load_ledger_slice(
    *,
    as_of: str | date,
    tf: str,
    columns: tuple[str, ...],
    symbols: tuple[str, ...] | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    enforce_eligibility: bool = True,
) -> pd.DataFrame:
    """Load PIT-safe ledger slice from SQLite or parquet storage."""
    needed = set(columns) | {"symbol", "tf", "date", "knowledge_date"}
    if enforce_eligibility:
        needed |= {"is_listed", "is_trading"}

    if not ledger_path.exists():
        return pd.DataFrame(columns=sorted(needed))

    suffix = ledger_path.suffix.lower()
    if suffix in _PARQUET_LEDGER_SUFFIXES:
        df = _load_parquet_ledger_slice(
            as_of=as_of,
            tf=tf,
            needed=needed,
            symbols=symbols,
            ledger_path=ledger_path,
            enforce_eligibility=enforce_eligibility,
        )
    elif suffix in _SQLITE_LEDGER_SUFFIXES:
        df = _load_sqlite_ledger_slice(
            as_of=as_of,
            tf=tf,
            needed=needed,
            symbols=symbols,
            ledger_path=ledger_path,
            enforce_eligibility=enforce_eligibility,
        )
    else:
        raise ValueError(f"unsupported ledger backend: {ledger_path.suffix or '<empty>'}")

    return query_ledger_as_of(
        df,
        as_of=as_of,
        tf=tf,
        symbols=symbols,
        enforce_eligibility=enforce_eligibility,
    )


def _load_sqlite_ledger_slice(
    *,
    as_of: str | date,
    tf: str,
    needed: set[str],
    symbols: tuple[str, ...] | None,
    ledger_path: Path,
    enforce_eligibility: bool,
) -> pd.DataFrame:
    as_of_date = _to_date(as_of)
    as_of_str = as_of_date.isoformat()
    cols_str = ", ".join([f'"{col}"' for col in sorted(needed)])
    query = f"SELECT {cols_str} FROM ledger WHERE tf = ? AND date <= ? AND knowledge_date <= ?"  # noqa: S608
    params: list[str] = [tf, as_of_str, as_of_str]
    if symbols is not None:
        placeholders = ", ".join(["?"] * len(symbols))
        query += f" AND symbol IN ({placeholders})"
        params.extend(symbols)
    if enforce_eligibility:
        query += " AND is_listed = 1 AND is_trading = 1"
    logger.info(
        "[SQL-DB]   🔑 Loading ledger slice from %s (as_of=%s, tf=%s)",
        ledger_path.name,
        as_of_str,
        tf,
    )
    try:
        conn = sqlite3.connect(str(ledger_path))
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    except (sqlite3.DatabaseError, pd.errors.DatabaseError, ValueError) as exc:
        raise ValueError(
            f"Failed to load sqlite ledger slice from {ledger_path}: {exc}"
        ) from exc


def _load_parquet_ledger_slice(
    *,
    as_of: str | date,
    tf: str,
    needed: set[str],
    symbols: tuple[str, ...] | None,
    ledger_path: Path,
    enforce_eligibility: bool,
) -> pd.DataFrame:
    try:
        logger.info(
            "[PARQUET] Loading ledger slice from %s (as_of=%s, tf=%s)",
            ledger_path.name,
            _to_date(as_of).isoformat(),
            tf,
        )
        frame = pd.read_parquet(ledger_path)
    except Exception as exc:
        raise ValueError(f"Failed to load parquet ledger slice from {ledger_path}: {exc}") from exc

    missing = needed.difference(frame.columns)
    synth_allowed = {"is_listed", "is_trading"} if enforce_eligibility else set()
    extra_missing = missing.difference(synth_allowed)
    if extra_missing:
        raise ValueError(
            f"Parquet ledger missing required columns: {sorted(extra_missing)} from {ledger_path}"
        )
    for column in synth_allowed.intersection(missing):
        frame[column] = True

    return frame


def update_ledger(new_rows: pd.DataFrame, *, ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append new rows to ledger storage in an idempotent way using SQLite."""
    if new_rows.empty:
        return
    import sqlite3
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[SQL-DB]   💾 Updating ledger: {len(new_rows)} rows -> {ledger_path.name}")
    conn = sqlite3.connect(str(ledger_path))
    try:
        new_rows.to_sql("temp_ledger", conn, if_exists="replace", index=False)
        
        cursor = conn.cursor()
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ledger'")
        if cursor.fetchone()[0] == 0:
            cursor.execute("CREATE TABLE ledger AS SELECT * FROM temp_ledger WHERE 1=0")
            cursor.execute(
                "CREATE UNIQUE INDEX idx_ledger ON ledger (symbol, tf, date, knowledge_date)"
            )
            
        cols = ", ".join([f'"{col}"' for col in new_rows.columns])
        cursor.execute(f"INSERT OR REPLACE INTO ledger ({cols}) SELECT {cols} FROM temp_ledger")  # noqa: S608
        cursor.execute("DROP TABLE temp_ledger")
        conn.commit()
    finally:
        conn.close()


def apply_structure_stage(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply structure filters in vectorized form."""
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=["symbol", "stage", "passed", "reason"])

    symbol = frame["symbol"].astype("string")
    upper = symbol.str.upper()
    is_perp = frame.get("contract_type", pd.Series("", index=frame.index)).eq("PERPETUAL")
    is_usdt_quote = frame.get("quote_asset", pd.Series("", index=frame.index)).eq("USDT")
    is_usdt_margin = frame.get("margin_asset", pd.Series("", index=frame.index)).eq("USDT")
    is_trading = frame.get("status", pd.Series("", index=frame.index)).eq("TRADING")
    multiplier = pd.to_numeric(
        frame.get("contract_multiplier", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    multiplier_valid = multiplier.notna() & np.isfinite(multiplier) & (multiplier > 0.0)
    leveraged = np.logical_or.reduce(
        [upper.str.contains(p, regex=False) for p in LEVERAGED_TOKEN_PATTERNS]
    )

    usdt_quote_or_margin = is_usdt_quote | is_usdt_margin
    pass_mask = is_perp & usdt_quote_or_margin & is_trading & multiplier_valid & (~leveraged)
    reasons = np.where(~is_perp, "not_perpetual", "")
    reasons = np.where(
        (reasons == "") & (~usdt_quote_or_margin),
        "not_usdt_quote_or_margin",
        reasons,
    )
    reasons = np.where((reasons == "") & (~is_trading), "not_trading", reasons)
    reasons = np.where(
        (reasons == "") & (~multiplier_valid),
        "invalid_contract_multiplier",
        reasons,
    )
    reasons = np.where((reasons == "") & leveraged, "leveraged_token_pattern", reasons)
    reasons = pd.Series(np.where(reasons == "", "pass", reasons), index=frame.index, dtype="string")

    report = pd.DataFrame(
        {
            "symbol": symbol,
            "stage": "stage1_structure",
            "passed": pass_mask.astype(bool),
            "reason": reasons,
        }
    )
    return frame.loc[pass_mask].copy(), report


def _to_utc_date_str(value: Any) -> str | None:
    """Convert timestamp-like input into YYYY-MM-DD in UTC."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        ts = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
        return ts.date().isoformat()
    if isinstance(value, str) and value:
        return value[:10]
    return None


def normalize_exchange_info(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize exchangeInfo symbols into vectorizable metadata frame.

    Args:
        records: Raw symbol records from exchange info.

    Returns:
        DataFrame with normalized columns required by Stage 1/5.

    """
    if not records:
        return pd.DataFrame(
            columns=[
                "symbol",
                "pair",
                "contract_type",
                "status",
                "quote_asset",
                "margin_asset",
                "onboard_date",
                "delivery_date",
            ]
        )

    frame = pd.DataFrame.from_records(records)
    required = {
        "symbol": "",
        "pair": "",
        "contractType": "",
        "status": "",
        "quoteAsset": "",
        "marginAsset": "",
        "onboardDate": None,
        "deliveryDate": None,
    }
    for col, default in required.items():
        if col not in frame.columns:
            frame[col] = default

    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype("string"),
            "pair": frame["pair"].astype("string"),
            "contract_type": frame["contractType"].astype("string"),
            "status": frame["status"].astype("string"),
            "quote_asset": frame["quoteAsset"].astype("string"),
            "margin_asset": frame["marginAsset"].astype("string"),
            "onboard_date": frame["onboardDate"].map(_to_utc_date_str).astype("string"),
            "delivery_date": frame["deliveryDate"].map(_to_utc_date_str).astype("string"),
            "is_listed": frame["onboardDate"].notna(),
            "is_trading": frame["status"].astype("string").eq("TRADING"),
        }
    )
    return out
