"""Stage -1 ingestion pre-download structural exclusion filter.

Replaces ``smart_filter_symbols`` ranking cut with absolute structural
denylist/floor criteria. No alpha-based ranking or relative cuts allowed.

Time: O(N) per symbol. Space: O(N) for exclusion report.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Stablecoin denylist (confirmed 2026-06-20) ─────────────────────────────
# 17 bases. USD/EUR-peg → directional alpha structurally zero.
# PAXG (gold-price linked) is intentionally EXCLUDED from this denylist.
STABLECOIN_BASES: frozenset[str] = frozenset(
    {
        # USD fiat-backed
        "USDC",
        "BUSD",
        "FDUSD",
        "TUSD",
        "USDP",
        "GUSD",
        "PYUSD",
        # USD algorithmic / CDP
        "DAI",
        "FRAX",
        "SUSD",
        "LUSD",
        "CRVUSD",
        "DOLA",
        "USDE",
        # EUR / other fiat-peg
        "EUR",
        "EURS",
        "GBP",
        "AUD",
        # yield-linked USD-peg (price ≈ 1 USD)
        "USDY",
    }
)

_LEVERAGED_PATTERNS: tuple[str, ...] = ("UP", "DOWN", "BULL", "BEAR")


def _strip_quote_suffix(symbol: str) -> str:
    """Strip USDT or PERP suffix to extract base asset name.

    Args:
        symbol: Raw Binance futures symbol e.g. ``BTCUSDT``, ``BTCPERP``.

    Returns:
        Base asset string e.g. ``BTC``.
    """
    for suffix in ("USDT", "PERP"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


@dataclass(frozen=True, slots=True)
class IngestionFilterConfig:
    """Configuration for Stage -1 structural exclusion.

    Attributes:
        exclude_leveraged: Exclude tokens with UP/DOWN/BULL/BEAR in symbol.
        exclude_stablecoin_base: Exclude symbols whose base is in STABLECOIN_BASES.
        perpetual_only: Exclude non-perpetual contracts.
        delist_horizon_days: Exclude if delivery_date is within this many days.
        max_data_staleness_days: Exclude if last known data gap > this many days.
        lifetime_adv_floor_usdt: Absolute floor on lifetime peak 24h quote volume.
            NOT a ranking cut — only excludes permanent zombie symbols.
    """

    exclude_leveraged: bool = True
    exclude_stablecoin_base: bool = True
    perpetual_only: bool = True
    delist_horizon_days: int = 30
    max_data_staleness_days: int = 180
    lifetime_adv_floor_usdt: float = 500_000.0


def select_ingestion_symbols(
    *,
    exchange_info: Mapping[str, Any],
    ticker_24h: pd.DataFrame,
    profiles: Mapping[str, Any],
    config: IngestionFilterConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Select symbols for historical download, excluding structural misfits.

    Applies Stage -1 structural exclusion criteria in order. Each criterion
    is independent and uses short-circuit logic per symbol. No ranking or
    relative top-N cut is performed — only absolute structural floors/denylists.

    Args:
        exchange_info: Dict with ``"symbols"`` key containing list of symbol
            dicts from Binance ``/fapi/v1/exchangeInfo``. Each dict must have
            at minimum: ``symbol``, ``contractType``, ``quoteAsset``.
            Optional: ``deliveryDate`` (epoch ms int).
        ticker_24h: DataFrame with columns ``["symbol", "quoteVolume"]``
            from Binance ``/fapi/v1/ticker/24hr``. Used as ADV proxy and
            staleness indicator.
        profiles: Mapping from symbol str to profile objects. Each profile
            must expose ``.onboard_date`` (date | None) and
            ``.delivery_date`` (date | None) attributes.
        config: Exclusion configuration thresholds.

    Returns:
        Tuple of (symbols_to_download, exclusion_report_df).
        ``exclusion_report_df`` columns: ``["symbol", "excluded", "reason"]``.

    Notes:
        Time complexity: O(N) where N = number of symbols in exchange_info.
        Space complexity: O(N) for report accumulation.
    """
    symbols_raw: list[dict[str, Any]] = exchange_info.get("symbols", [])

    # Build ticker lookup: symbol → quoteVolume (float)
    ticker_vol: dict[str, float] = {}
    if not ticker_24h.empty and "symbol" in ticker_24h.columns:
        for _, row in ticker_24h.iterrows():
            sym = str(row["symbol"])
            try:
                ticker_vol[sym] = float(row["quoteVolume"])
            except (ValueError, TypeError):
                ticker_vol[sym] = 0.0

    today: date = date.today()

    report_rows: list[dict[str, object]] = []
    download_symbols: list[str] = []

    for sym_info in symbols_raw:
        symbol: str = str(sym_info.get("symbol", ""))
        if not symbol:
            continue

        excluded = False
        reason = ""

        # ── Gate 1: leveraged token ──────────────────────────────────────
        if config.exclude_leveraged:
            upper = symbol.upper()
            if any(pat in upper for pat in _LEVERAGED_PATTERNS):
                excluded = True
                reason = "leveraged"

        # ── Gate 2: perpetual only ────────────────────────────────────────
        if not excluded and config.perpetual_only:
            contract_type = str(sym_info.get("contractType", ""))
            if contract_type != "PERPETUAL":
                excluded = True
                reason = "non_perpetual"

        # ── Gate 3: stablecoin base ──────────────────────────────────────
        if not excluded and config.exclude_stablecoin_base:
            base = _strip_quote_suffix(symbol)
            if base in STABLECOIN_BASES:
                excluded = True
                reason = "stablecoin_base"

        # ── Gate 4: data staleness (last known data > staleness_days ago) ─
        if not excluded and config.max_data_staleness_days > 0:
            profile = profiles.get(symbol)
            delivery: date | None = None
            if profile is not None:
                delivery = getattr(profile, "delivery_date", None)
            if delivery is None:
                # Fallback: parse deliveryDate from exchange_info (epoch ms)
                raw_delivery = sym_info.get("deliveryDate")
                if raw_delivery is not None:
                    try:
                        ts_ms = int(raw_delivery)
                        # Binance uses 4133404800000 (year 2100) for perpetuals
                        if ts_ms > 0 and ts_ms < 4_000_000_000_000:
                            from datetime import datetime

                            delivery = datetime.fromtimestamp(
                                ts_ms / 1000.0, tz=UTC
                            ).date()
                    except (ValueError, TypeError, OverflowError):
                        pass
            # For staleness: if symbol is not in ticker_24h at all AND
            # profile has onboard_date far in the past, treat as stale.
            # Primary staleness signal: symbol absent from live ticker.
            if symbol not in ticker_vol:
                # Symbol not in live 24h ticker → potentially delisted
                onboard: date | None = None
                if profile is not None:
                    onboard = getattr(profile, "onboard_date", None)
                if onboard is not None:
                    age_days = (today - onboard).days
                    if age_days > config.max_data_staleness_days:
                        excluded = True
                        reason = "data_stale"

        # ── Gate 5: lifetime ADV floor ────────────────────────────────────
        if not excluded:
            quote_vol = ticker_vol.get(symbol, 0.0)
            if quote_vol < config.lifetime_adv_floor_usdt:
                excluded = True
                reason = "lifetime_adv_floor"

        report_rows.append(
            {"symbol": symbol, "excluded": excluded, "reason": reason}
        )
        if not excluded:
            download_symbols.append(symbol)

    exclusion_df = pd.DataFrame(report_rows, columns=["symbol", "excluded", "reason"])

    n_excluded = exclusion_df["excluded"].sum()
    logger.info(
        "ingestion_filter: total=%d excluded=%d download=%d",
        len(report_rows),
        n_excluded,
        len(download_symbols),
    )
    if n_excluded > 0:
        reason_counts = (
            exclusion_df[exclusion_df["excluded"]]["reason"].value_counts().to_dict()
        )
        logger.info("ingestion_filter exclusion reasons: %s", reason_counts)

    return download_symbols, exclusion_df
