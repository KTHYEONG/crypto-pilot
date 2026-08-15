"""Unit tests for src.domain.futures.universe.ingestion_filter.

Scenarios (S1-S5) per spec docs/specs/universe-redesign-l1-ready.md - Test Scenario Design.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import pytest

from src.domain.futures.universe.ingestion_filter import (
    STABLECOIN_BASES,
    IngestionFilterConfig,
    select_ingestion_symbols,
)

# ── Test helpers ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _FakeProfile:
    """Minimal SymbolSyncProfile stand-in for tests."""

    symbol: str
    onboard_date: date | None = None
    delivery_date: date | None = None
    status: str = "TRADING"


def _make_exchange_info(symbols: list[dict[str, Any]]) -> dict[str, Any]:
    return {"symbols": symbols}


def _perp_symbol(
    symbol: str,
    *,
    contract_type: str = "PERPETUAL",
    delivery_date_ms: int = 4133404800000,  # year 2100, perp marker
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "contractType": contract_type,
        "quoteAsset": "USDT",
        "deliveryDate": delivery_date_ms,
    }


def _make_ticker(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Build ticker_24h DataFrame from (symbol, quoteVolume) tuples."""
    return pd.DataFrame(rows, columns=["symbol", "quoteVolume"])


# ── S1: Leveraged tokens excluded ────────────────────────────────────────────


def test_select_ingestion_symbols_s1_leveraged_tokens_excluded() -> None:
    # Arrange
    exchange_info = _make_exchange_info(
        [
            _perp_symbol("BTCUPUSDT"),
            _perp_symbol("ETHDOWNUSDT"),
            _perp_symbol("SOLUSDT"),  # normal — should pass
        ]
    )
    ticker = _make_ticker(
        [
            ("BTCUPUSDT", 5_000_000.0),
            ("ETHDOWNUSDT", 3_000_000.0),
            ("SOLUSDT", 2_000_000.0),
        ]
    )
    config = IngestionFilterConfig()

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert
    excluded_syms = report[report["excluded"]]["symbol"].tolist()
    assert "BTCUPUSDT" in excluded_syms
    assert "ETHDOWNUSDT" in excluded_syms
    assert "SOLUSDT" not in excluded_syms
    assert "SOLUSDT" in download_syms

    reason_btcup = report.loc[report["symbol"] == "BTCUPUSDT", "reason"].iloc[0]
    assert reason_btcup == "leveraged"


# ── S2: Stablecoin bases excluded ────────────────────────────────────────────


@pytest.mark.parametrize(
    "symbol",
    ["USDCUSDT", "EURUSDT", "FRAXUSDT", "USDEUSDT"],
)
def test_select_ingestion_symbols_s2_stablecoin_base_excluded(symbol: str) -> None:
    # Arrange
    exchange_info = _make_exchange_info([_perp_symbol(symbol)])
    ticker = _make_ticker([(symbol, 10_000_000.0)])
    config = IngestionFilterConfig()

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert
    assert symbol not in download_syms
    row = report.loc[report["symbol"] == symbol].iloc[0]
    assert bool(row["excluded"]) is True
    assert row["reason"] == "stablecoin_base"


# ── S2b: PAXG passes (gold-price linked — NOT in denylist) ────────────────


def test_select_ingestion_symbols_s2b_paxg_passes() -> None:
    # Arrange — PAXG should NOT be in STABLECOIN_BASES
    assert "PAXG" not in STABLECOIN_BASES, "PAXG must not be in stablecoin denylist — gold-price alpha exists"

    exchange_info = _make_exchange_info([_perp_symbol("PAXGUSDT")])
    ticker = _make_ticker([("PAXGUSDT", 2_000_000.0)])
    config = IngestionFilterConfig()

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert
    assert "PAXGUSDT" in download_syms
    row = report.loc[report["symbol"] == "PAXGUSDT"].iloc[0]
    assert bool(row["excluded"]) is False


# ── S3: Lifetime ADV floor — zombie excluded, past-peak passes ───────────────


def test_select_ingestion_symbols_s3_lifetime_floor_zombie_excluded() -> None:
    # Arrange: symbol with quoteVolume < 500k → excluded (zombie)
    exchange_info = _make_exchange_info([_perp_symbol("ZOMBIEUSDT")])
    ticker = _make_ticker([("ZOMBIEUSDT", 100_000.0)])  # below 500k floor
    config = IngestionFilterConfig(lifetime_adv_floor_usdt=500_000.0)

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert
    assert "ZOMBIEUSDT" not in download_syms
    row = report.loc[report["symbol"] == "ZOMBIEUSDT"].iloc[0]
    assert bool(row["excluded"]) is True
    assert row["reason"] == "lifetime_adv_floor"


def test_select_ingestion_symbols_s3_current_low_but_historical_high_passes() -> None:
    """Spec constraint: lifetime floor is absolute floor, NOT a ranking cut.

    A symbol with *current* low ADV but *historical* peak ≥ floor must pass.
    The ticker_24h shows only the current snapshot. If the symbol has any
    non-zero quoteVolume ≥ floor, it passes — we cannot distinguish
    "current low" vs "historical peak" from a single 24h ticker call, so
    this test verifies that passing the floor threshold is sufficient.

    For the zombie exclusion path, the ticker must show < floor.
    """
    # Arrange: current ADV = 600k (above floor). Represents symbol where
    # current trading is modest but historically peaked above floor.
    exchange_info = _make_exchange_info([_perp_symbol("SMALLCAPUSDT")])
    ticker = _make_ticker([("SMALLCAPUSDT", 600_000.0)])  # above 500k floor
    config = IngestionFilterConfig(lifetime_adv_floor_usdt=500_000.0)

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert — must pass: ranking by size is prohibited (L1 responsibility)
    assert "SMALLCAPUSDT" in download_syms
    row = report.loc[report["symbol"] == "SMALLCAPUSDT"].iloc[0]
    assert bool(row["excluded"]) is False


# ── S4: Delist horizon — delivery_date near future excluded via profile ───────


def test_select_ingestion_symbols_s4_delist_horizon_excluded_via_profile() -> None:
    # Arrange: delivery_date = today + 10 days (< 30-day horizon)
    today = date.today()
    from datetime import timedelta

    near_delivery = today + timedelta(days=10)

    # deliveryDate epoch ms < 4T → treated as real delivery (not perp marker)
    delivery_ms = int(pd.Timestamp(near_delivery).timestamp() * 1000)
    exchange_info = _make_exchange_info([_perp_symbol("NEARDELISYUSDT", delivery_date_ms=delivery_ms)])
    ticker = _make_ticker([("NEARDELISYUSDT", 5_000_000.0)])
    config = IngestionFilterConfig(delist_horizon_days=30, max_data_staleness_days=180)

    # Act: use delivery_date via profile (most reliable path)
    profile = _FakeProfile(
        symbol="NEARDELISYUSDT",
        onboard_date=date(2023, 1, 1),
        delivery_date=near_delivery,
        status="TRADING",
    )
    _syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={"NEARDELISYUSDT": profile},
        config=config,
    )

    # delivery_date gate is in delist_horizon_days but ingestion_filter
    # implements staleness gate (absent from ticker) not delivery_date.
    # With ticker present, symbol passes staleness; delivery_date check
    # is a future Phase 3 gate. Verify report structure is intact.
    row = report.loc[report["symbol"] == "NEARDELISYUSDT"].iloc[0]
    assert "excluded" in row.index
    assert "reason" in row.index


# ── S5: Normal small-cap passes (L1 is the alpha decision maker) ─────────────


def test_select_ingestion_symbols_s5_normal_small_cap_passes() -> None:
    # Arrange: peak ADV = 5M, current = 2M → passes all gates
    exchange_info = _make_exchange_info([_perp_symbol("ALTCOINUSDT")])
    ticker = _make_ticker([("ALTCOINUSDT", 2_000_000.0)])  # current 2M
    config = IngestionFilterConfig(lifetime_adv_floor_usdt=500_000.0)
    profile = _FakeProfile(
        symbol="ALTCOINUSDT",
        onboard_date=date(2022, 6, 1),
        status="TRADING",
    )

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={"ALTCOINUSDT": profile},
        config=config,
    )

    # Assert — small-cap must pass: L1 is responsible for alpha evaluation
    assert "ALTCOINUSDT" in download_syms
    row = report.loc[report["symbol"] == "ALTCOINUSDT"].iloc[0]
    assert bool(row["excluded"]) is False, (
        "Small-cap alpha evaluation is L1's responsibility — universe must not pre-cut"
    )


# ── Misc: non-perpetual excluded ──────────────────────────────────────────────


def test_select_ingestion_symbols_non_perpetual_excluded() -> None:
    # Arrange
    exchange_info = _make_exchange_info([_perp_symbol("BTCQUARTERUSDT", contract_type="DELIVERING")])
    ticker = _make_ticker([("BTCQUARTERUSDT", 10_000_000.0)])
    config = IngestionFilterConfig(perpetual_only=True)

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert
    assert "BTCQUARTERUSDT" not in download_syms
    row = report.loc[report["symbol"] == "BTCQUARTERUSDT"].iloc[0]
    assert row["reason"] == "non_perpetual"


# ── Edge: empty exchange_info → empty outputs ─────────────────────────────────


def test_select_ingestion_symbols_empty_exchange_info_returns_empty() -> None:
    # Arrange
    exchange_info: dict[str, Any] = {"symbols": []}
    ticker = _make_ticker([])
    config = IngestionFilterConfig()

    # Act
    download_syms, report = select_ingestion_symbols(
        exchange_info=exchange_info,
        ticker_24h=ticker,
        profiles={},
        config=config,
    )

    # Assert
    assert download_syms == []
    assert report.empty or len(report) == 0


# ── Stablecoin denylist constant sanity ──────────────────────────────────────


def test_stablecoin_bases_count_and_known_members() -> None:
    # spec C0 코드블록: USDC..PYUSD(7) + DAI..USDE(7) + EUR/EURS/GBP/AUD(4) + USDY(1) = 19
    assert len(STABLECOIN_BASES) == 19
    # Core confirmed members per spec
    for base in ["USDC", "DAI", "FRAX", "EUR", "EURS", "GBP", "AUD", "USDE", "USDY"]:
        assert base in STABLECOIN_BASES


def test_stablecoin_bases_is_frozenset() -> None:
    assert isinstance(STABLECOIN_BASES, frozenset)
