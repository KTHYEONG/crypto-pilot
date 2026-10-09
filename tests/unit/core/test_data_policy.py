"""Invariant scenarios for registry-driven symbol exclusions (spec 34 part 4)."""

from __future__ import annotations

from datetime import UTC

import pandas as pd

from src.core import data_policy as data_policy_mod
from src.core.data_policy import source_gap_excluded_symbols


def _interval(symbol: str, reason: str, extent: str, start: str, end: str | None = None):
    from src.core.source_gaps import SourceGapInterval

    def _dt(value: str):
        return pd.Timestamp(value, tz="UTC").to_pydatetime().astimezone(UTC)

    verified = _dt("2026-01-01T00:00:00Z")
    return SourceGapInterval(
        symbol=symbol, plane="ohlcv_3m", start=_dt(start),
        end=_dt(end) if end is not None else None, reason=reason,
        evidence="test fixture", verified_at=verified, resolved_at=None,
        extent=extent,  # type: ignore[arg-type]
    )


def _settlement_registry(*payloads) -> object:
    import json

    from src.core.instrument_settlements import parse_instrument_settlement_registry

    raw = ("\n".join(json.dumps(p) for p in payloads) + "\n").encode() if payloads else b""
    if not payloads:
        from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY

        return EMPTY_SETTLEMENT_REGISTRY
    return parse_instrument_settlement_registry(raw, source="test")


def _luna_settlement() -> dict:
    symbol = "LUNAUSDT"
    delivery = pd.Timestamp("2022-05-12T15:33:00Z")
    announced = delivery - pd.Timedelta(days=7)
    return {
        "kind": "settlement", "symbol": symbol,
        "event_id": f"{symbol}:{int(delivery.value // 1_000_000)}",
        "announced_at": announced.isoformat().replace("+00:00", "Z"),
        "announcement_source": "proxy_lead", "announcement_evidence": "",
        "last_trade_at": delivery.isoformat().replace("+00:00", "Z"),
        "delivery_at": delivery.isoformat().replace("+00:00", "Z"),
        "settlement_price": 1.25, "price_source": "flat_1h_klines",
        "price_evidence": "flat bars", "fee_bps": 5.0,
        "evidence_digest": "sha256:" + "ab" * 32, "verified_at": "2026-07-01T00:00:00Z",
    }


def test_settled_delisting_is_not_excluded(monkeypatch) -> None:
    intervals = (_interval("LUNAUSDT", "DELISTED", "UNSCOPED", "2021-01-01T00:00:00Z"),)
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: intervals)
    monkeypatch.setattr(
        data_policy_mod, "load_instrument_settlement_registry",
        lambda *a, **k: _settlement_registry(_luna_settlement()),
    )
    assert "LUNAUSDT" not in source_gap_excluded_symbols()


def test_unsettled_delisted_still_excluded(monkeypatch) -> None:
    from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY

    intervals = (_interval("LUNAUSDT", "DELISTED", "UNSCOPED", "2021-01-01T00:00:00Z"),)
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: intervals)
    monkeypatch.setattr(
        data_policy_mod, "load_instrument_settlement_registry",
        lambda *a, **k: EMPTY_SETTLEMENT_REGISTRY,
    )
    assert "LUNAUSDT" in source_gap_excluded_symbols()


def test_pre_delivery_unscoped_still_excludes(monkeypatch) -> None:
    import json

    from src.core.instrument_settlements import parse_instrument_settlement_registry

    symbol = "BTCSTUSDT"
    delivery = pd.Timestamp("2021-03-12T02:00:00Z")
    announced = delivery - pd.Timedelta(days=7)
    payload = {
        "kind": "settlement", "symbol": symbol,
        "event_id": f"{symbol}:{int(delivery.value // 1_000_000)}",
        "announced_at": announced.isoformat().replace("+00:00", "Z"),
        "announcement_source": "proxy_lead", "announcement_evidence": "",
        "last_trade_at": delivery.isoformat().replace("+00:00", "Z"),
        "delivery_at": delivery.isoformat().replace("+00:00", "Z"),
        "settlement_price": 1.25, "price_source": "flat_1h_klines",
        "price_evidence": "flat bars", "fee_bps": 5.0,
        "evidence_digest": "sha256:" + "ab" * 32, "verified_at": "2026-07-01T00:00:00Z",
    }
    registry = parse_instrument_settlement_registry(
        (json.dumps(payload) + "\n").encode(), source="test",
    )
    intervals = (
        _interval("BTCSTUSDT", "SOURCE_ABSENT", "UNSCOPED", "2021-01-01T00:00:00Z", "2021-03-04T07:00:00Z"),
    )
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: intervals)
    monkeypatch.setattr(
        data_policy_mod, "load_instrument_settlement_registry", lambda *a, **k: registry,
    )
    assert "BTCSTUSDT" in source_gap_excluded_symbols()


def test_committed_registries_exclude_nothing_after_edge_rescope() -> None:
    excluded = source_gap_excluded_symbols()
    assert "LUNAUSDT" not in excluded
    assert excluded == frozenset()


def test_edge_scoped_symbols_are_not_excluded(monkeypatch) -> None:
    intervals = (
        _interval("EDGEUSDT", "SOURCE_ABSENT", "LISTING_EDGE", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"),
        _interval("TAILUSDT", "SOURCE_ABSENT", "OPEN_EDGE", "2022-06-01T00:00:00Z"),
    )
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: intervals)
    monkeypatch.setattr(
        data_policy_mod, "load_instrument_settlement_registry",
        lambda *a, **k: _settlement_registry(),
    )
    assert source_gap_excluded_symbols() == frozenset()


def test_delisted_superseded_by_settlement_stays_non_excluding(monkeypatch) -> None:
    intervals = (_interval("LUNAUSDT", "DELISTED", "UNSCOPED", "2021-01-01T00:00:00Z"),)
    monkeypatch.setattr(data_policy_mod, "active_intervals", lambda **kwargs: intervals)
    monkeypatch.setattr(
        data_policy_mod, "load_instrument_settlement_registry",
        lambda *a, **k: _settlement_registry(_luna_settlement()),
    )
    assert "LUNAUSDT" not in source_gap_excluded_symbols()
