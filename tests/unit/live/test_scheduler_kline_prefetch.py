# ruff: noqa
"""Execution parity part 3: kline prefetch posted before the release."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import src.live.scheduler as sched
from src.live.runner import CycleReport
from src.live.scheduler import DECISION_RELEASE_OFFSET, run_daemon
from src.live.settings import LiveSettings

TARGET = pd.Timestamp("2026-08-24 00:00Z")


def _report(decision_time: pd.Timestamp = TARGET) -> CycleReport:
    return CycleReport(status="COMPLETE", reason=None, decision_time=decision_time, intent_count=0)


def _install_daemon_stubs(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001, ANN202
    monkeypatch.setattr(sched, "_resolve_heartbeat_path", lambda s: tmp_path / "hb.json")
    monkeypatch.setattr(sched, "run_shadow_cycle", lambda s, d, a, now=None, **k: _report(d))
    monkeypatch.setattr(sched, "prune_old_audit_logs", lambda *a, **k: 0)


def _run_with_clock(monkeypatch, tmp_path: Path, settings: LiveSettings, start: pd.Timestamp, fns: dict) -> tuple[list, Any]:  # noqa: ANN001, ANN202
    _install_daemon_stubs(monkeypatch, tmp_path)
    cur = [start]

    def now_fn() -> pd.Timestamp:
        return cur[0]

    def sleep_fn(seconds: float) -> None:
        cur[0] += pd.Timedelta(seconds=seconds)

    (tmp_path / "w.parquet").touch()
    run_daemon(
        settings,
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=1,
        refresh_fn=fns.get("refresh", lambda *a, **k: None),
        signal_step_fn=fns.get("signal", lambda *a, **k: None),
        venue_fn=fns.get("venue", lambda *a, **k: None),
        prefetch_fn=fns.get("prefetch", lambda *a, **k: None),
        kline_prefetch_fn=fns.get("kline_prefetch"),
        prune_fn=lambda: None,
    )
    return cur, now_fn


def test_kline_prefetch_fires_once_before_release(tmp_path, monkeypatch) -> None:
    settings = LiveSettings(funding_prefetch_enabled=False, daemon_catchup_buffer_minutes=0.25)
    wait_until = TARGET + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=0.25)
    events: list[tuple[str, Any, pd.Timestamp]] = []
    holder: dict[str, Any] = {}

    def fake_kline(t: pd.Timestamp) -> None:
        events.append(("kline", t, holder["now"]()))
        return None

    def fake_refresh(t: pd.Timestamp) -> None:
        events.append(("refresh", t, holder["now"]()))
        return None

    _install_daemon_stubs(monkeypatch, tmp_path)
    cur = [wait_until - pd.Timedelta(minutes=30)]

    def now_fn() -> pd.Timestamp:
        return cur[0]

    def sleep_fn(seconds: float) -> None:
        cur[0] += pd.Timedelta(seconds=seconds)

    holder["now"] = now_fn
    (tmp_path / "w.parquet").touch()
    run_daemon(
        settings,
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=1,
        refresh_fn=fake_refresh,
        signal_step_fn=lambda *a, **k: None,
        venue_fn=lambda *a, **k: None,
        prefetch_fn=lambda *a, **k: None,
        kline_prefetch_fn=fake_kline,
        prune_fn=lambda: None,
    )
    kinds = [kind for kind, _, _ in events]
    assert kinds.count("kline") == 1
    assert kinds.index("kline") < kinds.index("refresh")
    _, kline_target, kline_at = next(e for e in events if e[0] == "kline")
    assert kline_target == TARGET
    assert wait_until - pd.Timedelta(minutes=10) <= kline_at < wait_until


def test_kline_prefetch_failure_never_blocks_cycle(tmp_path, monkeypatch, caplog) -> None:
    settings = LiveSettings(funding_prefetch_enabled=False, daemon_catchup_buffer_minutes=0.25)
    wait_until = TARGET + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=0.25)
    refresh_calls: list[pd.Timestamp] = []

    def fake_kline(t: pd.Timestamp) -> None:
        raise RuntimeError("boom")

    def fake_refresh(t: pd.Timestamp) -> None:
        refresh_calls.append(t)
        return None

    with caplog.at_level(logging.ERROR, logger="LiveScheduler"):
        _run_with_clock(
            monkeypatch, tmp_path, settings,
            wait_until - pd.Timedelta(minutes=30),
            {"refresh": fake_refresh, "kline_prefetch": fake_kline},
        )
    assert refresh_calls == [TARGET]
    assert "kline prefetch failed" in caplog.text


def test_disabled_kline_prefetch_is_never_called(tmp_path, monkeypatch) -> None:
    settings = LiveSettings(
        funding_prefetch_enabled=False,
        kline_prefetch_enabled=False,
        daemon_catchup_buffer_minutes=0.25,
    )
    wait_until = TARGET + DECISION_RELEASE_OFFSET + pd.Timedelta(minutes=0.25)
    calls: list[pd.Timestamp] = []

    def fake_kline(t: pd.Timestamp) -> None:
        calls.append(t)
        return None

    _run_with_clock(
        monkeypatch, tmp_path, settings,
        wait_until - pd.Timedelta(minutes=30),
        {"kline_prefetch": fake_kline},
    )
    assert calls == []


def test_slow_prefetch_does_not_add_stale_wait(tmp_path, monkeypatch) -> None:
    _install_daemon_stubs(monkeypatch, tmp_path)
    wait_until = TARGET + DECISION_RELEASE_OFFSET + pd.Timedelta(seconds=15)
    cur = [wait_until - pd.Timedelta(minutes=4)]
    events = []

    def prefetch(target):
        events.append(("prefetch", cur[0]))
        cur[0] += pd.Timedelta(minutes=3)

    def refresh(target):
        events.append(("refresh", cur[0]))

    run_daemon(
        LiveSettings(funding_prefetch_enabled=False, kline_prefetch_lead_minutes=2),
        tmp_path / "w.parquet", tmp_path / "state.json",
        now_fn=lambda: cur[0],
        sleep_fn=lambda seconds: cur.__setitem__(0, cur[0] + pd.Timedelta(seconds=seconds)),
        max_iterations=1, refresh_fn=refresh, kline_prefetch_fn=prefetch,
        signal_step_fn=lambda target: None, venue_fn=lambda target: None, prune_fn=lambda: None,
    )
    assert events == [("prefetch", wait_until - pd.Timedelta(minutes=2)),
                      ("refresh", wait_until + pd.Timedelta(minutes=1))]


def test_wait_ends_at_release_plus_buffer(tmp_path, monkeypatch) -> None:
    settings = LiveSettings(funding_prefetch_enabled=False, daemon_catchup_buffer_minutes=0.25)
    wait_until = TARGET + DECISION_RELEASE_OFFSET + pd.Timedelta(seconds=15)
    refresh_at: list[pd.Timestamp] = []

    def fake_refresh(t: pd.Timestamp) -> None:
        refresh_at.append(holder_now[0])
        return None

    _install_daemon_stubs(monkeypatch, tmp_path)
    holder_now = [wait_until - pd.Timedelta(seconds=60)]

    def now_fn() -> pd.Timestamp:
        return holder_now[0]

    def sleep_fn(seconds: float) -> None:
        holder_now[0] += pd.Timedelta(seconds=seconds)

    (tmp_path / "w.parquet").touch()
    run_daemon(
        settings,
        tmp_path / "w.parquet",
        tmp_path / "state.json",
        sleep_fn=sleep_fn,
        now_fn=now_fn,
        max_iterations=1,
        refresh_fn=fake_refresh,
        signal_step_fn=lambda *a, **k: None,
        venue_fn=lambda *a, **k: None,
        prefetch_fn=lambda *a, **k: None,
        kline_prefetch_fn=lambda *a, **k: None,
        prune_fn=lambda: None,
    )
    assert len(refresh_at) == 1
    assert wait_until <= refresh_at[0] <= wait_until + pd.Timedelta(seconds=sched.DAEMON_POLL_INTERVAL_SECONDS)


def test_default_kline_prefetch_returns_none_without_listing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    assert sched._default_kline_prefetch(settings, tmp_path / "w.parquet", TARGET) is None


def test_default_kline_prefetch_uses_universe_and_refresh_args(monkeypatch, tmp_path) -> None:
    from src.live.data_refresh import RefreshReport
    from src.live.venue_listing import VenueListingEntry, VenueListingSnapshot, write_venue_listing_snapshot

    listing_root = tmp_path / "data" / "state" / "venue_listing"
    listing_root.mkdir(parents=True)
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    write_venue_listing_snapshot(
        VenueListingSnapshot(
            captured_at=now,
            entries={"BTCUSDT": VenueListingEntry(symbol="BTCUSDT", status="TRADING", contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT", delivery_time=None, announced_delisting=False, delisting_first_seen_at=None)},
        ),
        listing_root, slot_day=now,
    )
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_text('{"captured_at": "2026-09-01T00:00:00+00:00", "symbols": []}', encoding="utf-8")
    (tmp_path / "ledger.json").write_text('{"positions": {}, "cash_usdt": "1000"}', encoding="utf-8")
    seen: dict = {}

    def _fake_refresh(*a, **k):
        seen.update(k)
        return RefreshReport(total=1, fresh=0, refreshed=1, failed=0, deadline_skipped=0, elapsed_s=1.5, deadline_hit=False, staleness_hours=1.0, ok=True)

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _fake_refresh)
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    report = sched._default_kline_prefetch(settings, tmp_path / "w.parquet", now)
    assert report is not None and report.refreshed == 1
    assert seen["symbols"] == ["BTCUSDT"]
    assert seen["max_workers"] == settings.refresh_max_workers == 32
    assert seen["deadline_s"] == settings.refresh_deadline_s


def test_default_kline_prefetch_tolerates_missing_non_crypto_file(monkeypatch, tmp_path) -> None:
    from src.live.data_refresh import RefreshReport
    from src.live.venue_listing import VenueListingEntry, VenueListingSnapshot, write_venue_listing_snapshot

    listing_root = tmp_path / "data" / "state" / "venue_listing"
    listing_root.mkdir(parents=True)
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    write_venue_listing_snapshot(
        VenueListingSnapshot(
            captured_at=now,
            entries={"BTCUSDT": VenueListingEntry(symbol="BTCUSDT", status="TRADING", contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT", delivery_time=None, announced_delisting=False, delisting_first_seen_at=None)},
        ),
        listing_root, slot_day=now,
    )
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "ledger.json").write_text('{"positions": {}, "cash_usdt": "1000"}', encoding="utf-8")
    seen: dict = {}

    def _fake_refresh(*a, **k):
        seen.update(k)
        return RefreshReport(total=1, fresh=1, refreshed=0, failed=0, deadline_skipped=0, elapsed_s=0.2, deadline_hit=False, staleness_hours=0.5, ok=True)

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _fake_refresh)
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    report = sched._default_kline_prefetch(settings, tmp_path / "w.parquet", now)
    assert report is not None and report.fresh == 1
    assert seen["symbols"] == ["BTCUSDT"]


def test_default_kline_prefetch_failure_returns_none(monkeypatch, tmp_path, caplog) -> None:
    from src.live.venue_listing import VenueListingEntry, VenueListingSnapshot, write_venue_listing_snapshot

    listing_root = tmp_path / "data" / "state" / "venue_listing"
    listing_root.mkdir(parents=True)
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    write_venue_listing_snapshot(
        VenueListingSnapshot(
            captured_at=now,
            entries={"BTCUSDT": VenueListingEntry(symbol="BTCUSDT", status="TRADING", contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT", delivery_time=None, announced_delisting=False, delisting_first_seen_at=None)},
        ),
        listing_root, slot_day=now,
    )
    monkeypatch.setattr(sched, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sched, "NON_CRYPTO_SYMBOLS_PATH", tmp_path / "non_crypto.json")
    (tmp_path / "non_crypto.json").write_text('{"symbols": []}', encoding="utf-8")
    (tmp_path / "ledger.json").write_text('{"positions": {}, "cash_usdt": "1000"}', encoding="utf-8")

    def _boom(*a, **k):
        raise RuntimeError("venue down")

    monkeypatch.setattr("src.live.data_refresh.refresh_live_market_data", _boom)
    settings = LiveSettings(ledger_path=str(tmp_path / "ledger.json"))
    with caplog.at_level(logging.WARNING, logger="LiveScheduler"):
        assert sched._default_kline_prefetch(settings, tmp_path / "w.parquet", now) is None
    assert "stage=kline_prefetch status=failed" in caplog.text
