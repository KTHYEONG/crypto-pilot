# ruff: noqa
"""Execution parity part 3: execution-latency audit after the first post."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
RELEASE_AT = DECISION_TIME + pd.Timedelta(hours=23)


def _seed_weights(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    closes = pd.DataFrame(
        100.0, index=pd.DatetimeIndex(frame.index), columns=list(frame.columns), dtype="float64",
    )
    from src.live.deployed_weights import decision_ohlcv_close_path

    closes.to_parquet(decision_ohlcv_close_path(path), index=True)
    return path


def _install_stubs(tmp_path: Path, monkeypatch, *, post: bool) -> dict:  # noqa: ANN001, ANN202
    import src.live.orderbook as ob_mod
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from tests.unit.live._runner_stubs import StubMarketClient, StubOrderClient

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / "shadow_cycle.jsonl")
    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        if post and intents:
            audit.record(
                "order_posted", symbol=intents[0].symbol, client_order_id="lat1",
                time_in_force="GTX", price="100", quantity="1",
                bid="99", ask="101", phase="passive",
            )
        return tuple(
            ExecutionOutcome(
                symbol=intent.symbol, filled_qty=intent.quantity,
                unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"),
                chases=0, status="FILLED",
            )
            for intent in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)
    return {}


def _latency_events(tmp_path: Path) -> list[dict]:
    audit_path = tmp_path / "shadow_cycle.jsonl"
    return [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "execution_latency"
    ]


def test_latency_audit_after_first_post(tmp_path, monkeypatch, caplog) -> None:
    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings

    weights_path = _seed_weights(tmp_path)
    _install_stubs(tmp_path, monkeypatch, post=True)
    settings = LiveSettings(
        ledger_path=str(tmp_path / "ledger.json"),
        order_journal_path=str(tmp_path / "journal.jsonl"),
        exec_depth_capture_enabled=False,
    )
    now = RELEASE_AT + pd.Timedelta(seconds=42)
    monkeypatch.setattr(runner_mod, "_clock", lambda: now.timestamp())
    token = runner_mod.cycle_stage_durations.set((12.0, 23.0))
    with caplog.at_level(logging.INFO, logger="LiveRunner"):
        report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=now)
    runner_mod.cycle_stage_durations.reset(token)
    assert report.status == "COMPLETE"
    events = _latency_events(tmp_path)
    assert len(events) == 1
    row = events[0]
    assert row["seconds_after_release"] == 42
    assert pd.Timestamp(row["release_at"]) == RELEASE_AT
    assert abs((pd.Timestamp(row["first_post_at"]) - now).total_seconds()) < 5.0
    for field in ("refresh_s", "signal_s", "depth_wait_s"):
        assert field in row and row[field] >= 0
    assert (row["refresh_s"], row["signal_s"]) == (12.0, 23.0)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "execution_latency" in r.getMessage()]


def test_slow_cycle_warns_with_breakdown(tmp_path, monkeypatch, caplog) -> None:
    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings

    weights_path = _seed_weights(tmp_path)
    _install_stubs(tmp_path, monkeypatch, post=True)
    monkeypatch.setattr(runner_mod, "_clock", lambda: (RELEASE_AT + pd.Timedelta(seconds=75)).timestamp())
    settings = LiveSettings(
        ledger_path=str(tmp_path / "ledger.json"),
        order_journal_path=str(tmp_path / "journal.jsonl"),
        exec_depth_capture_enabled=False,
    )
    with caplog.at_level(logging.INFO, logger="LiveRunner"):
        report = runner_mod.run_shadow_cycle(
            settings, DECISION_TIME, weights_path, now=RELEASE_AT + pd.Timedelta(seconds=75),
        )
    assert report.status == "COMPLETE"
    events = _latency_events(tmp_path)
    assert len(events) == 1
    assert events[0]["seconds_after_release"] == 75
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "execution_latency" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "75" in message and "refresh_s=" in message and "signal_s=" in message and "depth_wait_s=" in message


def test_first_post_audit_proxy_delegates_and_fires_once(tmp_path) -> None:
    from src.live.audit import AuditLog
    from src.live.runner import _FirstPostAudit

    audit = AuditLog(tmp_path / "a.jsonl")
    calls: list[int] = []
    proxy = _FirstPostAudit(audit, lambda: calls.append(1))
    assert proxy.path == audit.path
    assert proxy.context is audit.context
    proxy.record("intent_outcome", symbol="AAAUSDT")
    assert calls == []
    proxy.record("order_posted", symbol="AAAUSDT")
    proxy.record("order_posted", symbol="AAAUSDT")
    assert calls == [1]


def test_latency_audit_failure_never_halts_cycle(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings

    weights_path = _seed_weights(tmp_path)
    _install_stubs(tmp_path, monkeypatch, post=True)

    def _boom(*a, **k):
        raise RuntimeError("audit disk full")

    monkeypatch.setattr(runner_mod, "_record_execution_latency", _boom)
    settings = LiveSettings(
        ledger_path=str(tmp_path / "ledger.json"),
        order_journal_path=str(tmp_path / "journal.jsonl"),
        exec_depth_capture_enabled=False,
    )
    with caplog.at_level(logging.WARNING, logger="LiveRunner"):
        report = runner_mod.run_shadow_cycle(
            settings, DECISION_TIME, weights_path, now=RELEASE_AT + pd.Timedelta(seconds=42),
        )
    assert report.status == "COMPLETE"
    assert "execution_latency audit failed" in caplog.text


def test_latency_recorded_once_when_nothing_posted(tmp_path, monkeypatch) -> None:
    import src.live.runner as runner_mod
    from src.live.settings import LiveSettings

    weights_path = _seed_weights(tmp_path)
    _install_stubs(tmp_path, monkeypatch, post=False)
    settings = LiveSettings(
        ledger_path=str(tmp_path / "ledger.json"),
        order_journal_path=str(tmp_path / "journal.jsonl"),
        exec_depth_capture_enabled=False,
    )
    now = RELEASE_AT + pd.Timedelta(seconds=42)
    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=now)
    assert report.status == "COMPLETE"
    events = _latency_events(tmp_path)
    assert len(events) == 1
    assert events[0]["seconds_after_release"] is None
    assert events[0]["first_post_at"] is None
