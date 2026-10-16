"""Hermetic, clock-controlled two-account drill sessions."""

from __future__ import annotations

import time
from decimal import Decimal

import pytest
import pandas as pd

from src.live import order_journal
from src.live.testnet_drill import session as session_module
from tests.unit.live.testnet_drill._exchange import Exchange
from tests.unit.live.testnet_drill._stubs import make_config, make_settings, patch_drill_root


@pytest.fixture
def exchange_session(tmp_path, monkeypatch):
    patch_drill_root(monkeypatch, tmp_path)
    now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(order_journal, "_utc_now", lambda: pd.Timestamp(now[0], unit="s", tz="UTC"))
    opened = []

    def factory(*, config=None, settings=None):
        venue = Exchange()
        monkeypatch.setattr(session_module, "KeepAliveTransport", lambda _url: venue)
        config_values = {"probe_notional_usdt": Decimal("20"), "max_probe_notional_usdt": Decimal("30"),
                         "scenario_timeout_s": 500, "settle_wait_s": 0.001, "settle_poll_s": 0.001}
        config_values.update(config or {})
        setting_values = {"maker_fee_bps": 5.0, "taker_fee_bps": 5.0, "passive_timeout_minutes": 1,
                          "repeg_interval_s": 6.0}
        setting_values.update(settings or {})
        session = session_module.open_drill_session(make_settings(**setting_values), make_config(**config_values),
            run_id=f"dual{len(opened)}", inner=venue)
        opened.append(session)
        return session, venue, now

    yield factory
    for session in opened:
        session_module.close_session_handles(session)
