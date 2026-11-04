"""Production step wiring must accept the single positional decision time run_daemon passes."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.live import scheduler
from src.live.settings import LiveSettings


@pytest.mark.parametrize("name", ["signal", "refresh", "venue", "prefetch"])
def test_default_step_fns_bind_decision_time_positionally(name: str, tmp_path: Path) -> None:
    fns = scheduler.default_step_fns(LiveSettings(), tmp_path / "w.parquet")
    bound = inspect.signature(fns[name]).bind(pd.Timestamp("2026-09-26", tz="UTC"))
    assert len(bound.args) == 1


def test_default_refresh_receives_decision_time_in_its_own_parameter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake(settings: LiveSettings, weights_path: Path, decision_time: pd.Timestamp) -> str:
        seen.update(settings=settings, weights_path=weights_path, decision_time=decision_time)
        return "ok"

    monkeypatch.setattr(scheduler, "_default_data_refresh", fake)
    monkeypatch.setattr(scheduler, "_default_funding_prefetch", fake)
    settings = LiveSettings()
    target = pd.Timestamp("2026-09-26", tz="UTC")
    for name in ("refresh", "prefetch"):
        seen.clear()
        assert scheduler.default_step_fns(settings, tmp_path / "w.parquet")[name](target) == "ok"
        assert seen["settings"] is settings
        assert seen["decision_time"] == target


def test_funding_info_failure_degrades_to_inference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from types import SimpleNamespace

    from src.live import data_refresh, venue_listing
    from src.market_data.binance import funding_info

    def boom() -> None:
        raise RuntimeError("fundingInfo request failed")

    seen: dict[str, object] = {}

    def fake_tails(*args: object, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(total=0, fetched=0, fresh=0, failed=0, elapsed_s=0.0)

    monkeypatch.setattr(funding_info, "fetch_funding_info", boom)
    monkeypatch.setattr(scheduler, "FUNDING_INTERVAL_SNAPSHOT_DIR", tmp_path / "fi")
    monkeypatch.setattr(venue_listing, "latest_venue_listing_or_none", lambda root: object())
    monkeypatch.setattr(data_refresh, "build_refresh_universe", lambda *a, **k: SimpleNamespace(trading=(), tracked_pending=()))
    monkeypatch.setattr(data_refresh, "refresh_funding_tails", fake_tails)
    monkeypatch.setattr(scheduler, "_required_symbols", lambda settings, path: frozenset())

    with caplog.at_level("WARNING", logger="LiveScheduler"):
        scheduler._default_funding_prefetch(LiveSettings(), tmp_path / "w.parquet", pd.Timestamp("2026-11-03", tz="UTC"))

    assert seen["funding_intervals"] is None
    assert sum("stage=funding_info status=UNAVAILABLE" in r.message for r in caplog.records) == 1


def test_funding_info_snapshot_persisted_and_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.live import data_refresh, venue_listing
    from src.market_data.binance import funding_info

    snapshot = funding_info.parse_funding_info(
        [{"symbol": "QUSDT", "fundingIntervalHours": 4}], captured_at=pd.Timestamp("2026-11-03T23:00:30", tz="UTC"),
    )
    seen: dict[str, object] = {}

    def fake_tails(*args: object, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(total=0, fetched=0, fresh=0, failed=0, elapsed_s=0.0)

    root = tmp_path / "fi"
    monkeypatch.setattr(funding_info, "fetch_funding_info", lambda: snapshot)
    monkeypatch.setattr(scheduler, "FUNDING_INTERVAL_SNAPSHOT_DIR", root)
    monkeypatch.setattr(venue_listing, "latest_venue_listing_or_none", lambda r: object())
    monkeypatch.setattr(data_refresh, "build_refresh_universe", lambda *a, **k: SimpleNamespace(trading=(), tracked_pending=()))
    monkeypatch.setattr(data_refresh, "refresh_funding_tails", fake_tails)
    monkeypatch.setattr(scheduler, "_required_symbols", lambda settings, path: frozenset())

    scheduler._default_funding_prefetch(LiveSettings(), tmp_path / "w.parquet", pd.Timestamp("2026-11-03", tz="UTC"))

    assert seen["funding_intervals"] is snapshot
    persisted = funding_info.latest_funding_interval_snapshot_or_none(root)
    assert persisted is not None
    assert dict(persisted.interval_ms) == dict(snapshot.interval_ms)
