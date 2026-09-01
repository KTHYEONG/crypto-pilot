# ruff: noqa
def test_refresh_live_market_data_skips_fresh_symbols(tmp_path, monkeypatch) -> None:
    # Given: one dev symbol with a parquet whose tail == now
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    ts = [int((now - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / "BTCUSDT.parquet", index=False)
    monkeypatch.setattr(data_refresh, "symbol_partition", lambda s: "dev")

    calls: list[str] = []

    class _Collector:
        def __getattr__(self, _name):
            def _rec(*a, **k):
                calls.append(a[0] if a else "?")
                return True
            return _rec

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path,
        now=now,
        lookback_days=40,
        max_workers=2,
        deadline_s=30.0,
        freshness_floor_hours=1.5,
        min_symbols=1,
        max_fail_fraction=0.15,
        collector=_Collector(),
    )

    # Then
    assert report.total == 1
    assert report.fresh == 1
    assert report.refreshed == 0
    assert calls == []
    assert report.ok is True


def test_refresh_live_market_data_fetches_stale_symbol_with_tail_window(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    tail = now - pd.Timedelta(days=1)
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    ts = [int((tail - pd.Timedelta(hours=h)).value // 10**6) for h in range(72)]
    pd.DataFrame({"timestamp": ts, "close": [1.0] * 72}).to_parquet(d / "ETHUSDT.parquet", index=False)
    monkeypatch.setattr(data_refresh, "symbol_partition", lambda s: "dev")

    seen: dict[str, str] = {}

    def _fake_one(collector, symbol, start, end):
        seen["start"] = start
        seen["end"] = end
        return True

    monkeypatch.setattr(data_refresh, "_refresh_one_symbol_tail", _fake_one)

    report = data_refresh.refresh_live_market_data(
        tmp_path, now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        freshness_floor_hours=1.5, min_symbols=1, max_fail_fraction=0.15, collector=object(),
    )

    assert report.refreshed == 1
    assert report.fresh == 0
    started = pd.Timestamp(seen["start"])
    assert started >= now - pd.Timedelta(days=40)
    assert started <= tail  # never fetches from beyond the existing tail
    assert report.ok is True


def test_refresh_live_market_data_cold_universe_raises_without_network(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    from src.live import data_refresh

    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1, 2, 3], "close": [1.0, 1.0, 1.0]}).to_parquet(d / "BTCUSDT.parquet", index=False)
    monkeypatch.setattr(data_refresh, "symbol_partition", lambda s: "dev")

    def _boom(*a, **k):
        raise AssertionError("collector must not be constructed on cold universe")

    monkeypatch.setattr(data_refresh, "DataCollector", _boom)

    with pytest.raises(data_refresh.ColdUniverseError):
        data_refresh.refresh_live_market_data(
            tmp_path, now=pd.Timestamp("2026-09-01T00:00:00Z"), lookback_days=40,
            max_workers=2, deadline_s=30.0, freshness_floor_hours=1.5,
            min_symbols=100, max_fail_fraction=0.15,
        )


def test_refresh_live_market_data_deadline_stops_further_fetches(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    for sym in ("AUSDT", "BUSDT", "CUSDT"):
        ts = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / f"{sym}.parquet", index=False)
    monkeypatch.setattr(data_refresh, "symbol_partition", lambda s: "dev")

    def _must_not_fetch(*a, **k):
        raise AssertionError("no fetch past the deadline")

    monkeypatch.setattr(data_refresh, "_refresh_one_symbol_tail", _must_not_fetch)

    report = data_refresh.refresh_live_market_data(
        tmp_path, now=now, lookback_days=40, max_workers=2, deadline_s=0.0,
        freshness_floor_hours=1.5, min_symbols=1, max_fail_fraction=1.0, collector=object(),
    )

    assert report.deadline_hit is True
    assert report.deadline_skipped == 3
    assert report.refreshed == 0 and report.failed == 0


def test_refresh_live_market_data_ok_false_on_excess_failures(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    for sym in ("AUSDT", "BUSDT"):
        ts = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / f"{sym}.parquet", index=False)
    monkeypatch.setattr(data_refresh, "symbol_partition", lambda s: "dev")
    monkeypatch.setattr(data_refresh, "_refresh_one_symbol_tail", lambda *a, **k: False)

    report = data_refresh.refresh_live_market_data(
        tmp_path, now=now, lookback_days=40, max_workers=2, deadline_s=30.0,
        freshness_floor_hours=1.5, min_symbols=2, max_fail_fraction=0.15, collector=object(),
    )

    assert report.failed == 2
    assert report.ok is False


def test_market_data_staleness_hours_p90_ignores_delisted_outliers(tmp_path, monkeypatch) -> None:
    import math
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    monkeypatch.setattr(data_refresh, "symbol_partition", lambda s: "dev")
    assert math.isinf(data_refresh.market_data_staleness_hours(tmp_path, now=now))

    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    # 18 healthy symbols at 2h, 2 delisted at 2000h -> p90 must stay ~2h, not blow up
    plan = [(f"H{i}USDT", 2) for i in range(40)] + [("DEADAUSDT", 2000), ("DEADBUSDT", 2000)]
    for sym, lag_h in plan:
        ts = [int((now - pd.Timedelta(hours=lag_h + k)).value // 10**6) for k in range(10)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 10}).to_parquet(d / f"{sym}.parquet", index=False)

    got = data_refresh.market_data_staleness_hours(tmp_path, now=now)
    assert got < 48.0  # robust to the 2 delisted outliers

    # a real systemic outage: every symbol stale -> metric rises
    for sym, _ in plan:
        ts = [int((now - pd.Timedelta(hours=200 + k)).value // 10**6) for k in range(10)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 10}).to_parquet(d / f"{sym}.parquet", index=False)
    assert data_refresh.market_data_staleness_hours(tmp_path, now=now) > 150.0
