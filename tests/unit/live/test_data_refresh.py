# ruff: noqa
def test_refresh_live_market_data_skips_fresh_symbols(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_ohlcv(symbol: str, tail: pd.Timestamp) -> None:
        stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

    def _write_funding(symbol: str, last: pd.Timestamp) -> None:
        stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    _write_ohlcv("BTCUSDT", now)
    _write_funding("BTCUSDT", now)
    calls: list[str] = []

    class _Collector:
        def __getattr__(self, _name):
            def _rec(*a, **k):
                calls.append(a[0] if a else "?")
                return True
            return _rec

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["BTCUSDT"], now=now, lookback_days=40, max_workers=2, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert report.total == 1
    assert report.fresh == 1
    assert report.refreshed == 0
    assert report.funding_stale == 0
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

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    seen: dict[str, str] = {}

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            seen["start"] = start
            seen["end"] = end
            stamps = [_ms(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(d / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            seen["funding_start"] = start
            funding_dir = tmp_path / "funding"
            funding_dir.mkdir(parents=True, exist_ok=True)
            stamps = [_ms(now - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
            pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["ETHUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert report.refreshed == 1
    assert report.fresh == 0
    started = pd.Timestamp(seen["start"])
    assert started >= now - pd.Timedelta(days=40)
    assert started <= tail  # never fetches from beyond the existing tail
    assert seen["funding_start"] == str(now - pd.Timedelta(days=40))
    assert report.ok is True


def test_refresh_live_market_data_cold_universe_raises_without_network(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    from src.live import data_refresh

    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1, 2, 3], "close": [1.0, 1.0, 1.0]}).to_parquet(d / "BTCUSDT.parquet", index=False)

    def _boom(*a, **k):
        raise AssertionError("collector must not be constructed on cold universe")

    monkeypatch.setattr(data_refresh, "DataCollector", _boom)

    with pytest.raises(data_refresh.ColdUniverseError):
        data_refresh.refresh_live_market_data(
            tmp_path, symbols=["BTCUSDT"], now=pd.Timestamp("2026-09-01T00:00:00Z"), lookback_days=40,
            max_workers=2, deadline_s=30.0, min_symbols=100, max_fail_fraction=0.15,
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

    class _MustNotFetch:
        def ensure_ohlcv_data(self, *a, **k):
            raise AssertionError("no fetch past the deadline")

        def ensure_funding_data(self, *a, **k):
            raise AssertionError("no fetch past the deadline")

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT", "CUSDT"], now=now, lookback_days=40, max_workers=2, deadline_s=0.0,
        min_symbols=1, max_fail_fraction=1.0, collector=_MustNotFetch(),
    )

    assert report.deadline_hit is True
    assert report.deadline_skipped == 3
    assert report.refreshed == 0 and report.failed == 0
    # 마감으로 건너뛴 심볼은 성공으로 집계되지 않는다
    assert report.ok is False


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
    calls: list[str] = []

    class _FailingCollector:
        def ensure_ohlcv_data(self, symbol, *a, **k):
            calls.append(symbol)
            raise RuntimeError("venue 5xx")

        def ensure_funding_data(self, *a, **k):
            raise AssertionError("funding must not run after a klines failure")

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT"], now=now, lookback_days=40, max_workers=2, deadline_s=30.0,
        min_symbols=2, max_fail_fraction=0.15, collector=_FailingCollector(),
    )

    assert sorted(calls) == ["AUSDT", "BUSDT"]

    assert report.failed == 2
    assert report.ok is False


def test_market_data_staleness_hours_p90_ignores_delisted_outliers(tmp_path, monkeypatch) -> None:
    import math
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    assert math.isinf(data_refresh.market_data_staleness_hours(tmp_path, now=now))

    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    plan = [(f"H{i}USDT", 2) for i in range(40)] + [("DEADAUSDT", 2000), ("DEADBUSDT", 2000)]
    for sym, lag_h in plan:
        ts = [int((now - pd.Timedelta(hours=lag_h + k)).value // 10**6) for k in range(10)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 10, "volume": [1.0] * 10}).to_parquet(d / f"{sym}.parquet", index=False)

    got = data_refresh.market_data_staleness_hours(tmp_path, now=now)
    assert got < 48.0

    for sym, _ in plan:
        ts = [int((now - pd.Timedelta(hours=200 + k)).value // 10**6) for k in range(10)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 10, "volume": [1.0] * 10}).to_parquet(d / f"{sym}.parquet", index=False)
    assert data_refresh.market_data_staleness_hours(tmp_path, now=now) > 150.0


def test_refresh_live_market_data_counts_funding_failure_as_failed(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh
    from src.market_data.binance.futures import BinanceFundingFetchError

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    ts = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / "AUSDT.parquet", index=False)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            return None

        def ensure_funding_data(self, symbol, start, end):
            raise BinanceFundingFetchError(symbol=symbol, http_code=403, url="https://fapi.binance.com/fapi/v1/fundingRate")

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert report.failed == 1
    assert report.refreshed == 0
    assert report.ok is False


def test_refresh_live_market_data_aborts_remaining_symbols_on_ip_block(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh
    from src.market_data.binance.futures import BinanceIpBlockedError

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    ts = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    for sym in ("AUSDT", "BUSDT", "CUSDT"):
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / f"{sym}.parquet", index=False)
    calls: list[str] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            calls.append(symbol)
            raise BinanceIpBlockedError(http_code=418, url="https://fapi.binance.com/fapi/v1/klines")

        def ensure_funding_data(self, symbol, start, end):
            raise AssertionError("funding must not run after a klines block")

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT", "CUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert calls == ["AUSDT"]
    assert report.ip_blocked is True
    assert report.failed == 3
    assert report.refreshed == 0
    assert report.ok is False


def test_refresh_live_market_data_keeps_fresh_count_during_ip_block(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_ohlcv(symbol: str, tail: pd.Timestamp) -> None:
        stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

    def _write_funding(symbol: str, last: pd.Timestamp) -> None:
        stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    from src.market_data.binance.futures import BinanceIpBlockedError

    _write_ohlcv("AUSDT", now - pd.Timedelta(days=5))
    _write_ohlcv("ZUSDT", now)
    _write_funding("ZUSDT", now)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            raise BinanceIpBlockedError(http_code=429, url="https://fapi.binance.com/fapi/v1/klines")

        def ensure_funding_data(self, symbol, start, end):
            raise AssertionError("funding must not run after a klines block")

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "ZUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert report.fresh == 1
    assert report.failed == 1
    assert report.ip_blocked is True
    assert report.ok is False


def test_refresh_live_market_data_ignores_legacy_temp_artifacts(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_ohlcv(symbol: str, tail: pd.Timestamp) -> None:
        stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

    def _write_funding(symbol: str, last: pd.Timestamp) -> None:
        stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    _write_ohlcv("AUSDT", now)
    _write_funding("AUSDT", now)
    (ohlcv_dir / "AUSDT.tmp.parquet").write_bytes((ohlcv_dir / "AUSDT.parquet").read_bytes())

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=object(),
    )

    assert report.total == 1
    assert report.fresh == 1
    assert report.ip_blocked is False


def test_market_data_staleness_hours_ignores_legacy_temp_artifacts(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    healthy = [int((now - pd.Timedelta(hours=2 + k)).value // 10**6) for k in range(10)]
    stale = [int((now - pd.Timedelta(hours=2000 + k)).value // 10**6) for k in range(10)]
    pd.DataFrame({"timestamp": healthy, "close": [1.0] * 10, "volume": [1.0] * 10}).to_parquet(d / "AUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": stale, "close": [1.0] * 10, "volume": [1.0] * 10}).to_parquet(d / "AUSDT.tmp.parquet", index=False)

    got = data_refresh.market_data_staleness_hours(tmp_path, now=now)

    assert got == 2.0

def test_refresh_live_market_data_refreshes_fresh_ohlcv_with_stale_funding(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_ohlcv(symbol: str, tail: pd.Timestamp) -> None:
        stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

    def _write_funding(symbol: str, last: pd.Timestamp) -> None:
        stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    _write_ohlcv("AUSDT", now)
    _write_funding("AUSDT", now - pd.Timedelta(hours=24))
    calls: list[str] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            return None

        def ensure_funding_data(self, symbol, start, end):
            calls.append(symbol)
            _write_funding(symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert calls == ["AUSDT"]
    assert report.fresh == 0
    assert report.refreshed == 1


def test_refresh_live_market_data_reports_funding_stale_after_run(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_ohlcv(symbol: str, tail: pd.Timestamp) -> None:
        stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

    def _write_funding(symbol: str, last: pd.Timestamp) -> None:
        stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    _write_ohlcv("AUSDT", now)
    _write_funding("AUSDT", now)
    _write_ohlcv("BUSDT", now)
    _write_funding("BUSDT", now - pd.Timedelta(hours=24))

    def _ms2(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            stamps = [_ms2(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            return None

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert report.fresh == 1
    assert report.refreshed == 0
    assert report.incomplete == 1
    assert ("BUSDT",) == report.not_current_sample
    assert report.funding_stale == 1


def test_funding_fresh_on_disk_treats_missing_or_unreadable_file_as_stale(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_ohlcv(symbol: str, tail: pd.Timestamp) -> None:
        stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

    def _write_funding(symbol: str, last: pd.Timestamp) -> None:
        stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)
    assert data_refresh._funding_fresh_on_disk(tmp_path, "MISSINGUSDT", now) is False
    (funding_dir / "BROKENUSDT.parquet").write_bytes(b"not a parquet")
    assert data_refresh._funding_fresh_on_disk(tmp_path, "BROKENUSDT", now) is False
    pd.DataFrame({"funding_rate": [0.0001]}).to_parquet(funding_dir / "NOTSUSDT.parquet", index=False)
    assert data_refresh._funding_fresh_on_disk(tmp_path, "NOTSUSDT", now) is False
    _write_funding("OKUSDT", now)
    assert data_refresh._funding_fresh_on_disk(tmp_path, "OKUSDT", now) is True


def test_market_data_staleness_hours_ignores_symbols_without_recent_volume(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    healthy = [int((now - pd.Timedelta(hours=2 + k)).value // 10**6) for k in range(10)]
    zombie = [int((now - pd.Timedelta(hours=k)).value // 10**6) for k in range(10)]
    pd.DataFrame({"timestamp": healthy, "close": [1.0] * 10, "volume": [3.0] * 10}).to_parquet(d / "AUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": zombie, "close": [1.0] * 10, "volume": [0.0] * 10}).to_parquet(d / "ZUSDT.parquet", index=False)

    assert data_refresh.market_data_staleness_hours(tmp_path, now=now) == 2.0


def test_funding_fresh_on_disk_treats_internal_gap_as_stale_when_window_given(tmp_path) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)
    # Given: 꼬리는 최신(now)이지만 now-64h ~ now-16h 사이가 비어 있는 8h 간격 펀딩
    offsets_h = [72, 64, 16, 8, 0]
    stamps = [int((now - pd.Timedelta(hours=o)).value // 10**6) for o in offsets_h]
    pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * len(stamps)}).to_parquet(funding_dir / "GAPUSDT.parquet", index=False)

    # When/Then: 창을 주지 않으면 기존 꼬리 판정 유지, 창을 주면 내부 공백 때문에 stale
    assert data_refresh._funding_fresh_on_disk(tmp_path, "GAPUSDT", now) is True
    assert data_refresh._funding_fresh_on_disk(tmp_path, "GAPUSDT", now, window_start=now - pd.Timedelta(days=40)) is False
    assert data_refresh._funding_fresh_on_disk(tmp_path, "GAPUSDT", now, window_start=now - pd.Timedelta(hours=16)) is True

def test_refresh_live_market_data_refetches_gap_symbol_with_lookback_funding_start(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    # Given: OHLCV 최신 + 펀딩 꼬리 최신이지만 내부 공백이 있는 심볼
    ohlcv_stamps = [_ms(now - pd.Timedelta(hours=h)) for h in range(48)]
    pd.DataFrame({"timestamp": ohlcv_stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / "GAPUSDT.parquet", index=False)
    funding_stamps = [_ms(now - pd.Timedelta(hours=o)) for o in (72, 64, 16, 8, 0)]
    pd.DataFrame({"timestamp": funding_stamps, "funding_rate": [0.0001] * 5}).to_parquet(funding_dir / "GAPUSDT.parquet", index=False)
    seen: dict[str, object] = {}

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            seen["symbol"] = symbol
            stamps = [_ms(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            seen["funding_start"] = start
            stamps = [_ms(now - pd.Timedelta(hours=8 * k)) for k in range(120)]
            pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 120}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["GAPUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    # Then: 공백 심볼은 fresh 로 건너뛰지 않고 lookback 창 start 로 재조회, 디스크 검증 후 refreshed
    assert seen == {"symbol": "GAPUSDT", "funding_start": str(now - pd.Timedelta(days=40))}
    assert report.fresh == 0
    assert report.refreshed == 1
    assert report.funding_stale == 0


def test_parse_listed_symbols_accepts_any_status_and_rejects_malformed() -> None:
    from src.live.data_refresh import parse_listed_symbols

    payload = {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}, {"symbol": "ZOMBIEUSDT", "status": "SETTLING"}, "junk", {"status": "TRADING"}]}
    assert parse_listed_symbols(payload) == frozenset({"BTCUSDT", "ZOMBIEUSDT"})
    assert parse_listed_symbols({}) is None
    assert parse_listed_symbols({"symbols": "nope"}) is None
    assert parse_listed_symbols({"symbols": []}) is None
    assert parse_listed_symbols({"symbols": [{"status": "TRADING"}]}) is None


def test_fetch_listed_symbols_fails_open_on_transport_or_payload_errors() -> None:
    import json
    import urllib.error

    from src.live.data_refresh import EXCHANGE_INFO_TIMEOUT_S, EXCHANGE_INFO_URL, fetch_listed_symbols

    seen: list[tuple[str, float]] = []

    class _Resp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return self._body

    def _ok(url, timeout):
        seen.append((url, timeout))
        return _Resp(json.dumps({"symbols": [{"symbol": "BTCUSDT"}]}).encode("utf-8"))

    assert fetch_listed_symbols(opener=_ok) == frozenset({"BTCUSDT"})
    assert seen == [(EXCHANGE_INFO_URL, EXCHANGE_INFO_TIMEOUT_S)]

    def _blocked(url, timeout):
        raise urllib.error.HTTPError(url, 418, "teapot", None, None)

    def _timeout(url, timeout):
        raise TimeoutError("slow")

    assert fetch_listed_symbols(opener=_blocked) is None
    assert fetch_listed_symbols(opener=_timeout) is None
    assert fetch_listed_symbols(opener=lambda url, timeout: _Resp(b"{not json")) is None
    assert fetch_listed_symbols(opener=lambda url, timeout: _Resp(b"{}")) is None
    assert fetch_listed_symbols(opener=lambda url, timeout: _Resp(b"[]")) is None


def test_split_absent_symbols_guards_untrusted_listing(monkeypatch) -> None:
    from src.live import data_refresh

    symbols = [f"S{i:02d}USDT" for i in range(40)]
    listed = frozenset(symbols[:-1])

    assert data_refresh.split_absent_symbols(symbols, None) == (symbols, [])
    assert data_refresh.split_absent_symbols(symbols, listed) == (symbols[:-1], ["S39USDT"])
    # 제거 비율이 상한을 넘으면 목록을 신뢰하지 않고 원래 유니버스를 유지
    assert data_refresh.split_absent_symbols(symbols, frozenset(symbols[:30])) == (symbols, [])
    # 상한은 호출 시점 모듈 상수로 읽는다
    monkeypatch.setattr(data_refresh, "ABSENT_MAX_FRACTION", 0.5)
    assert data_refresh.split_absent_symbols(symbols, frozenset(symbols[:30])) == (symbols[:30], symbols[30:])
    assert data_refresh.split_absent_symbols([], listed) == ([], [])


def test_refresh_live_market_data_excludes_absent_symbols(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    stale = now - pd.Timedelta(days=2)
    for sym in ("AAAUSDT", "BBBUSDT", "GONEUSDT"):
        stamps = [int((stale - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{sym}.parquet", index=False)
    monkeypatch.setattr(data_refresh, "ABSENT_MAX_FRACTION", 0.5)
    refreshed: list[str] = []

    def _ms2(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    def _write_current(symbol: str) -> None:
        stamps = [_ms2(now - pd.Timedelta(hours=h)) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)
        funding_dir = tmp_path / "funding"
        funding_dir.mkdir(parents=True, exist_ok=True)
        fstamps = [_ms2(now - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
        pd.DataFrame({"timestamp": fstamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            refreshed.append(symbol)
            _write_current(symbol)

        def ensure_funding_data(self, symbol, start, end):
            return None
    caplog.set_level(logging.INFO, logger="src.live.data_refresh")

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AAAUSDT", "BBBUSDT", "GONEUSDT"], now=now, lookback_days=40, max_workers=2, deadline_s=30.0, min_symbols=2, max_fail_fraction=0.15, collector=_Collector(), listed_symbols=frozenset({"AAAUSDT", "BBBUSDT"}),
    )

    # Then: 거래소에 없는 심볼은 요청/집계/신선도 계산에서 모두 빠진다
    assert sorted(refreshed) == ["AAAUSDT", "BBBUSDT"]
    assert (report.total, report.refreshed, report.failed, report.absent) == (2, 2, 0, 1)
    assert report.ok is True
    messages = [record.getMessage() for record in caplog.records]
    assert any("absent_symbols=1 sample=GONEUSDT" in m for m in messages)
    assert any("absent=1" in m and "stage=refresh_live_market_data total=2" in m for m in messages)

    # Given: 목록 미제공(None)은 기존 동작
    refreshed.clear()
    for sym in ("AAAUSDT", "BBBUSDT", "GONEUSDT"):
        stamps = [int((stale - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{sym}.parquet", index=False)
    legacy = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AAAUSDT", "BBBUSDT", "GONEUSDT"], now=now, lookback_days=40, max_workers=2, deadline_s=30.0, min_symbols=2, max_fail_fraction=0.15, collector=_Collector(),
    )
    assert sorted(refreshed) == ["AAAUSDT", "BBBUSDT", "GONEUSDT"]
    assert (legacy.total, legacy.absent) == (3, 0)


def test_refresh_live_market_data_funding_block_keeps_ohlcv_refresh_for_remaining_symbols(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh
    from src.market_data.binance.futures import BinanceIpBlockedError

    # Given
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    ts = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    for sym in ("AUSDT", "BUSDT", "CUSDT"):
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / f"{sym}.parquet", index=False)
    ohlcv_calls: list[str] = []
    funding_calls: list[str] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            ohlcv_calls.append(symbol)

        def ensure_funding_data(self, symbol, start, end):
            funding_calls.append(symbol)
            raise BinanceIpBlockedError(http_code=403, url="https://fapi.binance.com/fapi/v1/fundingRate")

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT", "CUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    # Then
    assert ohlcv_calls == ["AUSDT", "BUSDT", "CUSDT"]
    assert funding_calls == ["AUSDT"]
    assert report.funding_blocked is True
    assert report.ip_blocked is False
    assert report.failed == 3
    assert report.refreshed == 0
    assert report.ok is False

def test_refresh_live_market_data_klines_block_still_aborts_remaining_symbols(tmp_path, monkeypatch) -> None:
    import pandas as pd
    from src.live import data_refresh
    from src.market_data.binance.futures import BinanceIpBlockedError

    # Given
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    ts = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    for sym in ("AUSDT", "BUSDT", "CUSDT"):
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 48}).to_parquet(d / f"{sym}.parquet", index=False)
    ohlcv_calls: list[str] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            ohlcv_calls.append(symbol)
            raise BinanceIpBlockedError(http_code=418, url="https://fapi.binance.com/fapi/v1/klines")

        def ensure_funding_data(self, symbol, start, end):
            raise AssertionError("funding must not run after a klines block")

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT", "CUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    # Then
    assert ohlcv_calls == ["AUSDT"]
    assert report.ip_blocked is True
    assert report.funding_blocked is False
    assert report.failed == 3
    assert report.ok is False


def test_listed_crypto_perpetuals_split_excludes_non_coin() -> None:
    from src.live.data_refresh import listed_crypto_perpetuals

    payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "AAPLUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "EQUITY"},
            {"symbol": "ETHUSDT", "status": "SETTLING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "COIN"},
            {"symbol": "BTCUSDT_260925", "status": "TRADING", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT", "underlyingType": "COIN"},
            "junk",
            {"status": "TRADING"},
        ]
    }

    crypto, non_crypto = listed_crypto_perpetuals(payload)

    assert crypto == frozenset({"BTCUSDT"})
    assert non_crypto == frozenset({"AAPLUSDT"})


def test_listed_crypto_perpetuals_empty_crypto_fails_closed() -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live.data_refresh import listed_crypto_perpetuals

    with pytest.raises(DataIntegrityError):
        listed_crypto_perpetuals({"symbols": []})
    with pytest.raises(DataIntegrityError):
        listed_crypto_perpetuals({})
    with pytest.raises(DataIntegrityError):
        listed_crypto_perpetuals(
            {"symbols": [{"symbol": "AAPLUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "underlyingType": "EQUITY"}]}
        )


def test_refresh_touches_only_given_symbols(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    for sym in ("BTCUSDT", "AAPLUSDT"):
        stamps = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
        pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{sym}.parquet", index=False)
    seen: list[str] = []

    def _ms3(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            seen.append(f"ohlcv:{symbol}")
            stamps = [_ms3(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            seen.append(f"funding:{symbol}")
            funding_dir = tmp_path / "funding"
            funding_dir.mkdir(parents=True, exist_ok=True)
            stamps = [_ms3(now - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
            pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["BTCUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert set(seen) == {"ohlcv:BTCUSDT", "funding:BTCUSDT"}
    assert report.total == 1


def test_missing_file_seeds_long_window(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    old = now - pd.Timedelta(days=50)
    stamps = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / "BTCUSDT.parquet", index=False)
    seen: dict[str, str] = {}

    def _ms4(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            seen[symbol] = start
            stamps = [_ms4(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            funding_dir = tmp_path / "funding"
            funding_dir.mkdir(parents=True, exist_ok=True)
            stamps = [_ms4(now - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
            pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    data_refresh.refresh_live_market_data(
        tmp_path, symbols=["BTCUSDT", "SOLUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=1.0, collector=_Collector(),
        seed_lookback_days=150,
    )

    assert pd.Timestamp(seen["SOLUSDT"]) == now - pd.Timedelta(days=150)
    assert pd.Timestamp(seen["BTCUSDT"]) == now - pd.Timedelta(days=40)


def _delist_entry(symbol, *, status="TRADING", contract_type="PERPETUAL", underlying_type="COIN", quote_asset="USDT", delivery_time=None, announced=False):
    from src.live.venue_listing import VenueListingEntry

    return VenueListingEntry(
        symbol=symbol, status=status, contract_type=contract_type, underlying_type=underlying_type,
        quote_asset=quote_asset, delivery_time=delivery_time, announced_delisting=announced,
        delisting_first_seen_at=None,
    )


def _delist_snapshot(entries, captured_at):
    from src.live.venue_listing import VenueListingSnapshot

    return VenueListingSnapshot(captured_at=captured_at, entries={e.symbol: e for e in entries})


def test_build_refresh_universe_settled_held_symbol_is_klines_only(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 거래 중 종목과 어제 인도된 SETTLING 보유 종목이 있는 상장 스냅샷
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    delivery = now - pd.Timedelta(days=1)
    listing = _delist_snapshot(
        [
            _delist_entry("BTCUSDT"),
            _delist_entry("XSETTLEUSDT", status="SETTLING", delivery_time=delivery),
        ],
        now,
    )

    # When
    universe = data_refresh.build_refresh_universe(
        listing, required_symbols=["XSETTLEUSDT"], non_crypto=frozenset(), now=now,
    )

    # Then: 인도 완료 보유 종목은 klines-only 추적 버킷에, 거래 버킷 밖에는 없다
    assert "XSETTLEUSDT" in universe.tracked_settled
    assert "XSETTLEUSDT" not in universe.trading
    assert "XSETTLEUSDT" not in universe.tracked_pending
    assert "XSETTLEUSDT" not in universe.unlisted_required
    assert universe.trading == ("BTCUSDT",)

    # When: klines-only 새로고침을 수행
    calls: list[str] = []

    def _ms5(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            calls.append(f"ohlcv:{symbol}")
            ohlcv_dir = tmp_path / "ohlcv" / "1h"
            ohlcv_dir.mkdir(parents=True, exist_ok=True)
            stamps = [_ms5(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            calls.append(f"funding:{symbol}")

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["XSETTLEUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
        klines_only_symbols=frozenset({"XSETTLEUSDT"}),
    )

    # Then: klines 만 요청하고 펀딩은 요청하지 않는다
    assert calls == ["ohlcv:XSETTLEUSDT"]
    assert report.refreshed == 1
    assert report.failed == 0


def test_build_refresh_universe_klines_only_success_counts_refreshed_not_failed(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 디스크 꼬리가 오래된 klines-only 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    old = now - pd.Timedelta(days=5)
    stamps = [int((old - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / "KONLYUSDT.parquet", index=False)
    ohlcv_calls: list[str] = []

    def _ms6(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            ohlcv_calls.append(symbol)
            stamps = [_ms6(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            raise AssertionError("funding must never be requested for klines-only symbols")

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["KONLYUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
        klines_only_symbols=frozenset({"KONLYUSDT"}),
    )

    # Then: kline 성공은 refreshed 로 집계되고 실패가 아니다
    assert ohlcv_calls == ["KONLYUSDT"]
    assert report.refreshed == 1
    assert report.failed == 0
    assert report.ok is True


def test_build_refresh_universe_klines_only_kline_failure_still_counts_failed(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: kline 호출이 실패하는 klines-only 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    (tmp_path / "ohlcv" / "1h").mkdir(parents=True)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            raise RuntimeError("kline endpoint down")

        def ensure_funding_data(self, symbol, start, end):
            raise AssertionError("funding must never be requested for klines-only symbols")

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["KFAILUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=1.0, collector=_Collector(),
        klines_only_symbols=frozenset({"KFAILUSDT"}),
    )

    # Then: 실제 kline 실패는 실패로 집계된다
    assert report.refreshed == 0
    assert report.failed == 1


def test_build_refresh_universe_klines_only_fresh_skips_funding_requirement(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: klines 꼬리는 최신이지만 펀딩 파일이 없는 klines-only 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)

    def _ms(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    stamps = [_ms(now - pd.Timedelta(hours=h)) for h in range(48)]
    pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / "KFRESHUSDT.parquet", index=False)

    class _Collector:
        def __getattr__(self, _name):
            raise AssertionError("no fetch needed for a fresh klines-only symbol")

    # When
    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["KFRESHUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
        klines_only_symbols=frozenset({"KFRESHUSDT"}),
    )

    # Then: 펀딩 없이 fresh 로 건너뛴다
    assert report.fresh == 1
    assert report.refreshed == 0
    assert report.failed == 0


def test_build_refresh_universe_announced_but_trading_keeps_funding(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 5일 뒤 인도 예정이지만 아직 TRADING 인 보유 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    delivery = now + pd.Timedelta(days=5)
    listing = _delist_snapshot(
        [
            _delist_entry("BTCUSDT"),
            _delist_entry("YANNUSDT", delivery_time=delivery, announced=True),
        ],
        now,
    )

    # When
    universe = data_refresh.build_refresh_universe(
        listing, required_symbols=["YANNUSDT"], non_crypto=frozenset(), now=now,
    )

    # Then: 아직 거래 버킷에 머물고 펀딩 추적 대상이다
    assert "YANNUSDT" in universe.trading
    assert "YANNUSDT" not in universe.tracked_settled
    assert "YANNUSDT" not in universe.tracked_pending
    assert "YANNUSDT" not in universe.unlisted_required

    # When: 일반 새로고침을 수행
    calls: list[str] = []

    def _ms7(ts: pd.Timestamp) -> int:
        return int(ts.value // 10**6)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            calls.append(f"ohlcv:{symbol}")
            ohlcv_dir = tmp_path / "ohlcv" / "1h"
            ohlcv_dir.mkdir(parents=True, exist_ok=True)
            stamps = [_ms7(now - pd.Timedelta(hours=h)) for h in range(48)]
            pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)

        def ensure_funding_data(self, symbol, start, end):
            calls.append(f"funding:{symbol}")
            funding_dir = tmp_path / "funding"
            funding_dir.mkdir(parents=True, exist_ok=True)
            stamps = [_ms7(now - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
            pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["YANNUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    # Then: 펀딩이 요청된다
    assert calls == ["ohlcv:YANNUSDT", "funding:YANNUSDT"]
    assert report.refreshed == 1


def test_build_refresh_universe_unlisted_required_symbol_surfaced() -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 장부에 있지만 상장 스냅샷에 없는 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    listing = _delist_snapshot([_delist_entry("BTCUSDT")], now)

    # When
    universe = data_refresh.build_refresh_universe(
        listing, required_symbols=["ZVANISHUSDT"], non_crypto=frozenset(), now=now,
    )

    # Then: 가져오기 목록이 아니라 미상장 버킷에 보고된다
    assert universe.unlisted_required == ("ZVANISHUSDT",)
    assert "ZVANISHUSDT" not in universe.trading
    assert "ZVANISHUSDT" not in universe.tracked_settled
    assert "ZVANISHUSDT" not in universe.tracked_pending


def test_build_refresh_universe_non_coin_required_goes_unlisted() -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 상장 스냅샷에 EQUITY 로 기록된 종목과 non_crypto 집합에 든 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    listing = _delist_snapshot(
        [
            _delist_entry("BTCUSDT"),
            _delist_entry("EQTYUSDT", underlying_type="EQUITY"),
        ],
        now,
    )

    # When
    universe = data_refresh.build_refresh_universe(
        listing, required_symbols=["EQTYUSDT", "IDXUSDT"], non_crypto=frozenset({"IDXUSDT"}), now=now,
    )

    # Then: Non-COIN 종목은 절대 추가되지 않고 미상장으로 보고된다
    assert universe.unlisted_required == ("EQTYUSDT", "IDXUSDT")
    assert "EQTYUSDT" not in universe.trading
    assert "EQTYUSDT" not in universe.tracked_settled
    assert "EQTYUSDT" not in universe.tracked_pending


def test_build_refresh_universe_unrequired_settled_symbol_not_tracked() -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 인도 완료된 SETTLING 종목에 장부/비중 노출이 전혀 없음
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    listing = _delist_snapshot(
        [
            _delist_entry("BTCUSDT"),
            _delist_entry("WOLDUSDT", status="SETTLING", delivery_time=now - pd.Timedelta(days=2)),
        ],
        now,
    )

    # When
    universe = data_refresh.build_refresh_universe(
        listing, required_symbols=[], non_crypto=frozenset(), now=now,
    )

    # Then: 어느 버킷에도 없다
    assert "WOLDUSDT" not in universe.trading
    assert "WOLDUSDT" not in universe.tracked_settled
    assert "WOLDUSDT" not in universe.tracked_pending
    assert "WOLDUSDT" not in universe.unlisted_required


def test_build_refresh_universe_settling_before_delivery_is_pending() -> None:
    import pandas as pd

    from src.live import data_refresh

    # Given: 인도 전 SETTLING, 인도 시각 없는 CLOSE, CLOSE 인도 완료 종목
    now = pd.Timestamp("2026-09-01T00:00:00Z")
    listing = _delist_snapshot(
        [
            _delist_entry("BTCUSDT"),
            _delist_entry("SPREUSDT", status="SETTLING", delivery_time=now + pd.Timedelta(days=1)),
            _delist_entry("SNONEUSDT", status="CLOSE"),
            _delist_entry("SCLOSEDUSDT", status="CLOSE", delivery_time=now - pd.Timedelta(hours=1)),
        ],
        now,
    )

    # When
    universe = data_refresh.build_refresh_universe(
        listing, required_symbols=["SPREUSDT", "SNONEUSDT", "SCLOSEDUSDT"], non_crypto=frozenset(), now=now,
    )

    # Then: 인도 전/시각 미상은 pending, 인도 완료 CLOSE 는 settled
    assert universe.tracked_pending == ("SNONEUSDT", "SPREUSDT")
    assert universe.tracked_settled == ("SCLOSEDUSDT",)


def test_build_refresh_universe_buckets_disjoint_sorted_and_guarded() -> None:
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    listing = _delist_snapshot(
        [
            _delist_entry("BTCUSDT"),
            _delist_entry("ETHUSDT"),
            _delist_entry("XSETTLEUSDT", status="SETTLING", delivery_time=now - pd.Timedelta(days=1)),
            _delist_entry("YPREUSDT", status="SETTLING", delivery_time=now + pd.Timedelta(days=1)),
        ],
        now,
    )

    # When: 네 버킷 경계에 걸친 required 집합
    universe = data_refresh.build_refresh_universe(
        listing,
        required_symbols=["YPREUSDT", "XSETTLEUSDT", "BTCUSDT", "ZMISSINGUSDT", "BTCUSDT"],
        non_crypto=frozenset(),
        now=now,
    )

    # Then: 네 튜플은 서로소이고 정렬되어 있으며, 거래 중 required 는 trading 에만 남는다
    buckets = [universe.trading, universe.tracked_settled, universe.tracked_pending, universe.unlisted_required]
    assert universe.trading == ("BTCUSDT", "ETHUSDT")
    assert universe.tracked_settled == ("XSETTLEUSDT",)
    assert universe.tracked_pending == ("YPREUSDT",)
    assert universe.unlisted_required == ("ZMISSINGUSDT",)
    for bucket in buckets:
        assert tuple(sorted(bucket)) == bucket
    assert len({s for bucket in buckets for s in bucket}) == sum(len(b) for b in buckets)

    # When/Then: TRADING 씨앗이 비면 닫힌 실패, naive now 도 닫힌 실패
    empty_listing = _delist_snapshot(
        [_delist_entry("XSETTLEUSDT", status="SETTLING", delivery_time=now - pd.Timedelta(days=1))],
        now,
    )
    with pytest.raises(DataIntegrityError):
        data_refresh.build_refresh_universe(empty_listing, required_symbols=[], non_crypto=frozenset(), now=now)
    with pytest.raises(DataIntegrityError):
        data_refresh.build_refresh_universe(
            listing, required_symbols=[], non_crypto=frozenset(), now=pd.Timestamp("2026-09-01T00:00:00"),
        )


def _edge06_ms(ts) -> int:
    import pandas as pd

    return int(pd.Timestamp(ts).value // 10**6)


def _edge06_write_klines(root, symbol: str, tail) -> None:
    import pandas as pd

    ohlcv_dir = root / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    stamps = [_edge06_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
    pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)


def _edge06_write_funding(root, symbol: str, last) -> None:
    import pandas as pd

    funding_dir = root / "funding"
    funding_dir.mkdir(parents=True, exist_ok=True)
    stamps = [_edge06_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
    pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(funding_dir / f"{symbol}.parquet", index=False)


def test_edge06_deadline_skip_is_not_ok(tmp_path, monkeypatch) -> None:
    import time

    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    symbols = [f"S{i:04d}USDT" for i in range(500)]
    required = frozenset({"S0499USDT"})
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: clock["t"])

    kline_calls: list[str] = []
    funding_calls: list[str] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            kline_calls.append(symbol)
            clock["t"] += 1.0
            _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            funding_calls.append(symbol)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=symbols, now=now, lookback_days=40, max_workers=1, deadline_s=239.5,
        min_symbols=100, max_fail_fraction=0.15, collector=_Collector(),
        required_symbols=required,
    )

    assert report.total == 500
    assert report.deadline_skipped == 260
    assert report.incomplete == 240
    assert report.ok is False
    assert report.deadline_hit is True
    assert report.fresh + report.refreshed + report.failed + report.deadline_skipped + report.incomplete == 500
    assert kline_calls[0] == "S0499USDT"
    skipped = set(symbols) - set(kline_calls)
    assert skipped == set(sorted(s for s in symbols if s not in required)[-260:])
    assert "S0499USDT" not in skipped
    assert funding_calls == []


def test_edge06_swallowed_empty_response_counts_incomplete(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            return None

        def ensure_funding_data(self, symbol, start, end):
            return None

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["EUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert report.refreshed == 0
    assert report.incomplete == 1
    assert report.not_current_sample == ("EUSDT",)
    assert report.ok is False


def test_edge06_required_incomplete_forces_not_ok(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    symbols = [f"S{i:03d}USDT" for i in range(200)]
    hole = "S042USDT"

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            if symbol != hole:
                _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            if symbol != hole:
                _edge06_write_funding(tmp_path, symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=symbols, now=now, lookback_days=40, max_workers=4, deadline_s=60.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
        required_symbols=frozenset({hole}),
    )

    assert report.incomplete == 1
    assert report.required_incomplete == (hole,)
    assert report.ok is False


def test_edge06_required_symbols_fetched_first(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    kline_calls: list[str] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            kline_calls.append(symbol)
            _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            _edge06_write_funding(tmp_path, symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT", "CUSDT", "ZUSDT"], now=now, lookback_days=40, max_workers=1,
        deadline_s=30.0, min_symbols=1, max_fail_fraction=0.15,
        collector=_Collector(), required_symbols=frozenset({"ZUSDT"}),
    )

    assert kline_calls == ["ZUSDT", "AUSDT", "BUSDT", "CUSDT"]
    assert report.ok is True


def test_edge06_klines_phase_precedes_funding_phase(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    events: list[tuple[str, str]] = []

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            events.append(("ohlcv", symbol))
            _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            events.append(("funding", symbol))
            _edge06_write_funding(tmp_path, symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT", "CUSDT"], now=now, lookback_days=40, max_workers=2,
        deadline_s=30.0, min_symbols=1, max_fail_fraction=0.15,
        collector=_Collector(),
    )

    ohlcv_idx = [i for i, (plane, _) in enumerate(events) if plane == "ohlcv"]
    funding_idx = [i for i, (plane, _) in enumerate(events) if plane == "funding"]
    assert len(ohlcv_idx) == 3 and len(funding_idx) == 3
    assert max(ohlcv_idx) < min(funding_idx)
    assert report.refreshed == 3


def test_edge06_funding_deadline_keeps_klines_intact(tmp_path, monkeypatch) -> None:
    import time

    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    symbols = [f"D{i}USDT" for i in range(5)]
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: clock["t"])

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            clock["t"] += 10.0
            _edge06_write_funding(tmp_path, symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=symbols, now=now, lookback_days=40, max_workers=1, deadline_s=25.0,
        min_symbols=1, max_fail_fraction=1.0, collector=_Collector(),
    )

    expected_tail = now.floor("h") - pd.Timedelta(hours=1)
    for sym in symbols:
        tail = data_refresh._disk_tail_ts(tmp_path, sym, now)
        assert tail is not None and tail >= expected_tail
    assert report.refreshed == 3
    assert report.incomplete == 2
    assert report.deadline_skipped == 0
    assert report.deadline_hit is True
    assert report.ok is True


def test_edge06_counts_partition_the_universe(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    _edge06_write_klines(tmp_path, "FRESHUSDT", now)
    _edge06_write_funding(tmp_path, "FRESHUSDT", now)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            if symbol == "FAILUSDT":
                raise RuntimeError("kline endpoint down")
            if symbol == "OKUSDT":
                _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            if symbol == "OKUSDT":
                _edge06_write_funding(tmp_path, symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["FRESHUSDT", "OKUSDT", "FAILUSDT", "EMPTYUSDT"], now=now, lookback_days=40,
        max_workers=1, deadline_s=30.0, min_symbols=1, max_fail_fraction=1.0,
        collector=_Collector(),
    )

    assert (report.fresh, report.refreshed, report.failed, report.incomplete) == (1, 1, 1, 1)
    assert report.deadline_skipped == 0
    assert report.fresh + report.refreshed + report.failed + report.deadline_skipped + report.incomplete == report.total == 4


def test_edge06_prefetch_leaves_nightly_funding_fresh(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    prefetch_now = pd.Timestamp("2026-09-01T20:15:00Z")
    nightly_now = pd.Timestamp("2026-09-01T23:03:00Z")
    symbols = ["A4HUSDT", "B8HUSDT"]
    intervals = {"A4HUSDT": 4 * 3_600_000, "B8HUSDT": 8 * 3_600_000}
    current = {"now": prefetch_now}
    funding_calls: list[str] = []
    ohlcv_calls: list[str] = []

    def _published(now, interval_ms: int) -> int:
        as_of_ms = int(now.value // 1_000_000) - 5 * 60_000
        return (as_of_ms // interval_ms) * interval_ms

    def _settle(symbol: str, now) -> None:
        import os

        interval_ms = intervals[symbol]
        window_start_ms = int((now - pd.Timedelta(days=40)).value // 1_000_000)
        first = (window_start_ms // interval_ms) * interval_ms
        stamps = list(range(first, _published(now, interval_ms) + 1, interval_ms))
        path = tmp_path / "funding" / f"{symbol}.parquet"
        if path.exists():
            old = pd.read_parquet(path, columns=["timestamp"])["timestamp"].tolist()
            stamps = sorted(set(stamps) | {int(v) for v in old})
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * len(stamps)}).to_parquet(path, index=False)

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            ohlcv_calls.append(symbol)
            _edge06_write_klines(tmp_path, symbol, current["now"])

        def ensure_funding_data(self, symbol, start, end):
            funding_calls.append(symbol)
            _settle(symbol, current["now"])

    prefetch = data_refresh.refresh_funding_tails(
        tmp_path, now=prefetch_now, lookback_days=40, symbols=symbols,
        deadline_s=60.0, max_workers=2, collector=_Collector(),
    )
    assert prefetch.fetched == 2
    assert prefetch.failed == 0

    current["now"] = nightly_now
    for sym in symbols:
        _edge06_write_klines(tmp_path, sym, nightly_now - pd.Timedelta(hours=1))
    funding_calls.clear()
    ohlcv_calls.clear()

    nightly = data_refresh.refresh_live_market_data(
        tmp_path, symbols=symbols, now=nightly_now, lookback_days=40, max_workers=2, deadline_s=60.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert funding_calls == []
    assert nightly.fresh == 2
    assert nightly.ok is True


def test_edge06_settings_refresh_and_venue_fields() -> None:
    import pytest

    from src.live.settings import LiveSettings, refresh_settings_fields

    settings = LiveSettings()
    assert settings.funding_prefetch_enabled is True
    assert settings.funding_prefetch_offset_hours == 20.25
    assert settings.refresh_decision_bar_max_missing_fraction == 0.05
    assert settings.venue_rules_warn_age_days == 2.0
    assert settings.venue_rules_max_age_days == 7.0
    assert settings.venue_rules_max_rejected_fraction == 0.05
    assert "refresh_decision_bar_max_missing_fraction" in refresh_settings_fields()

    with pytest.raises(Exception):
        LiveSettings(funding_prefetch_offset_hours=22.5)
    with pytest.raises(Exception):
        LiveSettings(funding_prefetch_offset_hours=0.0)
    with pytest.raises(Exception):
        LiveSettings(venue_rules_warn_age_days=7.0, venue_rules_max_age_days=7.0)
    with pytest.raises(Exception):
        LiveSettings(refresh_decision_bar_max_missing_fraction=1.5)


def test_edge06_prefetch_counts_fresh_deadline_and_blocked(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh
    from src.market_data.binance.futures import BinanceIpBlockedError

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    _edge06_write_funding(tmp_path, "FRESHUSDT", now)

    class _NeverCalled:
        def ensure_funding_data(self, symbol, start, end):
            raise AssertionError("fresh symbol must not be fetched")

    fresh_only = data_refresh.refresh_funding_tails(
        tmp_path, now=now, lookback_days=40, symbols=["FRESHUSDT"],
        deadline_s=60.0, max_workers=1, collector=_NeverCalled(),
    )
    assert (fresh_only.fresh, fresh_only.fetched, fresh_only.failed) == (1, 0, 0)

    class _BlockThenSkip:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def ensure_funding_data(self, symbol, start, end):
            self.calls.append(symbol)
            raise BinanceIpBlockedError(http_code=403, url="https://fapi.binance.com/fapi/v1/fundingRate")

    collector = _BlockThenSkip()
    blocked = data_refresh.refresh_funding_tails(
        tmp_path, now=now, lookback_days=40, symbols=["AUSDT", "BUSDT"],
        deadline_s=60.0, max_workers=1, collector=collector,
    )
    assert blocked.funding_blocked is True
    assert blocked.failed == 2
    assert collector.calls == ["AUSDT"]

    expired = data_refresh.refresh_funding_tails(
        tmp_path, now=now, lookback_days=40, symbols=["CUSDT"],
        deadline_s=0.0, max_workers=1, collector=_NeverCalled(),
    )
    assert expired.deadline_skipped == 1

    class _Flaky:
        def ensure_funding_data(self, symbol, start, end):
            raise RuntimeError("transport down")

    flaky = data_refresh.refresh_funding_tails(
        tmp_path, now=now, lookback_days=40, symbols=["DUSDT"],
        deadline_s=60.0, max_workers=1, collector=_Flaky(),
    )
    assert flaky.failed == 1
    assert flaky.funding_blocked is False


def test_edge06_prefetch_defaults_collector_without_network(tmp_path) -> None:
    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    _edge06_write_funding(tmp_path, "FRESHUSDT", now)
    report = data_refresh.refresh_funding_tails(
        tmp_path, now=now, lookback_days=40, symbols=["FRESHUSDT"],
        deadline_s=60.0, max_workers=1,
    )
    assert (report.fresh, report.fetched, report.failed) == (1, 0, 0)
    assert report.funding_blocked is False


def test_edge06_funding_phase_deadline_keeps_klines_current(tmp_path, monkeypatch) -> None:
    import time

    import pandas as pd

    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: clock["t"])

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            _edge06_write_klines(tmp_path, symbol, now)

        def ensure_funding_data(self, symbol, start, end):
            clock["t"] += 1000.0
            _edge06_write_funding(tmp_path, symbol, now)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=["AUSDT", "BUSDT"], now=now, lookback_days=40, max_workers=1,
        deadline_s=10.0, min_symbols=1, max_fail_fraction=0.0,
        collector=_Collector(),
    )
    assert report.deadline_hit is True
    assert report.incomplete >= 1
    assert report.ok is False


def test_refresh_reads_each_kline_file_at_most_before_and_after(tmp_path, monkeypatch) -> None:
    """디스크 tail은 갱신 전·후 각 한 번만 읽고, 이미 current인 심볼은 재읽기하지 않는다."""
    import collections

    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:30:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    current_tail = data_refresh._expected_kline_tail(now)
    stale_tail = now - pd.Timedelta(days=2)
    for sym, tail in (("CURUSDT", current_tail), ("OLDUSDT", stale_tail)):
        ts = [int((tail - pd.Timedelta(hours=h)).value // 10**6) for h in range(24)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 24, "volume": [1.0] * 24}).to_parquet(d / f"{sym}.parquet", index=False)
    reads: collections.Counter[str] = collections.Counter()
    real = data_refresh._kline_tail_state

    def _counting(root, sym, at):
        reads[sym] += 1
        return real(root, sym, at)

    monkeypatch.setattr(data_refresh, "_kline_tail_state", _counting)
    monkeypatch.setattr(data_refresh, "_funding_fresh_on_disk", lambda *a, **k: True)

    class _Collector:
        def ensure_ohlcv_data(self, *a, **k):
            return None

        def ensure_funding_data(self, *a, **k):
            return None

    data_refresh.refresh_live_market_data(
        tmp_path, symbols=["CURUSDT", "OLDUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=1.0, collector=_Collector(),
    )
    assert reads["CURUSDT"] == 1
    assert reads["OLDUSDT"] == 2


def test_kline_tail_state_treats_unreadable_or_timestampless_files_as_absent(tmp_path) -> None:
    """An unreadable parquet or one without valid timestamps has no tail and never counts as fresh."""
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    (d / "BADUSDT.parquet").write_bytes(b"not a parquet file")
    pd.DataFrame({"timestamp": [None, None], "close": [1.0, 1.0], "volume": [1.0, 1.0]}).to_parquet(
        d / "NOTSUSDT.parquet", index=False
    )

    assert data_refresh._kline_tail_state(tmp_path, "BADUSDT", now) == (None, False)
    assert data_refresh._kline_tail_state(tmp_path, "NOTSUSDT", now) == (None, False)


def test_market_data_staleness_hours_restricts_to_requested_symbols(tmp_path) -> None:
    """Staleness is computed only over the requested symbols; other files on disk are ignored."""
    import pandas as pd
    from src.live import data_refresh

    now = pd.Timestamp("2026-09-01T00:00:00Z")
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    for sym, lag_h in (("FRESHUSDT", 2), ("OLDUSDT", 30)):
        ts = [int((now - pd.Timedelta(hours=lag_h + k)).value // 10**6) for k in range(10)]
        pd.DataFrame({"timestamp": ts, "close": [1.0] * 10, "volume": [1.0] * 10}).to_parquet(d / f"{sym}.parquet", index=False)

    assert data_refresh.market_data_staleness_hours(tmp_path, now=now, symbols=["FRESHUSDT"]) == 2.0
    assert data_refresh.market_data_staleness_hours(tmp_path, now=now, symbols=["OLDUSDT"]) == 30.0
