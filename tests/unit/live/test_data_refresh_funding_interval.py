# ruff: noqa
def _qusdt_cadence_change_root(tmp_path, *, last_settlement="2026-11-03T20:00:00Z", symbol="QUSDT"):
    import pandas as pd

    now = pd.Timestamp("2026-11-03T23:00:30Z")
    ohlcv_dir = tmp_path / "ohlcv" / "1h"
    ohlcv_dir.mkdir(parents=True)
    funding_dir = tmp_path / "funding"
    funding_dir.mkdir(parents=True)
    tail = pd.Timestamp("2026-11-03T22:00:00Z")
    stamps = [int((tail - pd.Timedelta(hours=h)).value // 10**6) for h in range(48)]
    pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(ohlcv_dir / f"{symbol}.parquet", index=False)
    hourly = [pd.Timestamp(f"2026-11-03T{hour:02d}:00:00Z") for hour in range(10, 17)]
    series = [*hourly, pd.Timestamp(last_settlement)]
    pd.DataFrame({"timestamp": [int(t.value // 10**6) for t in series], "funding_rate": [0.0001] * len(series)}).to_parquet(
        funding_dir / f"{symbol}.parquet", index=False)
    return now


def _snapshot(captured_at, **hours):
    import pandas as pd
    from src.market_data.binance.funding_info import FundingIntervalSnapshot

    return FundingIntervalSnapshot(
        captured_at=pd.Timestamp(captured_at), interval_ms={sym: h * 3_600_000 for sym, h in hours.items()})


class _RecordingCollector:
    def __init__(self):
        self.funding_calls: list[tuple] = []

    def ensure_ohlcv_data(self, *a, **k):
        return None

    def ensure_funding_data(self, symbol, start, end, **kwargs):
        self.funding_calls.append((symbol, kwargs))


def _refresh_qusdt(tmp_path, now, collector, snapshot):
    from src.live import data_refresh

    return data_refresh.refresh_live_market_data(
        tmp_path, symbols=["QUSDT"], now=now, lookback_days=40, max_workers=1, deadline_s=30.0,
        min_symbols=1, max_fail_fraction=0.15, collector=collector,
        required_symbols=frozenset({"QUSDT"}), funding_intervals=snapshot,
    )


def test_interval_change_not_flagged_when_attested(tmp_path) -> None:
    now = _qusdt_cadence_change_root(tmp_path)
    collector = _RecordingCollector()

    report = _refresh_qusdt(tmp_path, now, collector, _snapshot(now, QUSDT=4))

    assert report.required_incomplete == ()
    assert report.ok is True
    assert report.fresh == 1
    assert collector.funding_calls == []


def test_without_snapshot_behaviour_is_stale_and_fetches(tmp_path) -> None:
    now = _qusdt_cadence_change_root(tmp_path)
    collector = _RecordingCollector()

    report = _refresh_qusdt(tmp_path, now, collector, None)

    assert report.required_incomplete == ("QUSDT",)
    assert report.ok is False
    assert collector.funding_calls == [("QUSDT", {})]


def test_unlisted_symbol_falls_back_to_inference_not_8h(tmp_path) -> None:
    now = _qusdt_cadence_change_root(tmp_path)
    collector = _RecordingCollector()

    report = _refresh_qusdt(tmp_path, now, collector, _snapshot(now, BTCUSDT=8))

    assert report.required_incomplete == ("QUSDT",)
    assert collector.funding_calls == [("QUSDT", {})]


def test_attested_interval_still_catches_real_gap(tmp_path) -> None:
    now = _qusdt_cadence_change_root(tmp_path)
    collector = _RecordingCollector()

    report = _refresh_qusdt(tmp_path, now, collector, _snapshot(now, QUSDT=1))

    assert report.required_incomplete == ("QUSDT",)
    assert collector.funding_calls == [("QUSDT", {"interval_ms": 3_600_000})]


def test_prefetch_uses_attested_interval(tmp_path) -> None:
    from src.live import data_refresh

    now = _qusdt_cadence_change_root(tmp_path)
    collector = _RecordingCollector()

    report = data_refresh.refresh_funding_tails(
        tmp_path, now=now, lookback_days=40, symbols=["QUSDT"], deadline_s=30.0, max_workers=1,
        collector=collector, funding_intervals=_snapshot(now, QUSDT=4),
    )

    assert report.fresh == 1
    assert report.fetched == 0
    assert collector.funding_calls == []
