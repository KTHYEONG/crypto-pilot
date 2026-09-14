# ruff: noqa


def test_repair_ohlcv_file_leaves_healthy_file_untouched(tmp_path) -> None:
    import pandas as pd
    from src.live.data_repair import REPAIR_REQUIRED_COLUMNS, repair_ohlcv_file
    from src.market_data.storage.ohlcv import write_ohlcv

    now = pd.Timestamp("2026-09-14T03:04:05Z")
    target = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    target.parent.mkdir(parents=True)
    idx = pd.date_range("2026-09-13", periods=3, freq="1h", tz="UTC")
    valid = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in REPAIR_REQUIRED_COLUMNS})
    valid["timestamp"] = (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    class _Collector:
        def __init__(self, frame):
            self.frame = frame
            self.calls: list[tuple] = []

        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            self.calls.append((symbol, timeframe, start, end))
            if self.frame is not None:
                write_ohlcv(target, self.frame, timeframe=timeframe)

    write_ohlcv(target, valid, timeframe="1h")
    before = target.read_bytes()
    collector = _Collector(valid)

    result = repair_ohlcv_file("XUSDT", now=now, lookback_days=430, futures_root=tmp_path, collector=collector)

    assert result.status == "healthy"
    assert result.moved_to is None
    assert result.rows == 3
    assert collector.calls == []
    assert target.read_bytes() == before


def test_repair_ohlcv_file_moves_corrupt_file_aside_and_refetches_full_window(tmp_path) -> None:
    import pandas as pd
    from src.live.data_repair import REPAIR_REQUIRED_COLUMNS, repair_ohlcv_file
    from src.market_data.storage.ohlcv import write_ohlcv

    now = pd.Timestamp("2026-09-14T03:04:05Z")
    target = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    target.parent.mkdir(parents=True)
    idx = pd.date_range("2026-09-13", periods=3, freq="1h", tz="UTC")
    valid = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in REPAIR_REQUIRED_COLUMNS})
    valid["timestamp"] = (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    class _Collector:
        def __init__(self, frame):
            self.frame = frame
            self.calls: list[tuple] = []

        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            self.calls.append((symbol, timeframe, start, end))
            if self.frame is not None:
                write_ohlcv(target, self.frame, timeframe=timeframe)

    target.write_bytes(b"not a parquet")
    collector = _Collector(valid)

    result = repair_ohlcv_file("XUSDT", now=now, lookback_days=430, futures_root=tmp_path, collector=collector)

    moved = target.parent / "XUSDT.corrupt-20260914T030405Z"
    assert result.status == "repaired"
    assert result.moved_to == moved
    assert moved.read_bytes() == b"not a parquet"
    assert result.rows == 3
    assert collector.calls == [("XUSDT", "1h", str(now - pd.Timedelta(days=430)), str(now))]
    assert not moved.name.endswith(".parquet")


def test_repair_ohlcv_file_refetches_missing_file(tmp_path) -> None:
    import pandas as pd
    from src.live.data_repair import REPAIR_REQUIRED_COLUMNS, repair_ohlcv_file
    from src.market_data.storage.ohlcv import write_ohlcv

    now = pd.Timestamp("2026-09-14T03:04:05Z")
    target = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    target.parent.mkdir(parents=True)
    idx = pd.date_range("2026-09-13", periods=3, freq="1h", tz="UTC")
    valid = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in REPAIR_REQUIRED_COLUMNS})
    valid["timestamp"] = (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    class _Collector:
        def __init__(self, frame):
            self.frame = frame
            self.calls: list[tuple] = []

        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            self.calls.append((symbol, timeframe, start, end))
            if self.frame is not None:
                write_ohlcv(target, self.frame, timeframe=timeframe)

    collector = _Collector(valid)

    result = repair_ohlcv_file("XUSDT", now=now, lookback_days=30, futures_root=tmp_path, collector=collector)

    assert result.status == "repaired"
    assert result.moved_to is None
    assert len(collector.calls) == 1


def test_repair_ohlcv_file_fails_closed_when_refetch_lacks_required_columns(tmp_path) -> None:
    import pandas as pd
    from src.live.data_repair import REPAIR_REQUIRED_COLUMNS, repair_ohlcv_file
    from src.market_data.storage.ohlcv import write_ohlcv

    now = pd.Timestamp("2026-09-14T03:04:05Z")
    target = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    target.parent.mkdir(parents=True)
    idx = pd.date_range("2026-09-13", periods=3, freq="1h", tz="UTC")
    valid = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in REPAIR_REQUIRED_COLUMNS})
    valid["timestamp"] = (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    class _Collector:
        def __init__(self, frame):
            self.frame = frame
            self.calls: list[tuple] = []

        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            self.calls.append((symbol, timeframe, start, end))
            if self.frame is not None:
                write_ohlcv(target, self.frame, timeframe=timeframe)

    import pytest
    from src.common.errors import DataIntegrityError

    target.write_bytes(b"not a parquet")
    collector = _Collector(valid.drop(columns=["taker_buy_quote"]))

    with pytest.raises(DataIntegrityError, match="ohlcv repair did not produce a valid file"):
        repair_ohlcv_file("XUSDT", now=now, lookback_days=430, futures_root=tmp_path, collector=collector)

    assert (target.parent / "XUSDT.corrupt-20260914T030405Z").read_bytes() == b"not a parquet"


def test_repair_ohlcv_file_treats_empty_file_as_invalid(tmp_path) -> None:
    import pandas as pd
    from src.live.data_repair import REPAIR_REQUIRED_COLUMNS, repair_ohlcv_file
    from src.market_data.storage.ohlcv import write_ohlcv

    now = pd.Timestamp("2026-09-14T03:04:05Z")
    target = tmp_path / "ohlcv" / "1h" / "XUSDT.parquet"
    target.parent.mkdir(parents=True)
    idx = pd.date_range("2026-09-13", periods=3, freq="1h", tz="UTC")
    valid = pd.DataFrame({column: [1.0, 1.0, 1.0] for column in REPAIR_REQUIRED_COLUMNS})
    valid["timestamp"] = (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")

    class _Collector:
        def __init__(self, frame):
            self.frame = frame
            self.calls: list[tuple] = []

        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            self.calls.append((symbol, timeframe, start, end))
            if self.frame is not None:
                write_ohlcv(target, self.frame, timeframe=timeframe)

    valid.head(0).to_parquet(target, index=False)
    collector = _Collector(valid)

    result = repair_ohlcv_file("XUSDT", now=now, lookback_days=430, futures_root=tmp_path, collector=collector)

    assert result.status == "repaired"
    assert result.moved_to == target.parent / "XUSDT.corrupt-20260914T030405Z"
    assert result.rows == 3
