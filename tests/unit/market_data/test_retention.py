def test_prune_market_data_shrinks_only_old_rows(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import prune_market_data

    now = pd.Timestamp("2026-09-01", tz="UTC")
    old = pd.date_range("2025-01-01", periods=100, freq="1D", tz="UTC")
    recent = pd.date_range("2026-06-01", periods=100, freq="1D", tz="UTC")
    idx = old.append(recent)
    df = pd.DataFrame({
        "timestamp": ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    df.to_parquet(d / "BTCUSDT.parquet", index=False)

    result = prune_market_data(tmp_path, 220, now=now)

    out = pd.read_parquet(d / "BTCUSDT.parquet")
    cutoff_ms = int((now - pd.Timedelta(days=220)).timestamp() * 1000)
    assert (out["timestamp"] >= cutoff_ms).all()
    assert len(out) == 100  # only the 'recent' block survives
    assert result["ohlcv/1h"]["files_pruned"] == 1
    assert result["ohlcv/1h"]["rows_removed"] == 100


def test_prune_market_data_rejects_below_floor_retention(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.market_data.retention import MARKET_DATA_MIN_RETENTION_DAYS, prune_market_data

    with pytest.raises(ValueError, match=r"data_retention_days|retention"):
        prune_market_data(tmp_path, MARKET_DATA_MIN_RETENTION_DAYS - 1, now=pd.Timestamp("2026-09-01", tz="UTC"))


def test_prune_market_data_skips_non_integer_timestamp_file(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import prune_market_data

    d = tmp_path / "funding"
    d.mkdir(parents=True)
    p = d / "WEIRD.parquet"
    pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=50, freq="1D", tz="UTC"), "funding_rate": 0.0001}).to_parquet(p, index=False)
    before = p.read_bytes()

    result = prune_market_data(tmp_path, 220, now=pd.Timestamp("2026-09-01", tz="UTC"))

    assert p.read_bytes() == before
    assert result["funding"]["files_skipped"] == 1
    assert result["funding"]["files_pruned"] == 0


def test_prune_market_data_never_writes_empty_frame(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import prune_market_data

    idx = pd.date_range("2020-01-01", periods=80, freq="1D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    p = d / "DEADUSDT.parquet"
    df.to_parquet(p, index=False)
    before = p.read_bytes()

    result = prune_market_data(tmp_path, 220, now=pd.Timestamp("2026-09-01", tz="UTC"))

    assert p.exists()
    assert p.read_bytes() == before
    assert result["ohlcv/1h"]["files_pruned"] == 0
    assert result["ohlcv/1h"]["files_skipped"] == 1


def test_prune_market_data_noop_when_all_rows_recent(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import prune_market_data

    idx = pd.date_range("2026-07-01", periods=60, freq="1D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    d = tmp_path / "markPriceKlines" / "1h"
    d.mkdir(parents=True)
    p = d / "ETHUSDT.parquet"
    df.to_parquet(p, index=False)
    before = p.read_bytes()

    result = prune_market_data(tmp_path, 220, now=pd.Timestamp("2026-09-01", tz="UTC"))

    assert p.read_bytes() == before
    assert result["markPriceKlines/1h"]["files_pruned"] == 0


def test_prune_orderbook_history_removes_only_old_dailies(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import prune_orderbook_history

    for tag in ("20250101", "20250601", "20260815", "20260830"):
        (tmp_path / f"live_orderbook_{tag}.parquet").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"keep")

    removed = prune_orderbook_history(tmp_path, 365, now=pd.Timestamp("2026-09-01", tz="UTC"))

    assert removed == 2
    remaining = sorted(p.name for p in tmp_path.glob("live_orderbook_*.parquet"))
    assert remaining == ["live_orderbook_20260815.parquet", "live_orderbook_20260830.parquet"]
    assert (tmp_path / "notes.txt").exists()


def test_check_orderbook_prune_impending_detects_warning(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import check_orderbook_prune_impending

    # now: 2026-09-01, retention 365 days -> expiry of 2025-09-05 is 2026-09-05 (4 days left)
    (tmp_path / "live_orderbook_20250905.parquet").write_bytes(b"x")
    (tmp_path / "live_orderbook_20260801.parquet").write_bytes(b"x")

    impending, days_left, earliest = check_orderbook_prune_impending(
        tmp_path, 365, now=pd.Timestamp("2026-09-01", tz="UTC"), warning_days=7
    )
    assert impending is True
    assert days_left == 4
    assert earliest == "2025-09-05"


def test_check_orderbook_prune_impending_false_when_safe(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import check_orderbook_prune_impending

    # now: 2026-09-01, retention 365 days -> expiry of 2026-01-01 is 2027-01-01 (122 days left)
    (tmp_path / "live_orderbook_20260101.parquet").write_bytes(b"x")

    impending, days_left, earliest = check_orderbook_prune_impending(
        tmp_path, 365, now=pd.Timestamp("2026-09-01", tz="UTC"), warning_days=7
    )
    assert impending is False
    assert days_left > 7
    assert earliest == "2026-01-01"


def test_check_orderbook_prune_impending_empty_dir(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import check_orderbook_prune_impending

    impending, days_left, earliest = check_orderbook_prune_impending(
        tmp_path, 365, now=pd.Timestamp("2026-09-01", tz="UTC"), warning_days=7
    )
    assert impending is False
    assert days_left == 0
    assert earliest is None
