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

    result = prune_market_data(tmp_path, 450, now=now)

    out = pd.read_parquet(d / "BTCUSDT.parquet")
    cutoff_ms = int((now - pd.Timedelta(days=450)).timestamp() * 1000)
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

    result = prune_market_data(tmp_path, 450, now=pd.Timestamp("2026-09-01", tz="UTC"))

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

    result = prune_market_data(tmp_path, 450, now=pd.Timestamp("2026-09-01", tz="UTC"))

    assert p.exists()
    assert p.read_bytes() == before
    assert result["ohlcv/1h"]["files_pruned"] == 0
    assert result["ohlcv/1h"]["files_skipped"] == 1


def test_prune_market_data_leaves_retired_mark_cache_untouched(tmp_path) -> None:
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

    result = prune_market_data(tmp_path, 450, now=pd.Timestamp("2026-09-01", tz="UTC"))

    assert p.read_bytes() == before
    assert result["markPriceKlines/1h"] == {"files_pruned": 0, "rows_removed": 0, "files_skipped": 0}


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

def test_prune_market_data_never_prunes_retired_metrics_feed(tmp_path) -> None:
    import pandas as pd
    from src.market_data.retention import prune_market_data

    now = pd.Timestamp("2026-09-01", tz="UTC")
    old = pd.date_range("2025-01-01", periods=50, freq="1D", tz="UTC")
    recent = pd.date_range("2026-06-01", periods=50, freq="1D", tz="UTC")
    idx = old.append(recent)
    df = pd.DataFrame({
        "timestamp": ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64"),
        "sum_open_interest": 1.0,
        "sum_open_interest_value": 1.0,
        "long_short_ratio": 1.0,
        "top_trader_long_short_ratio": 1.0,
        "sum_taker_long_short_vol_ratio": 1.0,
        "datetime": idx,
        "available_at": idx + pd.Timedelta(minutes=5),
        "symbol": "AAAUSDT",
    })
    d = tmp_path / "metrics" / "1d"
    d.mkdir(parents=True)
    df.to_parquet(d / "AAAUSDT.parquet", index=False)
    before = (d / "AAAUSDT.parquet").read_bytes()

    result = prune_market_data(tmp_path, 450, now=now)

    # Retired feed: ordinary MHS retention leaves every row untouched.
    assert (d / "AAAUSDT.parquet").read_bytes() == before
    assert result["metrics/1d"] == {"files_pruned": 0, "rows_removed": 0, "files_skipped": 0}


def test_quarantine_retired_feeds_dry_run_enumerates_exact_targets(tmp_path) -> None:
    import json

    from src.market_data.retention import quarantine_retired_mhs_feeds

    (tmp_path / "markPriceKlines" / "1h").mkdir(parents=True)
    (tmp_path / "metrics" / "1d").mkdir(parents=True)
    (tmp_path / "ohlcv" / "1h").mkdir(parents=True)
    (tmp_path / "funding").mkdir(parents=True)
    (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").write_bytes(b"a" * 10)
    (tmp_path / "markPriceKlines" / "1h" / "AUSDT.coverage.json").write_bytes(b"c" * 5)
    (tmp_path / "metrics" / "1d" / "BUSDT.parquet").write_bytes(b"b" * 7)
    (tmp_path / "markPriceKlines" / "1h" / "X.tmp.parquet").write_bytes(b"temp")
    (tmp_path / "ohlcv" / "1h" / "AUSDT.parquet").write_bytes(b"keep-1h")
    (tmp_path / "funding" / "AUSDT.parquet").write_bytes(b"keep-funding")

    manifest = tmp_path / "manifest.json"
    report = quarantine_retired_mhs_feeds(
        tmp_path, tmp_path / "recovery", dry_run=True, manifest_path=manifest
    )

    # Exact deletion targets only; temp artifacts and live feeds excluded.
    assert report["targets"] == 3
    assert report["moved"] == 0
    assert report["bytes"] == 22
    assert report["recovery_dir"] == str(tmp_path / "recovery")
    assert (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry["relative_path"] for entry in payload["files"]] == [
        "markPriceKlines/1h/AUSDT.coverage.json",
        "markPriceKlines/1h/AUSDT.parquet",
        "metrics/1d/BUSDT.parquet",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in payload["files"])
    assert payload["moved"] is False


def test_quarantine_retired_feeds_blocked_by_active_readers(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.market_data.retention import quarantine_retired_mhs_feeds, retired_feed_active_readers

    (tmp_path / "markPriceKlines" / "1h").mkdir(parents=True)
    (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").write_bytes(b"a")

    # The opt-in parity gate and live decision-mark recording still consume
    # mark files, so physical removal stays blocked without operator override.
    assert len(retired_feed_active_readers()) > 0
    with pytest.raises(DataIntegrityError, match="active readers"):
        quarantine_retired_mhs_feeds(tmp_path, tmp_path / "recovery", dry_run=False)
    assert (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").exists()


def test_quarantine_retired_feeds_apply_moves_with_recoverable_manifest(tmp_path) -> None:
    import json

    from src.market_data.retention import quarantine_retired_mhs_feeds

    (tmp_path / "markPriceKlines" / "1h").mkdir(parents=True)
    (tmp_path / "metrics" / "1d").mkdir(parents=True)
    (tmp_path / "ohlcv" / "3m").mkdir(parents=True)
    (tmp_path / "mhs_execution").mkdir(parents=True)
    (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").write_bytes(b"a" * 4)
    (tmp_path / "metrics" / "1d" / "BUSDT.parquet").write_bytes(b"b" * 6)
    (tmp_path / "ohlcv" / "3m" / "AUSDT.parquet").write_bytes(b"keep-3m")
    (tmp_path / "mhs_execution" / "input_manifest.json").write_bytes(b"keep-run")

    report = quarantine_retired_mhs_feeds(
        tmp_path, tmp_path / "recovery", dry_run=False, allow_active_readers=True
    )

    assert report == {
        "targets": 2,
        "moved": 2,
        "bytes": 10,
        "recovery_dir": str(tmp_path / "recovery"),
        "manifest": str(tmp_path / "recovery" / "retired_feeds_manifest.json"),
    }
    assert (tmp_path / "recovery" / "markPriceKlines" / "1h" / "AUSDT.parquet").read_bytes() == b"a" * 4
    assert (tmp_path / "recovery" / "metrics" / "1d" / "BUSDT.parquet").read_bytes() == b"b" * 6
    assert not (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").exists()
    # 3m execution corpus, backtest runs and unrelated data are never touched.
    assert (tmp_path / "ohlcv" / "3m" / "AUSDT.parquet").read_bytes() == b"keep-3m"
    assert (tmp_path / "mhs_execution" / "input_manifest.json").read_bytes() == b"keep-run"
    payload = json.loads((tmp_path / "recovery" / "retired_feeds_manifest.json").read_text(encoding="utf-8"))
    assert payload["moved"] is True
    assert len(payload["files"]) == 2


def test_prune_market_data_uses_hidden_temp_and_skips_legacy_tmp_files(tmp_path, monkeypatch) -> None:
    import os
    import pathlib
    import threading
    import pandas as pd
    from src.market_data.retention import prune_market_data

    now = pd.Timestamp("2026-09-01", tz="UTC")
    idx = pd.date_range("2025-01-01", periods=100, freq="1D", tz="UTC").append(
        pd.date_range("2026-06-01", periods=100, freq="1D", tz="UTC")
    )
    df = pd.DataFrame({
        "timestamp": ((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    d = tmp_path / "ohlcv" / "1h"
    d.mkdir(parents=True)
    df.to_parquet(d / "BTCUSDT.parquet", index=False)
    df.to_parquet(d / "BTCUSDT.tmp.parquet", index=False)
    leftover_before = (d / "BTCUSDT.tmp.parquet").read_bytes()

    replaced: list[str] = []
    original_replace = pathlib.Path.replace

    def _spy(self, target):
        replaced.append(self.name)
        return original_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", _spy)

    result = prune_market_data(tmp_path, 450, now=now)

    assert replaced == [f".BTCUSDT.parquet.{os.getpid()}.{threading.get_ident()}.prune.tmp"]
    assert (d / "BTCUSDT.tmp.parquet").read_bytes() == leftover_before
    assert result["ohlcv/1h"]["files_pruned"] == 1

