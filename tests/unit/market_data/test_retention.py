import json
import os
from pathlib import Path

import pandas as pd

from src.market_data.streams.normalizer import NormalizerConfig
from src.market_data.streams.retention import (
    BackupStatus,
    prune_backed_up,
    read_backup_status,
    sweep_partials,
)


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
    from src.market_data.retention import quarantine_retired_mhs_feeds, retired_feed_active_readers

    (tmp_path / "markPriceKlines" / "1h").mkdir(parents=True)
    (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").write_bytes(b"a")

    # The OHLCV-only contract removed the last mark reader, so physical
    # removal proceeds without operator override.
    assert retired_feed_active_readers() == ()
    report = quarantine_retired_mhs_feeds(tmp_path, tmp_path / "recovery", dry_run=False)
    assert report["moved"] == 1
    assert not (tmp_path / "markPriceKlines" / "1h" / "AUSDT.parquet").exists()


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



def _norm_now() -> pd.Timestamp:
    return pd.Timestamp("2026-09-26T12:00:00Z")


def _norm_status(started_days_ago: float = 1.0, finished_days_ago: float = 0.5) -> BackupStatus:
    now = _norm_now()
    return BackupStatus(
        started_at=now - pd.Timedelta(days=started_days_ago),
        finished_at=now - pd.Timedelta(days=finished_days_ago),
    )


def _touch_old(path: Path, days_ago: float = 40.0) -> None:
    old = _norm_now().value // 1_000_000_000 - int(days_ago * 86400)
    os.utime(path, (old, old))


def _touch_now(path: Path) -> None:
    fresh = _norm_now().value // 1_000_000_000
    os.utime(path, (fresh, fresh))


def _archive_pair(root: Path, stream: str, day: str, days_ago: float = 40.0) -> None:
    stream_dir = root / "raw" / "archive" / stream
    stream_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{day}.jsonl.xz", f"{day}.manifest.json"):
        (stream_dir / name).write_bytes(b"x")
        _touch_old(stream_dir / name, days_ago)


def test_backup_gated_prune_removes_old_pair_keeps_recent(tmp_path, monkeypatch) -> None:
    """A 31-day archive pair with a covering status is removed manifest-last; 29-day kept."""
    _archive_pair(tmp_path, "book_ticker", "20260826")
    _archive_pair(tmp_path, "book_ticker", "20260828")
    removed: list[str] = []
    real_unlink = Path.unlink

    def recording_unlink(self: Path, *args: object, **kwargs: object) -> None:
        removed.append(str(self))
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    report = prune_backed_up(
        tmp_path, tmp_path / "liq", status=_norm_status(), config=NormalizerConfig(), now=_norm_now()
    )
    assert report.prune_blocked is False
    assert not (tmp_path / "raw" / "archive" / "book_ticker" / "20260826.jsonl.xz").exists()
    assert not (tmp_path / "raw" / "archive" / "book_ticker" / "20260826.manifest.json").exists()
    assert (tmp_path / "raw" / "archive" / "book_ticker" / "20260828.jsonl.xz").exists()
    archive_pos = next(i for i, p in enumerate(removed) if p.endswith("20260826.jsonl.xz"))
    manifest_pos = next(i for i, p in enumerate(removed) if p.endswith("20260826.manifest.json"))
    assert archive_pos < manifest_pos


def test_prune_keeps_unit_modified_after_backup_start(tmp_path) -> None:
    """An eligible day touched after the backup started is kept without blocking."""

    _archive_pair(tmp_path, "book_ticker", "20260826")
    fresh = tmp_path / "raw" / "archive" / "book_ticker" / "20260826.jsonl.xz"
    now = _norm_now()
    os.utime(fresh, (now.value // 1_000_000_000, now.value // 1_000_000_000))

    report = prune_backed_up(
        tmp_path, tmp_path / "liq", status=_norm_status(), config=NormalizerConfig(), now=_norm_now()
    )
    assert report.prune_blocked is False
    assert fresh.exists()


def test_prune_blocked_without_fresh_successful_status(tmp_path) -> None:
    """Missing, stale or failed backup evidence blocks every deletion with its reason."""


    _archive_pair(tmp_path, "book_ticker", "20260826")
    config = NormalizerConfig()
    missing = prune_backed_up(tmp_path, tmp_path / "liq", status=None, config=config, now=_norm_now())
    assert (missing.prune_blocked, missing.blocked_reason) == (True, "status_missing")
    stale = prune_backed_up(
        tmp_path, tmp_path / "liq", status=_norm_status(started_days_ago=5.0, finished_days_ago=4.0),
        config=config, now=_norm_now(),
    )
    assert (stale.prune_blocked, stale.blocked_reason) == (True, "status_stale")
    bad = tmp_path / "last_success.json"
    bad.write_text(json.dumps({"started_at": "2026-09-26T00:00:00+00:00",
                               "finished_at": "2026-09-26T01:00:00+00:00", "rc": 1}))
    assert read_backup_status(bad) is None
    invalid = prune_backed_up(tmp_path, tmp_path / "liq", status=None, config=config, now=_norm_now(),
                              backup_status_path=bad)
    assert (invalid.prune_blocked, invalid.blocked_reason) == (True, "status_invalid")
    assert (tmp_path / "raw" / "archive" / "book_ticker" / "20260826.jsonl.xz").exists()


def test_parquet_partitions_follow_own_window(tmp_path) -> None:
    """181-day snapshot and liquidation partitions prune; 179-day stays."""

    old_day, keep_day = "20260328", "20260330"
    old_dir = tmp_path / "book_ticker" / old_day
    old_dir.mkdir(parents=True)
    (old_dir / "10.parquet").write_bytes(b"p")
    _touch_old(old_dir / "10.parquet")
    keep_dir = tmp_path / "book_ticker" / keep_day
    keep_dir.mkdir(parents=True)
    (keep_dir / "10.parquet").write_bytes(b"p")
    _touch_old(keep_dir / "10.parquet")
    liq = tmp_path / "liq"
    liq.mkdir()
    (liq / f"liquidations_{old_day}_10.parquet").write_bytes(b"p")
    _touch_old(liq / f"liquidations_{old_day}_10.parquet")
    young_liq = liq / "liquidations_20260330_10.parquet"
    young_liq.write_bytes(b"p")
    _touch_old(young_liq)
    fresh_liq = liq / "liquidations_20260327_10.parquet"
    fresh_liq.write_bytes(b"p")
    report = prune_backed_up(
        tmp_path, liq, status=_norm_status(), config=NormalizerConfig(), now=_norm_now()
    )
    assert report.prune_blocked is False
    assert not (old_dir / "10.parquet").exists()
    assert (keep_dir / "10.parquet").exists()
    assert not (liq / f"liquidations_{old_day}_10.parquet").exists()
    assert young_liq.exists()
    assert fresh_liq.exists()


def test_protected_trees_never_touched(tmp_path) -> None:
    """Coverage, reference, quarantine, exec_depth and heartbeat files survive pruning."""

    keep = [
        tmp_path / "coverage" / "liquidations" / "20200101.jsonl",
        tmp_path / "reference" / "exchange_info" / "20200101.json.gz",
        tmp_path / "quarantine" / "x.parquet",
        tmp_path / "exec_depth" / "y.parquet",
        tmp_path / "recorder_heartbeat.json",
        tmp_path / "raw" / "capture_blue.json",
        tmp_path / "raw" / "normalizer_checkpoint.json",
    ]
    for path in keep:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
        _touch_old(path, days_ago=400.0)
    _archive_pair(tmp_path, "book_ticker", "20260826")
    report = prune_backed_up(
        tmp_path, tmp_path / "liq", status=_norm_status(), config=NormalizerConfig(), now=_norm_now()
    )
    assert report.prune_blocked is False
    assert all(p.exists() for p in keep)


def test_sweep_removes_only_old_temps(tmp_path) -> None:
    """Old partials and parquet tmps are swept; fresh ones survive."""

    raw = tmp_path / "raw"
    raw.mkdir()
    old_partial = raw / "x.partial"
    old_partial.write_bytes(b"o")
    _touch_old(old_partial, days_ago=2.0)
    fresh_partial = raw / "y.partial"
    fresh_partial.write_bytes(b"f")
    _touch_now(fresh_partial)
    book = tmp_path / "book_ticker" / "20260926"
    book.mkdir(parents=True)
    old_tmp = book / ".10.parquet.1.2.tmp"
    old_tmp.write_bytes(b"o")
    _touch_old(old_tmp, days_ago=2.0)
    fresh_tmp = book / ".11.parquet.1.2.tmp"
    fresh_tmp.write_bytes(b"f")
    _touch_now(fresh_tmp)
    swept = sweep_partials(tmp_path, tmp_path / "liq", now=_norm_now(), max_age_s=3600.0)
    assert swept == 2
    assert not old_partial.exists()
    assert fresh_partial.exists()
    assert not old_tmp.exists()
    assert fresh_tmp.exists()


def test_empty_dirs_removed_roots_kept(tmp_path) -> None:
    """Empty day dirs vanish; the dataset roots themselves remain."""

    day_dir = tmp_path / "raw" / "hot" / "book_ticker" / "20260901"
    day_dir.mkdir(parents=True)
    book_root = tmp_path / "book_ticker"
    book_root.mkdir(parents=True)
    report = prune_backed_up(
        tmp_path, tmp_path / "liq", status=_norm_status(), config=NormalizerConfig(), now=_norm_now()
    )
    assert report.prune_blocked is False
    assert not day_dir.exists()
    assert book_root.exists()


def test_read_backup_status_variants(tmp_path) -> None:
    """Absent, corrupt, failed-rc and naive timestamps all read as None."""
    assert read_backup_status(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert read_backup_status(bad) is None
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[1]")
    assert read_backup_status(scalar) is None
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"started_at": "2026-09-26T00:00:00+00:00",
                                  "finished_at": "2026-09-26T01:00:00+00:00", "rc": 2}))
    assert read_backup_status(failed) is None
    naive = tmp_path / "naive.json"
    naive.write_text(json.dumps({"started_at": "2026-09-26T00:00:00",
                                 "finished_at": "2026-09-26T01:00:00", "rc": 0}))
    assert read_backup_status(naive) is None
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"started_at": "2026-09-26T00:00:00+00:00",
                              "finished_at": "2026-09-26T01:00:00+00:00", "rc": 0}))
    status = read_backup_status(ok)
    assert status is not None
    assert status.finished_at > status.started_at


def test_sweep_skips_missing_trees_and_failures(tmp_path, monkeypatch) -> None:
    """Absent trees sweep zero; unlink races are skipped silently."""
    assert sweep_partials(tmp_path, tmp_path / "liq", now=_norm_now(), max_age_s=3600.0) == 0
    raw = tmp_path / "raw"
    raw.mkdir()
    doomed = raw / "z.partial"
    doomed.write_bytes(b"x")
    _touch_old(doomed, days_ago=2.0)

    def _fail_unlink(self, *args, **kwargs) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", _fail_unlink)
    assert sweep_partials(tmp_path, tmp_path / "liq", now=_norm_now(), max_age_s=3600.0) == 0
    assert doomed.exists()


def test_prune_skips_foreign_names_and_missing_files(tmp_path) -> None:
    """Non-day archive names and vanished files never break the pass."""
    stream_dir = tmp_path / "raw" / "archive" / "book_ticker"
    stream_dir.mkdir(parents=True)
    (stream_dir / "notes.txt").write_bytes(b"x")
    (stream_dir / "2026-09-20.jsonl.xz").write_bytes(b"x")
    liq = tmp_path / "liq"
    liq.mkdir()
    (liq / "scratch.parquet").write_bytes(b"x")
    (liq / "liquidations_20200101_10.parquet").write_bytes(b"p")
    _touch_old(liq / "liquidations_20200101_10.parquet", days_ago=400.0)
    report = prune_backed_up(tmp_path, liq, status=_norm_status(), config=NormalizerConfig(), now=_norm_now())
    assert report.prune_blocked is False
    assert (stream_dir / "notes.txt").exists()
    assert (stream_dir / "2026-09-20.jsonl.xz").exists()
    assert (liq / "scratch.parquet").exists()
    assert not (liq / "liquidations_20200101_10.parquet").exists()


def test_prune_unlink_failures_are_counted_not_raised(tmp_path, monkeypatch) -> None:
    """A locked file keeps the unit alive without failing the pass."""
    _archive_pair(tmp_path, "book_ticker", "20260826")
    real_unlink = Path.unlink
    calls = {"n": 0}

    def _flaky_unlink(self, *args, **kwargs) -> None:
        if self.name.endswith(".jsonl.xz") and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("locked")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    report = prune_backed_up(tmp_path, tmp_path / "liq", status=_norm_status(),
                             config=NormalizerConfig(), now=_norm_now())
    assert report.prune_blocked is False
    assert (tmp_path / "raw" / "archive" / "book_ticker" / "20260826.manifest.json").exists()


def test_status_key_and_type_errors_read_none(tmp_path) -> None:
    """Timestamp-less or mistyped status files read as None."""
    bad = tmp_path / "keys.json"
    bad.write_text(json.dumps({"rc": 0}))
    assert read_backup_status(bad) is None
    mistyped = tmp_path / "types.json"
    mistyped.write_text(json.dumps({"started_at": 123, "finished_at": [], "rc": 0}))
    assert read_backup_status(mistyped) is None


def test_older_than_false_on_vanished_paths(tmp_path) -> None:
    """Vanished paths are never older than anything."""
    from src.market_data.streams.retention import _older_than

    assert _older_than(tmp_path / "missing", 10**20) is False


def test_sweep_unlink_failures_skipped(tmp_path, monkeypatch) -> None:
    """A locked temp file is skipped without failing the sweep."""
    raw = tmp_path / "raw"
    raw.mkdir()
    doomed = raw / "z.partial"
    doomed.write_bytes(b"x")
    _touch_old(doomed, days_ago=2.0)
    book = tmp_path / "book_ticker" / "20260926"
    book.mkdir(parents=True)
    doomed_tmp = book / ".10.parquet.1.2.tmp"
    doomed_tmp.write_bytes(b"x")
    _touch_old(doomed_tmp, days_ago=2.0)

    def _flaky_unlink(self, *args, **kwargs) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    assert sweep_partials(tmp_path, tmp_path / "liq", now=_norm_now(), max_age_s=3600.0) == 0
    assert doomed.exists()
    assert doomed_tmp.exists()
    monkeypatch.undo()
    assert sweep_partials(tmp_path, tmp_path / "liq", now=_norm_now(), max_age_s=3600.0) == 2


def test_prune_skips_foreign_day_names(tmp_path) -> None:
    """Non-day archive and snapshot names are never parsed as days."""
    stream_dir = tmp_path / "raw" / "archive" / "book_ticker"
    stream_dir.mkdir(parents=True)
    (stream_dir / "latest.jsonl.xz").write_bytes(b"x")
    snap_base = tmp_path / "book_ticker"
    (snap_base / "latest").mkdir(parents=True)
    (snap_base / "latest" / "10.parquet").write_bytes(b"x")
    liq = tmp_path / "liq"
    liq.mkdir()
    (liq / "notes.txt").write_bytes(b"x")
    (liq / "liquidations_latest.parquet").write_bytes(b"x")
    report = prune_backed_up(tmp_path, liq, status=_norm_status(), config=NormalizerConfig(), now=_norm_now())
    assert report.prune_blocked is False
    assert (stream_dir / "latest.jsonl.xz").exists()
    assert (snap_base / "latest").exists()


def test_backed_up_check_false_on_vanished_files(tmp_path) -> None:
    """A unit whose files vanish mid-check is kept, never pruned."""
    from src.market_data.streams.retention import _unit_backed_up

    assert _unit_backed_up([tmp_path / "missing"], 0) is False


def test_snapshot_and_liquidation_unlink_failures_kept(tmp_path, monkeypatch) -> None:
    """Locked derived files keep their day units without failing the pass."""
    old_day = "20260328"
    old_dir = tmp_path / "book_ticker" / old_day
    old_dir.mkdir(parents=True)
    (old_dir / "10.parquet").write_bytes(b"p")
    _touch_old(old_dir / "10.parquet")
    liq = tmp_path / "liq"
    liq.mkdir()
    liq_file = liq / f"liquidations_{old_day}_10.parquet"
    liq_file.write_bytes(b"p")
    _touch_old(liq_file)
    real_unlink = Path.unlink

    def _flaky_unlink(self, *args, **kwargs) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", _flaky_unlink)
    report = prune_backed_up(tmp_path, liq, status=_norm_status(), config=NormalizerConfig(), now=_norm_now())
    assert report.prune_blocked is False
    assert (old_dir / "10.parquet").exists()
    assert liq_file.exists()


def test_rmdir_failures_skipped(tmp_path, monkeypatch) -> None:
    """A locked empty directory is skipped without failing the pass."""
    from src.market_data.streams.retention import _remove_empty_dirs

    day_dir = tmp_path / "raw" / "hot" / "book_ticker" / "20260901"
    day_dir.mkdir(parents=True)

    def _flaky_rmdir(self, *args, **kwargs) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "rmdir", _flaky_rmdir)
    assert _remove_empty_dirs([tmp_path / "raw" / "hot"]) == 0
    assert day_dir.exists()
