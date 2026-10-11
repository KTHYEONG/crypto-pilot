from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.cli.commands.archive import add_archive_commands
from src.cli.main import build_root_parser
from src.market_data.public_archive.catalog import daily_key
from src.market_data.public_archive.roster import LiquidRoster, save_roster
from src.market_data.public_archive.store import load_coverage

_METRICS_HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)


def _zip_csv(body: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("data.csv", body)
    return buffer.getvalue()


class FakeDownloader:
    def __init__(self, symbols: list[str], days: list[date], *, corrupt: set[date] | None = None) -> None:
        self.keys: list[str] = []
        self.blobs: dict[str, bytes] = {}
        self.fetch_calls: list[str] = []
        for symbol in symbols:
            for day in days:
                stamp = pd.Timestamp(day, tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
                payload = _zip_csv(f"{_METRICS_HEADER}\n{stamp},{symbol},1,2,3,4,5,6\n")
                key = daily_key("metrics", symbol, day)
                stripped = key.removeprefix("data/futures/um/")
                digest = hashlib.sha256(payload).hexdigest()
                if corrupt and day in corrupt:
                    digest = "0" * 64
                self.keys.extend([key, key + ".CHECKSUM"])
                self.blobs[stripped] = payload
                self.blobs[stripped + ".CHECKSUM"] = f"{digest}  {key}\n".encode()
                agg_key = daily_key("aggTrades", symbol, day)
                self.keys.extend([agg_key, agg_key + ".CHECKSUM"])

    def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str] | None:
        del timeout
        return [k for k in self.keys if k.startswith(prefix)]

    def list_symbols_from_s3_xml_listing(self) -> list[str]:
        return ["BTCUSDT"]
    def fetch_archive_bytes(self, *parts: str) -> bytes | None:
        key = "/".join(parts)
        self.fetch_calls.append(key)
        return self.blobs.get(key)

    def list_all_symbols(self, **kwargs: object) -> list[str]:
        del kwargs
        return ["BTCUSDT"]


def _collect_ns(tmp_path: Path, argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_archive_commands(parser)
    ns = parser.parse_args([*argv, "--root", str(tmp_path)])
    return ns


def test_dry_run_default_writes_nothing(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "BTCUSDT",
        "--start", "2024-01-15", "--end", "2024-01-16",
    ])
    ns.downloader = FakeDownloader(["BTCUSDT"], days)
    ns.handler(ns)
    assert list(tmp_path.iterdir()) == []
    assert ns.downloader.fetch_calls == []


def test_exit_code_reflects_failures(tmp_path: Path) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "BTCUSDT",
        "--start", "2024-01-15", "--end", "2024-01-16", "--execute",
    ])
    ns.downloader = FakeDownloader(["BTCUSDT"], days, corrupt={date(2024, 1, 16)})
    with pytest.raises(SystemExit) as exc:
        ns.handler(ns)
    assert exc.value.code == 3
    coverage = load_coverage(tmp_path, "metrics")
    assert coverage[("BTCUSDT", days[0])].status == "ok"
    assert coverage[("BTCUSDT", days[1])].status == "failed"


def test_execute_success_exits_zero(tmp_path: Path) -> None:
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "BTCUSDT",
        "--start", "2024-01-15", "--end", "2024-01-15", "--execute",
    ])
    ns.downloader = FakeDownloader(["BTCUSDT"], [date(2024, 1, 15)])
    ns.handler(ns)
    assert (tmp_path / "metrics" / "BTCUSDT" / "202401.parquet").exists()


def test_roster_write_gated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive.roster import roster_path

    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path / "empty_lake")

    parser = argparse.ArgumentParser()
    add_archive_commands(parser)
    target = roster_path(tmp_path)
    save_roster(target, LiquidRoster(
        version=1, as_of=pd.Timestamp("2024-01-01", tz="UTC"),
        top_n=60, lookback_days=30, symbols=("BTCUSDT",),
    ))
    before = target.read_bytes()
    ns = parser.parse_args(["roster", "--root", str(tmp_path)])
    ns.handler(ns)
    assert target.read_bytes() == before


def test_aggtrades_requires_roster(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "aggTrades", "--symbols", "BTCUSDT",
        "--start", "2024-01-15", "--end", "2024-01-16",
    ])
    ns.downloader = FakeDownloader(["BTCUSDT"], [date(2024, 1, 15)])
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "aggTrades" in out
    assert list(tmp_path.iterdir()) == []


def test_archive_registered_in_main(tmp_path: Path) -> None:
    parser = build_root_parser([])
    ns = parser.parse_args(["archive", "status", "--root", str(tmp_path)])
    assert callable(ns.handler)
    ns.handler(ns)


def test_root_and_downloader_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.commands import archive as archive_mod

    monkeypatch.setattr("src.common.paths.PUBLIC_ARCHIVE_DIR", tmp_path / "default_root")

    class _StubDownloader:
        pass

    monkeypatch.setattr(
        "src.market_data.binance.vision.BinanceVisionDownloader", _StubDownloader
    )
    ns = argparse.Namespace(root=None, downloader=None)
    assert archive_mod._root(ns) == tmp_path / "default_root"
    assert isinstance(archive_mod._downloader(ns), _StubDownloader)
    assert archive_mod._downloader(argparse.Namespace(downloader="fake")) == "fake"


def test_daily_quote_volume_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.commands import archive as archive_mod

    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path / "missing")
    assert archive_mod._daily_quote_volume().empty  # type: ignore[union-attr]
    lake = tmp_path / "futures" / "ohlcv" / "1h"
    lake.mkdir(parents=True)
    stamps = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    ms = stamps.tz_localize(None).to_numpy().astype("datetime64[ms]").astype("int64")
    pd.DataFrame({"timestamp": ms, "quote_vol": 1.0}).to_parquet(lake / "AUSDT.parquet")
    pd.DataFrame({"timestamp": ms, "close": 1.0}).to_parquet(lake / "BUSDT.parquet")
    (lake / "CUSDT.parquet").write_bytes(b"not-a-parquet")
    (lake / "IGNORED.parquet").write_bytes(b"not-a-perpetual")
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path / "futures")
    frame = archive_mod._daily_quote_volume()
    assert list(frame.columns) == ["AUSDT"]
    assert frame["AUSDT"].sum() == 48.0


def test_roster_dry_run_reports_additions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.commands import archive

    frame = pd.DataFrame({"BTCUSDT": [10.0, 10.0]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    monkeypatch.setattr(archive, "_daily_quote_volume", lambda: frame)
    ns = _collect_ns(tmp_path, ["roster", "--lookback-days", "1"])
    ns.handler(ns)
    assert list(tmp_path.iterdir()) == []


def test_collect_planning_failure_exits_two(tmp_path: Path) -> None:
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "BTCUSDT",
        "--start", "2024-01-15", "--end", "2024-01-16",
    ])

    class _Failing(FakeDownloader):
        def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str] | None:
            del prefix, timeout
            return None

    ns.downloader = _Failing(["BTCUSDT"], [date(2024, 1, 15)])
    with pytest.raises(SystemExit) as exc:
        ns.handler(ns)
    assert exc.value.code == 2


def test_collect_symbols_all_and_status(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "all",
        "--start", "2024-01-15", "--end", "2024-01-15", "--execute",
    ])
    ns.downloader = FakeDownloader(["BTCUSDT"], [date(2024, 1, 15)])
    ns.handler(ns)
    status_ns = _collect_ns(tmp_path, ["status"])
    status_ns.handler(status_ns)
    out = capsys.readouterr().out
    assert "status dataset=metrics symbols=1" in out
    assert "heartbeat ts=" in out


def test_roster_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive.roster import load_roster, roster_path

    lake = tmp_path / "futures" / "ohlcv" / "1h"
    lake.mkdir(parents=True)
    stamps = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    ms = stamps.tz_localize(None).to_numpy().astype("datetime64[ms]").astype("int64")
    vols_a = [10.0] * 48 + [1.0] * 48
    vols_b = [1.0] * 48 + [10.0] * 48
    pd.DataFrame({"timestamp": ms, "quote_vol": vols_a}).to_parquet(lake / "AUSDT.parquet")
    pd.DataFrame({"timestamp": ms, "quote_vol": vols_b}).to_parquet(lake / "BUSDT.parquet")
    monkeypatch.setattr("src.common.paths.FUTURES_DATA_DIR", tmp_path / "futures")
    parser = argparse.ArgumentParser()
    add_archive_commands(parser)
    ns = parser.parse_args(["roster", "--top-n", "1", "--lookback-days", "1", "--write", "--root", str(tmp_path)])
    ns.handler(ns)
    roster = load_roster(roster_path(tmp_path))
    assert roster is not None
    assert roster.symbols == ("AUSDT", "BUSDT")
    assert roster.version == 1


def test_collect_uses_listing_fallback(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    inner = FakeDownloader(["BTCUSDT"], [date(2024, 1, 15)])

    class _Fallback:
        def list_object_keys(self, *, prefix: str, timeout: int | None = None) -> list[str] | None:
            return inner.list_object_keys(prefix=prefix, timeout=timeout)

        def fetch_archive_bytes(self, *parts: str) -> bytes | None:
            return inner.fetch_archive_bytes(*parts)

        def list_symbols_from_s3_xml_listing(self, **kwargs: object) -> list[str]:
            del kwargs
            return ["BTCUSDT"]

    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "all",
        "--start", "2024-01-15", "--end", "2024-01-15",
    ])
    ns.downloader = _Fallback()
    ns.handler(ns)
    assert "plan dataset=metrics symbols=1 days=1" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_collect_empty_universe_exits_two(tmp_path: Path) -> None:
    class _Empty(FakeDownloader):
        def list_all_symbols(self, **kwargs: object) -> list[str]:
            del kwargs
            return []

    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "metrics", "--symbols", "all",
        "--start", "2024-01-15", "--end", "2024-01-15",
    ])
    ns.downloader = _Empty(["BTCUSDT"], [date(2024, 1, 15)])
    with pytest.raises(SystemExit) as exc:
        ns.handler(ns)
    assert exc.value.code == 2


def test_collect_aggtrades_with_roster(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from src.market_data.public_archive.roster import roster_path, save_roster

    save_roster(roster_path(tmp_path), LiquidRoster(
        version=1, as_of=pd.Timestamp("2024-01-20", tz="UTC"),
        top_n=60, lookback_days=30, symbols=("BTCUSDT",),
    ))
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "aggTrades", "--symbols", "BTCUSDT",
        "--start", "2024-01-15", "--end", "2024-01-15",
    ])
    ns.downloader = FakeDownloader(["BTCUSDT"], [date(2024, 1, 15)])
    ns.handler(ns)
    assert "plan dataset=aggTrades symbols=1 days=1" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["_roster"]


def test_each_dataset_uses_its_bucket_universe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service

    captured = {}

    class _Listed(FakeDownloader):
        def list_all_symbols(self, **kwargs: object) -> list[str]:
            prefix = str(kwargs["dataset_prefix"])
            return ["DELISTEDUSDT"] if "metrics" in prefix else ["DEPTHUSDT"]

    def capture(*args, **kwargs):
        captured.update(kwargs["symbols"])
        return []

    monkeypatch.setattr(service, "plan_collection", capture)
    ns = _collect_ns(tmp_path, ["collect", "--dataset", "all"])
    ns.downloader = _Listed([], [])
    ns.handler(ns)
    assert captured == {"metrics": ["DELISTEDUSDT"], "bookDepth": ["DEPTHUSDT"]}
