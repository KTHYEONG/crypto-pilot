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

    def list_symbols_from_s3_xml_listing(self, **kwargs: object) -> list[str]:
        del kwargs
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


def test_aggtrades_uses_bucket_universe(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    days = [date(2024, 1, 15), date(2024, 1, 16)]

    class _Universe(FakeDownloader):
        def list_all_symbols(self, **kwargs: object) -> list[str]:
            del kwargs
            return ["BTCUSDT", "ETHUSDT", "AAVUSDT", "BTCUSDT_250627"]

    downloader = _Universe(["BTCUSDT", "ETHUSDT", "AAVUSDT"], days)
    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "aggTrades", "--symbols", "all",
        "--start", "2024-01-15", "--end", "2024-01-16",
    ])
    ns.downloader = downloader
    ns.handler(ns)
    out = capsys.readouterr().out
    assert "plan dataset=aggTrades symbols=3 days=6" in out
    assert "roster" not in out
    assert not (tmp_path / "_roster").exists()


def test_roster_command_removed(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    add_archive_commands(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["roster", "--root", str(tmp_path)])


def test_explicit_symbols_pass_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import service

    captured: dict[str, list[str]] = {}

    def capture(*args: object, **kwargs: object) -> list:
        symbols = kwargs["symbols"]  # type: ignore[index]
        captured.update(symbols)  # type: ignore[arg-type]
        return []

    monkeypatch.setattr(service, "plan_collection", capture)
    ns = _collect_ns(tmp_path, ["collect", "--dataset", "all", "--symbols", "BTCUSDT,ETHUSDT"])
    ns.downloader = FakeDownloader(["BTCUSDT"], [date(2024, 1, 15)])
    ns.handler(ns)
    assert captured == {
        "metrics": ["BTCUSDT", "ETHUSDT"],
        "bookDepth": ["BTCUSDT", "ETHUSDT"],
        "aggTrades": ["BTCUSDT", "ETHUSDT"],
    }


def test_aggtrades_listing_failure_exits_two(tmp_path: Path) -> None:
    class _Empty(FakeDownloader):
        def list_all_symbols(self, **kwargs: object) -> list[str]:
            del kwargs
            return []

    ns = _collect_ns(tmp_path, [
        "collect", "--dataset", "aggTrades", "--symbols", "all",
        "--start", "2024-01-15", "--end", "2024-01-16",
    ])
    ns.downloader = _Empty(["BTCUSDT"], [date(2024, 1, 15)])
    with pytest.raises(SystemExit) as exc:
        ns.handler(ns)
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", ["none", "exception"])
def test_universe_listing_errors_fail_closed(tmp_path: Path, failure: str) -> None:
    class _Broken(FakeDownloader):
        def list_all_symbols(self, **kwargs: object) -> list[str] | None:
            if failure == "exception":
                raise OSError("listing unavailable")
            return None

    ns = _collect_ns(tmp_path, ["collect", "--dataset", "all", "--execute"])
    ns.downloader = _Broken([], [])
    with pytest.raises(SystemExit) as exc:
        ns.handler(ns)
    assert exc.value.code == 2
    assert ns.downloader.fetch_calls == []
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
    assert captured == {
        "metrics": ["DELISTEDUSDT"],
        "bookDepth": ["DEPTHUSDT"],
        "aggTrades": ["DEPTHUSDT"],
    }


def _prune_ns(tmp_path: Path, argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_archive_commands(parser)
    return parser.parse_args([*argv, "--root", str(tmp_path)])


def _write_status(path: Path) -> None:
    import json

    finished = pd.Timestamp.now(tz="UTC")
    started = finished - pd.Timedelta(hours=1)
    path.write_text(
        json.dumps({"started_at": started.isoformat(), "finished_at": finished.isoformat(), "rc": 0}),
        encoding="utf-8",
    )


def _old_partition(root: Path) -> Path:
    import os
    import time

    target = root / "metrics" / "BTCUSDT" / "202401.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"parquet-stub")
    old = time.mktime(pd.Timestamp("2024-01-10T00:00:00Z").to_pydatetime().timetuple())
    os.utime(target, (old, old))
    return target


def test_prune_prunes_and_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import json

    target = _old_partition(tmp_path)
    status_file = tmp_path / "last_success.json"
    _write_status(status_file)
    ns = _prune_ns(tmp_path, ["prune", "--backup-status", str(status_file)])
    ns.handler(ns)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["prune_blocked"] is False
    assert payload["pruned_files"] == 1
    assert not target.exists()


def test_prune_blocked_without_status_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import json

    target = _old_partition(tmp_path)
    ns = _prune_ns(tmp_path, ["prune", "--backup-status", str(tmp_path / "missing.json")])
    ns.handler(ns)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["prune_blocked"] is True
    assert payload["blocked_reason"] == "status_missing"
    assert target.exists()


def test_prune_io_error_exits_four(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("injected")

    target = _old_partition(tmp_path)
    monkeypatch.setattr(Path, "unlink", _boom)
    status_file = tmp_path / "last_success.json"
    _write_status(status_file)
    ns = _prune_ns(tmp_path, ["prune", "--backup-status", str(status_file)])
    with pytest.raises(SystemExit) as exc:
        ns.handler(ns)
    assert exc.value.code == 4
    assert target.exists()


def test_prune_default_mount_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.commands import archive as archive_mod

    status_file = tmp_path / "last_success.json"
    _write_status(status_file)
    monkeypatch.setattr(archive_mod, "ARCHIVE_BACKUP_MOUNT", status_file)
    target = _old_partition(tmp_path)
    ns = _prune_ns(tmp_path, ["prune"])
    ns.handler(ns)
    assert not target.exists()


def test_prune_default_repo_status_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import json

    from src.cli.commands import archive as archive_mod

    monkeypatch.setattr(archive_mod, "ARCHIVE_BACKUP_MOUNT", tmp_path / "absent.json")
    monkeypatch.setattr("src.common.paths.BASE_DIR", tmp_path / "empty_base")
    target = _old_partition(tmp_path)
    ns = _prune_ns(tmp_path, ["prune"])
    ns.handler(ns)
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["prune_blocked"] is True
    assert target.exists()


@pytest.mark.parametrize(("flag", "start", "days"), [("1", None, 1), ("0", None, 2), ("1", "2024-01-01", 2)])
def test_collect_incremental_env_and_explicit_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    flag: str, start: str | None, days: int,
) -> None:
    monkeypatch.setenv("ARCHIVE_INCREMENTAL", flag)
    argv = ["collect", "--dataset", "metrics", "--symbols", "BTCUSDT", "--end", "2024-01-31"]
    if start is not None:
        argv.extend(["--start", start])
    ns = _collect_ns(tmp_path, argv)
    ns.downloader = FakeDownloader(["BTCUSDT"], [date(2024, 1, 1), date(2024, 1, 20)])
    ns.handler(ns)
    assert f"days={days} files={days}" in capsys.readouterr().out
    assert ns.downloader.fetch_calls == []
