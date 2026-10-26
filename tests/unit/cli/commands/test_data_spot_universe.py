from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest


def _parser():
    import argparse

    from src.cli.commands.data import add_data_commands

    parser = argparse.ArgumentParser()
    add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    return parser


def _seat_mask(symbols_days: dict[str, tuple[str, str]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2023-01-01", "2023-03-31", freq="D", tz="UTC"))
    mask = pd.DataFrame(False, index=idx, columns=sorted(symbols_days))
    for symbol, (start, end) in symbols_days.items():
        mask.loc[start:end, symbol] = True
    return mask


def _stub_census(monkeypatch, mask: pd.DataFrame) -> None:
    import src.market_data.public_archive.mask as mask_mod

    monkeypatch.setattr(mask_mod, "load_daily_census", lambda root: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(mask_mod, "seat_day_mask", lambda *a, **k: mask)


def _stub_spot_client(monkeypatch, present: set[str]) -> list[tuple]:
    import src.market_data.binance.spot as spot_mod

    calls: list[tuple] = []

    class _Fake:
        def fetch_spot_ohlcv(self, symbol, timeframe, start, end):
            calls.append((symbol, timeframe, str(start), str(end)))
            if symbol in present:
                return pd.DataFrame({"timestamp": [1]})
            return pd.DataFrame()

    monkeypatch.setattr(spot_mod, "BinanceSpotClient", _Fake)
    return calls


def test_dry_run_never_downloads_klines(tmp_path, monkeypatch) -> None:
    import src.market_data.services.spot_collection as spot_collection_mod

    mask = _seat_mask({"PEPEUSDT": ("2023-01-01", "2023-03-10")})
    _stub_census(monkeypatch, mask)
    probe_calls = _stub_spot_client(monkeypatch, {"PEPEUSDT"})
    ensured: list = []
    monkeypatch.setattr(spot_collection_mod, "ensure_spot_ohlcv", lambda *a, **k: ensured.append(a))

    out = tmp_path / "companions.json"
    args = _parser().parse_args([
        "data", "collect", "spot-universe",
        "--start", "2023-01-01", "--end", "2023-03-10", "--output", str(out),
    ])
    args.handler(args)

    assert probe_calls
    assert ensured == []
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["futures_symbol"] == "PEPEUSDT"
    assert payload[0]["spot_symbol"] == "PEPEUSDT"


def test_shared_spot_symbol_collected_once_from_earliest_day(tmp_path, monkeypatch) -> None:
    import src.market_data.services.spot_collection as spot_collection_mod

    mask = _seat_mask({
        "PEPEUSDT": ("2023-03-01", "2023-03-10"),
        "1000PEPEUSDT": ("2023-01-01", "2023-03-10"),
    })
    _stub_census(monkeypatch, mask)
    _stub_spot_client(monkeypatch, {"PEPEUSDT"})
    ensured: list = []
    monkeypatch.setattr(spot_collection_mod, "ensure_spot_ohlcv", lambda *a, **k: ensured.append(a))

    out = tmp_path / "companions.json"
    args = _parser().parse_args([
        "data", "collect", "spot-universe",
        "--start", "2023-01-01", "--end", "2023-03-31", "--output", str(out), "--execute",
    ])
    args.handler(args)

    assert ensured == [("PEPEUSDT", "1h", "2023-01-01", "2023-03-31")]


def test_partial_failure_exits_3_without_aborting_others(tmp_path, monkeypatch, caplog) -> None:
    import src.market_data.services.spot_collection as spot_collection_mod

    mask = _seat_mask({
        "AAAUSDT": ("2023-01-01", "2023-03-10"),
        "BBBUSDT": ("2023-01-01", "2023-03-10"),
    })
    _stub_census(monkeypatch, mask)
    _stub_spot_client(monkeypatch, {"AAAUSDT", "BBBUSDT"})

    collected: list[str] = []

    def _flaky(spot, timeframe, start, end):
        collected.append(spot)
        if spot == "AAAUSDT":
            raise RuntimeError("boom")

    monkeypatch.setattr(spot_collection_mod, "ensure_spot_ohlcv", _flaky)

    out = tmp_path / "companions.json"
    args = argparse.Namespace(
        start="2023-01-01", end="2023-03-10", breadth=30, timeframe="1h",
        execute=True, output=str(out),
    )
    from src.cli.commands.data_spot_universe import run_spot_universe

    with caplog.at_level("INFO", logger="src.cli.commands.data_spot_universe"), pytest.raises(SystemExit) as exc:
        run_spot_universe(args)

    assert exc.value.code == 3
    assert sorted(collected) == ["AAAUSDT", "BBBUSDT"]
    assert "AAAUSDT" in caplog.text


def test_midspan_retry_maps_pair_listed_midspan(tmp_path, monkeypatch) -> None:
    import src.market_data.services.spot_collection as spot_collection_mod

    mask = _seat_mask({"MIDUSDT": ("2023-01-01", "2023-03-10")})
    _stub_census(monkeypatch, mask)

    import src.market_data.binance.spot as spot_mod

    class _FakeMid:
        def fetch_spot_ohlcv(self, symbol, timeframe, start, end):
            if str(start).startswith("2023-02"):
                return pd.DataFrame({"timestamp": [1]})
            return pd.DataFrame()

    monkeypatch.setattr(spot_mod, "BinanceSpotClient", _FakeMid)
    monkeypatch.setattr(spot_collection_mod, "ensure_spot_ohlcv", lambda *a, **k: None)

    out = tmp_path / "companions.json"
    args = _parser().parse_args([
        "data", "collect", "spot-universe",
        "--start", "2023-01-01", "--end", "2023-03-10", "--output", str(out),
    ])
    args.handler(args)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload[0]["status"] == "mapped"
    assert payload[0]["spot_symbol"] == "MIDUSDT"


def test_no_spot_history_written_and_skipped_on_execute(tmp_path, monkeypatch) -> None:
    import src.market_data.services.spot_collection as spot_collection_mod

    mask = _seat_mask({
        "AAAUSDT": ("2023-01-01", "2023-03-10"),
        "ZZZUSDT": ("2023-01-01", "2023-03-10"),
    })
    _stub_census(monkeypatch, mask)
    _stub_spot_client(monkeypatch, {"AAAUSDT"})
    ensured: list = []
    monkeypatch.setattr(spot_collection_mod, "ensure_spot_ohlcv", lambda *a, **k: ensured.append(a))

    out = tmp_path / "companions.json"
    args = _parser().parse_args([
        "data", "collect", "spot-universe",
        "--start", "2023-01-01", "--end", "2023-03-10", "--output", str(out), "--execute",
    ])
    args.handler(args)

    payload = {row["futures_symbol"]: row for row in json.loads(out.read_text(encoding="utf-8"))}
    assert payload["ZZZUSDT"]["status"] == "no_spot_history"
    assert payload["ZZZUSDT"]["spot_symbol"] is None
    assert ensured
    assert all(call[0] == "AAAUSDT" for call in ensured)


def test_invalid_range_rejected(tmp_path) -> None:
    from src.cli.commands.data_spot_universe import run_spot_universe

    args = argparse.Namespace(
        start="2023-03-10", end="2023-01-01", breadth=30, timeframe="1h",
        execute=False, output=str(tmp_path / "never.json"),
    )
    with pytest.raises(ValueError, match="invalid range"):
        run_spot_universe(args)


def test_progress_logged_during_collection(tmp_path, monkeypatch, caplog) -> None:
    import time

    import src.market_data.services.spot_collection as spot_collection_mod

    mask = _seat_mask({
        "AAAUSDT": ("2023-01-01", "2023-03-10"),
        "BBBUSDT": ("2023-01-01", "2023-03-10"),
    })
    _stub_census(monkeypatch, mask)
    _stub_spot_client(monkeypatch, {"AAAUSDT", "BBBUSDT"})
    monkeypatch.setattr(spot_collection_mod, "ensure_spot_ohlcv", lambda *a, **k: None)
    ticks = iter([0.0, 31.0, 62.0, 93.0, 124.0, 155.0, 186.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks, 999.0))

    out = tmp_path / "companions.json"
    args = argparse.Namespace(
        start="2023-01-01", end="2023-03-10", breadth=30, timeframe="1h",
        execute=True, output=str(out),
    )
    from src.cli.commands.data_spot_universe import run_spot_universe

    with caplog.at_level("INFO", logger="src.cli.commands.data_spot_universe"):
        run_spot_universe(args)

    assert "stage=spot_universe" in caplog.text
    assert out.exists()


@pytest.mark.parametrize("code", [-1121, -1100])
def test_transport_invalid_symbol_is_distinct_from_request_failure(tmp_path, monkeypatch, code) -> None:
    import io
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlsplit

    from src.market_data.binance.futures import BinanceKlinePermanentError

    _stub_census(monkeypatch, _seat_mask({"1000PEPEUSDT": ("2023-01-01", "2023-03-10")}))
    calls = []

    class Response:
        def __init__(self, empty=False):
            self.empty = empty

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            if self.empty:
                return b"[]"
            return json.dumps([[1672531200000, "1", "1", "1", "1", "1", 0, "1", 1, "1", "1"]]).encode()

    def fetch(request, **kwargs):
        calls.append(request.full_url)
        if "symbol=1000PEPEUSDT" in request.full_url:
            raise HTTPError(request.full_url, 400, "bad request", {}, io.BytesIO(json.dumps({"code": code}).encode()))
        since = int(parse_qs(urlsplit(request.full_url).query)["startTime"][0])
        return Response(empty=since > 1672531200000)

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    monkeypatch.setattr("time.sleep", lambda _: None)
    out = tmp_path / "companions.json"
    args = _parser().parse_args([
        "data", "collect", "spot-universe", "--end", "2023-03-10", "--output", str(out),
    ])
    if code == -1121:
        args.handler(args)
        assert json.loads(out.read_text())[0]["spot_symbol"] == "PEPEUSDT"
        assert calls
    else:
        with pytest.raises(BinanceKlinePermanentError):
            args.handler(args)
        assert not out.exists()


def test_transport_exhaustion_preserves_existing_manifest(tmp_path, monkeypatch) -> None:
    _stub_census(monkeypatch, _seat_mask({"BTCUSDT": ("2023-01-01", "2023-03-10")}))

    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    monkeypatch.setattr("time.sleep", lambda _: None)
    out = tmp_path / "companions.json"
    out.write_text("previous")
    args = _parser().parse_args([
        "data", "collect", "spot-universe", "--end", "2023-03-10", "--output", str(out),
    ])
    with pytest.raises(OSError, match="retries exhausted"):
        args.handler(args)
    assert out.read_text() == "previous"
