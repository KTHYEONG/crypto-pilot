from __future__ import annotations

from argparse import Namespace

from src.cli import collect_data


def test_spot_ohlcv_command_wires_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    class FakeCollector:
        def ensure_spot_ohlcv(self, symbol: str, timeframe: str, start: str, end: str) -> None:
            calls.append((symbol, timeframe, start, end))

    monkeypatch.setattr(collect_data, "SpotDataCollector", FakeCollector)
    collect_data._spot_ohlcv(
        Namespace(symbol="BTCUSDT", timeframe="1h", start="2024-01-01", end="2024-01-02")
    )

    assert calls == [("BTCUSDT", "1h", "2024-01-01", "2024-01-02")]


def test_import_borrow_command_wires_explicit_source(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fake_import(symbol: str, source: str, source_id: str, rate_period: str) -> None:
        calls.append((symbol, source, source_id, rate_period))

    monkeypatch.setattr(collect_data, "import_quote_borrow_history", fake_import)
    collect_data._import_borrow(
        Namespace(
            symbol="BTCUSDT",
            source="borrow-export.parquet",
            source_id="operator:export-v1",
            rate_period="hourly",
        )
    )

    assert calls == [("BTCUSDT", "borrow-export.parquet", "operator:export-v1", "hourly")]


def test_collect_borrow_command_wires_signed_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(collect_data, "load_dotenv", lambda *args: True)
    monkeypatch.setattr(
        collect_data,
        "collect_binance_quote_borrow_history",
        lambda symbol, asset, start, end: calls.append((symbol, asset, start, end)),
    )

    collect_data._collect_borrow(
        Namespace(symbol="BTCUSDT", asset="USDT", start="2023-01-01", end="2023-01-31"),
    )

    assert calls == [("BTCUSDT", "USDT", "2023-01-01", "2023-01-31")]


def test_futures_ohlcv_command_wires_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    class FakeCollector:
        def ensure_ohlcv_data(self, symbol: str, timeframe: str, start: str, end: str) -> None:
            calls.append((symbol, timeframe, start, end))

    monkeypatch.setattr(collect_data, "DataCollector", FakeCollector)
    collect_data._ohlcv(
        Namespace(symbol="BTCUSDT", timeframe="1h", start="2024-01-01", end="2024-01-02")
    )

    assert calls == [("BTCUSDT", "1h", "2024-01-01", "2024-01-02")]
