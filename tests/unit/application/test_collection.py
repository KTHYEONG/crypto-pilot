from __future__ import annotations

from src.application import collection


def test_spot_ohlcv_command_wires_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    class FakeCollector:
        def ensure_spot_ohlcv(self, symbol: str, timeframe: str, start: str, end: str) -> None:
            calls.append((symbol, timeframe, start, end))

    monkeypatch.setattr(collection, "SpotDataCollector", FakeCollector)
    collection.collect_spot_ohlcv("BTCUSDT", "1h", "2024-01-01", "2024-01-02")

    assert calls == [("BTCUSDT", "1h", "2024-01-01", "2024-01-02")]


def test_import_borrow_command_wires_explicit_source(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    def fake_import(symbol: str, source: str, source_id: str, rate_period: str) -> None:
        calls.append((symbol, source, source_id, rate_period))

    monkeypatch.setattr(collection, "import_quote_borrow_history", fake_import)
    collection.import_borrow("BTCUSDT", "borrow-export.parquet", "operator:export-v1", "hourly")

    assert calls == [("BTCUSDT", "borrow-export.parquet", "operator:export-v1", "hourly")]


def test_collect_borrow_command_wires_signed_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(collection, "load_dotenv", lambda *args: True)
    monkeypatch.setattr(
        collection,
        "collect_binance_quote_borrow_history",
        lambda symbol, asset, start, end: calls.append((symbol, asset, start, end)),
    )

    collection.collect_borrow("BTCUSDT", "USDT", "2023-01-01", "2023-01-31")

    assert calls == [("BTCUSDT", "USDT", "2023-01-01", "2023-01-31")]


def test_futures_ohlcv_command_wires_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []

    class FakeCollector:
        def ensure_ohlcv_data(self, symbol: str, timeframe: str, start: str, end: str) -> None:
            calls.append((symbol, timeframe, start, end))

    monkeypatch.setattr(collection, "DataCollector", FakeCollector)
    collection.collect_ohlcv("BTCUSDT", "1h", "2024-01-01", "2024-01-02")

    assert calls == [("BTCUSDT", "1h", "2024-01-01", "2024-01-02")]

def test_funding_command_wires_collector(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeCollector:
        def ensure_funding_data(self, symbol: str, start: str, end: str) -> None:
            calls.append((symbol, start, end))

    monkeypatch.setattr(collection, "DataCollector", FakeCollector)
    collection.collect_funding("BTCUSDT", "2022-04-01", "2025-01-01")

    assert calls == [("BTCUSDT", "2022-04-01", "2025-01-01")]
