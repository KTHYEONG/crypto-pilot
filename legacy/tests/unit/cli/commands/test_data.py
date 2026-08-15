from __future__ import annotations

import argparse

from src.cli.commands import data


def test_metrics_handler_dispatches_requested_range(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        data.collection,
        "collect_metrics",
        lambda symbol, start, end: calls.append((symbol, start, end)),
    )

    data._metrics(
        argparse.Namespace(symbol="BTCUSDT", start="2022-04-01", end="2025-01-01"),
    )

    assert calls == [("BTCUSDT", "2022-04-01", "2025-01-01")]

def test_indicator_klines_handler_dispatches_requested_range(monkeypatch) -> None:
    # SCENARIO_CLI_WIRING_05: indicator-klines parses and dispatches to the wrapper.
    calls: list[tuple[str, str, str, str, str]] = []
    monkeypatch.setattr(
        data.collection,
        "collect_indicator_klines",
        lambda dataset, symbol, timeframe, start, end: calls.append(
            (dataset, symbol, timeframe, start, end),
        ),
    )

    data._indicator_klines(
        argparse.Namespace(
            dataset="premiumIndexKlines", symbol="BTCUSDT", timeframe="4h",
            start="2022-04-01", end="2025-01-01",
        ),
    )

    assert calls == [("premiumIndexKlines", "BTCUSDT", "4h", "2022-04-01", "2025-01-01")]


def test_bookdepth_handler_dispatches_requested_range(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        data.collection,
        "collect_bookdepth",
        lambda symbol, start, end: calls.append((symbol, start, end)),
    )

    data._bookdepth(
        argparse.Namespace(symbol="BTCUSDT", start="2022-04-01", end="2025-01-01"),
    )

    assert calls == [("BTCUSDT", "2022-04-01", "2025-01-01")]


def test_all_collect_subcommands_resolve_single_start_default() -> None:
    # SCENARIO_MHS_GAP_HARDENING_06: all 7 `data collect` subcommands resolve
    # --start through the single _DEFAULT_COLLECTION_START constant.
    parser = argparse.ArgumentParser()
    data.add_data_commands(parser)
    cases = {
        "futures-ohlcv": ["futures-ohlcv", "BTCUSDT", "1h"],
        "spot-ohlcv": ["spot-ohlcv", "BTCUSDT", "1h"],
        "funding": ["funding", "BTCUSDT", "--end", "2025-01-01"],
        "metrics": ["metrics", "BTCUSDT", "--end", "2025-01-01"],
        "indicator-klines": ["indicator-klines", "premiumIndexKlines", "BTCUSDT", "1h"],
        "bookdepth": ["bookdepth", "BTCUSDT"],
        "collect-borrow": ["collect-borrow", "BTCUSDT", "--end", "2025-01-01"],
    }
    for subcommand, argv in cases.items():
        args = parser.parse_args(["collect", *argv])
        assert args.start == "2022-04-01", subcommand
        assert args.start == data._DEFAULT_COLLECTION_START, subcommand
