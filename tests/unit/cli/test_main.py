from __future__ import annotations

import pytest

from src.market_data.services import collection
from src.cli.main import build_root_parser, main


def test_root_parser_exposes_the_two_groups() -> None:
    parser = build_root_parser()
    assert parser.parse_args(["data", "collect", "funding", "BTCUSDT", "--end", "2025-01-01"]).group == "data"
    assert parser.parse_args(["research", "run", "portfolio", "mhs-horizon-diagnostic"]).group == "research"


def test_root_parser_does_not_expose_provenance_group() -> None:
    # SCENARIO_MHS_REFACTOR_09: the provenance group was removed during the
    # legacy isolation refactor; only data + research remain.
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["provenance", "compare-runs"])


def test_root_parser_requires_a_group() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([])


def test_root_parser_requires_run_command_and_evaluation() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research"])
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run"])
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "portfolio"])


def test_data_collect_funding_subcommand_parses_and_dispatches(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        collection, "collect_funding",
        lambda symbol, start, end: calls.append((symbol, start, end)),
    )
    main([
        "data", "collect", "funding", "BTCUSDT",
        "--start", "2022-04-01", "--end", "2025-01-01",
    ])
    assert calls == [("BTCUSDT", "2022-04-01", "2025-01-01")]


def test_data_collect_futures_ohlcv_parses_and_dispatches(monkeypatch) -> None:
    calls: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(
        collection, "collect_ohlcv",
        lambda symbol, timeframe, start, end: calls.append((symbol, timeframe, start, end)),
    )
    main(["data", "collect", "futures-ohlcv", "BTCUSDT", "1h", "--start", "2024-01-01"])
    assert calls[0][0:3] == ("BTCUSDT", "1h", "2024-01-01")
    assert calls[0][3], "end defaults to now and must be non-empty"


def test_data_collect_metrics_parses_and_dispatches(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        collection, "collect_metrics",
        lambda symbol, start, end: calls.append((symbol, start, end)),
    )
    main([
        "data", "collect", "metrics", "BTCUSDT",
        "--start", "2022-04-01", "--end", "2025-01-01",
    ])
    assert calls == [("BTCUSDT", "2022-04-01", "2025-01-01")]
