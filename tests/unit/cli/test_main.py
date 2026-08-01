from __future__ import annotations

import pytest

from src.application.data import collection
from src.cli.main import build_root_parser, main


def test_root_parser_groups_research_commands() -> None:
    # PL-CLI-001: the parser maps expert command arguments to the frozen
    # request defaults.
    args = build_root_parser().parse_args(
        ["research", "run", "expert-portfolio", "--library-id", "valid_library"],
    )
    assert args.group == "research"
    assert args.research_command == "run"
    assert args.run_command == "expert-portfolio"
    assert args.library_id == "valid_library"


def test_root_parser_exposes_the_three_groups() -> None:
    parser = build_root_parser()
    assert parser.parse_args(["data", "collect", "funding", "BTCUSDT", "--end", "2025-01-01"]).group == "data"
    assert parser.parse_args(["research", "run", "baseline"]).group == "research"
    assert parser.parse_args(["provenance", "compare-runs"]).group == "provenance"


def test_root_parser_exposes_oi_deleveraging_evaluation() -> None:
    args = build_root_parser().parse_args(
        ["research", "run", "oi-deleveraging", "--symbol", "BTCUSDT"],
    )
    assert args.run_command == "oi-deleveraging"
    assert args.symbol == "BTCUSDT"


def test_root_parser_requires_a_group() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([])


def test_root_parser_requires_run_command_and_evaluation() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research"])
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run"])


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
