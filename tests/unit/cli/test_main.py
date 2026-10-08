from __future__ import annotations

import pytest

from src.market_data.services import collection
from src.cli.main import build_root_parser, main


def test_root_parser_exposes_the_two_groups() -> None:
    parser = build_root_parser()
    assert parser.parse_args(["data", "collect", "funding", "BTCUSDT", "--end", "2025-01-01"]).group == "data"
    assert build_root_parser(["lab", "horizon-diagnostic"]).parse_args(["lab", "horizon-diagnostic"]).group == "lab"


def test_root_parser_does_not_expose_provenance_group() -> None:
    # SCENARIO_MHS_REFACTOR_09: the provenance group was removed during the
    # legacy isolation refactor; only data + backtest + lab + live + ops remain.
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["provenance", "compare-runs"])


def test_root_parser_requires_a_group() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([])


def test_root_parser_requires_lab_command() -> None:
    with pytest.raises(SystemExit):
        build_root_parser(["lab"]).parse_args(["lab"])


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


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "collect", "metrics", "BTCUSDT", "--end", "2025-01-01"],
        ["data", "collect", "indicator-klines", "markPriceKlines", "BTCUSDT", "1h"],
        ["data", "collect", "bookdepth", "BTCUSDT"],
    ],
)
def test_data_collect_retired_collectors_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_root_parser().parse_args(argv)
    assert exc_info.value.code == 2


def test_lab_group_selection_skips_global_options() -> None:
    from src.cli.main import _lab_group_selected

    assert _lab_group_selected(["lab", "process-backtest"]) is True
    assert _lab_group_selected(["--log-level", "DEBUG", "lab"]) is True
    assert _lab_group_selected(["--debug-streams", "lab"]) is True
    assert _lab_group_selected(["live", "status"]) is False
    assert _lab_group_selected([]) is False
    assert _lab_group_selected(["--help"]) is False
    assert _lab_group_selected(["--log-level", "lab"]) is False


def test_root_parser_lazily_registers_lab_from_parse_arguments() -> None:
    parser = build_root_parser([])
    assert parser.parse_args(["live", "status"]).group == "live"
    args = parser.parse_args(["lab", "horizon-diagnostic", "--start", "2021-01-01"])
    assert args.lab_command == "horizon-diagnostic"
    assert args.start == "2021-01-01"
    assert parser.parse_args(["lab", "process-backtest"]).lab_command == "process-backtest"
