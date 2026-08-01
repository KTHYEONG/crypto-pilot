from __future__ import annotations

import sys

from src.application import collection
from src.cli import collect_data


def test_collect_data_funding_subcommand_parses_args(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        collection,
        "collect_funding",
        lambda symbol, start, end: calls.append((symbol, start, end)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_data",
            "funding",
            "BTCUSDT",
            "--start",
            "2022-04-01",
            "--end",
            "2025-01-01",
        ],
    )

    collect_data.main()

    assert calls == [("BTCUSDT", "2022-04-01", "2025-01-01")]
