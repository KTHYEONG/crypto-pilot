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
