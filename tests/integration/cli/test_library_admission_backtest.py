from __future__ import annotations

import argparse
import json

import pytest

from src.cli.main import build_root_parser


def _find_parser(root: argparse.ArgumentParser, path: list[str]) -> argparse.ArgumentParser:
    current = root
    for name in path:
        sub = next(
            action for action in current._actions
            if isinstance(action, argparse._SubParsersAction)  # type: ignore[attr-defined]
        )
        current = sub.choices[name]
    return current


def test_backtest_cli_accepts_proposal_id_and_has_no_mutation_switches() -> None:
    parser = _find_parser(
        build_root_parser(), ["research", "run", "library-admission-backtest"],
    )
    options = {
        option
        for action in parser._actions  # type: ignore[attr-defined]
        for option in action.option_strings
    }
    assert "--proposal-id" in options
    assert "--expert-id" in options
    assert "--unseal-holdout" not in options
    assert "--register" not in options
    assert "--promote" not in options
    assert "--no-log-run" in options

    args = build_root_parser().parse_args([
        "research", "run", "library-admission-backtest",
        "--proposal-id",
        "lae-v1:technical_macd_histogram_regime_long_v1:BTCUSDT|technical_rsi_trend_pullback_long_v1:ETHUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "60",
        "--router-volatility-lookback-bars", "20",
        "--router-min-context-history-bars", "30",
        "--start", "2024-01-01",
    ])
    assert callable(args.handler)


def test_backtest_cli_emits_report_without_registry_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # LAE-10-PROPOSAL-BACKTEST-NO-MUTATION: CLI delegates to the in-memory
    # service and emits its report without registration switches.
    captured: dict[str, object] = {}

    class _Report:
        def to_report_dict(self) -> dict[str, object]:
            return {"status": "COMPLETE", "proposal_id": "lae-v1:a"}

    def fake_run(request):
        captured["expert_ids"] = request.expert_ids
        return _Report()

    monkeypatch.setattr(
        "src.application.research.expert_portfolio.admission_backtest.run_technical_library_admission_backtest",
        fake_run,
    )
    args = build_root_parser().parse_args([
        "research", "run", "library-admission-backtest",
        "--expert-id", "technical_macd_histogram_regime_long_v1:BTCUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "60",
        "--router-volatility-lookback-bars", "20",
        "--router-min-context-history-bars", "30",
    ])
    args.handler(args)
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "COMPLETE"
    assert captured["expert_ids"] == (
        "technical_macd_histogram_regime_long_v1:BTCUSDT",
    )


def test_backtest_cli_decodes_proposal_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class _Report:
        def to_report_dict(self) -> dict[str, object]:
            return {"status": "COMPLETE"}

    def fake_run(request):
        captured["expert_ids"] = request.expert_ids
        return _Report()

    monkeypatch.setattr(
        "src.application.research.expert_portfolio.admission_backtest.run_technical_library_admission_backtest",
        fake_run,
    )
    args = build_root_parser().parse_args([
        "research", "run", "library-admission-backtest",
        "--proposal-id",
        "lae-v1:technical_macd_histogram_regime_long_v1:BTCUSDT|technical_rsi_trend_pullback_long_v1:ETHUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "60",
        "--router-volatility-lookback-bars", "20",
        "--router-min-context-history-bars", "30",
    ])
    args.handler(args)
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"
    assert captured["expert_ids"] == (
        "technical_macd_histogram_regime_long_v1:BTCUSDT",
        "technical_rsi_trend_pullback_long_v1:ETHUSDT",
    )
