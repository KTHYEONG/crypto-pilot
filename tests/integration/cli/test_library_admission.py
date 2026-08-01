from __future__ import annotations

import argparse
import json

import pytest

from src.cli.main import build_root_parser
from src.research.expert_portfolio.contracts import (
    CandidateAdmissionResult,
    ContextualRouterSpec,
    ExpertDefinition,
    LibraryAdmissionConfig,
    LibraryAdmissionReport,
)


def _find_parser(root: argparse.ArgumentParser, path: list[str]) -> argparse.ArgumentParser:
    current = root
    for name in path:
        sub = next(
            a for a in current._actions if isinstance(a, argparse._SubParsersAction)  # type: ignore[attr-defined]
        )
        current = sub.choices[name]
    return current


def _admission_parser_options() -> set[str]:
    parser = _find_parser(
        build_root_parser(), ["research", "run", "library-admission"],
    )
    return {
        option
        for action in parser._actions  # type: ignore[attr-defined]
        for option in action.option_strings
    }


def test_library_admission_cli_forbids_promotion_switches() -> None:
    # LAE-08: the diagnostic exposes no holdout-unseal, register, or promote
    # switch; saving JSON is review evidence, never a registration.
    options = _admission_parser_options()
    assert "--unseal-holdout" not in options
    assert "--register" not in options
    assert "--promote" not in options


def test_library_admission_cli_requires_every_policy_and_router_field() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "library-admission"])
    args = build_root_parser().parse_args([
        "research", "run", "library-admission",
        "--candidate-source", "technical_macd_histogram_regime_long_v1",
        "--candidate-source", "technical_rsi_trend_pullback_long_v1",
        "--symbols", "BTCUSDT", "ETHUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "60",
        "--router-volatility-lookback-bars", "20",
        "--router-min-context-history-bars", "30",
        "--min-experts", "1",
        "--max-experts", "2",
        "--min-closed-trades", "1",
        "--min-active-return-bars", "1",
        "--max-abs-pairwise-log-return-correlation", "0.8",
        "--max-joint-negative-return-rate", "0.5",
        "--min-context-covered-states", "1",
        "--max-combinations", "100",
        "--max-workers", "2",
        "--start", "2024-01-01",
    ])
    assert callable(args.handler)
    assert args.max_workers == 2
    assert args.min_context_covered_states == 1


def _canned_report() -> LibraryAdmissionReport:
    return LibraryAdmissionReport(
        status="COMPLETE",
        window_start="2024-01-01T00:00:00+00:00",
        window_end="2025-12-31T20:00:00+00:00",
        experts=(
            ExpertDefinition(
                "technical_macd_histogram_regime_long_v1:BTCUSDT",
                "technical_macd_histogram_regime_long_v1",
                "macd_histogram_regime",
                ("BTCUSDT",),
                "run_technical_expert",
                "h" * 64,
            ),
        ),
        candidates=(
            CandidateAdmissionResult(
                "technical_macd_histogram_regime_long_v1:BTCUSDT",
                3, 5, True, None,
            ),
        ),
        proposals=(),
        context_coverage={"up_low_vol": 5},
        covered_states=1,
        coverage_sufficient=True,
        router=ContextualRouterSpec("BTCUSDT", 60, 20, 30),
        admission=LibraryAdmissionConfig(1, 1, 1, 1, 0.8, 0.5, 1, 1),
        code_hash="c" * 64,
        data_hashes={"BTCUSDT": {"perp_ohlcv": "a" * 64, "funding": "b" * 64}},
    )


def test_library_admission_cli_emits_deterministic_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # LAE-08: the handler emits deterministic JSON to stdout; a repeated run is
    # byte-identical and no ledger/catalog mutation occurs.
    args = build_root_parser().parse_args([
        "research", "run", "library-admission",
        "--candidate-source", "technical_macd_histogram_regime_long_v1",
        "--symbols", "BTCUSDT",
        "--router-context-symbol", "BTCUSDT",
        "--router-trend-lookback-bars", "60",
        "--router-volatility-lookback-bars", "20",
        "--router-min-context-history-bars", "30",
        "--min-experts", "1",
        "--max-experts", "1",
        "--min-closed-trades", "1",
        "--min-active-return-bars", "1",
        "--max-abs-pairwise-log-return-correlation", "0.8",
        "--max-joint-negative-return-rate", "0.5",
        "--min-context-covered-states", "1",
        "--max-combinations", "10",
    ])
    monkeypatch.setattr(
        "src.cli.commands.research.run_technical_library_admission",
        lambda request: _canned_report(),
    )

    args.handler(args)
    first = capsys.readouterr().out
    data = json.loads(first)
    assert data["status"] == "COMPLETE"
    assert data["router"]["trend_lookback_bars"] == 60
    assert data["candidates"][0]["admitted"] is True
    assert isinstance(data["fingerprint"], dict)

    args.handler(args)
    second = capsys.readouterr().out
    assert first == second
