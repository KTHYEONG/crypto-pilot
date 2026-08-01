from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.cli.commands.provenance import add_provenance_commands
from src.cli.main import build_root_parser
from src.research.provenance.ledger import RUNS_LOG_PATH

_POPULATED = pd.DataFrame([{
    "ts": "2026-07-31T00:00:00+00:00",
    "record_type": "evaluation",
    "schema_version": 1,
    "symbol": "BTCUSDT",
    "end": "2025-12-31",
    "metrics.trade_count": 30,
    "metrics.cagr": 0.2,
    "metrics.mdd": -0.1,
    "metrics.sharpe": 1.5,
    "metrics.profit_factor": 1.5,
    "metrics.win_rate": 0.5,
    "reliability.observation.verdict": "PASS",
    "reliability.observation.lcb90_cagr": 0.16,
    "reliability.fold_distribution.max_period_contribution": 0.2,
    "reliability.stress_test.verdict": "PASS",
}])


def test_register_expert_library_command_parses_and_dispatches(
    monkeypatch, capsys,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_register(library_id: str, *, catalog=None, ledger_path=None):
        calls.append((library_id, ledger_path))
        return SimpleNamespace(registration_id="reg-123")

    monkeypatch.setattr(
        "src.cli.commands.provenance.register_expert_library", fake_register,
    )
    args = build_root_parser().parse_args([
        "provenance", "register", "expert-library", "--library-id", "valid_library",
    ])
    args.handler(args)
    assert calls == [("valid_library", None)]
    assert "reg-123" in capsys.readouterr().out


def test_compare_runs_renders_evaluations_only(monkeypatch, capsys) -> None:
    from src.cli.commands.provenance import compare_runs_command

    monkeypatch.setattr(
        "src.cli.commands.provenance.load_evaluation_runs",
        lambda ledger_path=RUNS_LOG_PATH: _POPULATED,
    )
    args = build_root_parser().parse_args(["provenance", "compare-runs", "--last", "5"])
    compare_runs_command(args)
    assert "BTCUSDT" in capsys.readouterr().out


def test_compare_runs_renders_empty(monkeypatch, capsys) -> None:
    from src.cli.commands.provenance import compare_runs_command

    monkeypatch.setattr(
        "src.cli.commands.provenance.load_evaluation_runs",
        lambda ledger_path=RUNS_LOG_PATH: pd.DataFrame(),
    )
    args = build_root_parser().parse_args(["provenance", "compare-runs"])
    compare_runs_command(args)
    assert "No evaluation runs recorded yet" in capsys.readouterr().out


def test_compare_runs_honours_ledger_path_override(
    monkeypatch, tmp_path: Path, capsys,
) -> None:
    from src.cli.commands.provenance import compare_runs_command

    seen: list[Path] = []
    monkeypatch.setattr(
        "src.cli.commands.provenance.load_evaluation_runs",
        lambda ledger_path=RUNS_LOG_PATH: seen.append(Path(ledger_path)) or pd.DataFrame(),
    )
    override = tmp_path / "ledger.jsonl"
    args = build_root_parser().parse_args([
        "provenance", "compare-runs", "--ledger-path", str(override),
    ])
    compare_runs_command(args)
    assert seen == [override]


def test_add_provenance_commands_attaches_group() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    add_provenance_commands(parser)
    assert parser.parse_args(["compare-runs"]).provenance_command == "compare-runs"
