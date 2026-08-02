from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.cli.main import main as root_main


def _dispatch_legacy(monkeypatch, canonical_target: str, argv: list[str]):
    """Run a legacy module and capture the typed request it produces.

    ``canonical_target`` is the canonical application module attribute that the
    delegated handler calls (e.g. ``src.application.research.baseline.evaluation.
    run_baseline_evaluation``), so the patch is observable through the leaf
    command modules.
    """
    calls: list[object] = []
    monkeypatch.setattr(canonical_target, calls.append)
    monkeypatch.setattr(sys, "argv", argv)
    return calls


def test_legacy_run_backtest_equivalent_to_research_run_baseline(monkeypatch) -> None:
    legacy = _dispatch_legacy(
        monkeypatch,
        "src.application.research.baseline.evaluation.run_baseline_evaluation",
        ["run_backtest", "--symbol", "ETHUSDT", "--no-log-run"],
    )
    from src.cli.adapters import run_backtest

    run_backtest.main()
    assert len(legacy) == 1
    assert legacy[0].symbol == "ETHUSDT"
    assert legacy[0].log_run is False
    assert legacy[0].unseal_holdout is False

    grouped: list[object] = []
    monkeypatch.setattr(
        "src.application.research.baseline.evaluation.run_baseline_evaluation", grouped.append,
    )
    root_main(["research", "run", "single", "baseline", "--symbol", "ETHUSDT", "--no-log-run"])
    assert grouped == legacy


def test_legacy_portfolio_equivalent_to_research_run_portfolio(monkeypatch) -> None:
    legacy = _dispatch_legacy(
        monkeypatch,
        "src.application.research.portfolio.evaluation.run_portfolio_evaluation",
        ["run_portfolio_backtest", "--symbols", "BTCUSDT", "ETHUSDT", "--no-log-run"],
    )
    from src.cli.adapters import run_portfolio_backtest

    run_portfolio_backtest.main()
    assert legacy[0].symbols == ("BTCUSDT", "ETHUSDT")
    assert legacy[0].log_run is False


def test_legacy_cash_carry_equivalent_to_research_run_cash_carry(monkeypatch) -> None:
    legacy = _dispatch_legacy(
        monkeypatch,
        "src.application.research.carry.evaluation.run_cash_carry_evaluation",
        ["run_cash_carry_backtest", "run", "--symbol", "BTCUSDT", "--no-log-run"],
    )
    from src.cli.adapters import run_cash_carry_backtest

    run_cash_carry_backtest.main()
    assert legacy[0].symbol == "BTCUSDT"
    assert legacy[0].log_run is False


def test_legacy_expert_portfolio_equivalent_to_research_run_expert_portfolio(
    monkeypatch,
) -> None:
    legacy = _dispatch_legacy(
        monkeypatch,
        "src.application.research.expert.evaluation.run_expert_portfolio_evaluation",
        ["run_expert_portfolio_backtest", "--library-id", "pair_residual_v1", "--no-log-run"],
    )
    from src.cli.adapters import run_expert_portfolio_backtest

    run_expert_portfolio_backtest.main()
    assert legacy[0].library_id == "pair_residual_v1"
    assert legacy[0].log_run is False

    grouped: list[object] = []
    monkeypatch.setattr(
        "src.application.research.expert.evaluation.run_expert_portfolio_evaluation",
        grouped.append,
    )
    root_main([
        "research", "run", "expert", "eval",
        "--library-id", "pair_residual_v1", "--no-log-run",
    ])
    assert grouped == legacy


def test_legacy_sleeve_blend_equivalent_to_research_run_sleeve_blend(monkeypatch) -> None:
    legacy = _dispatch_legacy(
        monkeypatch,
        "src.application.research.blend.evaluation.run_sleeve_blend_evaluation",
        ["run_sleeve_blend_backtest", "--candidate-kind", "funding_signed_directional_v1",
         "--no-log-run"],
    )
    from src.cli.adapters import run_sleeve_blend_backtest

    run_sleeve_blend_backtest.main()
    assert legacy[0].candidate_kind == "funding_signed_directional_v1"


def test_legacy_collect_data_equivalent_to_data_collect(monkeypatch) -> None:
    from src.application.data import collection

    calls: list[str] = []
    monkeypatch.setattr(
        collection, "collect_spot_ohlcv",
        lambda *args: calls.append("spot_ohlcv"),
    )
    monkeypatch.setattr(sys, "argv", [
        "collect_data", "spot-ohlcv", "BTCUSDT", "1h", "--start", "2024-01-01",
    ])
    from src.cli.adapters import collect_data

    collect_data.main()
    assert calls == ["spot_ohlcv"]

    monkeypatch.setattr(sys, "argv", ["collect_data", "ohlcv", "BTCUSDT", "1h"])
    monkeypatch.setattr(collection, "collect_ohlcv", lambda *args: calls.append("futures_ohlcv"))
    collect_data.main()
    assert calls == ["spot_ohlcv", "futures_ohlcv"]


def test_sealed_holdout_policy_shared_across_clis() -> None:
    from src.research.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end

    assert resolve_evaluation_end(None, unseal_holdout=False) == HOLDOUT_CUTOFF
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        resolve_evaluation_end("2026-01-01", unseal_holdout=False)
    assert resolve_evaluation_end("2026-01-01", unseal_holdout=True) == "2026-01-01"


def test_legacy_compare_runs_renders_populated(monkeypatch, capsys) -> None:
    from src.cli.adapters import compare_runs

    populated = pd.DataFrame([{
        "ts": "2026-07-31T00:00:00+00:00", "git_sha": "abc", "git_dirty": False,
        "symbol": "BTCUSDT", "end": "2025-12-31",
        "metrics.trade_count": 30, "metrics.cagr": 0.2, "metrics.mdd": -0.1,
        "metrics.sharpe": 1.5, "metrics.profit_factor": 1.5, "metrics.win_rate": 0.5,
        "reliability.observation.verdict": "PASS",
        "reliability.observation.lcb90_cagr": 0.16,
        "reliability.fold_distribution.max_period_contribution": 0.2,
        "reliability.stress_test.verdict": "PASS",
    }])
    monkeypatch.setattr(
        "src.cli.commands.provenance.load_evaluation_runs",
        lambda ledger_path=None: populated,
    )
    monkeypatch.setattr(sys, "argv", ["compare_runs", "--last", "5"])
    compare_runs.main()
    assert "BTCUSDT" in capsys.readouterr().out


_LEGACY_MODULES = (
    "collect_data",
    "run_backtest",
    "run_cash_carry_backtest",
    "run_portfolio_backtest",
    "run_sleeve_blend_backtest",
    "run_expert_portfolio_backtest",
    "compare_runs",
    "register_directional_candidate",
)
_ALLOWED_FROM_MODULES = ("src.cli", "logging", "src.research.evaluation.policy")
_ALLOWED_IMPORTS = ("src.cli.compat", "logging")


def test_legacy_cli_modules_are_parser_free_adapters() -> None:
    """RF-CLI-01: shims contain no parser or business logic, only forwarding."""
    for name in _LEGACY_MODULES:
        path = Path("src/cli/adapters") / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name in _ALLOWED_IMPORTS, (
                        f"{path}: adapter imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                assert node.module in _ALLOWED_FROM_MODULES, (
                    f"{path}: adapter imports {node.module}"
                )
            elif isinstance(node, ast.FunctionDef) and node.name == "main":
                # main() only calls a compat forwarder.
                assert any(
                    isinstance(c, ast.Expr)
                    and isinstance(c.value, ast.Call)
                    and isinstance(c.value.func, ast.Attribute)
                    and isinstance(c.value.func.value, ast.Name)
                    and c.value.func.value.id == "compat"
                    for c in node.body
                ), f"{path}: main() must only forward to compat"


def test_legacy_register_directional_candidate_is_importable() -> None:
    """RF-CLI-01: the retired-migration adapter remains importable at its new home."""
    from src.cli.adapters import register_directional_candidate as adapter

    assert callable(adapter.main)


def test_legacy_register_directional_candidate_dispatches_retired_migration(
    monkeypatch,
) -> None:
    """RF-CLI-01: invoking the adapter runs the idempotent RETIRED migration."""
    from src.cli import compat
    from src.cli.adapters import register_directional_candidate as adapter

    calls: list[object] = []
    monkeypatch.setattr(compat, "register_directional_candidate", lambda *a: calls.append(a))
    adapter.main()
    assert len(calls) == 1


def test_compat_dispatcher_remains_importable() -> None:
    """RF-CLI-01: the compat dispatcher keeps every legacy forwarder importable."""
    import src.cli.compat as compat_module

    for name in (
        "run_collect_data",
        "run_backtest",
        "run_cash_carry_backtest",
        "run_portfolio_backtest",
        "run_sleeve_blend_backtest",
        "run_expert_portfolio_backtest",
        "compare_runs",
        "register_directional_candidate",
    ):
        assert callable(getattr(compat_module, name))
