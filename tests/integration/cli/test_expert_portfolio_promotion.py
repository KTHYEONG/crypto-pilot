from __future__ import annotations

import sys

import pytest

from src.cli import run_expert_portfolio_backtest as cli
from src.research.evaluation.policy import resolve_evaluation_end


def test_cli_refuses_unregistered_library_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EP-CLI-01: the sealed CLI only accepts a registered library id; running it
    # with an unregistered library must fail closed before any promotion claim.
    monkeypatch.setattr(sys, "argv", [
        "run_expert_portfolio_backtest", "--library-id", "no_such_library", "--no-log-run",
    ])
    with pytest.raises(ValueError, match="not registered"):
        cli.main()


def test_cli_uses_the_shared_sealed_holdout_policy() -> None:
    # EP-CLI-02: the expert-portfolio CLI shares the same sealed-window policy as
    # every other evaluation CLI; an explicit end past the cutoff stays sealed.
    assert resolve_evaluation_end(None, unseal_holdout=False) is not None
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        resolve_evaluation_end("2026-01-01", unseal_holdout=False)


def test_cli_requires_registered_library_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_expert_portfolio_backtest", "--no-log-run"])
    with pytest.raises(SystemExit):
        cli.main()
