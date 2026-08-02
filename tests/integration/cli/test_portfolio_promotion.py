from __future__ import annotations

import logging
import sys

import pytest

from src.cli.adapters import run_portfolio_backtest as cli


@pytest.mark.slow
def test_sealed_portfolio_cli_composes_all_required_gates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # SC-PORT-07: a sealed portfolio run reports the liquid universe and all gate
    # evidence and never unseals the holdout. Fail-closed means HOLDOUT_PASS is
    # unreachable without the holdout.
    monkeypatch.setattr(sys, "argv", [
        "run_portfolio_backtest",
        "--no-log-run",
        "--symbols", "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    ])
    with caplog.at_level(logging.INFO):
        cli.main()

    assert "[EVAL] portfolio universe as_of=" in caplog.text
    assert "selected=[]" not in caplog.text
    assert "[EVAL] reliability observation=" in caplog.text
    assert "[EVAL] reliability fold max_period_contribution=" in caplog.text
    assert "[EVAL] reliability stress_test=" in caplog.text
    assert "[EVAL] promotion status=" in caplog.text
    assert "observation=" in caplog.text
    assert "fold_gate=" in caplog.text
    assert "stress=" in caplog.text
    assert "[EVAL] holdout unsealed" not in caplog.text
    assert "[EVAL] reliability holdout=" not in caplog.text
