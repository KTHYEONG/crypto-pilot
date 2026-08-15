from __future__ import annotations

import logging
import sys

import pytest

from src.cli.adapters import run_backtest as cli


@pytest.mark.slow
def test_sealed_cli_logs_promotion_without_holdout_override(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # SC-CLI-01: a sealed default run logs the promotion result and never
    # unseals the holdout. v1 baseline remains REJECTED: that proves the
    # gate was not weakened.
    monkeypatch.setattr(sys, "argv", ["run_backtest", "--no-log-run"])
    with caplog.at_level(logging.INFO):
        cli.main()

    assert "[EVAL] promotion status=REJECTED" in caplog.text
    assert "observation=" in caplog.text
    assert "fold_gate=" in caplog.text
    assert "stress=" in caplog.text
    assert "[EVAL] holdout unsealed" not in caplog.text
    assert "[EVAL] reliability holdout=" not in caplog.text
