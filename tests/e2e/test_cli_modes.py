from __future__ import annotations

import sys
from typing import Any

import pytest

from src.execution import opt_main_futures


def test_rejects_legacy_alpha_only_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "quick-backtest", "--symbols", "BTCUSDT", "--alpha-only"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_rejects_legacy_hmm_only_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "quick-backtest", "--symbols", "BTCUSDT", "--hmm-only"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_strategy_mode_uses_default_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_run_pipeline(
        run_config: Any,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> opt_main_futures.RunnerResult:
        _ = seed
        _ = resume
        captured["strategy"] = run_config.strategy
        return opt_main_futures.RunnerResult(exit_code=0, reason="ok")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "strategy", "--symbols", "BTCUSDT"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 0
    assert captured["strategy"] == "ml_lambdamart_v1"


def test_strategy_smoke_mode_uses_default_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run_pipeline(
        run_config: Any,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> opt_main_futures.RunnerResult:
        _ = seed
        _ = resume
        captured["strategy"] = run_config.strategy
        return opt_main_futures.RunnerResult(exit_code=0, reason="ok")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "strategy-smoke", "--symbols", "BTCUSDT"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 0
    assert captured["strategy"] == "ml_lambdamart_v1"


def test_rejects_legacy_full_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "full", "--symbols", "BTCUSDT"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_strategy_smoke_mode_enters_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_mode = {"value": None}

    def fake_run_pipeline(
        run_config: Any,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> opt_main_futures.RunnerResult:
        _ = seed
        _ = resume
        captured_mode["value"] = run_config.mode
        return opt_main_futures.RunnerResult(exit_code=0, reason="strategy_smoke_done")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opt_main_futures.py",
            "--mode",
            "strategy-smoke",
            "--strategy",
            "momentum_v0",
            "--symbols",
            "BTCUSDT",
            "--trials",
            "1",
        ],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 0
    assert captured_mode["value"] == "strategy-smoke"
