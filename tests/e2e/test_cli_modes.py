from __future__ import annotations

import sys

from src.execution import opt_main_futures


def test_rejects_legacy_alpha_only_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "quick-backtest", "--symbols", "BTCUSDT", "--alpha-only"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_rejects_legacy_hmm_only_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "quick-backtest", "--symbols", "BTCUSDT", "--hmm-only"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_strategy_mode_requires_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "strategy", "--symbols", "BTCUSDT"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_strategy_smoke_mode_requires_strategy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "strategy-smoke", "--symbols", "BTCUSDT"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_rejects_legacy_full_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "full", "--symbols", "BTCUSDT"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_strategy_smoke_mode_enters_pipeline(monkeypatch) -> None:
    captured_mode = {"value": None}

    def fake_run_pipeline(run_config, *, seed: int = 42, resume: bool = False):
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
