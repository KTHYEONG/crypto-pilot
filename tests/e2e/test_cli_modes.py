from __future__ import annotations

import sys
from typing import Any

import pytest

from src.execution import opt_main_futures


def test_rejects_legacy_alpha_only_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--alpha-only는 제거되어 unrecognized argument 오류(exit 2)를 반환해야 한다."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--phase", "strategy", "--alpha-only"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_rejects_legacy_hmm_only_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--phase", "strategy", "--hmm-only"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_rejects_legacy_mode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--mode", "strategy"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 2


def test_strategy_mode_uses_default_full(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_run_pipeline(
        run_config: Any,
        *,
        seed: int = 42,
        resume: bool = False,
    ) -> opt_main_futures.RunnerResult:
        _ = seed
        _ = resume
        captured["phase"] = run_config.phase
        return opt_main_futures.RunnerResult(exit_code=0, reason="ok")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--phase", "full"],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 0
    assert captured["phase"] == "full"


def test_accepts_full_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["opt_main_futures.py", "--phase", "full"],
    )
    # This should now succeed as full is an active phase
    monkeypatch.setattr(
        opt_main_futures,
        "run_pipeline",
        lambda *a, **kw: opt_main_futures.RunnerResult(exit_code=0, reason="ok"),
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 0


def test_full_mode_enters_pipeline(
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
        captured_mode["value"] = run_config.phase
        return opt_main_futures.RunnerResult(exit_code=0, reason="full_done")

    monkeypatch.setattr(opt_main_futures, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opt_main_futures.py",
            "--phase",
            "full",
            "--trials",
            "1",
        ],
    )
    exit_code = opt_main_futures.main()
    assert exit_code == 0
    assert captured_mode["value"] == "full"
