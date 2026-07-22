from __future__ import annotations

from src.application.futures.runner.compound_main import run_compound_main


def test_compound_main_is_callable() -> None:
    assert callable(run_compound_main)
