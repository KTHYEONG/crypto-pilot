from __future__ import annotations

import importlib


def test_opt_main_futures_remains_thin_entrypoint() -> None:
    module = importlib.import_module("src.execution.opt_main_futures")

    assert hasattr(module, "main")
    assert not hasattr(module, "_run_strategy_stage")
