from __future__ import annotations

import importlib


def test_opt_main_futures_exposes_single_cli_entrypoint() -> None:
    module = importlib.import_module("src.execution.opt_main_futures")

    assert callable(module.cli)
    assert "run_multiscale_cli" not in module.__dict__
