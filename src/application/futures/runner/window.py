from __future__ import annotations

from typing import Any

from src.application.futures.runner.models import RunWindow


def resolve_run_window(reference_date: str | None) -> RunWindow:
    raise NotImplementedError("resolve_run_window: delegate to opt_main_futures window logic")


def resolve_layered_window(reference_date: str | None) -> Any | None:
    raise NotImplementedError("resolve_layered_window: delegate to opt_main_futures layered window logic")
