"""Wiring smoke test for src.mhs.pipeline.stages.panel.load_panel.

Verifies the S1 stage still threads the panel/funding load correctly after
the P4 refactor moved its ``_guard_stage_or_breach`` dependency behind the
``stage_services`` seam (previously a private ``evaluation.`` attribute
lookup). Every collaborator is monkeypatched so this stays a fast unit test;
the real end-to-end path is covered by tests/integration/mhs/test_golden_identity.py.
"""

from __future__ import annotations
import src.mhs.evaluation.guards as guards_mod

import pandas as pd
import pytest

import src.mhs.pipeline.stages.panel as panel_stage
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


def _bare_context() -> PipelineContext:
    return PipelineContext(
        config=MhsRunConfig(),
        resolved_end=None,
        start=_GRID[0],
        end=_GRID[-1],
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=pd.DatetimeIndex([]),
        close=pd.DataFrame(),
        opens=pd.DataFrame(),
        quote_vol=pd.DataFrame(),
        taker_buy_quote=None,
        symbols=[],
    )


def test_load_panel_threads_symbols_and_funding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(panel_stage, "_resolve_ram_budget", lambda *_a, **_k: (None, None), raising=False)
    monkeypatch.setattr(
        panel_stage,
        "load_base_panel",
        lambda *_a, **_k: {
            "close": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
            "open": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
            "quote_vol": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
            "taker_buy_quote": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
        },
    )
    monkeypatch.setattr(panel_stage, "_guard_stage_or_breach", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(guards_mod, "_guard_stage_or_breach", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        panel_stage,
        "_load_funding_series",
        lambda symbols: ({s: pd.Series(0.0001, index=_GRID) for s in symbols}, []), raising=False
    )
    monkeypatch.setattr(
        panel_stage,
        "bar_funding_panel",
        lambda funding_window, grid: pd.DataFrame(0.0001, index=grid, columns=list(funding_window)),
    )

    ctx = _bare_context()
    load_panel_telemetry = StageTelemetry(log_run=False)

    panel_stage.load_panel(ctx, load_panel_telemetry)

    assert ctx._terminal_report is None
    assert ctx.symbols == _SYMS
    assert ctx.funded == _SYMS
    assert ctx.aligned_symbols == _SYMS
    assert list(ctx.bar_funding.columns) == _SYMS
    assert ctx.recorder is not None


def test_load_panel_short_circuits_on_guard_breach(monkeypatch: pytest.MonkeyPatch) -> None:
    """A breach at the first checkpoint sets `_terminal_report` and returns
    before the funding load ever runs."""
    monkeypatch.setattr(panel_stage, "_resolve_ram_budget", lambda *_a, **_k: (None, None), raising=False)
    monkeypatch.setattr(
        panel_stage,
        "load_base_panel",
        lambda *_a, **_k: {
            "close": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
            "open": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
            "quote_vol": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
            "taker_buy_quote": pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
        },
    )
    monkeypatch.setattr(panel_stage, "_guard_stage_or_breach", lambda *_a, **_k: "TERMINAL", raising=False)
    monkeypatch.setattr(guards_mod, "_guard_stage_or_breach", lambda *_a, **_k: "TERMINAL", raising=False)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("funding load must not run after a guard breach")

    monkeypatch.setattr(panel_stage, "_load_funding_series", _boom, raising=False)

    ctx = _bare_context()
    panel_stage.load_panel(ctx, StageTelemetry(log_run=False))

    assert ctx._terminal_report == "TERMINAL"