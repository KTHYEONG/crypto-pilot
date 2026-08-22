"""Wiring smoke test for src.mhs.pipeline.stages.committee.build_committee.

Verifies the S4 stage reaches ``phase_diagnostics``/``active_blend_book_and_grid``
through the ``stage_services`` seam after the P4 refactor (previously private
``evaluation.`` attribute lookups). Every collaborator besides the seam is
monkeypatched so this exercises the non-committee-capital branch as a fast,
deterministic unit test; the full numeric behaviour is covered by the
evaluation/golden-identity suites.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import src.mhs.pipeline.stages.committee as committee_stage
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


class _FakeBand:
    def __init__(self, sign: int) -> None:
        self.sign = sign


class _FakeSpec:
    def __init__(self, horizon_hours: int) -> None:
        self.band = _FakeBand(1)
        self.horizon_hours = horizon_hours
        self.min_symbols = 1


def _bare_context() -> PipelineContext:
    frame = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    ctx = PipelineContext(
        config=dataclasses.replace(
            MhsRunConfig(),
            committee_capital=False,
            discovery_gate=False,
            trend_sleeve=False,
            trend_efficiency_overlay=False,
        ),
        resolved_end=None,
        start=_GRID[0],
        end=_GRID[-1],
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=_GRID,
        close=frame,
        opens=frame,
        quote_vol=frame,
        taker_buy_quote=None,
        symbols=_SYMS,
    )
    ctx.log_close = frame
    ctx.eligible = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.bar_funding = pd.DataFrame(0.0001, index=_GRID, columns=_SYMS)
    ctx.fast_grid = _GRID
    ctx.slow_grid = _GRID
    ctx.fast = _FakeSpec(48)
    ctx.slow = _FakeSpec(168)
    ctx.w_fast_1h = frame
    ctx.w_slow_1h = frame
    ctx.execution_mask = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    return ctx


def test_build_committee_reaches_seam_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_phase_diagnostics(*_a: object, **_k: object) -> str:
        calls.append("_phase_diagnostics")
        return "phase-result"

    def _fake_active_blend_book_and_grid(fast, slow, fast_grid, slow_grid):
        calls.append("_active_blend_book_and_grid")
        return slow, slow_grid

    monkeypatch.setattr(committee_stage, "_phase_diagnostics", _fake_phase_diagnostics)
    monkeypatch.setattr(
        committee_stage, "_active_blend_book_and_grid", _fake_active_blend_book_and_grid,
    )
    monkeypatch.setattr(
        committee_stage, "realized_vol",
        lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "horizon_log_return",
        lambda log_close, horizon: pd.DataFrame(0.0, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage, "efficiency_ratio",
        lambda log_close, horizon: pd.DataFrame(0.5, index=log_close.index, columns=log_close.columns),
    )
    monkeypatch.setattr(
        committee_stage._statistics, "_xs_rank_ic", lambda *_a, **_k: {"mean_ic": 0.0},
    )
    monkeypatch.setattr(
        committee_stage._statistics, "_date_clustered_ols", lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        committee_stage._scaling, "_regime_cash_scale",
        lambda vol_mean: pd.Series(1.0, index=vol_mean.index),
    )

    ctx = _bare_context()
    committee_stage.build_committee(ctx, StageTelemetry(log_run=False))

    assert calls == [
        "_phase_diagnostics", "_phase_diagnostics", "_active_blend_book_and_grid", "_phase_diagnostics",
    ]
    assert ctx.phase_fast == "phase-result"
    assert ctx.phase_slow == "phase-result"
    assert ctx.phase_blend == "phase-result"
    assert ctx.committee_execution_book is None
