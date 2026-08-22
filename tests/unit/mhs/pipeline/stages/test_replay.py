"""Wiring smoke test for src.mhs.pipeline.stages.replay.run_replays.

Verifies the S6 stage reaches ``guard_stage_or_breach``/``run_books_concurrent``
through the ``stage_services`` seam after the P4 refactor (previously private
``evaluation.`` attribute lookups).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import src.mhs.pipeline.stages.replay as replay_stage
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


def _bare_context() -> PipelineContext:
    frame = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    ctx = PipelineContext(
        config=dataclasses.replace(
            MhsRunConfig(), committee_capital=True, execution_timeframe="3m",
            committee_member_attribution=False,
        ),
        resolved_end=None,
        start=_GRID[0],
        end=_GRID[-1],
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="/does/not/matter",
        grid_1h=_GRID,
        close=frame,
        opens=frame,
        quote_vol=frame,
        taker_buy_quote=None,
        symbols=_SYMS,
    )
    ctx.w_fast_execution = frame
    ctx.w_slow_execution = frame
    ctx.w_fast = frame
    ctx.w_slow = frame
    ctx.blend_1h = frame
    ctx.fast_grid = _GRID
    ctx.slow_grid = _GRID
    ctx.fast = "fast-spec"
    ctx.slow = "slow-spec"
    ctx.bar_funding = frame
    ctx.phase_fast = "phase-fast"
    ctx.phase_slow = "phase-slow"
    ctx.phase_blend = "phase-blend"
    ctx.funding_by_symbol = {}
    ctx.execution_mask = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.regime_scale = pd.Series(1.0, index=_GRID)
    ctx.committee_execution_book = frame
    ctx.committee_member_books = None
    ctx.funded = _SYMS
    return ctx


class _FakeBookReport:
    """Stands in for MhsBookReport: only `.failure` is read after replay."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failure = None


def test_run_replays_reaches_seam_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(replay_stage.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(replay_stage, "_prewarm_mark_frames", lambda symbols: calls.append("prewarm"))

    def _fake_guard_stage_or_breach(*_a: object, **_k: object) -> None:
        calls.append("_guard_stage_or_breach")
        return None

    fast_report = _FakeBookReport("fast")
    slow_report = _FakeBookReport("slow")
    blend_report = _FakeBookReport("blend")

    def _fake_run_books_concurrent(*_a: object, **_k: object):
        calls.append("_run_books_concurrent")
        return (fast_report, slow_report, blend_report, {}, {})

    monkeypatch.setattr(replay_stage, "_guard_stage_or_breach", _fake_guard_stage_or_breach)
    monkeypatch.setattr(replay_stage, "_run_books_concurrent", _fake_run_books_concurrent)

    ctx = _bare_context()
    ctx.recorder = type("_R", (), {"record": lambda self, *a, **k: None})()

    replay_stage.run_replays(ctx, StageTelemetry(log_run=False))

    assert calls == ["_guard_stage_or_breach", "prewarm", "_run_books_concurrent", "_guard_stage_or_breach"]
    assert ctx.books == {"fast_reversal": fast_report, "slow_momentum": slow_report}
    assert ctx.blend_report is blend_report
    assert ctx.committee_member_attribution is None
    assert ctx._terminal_report is None
