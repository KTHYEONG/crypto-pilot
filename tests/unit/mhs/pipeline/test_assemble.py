"""Unit-level isolation for src.mhs.pipeline.stages.assemble.assemble_report.

The full six-stage pipeline is exercised end-to-end by
tests/integration/mhs/test_golden_identity.py; this module isolates
assemble_report's own field-mapping logic -- in particular the `worker_plan`
field wired from `ctx.recorder.worker_plan` for measurement correctness --
without running a real pipeline pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.application.research.mhs.resources import _StageRecorder
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.pipeline.stages.assemble import assemble_report
from src.mhs.telemetry import StageTelemetry
from src.research.evaluation.policy import HOLDOUT_CUTOFF


def _bare_context(recorder: _StageRecorder) -> PipelineContext:
    """Minimal PipelineContext: only the fields assemble_report reads."""
    grid = pd.DatetimeIndex([])
    ctx = PipelineContext(
        config=MhsRunConfig(),
        resolved_end=None,
        start=pd.Timestamp("2021-01-01", tz="UTC"),
        end=pd.Timestamp("2021-01-02", tz="UTC"),
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=grid,
        close=pd.DataFrame(),
        opens=pd.DataFrame(),
        quote_vol=pd.DataFrame(),
        taker_buy_quote=None,
        symbols=[],
    )
    ctx.run_start = 0.0
    ctx.recorder = recorder
    ctx.telemetry = StageTelemetry(log_run=False)
    return ctx


def test_assemble_report_carries_recorder_worker_plan() -> None:
    """SCENARIO_MHS_PERF_P0_04: the granted-worker-count decisions recorded
    during the run land on the assembled report's `worker_plan` field."""
    recorder = _StageRecorder(log_run=False)
    recorder.record_worker_plan(
        "books", requested=3, granted=1, available_bytes=6_000_000_000, reserve_bytes=1_000_000_000,
    )
    ctx = _bare_context(recorder)

    report = assemble_report(ctx, ctx.telemetry)

    assert report.worker_plan == {"books": 1}


def test_assemble_report_worker_plan_empty_when_no_fork_point_ran() -> None:
    """A run with no fork-point decisions (e.g. no book replay) yields an
    empty worker_plan, never a crash."""
    recorder = _StageRecorder(log_run=False)
    ctx = _bare_context(recorder)

    report = assemble_report(ctx, ctx.telemetry)

    assert report.worker_plan == {}


def _stub_blend_equity() -> pd.Series:
    """봉인 경계를 넘는 합성 equity: cutoff 이후 양의 드리프트 꼬리 포함."""
    rng = np.random.default_rng(20260825)
    hours = pd.date_range("2025-01-01 00:00", "2026-06-30 23:00", freq="1h", tz="UTC")
    post = hours > HOLDOUT_CUTOFF
    hourly = np.where(post, 0.0002, rng.normal(0.0, 0.001, len(hours)))
    return pd.Series(np.cumprod(1.0 + hourly), index=hours)


# SCENARIO_ASSEMBLE_REPORT_HOLDOUT_TAIL_WIRING
def test_assemble_report_holdout_tail_none_when_no_blend_report() -> None:
    recorder = _StageRecorder(log_run=False)
    ctx = _bare_context(recorder)
    assert ctx.blend_report is None

    report = assemble_report(ctx, ctx.telemetry)

    assert report.holdout_tail is None


# SCENARIO_ASSEMBLE_REPORT_HOLDOUT_TAIL_WIRING (populated past the boundary)
def test_assemble_report_holdout_tail_populated_past_sealed_boundary() -> None:
    recorder = _StageRecorder(log_run=False)
    ctx = _bare_context(recorder)
    ctx.blend_report = SimpleNamespace(
        primary=SimpleNamespace(
            ledger=SimpleNamespace(equity=_stub_blend_equity(), mark_source="OHLCV"),
        ),
    )

    report = assemble_report(ctx, ctx.telemetry)

    assert report.holdout_tail is not None
    assert {"n_days", "geometric_cagr", "max_drawdown", "naive_sharpe"} <= set(
        report.holdout_tail
    )
