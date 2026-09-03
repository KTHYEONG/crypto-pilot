"""Unit-level isolation for src.mhs.pipeline.stages.assemble.assemble_report.

The full six-stage pipeline is exercised end-to-end by
tests/integration/mhs/test_golden_identity.py; this module isolates
assemble_report's own field-mapping logic -- in particular the `worker_plan`
field wired from `ctx.recorder.worker_plan` for measurement correctness --
without running a real pipeline pass.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.mhs.resources import _StageRecorder
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.pipeline.stages.assemble import assemble_report
from src.mhs.params import COMMITTEE_OOS_START
from src.mhs.telemetry import StageTelemetry
from src.quant.evaluation.policy import HOLDOUT_CUTOFF


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


# SCENARIO_ASSEMBLE_WIRES_PARAMETER_OOS_SPLIT_AT_COMMITTEE_OOS_START
def test_assemble_wires_parameter_oos_split(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"boundary": "sentinel"}
    calls: list[tuple[pd.Series, pd.Timestamp]] = []

    def _spy(equity: pd.Series, boundary: pd.Timestamp) -> dict[str, Any] | None:
        calls.append((equity, boundary))
        return sentinel

    monkeypatch.setattr(
        "src.mhs.pipeline.stages.assemble.parameter_oos_split_evidence", _spy
    )

    recorder = _StageRecorder(log_run=False)
    ctx = _bare_context(recorder)
    equity = _stub_blend_equity()
    ctx.blend_report = SimpleNamespace(
        primary=SimpleNamespace(
            ledger=SimpleNamespace(equity=equity, mark_source="OHLCV"),
        ),
    )

    report = assemble_report(ctx, ctx.telemetry)

    # 스파이가 반환한 센티넬 dict가 그대로 보고서에 전달된다.
    assert report.parameter_oos_split is sentinel
    assert len(calls) == 1
    called_equity, called_boundary = calls[0]
    assert called_boundary == COMMITTEE_OOS_START
    assert called_equity is equity

    # blend_report가 None이면 스파이 호출 없이 None.
    calls.clear()
    bare_ctx = _bare_context(_StageRecorder(log_run=False))
    bare_report = assemble_report(bare_ctx, bare_ctx.telemetry)
    assert bare_report.parameter_oos_split is None
    assert calls == []

    # I-OBSERVATIONAL: split 결과와 무관하게 research_go는 동일(게이트 아님).
    go_with_sentinel = report.research_go
    ctx2 = _bare_context(_StageRecorder(log_run=False))
    ctx2.blend_report = SimpleNamespace(
        primary=SimpleNamespace(
            ledger=SimpleNamespace(equity=equity, mark_source="OHLCV"),
        ),
    )
    monkeypatch.setattr(
        "src.mhs.pipeline.stages.assemble.parameter_oos_split_evidence",
        lambda *_args: None,
    )
    report_none = assemble_report(ctx2, ctx2.telemetry)
    assert report_none.parameter_oos_split is None
    assert report_none.research_go == go_with_sentinel


# SCENARIO_MHS_TRIAL_POOL_DISCLOSURE_IN_REPORT_AND_HISTORY (assemble wiring)
def test_SCENARIO_MHS_TRIAL_POOL_DISCLOSURE_IN_REPORT_AND_HISTORY(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.mhs.contracts import MhsResearchGoResult
    from src.mhs.run_history import (
        append_run_history_record,
        trial_pool_disclosure,
    )

    recorder = _StageRecorder(log_run=False)
    ctx = _bare_context(recorder)
    go_before = MhsResearchGoResult(
        eligible=False,
        reason_codes=("SOME_BLOCKING_CODE",),
        evaluated_folds=0,
        folds_passed=0,
    )
    ctx.research_go = go_before

    # Unset disclosure stays None on the report.
    unset_report = assemble_report(ctx, ctx.telemetry)
    assert unset_report.trial_pool is None

    # The disclosure payload is attached verbatim and adds no GO reason code.
    history_dir = tmp_path / "history"
    window = ("2021-01-01T00:00:00+00:00", "2025-12-31T23:59:59+00:00")
    gap_code = "RELEVANT_EXECUTION_DATA_GAP"

    def _record(run_id: str, flags: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": overrides.get("status", "COMPLETE"),
            "flags": flags,
            "start": window[0],
            "resolved_end": overrides.get("resolved_end", window[1]),
            "blend": {"primary_naive_sharpe": overrides.get("sharpe", 2.0)},
            "research_go": {
                "reason_codes": list(overrides.get("reason_codes", ())),
                "data_integrity_reason_codes": [],
            },
        }

    for record in (
        _record("clean1", {"u": 1}, sharpe=1.5),
        _record("gap", {"u": 2}, reason_codes=(gap_code,)),
        _record("halted", {"u": 3}, status="RUNNING"),
        _record("nonfinite", {"u": 4}, sharpe=None),
    ):
        append_run_history_record(record, history_dir)

    disclosure = trial_pool_disclosure(window, history_dir)
    expected_keys = {
        "n_history_records",
        "n_trial_records",
        "excluded_data_integrity",
        "excluded_not_complete",
        "excluded_nonfinite_blend",
        "distinct_trial_keys",
        "neutral_flags_dropped",
        "pool_window_span_days",
        "ledger_size",
        "source",
    }
    assert expected_keys <= set(disclosure)
    assert disclosure["n_history_records"] == 4
    assert disclosure["n_trial_records"] == 1
    assert disclosure["excluded_data_integrity"] == 1
    assert disclosure["excluded_not_complete"] == 1
    assert disclosure["excluded_nonfinite_blend"] == 1
    assert (
        disclosure["n_trial_records"]
        + disclosure["excluded_data_integrity"]
        + disclosure["excluded_not_complete"]
        + disclosure["excluded_nonfinite_blend"]
        == disclosure["n_history_records"]
    )
    assert disclosure["source"] == "constant_plus_ledger"

    ctx.trial_pool = disclosure
    wired_report = assemble_report(ctx, ctx.telemetry)
    assert wired_report.trial_pool is disclosure
    assert wired_report.trial_pool == disclosure
    # I-OBSERVATIONAL: the disclosure never touches the GO gate decision.
    assert wired_report.research_go.reason_codes == go_before.reason_codes
