"""Wiring smoke test for src.mhs.pipeline.stages.fold.run_folds.

Verifies the S7 stage reaches ``run_post_book_concurrently``/
``guard_stage_or_breach``/``fold_blend_parity``/``fold_growth_concentration``/
``load_feature_panels``/``committee_diagnostic`` through the
``stage_services`` seam after the P4 refactor (previously private
``evaluation.`` attribute lookups).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import src.mhs.pipeline.stages.fold as fold_stage
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


class _RecordingRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, stage: str, **_kwargs: object) -> None:
        self.calls.append(stage)


def _bare_context(*, committee_book: bool = False) -> PipelineContext:
    frame = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    ctx = PipelineContext(
        config=dataclasses.replace(
            MhsRunConfig(), multi_feature_book=False, committee_book=committee_book,
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
    ctx.recorder = _RecordingRecorder()
    ctx.blend_report = None
    ctx.execution_symbols = _SYMS
    ctx.minute_grid = _GRID
    ctx.signal_48h = frame
    ctx.eligible = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.bar_funding = frame
    ctx.fast = "fast-spec"
    ctx.fold_funding = {}
    ctx.initial_equity = 1.0
    ctx.fold_slow_horizons = {}
    ctx.fold_fast_horizons = {}
    ctx.fold_funding_carry = {}
    ctx._fold_committee_weights = None
    ctx.blend_traces = {}
    ctx.book_reasons = ()
    ctx.aligned_symbols = _SYMS
    ctx.execution_mask = pd.DataFrame(True, index=_GRID, columns=_SYMS)
    ctx.funding_window = {}
    return ctx


def test_run_folds_reaches_seam_functions_default_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_run_post_book_concurrently(*_a: object, **_k: object):
        calls.append("_run_post_book_concurrently")
        return (None, None, {}, {}, [], "deployment-stub")

    def _fake_guard_stage_or_breach(*_a: object, **_k: object) -> None:
        calls.append("_guard_stage_or_breach")
        return None

    def _fake_fold_blend_parity(*_a: object, **_k: object) -> tuple[object, tuple]:
        calls.append("_fold_blend_parity")
        return ("parity-stub", ())

    def _fake_fold_growth_concentration(*_a: object, **_k: object) -> tuple[object, tuple]:
        calls.append("_fold_growth_concentration")
        return ("concentration-stub", ())

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("committee/multi-feature diagnostics must not run when both flags are off")

    monkeypatch.setattr(fold_stage, "_run_post_book_concurrently", _fake_run_post_book_concurrently)
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", _fake_guard_stage_or_breach)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", _fake_fold_blend_parity)
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", _fake_fold_growth_concentration)
    monkeypatch.setattr(fold_stage, "_load_feature_panels", _boom)
    monkeypatch.setattr(fold_stage, "_committee_diagnostic", _boom)
    monkeypatch.setattr(fold_stage, "_multi_feature_diagnostic", _boom)
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=False)
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))

    assert calls == [
        "_run_post_book_concurrently", "_guard_stage_or_breach",
        "_fold_blend_parity", "_fold_growth_concentration",
    ]
    assert ctx.folds == ()
    assert ctx.fold_blend_parity == "parity-stub"
    assert ctx.fold_growth_concentration == "concentration-stub"
    assert ctx.deployment is not None


def test_run_folds_reaches_committee_diagnostic_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        fold_stage, "_run_post_book_concurrently",
        lambda *_a, **_k: (None, None, {}, {}, [], "deployment-stub"),
    )
    monkeypatch.setattr(fold_stage, "_guard_stage_or_breach", lambda *_a, **_k: None)
    monkeypatch.setattr(fold_stage, "_fold_blend_parity", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(fold_stage, "_fold_growth_concentration", lambda *_a, **_k: (None, ()))
    monkeypatch.setattr(
        fold_stage, "_load_feature_panels",
        lambda *_a, **_k: calls.append("_load_feature_panels") or "panels-stub",
    )
    monkeypatch.setattr(
        fold_stage, "_committee_diagnostic",
        lambda *_a, **_k: calls.append("_committee_diagnostic") or "committee-diag-stub",
    )
    monkeypatch.setattr(
        fold_stage._statistics, "_deflated_sharpe_evidence", lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_mhs_research_go",
        lambda *_a, **_k: type("_RG", (), {"eligible": False})(),
    )
    monkeypatch.setattr(
        fold_stage._research_go, "_resolved_growth_envelope",
        lambda _config: type("_Env", (), {"max_drawdown": -0.5})(),
    )

    ctx = _bare_context(committee_book=True)
    fold_stage.run_folds(ctx, StageTelemetry(log_run=False))

    assert calls == ["_load_feature_panels", "_committee_diagnostic"]
    assert ctx.committee_diagnostic == "committee-diag-stub"
