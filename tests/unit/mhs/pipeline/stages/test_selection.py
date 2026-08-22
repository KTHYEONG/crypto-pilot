"""Wiring smoke test for src.mhs.pipeline.stages.selection.select_horizons.

Verifies the S2 stage's fold-safe horizon path still reaches
``candidate_weight_books``/``run_fold_safe_discovery_parallel`` through the
``stage_services`` seam after the P4 refactor (previously a private
``evaluation.`` attribute lookup). The default (opt-out) path is left to the
existing evaluation/golden-identity suites; this isolates the opt-in wiring.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

import src.mhs.pipeline.stages.selection as selection_stage
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


def _bare_context(*, fold_safe: bool) -> PipelineContext:
    close = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    ctx = PipelineContext(
        config=dataclasses.replace(MhsRunConfig(), fold_safe_horizon_selection=fold_safe),
        resolved_end=None,
        start=_GRID[0],
        end=_GRID[-1],
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=_GRID,
        close=close,
        opens=pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
        quote_vol=pd.DataFrame(1.0, index=_GRID, columns=_SYMS),
        taker_buy_quote=None,
        symbols=_SYMS,
    )
    ctx.bar_funding = pd.DataFrame(0.0001, index=_GRID, columns=_SYMS)
    return ctx


def test_select_horizons_reaches_fold_safe_discovery_via_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_stage,
        "liquid_half_eligibility",
        lambda quote_vol, **_k: pd.DataFrame(True, index=quote_vol.index, columns=quote_vol.columns),
    )
    monkeypatch.setattr(
        selection_stage,
        "_fill_mark_parity_eligibility",
        lambda close, eligible, _gate: (eligible, None),
    )

    calls: list[str] = []

    def _fake_candidate_weight_books(*_a: object, **_k: object) -> str:
        calls.append("_candidate_weight_books")
        return "candidate-books"

    def _fake_run_fold_safe_discovery_parallel(*_a: object, **_k: object) -> tuple[dict, dict, dict]:
        calls.append("_run_fold_safe_discovery_parallel")
        return ({2: 168}, {2: (48, "raw")}, {2: (None, None, "raw", None)})

    monkeypatch.setattr(selection_stage, "_candidate_weight_books", _fake_candidate_weight_books)
    monkeypatch.setattr(
        selection_stage, "_run_fold_safe_discovery_parallel", _fake_run_fold_safe_discovery_parallel,
    )

    ctx = _bare_context(fold_safe=True)
    ctx.recorder = None

    selection_stage.select_horizons(ctx, StageTelemetry(log_run=False))

    assert calls == ["_candidate_weight_books", "_run_fold_safe_discovery_parallel"]
    assert ctx.candidate_books == "candidate-books"
    assert ctx.fold_slow_horizons == {2: 168}
    assert ctx.top_level_horizon == 168
    assert ctx.slow.horizon_hours == 168
    assert np.array_equal(ctx.log_close.to_numpy(), np.log(pd.DataFrame(1.0, index=_GRID, columns=_SYMS).to_numpy()))


def test_select_horizons_skips_fold_safe_discovery_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_stage,
        "liquid_half_eligibility",
        lambda quote_vol, **_k: pd.DataFrame(True, index=quote_vol.index, columns=quote_vol.columns),
    )
    monkeypatch.setattr(
        selection_stage,
        "_fill_mark_parity_eligibility",
        lambda close, eligible, _gate: (eligible, None),
    )

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("fold-safe discovery must not run when the flag is off")

    monkeypatch.setattr(selection_stage, "_candidate_weight_books", _boom)
    monkeypatch.setattr(selection_stage, "_run_fold_safe_discovery_parallel", _boom)

    ctx = _bare_context(fold_safe=False)
    selection_stage.select_horizons(ctx, StageTelemetry(log_run=False))

    assert ctx.candidate_books is None
    assert ctx.fold_slow_horizons == {}
