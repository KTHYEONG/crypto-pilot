"""Wiring smoke test for src.mhs.pipeline.stages.book.build_books.

Verifies the S3 stage reaches ``signal_ema_span``/``book_weights``/
``horizon_ensemble_execution_weights`` through the ``stage_services`` seam
after the P4 refactor (previously private ``evaluation.`` attribute lookups).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import src.mhs.pipeline.stages.book as book_stage
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

_GRID = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
_SYMS = ["AAAUSDT", "BBBUSDT"]


class _FakeBand:
    def __init__(self, sign: int) -> None:
        self.sign = sign


class _FakeSpec:
    def __init__(self, sign: int, horizon_hours: int, step_hours: int) -> None:
        self.band = _FakeBand(sign)
        self.horizon_hours = horizon_hours
        self.step_hours = step_hours
        self.min_symbols = 1


def _bare_context() -> PipelineContext:
    frame = pd.DataFrame(1.0, index=_GRID, columns=_SYMS)
    ctx = PipelineContext(
        config=dataclasses.replace(
            MhsRunConfig(), fast_book_mode="single_horizon", execution_coverage_gate=False,
            beta_neutralize=False, committee_capital=False,
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
    ctx.fast_grid = _GRID
    ctx.slow_grid = _GRID
    ctx.fast = _FakeSpec(sign=1, horizon_hours=48, step_hours=6)
    ctx.slow = _FakeSpec(sign=1, horizon_hours=168, step_hours=24)
    return ctx


def test_build_books_reaches_seam_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_signal_ema_span(sign: int, horizon_hours: int, step_hours: int) -> int:
        calls.append("_signal_ema_span")
        return 4

    def _fake_book_weights(log_close, eligible, spec, grid, *, ema_span):
        calls.append("_book_weights")
        return pd.DataFrame(0.0, index=grid, columns=log_close.columns)

    def _fake_horizon_ensemble_execution_weights(*_a: object, **_k: object) -> pd.DataFrame:
        calls.append("_horizon_ensemble_execution_weights")
        return pd.DataFrame(0.0, index=_GRID, columns=_SYMS)

    monkeypatch.setattr(book_stage, "_signal_ema_span", _fake_signal_ema_span)
    monkeypatch.setattr(book_stage, "_book_weights", _fake_book_weights)
    monkeypatch.setattr(
        book_stage, "_horizon_ensemble_execution_weights", _fake_horizon_ensemble_execution_weights,
    )
    monkeypatch.setattr(
        book_stage, "_pit_execution_mask",
        lambda quote_vol, eligible, universe_size: pd.DataFrame(True, index=quote_vol.index, columns=quote_vol.columns),
    )
    monkeypatch.setattr(
        book_stage, "inverse_realized_vol_tilt", lambda w, vol: w,
    )
    monkeypatch.setattr(book_stage, "realized_vol", lambda log_close, horizon: pd.DataFrame(0.1, index=log_close.index, columns=log_close.columns))
    monkeypatch.setattr(
        book_stage, "renormalize_within_mask", lambda w, mask, min_symbols: w,
    )

    ctx = _bare_context()
    book_stage.build_books(ctx, StageTelemetry(log_run=False))

    assert calls == [
        "_signal_ema_span", "_signal_ema_span", "_book_weights", "_book_weights",
        "_horizon_ensemble_execution_weights",
    ]
    assert ctx.w_fast_execution is not None
    assert ctx.w_slow_execution is not None
