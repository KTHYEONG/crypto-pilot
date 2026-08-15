from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.application.research.expert import exit_sweep as app
from src.research.contracts import CostModel
from src.research.expert_portfolio.admission_reports import (
    ExitSweepCellResult,
    ExitSweepFamilySummary,
    TechnicalExpertExitSweepReport,
)
from src.research.expert_portfolio.admission_types import (
    ExitSweepSetting,
    TechnicalExpertExitSweepRequest,
)


class _FakeFuture:
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)

    def cancel(self) -> bool:
        return True


class _FakeExecutor:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.submits: list[tuple[object, ...]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def submit(self, fn, *args, **kwargs):
        self.submits.append((args, kwargs))
        return _FakeFuture(fn, *args, **kwargs)


class _FakeResult:
    def __init__(self) -> None:
        index = pd.date_range("2024-01-01", periods=500, freq="4h", tz="UTC")
        self.equity = pd.Series(np.linspace(10_000.0, 11_000.0, len(index)), index=index)
        self.trades = pd.DataFrame({"pnl": [1.0, 2.0]})


class _FakeMetrics:
    def __init__(self, cagr: float, trade_count: int) -> None:
        self.cagr = cagr
        self.trade_count = trade_count


class _FakeGate:
    def __init__(self, lcb90_cagr: float, verdict: str) -> None:
        self.lcb90_cagr = lcb90_cagr
        self.verdict = verdict


def _fake_frame(symbol: str) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2024-01-01", periods=500, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.full(len(index), 100.0),
            "high": np.full(len(index), 101.0),
            "low": np.full(len(index), 99.0),
            "close": np.full(len(index), 100.5),
        },
        index=index,
    )
    funding = pd.Series(np.full(len(index), 1e-4), index=index)
    return frame, funding


def test_exit_sweep_worker_loads_each_pair_once(monkeypatch) -> None:
    # TES-01-SINGLE-LOAD-PER-PAIR: one (symbol, timeframe) load is shared by
    # every (candidate, setting) cell; the worker never reloads per cell.
    load_calls: list[tuple[str, str]] = []

    def _counting_load(symbol, start, end, *, timeframe="4h"):
        load_calls.append((symbol, timeframe))
        return _fake_frame(symbol)

    monkeypatch.setattr(app, "_load_technical_market_data", _counting_load)
    monkeypatch.setattr(app, "resolve_technical_candidate", lambda source, timeframe="4h": object())
    monkeypatch.setattr(app, "run_technical_expert_backtest", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(app, "compute_metrics", lambda equity, trades: _FakeMetrics(0.05, 10))
    monkeypatch.setattr(
        app, "compute_equity_reliability_gate",
        lambda equity, count: _FakeGate(0.02, "PASS"),
    )

    settings = (
        ExitSweepSetting(None, None, False),
        ExitSweepSetting("fixed_pct", 0.03, False),
        ExitSweepSetting("fixed_pct", 0.03, True),
        ExitSweepSetting("atr_multiple", 1.5, False),
        ExitSweepSetting("atr_multiple", 1.5, True),
    )
    cells = app._sweep_symbol_timeframe_worker(
        "BTCUSDT",
        "4h",
        (
            "technical_ema_alignment_long_v1",
            "technical_rsi_trend_pullback_long_v1",
            "technical_ichimoku_cloud_long_v1",
        ),
        settings,
        atr_period=14,
        start=None,
        end=None,
        costs=CostModel(),
    )
    assert load_calls == [("BTCUSDT", "4h")]
    assert len(cells) == 15
    assert all(isinstance(cell, ExitSweepCellResult) for cell in cells)


def test_exit_sweep_settings_grid_order_and_count() -> None:
    # TES-02-SETTINGS-GRID-ORDER-AND-COUNT: baseline first, then fixed_pct and
    # atr_multiple crossed with static/trailing in the declared order.
    request = TechnicalExpertExitSweepRequest(
        candidate_sources=("technical_ema_alignment_long_v1",),
        symbols=("BTCUSDT",),
        timeframes=("4h",),
        fixed_pct_values=(0.03, 0.05),
        atr_multiple_values=(1.5,),
        include_baseline=True,
    )
    settings = request.settings()
    assert len(settings) == 7
    assert settings[0] == ExitSweepSetting(None, None, False)
    assert settings[1] == ExitSweepSetting("fixed_pct", 0.03, False)
    assert settings[2] == ExitSweepSetting("fixed_pct", 0.03, True)
    assert settings[3] == ExitSweepSetting("fixed_pct", 0.05, False)
    assert settings[4] == ExitSweepSetting("fixed_pct", 0.05, True)
    assert settings[5] == ExitSweepSetting("atr_multiple", 1.5, False)
    assert settings[6] == ExitSweepSetting("atr_multiple", 1.5, True)
    assert settings[0].label() == "baseline_no_stop"
    assert settings[4].label() == "fixed_pct_0.05_trailing"

    no_baseline = replace(request, include_baseline=False)
    assert len(no_baseline.settings()) == 6
    assert no_baseline.settings()[0] == ExitSweepSetting("fixed_pct", 0.03, False)


def test_exit_sweep_family_summary_aggregation() -> None:
    # TES-03-FAMILY-SUMMARY-AGGREGATION: one summary per (candidate, timeframe,
    # setting) with symbol_count / mean_cagr / median_lcb90_cagr /
    # gate_pass_count computed across the swept symbols.
    baseline = ExitSweepSetting(None, None, False)
    cells = (
        ExitSweepCellResult("cand_a", "S1", "4h", baseline, 0.01, 0.005, True, 10),
        ExitSweepCellResult("cand_a", "S2", "4h", baseline, 0.03, 0.010, False, 20),
        ExitSweepCellResult("cand_a", "S3", "4h", baseline, 0.05, 0.015, True, 30),
        ExitSweepCellResult("cand_b", "S1", "4h", baseline, 0.10, 0.09, True, 5),
        ExitSweepCellResult("cand_b", "S2", "4h", baseline, 0.20, 0.11, True, 6),
        ExitSweepCellResult("cand_b", "S3", "4h", baseline, 0.30, 0.13, False, 7),
    )
    summary = app._aggregate_family_summary(cells, (baseline,))
    assert len(summary) == 2
    assert all(isinstance(entry, ExitSweepFamilySummary) for entry in summary)
    by_candidate = {entry.candidate: entry for entry in summary}
    cand_a = by_candidate["cand_a"]
    assert cand_a.symbol_count == 3
    assert cand_a.mean_cagr == pytest.approx(0.03)
    assert cand_a.median_lcb90_cagr == pytest.approx(0.010)
    assert cand_a.gate_pass_count == 2
    cand_b = by_candidate["cand_b"]
    assert cand_b.symbol_count == 3
    assert cand_b.mean_cagr == pytest.approx(0.20)
    assert cand_b.median_lcb90_cagr == pytest.approx(0.11)
    assert cand_b.gate_pass_count == 2


def test_exit_sweep_request_validation_fail_closed() -> None:
    # TES-04-REQUEST-VALIDATION-FAIL-CLOSED: structural and holdout guards fail
    # closed before any worker is dispatched.
    base = {
        "candidate_sources": ("technical_ema_alignment_long_v1",),
        "symbols": ("BTCUSDT",),
        "timeframes": ("4h",),
    }
    with pytest.raises(ValueError, match="candidate_sources"):
        TechnicalExpertExitSweepRequest(candidate_sources=(), symbols=("BTCUSDT",), timeframes=("4h",))
    with pytest.raises(ValueError, match="symbols"):
        TechnicalExpertExitSweepRequest(
            candidate_sources=("technical_ema_alignment_long_v1",), symbols=(), timeframes=("4h",),
        )
    with pytest.raises(ValueError, match="timeframes"):
        TechnicalExpertExitSweepRequest(
            candidate_sources=("technical_ema_alignment_long_v1",), symbols=("BTCUSDT",), timeframes=(),
        )
    with pytest.raises(ValueError, match="atr_period"):
        TechnicalExpertExitSweepRequest(**base, atr_period=0)
    with pytest.raises(ValueError, match="fixed_pct"):
        TechnicalExpertExitSweepRequest(**base, fixed_pct_values=(1.0,))
    with pytest.raises(ValueError, match="atr_multiple"):
        TechnicalExpertExitSweepRequest(**base, atr_multiple_values=(0.0,))
    with pytest.raises(ValueError, match="max_workers"):
        TechnicalExpertExitSweepRequest(**base, max_workers=0)
    with pytest.raises(RuntimeError, match="Holdout sealed"):
        TechnicalExpertExitSweepRequest(**base, end="2026-01-01")


def test_exit_sweep_report_to_report_dict_flattens_setting() -> None:
    baseline = ExitSweepSetting(None, None, False)
    trailing = ExitSweepSetting("atr_multiple", 2.5, True)
    report = TechnicalExpertExitSweepReport(
        cells=(
            ExitSweepCellResult("cand_a", "BTCUSDT", "4h", baseline, 0.01, 0.005, True, 10),
            ExitSweepCellResult("cand_a", "BTCUSDT", "4h", trailing, 0.02, 0.008, False, 12),
        ),
        family_summary=(
            ExitSweepFamilySummary("cand_a", "4h", baseline, 1, 0.01, 0.005, 1),
        ),
        execution_workers=1,
        wall_seconds=1.5,
    )
    payload = report.to_report_dict()
    assert set(payload) == {"cells", "family_summary", "execution_workers", "wall_seconds"}
    assert payload["execution_workers"] == 1
    assert payload["wall_seconds"] == 1.5
    cell = payload["cells"][1]
    assert cell["stop_loss_mode"] == "atr_multiple"
    assert cell["stop_loss_value"] == 2.5
    assert cell["trailing_stop"] is True
    assert cell["setting_label"] == "atr_multiple_2.5_trailing"
    assert cell["cagr"] == 0.02
    baseline_cell = payload["cells"][0]
    assert baseline_cell["setting_label"] == "baseline_no_stop"
    summary = payload["family_summary"][0]
    assert summary["symbol_count"] == 1
    assert summary["mean_cagr"] == 0.01
    assert summary["median_lcb90_cagr"] == 0.005
    assert summary["gate_pass_count"] == 1


def _tagged_worker(
    symbol: str,
    timeframe: str,
    candidate_sources: tuple[str, ...],
    settings: tuple[ExitSweepSetting, ...],
    atr_period: int,
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
) -> list[ExitSweepCellResult]:
    return [
        ExitSweepCellResult(
            candidate=candidate_sources[0],
            symbol=symbol,
            timeframe=timeframe,
            setting=settings[0],
            cagr=0.01,
            lcb90_cagr=0.005,
            gate_pass=True,
            trade_count=10,
        )
    ]


def test_exit_sweep_parallel_dispatch_covers_all_pairs(monkeypatch) -> None:
    # TES-05-PARALLEL-DISPATCH-COVERS-ALL-PAIRS: one task per distinct
    # (symbol, timeframe) pair, every pair represented exactly once.
    monkeypatch.setattr(app, "_sweep_symbol_timeframe_worker", _tagged_worker)
    fake_executor = _FakeExecutor(max_workers=2)
    monkeypatch.setattr(app, "ProcessPoolExecutor", lambda **kwargs: fake_executor)

    request = TechnicalExpertExitSweepRequest(
        candidate_sources=("technical_ema_alignment_long_v1",),
        symbols=("BTCUSDT", "ETHUSDT"),
        timeframes=("4h", "1d"),
        max_workers=2,
    )
    report = app.run_technical_expert_exit_sweep(request)

    assert fake_executor.max_workers == 2
    assert len(fake_executor.submits) == 4
    assert report.execution_workers == 2
    assert isinstance(report, TechnicalExpertExitSweepReport)
    assert len(report.cells) == 4
    pairs = {(cell.symbol, cell.timeframe) for cell in report.cells}
    assert pairs == {
        ("BTCUSDT", "4h"), ("ETHUSDT", "4h"), ("BTCUSDT", "1d"), ("ETHUSDT", "1d"),
    }
    assert len(pairs) == 4
