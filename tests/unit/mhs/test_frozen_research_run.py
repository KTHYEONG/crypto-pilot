"""Invariant scenarios for the frozen-MHS historical runner."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.frozen_research_candidate import (
    FROZEN_MHS_TOP20_V1,
    FrozenMhsCandidate,
)
from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod
from src.mhs.types import ExecutionSpec

import src.mhs.frozen_research_run as run_mod
from src.mhs.frozen_research_run import FrozenMhsBacktestRequest, run_frozen_mhs_backtest

_SYMBOLS = ("AAA", "BBB", "CCC", "DELISTED")


def _specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0)
    stress = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0)
    return base, stress


def _request(**overrides: object) -> FrozenMhsBacktestRequest:
    base, stress = _specs()
    params: dict[str, object] = {
        "source_start": pd.Timestamp("2021-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2021-04-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2021-04-10", tz="UTC"),
        "strategy": FROZEN_MHS_TOP20_V1,
        "initial_equity": 100000.0,
        "base_spec": base,
        "stress_spec": stress,
        "report_periods": (
            FrozenMhsReportPeriod(
                label="P1",
                start=pd.Timestamp("2021-04-01", tz="UTC"),
                end=pd.Timestamp("2021-04-05", tz="UTC"),
            ),
        ),
    }
    params.update(overrides)
    return FrozenMhsBacktestRequest(**params)  # type: ignore[arg-type]


def _daily(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    close = pd.DataFrame(100.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    qv = pd.DataFrame(5_000_000.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    qv["DELISTED"] = 0.0
    return close, qv


def _candidate(n_days: int = 10) -> FrozenMhsCandidate:
    labels = pd.DatetimeIndex(
        [pd.Timestamp("2021-04-01", tz="UTC") + pd.Timedelta(days=i) for i in range(n_days)], tz="UTC"
    )
    weights = pd.DataFrame(0.0, index=labels, columns=list(_SYMBOLS), dtype="float64")
    weights["AAA"] = 0.05
    weights["BBB"] = -0.05
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    return FrozenMhsCandidate(target_weights=weights, signal_available_at=avail, strategy=FROZEN_MHS_TOP20_V1)


def _evidence(coverage: float = 1.0) -> object:
    metrics = pd.DataFrame(
        {"base_coverage": [coverage], "stress_coverage": [coverage]},
        index=pd.Index(["P1"], name="period"),
    )
    return types.SimpleNamespace(period_metrics=metrics)


def _install_source(monkeypatch: pytest.MonkeyPatch, seen: dict) -> None:
    daily_close, daily_qv = _daily()

    def _fake_source(request: FrozenMhsBacktestRequest, budget: object, swap: object) -> tuple:
        seen["request"] = request
        idx = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
        panels = {
            key: pd.DataFrame(100.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
            for key in ("close", "quote_vol", "taker_buy_quote")
        }
        available = pd.DataFrame(
            np.broadcast_to((idx + pd.Timedelta(hours=1)).to_numpy()[:, None], (len(idx), len(_SYMBOLS))),
            index=idx, columns=list(_SYMBOLS),
        )
        return daily_close, daily_qv, panels, available, _SYMBOLS, {}, {}, "root"

    monkeypatch.setattr(run_mod, "_load_frozen_source", _fake_source)


def test_top20_request_retains_full_pit_census(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later-delisted symbol stays in the selection census while hourly planes prune to selected names."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    captured: dict = {}
    real_build = run_mod.build_frozen_mhs_candidate

    def _spy(panels: object, available: object, daily_close: object, daily_qv: object, census: object, **kwargs: object) -> FrozenMhsCandidate:
        captured["hourly_columns"] = list(panels["close"].columns)  # type: ignore[index]
        return real_build(panels, available, daily_close, daily_qv, census, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", _spy)
    monkeypatch.setattr(run_mod, "evaluate_frozen_mhs_research", lambda *a, **k: _evidence())
    run = run_frozen_mhs_backtest(_request())
    assert "DELISTED" in run.source_symbols
    assert list(run.candidate.target_weights.columns) == list(_SYMBOLS)
    assert set(captured["hourly_columns"]) <= set(_SYMBOLS)
    assert "DELISTED" not in captured["hourly_columns"]


def test_evaluation_slice_follows_complete_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Targets are built across source history, then scored rows begin exactly at evaluation start."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    order: list[str] = []
    full = _candidate(n_days=10)

    def _fake_build(*args: object, **kwargs: object) -> FrozenMhsCandidate:
        order.append("build")
        return full

    def _fake_evaluate(candidate: FrozenMhsCandidate, windows: object, **kwargs: object) -> object:
        order.append("evaluate")
        assert list(candidate.target_weights.index) == list(full.target_weights.index[3:7])
        return _evidence()

    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", _fake_build)
    monkeypatch.setattr(run_mod, "evaluate_frozen_mhs_research", _fake_evaluate)
    request = _request(
        evaluation_start=pd.Timestamp("2021-04-04", tz="UTC"),
        evaluation_end=pd.Timestamp("2021-04-08", tz="UTC"),
    )
    run = run_frozen_mhs_backtest(request)
    assert order == ["build", "evaluate"]
    assert run.candidate.target_weights.index[0] == pd.Timestamp("2021-04-04", tz="UTC")
    assert run.execution_start == pd.Timestamp("2021-04-03 23:00", tz="UTC")


def test_prior_release_reaches_first_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v1 23:00 release precedes the first entry and the stream opens at that release."""
    candidate = _candidate(n_days=3)
    captured: dict = {}

    def _fake_iter(weights: pd.DataFrame, signals: pd.DatetimeIndex, *args: object, **kwargs: object) -> object:
        captured["signals"] = signals
        captured["start"] = args[2] if len(args) > 2 else None
        captured["required"] = kwargs.get("required_symbols")
        captured["initial_required"] = captured["required"]()
        return iter([])

    monkeypatch.setattr(run_mod, "_iter_mhs_execution_windows", _fake_iter)
    budget = run_mod.resolve_mhs_memory_budget(None)
    stream = run_mod._frozen_window_stream(
        candidate, candidate.signal_available_at[0], candidate.signal_available_at[0] + pd.Timedelta(days=2),
        "root", {}, {}, _specs()[0], budget, [],
    )
    list(stream)
    assert captured["start"] == candidate.signal_available_at[0]
    assert captured["initial_required"] == frozenset()
    assert bool((captured["signals"] == candidate.signal_available_at).all())
    assert (candidate.target_weights.index[0] - candidate.signal_available_at[0]) == pd.Timedelta(hours=1)


def test_live_requirements_are_read_from_accumulator_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stream delegates carried-symbol requirements to the live replay state."""
    from src.mhs.execution import ExecutionReplayWindow

    candidate = _candidate(n_days=3)
    labels = list(candidate.target_weights.index)
    captured: dict = {}

    def _grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq="3min", tz="UTC")

    def _window(grid: pd.DatetimeIndex, rows: list[pd.Timestamp]) -> ExecutionReplayWindow:
        cols = ["AAA", "BBB"]
        frames = {name: pd.DataFrame(100.0, index=grid, columns=cols, dtype="float64") for name in ("highs", "lows", "closes", "marks")}
        params: dict[str, object] = {
            "window_start": grid[0], "window_end": grid[-1], "columns": _SYMBOLS, "symbols": ("AAA", "BBB"),
            "minute_grid": grid, "bar_funding": pd.DataFrame(0.0, index=grid, columns=cols),
            "target_weights": candidate.target_weights.loc[rows, cols].copy(),
            "signal_available_at": pd.DatetimeIndex([candidate.signal_available_at[candidate.target_weights.index.get_loc(label)] for label in rows], tz="UTC"),
            "quote_volumes": pd.DataFrame(1000.0, index=grid, columns=cols),
            "funding_known": pd.DataFrame(True, index=grid, columns=cols),
            "bar_available_at": grid + pd.Timedelta(minutes=3),
        }
        params.update(frames)
        return ExecutionReplayWindow(**params)  # type: ignore[arg-type]

    first = _window(_grid(labels[0] - pd.Timedelta(hours=1), labels[0] + pd.Timedelta(hours=2)), labels[:1])
    second = _window(_grid(labels[1] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2)), labels[1:2])

    live_accumulators: list = []

    def _live_required(accumulators: object) -> frozenset[str]:
        assert accumulators == [None]
        return frozenset({"AAA"}) if accumulators else frozenset()

    monkeypatch.setattr(run_mod, "live_required_symbols", _live_required)

    def _fake_iter(*args: object, **kwargs: object) -> object:
        captured["required"] = kwargs.get("required_symbols")
        live_accumulators.append([None])
        assert captured["required"]() == frozenset({"AAA"})
        return iter([first, second])

    monkeypatch.setattr(run_mod, "_iter_mhs_execution_windows", _fake_iter)
    budget = run_mod.resolve_mhs_memory_budget(None)
    out = list(
        run_mod._frozen_window_stream(
            candidate, labels[0] - pd.Timedelta(hours=1), labels[1] + pd.Timedelta(hours=2),
            "root", {}, {}, _specs()[0], budget, live_accumulators,
        )
    )
    assert len(out) == 2
    assert captured["required"] is not None


def test_missing_selected_3m_source_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PIT-selected target without complete 3m source returns no partial evidence."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", lambda *a, **k: _candidate(n_days=10))

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("missing 3m source")

    monkeypatch.setattr(run_mod, "_iter_mhs_execution_windows", _boom)
    with pytest.raises(DataIntegrityError, match=r"missing 3m|source|coverage|resolved"):
        run_frozen_mhs_backtest(_request())


def test_request_chronology_and_report_coverage_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid UTC order or a report period outside scoring coverage fails before metrics publish."""
    with pytest.raises(DataIntegrityError, match="source_start"):
        _request(
            source_start=pd.Timestamp("2021-05-01", tz="UTC"),
            evaluation_start=pd.Timestamp("2021-04-01", tz="UTC"),
        )
    with pytest.raises(DataIntegrityError, match="timezone-aware UTC"):
        _request(source_start=pd.Timestamp("2021-01-01"))
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", lambda *a, **k: _candidate(n_days=10))
    monkeypatch.setattr(run_mod, "evaluate_frozen_mhs_research", lambda *a, **k: _evidence(coverage=0.5))
    with pytest.raises(DataIntegrityError, match=r"not fully covered"):
        run_frozen_mhs_backtest(_request())


def test_request_validation_branches() -> None:
    """Every malformed request field fails closed with a field-specific reason."""
    base, stress = _specs()
    valid: dict[str, object] = {
        "source_start": pd.Timestamp("2021-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2021-04-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2021-04-10", tz="UTC"),
        "strategy": FROZEN_MHS_TOP20_V1,
        "initial_equity": 100000.0,
        "base_spec": base,
        "stress_spec": stress,
        "report_periods": (
            FrozenMhsReportPeriod(
                label="P1",
                start=pd.Timestamp("2021-04-01", tz="UTC"),
                end=pd.Timestamp("2021-04-05", tz="UTC"),
            ),
        ),
    }

    def _bad(field: str, value: object, match: str) -> None:
        params = dict(valid)
        params[field] = value
        with pytest.raises(DataIntegrityError, match=match):
            FrozenMhsBacktestRequest(**params)  # type: ignore[arg-type]

    _bad("source_start", pd.Timestamp("2021-01-01"), "timezone-aware UTC")
    _bad("source_start", "2021-01-01", "valid timestamp")
    _bad("evaluation_end", pd.Timestamp("2021-04-01", tz="UTC"), "source_start < evaluation_start")
    _bad("strategy", "frozen_mhs_top20_v1", "FrozenMhsStrategySpec")
    for bad_equity in (0.0, -10.0, float("nan"), True):
        _bad("initial_equity", bad_equity, "initial_equity")
    _bad("base_spec", ExecutionSpec(), "6 bps")
    _bad("stress_spec", ExecutionSpec(), "6 bps")
    _bad("base_spec", "cheap", "ExecutionSpec")
    _bad("report_periods", (), "report_periods")
    _bad("report_periods", ("P1",), "report_periods")
    _bad("data_root", "somewhere", "data_root")
    _bad("memory_budget", "unlimited", "memory_budget")


def test_load_frozen_source_reads_full_census(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The complete 1h archive census feeds daily panels before any roster selection."""
    from pathlib import Path as _Path

    from src.common.paths import FUTURES_DATA_DIR

    admitted: list[int] = []
    idx = pd.date_range("2021-01-01", periods=50, freq="h", tz="UTC")
    panel = {
        key: pd.DataFrame(100.0, index=idx, columns=["A", "B"], dtype="float64")
        for key in ("close", "quote_vol", "taker_buy_quote")
    }

    def _fake_panel(*args: object, **kwargs: object) -> dict:
        admit = kwargs.get("allocation_admission")
        assert admit is not None
        admit(1024)
        admitted.append(1024)
        return panel

    monkeypatch.setattr(run_mod, "load_base_panel", _fake_panel)
    monkeypatch.setattr(run_mod, "_load_funding_series", lambda symbols: ({"A": pd.Series(dtype="float64")}, {"B": "missing"}))
    monkeypatch.setattr(run_mod, "assert_mhs_stage_allocation", lambda **kwargs: admitted.append(0))
    request = _request()
    budget = run_mod.resolve_mhs_memory_budget(None)
    daily_close, daily_qv, panels, available, census, funding, failures, root = run_mod._load_frozen_source(
        request, budget, None
    )
    assert census == ("A", "B")
    assert root == str(FUTURES_DATA_DIR / "ohlcv")
    assert list(panels) == ["close", "quote_vol", "taker_buy_quote"]
    assert available.dtypes.map(str).eq("datetime64[ns, UTC]").all()
    assert failures == {"B": "missing"}
    assert admitted
    rooted = _request(data_root=_Path(tmp_path))
    assert run_mod._load_frozen_source(rooted, budget, None)[-1] == str(tmp_path)


def test_empty_selection_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy selecting no historical symbol returns no candidate."""
    daily_close, _ = _daily()
    dead_qv = daily_close * 0.0

    def _fake_source(request: FrozenMhsBacktestRequest, budget: object, swap: object) -> tuple:
        idx = pd.date_range("2021-01-01", periods=48, freq="h", tz="UTC")
        panels = {
            key: pd.DataFrame(100.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
            for key in ("close", "quote_vol", "taker_buy_quote")
        }
        available = pd.DataFrame(
            np.broadcast_to((idx + pd.Timedelta(hours=1)).to_numpy()[:, None], (len(idx), len(_SYMBOLS))),
            index=idx, columns=list(_SYMBOLS),
        )
        return daily_close, dead_qv, panels, available, _SYMBOLS, {}, {}, "root"

    monkeypatch.setattr(run_mod, "_load_frozen_source", _fake_source)
    with pytest.raises(DataIntegrityError, match=r"selects no historical symbol"):
        run_frozen_mhs_backtest(_request())


def test_empty_evaluation_slice_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An evaluation interval with no entry row returns no evidence."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", lambda *a, **k: _candidate(n_days=10))
    with pytest.raises(DataIntegrityError, match=r"no candidate entry row"):
        run_frozen_mhs_backtest(
            _request(
                evaluation_start=pd.Timestamp("2021-05-01", tz="UTC"),
                evaluation_end=pd.Timestamp("2021-05-10", tz="UTC"),
            )
        )
