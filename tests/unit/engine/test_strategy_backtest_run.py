"""Invariant scenarios for the strategy historical runner."""

from __future__ import annotations

import dataclasses
import types
from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.strategy.targets import (
    FLOW_MOM_TOP20,
    StrategyTargets,
)
from src.engine.backtest_evidence import StrategyReportPeriod
from src.core.source_gaps import SourceGapInterval
from src.core.types import ExecutionSpec

import src.engine.strategy_backtest as run_mod
from src.engine.strategy_backtest import (
    StrategyBacktestRequest,
    assert_strategy_execution_coverage,
    strategy_blocked_decisions,
    run_strategy_backtest,
)
from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY

_REAL_PRECHECK = run_mod.assert_strategy_execution_coverage


@pytest.fixture(autouse=True)
def _bypass_strategy_coverage_precheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic-census runner tests exercise non-coverage stages; the precheck has its own scenarios below."""
    monkeypatch.setattr(run_mod, "assert_strategy_execution_coverage", lambda *a, **k: None)

_SYMBOLS = ("AAA", "BBB", "CCC", "DELISTED")


def _specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar")
    stress = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0, decision_anchor="submit_bar")
    return base, stress


def _request(**overrides: object) -> StrategyBacktestRequest:
    base, stress = _specs()
    params: dict[str, object] = {
        "source_start": pd.Timestamp("2021-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2021-04-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2021-04-10", tz="UTC"),
        "strategy": FLOW_MOM_TOP20,
        "initial_equity": 100000.0,
        "base_spec": base,
        "stress_spec": stress,
        "report_periods": (
            StrategyReportPeriod(
                label="P1",
                start=pd.Timestamp("2021-04-01", tz="UTC"),
                end=pd.Timestamp("2021-04-05", tz="UTC"),
            ),
        ),
    }
    params.update(overrides)
    return StrategyBacktestRequest(**params)  # type: ignore[arg-type]


def _daily(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    close = pd.DataFrame(100.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    qv = pd.DataFrame(5_000_000.0, index=idx, columns=list(_SYMBOLS), dtype="float64")
    qv["DELISTED"] = 0.0
    return close, qv


def _candidate(n_days: int = 10) -> StrategyTargets:
    labels = pd.DatetimeIndex(
        [pd.Timestamp("2021-04-01", tz="UTC") + pd.Timedelta(days=i) for i in range(n_days)], tz="UTC"
    )
    weights = pd.DataFrame(0.0, index=labels, columns=list(_SYMBOLS), dtype="float64")
    weights["AAA"] = 0.05
    weights["BBB"] = -0.05
    avail = pd.DatetimeIndex([label - pd.Timedelta(hours=1) for label in labels], tz="UTC")
    return StrategyTargets(target_weights=weights, signal_available_at=avail, strategy=FLOW_MOM_TOP20)


def _evidence(coverage: float = 1.0) -> object:
    metrics = pd.DataFrame(
        {"base_coverage": [coverage], "stress_coverage": [coverage]},
        index=pd.Index(["P1"], name="period"),
    )
    return types.SimpleNamespace(period_metrics=metrics)


def _install_source(monkeypatch: pytest.MonkeyPatch, seen: dict) -> None:
    daily_close, daily_qv = _daily()

    def _fake_source(request: StrategyBacktestRequest, budget: object, swap: object) -> tuple:
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

    monkeypatch.setattr(run_mod, "_load_strategy_source", _fake_source)


def test_top20_request_retains_full_pit_census(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later-delisted symbol stays in the selection census while hourly planes prune to selected names."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    captured: dict = {}
    real_build = run_mod.build_strategy_targets

    def _spy(panels: object, available: object, daily_close: object, daily_qv: object, census: object, **kwargs: object) -> StrategyTargets:
        captured["hourly_columns"] = list(panels["close"].columns)  # type: ignore[index]
        return real_build(panels, available, daily_close, daily_qv, census, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(run_mod, "build_strategy_targets", _spy)
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence())
    run = run_strategy_backtest(_request())
    assert "DELISTED" in run.source_symbols
    assert list(run.candidate.target_weights.columns) == list(_SYMBOLS)
    assert set(captured["hourly_columns"]) <= set(_SYMBOLS)
    assert "DELISTED" not in captured["hourly_columns"]


@pytest.mark.parametrize("extent", ["OPEN_EDGE", "UNSCOPED"])
def test_first_evaluation_entry_gap_fails_before_target_build(monkeypatch: pytest.MonkeyPatch, extent: str) -> None:
    _install_source(monkeypatch, {})
    gap = _gap_interval("AAA", "SOURCE_ABSENT", extent, "2021-04-01T12:00:00Z", "2021-04-01T13:00:00Z")
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (gap,))

    def _unexpected_build(*args, **kwargs):
        raise AssertionError("lake coverage must fail before feature building or replay")

    monkeypatch.setattr(run_mod, "build_strategy_targets", _unexpected_build)
    with pytest.raises(run_mod.LakeCoverageError, match=r"AAA.*2021-03-31"):
        run_strategy_backtest(_request())


def test_evaluation_slice_follows_complete_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Targets are built across source history, then scored rows begin exactly at evaluation start."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    order: list[str] = []
    full = _candidate(n_days=10)

    def _fake_build(*args: object, **kwargs: object) -> StrategyTargets:
        order.append("build")
        return full

    def _fake_evaluate(candidate: StrategyTargets, windows: object, **kwargs: object) -> object:
        order.append("evaluate")
        assert list(candidate.target_weights.index) == list(full.target_weights.index[3:7])
        return _evidence()

    monkeypatch.setattr(run_mod, "build_strategy_targets", _fake_build)
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", _fake_evaluate)
    request = _request(
        evaluation_start=pd.Timestamp("2021-04-04", tz="UTC"),
        evaluation_end=pd.Timestamp("2021-04-08", tz="UTC"),
    )
    run = run_strategy_backtest(request)
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
    stream = run_mod._strategy_window_stream(
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
    from src.engine.execution import ExecutionReplayWindow

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
        run_mod._strategy_window_stream(
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
    monkeypatch.setattr(run_mod, "build_strategy_targets", lambda *a, **k: _candidate(n_days=10))

    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("missing 3m source")

    monkeypatch.setattr(run_mod, "_iter_mhs_execution_windows", _boom)
    with pytest.raises(DataIntegrityError, match=r"missing 3m|source|coverage|resolved"):
        run_strategy_backtest(_request())


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
    monkeypatch.setattr(run_mod, "build_strategy_targets", lambda *a, **k: _candidate(n_days=10))
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence(coverage=0.5))
    with pytest.raises(DataIntegrityError, match=r"not fully covered"):
        run_strategy_backtest(_request())


def test_request_validation_branches() -> None:
    """Every malformed request field fails closed with a field-specific reason."""
    base, stress = _specs()
    valid: dict[str, object] = {
        "source_start": pd.Timestamp("2021-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2021-04-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2021-04-10", tz="UTC"),
        "strategy": FLOW_MOM_TOP20,
        "initial_equity": 100000.0,
        "base_spec": base,
        "stress_spec": stress,
        "report_periods": (
            StrategyReportPeriod(
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
            StrategyBacktestRequest(**params)  # type: ignore[arg-type]

    _bad("source_start", pd.Timestamp("2021-01-01"), "timezone-aware UTC")
    _bad("source_start", "2021-01-01", "valid timestamp")
    _bad("evaluation_end", pd.Timestamp("2021-04-01", tz="UTC"), "source_start < evaluation_start")
    _bad("strategy", "frozen_mhs_top20_v2", "StrategySpec")
    for bad_equity in (0.0, -10.0, float("nan"), True):
        _bad("initial_equity", bad_equity, "initial_equity")
    _bad("base_spec", ExecutionSpec(), "6 bps")
    _bad("stress_spec", ExecutionSpec(), "6 bps")
    _bad("base_spec", "cheap", "ExecutionSpec")
    _bad("report_periods", (), "report_periods")
    _bad("report_periods", ("P1",), "report_periods")
    _bad("data_root", "somewhere", "data_root")
    _bad("memory_budget", "unlimited", "memory_budget")


def test_load_strategy_source_reads_full_census(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    daily_close, daily_qv, panels, available, census, funding, failures, root = run_mod._load_strategy_source(
        request, budget, None
    )
    assert census == ("A", "B")
    assert root == str(FUTURES_DATA_DIR / "ohlcv")
    assert list(panels) == ["close", "quote_vol", "taker_buy_quote"]
    assert available.dtypes.map(str).eq("datetime64[ns, UTC]").all()
    assert failures == {"B": "missing"}
    assert admitted
    rooted = _request(data_root=_Path(tmp_path))
    assert run_mod._load_strategy_source(rooted, budget, None)[-1] == str(tmp_path)


def test_empty_selection_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy selecting no historical symbol returns no candidate."""
    daily_close, _ = _daily()
    dead_qv = daily_close * 0.0

    def _fake_source(request: StrategyBacktestRequest, budget: object, swap: object) -> tuple:
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

    monkeypatch.setattr(run_mod, "_load_strategy_source", _fake_source)
    with pytest.raises(DataIntegrityError, match=r"selects no historical symbol"):
        run_strategy_backtest(_request())


def test_empty_evaluation_slice_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An evaluation interval with no entry row returns no evidence."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_strategy_targets", lambda *a, **k: _candidate(n_days=10))
    with pytest.raises(DataIntegrityError, match=r"no candidate entry row"):
        run_strategy_backtest(
            _request(
                evaluation_start=pd.Timestamp("2021-05-01", tz="UTC"),
                evaluation_end=pd.Timestamp("2021-05-10", tz="UTC"),
            )
        )


_GAP_SYMBOLS = ("AAA", "PUMPUSDT", "LUNAUSDT", "BBB")


def _gap_source(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    idx_days = pd.date_range("2021-01-01", periods=100, freq="D", tz="UTC")
    daily_close = pd.DataFrame(100.0, index=idx_days, columns=list(_GAP_SYMBOLS), dtype="float64")
    daily_qv = pd.DataFrame(5_000_000.0, index=idx_days, columns=list(_GAP_SYMBOLS), dtype="float64")
    real_roster = run_mod.build_pit_roster
    real_build = run_mod.build_strategy_targets

    def _fake_source(request: StrategyBacktestRequest, budget: object, swap: object) -> tuple:
        idx = pd.date_range("2021-01-01", periods=100 * 24, freq="h", tz="UTC")
        panels = {
            key: pd.DataFrame(100.0, index=idx, columns=list(_GAP_SYMBOLS), dtype="float64")
            for key in ("close", "quote_vol", "taker_buy_quote")
        }
        available = pd.DataFrame(
            np.broadcast_to((idx + pd.Timedelta(hours=1)).to_numpy()[:, None], (len(idx), len(_GAP_SYMBOLS))),
            index=idx, columns=list(_GAP_SYMBOLS),
        )
        return daily_close, daily_qv, panels, available, _GAP_SYMBOLS, {}, {}, "root"

    def _spy_roster(daily_close: object, daily_qv: object, census: object, **kwargs: object) -> object:
        captured.setdefault("roster_blocked_calls", []).append(kwargs.get("blocked_decisions"))
        captured["roster_blocked"] = kwargs.get("blocked_decisions")
        return real_roster(daily_close, daily_qv, census, **kwargs)  # type: ignore[arg-type]

    def _spy_build(
        panels: object, available: object, daily_close: object, daily_qv: object, census: object, **kwargs: object
    ) -> StrategyTargets:
        captured["candidate_blocked"] = kwargs.get("blocked_decisions")
        captured["hourly_columns"] = list(panels["close"].columns)  # type: ignore[index]
        return real_build(panels, available, daily_close, daily_qv, census, **kwargs)  # type: ignore[arg-type]

    # 레지스트리 파일 내용에 결합되면 정책 갱신마다 이 시나리오가 깨진다 — 구간을 주입해 밀봉한다.
    injected = (
        SourceGapInterval(
            symbol="PUMPUSDT", plane="ohlcv_3m",
            start=pd.Timestamp("2020-01-01", tz="UTC").to_pydatetime(), end=None,
            reason="SOURCE_ABSENT", evidence="probe fixture: blocks the whole scenario window",
            verified_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(), resolved_at=None,
            extent="INTERIOR",
        ),
        SourceGapInterval(
            symbol="LUNAUSDT", plane="ohlcv_3m",
            start=pd.Timestamp("2023-01-01", tz="UTC").to_pydatetime(),
            end=pd.Timestamp("2023-02-01", tz="UTC").to_pydatetime(),
            reason="SOURCE_ABSENT", evidence="probe fixture: bounded interval outside the window",
            verified_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(), resolved_at=None,
            extent="INTERIOR",
        ),
    )
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: injected)
    monkeypatch.setattr(run_mod, "_load_strategy_source", _fake_source)
    monkeypatch.setattr(run_mod, "build_pit_roster", _spy_roster)
    monkeypatch.setattr(run_mod, "build_strategy_targets", _spy_build)
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence())


def test_runner_threads_reviewed_exclusion_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _gap_source(monkeypatch, captured)
    run = run_strategy_backtest(_request())
    assert run.source_gap_excluded_symbols == ("PUMPUSDT",)
    assert run.source_gap_blocked_decisions == 100
    roster_blocked = captured["roster_blocked_calls"][0]
    candidate_blocked = captured["candidate_blocked"]
    pd.testing.assert_frame_equal(roster_blocked, candidate_blocked)
    assert bool(roster_blocked["PUMPUSDT"].all())
    assert not bool(roster_blocked["LUNAUSDT"].any())
    assert not bool(roster_blocked["AAA"].any())
    assert not bool(roster_blocked["BBB"].any())
    assert run.source_gap_blocked_decisions == int(roster_blocked.to_numpy().sum())
    assert list(run.candidate.target_weights.columns) == list(_GAP_SYMBOLS)
    assert bool((run.candidate.target_weights["PUMPUSDT"].to_numpy() == 0.0).all())
    assert "PUMPUSDT" in run.source_symbols


def test_partial_withdrawal_is_disclosed_without_full_symbol_exclusion(monkeypatch) -> None:
    from src.core.source_gaps import SourceGapInterval

    _gap_source(monkeypatch, {})
    gap = SourceGapInterval(
        symbol="PUMPUSDT", plane="ohlcv_3m",
        start=pd.Timestamp("2021-04-05", tz="UTC").to_pydatetime(),
        end=pd.Timestamp("2021-04-06", tz="UTC").to_pydatetime(),
        reason="SOURCE_ABSENT", evidence="bounded withdrawal",
        verified_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
        resolved_at=None, extent="INTERIOR",
    )
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (gap,))
    run = run_strategy_backtest(_request())
    assert run.source_gap_excluded_symbols == ()
    assert run.roster_seat_days > 0
    assert sum(int(entry["days"]) for entry in run.data_availability_withdrawals) > 0
    assert "LUNAUSDT" in run.source_symbols


def test_excluded_symbol_absent_from_hourly_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _gap_source(monkeypatch, captured)
    run_strategy_backtest(_request())
    assert "PUMPUSDT" not in captured["hourly_columns"]
    assert "LUNAUSDT" in captured["hourly_columns"]
    assert "AAA" in captured["hourly_columns"]


def test_blocked_decisions_uses_half_open_window(monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = _specs()
    interval = _gap_interval(
        "MANAUSDT", "SOURCE_ABSENT", "INTERIOR", "2022-02-26T00:00:00Z", "2022-03-01T00:00:00Z",
    )
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    days = pd.DatetimeIndex([pd.Timestamp("2022-02-28", tz="UTC")], tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("MANAUSDT",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert not bool(frame.iloc[0, 0])


def test_blocked_decisions_blocks_holding_window_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = _specs()
    interval = _gap_interval(
        "MANAUSDT", "SOURCE_ABSENT", "INTERIOR", "2022-02-26T00:00:00Z", "2022-03-01T00:00:00Z",
    )
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    days = pd.DatetimeIndex([pd.Timestamp("2022-02-27", tz="UTC")], tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("MANAUSDT", "AAA"), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert bool(frame.loc[days[0], "MANAUSDT"])
    assert not bool(frame.loc[days[0], "AAA"])


def test_blocked_decisions_empty_census_returns_empty_frame() -> None:
    base, _ = _specs()
    days = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, (), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert frame.shape == (3, 0)


def test_blocked_decisions_empty_registry_blocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: ())
    base, _ = _specs()
    days = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert not bool(frame.to_numpy().any())


def test_held_source_stays_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_strategy_targets", lambda *a, **k: _candidate(n_days=10))
    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("missing 3m source for held AAA")

    monkeypatch.setattr(run_mod, "_iter_mhs_execution_windows", _boom)
    with pytest.raises(DataIntegrityError, match=r"missing 3m source"):
        run_strategy_backtest(_request())


_SETTLEMENT = pd.Timedelta(minutes=30)


def _coverage_roster(
    days: pd.DatetimeIndex, grants: dict[str, list[pd.Timestamp]],
) -> pd.DataFrame:
    frame = pd.DataFrame(False, index=days, columns=sorted(grants), dtype=bool)
    for symbol, stamps in grants.items():
        for stamp in stamps:
            frame.loc[stamp, symbol] = True
    return frame


def _write_3m(root, symbol: str, idx: pd.DatetimeIndex):
    from pathlib import Path as _Path

    directory = _Path(root) / "ohlcv" / "3m"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "timestamp": [int(t.value // 10**6) for t in idx],
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    path = directory / f"{symbol}.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_assert_strategy_execution_coverage_blocks_deficient_symbol(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")], "BBB": []})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-09", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
        )
    message = str(exc_info.value)
    assert "AAA" in message
    assert "2022-01-10T00:30:00+00:00" in message
    assert "2022-01-09T00:00:00+00:00" in message
    assert "BBB" not in message


def test_assert_strategy_execution_coverage_honours_entry_hour(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    # entry_hour=0 기준으로는 충족되지만, 6시 진입 변형에서는 6시간이 더 필요하다.
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10 00:30", freq="3min", tz="UTC"))
    assert_strategy_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
    )
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=6, data_root=tmp_path / "ohlcv",
        )
    assert "2022-01-10T06:30:00+00:00" in str(exc_info.value)


def test_assert_strategy_execution_coverage_separates_unreadable_from_missing(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    corrupt = tmp_path / "ohlcv" / "3m" / "AAA.parquet"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
        )
    message = str(exc_info.value)
    assert "UNREADABLE" in message
    assert "MISSING" not in message


def test_assert_strategy_execution_coverage_ignores_unselected_symbols(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-05", tz="UTC")], "GHOST": []})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-02-01", freq="3min", tz="UTC"))
    assert_strategy_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
    )


def test_assert_strategy_execution_coverage_passes_delisted_symbol(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10T00:30", freq="3min", tz="UTC"))
    assert_strategy_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-06-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
    )


def test_assert_strategy_execution_coverage_includes_settlement_slack(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError, match="AAA"):
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
        )


def test_assert_strategy_execution_coverage_tolerates_one_bar(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10T00:27", freq="3min", tz="UTC"))
    assert_strategy_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
    )


def test_assert_strategy_execution_coverage_reports_all_deficient_symbols(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    grant = [pd.Timestamp("2022-01-08", tz="UTC")]
    roster = _coverage_roster(days, {"AAA": grant, "BBB": grant, "CCC": grant})
    for symbol in ("AAA", "BBB", "CCC"):
        _write_3m(tmp_path, symbol, pd.date_range("2022-01-01", "2022-01-05", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
        )
    message = str(exc_info.value)
    assert "3 symbol(s)" in message
    assert all(symbol in message for symbol in ("AAA", "BBB", "CCC"))


def test_assert_strategy_execution_coverage_caps_requirement_at_fence(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-09", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10", freq="3min", tz="UTC"))
    assert_strategy_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-01-10", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
    )


def test_assert_strategy_execution_coverage_treats_unreadable_archives_as_missing(tmp_path) -> None:
    from pathlib import Path as _Path

    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    grant = [pd.Timestamp("2022-01-05", tz="UTC")]
    roster = _coverage_roster(days, {"AAA": grant, "BBB": grant, "CCC": grant, "DDD": grant})
    three_m = _Path(tmp_path) / "ohlcv" / "3m"
    three_m.mkdir(parents=True)
    (three_m / "BBB.parquet").write_bytes(b"not a parquet file")
    pd.DataFrame({"timestamp": pd.Series(dtype="int64")}).to_parquet(
        three_m / "CCC.parquet", index=False,
    )
    pd.DataFrame({"timestamp": [float("nan")] * 3}).to_parquet(three_m / "DDD.parquet", index=False)
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
        )
    message = str(exc_info.value)
    assert "4 symbol(s)" in message
    assert all(symbol in message for symbol in ("AAA", "BBB", "CCC", "DDD"))
    assert "MISSING" in message


def test_run_strategy_backtest_blocks_before_candidate_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """결손이 있는 요청은 후보 생성과 윈도우 스트림보다 먼저 멈춘다."""
    monkeypatch.setattr(run_mod, "assert_strategy_execution_coverage", _REAL_PRECHECK)
    seen: dict = {}
    _install_source(monkeypatch, seen)
    days = pd.date_range("2021-04-01", periods=9, freq="D", tz="UTC")
    roster = _coverage_roster(
        days, {"AAA": [pd.Timestamp("2021-04-08", tz="UTC")], "BBB": [], "CCC": [], "DELISTED": []},
    )
    monkeypatch.setattr(run_mod, "build_pit_roster", lambda *a, **k: roster)
    _write_3m(tmp_path, "AAA", pd.date_range("2021-04-01", "2021-04-05", freq="3min", tz="UTC"))
    called: list[str] = []
    monkeypatch.setattr(
        run_mod, "build_strategy_targets",
        lambda *a, **k: (called.append("build"), _candidate(n_days=10))[1],
    )
    monkeypatch.setattr(
        run_mod, "evaluate_strategy_backtest",
        lambda *a, **k: (called.append("evaluate"), _evidence())[1],
    )
    with pytest.raises(DataIntegrityError, match="AAA"):
        run_strategy_backtest(_request(data_root=tmp_path / "ohlcv"))
    assert called == []


def test_request_rejects_decision_bar_anchor() -> None:
    """Orders sized off a mark published after submission cannot be evidenced."""
    base, stress = _specs()
    legacy = dataclasses.replace(base, decision_anchor="decision_bar")
    legacy_stress = dataclasses.replace(stress, decision_anchor="decision_bar")
    with pytest.raises(DataIntegrityError, match="submit_bar"):
        _request(base_spec=legacy, stress_spec=legacy_stress)


def test_request_rejects_mismatched_stress_anchor() -> None:
    """Base and stress must share the causal submit-bar anchor."""
    base, stress = _specs()
    drifted = dataclasses.replace(stress, decision_anchor="decision_bar")
    with pytest.raises(DataIntegrityError, match="submit_bar"):
        _request(base_spec=base, stress_spec=drifted)


def test_request_rejects_unknown_execution_bound() -> None:
    """Only the registered taker and strict-proxy crossing models are evidenced."""
    with pytest.raises(DataIntegrityError, match="execution_bound"):
        _request(execution_bound="OHLCV_PEG_CHASE_PROXY")


def test_runner_forwards_execution_bound_to_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A maker request reaches paired evaluation with the selected bound."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_strategy_targets", lambda *a, **k: _candidate(n_days=10))

    def _spy(candidate: object, windows: object, **kwargs: object) -> object:
        seen["execution_bound"] = kwargs.get("execution_bound")
        return _evidence()

    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", _spy)
    run_strategy_backtest(_request(execution_bound="OHLCV_STRICT_PROXY"))
    assert seen["execution_bound"] == "OHLCV_STRICT_PROXY"


def test_candidate_builder_matches_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner reuses the standalone candidate builder without replay divergence."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence())
    request = _request()
    direct, context = run_mod.build_request_targets(request)
    run = run_strategy_backtest(request)
    pd.testing.assert_frame_equal(direct.target_weights, run.candidate.target_weights)
    assert bool((direct.signal_available_at == run.candidate.signal_available_at).all())
    assert context.census == run.source_symbols


def test_coverage_check_reads_panel_tree_under_override(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"BTCUSDT": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "BTCUSDT", pd.date_range("2022-01-01", "2022-02-01", freq="3min", tz="UTC"))
    assert_strategy_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
    )


def test_coverage_check_under_override_still_fails_closed(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"BTCUSDT": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "BTCUSDT", pd.date_range("2022-01-01", "2022-01-09", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_strategy_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path / "ohlcv",
        )
    assert "BTCUSDT" in str(exc_info.value)
    assert "MISSING" not in str(exc_info.value)
    assert "available=2022-01-09T00:00:00+00:00" in str(exc_info.value)


def _settlement_record(
    symbol: str, announced_at: str, last_trade_at: str, delivery_at: str,
    source: str = "curated",
):
    from src.core.instrument_settlements import InstrumentSettlementRecord

    delivery = pd.Timestamp(delivery_at, tz="UTC")
    return InstrumentSettlementRecord(
        symbol=symbol,
        event_id=f"{symbol}:{int(delivery.value // 1_000_000)}",
        announced_at=pd.Timestamp(announced_at, tz="UTC"),
        announcement_source=source,  # type: ignore[arg-type]
        announcement_evidence="" if source == "proxy_lead" else "Binance notice",
        last_trade_at=pd.Timestamp(last_trade_at, tz="UTC"),
        delivery_at=delivery,
        settlement_price=1.0, price_source="flat_1h_klines",
        price_evidence="flat bars", fee_bps=5.0,
        evidence_digest="sha256:" + "ab" * 32,
        verified_at=pd.Timestamp("2026-07-01T00:00:00Z"),
    )


def _settlement_registry(records: list) -> object:
    from src.core.instrument_settlements import assemble_instrument_settlement_registry

    return assemble_instrument_settlement_registry(records, [])


def _gap_interval(symbol: str, reason: str, extent: str, start: str, end: str | None = None):

    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime().astimezone(UTC)
    end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime().astimezone(UTC) if end else None
    verified = pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime().astimezone(UTC)
    return SourceGapInterval(
        symbol=symbol, plane="ohlcv_3m", start=start_dt, end=end_dt, reason=reason,
        evidence="test fixture", verified_at=verified, resolved_at=None,
        extent=extent,  # type: ignore[arg-type]
    )


def test_no_block_before_announcement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A data-end OPEN_EDGE explained by settlement never anticipates the delisting."""
    from src.core.params import DELIST_ROSTER_BLOCK_LEAD

    base, _ = _specs()
    record = _settlement_record(
        "AAA", "2022-01-09T12:00:00Z", "2022-01-10T00:00:00Z", "2022-01-10T00:00:00Z",
    )
    registry = _settlement_registry([record])
    interval = _gap_interval("AAA", "SOURCE_ABSENT", "OPEN_EDGE", "2022-01-10T00:00:00Z")
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    days = pd.date_range("2022-01-05", periods=8, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=registry,
    )
    before = pd.Timestamp("2022-01-07", tz="UTC")
    assert not bool(frame.loc[before, "AAA"])
    announced_day = pd.Timestamp("2022-01-09", tz="UTC")
    info = announced_day + pd.Timedelta(hours=int(FLOW_MOM_TOP20.snapshot_hour_utc))
    horizon = announced_day + pd.Timedelta(days=1, hours=int(FLOW_MOM_TOP20.snapshot_hour_utc))
    horizon += DELIST_ROSTER_BLOCK_LEAD
    expect = (info >= record.announced_at) and (horizon >= record.delivery_at)
    assert bool(frame.loc[announced_day, "AAA"]) is expect


def test_announced_delisting_withdrawn_within_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = _specs()
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: ())
    record = _settlement_record(
        "AAA", "2022-01-01T00:00:00Z", "2022-01-08T00:00:00Z", "2022-01-08T00:00:00Z",
    )
    registry = _settlement_registry([record])
    days = pd.date_range("2021-12-28", periods=12, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=registry,
    )
    snapshot = int(FLOW_MOM_TOP20.snapshot_hour_utc)
    for day in days:
        horizon = day + pd.Timedelta(days=1, hours=snapshot) + pd.Timedelta(hours=48)
        announced = (day + pd.Timedelta(hours=snapshot)) >= record.announced_at
        assert bool(frame.loc[day, "AAA"]) is bool(announced and horizon >= record.delivery_at)


def test_superseded_open_edge_does_not_block(monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = _specs()
    record = _settlement_record(
        "AAA", "2021-12-25T00:00:00Z", "2022-01-01T00:00:00Z", "2022-01-01T00:00:00Z",
    )
    registry = _settlement_registry([record])
    interval = _gap_interval("AAA", "SOURCE_ABSENT", "OPEN_EDGE", "2022-01-01T00:00:00Z")
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    days = pd.date_range("2021-12-28", periods=6, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=registry,
    )
    horizon_hits = [
        bool(
            (day + pd.Timedelta(hours=int(FLOW_MOM_TOP20.snapshot_hour_utc)) >= record.announced_at)
            and (
                day + pd.Timedelta(days=1, hours=int(FLOW_MOM_TOP20.snapshot_hour_utc))
                + pd.Timedelta(hours=48) >= record.delivery_at
            )
        )
        for day in days
    ]
    assert frame["AAA"].tolist() == horizon_hits


def test_interior_gap_still_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = _specs()
    registry = EMPTY_SETTLEMENT_REGISTRY
    interval = _gap_interval(
        "AAA", "SOURCE_ABSENT", "INTERIOR", "2022-01-03T00:00:00Z", "2022-01-04T00:00:00Z",
    )
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    days = pd.DatetimeIndex([pd.Timestamp("2022-01-02", tz="UTC")], tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=registry,
    )
    assert bool(frame.iloc[0, 0])


def test_announcement_perturbation_is_pit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moving announced_at after T leaves pre-T blocked cells identical (I5)."""
    base, _ = _specs()
    interval = _gap_interval("AAA", "SOURCE_ABSENT", "OPEN_EDGE", "2022-01-12T00:00:00Z")
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    before = _settlement_record(
        "AAA", "2022-01-05T00:00:00Z", "2022-01-12T00:00:00Z", "2022-01-12T00:00:00Z",
    )
    after = _settlement_record(
        "AAA", "2022-01-10T00:00:00Z", "2022-01-12T00:00:00Z", "2022-01-12T00:00:00Z",
    )
    days = pd.date_range("2022-01-01", periods=12, freq="D", tz="UTC")
    first = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=_settlement_registry([before]),
    )
    second = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=_settlement_registry([after]),
    )
    cutoff = pd.Timestamp("2022-01-04T22:00:00Z")
    snapshot = int(FLOW_MOM_TOP20.snapshot_hour_utc)
    pre = [day for day in days if day + pd.Timedelta(hours=snapshot) <= cutoff]
    assert pre
    assert first.loc[pre, "AAA"].tolist() == second.loc[pre, "AAA"].tolist()
    assert not bool(second.loc[pre, "AAA"].any())
    assert bool(first.loc[pd.Timestamp("2022-01-09", tz="UTC"), "AAA"])
    assert not bool(second.loc[pd.Timestamp("2022-01-09", tz="UTC"), "AAA"])


def test_parity_with_live_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cause-1 cells match the live roster withdrawal for a midnight announcement."""
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        delisting_blocked_decisions as live_blocked,
    )

    base, _ = _specs()
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: ())
    announced = pd.Timestamp("2022-01-05T00:00:00Z")
    delivery = pd.Timestamp("2022-01-12T00:00:00Z")
    record = _settlement_record(
        "AAA", announced.isoformat(), "2022-01-12T00:00:00Z", delivery.isoformat(),
    )
    registry = _settlement_registry([record])
    days = pd.date_range("2022-01-01", periods=14, freq="D", tz="UTC")
    research = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=registry,
    )
    entry = VenueListingEntry(
        symbol="AAA", status="TRADING", contract_type="PERPETUAL",
        underlying_type="COIN", quote_asset="USDT", delivery_time=delivery,
        announced_delisting=True, delisting_first_seen_at=announced,
    )
    history = (
        VenueListingSnapshot(
            captured_at=announced, entries={"AAA": entry},
            slot_day=announced.normalize(),
        ),
    )
    live = live_blocked(
        history, days, ("AAA",),
        holding_end_offset=pd.Timedelta(
            days=1, hours=int(FLOW_MOM_TOP20.snapshot_hour_utc),
        ),
        lead=pd.Timedelta(hours=48),
    )
    assert research["AAA"].tolist() == live["AAA"].tolist()


def test_report_carries_delisting_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.engine.backtest_persist import strategy_backtest_payload

    import tests.unit.engine.test_strategy_backtest_report as report_helper

    run = report_helper._run()
    tagged = dataclasses.replace(run, delisting_blocked_decisions=7)
    payload = strategy_backtest_payload(tagged)
    assert payload["delisting_blocked_decisions"] == 7


def test_foreign_settlement_symbol_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A settlement record outside the census never touches the frame."""
    base, _ = _specs()
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: ())
    record = _settlement_record(
        "ZZZ", "2022-01-01T00:00:00Z", "2022-01-08T00:00:00Z", "2022-01-08T00:00:00Z",
    )
    registry = _settlement_registry([record])
    days = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
    frame = strategy_blocked_decisions(
        days, ("AAA",), strategy=FLOW_MOM_TOP20, base_spec=base,
        settlement_registry=registry,
    )
    assert not bool(frame.to_numpy().any())


@pytest.mark.parametrize(
    ("announcement", "expected"),
    [("2022-01-09T22:00:00Z", True), ("2022-01-09T22:00:00.000000001Z", False)],
)
def test_snapshot_announcement_boundary(monkeypatch, announcement, expected) -> None:
    """An announcement after the snapshot cannot withdraw that day's roster."""
    record = _settlement_record(
        "AAA", announcement, "2022-01-10T00:00:00Z", "2022-01-10T00:00:00Z",
    )
    interval = _gap_interval("AAA", "SOURCE_ABSENT", "OPEN_EDGE", "2022-01-10T00:00:00Z")
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    frame = strategy_blocked_decisions(
        pd.date_range("2022-01-09", periods=1, tz="UTC"), ("AAA",),
        strategy=FLOW_MOM_TOP20, base_spec=_specs()[0],
        settlement_registry=_settlement_registry([record]),
    )
    assert bool(frame.iloc[0, 0]) is expected


def test_runner_counts_delisting_cause_once(monkeypatch) -> None:
    """Overlapping causes retain the delisting count and bind to the loaded root."""
    _install_source(monkeypatch, {})
    record = _settlement_record(
        "AAA", "2021-04-01T00:00:00Z", "2021-04-05T00:00:00Z", "2021-04-05T00:00:00Z",
    )
    registry = _settlement_registry([record])

    def registry_for_root(root):
        assert root == "root"
        return registry

    monkeypatch.setattr(run_mod, "settlement_registry_for_root", registry_for_root)
    interval = _gap_interval("AAA", "SOURCE_ABSENT", "INTERIOR", "2021-04-03", "2021-04-04")
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: (interval,))
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence())
    result = run_strategy_backtest(_request())
    assert result.delisting_blocked_decisions == 9
    assert result.source_gap_blocked_decisions == 10
    assert result.settlement_registry is registry


def _write_audit_3m(ohlcv_root, symbol: str, last_bar: pd.Timestamp) -> None:
    from pathlib import Path as _Path

    directory = _Path(ohlcv_root) / "3m"
    directory.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(pd.Timestamp("2021-04-01", tz="UTC"), last_bar, freq="3min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": [int(t.value // 10**6) for t in idx],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "quote_vol": 1000.0,
    })
    frame.to_parquet(directory / f"{symbol}.parquet", index=False)


def _install_audit_panel(monkeypatch: pytest.MonkeyPatch, census: tuple[str, ...]) -> None:
    idx = pd.date_range("2021-04-01", periods=48, freq="h", tz="UTC")
    panel = {
        key: pd.DataFrame(100.0, index=idx, columns=list(census), dtype="float64")
        for key in ("close", "quote_vol", "taker_buy_quote")
    }

    def _fake_panel(*args: object, **kwargs: object) -> dict:
        admit = kwargs.get("allocation_admission")
        if callable(admit):
            admit(1024)
        return panel

    monkeypatch.setattr(run_mod, "load_base_panel", _fake_panel)
    monkeypatch.setattr(run_mod, "_load_funding_series", lambda symbols: ({}, {}))


def test_audit_horizon_equals_evaluation_end(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The settlement audit is fenced at evaluation end, matching replay coverage."""
    census = ("AAA",)
    _install_audit_panel(monkeypatch, census)
    captured: dict = {}
    monkeypatch.setattr(
        run_mod, "assert_settlement_registry_complete",
        lambda *a, **k: captured.update(k),
    )
    request = _request(data_root=tmp_path / "ohlcv")
    run_mod._load_strategy_source(request, run_mod.resolve_mhs_memory_budget(None), None)
    assert captured["audit_end"] == request.evaluation_end


def test_lake_ending_exactly_at_evaluation_end_passes_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """A live symbol whose last 3m bar closes the horizon needs no registry record."""
    request = _request(data_root=tmp_path / "ohlcv")
    _install_audit_panel(monkeypatch, ("AAA",))
    _write_audit_3m(tmp_path / "ohlcv", "AAA", request.evaluation_end - pd.Timedelta(minutes=3))
    run_mod._load_strategy_source(request, run_mod.resolve_mhs_memory_budget(None), None)


def test_data_ending_before_evaluation_end_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """A live symbol ending inside the horizon without a record still fails closed."""
    request = _request(data_root=tmp_path / "ohlcv")
    _install_audit_panel(monkeypatch, ("AAA",))
    _write_audit_3m(tmp_path / "ohlcv", "AAA", request.evaluation_end - pd.Timedelta(days=1))
    with pytest.raises(DataIntegrityError, match=r"settlement registry incomplete.*AAA"):
        run_mod._load_strategy_source(request, run_mod.resolve_mhs_memory_budget(None), None)


def test_shared_source_gives_identical_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path as _Path
    from tests.unit.engine.test_strategy_backtest_windows import _window

    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_strategy_targets", lambda *a, **k: _candidate(n_days=2))

    def stream(candidate, start, end, *args):
        grid = pd.date_range(start, end, freq="3min", tz="UTC")
        yield _window(grid, candidate, list(candidate.target_weights.index))

    monkeypatch.setattr(run_mod, "_strategy_window_stream", stream)
    request = _request(data_root=_Path("root"), evaluation_end=pd.Timestamp("2021-04-03", tz="UTC"),
                       report_periods=(StrategyReportPeriod("P1", pd.Timestamp("2021-04-01", tz="UTC"),
                                                            pd.Timestamp("2021-04-02", tz="UTC")),))
    source = run_mod.load_strategy_source(request)
    run = run_strategy_backtest(request, source=source)
    direct = run_strategy_backtest(request)
    pd.testing.assert_frame_equal(run.candidate.target_weights, direct.candidate.target_weights)
    assert bool((run.candidate.signal_available_at == direct.candidate.signal_available_at).all())
    pd.testing.assert_frame_equal(run.evidence.period_metrics, direct.evidence.period_metrics, check_exact=True)
    for case in ("base", "stress"):
        pd.testing.assert_series_equal(getattr(run.evidence, f"{case}_daily").returns,
                                       getattr(direct.evidence, f"{case}_daily").returns, check_exact=True)
        pd.testing.assert_frame_equal(getattr(run.evidence, case).simulated_fills, getattr(direct.evidence, case).simulated_fills,
                                      check_exact=True)
        assert not getattr(run.evidence, case).simulated_fills.empty


def test_foreign_source_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path as _Path

    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence())
    request = _request(data_root=_Path("root"))
    source = run_mod.load_strategy_source(request)
    other = _request(data_root=_Path("root"), evaluation_end=pd.Timestamp("2021-04-11", tz="UTC"))
    with pytest.raises(DataIntegrityError, match="does not match"):
        run_strategy_backtest(other, source=source)


def test_audit_once_per_neighbor_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import src.cli.commands.backtest as backtest_mod

    seen: dict = {"loads": 0}
    index = pd.date_range("2021-01-01", periods=2400, freq="h", tz="UTC")
    panels = {key: pd.DataFrame(100.0 if key == "close" else 300000.0, index=index, columns=list(_SYMBOLS))
              for key in ("close", "quote_vol", "taker_buy_quote")}

    def panel(*args, **kwargs):
        seen["loads"] += 1
        return panels

    monkeypatch.setattr(run_mod, "load_base_panel", panel)
    monkeypatch.setattr(run_mod, "_load_funding_series", lambda symbols: ({}, {}))
    monkeypatch.setattr(run_mod, "evaluate_strategy_backtest", lambda *a, **k: _evidence())
    import src.engine.backtest_persist as persist_mod

    monkeypatch.setattr(
        persist_mod, "persist_strategy_backtest",
        lambda run, output, **kwargs: __import__("pathlib").Path(output).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(backtest_mod, "_strategy_run_statistics", lambda run: {})
    audits = []
    monkeypatch.setattr(run_mod, "assert_settlement_registry_complete", lambda *a, **k: audits.append(k))
    monkeypatch.setattr(backtest_mod, "_record_trial_for_window", lambda *a, **k: None)
    from pathlib import Path as _Path

    request = _request(data_root=_Path("root"))
    source = run_mod.load_strategy_source(request)
    assert seen["loads"] == 1
    from src.strategy.targets import MemberSnapshotCache

    cache = MemberSnapshotCache()
    output = tmp_path / "unit" / "result.json"
    output.parent.mkdir(parents=True)
    backtest_mod._run_neighbor_set(
        FLOW_MOM_TOP20, source_start=request.source_start,
        start=request.evaluation_start, end=request.evaluation_end,
        base_spec=request.base_spec, stress_spec=request.stress_spec,
        report_periods=request.report_periods, data_root=request.data_root,
        budget=None, execution_bound="OHLCV_IMMEDIATE_TAKER",
        output=output, source=source, snapshot_cache=cache,
    )
    assert seen["loads"] == 1
    assert len(audits) == 1


def test_shared_cache_accepts_changing_selected_universe(monkeypatch) -> None:
    from pathlib import Path
    from src.strategy.targets import MemberSnapshotCache
    import src.strategy.targets as targets_mod

    _install_source(monkeypatch, {})
    request = _request(data_root=Path("root"), strategy=dataclasses.replace(FLOW_MOM_TOP20, breadth=1))
    source = run_mod.load_strategy_source(request)
    cache = MemberSnapshotCache()
    calls = []
    real_compute = targets_mod.compute_source_readiness

    def compute(*args):
        calls.append(1)
        return real_compute(*args)

    monkeypatch.setattr(targets_mod, "compute_source_readiness", compute)
    for breadth in (1, 3):
        variant = dataclasses.replace(request, strategy=dataclasses.replace(request.strategy, breadth=breadth))
        shared, _ = run_mod.build_request_targets(variant, source=source, snapshot_cache=cache)
        direct, _ = run_mod.build_request_targets(variant)
        pd.testing.assert_frame_equal(shared.target_weights, direct.target_weights, check_exact=True)
    assert len(calls) == 3


def test_source_registry_changes_and_read_errors_fail_closed(monkeypatch) -> None:
    from pathlib import Path

    _install_source(monkeypatch, {})
    request = _request(data_root=Path("root"))
    source = run_mod.load_strategy_source(request)
    monkeypatch.setattr(run_mod, "venue_halt_registry_for_root", lambda root: types.SimpleNamespace(digest="changed"))
    with pytest.raises(DataIntegrityError, match="does not match"):
        run_mod.build_request_targets(request, source=source)

    def unreadable(root):
        raise DataIntegrityError("registry unreadable")

    monkeypatch.setattr(run_mod, "venue_halt_registry_for_root", unreadable)
    with pytest.raises(DataIntegrityError, match="registry unreadable"):
        run_mod.load_strategy_source(request)
