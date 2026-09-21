"""Invariant scenarios for the frozen-MHS historical runner."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.frozen_research_candidate import (
    FROZEN_MHS_TOP20_V2,
    FrozenMhsCandidate,
)
from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod
from src.mhs.source_gaps import SourceGapInterval
from src.mhs.types import ExecutionSpec

import src.mhs.frozen_research_run as run_mod
from src.mhs.frozen_research_run import (
    FrozenMhsBacktestRequest,
    assert_frozen_execution_coverage,
    frozen_blocked_decisions,
    run_frozen_mhs_backtest,
)

_REAL_PRECHECK = run_mod.assert_frozen_execution_coverage


@pytest.fixture(autouse=True)
def _bypass_frozen_coverage_precheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic-census runner tests exercise non-coverage stages; the precheck has its own scenarios below."""
    monkeypatch.setattr(run_mod, "assert_frozen_execution_coverage", lambda *a, **k: None)

_SYMBOLS = ("AAA", "BBB", "CCC", "DELISTED")


def _specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar")
    stress = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=13.0, decision_anchor="submit_bar")
    return base, stress


def _request(**overrides: object) -> FrozenMhsBacktestRequest:
    base, stress = _specs()
    params: dict[str, object] = {
        "source_start": pd.Timestamp("2021-01-01", tz="UTC"),
        "evaluation_start": pd.Timestamp("2021-04-01", tz="UTC"),
        "evaluation_end": pd.Timestamp("2021-04-10", tz="UTC"),
        "strategy": FROZEN_MHS_TOP20_V2,
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
    return FrozenMhsCandidate(target_weights=weights, signal_available_at=avail, strategy=FROZEN_MHS_TOP20_V2)


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
        "strategy": FROZEN_MHS_TOP20_V2,
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
    _bad("strategy", "frozen_mhs_top20_v2", "FrozenMhsStrategySpec")
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


_GAP_SYMBOLS = ("AAA", "PUMPUSDT", "LUNAUSDT", "BBB")


def _gap_source(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    idx_days = pd.date_range("2021-01-01", periods=100, freq="D", tz="UTC")
    daily_close = pd.DataFrame(100.0, index=idx_days, columns=list(_GAP_SYMBOLS), dtype="float64")
    daily_qv = pd.DataFrame(5_000_000.0, index=idx_days, columns=list(_GAP_SYMBOLS), dtype="float64")
    real_roster = run_mod.build_frozen_pit_roster
    real_build = run_mod.build_frozen_mhs_candidate

    def _fake_source(request: FrozenMhsBacktestRequest, budget: object, swap: object) -> tuple:
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
        captured["roster_blocked"] = kwargs.get("blocked_decisions")
        return real_roster(daily_close, daily_qv, census, **kwargs)  # type: ignore[arg-type]

    def _spy_build(
        panels: object, available: object, daily_close: object, daily_qv: object, census: object, **kwargs: object
    ) -> FrozenMhsCandidate:
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
        ),
        SourceGapInterval(
            symbol="LUNAUSDT", plane="ohlcv_3m",
            start=pd.Timestamp("2023-01-01", tz="UTC").to_pydatetime(),
            end=pd.Timestamp("2023-02-01", tz="UTC").to_pydatetime(),
            reason="SOURCE_ABSENT", evidence="probe fixture: bounded interval outside the window",
            verified_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(), resolved_at=None,
        ),
    )
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: injected)
    monkeypatch.setattr(run_mod, "_load_frozen_source", _fake_source)
    monkeypatch.setattr(run_mod, "build_frozen_pit_roster", _spy_roster)
    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", _spy_build)
    monkeypatch.setattr(run_mod, "evaluate_frozen_mhs_research", lambda *a, **k: _evidence())


def test_runner_threads_reviewed_exclusion_set(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _gap_source(monkeypatch, captured)
    run = run_frozen_mhs_backtest(_request())
    assert run.source_gap_excluded_symbols == ("PUMPUSDT",)
    assert run.source_gap_blocked_decisions == 100
    roster_blocked = captured["roster_blocked"]
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
    assert "LUNAUSDT" in run.source_symbols


def test_excluded_symbol_absent_from_hourly_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _gap_source(monkeypatch, captured)
    run_frozen_mhs_backtest(_request())
    assert "PUMPUSDT" not in captured["hourly_columns"]
    assert "LUNAUSDT" in captured["hourly_columns"]
    assert "AAA" in captured["hourly_columns"]


def test_blocked_decisions_uses_half_open_window() -> None:
    base, _ = _specs()
    days = pd.DatetimeIndex([pd.Timestamp("2022-02-28", tz="UTC")], tz="UTC")
    frame = frozen_blocked_decisions(days, ("MANAUSDT",), strategy=FROZEN_MHS_TOP20_V2, base_spec=base)
    assert not bool(frame.iloc[0, 0])


def test_blocked_decisions_blocks_holding_window_overlap() -> None:
    base, _ = _specs()
    days = pd.DatetimeIndex([pd.Timestamp("2022-02-27", tz="UTC")], tz="UTC")
    frame = frozen_blocked_decisions(days, ("MANAUSDT", "AAA"), strategy=FROZEN_MHS_TOP20_V2, base_spec=base)
    assert bool(frame.loc[days[0], "MANAUSDT"])
    assert not bool(frame.loc[days[0], "AAA"])


def test_blocked_decisions_empty_census_returns_empty_frame() -> None:
    base, _ = _specs()
    days = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    frame = frozen_blocked_decisions(days, (), strategy=FROZEN_MHS_TOP20_V2, base_spec=base)
    assert frame.shape == (3, 0)


def test_blocked_decisions_empty_registry_blocks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_mod, "active_intervals", lambda **kwargs: ())
    base, _ = _specs()
    days = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    frame = frozen_blocked_decisions(days, ("AAA",), strategy=FROZEN_MHS_TOP20_V2, base_spec=base)
    assert not bool(frame.to_numpy().any())


def test_held_source_stays_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", lambda *a, **k: _candidate(n_days=10))
    def _boom(*args: object, **kwargs: object) -> object:
        raise DataIntegrityError("missing 3m source for held AAA")

    monkeypatch.setattr(run_mod, "_iter_mhs_execution_windows", _boom)
    with pytest.raises(DataIntegrityError, match=r"missing 3m source"):
        run_frozen_mhs_backtest(_request())


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


def test_assert_frozen_execution_coverage_blocks_deficient_symbol(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")], "BBB": []})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-09", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_frozen_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
        )
    message = str(exc_info.value)
    assert "AAA" in message
    assert "2022-01-10T00:30:00+00:00" in message
    assert "2022-01-09T00:00:00+00:00" in message
    assert "BBB" not in message


def test_assert_frozen_execution_coverage_honours_entry_hour(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    # entry_hour=0 기준으로는 충족되지만, 6시 진입 변형에서는 6시간이 더 필요하다.
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10 00:30", freq="3min", tz="UTC"))
    assert_frozen_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
    )
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_frozen_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=6, data_root=tmp_path,
        )
    assert "2022-01-10T06:30:00+00:00" in str(exc_info.value)


def test_assert_frozen_execution_coverage_separates_unreadable_from_missing(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    corrupt = tmp_path / "ohlcv" / "3m" / "AAA.parquet"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not a parquet file")
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_frozen_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
        )
    message = str(exc_info.value)
    assert "UNREADABLE" in message
    assert "MISSING" not in message


def test_assert_frozen_execution_coverage_ignores_unselected_symbols(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-05", tz="UTC")], "GHOST": []})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-02-01", freq="3min", tz="UTC"))
    assert_frozen_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
    )


def test_assert_frozen_execution_coverage_passes_delisted_symbol(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10T00:30", freq="3min", tz="UTC"))
    assert_frozen_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-06-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
    )


def test_assert_frozen_execution_coverage_includes_settlement_slack(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError, match="AAA"):
        assert_frozen_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
        )


def test_assert_frozen_execution_coverage_tolerates_one_bar(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-08", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10T00:27", freq="3min", tz="UTC"))
    assert_frozen_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
    )


def test_assert_frozen_execution_coverage_reports_all_deficient_symbols(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    grant = [pd.Timestamp("2022-01-08", tz="UTC")]
    roster = _coverage_roster(days, {"AAA": grant, "BBB": grant, "CCC": grant})
    for symbol in ("AAA", "BBB", "CCC"):
        _write_3m(tmp_path, symbol, pd.date_range("2022-01-01", "2022-01-05", freq="3min", tz="UTC"))
    with pytest.raises(DataIntegrityError) as exc_info:
        assert_frozen_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
        )
    message = str(exc_info.value)
    assert "3 symbol(s)" in message
    assert all(symbol in message for symbol in ("AAA", "BBB", "CCC"))


def test_assert_frozen_execution_coverage_caps_requirement_at_fence(tmp_path) -> None:
    days = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    roster = _coverage_roster(days, {"AAA": [pd.Timestamp("2022-01-09", tz="UTC")]})
    _write_3m(tmp_path, "AAA", pd.date_range("2022-01-01", "2022-01-10", freq="3min", tz="UTC"))
    assert_frozen_execution_coverage(
        roster, execution_end=pd.Timestamp("2022-01-10", tz="UTC"),
        settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
    )


def test_assert_frozen_execution_coverage_treats_unreadable_archives_as_missing(tmp_path) -> None:
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
        assert_frozen_execution_coverage(
            roster, execution_end=pd.Timestamp("2022-02-01", tz="UTC"),
            settlement=_SETTLEMENT, entry_hour_utc=0, data_root=tmp_path,
        )
    message = str(exc_info.value)
    assert "4 symbol(s)" in message
    assert all(symbol in message for symbol in ("AAA", "BBB", "CCC", "DDD"))
    assert "MISSING" in message


def test_run_frozen_mhs_backtest_blocks_before_candidate_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """결손이 있는 요청은 후보 생성과 윈도우 스트림보다 먼저 멈춘다."""
    monkeypatch.setattr(run_mod, "assert_frozen_execution_coverage", _REAL_PRECHECK)
    seen: dict = {}
    _install_source(monkeypatch, seen)
    days = pd.date_range("2021-04-01", periods=9, freq="D", tz="UTC")
    roster = _coverage_roster(
        days, {"AAA": [pd.Timestamp("2021-04-08", tz="UTC")], "BBB": [], "CCC": [], "DELISTED": []},
    )
    monkeypatch.setattr(run_mod, "build_frozen_pit_roster", lambda *a, **k: roster)
    _write_3m(tmp_path, "AAA", pd.date_range("2021-04-01", "2021-04-05", freq="3min", tz="UTC"))
    called: list[str] = []
    monkeypatch.setattr(
        run_mod, "build_frozen_mhs_candidate",
        lambda *a, **k: (called.append("build"), _candidate(n_days=10))[1],
    )
    monkeypatch.setattr(
        run_mod, "evaluate_frozen_mhs_research",
        lambda *a, **k: (called.append("evaluate"), _evidence())[1],
    )
    with pytest.raises(DataIntegrityError, match="AAA"):
        run_frozen_mhs_backtest(_request(data_root=tmp_path))
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
    monkeypatch.setattr(run_mod, "build_frozen_mhs_candidate", lambda *a, **k: _candidate(n_days=10))

    def _spy(candidate: object, windows: object, **kwargs: object) -> object:
        seen["execution_bound"] = kwargs.get("execution_bound")
        return _evidence()

    monkeypatch.setattr(run_mod, "evaluate_frozen_mhs_research", _spy)
    run_frozen_mhs_backtest(_request(execution_bound="OHLCV_STRICT_PROXY"))
    assert seen["execution_bound"] == "OHLCV_STRICT_PROXY"


def test_candidate_builder_matches_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner reuses the standalone candidate builder without replay divergence."""
    seen: dict = {}
    _install_source(monkeypatch, seen)
    monkeypatch.setattr(run_mod, "evaluate_frozen_mhs_research", lambda *a, **k: _evidence())
    request = _request()
    direct, context = run_mod.build_frozen_request_candidate(request)
    run = run_frozen_mhs_backtest(request)
    pd.testing.assert_frame_equal(direct.target_weights, run.candidate.target_weights)
    assert bool((direct.signal_available_at == run.candidate.signal_available_at).all())
    assert context.census == run.source_symbols
