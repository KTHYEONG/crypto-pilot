"""Prefix-invariance guards for the canonical process path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mhs.backtest.market_data import build_candidate_member_books
from src.mhs.params import PROCESS_FEATURE_CANDIDATES, PROCESS_MIN_SYMBOLS


def _synthetic_panels(n_bars: int = 1000, n_symbols: int = 10, seed: int = 401):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n_bars, freq="h", tz="UTC")
    columns = [f"P{i:02d}USDT" for i in range(n_symbols)]
    close = pd.DataFrame(
        np.exp(rng.normal(0, 0.01, (n_bars, n_symbols)).cumsum(axis=0)),
        index=index,
        columns=columns,
    )
    quote_vol = pd.DataFrame(
        np.abs(rng.normal(1e6, 1e5, (n_bars, n_symbols))), index=index, columns=columns
    )
    panels = {
        "close": close,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "quote_vol": quote_vol,
        "taker_buy_quote": quote_vol * 0.5,
    }
    return panels, index[::24]


def _full_knowledge(panels, funding: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=funding.index, columns=list(funding.columns))


def _books(panels, funding, known, decisions):
    eligible = pd.DataFrame(True, index=panels["close"].index, columns=panels["close"].columns)
    mask = eligible.copy()
    return build_candidate_member_books(
        panels, funding, eligible, mask, decisions, funding_known_1h=known
    )


def test_process_books_ignore_future_funding_disappearance() -> None:
    """Future funding disappearance: earlier books match; later unknown finance is invalid."""
    panels, decisions = _synthetic_panels()
    rng = np.random.default_rng(9)
    funding = pd.DataFrame(
        rng.normal(0, 1e-5, panels["close"].shape),
        index=panels["close"].index,
        columns=panels["close"].columns,
    )
    cutoff = decisions[20]
    vanished = funding.copy()
    vanished.loc[vanished.index > cutoff, ["P00USDT", "P01USDT"]] = np.nan
    known_before = _full_knowledge(panels, funding)
    known_after = known_before.copy()
    known_after.loc[known_after.index > cutoff, ["P00USDT", "P01USDT"]] = False
    before = _books(panels, funding.fillna(0.0), known_before, decisions)
    after = _books(panels, vanished.fillna(0.0), known_after, decisions)
    assert list(before.keys()) == list(after.keys())
    for name, book in before.items():
        pd.testing.assert_frame_equal(after[name].loc[:cutoff], book.loc[:cutoff])
    carry = next(k for k in before if k.startswith("funding_carry_"))
    late = decisions[decisions > cutoff + pd.Timedelta(hours=200)]
    assert len(late) > 0
    assert bool((after[carry].loc[late, ["P00USDT", "P01USDT"]] == 0.0).all().all())
    for name in PROCESS_FEATURE_CANDIDATES:
        pd.testing.assert_frame_equal(after[name], before[name])


def test_process_membership_ignores_static_source_gap_exclusions(monkeypatch) -> None:
    """Retrospective exclusion immunity: the static gap list cannot move history."""
    import src.mhs.backtest.market_data as bt_market
    import src.mhs.data_policy as data_policy

    panels, _ = _synthetic_panels(n_bars=800, n_symbols=6, seed=77)
    grid = panels["close"].index
    funding = {s: pd.Series(np.full(len(grid), 1e-6), index=grid) for s in panels["close"].columns}
    monkeypatch.setattr(bt_market, "load_base_panel", lambda *a, **k: dict(panels))
    monkeypatch.setattr(
        bt_market, "_load_funding_series", lambda syms: ({s: funding[s] for s in syms}, {})
    )
    monkeypatch.setattr(bt_market, "apply_dynamic_gap_exclusion", lambda mask, *a, **k: (mask, {}))
    start, end = grid[0], grid[0] + pd.Timedelta(days=30)
    baseline = bt_market.load_process_market_data(start, end)
    monkeypatch.setattr(
        data_policy,
        "SOURCE_GAP_EXCLUDED_SYMBOLS",
        frozenset({panels["close"].columns[0], "P99USDT"}),
    )
    rerun = bt_market.load_process_market_data(start, end)
    assert list(rerun.opens_1h.columns) == list(baseline.opens_1h.columns)
    assert list(rerun.bar_funding_1h.columns) == list(baseline.bar_funding_1h.columns)
    assert list(rerun.member_books.keys()) == list(baseline.member_books.keys())
    for name in rerun.member_books:
        pd.testing.assert_frame_equal(rerun.member_books[name], baseline.member_books[name])
    pd.testing.assert_frame_equal(rerun.execution_mask, baseline.execution_mask)


def test_process_population_invariance_under_future_only_evidence() -> None:
    """End-to-end population invariance: post-T perturbations leave the prefix exact."""
    panels, decisions = _synthetic_panels()
    base_columns = list(panels["close"].columns)
    cutoff = decisions[20]
    shocked = {k: v.copy() for k, v in panels.items()}
    future = pd.Series(np.nan, index=panels["close"].index, dtype="float64")
    future.loc[future.index > cutoff] = 123.0
    for key in shocked:
        shocked[key] = shocked[key].copy()
        shocked[key]["P10USDT"] = future if key != "quote_vol" else future * 1e4
        shocked[key]["P10USDT"] = shocked[key]["P10USDT"].astype("float64")
    shocked["close"].loc[shocked["close"].index > cutoff, "P00USDT"] *= 1.5
    shocked["close"].loc[
        (shocked["close"].index > cutoff) & (shocked["close"].index <= cutoff + pd.Timedelta(hours=12)),
        "P01USDT",
    ] = np.nan
    funding = pd.DataFrame(
        1e-6, index=panels["close"].index, columns=base_columns
    )
    funding_shocked = funding.copy()
    funding_shocked["P10USDT"] = np.nan
    funding_shocked.loc[funding_shocked.index > cutoff, "P10USDT"] = 2e-6
    known = pd.DataFrame(True, index=funding.index, columns=base_columns)
    known_shocked = pd.DataFrame(True, index=funding.index, columns=[*base_columns, "P10USDT"])
    known_shocked.loc[known_shocked.index <= cutoff, "P10USDT"] = False
    base_books = _books(panels, funding, known, decisions)
    joint_books = _books(shocked, funding_shocked.fillna(0.0), known_shocked, decisions)
    assert list(joint_books.keys()) == list(base_books.keys())
    for name, book in base_books.items():
        pd.testing.assert_frame_equal(
            joint_books[name].loc[:cutoff, base_columns], book.loc[:cutoff]
        )


def test_process_missing_feature_cannot_count_toward_minimum() -> None:
    """Missing feature cannot count: seven finite features never admit an eight minimum."""
    panels, decisions = _synthetic_panels(n_bars=800, n_symbols=8, seed=55)
    grid = panels["close"].index
    missing_bar = grid[600]
    panels["close"].loc[[missing_bar], "P07USDT"] = np.nan
    funding = pd.DataFrame(1e-6, index=grid, columns=list(panels["close"].columns))
    books = _books(panels, funding, _full_knowledge(panels, funding), decisions)
    book = books["mom_168h"]
    assert PROCESS_MIN_SYMBOLS == 8
    assert bool((book.loc[missing_bar] == 0.0).all())
    settled = decisions[decisions > missing_bar + pd.Timedelta(hours=168)][0]
    assert abs(float(book.loc[settled].abs().sum()) - 1.0) < 1e-9


def _causal_market(n_days: int = 90, seed: int = 901):
    from src.mhs.backtest.contracts import ProcessMarketData

    rng = np.random.default_rng(seed)
    symbols = ["C0USDT", "C1USDT", "C2USDT", "C3USDT"]
    decisions = pd.date_range("2021-01-01", periods=n_days, freq="24h", tz="UTC")
    grid = pd.date_range(decisions[0], decisions[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0, 0.002, (n_days, len(symbols))), axis=0),
        index=decisions,
        columns=symbols,
    )
    funding_step = pd.DataFrame(
        rng.normal(0, 1e-6, (n_days, len(symbols))), index=decisions, columns=symbols
    )
    drift_a = pd.DataFrame(0.0, index=decisions, columns=symbols)
    drift_a["C0USDT"] = 0.5
    drift_a["C1USDT"] = -0.5
    drift_b = pd.DataFrame(0.0, index=decisions, columns=symbols)
    drift_b["C2USDT"] = 0.5
    drift_b["C3USDT"] = -0.5
    wobble = pd.DataFrame(
        rng.normal(0, 0.01, (n_days, len(symbols))), index=decisions, columns=symbols
    )
    books = {"alpha": drift_a + wobble, "beta": drift_b - wobble}
    opens = pd.DataFrame(
        100.0 + rng.normal(0, 0.5, (len(grid), len(symbols))), index=grid, columns=symbols
    )
    return ProcessMarketData(
        grid_1h=grid,
        decision_grid=decisions,
        opens_1h=opens,
        bar_funding_1h=pd.DataFrame(0.0, index=grid, columns=symbols),
        log_close_step=log_close,
        funding_step=funding_step,
        member_books=books,
        execution_mask=pd.DataFrame(True, index=decisions, columns=symbols),
        funding_known_1h=pd.DataFrame(True, index=grid, columns=symbols),
    )


def _causal_run(data, schedule):
    from src.mhs.backtest.labels import ProcessClockSpec, build_proxy_member_returns
    from src.mhs.backtest.paths import run_process_paths

    clock = ProcessClockSpec(
        decision_period=pd.Timedelta(hours=24),
        bar_completion_lag=pd.Timedelta(hours=1),
        fit_latency=pd.Timedelta(0),
    )
    evidence = build_proxy_member_returns(
        data, clock=clock, one_way_bps=8.0, procedure_digest="p", input_manifest_digest=None
    )
    (path,) = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
        clock=clock, member_evidence=evidence,
    )
    return path, evidence


def test_process_holdings_continue_across_refit_boundary() -> None:
    """Continuous refit boundary: state continues and every post-warmup day is retained."""
    from src.mhs.books import scale_book_to_target_gross
    from src.mhs.params import PROCESS_SMOOTHING_HALFLIFE_DAYS
    from src.mhs.process import RefitPoint, ema_smoothing_rate

    data = _causal_market()
    decisions = data.decision_grid
    boundary = decisions[60]
    schedule = (
        RefitPoint(
            effective_from=decisions[30], effective_to=boundary, train_end=decisions[30]
        ),
        RefitPoint(
            effective_from=boundary, effective_to=decisions[80], train_end=boundary
        ),
    )
    path, _ = _causal_run(data, schedule)
    assert list(path.target_weights.index) == list(decisions[30:80])
    assert bool((path.unit_target_weights.abs().sum(axis=1) > 0).all())
    rate = ema_smoothing_rate(PROCESS_SMOOTHING_HALFLIFE_DAYS)
    aims = []
    for point, record in zip(schedule, path.refits, strict=True):
        rows = data.member_books["alpha"].index[
            (data.member_books["alpha"].index >= point.effective_from)
            & (data.member_books["alpha"].index < point.effective_to)
        ]
        combined = sum(
            (
                record.member_weights.get(name, 0.0) * data.member_books[name].loc[rows]
                for name in ("alpha", "beta")
            ),
            start=data.member_books["alpha"].loc[rows] * 0.0,
        )
        aims.append(scale_book_to_target_gross(combined, 1.0))
    aim = pd.concat(aims)
    state = np.zeros(len(data.member_books["alpha"].columns))
    expected = np.empty((len(aim), len(state)))
    for pos in range(len(aim)):
        state = state + rate * (aim.to_numpy()[pos] - state)
        expected[pos] = state
    pd.testing.assert_frame_equal(
        path.unit_target_weights,
        pd.DataFrame(expected, index=aim.index, columns=list(aim.columns)),
    )
    restart = rate * aim.loc[[boundary]]
    assert not np.allclose(
        path.unit_target_weights.loc[boundary].to_numpy(), restart.to_numpy(), atol=1e-12
    )


def test_inventory_replay_uses_path_signal_availability(monkeypatch) -> None:
    """Single availability authority: submission follows supplied timestamps, no second shift."""
    import src.mhs.backtest.inventory as bt_inventory
    from src.mhs.backtest.contracts import ProcessBacktestReport, ProcessPath
    from src.mhs.deploy_gate import DeployGateResult
    from src.mhs.process import ProcessExecutionPolicy
    from src.mhs.resources import MhsMemoryBudget

    targets = pd.DataFrame(
        [[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]],
        index=pd.date_range("2022-01-01", periods=3, freq="24h", tz="UTC"),
        columns=["AUSDT", "BUSDT"],
        dtype="float64",
    )
    delayed = pd.DatetimeIndex(targets.index + pd.Timedelta(hours=5))
    hourly = pd.date_range(targets.index[0], targets.index[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(0.0, index=targets.index),
        unit_daily_returns=pd.Series(0.0, index=targets.index),
        exposure=pd.Series(1.0, index=targets.index),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=targets,
        target_weights=targets,
        turnover_1h=pd.Series(0.0, index=hourly),
        signal_available_at=delayed,
        clock_mode="matured_labels",
    )
    proxy = ProcessBacktestReport(
        start=targets.index[0],
        end=targets.index[-1],
        certification_level="process_proxy_1h_ledger",
        n_candidates=2,
        base=path,
        stress=path,
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
    )
    captured: dict = {}

    def _fake_iter(decision_targets, signal_at, *args, **kwargs):
        from src.mhs.execution.contracts import ExecutionReplayWindow

        captured["signal_available_at"] = pd.DatetimeIndex(signal_at)
        windows = []
        for pos, day in enumerate(decision_targets.index):
            start = day if pos == 0 else decision_targets.index[pos - 1]
            grid = pd.date_range(start, day + pd.Timedelta(hours=23, minutes=57), freq="3min", tz="UTC")
            frame = pd.DataFrame(100.0, index=grid, columns=list(decision_targets.columns))
            known = pd.DataFrame(True, index=grid, columns=list(decision_targets.columns))
            windows.append(
                ExecutionReplayWindow(
                    window_start=grid[0],
                    window_end=grid[-1],
                    columns=tuple(decision_targets.columns),
                    symbols=tuple(decision_targets.columns),
                    minute_grid=grid,
                    highs=pd.DataFrame(100.5, index=grid, columns=list(decision_targets.columns)),
                    lows=pd.DataFrame(99.5, index=grid, columns=list(decision_targets.columns)),
                    closes=frame,
                    marks=frame.copy(),
                    bar_funding=pd.DataFrame(0.0, index=grid, columns=list(decision_targets.columns)),
                    target_weights=decision_targets.loc[[day]],
                    signal_available_at=pd.DatetimeIndex([pd.DatetimeIndex(signal_at)[pos]]),
                    quote_volumes=pd.DataFrame(1e6, index=grid, columns=list(decision_targets.columns)),
                    funding_known=known,
                    bar_available_at=grid,
                )
            )
        return iter(windows)

    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: proxy)
    monkeypatch.setattr(bt_inventory, "_load_funding_series", lambda syms: ({}, {}))
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _fake_iter)
    bt_inventory.evaluate_process_inventory_backtest(
        targets.index[0], targets.index[-1] + pd.Timedelta(days=1), memory_budget=MhsMemoryBudget()
    )
    assert captured["signal_available_at"].equals(delayed)
    assert not captured["signal_available_at"].equals(
        pd.DatetimeIndex(targets.index + pd.Timedelta(hours=1))
    )


def test_process_path_invariance_under_later_evidence_corruption() -> None:
    """Whole-path clock invariance: later-only corruption leaves pre-T outputs unchanged."""
    from src.mhs.process import RefitPoint

    data = _causal_market(n_days=60, seed=707)
    decisions = data.decision_grid
    cutoff = decisions[30]
    schedule = (
        RefitPoint(effective_from=decisions[20], effective_to=cutoff, train_end=decisions[20]),
        RefitPoint(effective_from=cutoff, effective_to=decisions[50], train_end=cutoff),
    )
    before, _ = _causal_run(data, schedule)
    import dataclasses

    shocked_books = {name: book.copy() for name, book in data.member_books.items()}
    for book in shocked_books.values():
        book.loc[book.index > cutoff] = book.loc[book.index > cutoff] * 17.0
    shocked_log = data.log_close_step.copy()
    shocked_log.loc[shocked_log.index > cutoff] = shocked_log.loc[shocked_log.index > cutoff] + 0.5
    shocked_funding = data.funding_step.copy()
    shocked_funding.loc[shocked_funding.index > cutoff] = 0.01
    shocked = dataclasses.replace(
        data,
        member_books=shocked_books,
        log_close_step=shocked_log,
        funding_step=shocked_funding,
    )
    after, _ = _causal_run(shocked, schedule)
    assert before.refits[0] == after.refits[0]
    pd.testing.assert_frame_equal(after.target_weights.loc[:cutoff], before.target_weights.loc[:cutoff])
    pd.testing.assert_series_equal(after.exposure.loc[:cutoff], before.exposure.loc[:cutoff])
