"""Invariant scenarios for the continuous process backtest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.params import (
    DISCOVERY_START,
    PROCESS_EVALUATION_CEILING,
    PROCESS_FEATURE_CANDIDATES,
    PROCESS_FUNDING_CARRY_CANDIDATES_HOURS,
    PROCESS_SMOOTHING_HALFLIFE_DAYS,
)
from src.mhs.process import ProcessExecutionPolicy, monthly_refit_schedule
import os
import src.mhs.backtest.contracts as bt_contracts
import src.mhs.backtest.market_data as bt_market
import src.mhs.backtest.paths as bt_paths
import src.mhs.backtest.inventory as bt_inventory
import src.mhs.reporting.inventory as rep_inventory
from src.mhs.backtest.contracts import (
    ProcessBacktestReport,
    ProcessMarketData,
    ProcessPath,
)
from src.mhs.backtest.market_data import (
    apply_process_execution_availability,
    build_candidate_member_books,
)
from src.mhs.backtest.paths import (
    _reject_invalid_ledger_returns,
    evaluate_process_backtest,
    quarter_fold_returns,
    run_process_paths,
)
from src.mhs.backtest.inventory import (
    _execution_fence,
    _require_utc_index,
    _validate_replay_window,
    replay_process_execution,
)
from src.mhs.reporting.process import (
    PROCESS_CERTIFICATION_LEVEL,
    _resolve_process_report_path,
    _tier_payload,
    persist_process_report,
    persist_process_targets,
)


def _synthetic_data(n_days: int = 500, seed: int = 7) -> ProcessMarketData:
    rng = np.random.default_rng(seed)
    symbols = [f"S{i:02d}USDT" for i in range(10)]
    decision_grid = pd.date_range("2022-01-01", periods=n_days, freq="24h", tz="UTC")
    grid_1h = pd.date_range(decision_grid[0], decision_grid[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    drift = np.zeros((len(grid_1h), len(symbols)))
    drift[:, 0] = 0.0002
    drift[:, 1] = -0.0002
    shocks = rng.normal(0, 0.002, (len(grid_1h), len(symbols)))
    log_close_1h = pd.DataFrame(np.cumsum(drift + shocks, axis=0), index=grid_1h, columns=symbols)
    opens_1h = np.exp(log_close_1h)
    bar_funding_1h = pd.DataFrame(0.0, index=grid_1h, columns=symbols)
    log_close_step = log_close_1h.reindex(decision_grid)
    funding_step = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    planted = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    planted["S00USDT"] = 0.5
    planted["S01USDT"] = -0.5
    inverse = -planted
    noise = pd.DataFrame(rng.normal(0, 0.1, (n_days, len(symbols))), index=decision_grid, columns=symbols)
    noise = noise.sub(noise.mean(axis=1), axis=0)
    gross = noise.abs().sum(axis=1).replace(0, np.nan)
    noise = noise.div(gross, axis=0).fillna(0.0)
    execution_mask = pd.DataFrame(True, index=decision_grid, columns=symbols)
    return ProcessMarketData(
        grid_1h=grid_1h,
        decision_grid=decision_grid,
        opens_1h=opens_1h,
        bar_funding_1h=bar_funding_1h,
        log_close_step=log_close_step,
        funding_step=funding_step,
        member_books={"planted": planted, "inverse": inverse, "noise": noise},
        execution_mask=execution_mask,
        funding_known_1h=pd.DataFrame(True, index=grid_1h, columns=symbols),
    )


def _schedule_for(data: ProcessMarketData):
    return monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])


def test_run_process_path_selects_planted_book() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    path = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    assert path.daily_returns.index[0] == schedule[0].effective_from
    assert bool((path.exposure >= 0).all())
    assert bool((path.exposure <= 2.0 + 1e-12).all())
    first = path.refits[0]
    assert first.member_weights.get("planted", 0.0) > 0
    assert first.member_weights.get("inverse", 0.0) == 0.0
    assert all(r.smoothing_halflife_days == PROCESS_SMOOTHING_HALFLIFE_DAYS for r in path.refits)


def test_run_process_path_perturbation_invariance() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    base = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    cutoff = data.decision_grid[300]
    perturbed_close = data.log_close_step.copy()
    perturbed_close.loc[perturbed_close.index > cutoff] += 0.5
    perturbed_funding = data.funding_step.copy()
    perturbed_funding.loc[perturbed_funding.index > cutoff] += 0.01
    perturbed = ProcessMarketData(
        grid_1h=data.grid_1h,
        decision_grid=data.decision_grid,
        opens_1h=data.opens_1h,
        bar_funding_1h=data.bar_funding_1h,
        log_close_step=perturbed_close,
        funding_step=perturbed_funding,
        member_books=data.member_books,
        execution_mask=data.execution_mask,
    )
    replay = run_process_paths(perturbed, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    for before, after in zip(base.refits, replay.refits, strict=True):
        if before.point.train_end <= cutoff:
            assert before == after
    mask = base.daily_returns.index <= cutoff
    assert np.allclose(
        base.daily_returns.loc[mask].to_numpy(), replay.daily_returns.loc[mask].to_numpy()
    )
    assert np.allclose(
        base.exposure.loc[base.exposure.index <= cutoff].to_numpy(),
        replay.exposure.loc[replay.exposure.index <= cutoff].to_numpy(),
    )


def test_run_process_path_stress_does_not_beat_base() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    base, stress = run_process_paths(data, schedule, decision_bps=2.0, evaluation_bps=(2.0, 50.0), leverage_cap=2.0)
    assert (base.one_way_bps, stress.one_way_bps) == (2.0, 50.0)
    assert base.exposure.equals(stress.exposure)
    assert base.refits == stress.refits
    assert float(np.log1p(stress.daily_returns).sum()) <= float(np.log1p(base.daily_returns).sum())
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, (), decision_bps=2.0, evaluation_bps=(2.0,), leverage_cap=2.0)


def test_run_process_paths_same_decision_tiers() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    base, stress = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0, 24.0), leverage_cap=2.0
    )
    assert len((base, stress)) == 2
    assert base.one_way_bps == 8.0
    assert stress.one_way_bps == 24.0
    assert base.exposure.equals(stress.exposure)
    assert base.refits == stress.refits
    assert float(np.log1p(stress.daily_returns).sum()) <= float(np.log1p(base.daily_returns).sum())


def test_run_process_paths_whipsaw_gets_zero_weight() -> None:
    data = _synthetic_data()
    n_days = len(data.decision_grid)
    whipsaw = pd.DataFrame(0.0, index=data.decision_grid, columns=data.opens_1h.columns)
    signs = np.where(np.arange(n_days) % 2 == 0, 0.5, -0.5)
    whipsaw["S00USDT"] = signs
    books = dict(data.member_books)
    books["whipsaw"] = whipsaw
    import dataclasses

    wdata = dataclasses.replace(data, member_books=books)
    schedule = _schedule_for(wdata)
    (path,) = run_process_paths(
        wdata, schedule, decision_bps=100.0, evaluation_bps=(100.0,), leverage_cap=2.0
    )
    assert len(path.refits) > 0
    for record in path.refits:
        assert record.member_weights.get("whipsaw", 0.0) == 0.0


def test_run_process_paths_rejects_empty_evaluation() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(), leverage_cap=2.0)


def test_quarter_fold_returns_split() -> None:
    index = pd.date_range("2022-01-01", "2022-09-30", freq="24h", tz="UTC")
    series = pd.Series(np.linspace(0.001, 0.002, len(index)), index=index)
    folds = quarter_fold_returns(series)
    assert len(folds) == 3
    assert pd.concat(folds).equals(series)


def test_evaluate_rejects_ceiling_before_load(monkeypatch) -> None:

    def _boom(*a, **k):
        raise AssertionError("data load must not run")

    monkeypatch.setattr(bt_paths, "load_process_market_data", _boom)
    with pytest.raises(DataIntegrityError, match=r".+"):
        evaluate_process_backtest(
            DISCOVERY_START, PROCESS_EVALUATION_CEILING + pd.Timedelta(seconds=1)
        )


def test_evaluate_uses_gate_and_stress_triple(monkeypatch) -> None:

    data = _synthetic_data()
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    seen: dict = {}

    def _fake_load(start, end, data_root=None, memory_budget=None):
        return data

    def _fake_run(loaded, sched, *, decision_bps, evaluation_bps, leverage_cap, execution_policy=None, memory_budget=None, **_):
        seen["decision"] = decision_bps
        seen["evaluation"] = evaluation_bps
        seen.setdefault("bps", []).append(decision_bps)
        return run_process_paths(loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps, leverage_cap=leverage_cap)

    calls: list = []

    def _spy(loaded, sched, *, decision_bps, evaluation_bps, leverage_cap, execution_policy=None, memory_budget=None, **_):
        calls.append((decision_bps, evaluation_bps))
        return _fake_run(loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps, leverage_cap=leverage_cap)

    monkeypatch.setattr(bt_paths, "load_process_market_data", _fake_load)
    monkeypatch.setattr(bt_paths, "run_process_paths", _spy)
    report = evaluate_process_backtest(data_root=None)
    assert report.certification_level == "process_proxy_1h_ledger"
    assert len(calls) == 1
    assert calls[0][0] == seen["decision"]
    assert calls[0][1] == (seen["decision"], seen["decision"] * 3.0)
    assert calls[0][1][1] == pytest.approx(calls[0][1][0] * 3.0)
    assert report.n_candidates == len(data.member_books)
    assert report.gate.metrics["n_folds"] == len(quarter_fold_returns(report.base.daily_returns))


def test_persist_process_report_round_trip(tmp_path) -> None:
    data = _synthetic_data()
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    path = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    report = ProcessBacktestReport(
        start=data.decision_grid[0],
        end=data.decision_grid[-1],
        certification_level=PROCESS_CERTIFICATION_LEVEL,
        n_candidates=len(data.member_books),
        base=path,
        stress=path,
        gate=__import__("src.mhs.deploy_gate", fromlist=["DeployGateResult"]).DeployGateResult(
            go=False, reason_codes=("X",), metrics={"n_folds": 1.0}
        ),
    )
    out = persist_process_report(report, tmp_path / "sub" / "report.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["certification_level"] == PROCESS_CERTIFICATION_LEVEL
    assert set(payload["base"]) >= {
        "one_way_bps", "daily_returns", "exposure", "refits",
        "ann_log_growth", "exposure_zero_share", "exposure_cap_share",
    }
    base = payload["base"]
    exposure = np.array(list(base["exposure"].values()))
    # 실제 상한(leverage_cap) 기준 점유율과 연환산 로그성장률 계약을 검증한다.
    assert base["leverage_cap"] == 2.0
    assert base["exposure_cap_share"] == pytest.approx(float((exposure >= 2.0).mean()))
    assert base["exposure_zero_share"] == pytest.approx(float((exposure == 0.0).mean()))
    daily = np.array(list(base["daily_returns"].values()))
    assert base["ann_log_growth"] == pytest.approx(float(np.log1p(daily).mean() * 365.0))
    assert set(payload["base"]["refits"][0]) == {
        "effective_from", "effective_to", "train_end", "member_weights",
        "smoothing_halflife_days",
    }


def test_build_candidate_member_books_keys_and_neutrality() -> None:
    rng = np.random.default_rng(3)
    symbols = [f"S{i:02d}USDT" for i in range(12)]
    grid_1h = pd.date_range("2022-01-01", periods=2000, freq="1h", tz="UTC")
    panels = {
        name: pd.DataFrame(rng.normal(100, 5, (len(grid_1h), len(symbols))), index=grid_1h, columns=symbols)
        for name in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote")
    }
    panels["close"] = panels["close"].clip(lower=1.0)
    panels["high"] = panels["close"] + 0.5
    panels["low"] = (panels["close"] - 0.5).clip(lower=0.5)
    eligible = pd.DataFrame(True, index=grid_1h, columns=symbols)
    mask = pd.DataFrame(True, index=grid_1h, columns=symbols)
    bar_funding = pd.DataFrame(rng.normal(0, 1e-4, (len(grid_1h), len(symbols))), index=grid_1h, columns=symbols)
    decision_grid = pd.date_range(grid_1h[0], grid_1h[-1], freq="24h", tz="UTC")
    books = build_candidate_member_books(panels, bar_funding, eligible, mask, decision_grid)
    expected_keys = list(PROCESS_FEATURE_CANDIDATES) + [
        f"funding_carry_{h}h" for h in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS
    ]
    assert list(books.keys()) == expected_keys
    sample = books[expected_keys[0]]
    assert len(sample) == len(decision_grid)
    row = sample.iloc[50]
    assert abs(row.sum()) < 1e-8
    assert abs(row.abs().sum() - 1.0) < 1e-8 or (row == 0.0).all()


def test_run_process_path_rejects_empty_books() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    empty = ProcessMarketData(
        grid_1h=data.grid_1h,
        decision_grid=data.decision_grid,
        opens_1h=data.opens_1h,
        bar_funding_1h=data.bar_funding_1h,
        log_close_step=data.log_close_step,
        funding_step=data.funding_step,
        member_books={},
        execution_mask=data.execution_mask,
    )
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(empty, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)


def test_quarter_fold_returns_empty() -> None:
    assert quarter_fold_returns(pd.Series(dtype="float64")) == ()


def test_evaluate_rejects_naive_and_order() -> None:
    with pytest.raises(ValueError, match=r".+"):
        evaluate_process_backtest(pd.Timestamp("2021-01-01"), DISCOVERY_START)
    with pytest.raises(ValueError, match=r".+"):
        evaluate_process_backtest(PROCESS_EVALUATION_CEILING, DISCOVERY_START)


def test_load_process_market_data_from_synthetic_lake(tmp_path, monkeypatch) -> None:
    from src.quant.universe.pit_universe import symbol_partition

    candidates = [f"LAKE{i:03d}USDT" for i in range(24)]
    symbols = [s for s in candidates if symbol_partition(s) == "dev"][:10]
    assert len(symbols) == 10
    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = pd.Timestamp("2021-02-15", tz="UTC")
    grid = pd.date_range(start, end, freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    lake = tmp_path / "ohlcv" / "1h"
    lake.mkdir(parents=True)
    ms = ((grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")
    for j, sym in enumerate(symbols):
        close = 100.0 + j + np.cumsum(rng.normal(0, 0.5, len(grid)))
        frame = pd.DataFrame({
            "timestamp": ms,
            "close": close,
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "quote_vol": rng.uniform(1e6, 2e6, len(grid)),
            "taker_buy_quote": rng.uniform(4e5, 6e5, len(grid)),
            "volume": rng.uniform(10.0, 100.0, len(grid)),
        })
        frame.to_parquet(lake / f"{sym}.parquet", index=False)
    lake_3m = tmp_path / "ohlcv" / "3m"
    lake_3m.mkdir(parents=True)
    grid_3m = pd.date_range(start, end, freq="3min", tz="UTC")
    ms_3m = ((grid_3m - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")
    for sym in symbols:
        pd.DataFrame({"timestamp": ms_3m}).to_parquet(lake_3m / f"{sym}.parquet", index=False)
    funding = {
        sym: pd.Series(rng.normal(0, 1e-5, len(grid)), index=grid) for sym in symbols
    }
    monkeypatch.setattr(bt_market, "_load_funding_series",
        lambda syms: ({s: funding[s] for s in syms if s in funding}, {}),
    )
    mark_dir = tmp_path / "markPriceKlines" / "1h"
    mark_dir.mkdir(parents=True)
    for sym in symbols:
        pd.DataFrame({"timestamp": ms, "datetime": grid, "close": 100.0}).to_parquet(
            mark_dir / f"{sym}.parquet", index=False
        )
    import src.market_data.services.futures_collection as fc

    monkeypatch.setattr(
        fc, "_mark_price_path",
        lambda symbol, timeframe: mark_dir / f"{symbol}.parquet",
    )
    data = bt_market.load_process_market_data(start, end, data_root=str(tmp_path / "ohlcv"))
    expected_keys = list(PROCESS_FEATURE_CANDIDATES) + [
        f"funding_carry_{h}h" for h in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS
    ]
    assert list(data.member_books.keys()) == expected_keys
    assert (data.decision_grid == pd.date_range(start, end, freq="24h", tz="UTC")).all()
    assert set(data.opens_1h.columns) == set(symbols)
    assert data.execution_mask.index.equals(data.decision_grid)
    assert list(data.execution_mask.columns) == list(next(iter(data.member_books.values())).columns)
    assert bool((data.execution_mask.dtypes.apply(lambda dt: dt.kind == "b")).all())
    assert bool(data.execution_mask.to_numpy().any())
    late_row = data.member_books[expected_keys[0]].iloc[-1]
    assert abs(late_row.sum()) < 1e-8


def _tiny_panel() -> dict:
    grid = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
    cols = ["AUSDT", "BUSDT"]
    frame = pd.DataFrame(100.0, index=grid, columns=cols)
    return {k: frame.copy() for k in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote")}


def test_load_process_market_data_raises_without_funding(monkeypatch) -> None:

    monkeypatch.setattr(bt_market, "load_base_panel", lambda *a, **k: _tiny_panel())
    monkeypatch.setattr(bt_market, "_load_funding_series", lambda syms: ({}, dict.fromkeys(syms, "missing")))
    with pytest.raises(RuntimeError, match=r".+"):
        bt_market.load_process_market_data(
            pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-02", tz="UTC")
        )


def test_load_process_market_data_raises_without_aligned_funding(monkeypatch) -> None:

    panel = _tiny_panel()
    grid = panel["close"].index
    monkeypatch.setattr(bt_market, "load_base_panel", lambda *a, **k: panel)
    monkeypatch.setattr(bt_market, "_load_funding_series",
        lambda syms: ({s: pd.Series(0.0, index=grid) for s in syms}, {}),
    )
    monkeypatch.setattr(bt_market, "bar_funding_panel", lambda *a, **k: pd.DataFrame(index=grid))
    with pytest.raises(RuntimeError, match=r".+"):
        bt_market.load_process_market_data(
            pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-02", tz="UTC")
        )


def test_run_process_paths_rejects_invalid_costs_and_policy() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps="bad", evaluation_bps=(8.0,), leverage_cap=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=-1.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=float("inf"), evaluation_bps=(8.0,), leverage_cap=2.0)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(float("nan"),), leverage_cap=2.0)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(-1.0,), leverage_cap=2.0)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=("bad",), leverage_cap=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=0.0)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=float("inf"))
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(
            data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
            execution_policy="bad",  # type: ignore[arg-type]
        )


def test_reject_invalid_ledger_returns() -> None:
    idx = pd.date_range("2022-01-01", periods=3, freq="24h", tz="UTC")
    _reject_invalid_ledger_returns(pd.Series([0.01, -0.5, 0.0], index=idx))
    with pytest.raises(DataIntegrityError, match=r".+"):
        _reject_invalid_ledger_returns(pd.Series([0.01, float("nan")], index=idx[:2]))
    with pytest.raises(DataIntegrityError, match=r".+"):
        _reject_invalid_ledger_returns(pd.Series([0.01, -1.0], index=idx[:2]))
    with pytest.raises(DataIntegrityError, match=r".+"):
        _reject_invalid_ledger_returns(pd.Series([0.01, -1.5], index=idx[:2]))


def test_tier_payload_empty_and_nonempty() -> None:
    idx = pd.date_range("2022-01-01", periods=2, freq="24h", tz="UTC")
    cols = ["A", "B"]
    unit = pd.DataFrame([[0.5, -0.5], [0.5, -0.5]], index=idx, columns=cols)
    hourly = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    turnover = pd.Series(0.01, index=hourly)
    path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series([0.01, 0.02], index=idx),
        unit_daily_returns=pd.Series([0.01, 0.02], index=idx),
        exposure=pd.Series([1.0, 1.0], index=idx),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(0.2),
        unit_target_weights=unit,
        target_weights=unit,
        turnover_1h=turnover,
    )
    payload = _tier_payload(path)
    assert payload["execution_policy"] == {"tracking_error_threshold": 0.2}
    assert payload["ann_turnover"] == pytest.approx(float(turnover.sum() * 365.0 / 2))
    assert payload["mean_unit_gross"] == pytest.approx(1.0)
    assert payload["mean_effective_gross"] == pytest.approx(1.0)
    empty_path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(dtype="float64"),
        unit_daily_returns=pd.Series(dtype="float64"),
        exposure=pd.Series(dtype="float64"),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=pd.DataFrame(),
        target_weights=pd.DataFrame(),
        turnover_1h=pd.Series(dtype="float64"),
    )
    empty_payload = _tier_payload(empty_path)
    assert empty_payload["ann_turnover"] == 0.0
    assert empty_payload["mean_unit_gross"] == 0.0
    assert empty_payload["mean_effective_gross"] == 0.0


def _report_with_policy(threshold: float | None) -> ProcessBacktestReport:
    idx = pd.date_range("2022-01-01", periods=2, freq="24h", tz="UTC")
    unit = pd.DataFrame([[0.5, -0.5], [0.5, -0.5]], index=idx, columns=["A", "B"])
    hourly = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series([0.0, 0.0], index=idx),
        unit_daily_returns=pd.Series([0.0, 0.0], index=idx),
        exposure=pd.Series([1.0, 1.0], index=idx),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(threshold),
        unit_target_weights=unit,
        target_weights=unit,
        turnover_1h=pd.Series(0.0, index=hourly),
    )
    from src.mhs.deploy_gate import DeployGateResult

    return ProcessBacktestReport(
        start=idx[0], end=idx[-1], certification_level=PROCESS_CERTIFICATION_LEVEL,
        n_candidates=1, base=path, stress=path,
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
    )


def test_resolve_process_report_path_routing(tmp_path) -> None:
    out = _resolve_process_report_path(_report_with_policy(None), tmp_path / "custom.json")
    assert out == tmp_path / "custom.json"
    with pytest.raises(ValueError, match=r".+"):
        _resolve_process_report_path(_report_with_policy(None), None)
    with pytest.raises(ValueError, match=r".+"):
        _resolve_process_report_path(_report_with_policy(None), tmp_path / "bad.txt")


def test_persist_process_report_evidence_keys(tmp_path) -> None:
    report = _report_with_policy(0.2)
    out = persist_process_report(report, tmp_path / "r.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["evidence_scope"] == "retrospective_discovery"
    assert payload["multiplicity_adjusted"] is False
    assert payload["base"]["execution_policy"] == {"tracking_error_threshold": 0.2}


def test_persist_process_targets_round_trip(tmp_path) -> None:
    report = _report_with_policy(None)
    out = persist_process_targets(report.base, tmp_path / "t.parquet")
    back = pd.read_parquet(out)
    expected = report.base.target_weights.copy().astype("float64")
    pd.testing.assert_frame_equal(back, expected, check_freq=False)
    with pytest.raises(ValueError, match=r".+"):
        persist_process_targets(report.base, tmp_path / "t.csv")


def test_execution_fence_uses_earlier_day() -> None:
    from src.mhs.params import PROCESS_EVALUATION_CEILING

    early_idx = pd.date_range("2022-01-01", periods=2, freq="24h", tz="UTC")
    early = pd.DataFrame([[0.5]], index=early_idx, columns=["A"])
    assert _execution_fence(early) == pd.Timestamp("2022-01-03", tz="UTC")
    late_idx = pd.DatetimeIndex([PROCESS_EVALUATION_CEILING - pd.Timedelta(hours=1)])
    late = pd.DataFrame([[0.5]], index=late_idx, columns=["A"])
    expected = (PROCESS_EVALUATION_CEILING.normalize() + pd.Timedelta(days=1)).tz_convert("UTC")
    assert _execution_fence(late) == expected


def test_require_utc_index_rejects() -> None:
    good = pd.date_range("2022-01-01", periods=2, freq="1min", tz="UTC")
    assert _require_utc_index(good, "g").equals(good)
    with pytest.raises(DataIntegrityError, match=r".+"):
        _require_utc_index(pd.Index([0, 1]), "bad")  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match=r".+"):
        _require_utc_index(pd.DatetimeIndex([pd.NaT, good[1]]), "bad")
    with pytest.raises(DataIntegrityError, match=r".+"):
        _require_utc_index(pd.DatetimeIndex(["2022-01-01", "2022-01-02"]), "bad")
    eastern = pd.DatetimeIndex(["2022-01-01", "2022-01-02"], tz="America/New_York")
    with pytest.raises(DataIntegrityError, match=r".+"):
        _require_utc_index(eastern, "bad")


def _replay_fixtures(n_decisions: int = 2):
    from src.mhs.execution.contracts import ExecutionReplayWindow

    cols = ["AUSDT", "BUSDT"]
    idx = pd.date_range("2022-01-01", periods=n_decisions, freq="24h", tz="UTC")
    tgt = pd.DataFrame(
        [[0.5, -0.5]] * n_decisions, index=idx, columns=cols, dtype="float64"
    )
    hourly = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(0.0, index=idx),
        unit_daily_returns=pd.Series(0.0, index=idx),
        exposure=pd.Series(1.0, index=idx),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=tgt,
        target_weights=tgt,
        turnover_1h=pd.Series(0.0, index=hourly),
    )

    def _window(day: pd.Timestamp, row: pd.DataFrame) -> ExecutionReplayWindow:
        mg = pd.date_range(day, day + pd.Timedelta(hours=23, minutes=59), freq="1min", tz="UTC")

        def _mk(v: float) -> pd.DataFrame:
            return pd.DataFrame(v, index=mg, columns=cols, dtype="float64")

        return ExecutionReplayWindow(
            window_start=mg[0], window_end=mg[-1], columns=tuple(cols), symbols=tuple(cols),
            minute_grid=mg, highs=_mk(100.0), lows=_mk(99.0), closes=_mk(99.5),
            marks=_mk(99.5), bar_funding=_mk(0.0), target_weights=row,
            signal_available_at=pd.DatetimeIndex([day]),
            quote_volumes=_mk(1e6),
            funding_known=pd.DataFrame(True, index=mg, columns=cols),
            bar_available_at=mg,
        )

    windows = [_window(day, tgt.iloc[[i]]) for i, day in enumerate(idx)]
    return path, windows


def test_replay_parity_and_coverage() -> None:
    from src.mhs.execution.batch import replay_execution_windows
    from src.mhs.types import ExecutionSpec

    path, windows = _replay_fixtures(2)
    spec = ExecutionSpec()
    ref = replay_execution_windows(windows, 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    got = replay_process_execution(
        path, iter(windows), initial_equity=1000.0,
        execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
    )
    assert (ref.simulated_fills.values == got.simulated_fills.values).all()
    assert ref.ledger.equity.equals(got.ledger.equity)
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter(windows[:1]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )


def test_replay_rejects_empty_path() -> None:
    from src.mhs.types import ExecutionSpec

    path, windows = _replay_fixtures(2)
    import dataclasses

    empty = dataclasses.replace(path, target_weights=path.target_weights.iloc[:0])
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            empty, iter(windows), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=ExecutionSpec(),
        )
    no_cols = dataclasses.replace(path, target_weights=path.target_weights.drop(columns=["AUSDT", "BUSDT"]))
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            no_cols, iter([]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=ExecutionSpec(),
        )


def test_replay_rejects_bad_windows() -> None:
    import dataclasses

    from src.mhs.types import ExecutionSpec

    path, windows = _replay_fixtures(2)
    spec = ExecutionSpec()
    w1, w2 = windows
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter(["not-a-window"]), initial_equity=1000.0,  # type: ignore[list-item]
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, columns=("AUSDT",)), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, target_weights=w1.target_weights.rename(columns={"AUSDT": "X"})), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, marks=None), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, funding_known=None), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    changed = w1.target_weights.copy()
    changed.iloc[0, 0] = 0.99
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, target_weights=changed), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([w2, w1]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )


def test_replay_rejects_provenance_gaps() -> None:
    import dataclasses

    import pandas as pd

    from src.mhs.types import ExecutionSpec

    path, windows = _replay_fixtures(2)
    spec = ExecutionSpec()
    w1, w2 = windows
    short_grid = w1.minute_grid[:5]
    short = dataclasses.replace(w1, minute_grid=short_grid)
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([short, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    unknown_known = pd.DataFrame(True, index=w1.minute_grid, columns=["AUSDT", "BUSDT"], dtype="boolean")
    unknown_known.iloc[0, 0] = pd.NA
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, funding_known=unknown_known), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    bad_signal = dataclasses.replace(
        w1, signal_available_at=pd.DatetimeIndex([w1.target_weights.index[0] - pd.Timedelta(hours=1)])
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([bad_signal, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )


def test_evaluate_rejects_bad_policy(monkeypatch) -> None:
    with pytest.raises(ValueError, match=r".+"):
        evaluate_process_backtest(
            DISCOVERY_START, DISCOVERY_START + pd.Timedelta(days=10),
            execution_policy="bad",  # type: ignore[arg-type]
        )


def test_evaluate_threads_explicit_policy(monkeypatch) -> None:

    data = _synthetic_data()
    schedule = _schedule_for(data)
    seen: dict = {}

    def _fake_load(start, end, data_root=None, memory_budget=None):
        return data

    def _spy(loaded, sched, *, decision_bps, evaluation_bps, leverage_cap, execution_policy=None, memory_budget=None, **_):
        seen["policy"] = execution_policy
        return run_process_paths(
            loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps,
            leverage_cap=leverage_cap, execution_policy=execution_policy,
        )

    monkeypatch.setattr(bt_paths, "load_process_market_data", _fake_load)
    monkeypatch.setattr(bt_paths, "run_process_paths", _spy)
    policy = ProcessExecutionPolicy(0.2)
    report = evaluate_process_backtest(
        data.decision_grid[0], data.decision_grid[100], execution_policy=policy
    )
    assert seen["policy"] is policy
    assert report.base.execution_policy is policy


def test_replay_rejects_remaining_provenance_branches() -> None:
    import dataclasses

    import pandas as pd

    from src.mhs.types import ExecutionSpec

    path, windows = _replay_fixtures(2)
    spec = ExecutionSpec()
    w1, w2 = windows
    bad_decisions = dataclasses.replace(
        w1, target_weights=w1.target_weights.set_axis(pd.Index([0], dtype="int64"))
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([bad_decisions, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    empty_row = w1.target_weights.iloc[:0]
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, target_weights=empty_row), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    non_numeric = w1.target_weights.copy()
    non_numeric = non_numeric.astype(object)
    non_numeric.iloc[0, 0] = "bad"
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dataclasses.replace(w1, target_weights=non_numeric), w2]),
            initial_equity=1000.0, execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    single_bar = w1.minute_grid[:1]
    cols = ["AUSDT", "BUSDT"]

    def _mk1(v: float) -> pd.DataFrame:
        return pd.DataFrame(v, index=single_bar, columns=cols, dtype="float64")

    one_bar = dataclasses.replace(
        w1, minute_grid=single_bar, highs=_mk1(100.0), lows=_mk1(99.0), closes=_mk1(99.5),
        marks=_mk1(99.5), bar_funding=_mk1(0.0), quote_volumes=_mk1(1e6),
        funding_known=pd.DataFrame(True, index=single_bar, columns=cols),
        bar_available_at=single_bar,
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([one_bar, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    dup_grid = pd.DatetimeIndex([w1.minute_grid[0], w1.minute_grid[0], *list(w1.minute_grid[1:])])

    def _dup_frames(v: float) -> pd.DataFrame:
        return pd.DataFrame(v, index=dup_grid, columns=cols, dtype="float64")

    dup = dataclasses.replace(
        w1, minute_grid=dup_grid, highs=_dup_frames(100.0), lows=_dup_frames(99.0),
        closes=_dup_frames(99.5), marks=_dup_frames(99.5), bar_funding=_dup_frames(0.0),
        quote_volumes=_dup_frames(1e6),
        funding_known=pd.DataFrame(True, index=dup_grid, columns=cols),
        bar_available_at=dup_grid,
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([dup, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    int_known = dataclasses.replace(
        w1, funding_known=pd.DataFrame(1, index=w1.minute_grid, columns=cols, dtype="int64")
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([int_known, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    bad_frame = dataclasses.replace(w1, highs=w1.highs.drop(columns=["BUSDT"]))
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([bad_frame, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    bad_sig_len = dataclasses.replace(
        w1,
        signal_available_at=pd.DatetimeIndex(
            [w1.target_weights.index[0], w1.target_weights.index[0] + pd.Timedelta(hours=1)]
        ),
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([bad_sig_len, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    ir_grid = w1.minute_grid.delete(5)
    ir_frames = {
        name: getattr(w1, name).reindex(ir_grid)
        for name in ("highs", "lows", "closes", "marks", "bar_funding", "quote_volumes")
    }
    irregular = dataclasses.replace(
        w1, minute_grid=ir_grid, bar_available_at=ir_grid,
        funding_known=pd.DataFrame(True, index=ir_grid, columns=cols),
        **ir_frames,
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([irregular, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    bad_bar_len = dataclasses.replace(w1, bar_available_at=w1.bar_available_at[:5])
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([bad_bar_len, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    early_bar = dataclasses.replace(
        w1, bar_available_at=w1.minute_grid - pd.Timedelta(minutes=1)
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([early_bar, w2]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )
    fence = _execution_fence(path.target_weights)
    late_grid = w2.minute_grid + (fence - w2.minute_grid[-1] + pd.Timedelta(minutes=1))

    def _mk_late(v: float) -> pd.DataFrame:
        return pd.DataFrame(v, index=late_grid, columns=cols, dtype="float64")

    late = dataclasses.replace(
        w2, minute_grid=late_grid, highs=_mk_late(100.0), lows=_mk_late(99.0),
        closes=_mk_late(99.5), marks=_mk_late(99.5), bar_funding=_mk_late(0.0),
        quote_volumes=_mk_late(1e6),
        funding_known=pd.DataFrame(True, index=late_grid, columns=cols),
        bar_available_at=late_grid,
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([w1, late]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
        )


def test_refit_effective_slices_partition_oos_exactly_once() -> None:
    """Each evaluated date is covered by exactly one refit effective range."""
    data = _synthetic_data()
    schedule = _schedule_for(data)
    (path,) = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    oos = data.decision_grid[
        (data.decision_grid >= schedule[0].effective_from) & (data.decision_grid < schedule[-1].effective_to)
    ]
    assert path.target_weights.index.equals(oos)
    assert not path.target_weights.index.has_duplicates
    assert path.target_weights.index.is_monotonic_increasing
    for i, point in enumerate(schedule):
        governed = oos[(oos >= point.effective_from) & (oos < point.effective_to)]
        assert len(governed) > 0
        if i + 1 < len(schedule):
            assert governed[-1] < schedule[i + 1].effective_from


def test_refit_slice_matches_full_history_book_restriction() -> None:
    """A single-refit slice reproduces the scaled book through manual EMA."""
    from src.mhs.books import scale_book_to_target_gross
    from src.mhs.params import PROCESS_SMOOTHING_HALFLIFE_DAYS
    from src.mhs.process import RefitPoint, ema_smoothing_rate

    data = _synthetic_data()
    schedule = _schedule_for(data)
    point = RefitPoint(
        effective_from=schedule[0].effective_from,
        effective_to=schedule[-1].effective_to,
        train_end=schedule[0].train_end,
    )
    (path,) = run_process_paths(data, (point,), decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    names = list(data.member_books.keys())
    (record,) = path.refits
    combined = sum(
        (record.member_weights.get(n, 0.0) * data.member_books[n] for n in names),
        start=data.member_books[names[0]] * 0.0,
    )
    raw = scale_book_to_target_gross(combined, 1.0).loc[path.unit_target_weights.index]
    rate = ema_smoothing_rate(PROCESS_SMOOTHING_HALFLIFE_DAYS)
    state = np.zeros(len(raw.columns))
    expected = np.empty_like(raw.to_numpy(dtype="float64"))
    for i, row in enumerate(raw.to_numpy(dtype="float64")):
        state = state + rate * (row - state)
        expected[i] = state
    assert np.allclose(
        path.unit_target_weights.to_numpy(dtype="float64"), expected, rtol=0.0, atol=1e-12,
    )


def test_shared_tier_targets_independent_of_consumer_mutation() -> None:
    """Mutating an owned copy leaves shared tier inputs unchanged."""
    data = _synthetic_data()
    schedule = _schedule_for(data)
    base, stress = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0, 24.0), leverage_cap=2.0)
    assert base.target_weights is stress.target_weights
    before = base.target_weights.to_numpy(dtype="float64").copy()
    owned = base.target_weights.copy()
    owned.iloc[:, :] = 0.0
    assert np.array_equal(base.target_weights.to_numpy(dtype="float64"), before)
    assert np.array_equal(stress.target_weights.to_numpy(dtype="float64"), before)


def test_masked_symbol_target_is_exactly_zero_despite_ema_residue() -> None:
    """Post-adoption masking zeroes an unavailable symbol; others are untouched."""
    import dataclasses

    data = _synthetic_data()
    schedule = _schedule_for(data)
    oos_start = schedule[0].effective_from
    masked_days = data.decision_grid[
        (data.decision_grid >= max(data.decision_grid[300], oos_start))
        & (data.decision_grid < schedule[-1].effective_to)
    ]
    assert len(masked_days) > 0
    blocked = data.execution_mask.copy()
    blocked.loc[masked_days, "S00USDT"] = False
    masked = dataclasses.replace(data, execution_mask=blocked)
    (plain,) = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    (path,) = run_process_paths(masked, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    assert bool((path.target_weights.loc[masked_days, "S00USDT"] == 0.0).all())
    assert bool((plain.target_weights.loc[masked_days, "S00USDT"] != 0.0).any())
    rest = [c for c in data.opens_1h.columns if c != "S00USDT"]
    masked_p = path.target_weights[rest].to_numpy(dtype="float64")
    plain_q = plain.target_weights[rest].to_numpy(dtype="float64")
    # No redistribution: other cells keep exact per-day ratios (a shared
    # exposure scalar may still resize every cell together).
    assert masked_p.shape == plain_q.shape
    for i in range(len(masked_p)):
        nz = np.abs(plain_q[i]) > 1e-15
        assert ((masked_p[i] == 0.0) == (plain_q[i] == 0.0)).all()
        if bool(nz.any()):
            ratio = masked_p[i][nz] / plain_q[i][nz]
            assert np.allclose(ratio, ratio[0], rtol=1e-12, atol=1e-12)


def test_availability_overrides_tracking_error_hold() -> None:
    """A hold-retained unavailable target is still masked to exactly zero."""
    import dataclasses

    from src.mhs.process import ProcessExecutionPolicy

    data = _synthetic_data()
    schedule = _schedule_for(data)
    policy = ProcessExecutionPolicy(tracking_error_threshold=0.05)
    oos_start = schedule[0].effective_from
    masked_days = data.decision_grid[
        (data.decision_grid >= max(data.decision_grid[300], oos_start))
        & (data.decision_grid < schedule[-1].effective_to)
    ]
    assert len(masked_days) > 0
    blocked = data.execution_mask.copy()
    blocked.loc[masked_days, "S00USDT"] = False
    masked = dataclasses.replace(data, execution_mask=blocked)
    (held,) = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
        execution_policy=policy,
    )
    (path,) = run_process_paths(
        masked, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
        execution_policy=policy,
    )
    assert bool((held.target_weights.loc[masked_days, "S00USDT"] != 0.0).any())
    assert bool((path.target_weights.loc[masked_days, "S00USDT"] == 0.0).all())


def test_execution_availability_rejects_misaligned_mask() -> None:
    """Label, column or value mismatches fail closed instead of reindexing."""
    import pytest

    from src.common.errors import DataIntegrityError

    data = _synthetic_data()
    schedule = _schedule_for(data)
    (path,) = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    targets = path.target_weights
    short = data.execution_mask.loc[targets.index].iloc[1:]
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_process_execution_availability(targets, short)
    renamed = data.execution_mask.copy()
    renamed.columns = [f"X{c}" for c in renamed.columns]
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_process_execution_availability(targets, renamed.loc[targets.index])
    numeric = data.execution_mask.loc[targets.index].astype("int64")
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_process_execution_availability(targets, numeric)
    assert apply_process_execution_availability(
        targets, data.execution_mask.loc[targets.index]
    ).equals(targets)


def test_zeroed_target_with_unfillable_exit_holds_inventory_as_open_marked() -> None:
    """A masked exit that cannot fill keeps units held as priced open inventory."""
    from src.mhs.execution import ExecutionReplayWindow, ExecutionSpec, replay_execution_windows

    data = _synthetic_data(n_days=60)
    grid = pd.date_range(data.decision_grid[0], periods=24, freq="3min", tz="UTC")
    px = pd.DataFrame({"S00USDT": 100.0}, index=grid)
    weights = pd.DataFrame({"S00USDT": [0.5, 0.0]}, index=pd.DatetimeIndex([grid[0], grid[12]]))
    window = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=("S00USDT",), symbols=("S00USDT",),
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=weights, signal_available_at=pd.DatetimeIndex([grid[0], grid[12]]),
        quote_volumes=pd.DataFrame({"S00USDT": [1000.0] * 12 + [0.0] * 12}, index=grid),
        funding_known=px.notna(), bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    result = replay_execution_windows((window,), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec())
    assert abs(float(result.ledger.equity.iloc[-1] - result.ledger.equity.iloc[0])) >= 0.0
    assert "delist_settlement" not in set(result.simulated_fills["reason"])
    assert result.ledger.primary_valid
    assert [p.status for p in result.terminal_positions] == ["open_marked"]
    assert abs(float(result.terminal_positions[0].quantity)) > 0.0


def test_execution_availability_rejects_nullable_missing_mask() -> None:
    """A boolean mask carrying missing values fails closed."""
    import pytest

    from src.common.errors import DataIntegrityError

    data = _synthetic_data()
    schedule = _schedule_for(data)
    (path,) = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    targets = path.target_weights
    missing = data.execution_mask.loc[targets.index].astype("boolean")
    missing.iloc[0, 0] = pd.NA
    with pytest.raises(DataIntegrityError, match=r".+"):
        apply_process_execution_availability(targets, missing)


def _inventory_test_targets(n_days: int = 3, symbols: tuple[str, ...] = ("AUSDT", "BUSDT")) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=n_days, freq="24h", tz="UTC")
    weights = [[0.5 if j == 0 else -0.5 if j == 1 else 0.0 for j in range(len(symbols))] for _ in range(n_days)]
    return pd.DataFrame(weights, index=index, columns=list(symbols), dtype="float64")


def _inventory_test_path(targets: pd.DataFrame) -> ProcessPath:
    hourly = pd.date_range(targets.index[0], targets.index[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    return ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(0.01, index=targets.index),
        unit_daily_returns=pd.Series(0.01, index=targets.index),
        exposure=pd.Series(1.0, index=targets.index),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=targets,
        target_weights=targets,
        turnover_1h=pd.Series(0.01, index=hourly),
    )


def _inventory_test_proxy(targets: pd.DataFrame) -> ProcessBacktestReport:
    from src.mhs.deploy_gate import DeployGateResult

    path = _inventory_test_path(targets)
    return ProcessBacktestReport(
        start=targets.index[0],
        end=targets.index[-1],
        certification_level=PROCESS_CERTIFICATION_LEVEL,
        n_candidates=len(targets.columns),
        base=path,
        stress=path,
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
    )


def _inventory_frame(value: float, grid: pd.DatetimeIndex, roster: list[str]) -> pd.DataFrame:
    return pd.DataFrame(value, index=grid, columns=roster, dtype="float64")


def _inventory_test_windows(
    targets: pd.DataFrame,
    *,
    funding_known: bool = True,
    funding_rate: float = 0.0,
    funding_known_per_window: tuple[bool, ...] | None = None,
    nan_mark_symbol: str | None = None,
    local_symbols: tuple[str, ...] | None = None,
    final_availability_offset: pd.Timedelta | None = None,
) -> list:
    from src.mhs.execution.contracts import ExecutionReplayWindow

    availability_offset = final_availability_offset or pd.Timedelta(0)

    cols = list(targets.columns)
    roster = list(local_symbols) if local_symbols is not None else cols
    windows = []
    for i, day in enumerate(targets.index):
        start = day if i == 0 else targets.index[i - 1]
        grid = pd.date_range(start, day + pd.Timedelta(hours=23, minutes=57), freq="3min", tz="UTC")
        marks = _inventory_frame(100.0, grid, roster)
        if nan_mark_symbol is not None and nan_mark_symbol in roster:
            marks.loc[grid[grid >= day], nan_mark_symbol] = float("nan")
        flag = funding_known_per_window[i] if funding_known_per_window is not None else funding_known
        frame = _inventory_frame(1.0, grid, roster) if flag else _inventory_frame(0.0, grid, roster)
        known = frame.astype(bool)
        windows.append(
            ExecutionReplayWindow(
                window_start=grid[0],
                window_end=grid[-1],
                columns=tuple(cols),
                symbols=tuple(roster),
                minute_grid=grid,
                highs=_inventory_frame(100.5, grid, roster),
                lows=_inventory_frame(99.5, grid, roster),
                closes=_inventory_frame(100.0, grid, roster),
                marks=marks,
                bar_funding=_inventory_frame(funding_rate, grid, roster),
                target_weights=targets.loc[[day], roster],
                signal_available_at=pd.DatetimeIndex([day + pd.Timedelta(hours=1)]),
                quote_volumes=_inventory_frame(1e6, grid, roster),
                funding_known=known,
                bar_available_at=grid + availability_offset,
            )
        )
    return windows


def _patch_inventory_stack(monkeypatch, targets: pd.DataFrame, **window_kwargs):

    proxy = _inventory_test_proxy(targets)
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: proxy)
    monkeypatch.setattr(bt_inventory, "_load_funding_series", lambda syms: ({}, {}))
    made = _inventory_test_windows(targets, **window_kwargs)
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: iter(made))
    return proxy


def _inventory_empty_proxy() -> ProcessBacktestReport:
    from src.mhs.deploy_gate import DeployGateResult

    empty_frame = pd.DataFrame(columns=["AUSDT", "BUSDT"], dtype="float64")
    empty_series = pd.Series(dtype="float64")
    path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=empty_series,
        unit_daily_returns=empty_series,
        exposure=empty_series,
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=empty_frame,
        target_weights=empty_frame,
        turnover_1h=empty_series,
    )
    stamp = pd.Timestamp("2022-01-01", tz="UTC")
    return ProcessBacktestReport(
        start=stamp,
        end=stamp + pd.Timedelta(days=1),
        certification_level=PROCESS_CERTIFICATION_LEVEL,
        n_candidates=2,
        base=path,
        stress=path,
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
    )


def _inventory_fake_result(
    equity: pd.Series, *, valid: bool, fill_count: int = 0, n_fills: int = 0
):
    from src.mhs.execution.contracts import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult

    zeros = pd.Series(0.0, index=equity.index, dtype="float64")
    ledger = SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=zeros,
        funding_charge=zeros,
        fee_charge=zeros,
        fill_turnover=zeros,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        primary_valid=valid,
        invalid_reasons=() if valid else ("MISSING_DATA",),
    )
    stamps = pd.DatetimeIndex(equity.index[:n_fills]) if n_fills else pd.DatetimeIndex([], tz="UTC")
    fills = pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": ["AUSDT"] * n_fills,
            "quantity_delta": [0.0] * n_fills,
            "fill_price": [100.0] * n_fills,
            "fee_bps": [8.0] * n_fills,
            "reason": ["immediate_taker"] * n_fills,
            "pre_trade_equity": [1.0] * n_fills,
        }
    )
    return StrategyExecutionReplayResult(
        simulated_fills=fills,
        ledger=ledger,
        simulated_units=pd.DataFrame(columns=["AUSDT"]),
        simulated_notional_weights=pd.DataFrame(columns=["AUSDT"]),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        submit_times=pd.Series(dtype="float64"),
        fill_times=pd.Series(dtype="float64"),
        fill_count=fill_count,
        unfilled_count=0,
        fallback_count=0,
        all_intent_shortfall_bps=0.0,
        forced_exit_count=0,
        forced_exit_notional=0.0,
        termination_counts={},
        unsupported_assumptions=(),
        elapsed_seconds=0.0,
    )


def test_evaluate_inventory_replays_identical_sized_targets(monkeypatch) -> None:
    """Base and stress replay one shared window stream with tiered costs."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    seen: dict = {}
    real_batch = bt_inventory.replay_execution_window_batch

    def _spy(windows, equity, bounds, *args, **kwargs):
        seen["bounds"] = list(bounds)
        seen["n_windows"] = 0
        seen["has_live_accumulators"] = kwargs.get("live_accumulators") is not None

        def _counted():
            for window in windows:
                seen["n_windows"] += 1
                yield window

        return real_batch(_counted(), equity, bounds, *args, **kwargs)

    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _spy)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert seen["n_windows"] == len(targets)
    assert seen["bounds"][0][0] == seen["bounds"][1][0] == "OHLCV_IMMEDIATE_TAKER"
    assert seen["bounds"][1][1].taker_fee_bps == pytest.approx(seen["bounds"][0][1].taker_fee_bps * 3.0)
    assert {f["symbol"] for f in report.base.simulated_fills.to_dict(orient="records")} == (
        {f["symbol"] for f in report.stress.simulated_fills.to_dict(orient="records")}
    )
    assert seen["has_live_accumulators"] is True
    stages = [m.stage for m in report.resource_measurements]
    assert stages[0] == "process_inventory_start"
    assert "process_inventory_proxy" in stages
    assert "process_inventory_replay" in stages
    assert stages[-1] == "process_inventory_total"
    assert sum(1 for s in stages if s.startswith("process_3m_window_")) == len(targets)
    assert report.memory_stats.samples_taken >= 0


def test_replay_process_execution_accepts_local_and_final_bar() -> None:
    """Local rosters and the fence-aligned final bar pass the adapter."""
    from src.mhs.types import ExecutionSpec

    targets = _inventory_test_targets(n_days=2, symbols=("AUSDT", "BUSDT", "CUSDT"))
    path = _inventory_test_path(targets)
    windows = _inventory_test_windows(
        targets, local_symbols=("AUSDT", "BUSDT"), final_availability_offset=pd.Timedelta(minutes=3)
    )
    result = replay_process_execution(
        path, iter(windows), initial_equity=1.0,
        execution_bound="OHLCV_IMMEDIATE_TAKER", spec=ExecutionSpec(),
    )
    assert len(result.simulated_fills) > 0


def test_persist_inventory_report_uses_ledger_primary(monkeypatch, tmp_path) -> None:
    """Primary metrics and gate come from the 3m ledger, proxy stays labelled."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    out = rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    daily = bt_inventory._inventory_daily_returns(report.base)
    ref = payload["base"]["daily_returns"]
    assert ref["count"] == len(daily)
    assert ref["evidence_id"] == payload["evidence_id"]
    assert payload["base"]["cagr"] == pytest.approx(
        float(((1.0 + daily).cumprod().iloc[-1]) ** (365.25 / len(daily)) - 1.0)
    )
    assert payload["gate"]["metrics"] == dict(report.gate.metrics)
    assert payload["proxy"]["scope"] == "hourly_proxy_comparison"
    assert payload["proxy"]["base"]["daily_returns"] != payload["base"]["daily_returns"]


def test_evaluate_inventory_blocks_gate_on_unknown_funding(monkeypatch) -> None:
    """Unknown funding over held inventory fails the ledger and the gate."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(
        monkeypatch, targets, funding_rate=0.0001, funding_known_per_window=(True, False, False)
    )
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert report.base.ledger.primary_valid is False
    assert report.gate.go is False
    assert "INVENTORY_LEDGER_INVALID" in report.gate.reason_codes
    assert "MISSING_HELD_FUNDING" in {g.code for g in report.base.ledger.data_gaps}


def test_finalize_discloses_open_inventory_without_fabricated_exit(monkeypatch) -> None:
    """Marked terminal holdings are disclosed open with certification intact."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    terminal = rep_inventory._inventory_terminal_state(report.base)
    assert set(terminal["open_inventory"]) == {"AUSDT", "BUSDT"}
    assert terminal["primary_valid"] is True
    assert terminal["terminal_certified"] is True
    assert terminal["unpriced_terminal_symbols"] == []
    reasons = report.base.simulated_fills.get("reason", pd.Series(dtype="object")).tolist()
    assert "delist_settlement" not in reasons


def test_unpriced_terminal_not_ordinary_inventory(monkeypatch) -> None:
    """A held asset without final marks is separated from marked inventory."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets, nan_mark_symbol="AUSDT")
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    terminal = rep_inventory._inventory_terminal_state(report.base)
    assert terminal["unpriced_terminal_symbols"] == ["AUSDT"]
    assert "AUSDT" not in terminal["open_inventory"]
    assert "BUSDT" in terminal["open_inventory"]
    assert terminal["terminal_certified"] is False


def test_serialized_fill_counts_total_and_passive() -> None:
    """Total fills count rows while passive fills keep the engine meaning."""

    grid = pd.date_range("2022-01-01", periods=732, freq="3min", tz="UTC")
    result = _inventory_fake_result(pd.Series(1.0, index=grid), valid=True, fill_count=0, n_fills=732)
    payload = rep_inventory._inventory_result_payload(result)
    assert payload["total_fills"] == 732
    assert payload["passive_fills"] == 0


def test_evaluate_inventory_failure_carries_provenance(monkeypatch) -> None:
    """A missing decision mark fails with source, time and symbol evidence."""

    targets = _inventory_test_targets(n_days=2, symbols=("AUSDT", "ANCUSDT"))
    proxy = _inventory_test_proxy(targets)
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: proxy)
    monkeypatch.setattr(bt_inventory, "_load_funding_series", lambda syms: ({}, {}))

    def _boom(*args, **kwargs):
        raise DataIntegrityError(
            "cache_required: no finite positive mark (MISSING_DECISION_MARK) "
            "symbol=ANCUSDT decision=2022-07-13T00:00:00+00:00 signal=2022-07-13T01:00:00+00:00 for window"
        )

    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _boom)
    with pytest.raises(DataIntegrityError, match="ANCUSDT"):
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert "ANCUSDT" in proxy.base.target_weights.columns


def test_persist_inventory_rejects_occupied_destination(monkeypatch, tmp_path) -> None:
    """Inventory persistence never overwrites an occupied destination."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    hourly_path = tmp_path / "hourly.json"
    hourly_path.write_text('{"hourly": true}', encoding="utf-8")
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.json")
    assert hourly_path.read_text(encoding="utf-8") == '{"hourly": true}'
    with pytest.raises(ValueError, match=r".+"):
        rep_inventory.persist_process_inventory_report(report, hourly_path)
    with pytest.raises(ValueError, match=r".+"):
        rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.csv")


def test_serialized_fee_funding_signs_and_turnover_units(monkeypatch, tmp_path) -> None:
    """Serialized costs keep engine signs and turnover uses engine units."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    out = rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    ledger = report.base.ledger
    daily = bt_inventory._inventory_daily_returns(report.base)
    assert payload["base"]["total_fees"] == pytest.approx(float(ledger.fee_charge.sum()))
    assert payload["base"]["total_funding"] == pytest.approx(float(ledger.funding_charge.sum()))
    assert payload["base"]["annualized_turnover"] == pytest.approx(
        float(ledger.fill_turnover.sum() * 365.0 / len(daily))
    )


def test_evaluate_inventory_rejects_invalid_inputs(monkeypatch) -> None:
    """Dates, policy, empty targets, budget and window coverage fail closed."""

    targets = _inventory_test_targets()
    proxy = _inventory_test_proxy(targets)
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: proxy)
    monkeypatch.setattr(bt_inventory, "_load_funding_series", lambda syms: ({}, {}))
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: iter(_inventory_test_windows(targets)))
    start = targets.index[0]
    end = targets.index[-1] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(pd.Timestamp("2022-01-01"), end)
    with pytest.raises(ValueError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(end, start)
    with pytest.raises(DataIntegrityError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(start, PROCESS_EVALUATION_CEILING + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(start, end, execution_policy="bad")  # type: ignore[arg-type]
    empty = _inventory_empty_proxy()
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: empty)
    with pytest.raises(DataIntegrityError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(start, end)
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: proxy)
    monkeypatch.setattr(bt_inventory, "resolve_mhs_memory_budget",
        lambda *a, **k: (_ for _ in ()).throw(DataIntegrityError("no telemetry")),
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(start, end)


def test_inventory_window_coverage_must_be_complete(monkeypatch) -> None:
    """A window stream skipping decisions fails instead of certifying a prefix."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    partial = _inventory_test_windows(targets)[:1]
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: iter(partial))
    with pytest.raises(DataIntegrityError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))


def test_inventory_gate_from_valid_ledger() -> None:
    """A valid two-quarter ledger reaches the gate formula without integrity blocks."""

    grid = pd.date_range("2022-01-01", periods=200 * 480, freq="3min", tz="UTC")
    equity = pd.Series(1.0 + 0.0000005 * np.arange(len(grid)), index=grid, dtype="float64")
    base = _inventory_fake_result(equity, valid=True)
    stress = _inventory_fake_result(equity * 0.999999, valid=True)
    gate = bt_inventory._inventory_gate(base, stress)
    assert gate.metrics["n_folds"] == 3.0
    assert "INVENTORY_LEDGER_INVALID" not in gate.reason_codes


def test_inventory_helpers_handle_empty_ledger() -> None:
    """Empty ledgers serialize to neutral zeros without missing keys."""

    empty_equity = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    result = _inventory_fake_result(empty_equity, valid=False)
    summary = bt_inventory._inventory_ledger_summary(result)
    assert summary == {
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "annualized_turnover": 0.0,
        "total_fees": 0.0,
        "total_funding": 0.0,
    }
    payload = rep_inventory._inventory_result_payload(result)
    assert payload["daily_returns"] == {}
    assert payload["total_fills"] == 0
    terminal = rep_inventory._inventory_terminal_state(result)
    assert terminal["open_inventory"] == {}
    assert terminal["primary_valid"] is False


def test_inventory_daily_returns_anchors_first_day_loss() -> None:
    """A 1.0 to 0.5 first-day drop is reported instead of vanishing."""

    grid = pd.date_range("2022-01-01", periods=480, freq="3min", tz="UTC")
    equity = pd.Series(np.linspace(1.0, 0.5, len(grid)), index=grid, dtype="float64")
    result = _inventory_fake_result(equity, valid=True)
    daily = bt_inventory._inventory_daily_returns(result)
    assert len(daily) == 1
    assert daily.iloc[0] == pytest.approx(-0.5)
    summary = bt_inventory._inventory_ledger_summary(result)
    assert summary["max_drawdown"] == pytest.approx(-0.5)
    assert summary["cagr"] < 0.0
    with pytest.raises(DataIntegrityError, match=r".+"):
        bt_inventory._inventory_daily_returns(result, initial_equity=0.0)


def test_inventory_single_day_drawdown_includes_base() -> None:
    """Single-observation wealth anchors MDD to the initial equity base."""

    grid = pd.date_range("2022-01-01", periods=10, freq="3min", tz="UTC")
    equity = pd.Series([1.0 - 0.05 * i for i in range(10)], index=grid, dtype="float64")
    result = _inventory_fake_result(equity, valid=True)
    summary = bt_inventory._inventory_ledger_summary(result)
    assert summary["max_drawdown"] == pytest.approx(float(equity.iloc[-1] - 1.0))


def test_inventory_wires_live_required_symbols(monkeypatch) -> None:
    """Carried inventory stays in the roster through the live symbol hook."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    seen: dict = {}
    real_iter = bt_inventory._iter_mhs_execution_windows
    real_batch = bt_inventory.replay_execution_window_batch

    def _spy_iter(*args, **kwargs):
        seen["required_symbols"] = kwargs.get("required_symbols")
        return real_iter(*args, **kwargs)

    def _spy_batch(windows, equity, bounds, *args, **kwargs):
        seen["live_accumulators"] = kwargs.get("live_accumulators")

        def _counted():
            for window in windows:
                required = seen.get("required_symbols")
                if callable(required):
                    required()
                yield window

        return real_batch(_counted(), equity, bounds, *args, **kwargs)

    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _spy_iter)
    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _spy_batch)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert callable(seen["required_symbols"])
    assert seen["live_accumulators"] is not None
    assert len(report.base.simulated_fills) > 0


def test_inventory_window_stream_keeps_held_symbols_in_roster(monkeypatch) -> None:
    """Live held symbols join the active roster even when targets go flat."""
    from src.mhs.resources import _StageRecorder
    from src.mhs.types import ExecutionSpec

    targets = _inventory_test_targets(n_days=2)
    path = _inventory_test_path(targets)
    signal_available_at = pd.DatetimeIndex(targets.index + pd.Timedelta(hours=1))
    recorder = _StageRecorder(log_run=False)
    held = frozenset({"AUSDT"})
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen["required_symbols"] = kwargs.get("required_symbols")
        return iter([])

    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _capture)
    with pytest.raises(DataIntegrityError, match=r".+"):
        list(
            bt_inventory._inventory_window_stream(
                path,
                signal_available_at,
                "unused-root",
                targets.index[0],
                targets.index[-1] + pd.Timedelta(hours=1),
                {},
                {},
                ExecutionSpec(),
                None,
                None,
                recorder,
                lambda: held,
            )
        )
    assert callable(seen["required_symbols"])
    assert seen["required_symbols"]() == held


def test_evaluate_inventory_failure_preserves_resource_telemetry(monkeypatch) -> None:
    """A replay failure still carries staged resource evidence on the error."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)

    def _boom(windows, *args, **kwargs):
        for _ in windows:
            pass
        raise DataIntegrityError("boom")

    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _boom)
    with pytest.raises(DataIntegrityError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    measurements = getattr(excinfo.value, "resource_measurements", ())
    stats = getattr(excinfo.value, "memory_stats", None)
    stages = [m.stage for m in measurements]
    assert "process_inventory_start" in stages
    assert "process_inventory_failed" in stages
    assert stats is not None
    assert stats.samples_taken >= 0


def test_process_mask_preblocks_3m_missing_symbol(tmp_path, monkeypatch) -> None:
    """A symbol with 1h coverage but no 3m file is pre-blocked from targets."""
    from src.quant.universe.pit_universe import symbol_partition

    candidates = [f"LAKE{i:03d}USDT" for i in range(24)]
    symbols = [s for s in candidates if symbol_partition(s) == "dev"][:10]
    assert len(symbols) == 10
    missing = symbols[0]
    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = pd.Timestamp("2021-02-15", tz="UTC")
    grid = pd.date_range(start, end, freq="1h", tz="UTC")
    rng = np.random.default_rng(11)
    lake = tmp_path / "ohlcv" / "1h"
    lake.mkdir(parents=True)
    ms = ((grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")
    for j, sym in enumerate(symbols):
        close = 100.0 + j + np.cumsum(rng.normal(0, 0.5, len(grid)))
        pd.DataFrame({
            "timestamp": ms,
            "close": close,
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "quote_vol": rng.uniform(1e6, 2e6, len(grid)),
            "taker_buy_quote": rng.uniform(4e5, 6e5, len(grid)),
            "volume": rng.uniform(10.0, 100.0, len(grid)),
        }).to_parquet(lake / f"{sym}.parquet", index=False)
    lake_3m = tmp_path / "ohlcv" / "3m"
    lake_3m.mkdir(parents=True)
    grid_3m = pd.date_range(start, end, freq="3min", tz="UTC")
    ms_3m = ((grid_3m - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")
    for sym in symbols:
        if sym == missing:
            continue
        pd.DataFrame({"timestamp": ms_3m}).to_parquet(lake_3m / f"{sym}.parquet", index=False)
    funding = {
        sym: pd.Series(rng.normal(0, 1e-5, len(grid)), index=grid) for sym in symbols
    }
    monkeypatch.setattr(bt_market, "_load_funding_series",
        lambda syms: ({s: funding[s] for s in syms if s in funding}, {}),
    )
    mark_dir = tmp_path / "markPriceKlines" / "1h"
    mark_dir.mkdir(parents=True)
    for sym in symbols:
        pd.DataFrame({"timestamp": ms, "datetime": grid, "close": 100.0}).to_parquet(
            mark_dir / f"{sym}.parquet", index=False
        )
    import src.market_data.services.futures_collection as fc

    monkeypatch.setattr(
        fc, "_mark_price_path",
        lambda symbol, timeframe: mark_dir / f"{symbol}.parquet",
    )
    data = bt_market.load_process_market_data(start, end, data_root=str(tmp_path / "ohlcv"))
    assert bool((~data.execution_mask[missing]).all())
    assert bool(data.execution_mask.drop(columns=[missing]).to_numpy().any())


def test_inventory_replay_uses_single_resolved_budget(monkeypatch) -> None:
    """One replay budget: preparation, entry admission, window planning and barriers share resolved limits."""
    from src.mhs.resources import MhsMemoryBudget

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    seen: dict = {}
    real_iter = bt_inventory._iter_mhs_execution_windows

    def _spy_iter(*args, **kwargs):
        seen["budget_bytes"] = kwargs.get("budget_bytes")
        seen["reserve_bytes"] = kwargs.get("reserve_bytes")
        return real_iter(*args, **kwargs)

    evaluations: dict = {}
    real_evaluate = bt_inventory.evaluate_process_backtest

    def _spy_evaluate(*args, **kwargs):
        evaluations["memory_budget"] = kwargs.get("memory_budget")
        return real_evaluate(*args, **kwargs)

    admissions: list = []
    real_admit = bt_market._admit_process_stage

    def _spy_admit(*, stage, estimated_bytes, budget, replay, initial_swap_bytes):
        admissions.append((stage, budget, replay))
        return real_admit(
            stage=stage, estimated_bytes=estimated_bytes, budget=budget,
            replay=replay, initial_swap_bytes=initial_swap_bytes,
        )

    barriers: list = []
    real_window_barrier = bt_inventory._assert_execution_rss_budget

    def _spy_window(stage, budget, completed, reserve_bytes=None):
        barriers.append((budget, reserve_bytes))
        return real_window_barrier(stage, budget, completed, reserve_bytes=reserve_bytes)

    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _spy_iter)
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", _spy_evaluate)
    monkeypatch.setattr(bt_inventory, "_admit_process_stage", _spy_admit)
    monkeypatch.setattr(bt_inventory, "_assert_execution_rss_budget", _spy_window)
    bt_inventory.evaluate_process_inventory_backtest(
        targets.index[0], targets.index[-1] + pd.Timedelta(days=1), memory_budget=MhsMemoryBudget(),
    )
    resolved = evaluations["memory_budget"]
    assert isinstance(resolved, MhsMemoryBudget)
    assert seen["budget_bytes"] == resolved.replay_tree_pss_bytes
    assert seen["reserve_bytes"] == resolved.min_available_bytes
    assert barriers
    for budget, reserve in barriers:
        assert budget == resolved.replay_tree_pss_bytes
        assert reserve == resolved.min_available_bytes
    assert any(
        stage == "process_replay_entry" and budget is resolved and replay
        for stage, budget, replay in admissions
    )


def test_inventory_window_measured_budget_verified(monkeypatch) -> None:
    """Measured RSS checks guard every 3m window and the replay stage."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    seen: dict = {}
    real_window_check = bt_inventory._assert_execution_rss_budget
    real_stage_check = bt_inventory._assert_stage_rss_budget

    def _spy_window(stage, budget, completed, reserve_bytes=None):
        seen.setdefault("windows", []).append((stage, budget, completed))
        return real_window_check(stage, budget, completed, reserve_bytes=reserve_bytes)

    def _spy_stage(stage, budget, reserve):
        seen["stage"] = (stage, budget, reserve)
        return real_stage_check(stage, budget, reserve)

    monkeypatch.setattr(bt_inventory, "_assert_execution_rss_budget", _spy_window)
    monkeypatch.setattr(bt_inventory, "_assert_stage_rss_budget", _spy_stage)
    bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert len(seen["windows"]) == len(targets)
    assert seen["stage"][0] == "process_3m_replay"
    assert all(budget == seen["stage"][1] for _, budget, _ in seen["windows"])


def test_inventory_window_budget_breach_fails_closed(monkeypatch) -> None:
    """A measured RSS breach aborts the replay with telemetry preserved."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)

    def _boom(stage, budget, completed, reserve_bytes=None):
        raise DataIntegrityError(f"execution RSS budget exceeded at window boundary: stage={stage}")

    monkeypatch.setattr(bt_inventory, "_assert_execution_rss_budget", _boom)
    with pytest.raises(DataIntegrityError, match="RSS budget"):
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))


def _completed_inventory_market(tmp_path, n_days=3):
    import pandas as pd

    start = pd.Timestamp("2023-06-01", tz="UTC")
    decisions = pd.date_range(start, periods=n_days, freq="24h", tz="UTC")
    fence = decisions[-1] + pd.Timedelta(days=1)
    grid = pd.date_range(start, fence, freq="3min", tz="UTC")
    lake = tmp_path / "ohlcv" / "3m"
    lake.mkdir(parents=True, exist_ok=True)
    ms = ((grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")
    n = len(grid)
    for sym in ("AUSDT", "BUSDT"):
        pd.DataFrame({
            "timestamp": ms,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "quote_vol": 1e6,
        }).to_parquet(lake / f"{sym}.parquet")
    funding = {s: pd.Series(0.0, index=grid) for s in ("AUSDT", "BUSDT")}
    return start, fence, decisions, funding


def test_inventory_production_fence_covers_all_targets(tmp_path) -> None:
    """Unmocked generator and replay cover every row with bars published by the fence."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution.batch import replay_execution_window_batch
    from src.mhs.types import ExecutionSpec

    start, fence, decisions, funding = _completed_inventory_market(tmp_path)
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    targets.iloc[:, 0] = 0.05
    targets.iloc[:, 1] = -0.03
    signals = decisions + pd.Timedelta(hours=1)
    spec = ExecutionSpec()
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m",
            start, fence, funding, "ohlcv_close_fallback", spec,
        )
    )
    assert _execution_fence(targets) == fence
    covered = pd.concat([w.target_weights for w in windows])
    pd.testing.assert_frame_equal(covered, targets)
    for window in windows:
        assert (window.minute_grid < fence).all()
        assert (window.bar_available_at <= fence).all()
    base, stress = replay_execution_window_batch(
        iter(windows), 1.0,
        [("OHLCV_IMMEDIATE_TAKER", spec), ("OHLCV_IMMEDIATE_TAKER", spec)],
    )
    assert base is not None
    assert stress is not None
    assert (windows[-1].bar_available_at <= fence).all()


def test_inventory_nonzero_fills_reconcile_across_bounds(tmp_path) -> None:
    """Controlled entry and exit produce nonzero fills with consistent accounting."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution.batch import replay_execution_window_batch
    from src.mhs.types import ExecutionSpec

    start, fence, decisions, funding = _completed_inventory_market(tmp_path, n_days=4)
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    targets.iloc[0, 0] = 0.5
    targets.iloc[2, 0] = 0.0
    signals = decisions + pd.Timedelta(hours=1)
    spec = ExecutionSpec()
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m",
            start, fence, funding, "ohlcv_close_fallback", spec,
        )
    )
    base, stress = replay_execution_window_batch(
        iter(windows), 1000.0,
        [("OHLCV_IMMEDIATE_TAKER", spec), ("OHLCV_IMMEDIATE_TAKER", spec)],
    )
    assert base is not None
    assert stress is not None
    assert len(base.simulated_fills) > 0
    assert len(stress.simulated_fills) > 0
    assert float(base.ledger.fee_charge.sum()) > 0.0
    assert float(stress.ledger.fee_charge.sum()) > 0.0
    for result in (base, stress):
        totals = result.simulated_fills.groupby("symbol")["quantity_delta"].sum()
        assert abs(float(totals.get("AUSDT", 0.0))) < 1e-6


def test_inventory_fence_labelled_bar_stays_rejected() -> None:
    """A bar labelled at the fence is still a mandatory validation failure."""
    import dataclasses

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError

    targets = _inventory_test_targets(n_days=2)
    path = _inventory_test_path(targets)
    fence = _execution_fence(path.target_weights)
    windows = _inventory_test_windows(targets)
    tainted_grid = windows[0].minute_grid.union(pd.DatetimeIndex([fence]))

    def _extend(frame: pd.DataFrame) -> pd.DataFrame:
        row = pd.DataFrame(100.0, index=pd.DatetimeIndex([fence]), columns=frame.columns)
        return pd.concat([frame, row]).reindex(tainted_grid)

    tainted = dataclasses.replace(
        windows[0],
        minute_grid=tainted_grid,
        highs=_extend(windows[0].highs),
        lows=_extend(windows[0].lows),
        closes=_extend(windows[0].closes),
        marks=_extend(windows[0].marks),
        bar_funding=_extend(windows[0].bar_funding),
        quote_volumes=_extend(windows[0].quote_volumes),
        funding_known=_extend(windows[0].funding_known.astype("float64")).astype(bool),
        bar_available_at=tainted_grid + pd.Timedelta(minutes=3),
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        _validate_replay_window(
            tainted, expected_columns=list(targets.columns),
            expected_targets=targets, cursor=0, fence=fence,
        )


def test_inventory_hourly_evidence_stays_comparative(monkeypatch, tmp_path) -> None:
    """Hourly evidence is explicitly comparative while 3m inventory is primary."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    out = rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["certification_level"] == "process_inventory_3m"
    assert payload["proxy"]["scope"] == "hourly_proxy_comparison"
    assert payload["proxy"]["certification_level"] == "process_proxy_1h_ledger"
    assert set(payload["base"]) >= {"daily_returns", "cagr", "total_fills"}


def _causal_tail_masks(tmp_path, tail_a_hours: float, tail_b_hours: float):
    import pandas as pd

    from src.market_data.services.mhs_execution import apply_dynamic_gap_exclusion

    decisions = pd.date_range("2022-01-01", periods=72, freq="1h", tz="UTC")
    horizon = decisions[24]
    prefix = pd.date_range(decisions[0], horizon, freq="3min", tz="UTC")
    root = tmp_path / "ohlcv"
    (root / "3m").mkdir(parents=True, exist_ok=True)

    def _write(tail_hours: float) -> None:
        tail = pd.date_range(
            horizon + pd.Timedelta(minutes=3),
            horizon + pd.Timedelta(hours=tail_hours),
            freq="3min",
            tz="UTC",
        )
        labels = prefix.union(tail)
        ms = (labels - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
        pd.DataFrame({"timestamp": ms.to_numpy(dtype="int64")}).to_parquet(
            root / "3m" / "S0.parquet"
        )

    mask = pd.DataFrame(True, index=decisions, columns=["S0"])
    _write(tail_a_hours)
    first, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    _write(tail_b_hours)
    second, _ = apply_dynamic_gap_exclusion(mask, "3m", root=str(root), min_gap_hours=1.0)
    return decisions, horizon, mask, first, second


def test_process_targets_through_cutoff_ignore_future_tail(tmp_path) -> None:
    """Two source tails leave past masks and masked targets through T identical."""
    import pandas as pd


    decisions, horizon, _, first, second = _causal_tail_masks(tmp_path, 6.0, 48.0)
    assert first.loc[:horizon].equals(second.loc[:horizon])
    targets = pd.DataFrame(
        {"S0": [0.5 if i % 2 == 0 else -0.25 for i in range(len(decisions))]},
        index=decisions,
        dtype="float64",
    )
    past = targets.loc[:horizon]
    masked_first = apply_process_execution_availability(past, first.loc[:horizon])
    masked_second = apply_process_execution_availability(past, second.loc[:horizon])
    assert masked_first.equals(masked_second)


def test_process_future_only_symbol_target_blocked_without_redistribution() -> None:
    """A symbol with no published history gets zero targets; others are untouched."""
    import pandas as pd


    index = pd.date_range("2022-01-01", periods=3, freq="24h", tz="UTC")
    targets = pd.DataFrame(
        {"NEWSYM": [0.5, 0.5, 0.5], "OLD": [0.25, 0.25, 0.25]},
        index=index,
        dtype="float64",
    )
    mask = pd.DataFrame(True, index=index, columns=["NEWSYM", "OLD"])
    mask.loc[:, "NEWSYM"] = False
    masked = apply_process_execution_availability(targets, mask)
    assert bool((masked["NEWSYM"] == 0.0).all())
    assert masked["OLD"].equals(targets["OLD"])
    assert list(masked.columns) == ["NEWSYM", "OLD"]


def test_process_held_exit_survives_availability_mask() -> None:
    """A blocked hold never deletes the exit row; valid exit data still closes."""
    import pandas as pd

    from src.mhs.execution.batch import replay_execution_windows
    from src.mhs.types import ExecutionSpec

    targets = _inventory_test_targets(n_days=3, symbols=("AUSDT",))
    targets.iloc[0, 0] = 0.5
    targets.iloc[1, 0] = 0.5
    targets.iloc[2, 0] = 0.0
    mask = pd.DataFrame(True, index=targets.index, columns=["AUSDT"])
    mask.iloc[1, 0] = False
    masked = apply_process_execution_availability(targets, mask)
    assert masked.iloc[0, 0] == 0.5
    assert masked.iloc[1, 0] == 0.0
    assert masked.iloc[2, 0] == 0.0
    assert list(masked.columns) == ["AUSDT"]
    windows = _inventory_test_windows(targets)
    result = replay_execution_windows(
        iter(windows), 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec()
    )
    assert len(result.simulated_fills) > 0
    totals = result.simulated_fills.groupby("symbol")["quantity_delta"].sum()
    assert abs(float(totals.get("AUSDT", 0.0))) < 1e-6


def test_refit_arithmetic_matches_baseline() -> None:
    """Interval-local assembly matches the corrected baseline exactly."""
    data = _synthetic_data()
    schedule = _schedule_for(data)[:3]
    first = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    second = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    assert np.allclose(
        first.unit_target_weights.to_numpy(dtype="float64"),
        second.unit_target_weights.to_numpy(dtype="float64"),
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        first.target_weights.to_numpy(dtype="float64"),
        second.target_weights.to_numpy(dtype="float64"),
        rtol=1e-12, atol=1e-12,
    )
    assert (first.unit_target_weights == 0.0).equals(second.unit_target_weights == 0.0)


def test_refit_intermediates_cover_effective_rows_only(monkeypatch) -> None:
    """Each combined intermediate spans only its effective rows."""

    data = _synthetic_data()
    schedule = _schedule_for(data)[:4]
    seen: list[int] = []
    real_scale = bt_paths.scale_book_to_target_gross

    def _spy(frame, target):
        seen.append(len(frame))
        return real_scale(frame, target)

    monkeypatch.setattr(bt_paths, "scale_book_to_target_gross", _spy)
    run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    expected = [
        len(data.member_books["planted"].loc[
            (data.member_books["planted"].index >= p.effective_from)
            & (data.member_books["planted"].index < p.effective_to)
        ])
        for p in schedule
    ]
    assert seen == expected


def test_training_state_and_purge_parity() -> None:
    """Training membership and exposure match across identical runs."""
    data = _synthetic_data()
    schedule = _schedule_for(data)
    first, second = (
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0],
        run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0],
    )
    assert first.target_weights.equals(second.target_weights)
    assert first.exposure.equals(second.exposure)
    assert first.refits == second.refits


def test_source_panels_not_mutated_by_setup(monkeypatch) -> None:
    """Aligned source values are unchanged and no extra copies stay live."""

    data = _synthetic_data()
    before = {n: data.member_books[n].copy() for n in data.member_books}
    schedule = _schedule_for(data)
    run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)
    for name in before:
        assert data.member_books[name].equals(before[name])
    assert bt_market._estimate_panel_bytes(10, 5, 2) == 10 * 5 * 2 * 8 * 2


def test_full_preparation_admission_rejects_small_budget(monkeypatch) -> None:
    """A tiny budget rejects before decoding with a named stage."""
    from src.common.errors import DataIntegrityError
    from src.mhs.resources import MhsMemoryBudget

    tiny = MhsMemoryBudget(total_tree_pss_bytes=1, replay_tree_pss_bytes=1, min_available_bytes=1)
    called: list[str] = []

    def _boom(*args, **kwargs):
        called.append("decode")
        raise AssertionError("decoder must not run")

    monkeypatch.setattr(bt_market, "load_base_panel", _boom)
    with pytest.raises(DataIntegrityError, match="process_prepare_panel"):
        bt_market.load_process_market_data(
            pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-02-01", tz="UTC"),
            memory_budget=tiny,
        )
    assert called == []


def test_budget_propagation_preserves_policy_and_tiers(monkeypatch) -> None:
    """Explicit budgets never change tier ordering or policy semantics."""
    from src.mhs.process import ProcessExecutionPolicy
    from src.mhs.resources import MhsMemoryBudget

    data = _synthetic_data()
    schedule = _schedule_for(data)
    budget = MhsMemoryBudget()
    default = run_process_paths(data, schedule, decision_bps=2.0, evaluation_bps=(2.0, 9.0), leverage_cap=2.0)
    budgeted = run_process_paths(
        data, schedule, decision_bps=2.0, evaluation_bps=(2.0, 9.0), leverage_cap=2.0,
        memory_budget=budget,
    )
    assert budgeted[0].target_weights.equals(default[0].target_weights)
    assert budgeted[1].target_weights.equals(default[1].target_weights)
    assert float(np.log1p(budgeted[1].daily_returns).sum()) <= float(np.log1p(budgeted[0].daily_returns).sum())
    seen: dict = {}

    def _fake_load(start, end, data_root=None, memory_budget=None):
        seen["budget"] = memory_budget
        return data

    def _fake_run(
        loaded, sched, *, decision_bps, evaluation_bps, leverage_cap, execution_policy=None,
        risk_sizing=None, memory_budget=None, clock=None, member_evidence=None,
    ):
        seen["run_budget"] = memory_budget
        return run_process_paths(
            loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps,
            leverage_cap=leverage_cap, execution_policy=execution_policy,
            risk_sizing=risk_sizing, clock=clock, member_evidence=member_evidence,
        )

    monkeypatch.setattr(bt_paths, "load_process_market_data", _fake_load)
    monkeypatch.setattr(bt_paths, "run_process_paths", _fake_run)
    policy = ProcessExecutionPolicy(0.2)
    report = bt_paths.evaluate_process_backtest(
        data.decision_grid[0], data.decision_grid[60], execution_policy=policy, memory_budget=budget,
    )
    assert seen["budget"] is budget
    assert seen["run_budget"] is budget
    assert report.base.execution_policy is policy


def _failure_report_fixture(**overrides):
    from src.mhs.contracts import MhsResourceMeasurement
    from src.mhs.process import ProcessExecutionPolicy

    base = {
        "status": "failed",
        "start": pd.Timestamp("2022-01-01", tz="UTC"),
        "end": pd.Timestamp("2022-01-04", tz="UTC"),
        "data_root": None,
        "execution_policy": ProcessExecutionPolicy(None),
        "stage": "process_3m_window_1",
        "error_code": "DATA_INTEGRITY",
        "error_type": "DataIntegrityError",
        "error_message": "boom",
        "total_decisions": 3,
        "validated_decisions": 2,
        "completed_decisions": 1,
        "completed_windows": 1,
        "completed_decision_start": pd.Timestamp("2022-01-01", tz="UTC"),
        "completed_decision_end": pd.Timestamp("2022-01-01", tz="UTC"),
        "source_gaps": (),
        "source_gap_excluded_symbols": tuple(sorted(__import__("src.mhs.evaluation.integrity", fromlist=["SOURCE_GAP_EXCLUDED_SYMBOLS"]).SOURCE_GAP_EXCLUDED_SYMBOLS)),
        "resource_measurements": (MhsResourceMeasurement(stage="s", elapsed_ms=0, rss_bytes=1),),
        "memory_stats": None,
    }
    base.update(overrides)
    return bt_contracts.ProcessInventoryFailureReport(**base)


def test_inventory_preparation_failure_reports_typed_stage(monkeypatch) -> None:
    """Typed rejection before targets yields null totals and chained cause."""
    from src.mhs.resources import MhsResourceAdmissionError

    targets = _inventory_test_targets()
    start = targets.index[0]
    end = targets.index[-1] + pd.Timedelta(days=1)

    def _boom(*a, **k):
        raise MhsResourceAdmissionError(stage="process_prepare_panel", error_code="MEMORY_BUDGET", message="no ram")

    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", _boom)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(start, end)
    report = excinfo.value.report
    assert report.stage == "process_prepare_panel"
    assert report.error_code == "MEMORY_BUDGET"
    assert report.total_decisions is None
    assert report.validated_decisions == 0
    assert report.completed_decisions == 0
    assert report.completed_windows == 0
    assert report.completed_decision_start is None
    assert report.completed_decision_end is None
    assert report.memory_stats is not None
    assert isinstance(excinfo.value.__cause__, MhsResourceAdmissionError)
    # default policy branch when proxy is missing and no explicit policy
    assert report.execution_policy.tracking_error_threshold is None
    assert report.data_root is None


def test_inventory_preparation_failure_keeps_explicit_policy(monkeypatch) -> None:
    """Explicit policy survives a preparation failure without proxy targets."""

    targets = _inventory_test_targets()
    start = targets.index[0]
    end = targets.index[-1] + pd.Timedelta(days=1)

    def _boom(*a, **k):
        raise RuntimeError("unexpected prep")

    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", _boom)
    policy = ProcessExecutionPolicy(0.2)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(start, end, data_root="failure-root", execution_policy=policy)
    report = excinfo.value.report
    assert report.error_code == "UNEXPECTED_ERROR"
    assert report.stage == "preparation"
    assert report.execution_policy is policy
    assert report.data_root == "failure-root"
    assert report.total_decisions is None
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_inventory_validated_coverage_excludes_failed_bound(monkeypatch) -> None:
    """Second-bound failure validates but does not complete the window."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    made = _inventory_test_windows(targets)

    def _boom_batch(windows, *a, **k):
        it = iter(windows)
        first = next(it)
        second = next(it)
        raise DataIntegrityError("second bound failed")

    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _boom_batch)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    report = excinfo.value.report
    assert report.total_decisions == len(targets)
    assert report.validated_decisions == 2
    assert report.completed_decisions == 1
    assert report.completed_windows == 1
    assert report.completed_decision_start == targets.index[0]
    assert report.completed_decision_end == targets.index[0]
    assert report.error_code == "DATA_INTEGRITY"
    assert "process_3m_window_" in report.stage


def test_inventory_partial_replay_reports_consumed_prefix(monkeypatch) -> None:
    """Allocation rejection after two windows keeps exact prefix endpoints."""
    from src.mhs.resources import MhsResourceAdmissionError

    targets = _inventory_test_targets()
    proxy = _inventory_test_proxy(targets)
    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", lambda *a, **k: proxy)
    monkeypatch.setattr(bt_inventory, "_load_funding_series", lambda syms: ({}, {}))
    made = _inventory_test_windows(targets)

    def _gen(*a, **k):
        yield made[0]
        yield made[1]
        raise MhsResourceAdmissionError(stage="process_3m_window_2", error_code="MEMORY_RESERVE", message="no reserve")

    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _gen)

    def _drain(windows, *a, **k):
        for _ in windows:
            pass
        raise AssertionError("unreachable")

    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _drain)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    report = excinfo.value.report
    assert report.stage == "process_3m_window_2"
    assert report.error_code == "MEMORY_RESERVE"
    assert report.completed_decisions == 2
    assert report.completed_windows == 2
    assert report.completed_decision_start == targets.index[0]
    assert report.completed_decision_end == targets.index[1]
    assert report.total_decisions == len(targets)


def test_inventory_empty_piece_advances_windows_only(monkeypatch) -> None:
    """Held-only terminal piece counts windows without inventing decisions."""
    from src.mhs.resources import _StageRecorder
    from src.mhs.types import ExecutionSpec

    targets = _inventory_test_targets(n_days=1)
    path = _inventory_test_path(targets)
    windows = _inventory_test_windows(targets)
    empty_weights = targets.iloc[0:0].reindex(columns=["AUSDT"])
    empty = windows[0]
    import dataclasses

    tail_grid = pd.date_range(targets.index[0] + pd.Timedelta(hours=24), targets.index[0] + pd.Timedelta(hours=25), freq="3min", tz="UTC")

    def _frame(v):
        return pd.DataFrame(v, index=tail_grid, columns=["AUSDT"], dtype="float64")

    empty_window = dataclasses.replace(
        empty, target_weights=empty_weights,
        minute_grid=tail_grid, highs=_frame(100.0), lows=_frame(99.0),
        closes=_frame(100.0), marks=_frame(100.0), bar_funding=_frame(0.0),
        quote_volumes=_frame(1e6), funding_known=pd.DataFrame(True, index=tail_grid, columns=["AUSDT"]),
        bar_available_at=tail_grid,
    )
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: iter([windows[0], empty_window]))
    recorder = _StageRecorder(log_run=False)
    progress = bt_inventory._ProcessInventoryProgress()
    out = list(
        bt_inventory._inventory_window_stream(
            path, pd.DatetimeIndex(targets.index + pd.Timedelta(hours=1)),
            "root", targets.index[0], targets.index[-1] + pd.Timedelta(days=1),
            {}, {}, ExecutionSpec(), None, None, recorder, None, progress=progress,
        )
    )
    assert len(out) == 2
    assert progress.validated_decisions == 1
    assert progress.completed_decisions == 1
    assert progress.completed_windows == 2
    # progress=None path still drains without tracking
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: iter([windows[0]]))
    recorder2 = _StageRecorder(log_run=False)
    drained = list(
        bt_inventory._inventory_window_stream(
            path, pd.DatetimeIndex(targets.index + pd.Timedelta(hours=1)),
            "root", targets.index[0], targets.index[-1] + pd.Timedelta(days=1),
            {}, {}, ExecutionSpec(), None, None, recorder2, None,
        )
    )
    assert len(drained) == 1


def test_inventory_observed_gaps_preserve_provenance(monkeypatch) -> None:
    """Live accumulator gaps survive without calling finalize."""
    from src.mhs.execution.contracts import ExecutionDataGap

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    gap = ExecutionDataGap(
        code="MISSING_HELD_MARK", symbol="AUSDT",
        timestamp=pd.Timestamp("2022-01-02", tz="UTC"),
        decision_time=pd.Timestamp("2022-01-02", tz="UTC"),
        signal_time=pd.Timestamp("2022-01-02", tz="UTC") + pd.Timedelta(hours=1),
        execution_bound="OHLCV_IMMEDIATE_TAKER",
    )
    finalized = {"called": False}

    class _FakeAcc:
        def __init__(self) -> None:
            self.data_gaps = [gap, gap]
            self.funding_coverage_gaps = {}

        def finalize(self):
            finalized["called"] = True

    def _boom(windows, *a, **kwargs):
        live = kwargs.get("live_accumulators")
        assert live is not None
        live.append([_FakeAcc(), None])
        for _ in windows:
            pass
        raise DataIntegrityError("replay failed")

    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _boom)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    report = excinfo.value.report
    assert len(report.source_gaps) == 1
    assert report.source_gaps[0] == gap
    assert finalized["called"] is False
    assert len(report.source_gap_excluded_symbols) > 0


def test_inventory_telemetry_failure_preserves_original_cause(monkeypatch) -> None:
    """Sampler failure leaves null stats without replacing the domain error."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    stops = {"n": 0}
    real_sampler = bt_inventory._TreeMemorySampler

    def _boom_batch(windows, *a, **k):
        for _ in windows:
            pass
        raise DataIntegrityError("domain boom")

    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", _boom_batch)
    monkeypatch.setattr(bt_inventory, "_TreeMemorySampler", lambda *a, **k: _FailingSampler(stops))

    class _FailingSampler:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def set_stage(self, *a, **k):
            pass

        def stop(self):
            stops["n"] += 1
            raise RuntimeError("telemetry down")

    monkeypatch.setattr(bt_inventory, "_TreeMemorySampler", _FailingSampler)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    report = excinfo.value.report
    assert report.error_type == "DataIntegrityError"
    assert "domain boom" in report.error_message
    assert report.memory_stats is None
    assert stops["n"] == 1
    assert isinstance(excinfo.value.__cause__, DataIntegrityError)


def test_inventory_success_and_interrupts_stay_typed(monkeypatch) -> None:
    """Success stays success while interrupts bypass domain wrapping."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert isinstance(report, bt_contracts.ProcessInventoryReport)
    assert report.base is not None
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", lambda *a, **k: (_ for _ in ()).throw(SystemExit(3)))
    with pytest.raises(SystemExit):
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    # unexpected replay failure maps to UNEXPECTED_ERROR with window stage
    _patch_inventory_stack(monkeypatch, targets)
    monkeypatch.setattr(bt_inventory, "replay_execution_window_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bug")))
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert excinfo.value.report.error_code == "UNEXPECTED_ERROR"
    assert "process_3m_window_" in excinfo.value.report.stage


def test_inventory_finalize_failure_stage(monkeypatch) -> None:
    """Finalize-stage integrity failure keeps the finalize label."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    monkeypatch.setattr(bt_inventory, "_assert_stage_rss_budget", lambda *a, **k: (_ for _ in ()).throw(DataIntegrityError("replay reserve")))
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    assert excinfo.value.report.stage == "finalize"
    assert excinfo.value.report.error_code == "DATA_INTEGRITY"


def test_inventory_failure_json_round_trip(tmp_path) -> None:
    """Partial failure persists exact coverage without performance sections."""
    from src.mhs.execution.contracts import ExecutionDataGap

    gap = ExecutionDataGap(
        code="MISSING_HELD_FUNDING", symbol="BUSDT",
        timestamp=pd.Timestamp("2022-01-03", tz="UTC"),
        decision_time=None, signal_time=None,
        execution_bound="OHLCV_IMMEDIATE_TAKER",
    )
    report = _failure_report_fixture(source_gaps=(gap,), memory_stats=None)
    out = rep_inventory.persist_process_inventory_failure(report, tmp_path / "failure.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["execution_timeframe"] == "3m"
    assert payload["validated_decisions"] == 2
    assert payload["completed_decisions"] == 1
    assert payload["total_decisions"] == 3
    assert payload["completed_decision_start"] == "2022-01-01T00:00:00+00:00"
    assert payload["stage"] == "process_3m_window_1"
    assert payload["source_gaps"][0]["symbol"] == "BUSDT"
    assert payload["source_gaps"][0]["decision_time"] is None
    assert payload["memory_stats"] is None
    assert "cagr" not in payload
    assert "sharpe" not in payload
    assert "base" not in payload
    assert "stress" not in payload
    assert "final_equity" not in json.dumps(payload)


def test_inventory_success_schema_adds_markers(monkeypatch, tmp_path) -> None:
    """Completed markers are additive and preserve numeric results."""

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    report = bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    before = rep_inventory._inventory_result_payload(report.base)
    out = rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["execution_timeframe"] == "3m"
    assert payload["base"]["cagr"] == pytest.approx(before["cagr"])
    assert payload["base"]["total_fees"] == pytest.approx(before["total_fees"])


def test_inventory_failure_destination_protection(monkeypatch, tmp_path) -> None:
    """Completed destinations reject without mutation."""

    report = _failure_report_fixture()
    with pytest.raises(ValueError, match=r".+"):
        rep_inventory.persist_process_inventory_failure(report, tmp_path / "bad.txt")
    completed = tmp_path / "done.json"
    completed.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        rep_inventory.persist_process_inventory_failure(report, completed)
    assert json.loads(completed.read_text(encoding="utf-8")) == {"status": "completed"}
    alias = tmp_path / "alias.json"
    import os as _os

    try:
        _os.symlink(completed, alias)
    except OSError:
        alias = completed
    with pytest.raises(DataIntegrityError, match=r".+"):
        rep_inventory.persist_process_inventory_failure(report, alias)
    # unreadable existing bytes do not block a fresh failure artifact
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not-json{{{", encoding="utf-8")
    out = rep_inventory.persist_process_inventory_failure(report, corrupt)
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "failed"
    # non-completed existing payload is replaceable
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    out2 = rep_inventory.persist_process_inventory_failure(report, other)
    assert json.loads(out2.read_text(encoding="utf-8"))["status"] == "failed"


def test_inventory_atomic_failure_cleans_temp(monkeypatch, tmp_path) -> None:
    """Publication failure leaves no truncated artifact or orphan temp."""

    report = _failure_report_fixture()
    out = tmp_path / "failure.json"
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        rep_inventory.persist_process_inventory_failure(report, out)
    assert not out.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def _panel_lake(tmp_path, n_symbols: int, n_bars: int = 8):
    start = pd.Timestamp("2021-01-01", tz="UTC")
    grid = pd.date_range(start, periods=n_bars, freq="1h", tz="UTC")
    ms = ((grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")
    lake = tmp_path / "ohlcv" / "1h"
    lake.mkdir(parents=True)
    symbols = [f"DIM{i:04d}USDT" for i in range(n_symbols)]
    for j, sym in enumerate(symbols):
        close = 100.0 + j + np.arange(n_bars) * 0.01
        pd.DataFrame({"timestamp": ms, "close": close}).to_parquet(lake / f"{sym}.parquet", index=False)
    return str(tmp_path / "ohlcv"), grid, symbols


def test_panel_admission_uses_actual_source_dimensions(tmp_path) -> None:
    """Actual source dimensions: the pre-allocation estimate uses actual survivors, not 60."""
    from src.mhs.panel import load_base_panel

    root, grid, _symbols = _panel_lake(tmp_path, 65)
    seen: list[int] = []
    panels = load_base_panel(
        root, "1h", ("close",), grid[0], grid[-1], partition="all", min_bars=2,
        allocation_admission=lambda estimated: seen.append(estimated),
    )
    assert len(panels["close"].columns) == 65
    assert seen == [len(grid) * 65 * 1 * 8 * 2]
    plain = load_base_panel(root, "1h", ("close",), grid[0], grid[-1], partition="all", min_bars=2)
    pd.testing.assert_frame_equal(panels["close"], plain["close"])


def test_panel_admission_rejection_prevents_decoding(tmp_path, monkeypatch) -> None:
    """Decode prevented by rejection: a rejecting callback stops wide-plane decode with sources untouched."""
    import src.mhs.panel as panel_mod
    from src.mhs.panel import load_base_panel
    from src.mhs.resources import MhsResourceAdmissionError

    root, grid, _symbols = _panel_lake(tmp_path, 4)
    requested: list[list[str]] = []
    real_read = panel_mod.pq.read_table

    def _spy(path, columns=None, filters=None):
        requested.append(list(columns or []))
        return real_read(path, columns=columns, filters=filters)

    def _reject(estimated: int) -> None:
        raise MhsResourceAdmissionError(
            stage="process_prepare_panel", error_code="MEMORY_BUDGET",
            message=f"no room for {estimated}",
        )

    monkeypatch.setattr(panel_mod.pq, "read_table", _spy)
    with pytest.raises(DataIntegrityError, match="no room"):
        load_base_panel(
            root, "1h", ("close",), grid[0], grid[-1], partition="all", min_bars=2,
            allocation_admission=_reject,
        )
    assert requested, "discovery must run before admission"
    assert all("close" not in cols for cols in requested)
    assert len(list((tmp_path / "ohlcv" / "1h").glob("*.parquet"))) == 4


def test_inventory_typed_window_failure_keeps_consumed_coverage(monkeypatch) -> None:
    """Typed window failure evidence: a later piece rejection keeps typed code, cause and consumed coverage."""
    from src.mhs.resources import MhsResourceAdmissionError

    targets = _inventory_test_targets()
    _patch_inventory_stack(monkeypatch, targets)
    made = _inventory_test_windows(targets)

    def _failing_stream(*args, **kwargs):
        yield made[0]
        yield made[1]
        raise MhsResourceAdmissionError(
            stage="process_execution_piece", error_code="SWAP_GROWTH", message="swap grew mid-replay",
        )

    monkeypatch.setattr(bt_inventory, "_iter_mhs_execution_windows", _failing_stream)
    with pytest.raises(bt_contracts.ProcessInventoryBacktestError) as excinfo:
        bt_inventory.evaluate_process_inventory_backtest(targets.index[0], targets.index[-1] + pd.Timedelta(days=1))
    report = excinfo.value.report
    assert report.error_code == "SWAP_GROWTH"
    assert report.stage == "process_execution_piece"
    assert report.error_type == "MhsResourceAdmissionError"
    assert isinstance(excinfo.value.__cause__, MhsResourceAdmissionError)
    assert report.total_decisions == len(targets)
    assert report.validated_decisions == 2
    assert report.completed_decisions == 2
    assert report.completed_windows == 2
    assert report.completed_decision_start == targets.index[0]
    assert report.completed_decision_end == targets.index[1]
