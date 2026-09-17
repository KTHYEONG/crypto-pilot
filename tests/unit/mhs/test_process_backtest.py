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
from src.mhs.process_backtest import (
    PROCESS_CERTIFICATION_LEVEL,
    PROCESS_POLICY_REPORT_PATH,
    PROCESS_REPORT_PATH,
    ProcessBacktestReport,
    ProcessMarketData,
    ProcessPath,
    _execution_fence,
    _reject_invalid_ledger_returns,
    _require_utc_index,
    _resolve_process_report_path,
    _tier_payload,
    build_candidate_member_books,
    evaluate_process_backtest,
    persist_process_report,
    persist_process_targets,
    quarter_fold_returns,
    replay_process_execution,
    run_process_paths,
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
    return ProcessMarketData(
        grid_1h=grid_1h,
        decision_grid=decision_grid,
        opens_1h=opens_1h,
        bar_funding_1h=bar_funding_1h,
        log_close_step=log_close_step,
        funding_step=funding_step,
        member_books={"planted": planted, "inverse": inverse, "noise": noise},
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
    import src.mhs.process_backtest as pb

    def _boom(*a, **k):
        raise AssertionError("data load must not run")

    monkeypatch.setattr(pb, "load_process_market_data", _boom)
    with pytest.raises(DataIntegrityError, match=r".+"):
        evaluate_process_backtest(
            DISCOVERY_START, PROCESS_EVALUATION_CEILING + pd.Timedelta(seconds=1)
        )


def test_evaluate_uses_gate_and_stress_triple(monkeypatch) -> None:
    import src.mhs.process_backtest as pb

    data = _synthetic_data()
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    seen: dict = {}

    def _fake_load(start, end, data_root=None):
        return data

    def _fake_run(loaded, sched, *, decision_bps, evaluation_bps, leverage_cap):
        seen["decision"] = decision_bps
        seen["evaluation"] = evaluation_bps
        seen.setdefault("bps", []).append(decision_bps)
        return run_process_paths(loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps, leverage_cap=leverage_cap)

    calls: list = []

    def _spy(loaded, sched, *, decision_bps, evaluation_bps, leverage_cap):
        calls.append((decision_bps, evaluation_bps))
        return _fake_run(loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps, leverage_cap=leverage_cap)

    monkeypatch.setattr(pb, "load_process_market_data", _fake_load)
    monkeypatch.setattr(pb, "run_process_paths", _spy)
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
    import src.mhs.process_backtest as pb
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
    funding = {
        sym: pd.Series(rng.normal(0, 1e-5, len(grid)), index=grid) for sym in symbols
    }
    monkeypatch.setattr(
        pb, "_load_funding_series",
        lambda syms: ({s: funding[s] for s in syms if s in funding}, {}),
    )
    data = pb.load_process_market_data(start, end, data_root=str(tmp_path / "ohlcv"))
    expected_keys = list(PROCESS_FEATURE_CANDIDATES) + [
        f"funding_carry_{h}h" for h in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS
    ]
    assert list(data.member_books.keys()) == expected_keys
    assert (data.decision_grid == pd.date_range(start, end, freq="24h", tz="UTC")).all()
    assert set(data.opens_1h.columns) == set(symbols)
    late_row = data.member_books[expected_keys[0]].iloc[-1]
    assert abs(late_row.sum()) < 1e-8


def _tiny_panel() -> dict:
    grid = pd.date_range("2021-01-01", periods=5, freq="1h", tz="UTC")
    cols = ["AUSDT", "BUSDT"]
    frame = pd.DataFrame(100.0, index=grid, columns=cols)
    return {k: frame.copy() for k in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote")}


def test_load_process_market_data_raises_without_funding(monkeypatch) -> None:
    import src.mhs.process_backtest as pb

    monkeypatch.setattr(pb, "load_base_panel", lambda *a, **k: _tiny_panel())
    monkeypatch.setattr(pb, "_load_funding_series", lambda syms: ({}, dict.fromkeys(syms, "missing")))
    with pytest.raises(RuntimeError, match=r".+"):
        pb.load_process_market_data(
            pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-02", tz="UTC")
        )


def test_load_process_market_data_raises_without_aligned_funding(monkeypatch) -> None:
    import src.mhs.process_backtest as pb

    panel = _tiny_panel()
    grid = panel["close"].index
    monkeypatch.setattr(pb, "load_base_panel", lambda *a, **k: panel)
    monkeypatch.setattr(
        pb, "_load_funding_series",
        lambda syms: ({s: pd.Series(0.0, index=grid) for s in syms}, {}),
    )
    monkeypatch.setattr(pb, "bar_funding_panel", lambda *a, **k: pd.DataFrame(index=grid))
    with pytest.raises(RuntimeError, match=r".+"):
        pb.load_process_market_data(
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
    assert _resolve_process_report_path(_report_with_policy(None), None) == PROCESS_REPORT_PATH
    assert _resolve_process_report_path(_report_with_policy(0.0), None) == PROCESS_POLICY_REPORT_PATH
    assert _resolve_process_report_path(_report_with_policy(0.2), None) == PROCESS_POLICY_REPORT_PATH
    out = _resolve_process_report_path(_report_with_policy(None), tmp_path / "custom.json")
    assert out == tmp_path / "custom.json"
    with pytest.raises(ValueError, match=r".+"):
        _resolve_process_report_path(_report_with_policy(None), tmp_path / "bad.txt")
    with pytest.raises(ValueError, match=r".+"):
        _resolve_process_report_path(_report_with_policy(0.2), PROCESS_REPORT_PATH)


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
    import src.mhs.process_backtest as pb

    data = _synthetic_data()
    schedule = _schedule_for(data)
    seen: dict = {}

    def _fake_load(start, end, data_root=None):
        return data

    def _spy(loaded, sched, *, decision_bps, evaluation_bps, leverage_cap, execution_policy=None):
        seen["policy"] = execution_policy
        return run_process_paths(
            loaded, sched, decision_bps=decision_bps, evaluation_bps=evaluation_bps,
            leverage_cap=leverage_cap, execution_policy=execution_policy,
        )

    monkeypatch.setattr(pb, "load_process_market_data", _fake_load)
    monkeypatch.setattr(pb, "run_process_paths", _spy)
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
