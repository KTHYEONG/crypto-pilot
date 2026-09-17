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
)
from src.mhs.process import monthly_refit_schedule
from src.mhs.process_backtest import (
    PROCESS_CERTIFICATION_LEVEL,
    ProcessBacktestReport,
    ProcessMarketData,
    build_candidate_member_books,
    evaluate_process_backtest,
    persist_process_report,
    quarter_fold_returns,
    run_process_path,
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
    path = run_process_path(data, schedule, one_way_bps=8.0, leverage_cap=2.0)
    assert path.daily_returns.index[0] == schedule[0].effective_from
    assert bool((path.exposure >= 0).all())
    assert bool((path.exposure <= 2.0 + 1e-12).all())
    first = path.refits[0]
    assert first.member_weights.get("planted", 0.0) > 0
    assert first.member_weights.get("inverse", 0.0) == 0.0


def test_run_process_path_perturbation_invariance() -> None:
    data = _synthetic_data()
    schedule = _schedule_for(data)
    base = run_process_path(data, schedule, one_way_bps=8.0, leverage_cap=2.0)
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
    replay = run_process_path(perturbed, schedule, one_way_bps=8.0, leverage_cap=2.0)
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
    base = run_process_path(data, schedule, one_way_bps=2.0, leverage_cap=2.0)
    stress = run_process_path(data, schedule, one_way_bps=50.0, leverage_cap=2.0)
    assert float(np.log1p(stress.daily_returns).sum()) <= float(np.log1p(base.daily_returns).sum())
    with pytest.raises(ValueError, match=r".+"):
        run_process_path(data, (), one_way_bps=2.0, leverage_cap=2.0)


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

    def _fake_run(loaded, sched, one_way_bps, leverage_cap):
        seen.setdefault("bps", []).append(one_way_bps)
        return run_process_path(loaded, sched, one_way_bps=one_way_bps, leverage_cap=leverage_cap)

    monkeypatch.setattr(pb, "load_process_market_data", _fake_load)
    monkeypatch.setattr(pb, "run_process_path", _fake_run)
    report = evaluate_process_backtest(data_root=None)
    assert report.certification_level == "process_proxy_1h_ledger"
    assert seen["bps"][1] == pytest.approx(seen["bps"][0] * 3.0)
    assert report.n_candidates == len(data.member_books)
    assert report.gate.metrics["n_folds"] == len(quarter_fold_returns(report.base.daily_returns))


def test_persist_process_report_round_trip(tmp_path) -> None:
    data = _synthetic_data()
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    path = run_process_path(data, schedule, one_way_bps=8.0, leverage_cap=2.0)
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
        run_process_path(empty, schedule, one_way_bps=8.0, leverage_cap=2.0)


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
