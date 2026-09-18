"""Invariant guards for causal risk sizing on process paths."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.mhs.backtest.contracts import ProcessBacktestReport
from src.mhs.backtest.paths import run_process_paths
from src.mhs.deploy_gate import DeployGateResult
from src.mhs.process import ProcessRiskSizingSpec, monthly_refit_schedule
from src.mhs.reporting.process import _tier_payload, persist_process_report


def _synthetic_data(n_days: int = 500, seed: int = 7):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessMarketData

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
    )


def _schedule(data):  # type: ignore[no-untyped-def]
    return monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])


def _candidate_spec(cap: float = 3.0) -> ProcessRiskSizingSpec:
    return ProcessRiskSizingSpec(
        annual_volatility_target=0.25,
        ewma_halflife_days=60,
        minimum_observations=60,
        leverage_cap=cap,
    )


def test_legacy_path_unchanged() -> None:
    data = _synthetic_data()
    schedule = _schedule(data)
    first = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    second = run_process_paths(data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0)[0]
    pd.testing.assert_frame_equal(first.target_weights, second.target_weights)
    pd.testing.assert_series_equal(first.exposure, second.exposure)
    pd.testing.assert_series_equal(first.daily_returns, second.daily_returns)
    assert first.risk_sizing is None


def test_specified_sizing_shared_across_tiers() -> None:
    data = _synthetic_data()
    schedule = _schedule(data)
    spec = _candidate_spec(cap=3.0)
    base, stress = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0, 24.0),
        leverage_cap=3.0, risk_sizing=spec,
    )
    pd.testing.assert_frame_equal(base.target_weights, stress.target_weights)
    pd.testing.assert_series_equal(base.exposure, stress.exposure)
    assert base.risk_sizing == spec
    assert stress.risk_sizing == spec


def test_cap_mismatch_rejected() -> None:
    data = _synthetic_data()
    schedule = _schedule(data)
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(
            data, schedule, decision_bps=8.0, evaluation_bps=(8.0,),
            leverage_cap=2.0, risk_sizing=_candidate_spec(cap=3.0),
        )
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(
            data, schedule, decision_bps=8.0, evaluation_bps=(8.0,),
            leverage_cap=2.0, risk_sizing="not-a-spec",  # type: ignore[arg-type]
        )


def test_sizing_waits_for_history() -> None:
    data = _synthetic_data()
    schedule = _schedule(data)
    (path,) = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,),
        leverage_cap=3.0, risk_sizing=_candidate_spec(cap=3.0),
    )
    assert bool((path.exposure.iloc[:60] == 0.0).all())
    gross = path.target_weights.abs().sum(axis=1)
    assert bool((gross[path.exposure == 0.0] == 0.0).all())


def test_execution_uses_supplied_targets_once() -> None:
    import inspect

    import src.mhs.backtest.inventory as bt_inventory

    data = _synthetic_data()
    schedule = _schedule(data)
    (path,) = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,),
        leverage_cap=3.0, risk_sizing=_candidate_spec(cap=3.0),
    )
    expected = path.unit_target_weights.mul(
        path.exposure.reindex(path.unit_target_weights.index).fillna(0.0), axis=0
    )
    pd.testing.assert_frame_equal(path.target_weights, expected)
    source = inspect.getsource(bt_inventory)
    assert "causal_volatility_scaled_exposure" not in source
    assert "volatility_scaled_exposure" not in source


def test_provenance_survives_report_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = _synthetic_data()
    schedule = _schedule(data)
    spec = _candidate_spec(cap=3.0)
    base, stress = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0, 24.0),
        leverage_cap=3.0, risk_sizing=spec,
    )
    gate = DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0})
    report = ProcessBacktestReport(
        start=data.decision_grid[0], end=data.decision_grid[-1],
        certification_level="process_proxy_1h_ledger",
        n_candidates=len(data.member_books), base=base, stress=stress, gate=gate,
    )
    out = persist_process_report(report, tmp_path / "report.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    for tier in ("base", "stress"):
        sizing = payload[tier]["risk_sizing"]
        assert sizing["annual_volatility_target"] == spec.annual_volatility_target
        assert sizing["ewma_halflife_days"] == spec.ewma_halflife_days
        assert sizing["minimum_observations"] == spec.minimum_observations
        assert sizing["leverage_cap"] == spec.leverage_cap
    legacy = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
    )[0]
    assert _tier_payload(legacy)["risk_sizing"] is None


def test_public_backtest_entry_forwards_risk_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    data = _synthetic_data()
    monkeypatch.setattr(bt_paths, "load_process_market_data", lambda *a, **k: data)
    real_run = bt_paths.run_process_paths
    seen: dict[str, object] = {}

    def _spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(bt_paths, "run_process_paths", _spy)
    spec = _candidate_spec(cap=3.0)
    report = bt_paths.evaluate_process_backtest(
        DISCOVERY_START, PROCESS_EVALUATION_CEILING, risk_sizing=spec,
    )
    assert seen.get("risk_sizing") is spec
    assert report.base.risk_sizing == spec
    assert report.stress.risk_sizing == spec
    pd.testing.assert_frame_equal(report.base.target_weights, report.stress.target_weights)
    pd.testing.assert_series_equal(report.base.exposure, report.stress.exposure)
    with pytest.raises(ValueError, match=r".+"):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START, PROCESS_EVALUATION_CEILING,
            risk_sizing="not-a-spec",  # type: ignore[arg-type]
        )


def test_inventory_entry_forwards_risk_sizing_to_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.inventory as bt_inventory
    from src.mhs.backtest.contracts import ProcessInventoryBacktestError
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    seen: dict[str, object] = {}

    def _stub(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        raise RuntimeError("sentinel-stop-before-replay")

    monkeypatch.setattr(bt_inventory, "evaluate_process_backtest", _stub)
    spec = _candidate_spec(cap=3.0)
    with pytest.raises(ProcessInventoryBacktestError):
        bt_inventory.evaluate_process_inventory_backtest(
            DISCOVERY_START, PROCESS_EVALUATION_CEILING, risk_sizing=spec,
        )
    assert seen.get("risk_sizing") is spec
    with pytest.raises(ValueError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(
            DISCOVERY_START, PROCESS_EVALUATION_CEILING,
            risk_sizing="not-a-spec",  # type: ignore[arg-type]
        )
