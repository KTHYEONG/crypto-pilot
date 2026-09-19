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
    funding_known_1h = pd.DataFrame(True, index=grid_1h, columns=symbols)
    return ProcessMarketData(
        grid_1h=grid_1h,
        decision_grid=decision_grid,
        opens_1h=opens_1h,
        bar_funding_1h=bar_funding_1h,
        log_close_step=log_close_step,
        funding_step=funding_step,
        member_books={"planted": planted, "inverse": inverse, "noise": noise},
        execution_mask=execution_mask,
        funding_known_1h=funding_known_1h,
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
        data,
        schedule,
        decision_bps=8.0,
        evaluation_bps=(8.0, 24.0),
        leverage_cap=3.0,
        risk_sizing=spec,
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
            data,
            schedule,
            decision_bps=8.0,
            evaluation_bps=(8.0,),
            leverage_cap=2.0,
            risk_sizing=_candidate_spec(cap=3.0),
        )
    with pytest.raises(ValueError, match=r".+"):
        run_process_paths(
            data,
            schedule,
            decision_bps=8.0,
            evaluation_bps=(8.0,),
            leverage_cap=2.0,
            risk_sizing="not-a-spec",  # type: ignore[arg-type]
        )


def test_sizing_waits_for_history() -> None:
    data = _synthetic_data()
    schedule = _schedule(data)
    (path,) = run_process_paths(
        data,
        schedule,
        decision_bps=8.0,
        evaluation_bps=(8.0,),
        leverage_cap=3.0,
        risk_sizing=_candidate_spec(cap=3.0),
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
        data,
        schedule,
        decision_bps=8.0,
        evaluation_bps=(8.0,),
        leverage_cap=3.0,
        risk_sizing=_candidate_spec(cap=3.0),
    )
    expected = path.unit_target_weights.mul(path.exposure.reindex(path.unit_target_weights.index).fillna(0.0), axis=0)
    pd.testing.assert_frame_equal(path.target_weights, expected)
    source = inspect.getsource(bt_inventory)
    assert "causal_volatility_scaled_exposure" not in source
    assert "volatility_scaled_exposure" not in source


def test_provenance_survives_report_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = _synthetic_data()
    schedule = _schedule(data)
    spec = _candidate_spec(cap=3.0)
    base, stress = run_process_paths(
        data,
        schedule,
        decision_bps=8.0,
        evaluation_bps=(8.0, 24.0),
        leverage_cap=3.0,
        risk_sizing=spec,
    )
    gate = DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0})
    report = ProcessBacktestReport(
        start=data.decision_grid[0],
        end=data.decision_grid[-1],
        certification_level="process_proxy_1h_ledger",
        n_candidates=len(data.member_books),
        base=base,
        stress=stress,
        gate=gate,
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
        data,
        schedule,
        decision_bps=8.0,
        evaluation_bps=(8.0,),
        leverage_cap=2.0,
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
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        risk_sizing=spec,
    )
    assert seen.get("risk_sizing") is spec
    assert report.base.risk_sizing == spec
    assert report.stress.risk_sizing == spec
    pd.testing.assert_frame_equal(report.base.target_weights, report.stress.target_weights)
    pd.testing.assert_series_equal(report.base.exposure, report.stress.exposure)
    with pytest.raises(ValueError, match=r".+"):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
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
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            risk_sizing=spec,
        )
    assert seen.get("risk_sizing") is spec
    with pytest.raises(ValueError, match=r".+"):
        bt_inventory.evaluate_process_inventory_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            risk_sizing="not-a-spec",  # type: ignore[arg-type]
        )


def _baseline_procedure():  # type: ignore[no-untyped-def]
    from src.mhs.backtest.paths import baseline_process_procedure

    return baseline_process_procedure(code_digest="ab" * 32)


def _custom_procedure(member_ids):  # type: ignore[no-untyped-def]
    import pandas as pd

    from src.mhs.backtest.certification import REQUIRED_CHECK_NAMES, ValidationInferenceSpec
    from src.mhs.backtest.journal import PROCEDURE_SCHEMA_VERSION, ProcessProcedureDefinition
    from src.mhs.backtest.labels import ProcessClockSpec
    from src.mhs.backtest.selection import NestedSelectionSpec, TrainingWindowSpec
    from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
    from src.mhs.params import (
        CLI_GROWTH_ENVELOPE_DEFAULT,
        EVIDENCE_GATE_ALPHA,
        GROWTH_RISK_ENVELOPES,
        NULL_BOOTSTRAP_SEED,
        PROCESS_MIN_TRAIN_DAYS,
        PROCESS_SMOOTHING_HALFLIFE_DAYS,
    )
    from src.mhs.process import ProcessExecutionPolicy
    from src.mhs.types import ExecutionSpec

    return ProcessProcedureDefinition(
        schema_version=PROCEDURE_SCHEMA_VERSION,
        code_digest="ab" * 32,
        data_policy=str(MHS_DATA_POLICY_DEFAULT),
        universe_partition="dev",
        member_ids=tuple(member_ids),
        clock=ProcessClockSpec(
            decision_period=pd.Timedelta(hours=24),
            bar_completion_lag=pd.Timedelta(hours=1),
            fit_latency=pd.Timedelta(0),
        ),
        selection=NestedSelectionSpec(
            policies=(
                TrainingWindowSpec(policy_id="expanding", kind="expanding", months=None),
                TrainingWindowSpec(policy_id="rolling_12m", kind="rolling", months=12),
                TrainingWindowSpec(policy_id="rolling_24m", kind="rolling", months=24),
                TrainingWindowSpec(policy_id="equal_member", kind="equal_member", months=None),
            ),
            control_policy_id="equal_member",
            minimum_inner_labels=PROCESS_MIN_TRAIN_DAYS,
            alpha=float(EVIDENCE_GATE_ALPHA),
            bootstrap_paths=500,
            seed=int(NULL_BOOTSTRAP_SEED),
            fit_latency=pd.Timedelta(0),
        ),
        inference=ValidationInferenceSpec(
            family_alpha=float(EVIDENCE_GATE_ALPHA),
            procedure_budget=4,
            look_budget=1,
            endpoint_budget=4,
            bootstrap_paths=6400,
            seed=int(NULL_BOOTSTRAP_SEED),
            resample_batch_paths=500,
            minimum_block_days=1,
        ),
        execution_policy=ProcessExecutionPolicy(),
        risk_sizing=None,
        sizing_source="hourly_proxy",
        member_evidence_source="daily_step_proxy",
        execution_spec=ExecutionSpec(),
        envelope=GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT],
        initial_equity=1.0,
        smoothing_halflife_days=float(PROCESS_SMOOTHING_HALFLIFE_DAYS),
        required_checks=tuple(REQUIRED_CHECK_NAMES),
    )


def test_baseline_procedure_freezes_declared_controls() -> None:
    from src.mhs.backtest.journal import process_procedure_digest
    from src.mhs.backtest.paths import baseline_process_procedure
    from src.common.errors import DataIntegrityError

    first = _baseline_procedure()
    second = _baseline_procedure()
    assert first == second
    assert process_procedure_digest(first) == process_procedure_digest(second)
    assert len(first.member_ids) == 18
    assert first.selection.control_policy_id == "equal_member"
    assert [p.policy_id for p in first.selection.policies] == [
        "expanding",
        "rolling_12m",
        "rolling_24m",
        "equal_member",
    ]
    assert first.selection.minimum_inner_labels == 365
    assert first.inference.procedure_budget == 4
    assert first.inference.look_budget == 1
    assert first.inference.endpoint_budget == 4
    assert first.inference.minimum_block_days == 1
    assert first.inference.bootstrap_paths >= first.inference.required_paths
    assert first.initial_equity == 1.0
    assert len(first.required_checks) == 11
    other = baseline_process_procedure(code_digest="cd" * 32)
    assert process_procedure_digest(other) != process_procedure_digest(first)
    with pytest.raises(DataIntegrityError):
        baseline_process_procedure(code_digest="")


def _stub_cold_control(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths
    from src.mhs.backtest.selection import RefitPolicyChoice

    def _cold(evidence, point, *, spec):  # type: ignore[no-untyped-def]
        return RefitPolicyChoice(
            point,
            "equal_member",
            None,
            None,
            None,
            0,
            None,
            ("INNER_EVIDENCE_INSUFFICIENT",),
        )

    monkeypatch.setattr(bt_paths, "choose_refit_policy", _cold)


def test_procedure_driven_path_uses_fixed_control(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    data = _synthetic_data()
    monkeypatch.setattr(bt_paths, "load_process_market_data", lambda *a, **k: data)
    _stub_cold_control(monkeypatch)
    procedure = _custom_procedure(list(data.member_books.keys()))
    report = bt_paths.evaluate_process_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
    )
    assert report.base.clock_mode == "matured_labels"
    assert len(report.base.policy_choices) == len(report.base.refits)
    assert report.base.policy_choices[0].policy_id == "equal_member"
    assert report.gate.go is False


def test_procedure_driven_proxy_blocks_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    data = _synthetic_data()
    monkeypatch.setattr(bt_paths, "load_process_market_data", lambda *a, **k: data)
    _stub_cold_control(monkeypatch)
    procedure = _custom_procedure(list(data.member_books.keys()))
    report = bt_paths.evaluate_process_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
    )
    assert report.gate.go is False
    assert "PROXY_NEVER_DEPLOYS" in report.gate.reason_codes


def test_native_member_evidence_identity_mismatch_rejected() -> None:
    import dataclasses

    import src.mhs.backtest.paths as bt_paths
    from src.common.errors import DataIntegrityError
    from src.mhs.backtest.journal import process_procedure_digest
    from src.mhs.backtest.labels import ProcessClockSpec
    from src.mhs.backtest.labels import build_proxy_member_returns as _build
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    data = _synthetic_data()
    procedure = _custom_procedure(list(data.member_books.keys()))
    clock = ProcessClockSpec(
        decision_period=pd.Timedelta(hours=24),
        bar_completion_lag=pd.Timedelta(hours=1),
        fit_latency=pd.Timedelta(0),
    )
    good = _build(
        data,
        clock=clock,
        one_way_bps=8.0,
        procedure_digest=process_procedure_digest(procedure),
        input_manifest_digest=None,
    )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure=procedure,
            member_evidence=dataclasses.replace(good, source="inventory_3m"),  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure=procedure,
            member_evidence=dataclasses.replace(good, procedure_digest="00" * 32),
        )
    renamed = good.returns.copy()
    renamed.columns = ["a", "b", "c"]
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure=procedure,
            member_evidence=dataclasses.replace(good, returns=renamed),
        )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            member_evidence=good,
        )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure=procedure,
            member_evidence="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure=procedure,
            execution_policy=__import__("src.mhs.process", fromlist=["ProcessExecutionPolicy"]).ProcessExecutionPolicy(
                tracking_error_threshold=0.5
            ),
        )
    with pytest.raises(DataIntegrityError):
        bt_paths.evaluate_process_backtest(
            DISCOVERY_START,
            PROCESS_EVALUATION_CEILING,
            procedure=procedure,
            risk_sizing=_candidate_spec(cap=3.0),
        )


def test_procedure_driven_same_decision_across_cost_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    data = _synthetic_data()
    monkeypatch.setattr(bt_paths, "load_process_market_data", lambda *a, **k: data)
    _stub_cold_control(monkeypatch)
    procedure = _custom_procedure(list(data.member_books.keys()))
    report = bt_paths.evaluate_process_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
    )
    pd.testing.assert_frame_equal(report.base.target_weights, report.stress.target_weights)
    pd.testing.assert_frame_equal(report.base.unit_target_weights, report.stress.unit_target_weights)
    assert report.base.signal_available_at is not None
    assert report.stress.signal_available_at is not None
    assert list(report.base.signal_available_at) == list(report.stress.signal_available_at)


def test_continuous_path_preserves_calendar_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths
    from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING

    data = _synthetic_data()
    monkeypatch.setattr(bt_paths, "load_process_market_data", lambda *a, **k: data)
    _stub_cold_control(monkeypatch)
    procedure = _custom_procedure(list(data.member_books.keys()))
    report = bt_paths.evaluate_process_backtest(
        DISCOVERY_START,
        PROCESS_EVALUATION_CEILING,
        procedure=procedure,
    )
    idx = report.base.target_weights.index
    assert idx.is_monotonic_increasing
    assert not idx.has_duplicates
    gaps = (idx[1:] - idx[:-1]).total_seconds() / 86400.0
    assert bool((gaps == 1.0).all())
    assert report.base.exposure.index.equals(idx)


def _audited_path():  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessPath
    from src.mhs.backtest.selection import RefitPolicyChoice
    from src.mhs.process import ProcessExecutionPolicy, RefitPoint
    from src.mhs.backtest.contracts import RefitRecord

    idx = pd.date_range("2022-01-01", periods=3, freq="24h", tz="UTC")
    unit = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=idx)
    hourly = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    first = RefitPoint(idx[0], idx[1], idx[0])
    second = RefitPoint(idx[1], idx[2] + pd.Timedelta(days=1), idx[1])
    refits = (
        RefitRecord(
            point=first,
            member_weights={"A": 0.5, "B": 0.5},
            smoothing_halflife_days=8.0,
            policy_id="equal_member",
            train_start=None,
            n_train_labels=0,
        ),
        RefitRecord(
            point=second,
            member_weights={"A": 0.7, "B": 0.3},
            smoothing_halflife_days=8.0,
            policy_id="expanding",
            train_start=idx[0] - pd.Timedelta(days=30),
            n_train_labels=400,
        ),
    )
    choices = (
        RefitPolicyChoice(first, "equal_member", None, None, None, 0, None, ("INNER_EVIDENCE_INSUFFICIENT",)),
        RefitPolicyChoice(
            second,
            "expanding",
            idx[0] - pd.Timedelta(days=30),
            idx[0],
            idx[1],
            120,
            0.02,
            ("PAIRED_LCB_IMPROVEMENT",),
        ),
    )
    return ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series([0.01, 0.02, 0.015], index=idx),
        unit_daily_returns=pd.Series([0.01, 0.02, 0.015], index=idx),
        exposure=pd.Series([1.0, 1.0, 1.0], index=idx),
        refits=refits,
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=unit,
        target_weights=unit,
        turnover_1h=pd.Series(0.01, index=hourly),
        risk_sizing=None,
        signal_available_at=pd.DatetimeIndex(idx + pd.Timedelta(hours=1)),
        clock_mode="matured_labels",
        policy_choices=choices,
    )


def test_proxy_targets_preserved_with_clock_provenance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.mhs.reporting.process import _tier_payload, persist_process_targets

    path = _audited_path()
    back = pd.read_parquet(persist_process_targets(path, tmp_path / "t.parquet"))
    pd.testing.assert_frame_equal(back, path.target_weights.copy().astype("float64"), check_freq=False)
    payload = _tier_payload(path)
    assert payload["signal_available_at"] == [ts.isoformat() for ts in path.signal_available_at]
    assert payload["clock_mode"] == "matured_labels"


def test_proxy_control_and_inactive_audits_preserved() -> None:
    from src.mhs.reporting.process import _tier_payload

    payload = _tier_payload(_audited_path())
    assert len(payload["policy_choices"]) == 2
    assert len(payload["refits"]) == 2
    assert len(payload["refit_audits"]) == 2
    first_choice = payload["policy_choices"][0]
    assert first_choice["policy_id"] == "equal_member"
    assert first_choice["paired_growth_lcb"] is None
    assert first_choice["reason_codes"] == ["INNER_EVIDENCE_INSUFFICIENT"]
    second_choice = payload["policy_choices"][1]
    assert second_choice["policy_id"] == "expanding"
    assert second_choice["paired_growth_lcb"] == 0.02
    assert payload["refit_audits"][0]["n_train_labels"] == 0
    assert payload["refit_audits"][1]["n_train_labels"] == 400


def test_proxy_boundary_labels_comparative(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from src.mhs.backtest.contracts import ProcessBacktestReport
    from src.mhs.reporting.process import persist_process_report

    path = _audited_path()
    report = ProcessBacktestReport(
        start=path.target_weights.index[0],
        end=path.target_weights.index[-1],
        certification_level="process_proxy_1h_ledger",
        n_candidates=2,
        base=path,
        stress=path,
        gate=DeployGateResult(go=False, reason_codes=("PROXY_NEVER_DEPLOYS",), metrics={}),
    )
    payload = json.loads(persist_process_report(report, tmp_path / "r.json").read_text(encoding="utf-8"))
    assert payload["research_diagnostics_only"] is True
    assert payload["deployment_eligible"] is False
    assert payload["gate"]["go"] is False
    assert payload["base"]["clock_mode"] == "matured_labels"
    assert payload["certification_level"] == "process_proxy_1h_ledger"


def test_proxy_frozen_training_sample_disclosure() -> None:
    from src.mhs.reporting.process import _tier_payload

    payload = _tier_payload(_audited_path())
    audits = payload["refit_audits"]
    assert audits[0]["train_start"] is None
    assert audits[1]["train_start"] is not None
    assert audits[0]["n_train_labels"] != audits[1]["n_train_labels"]
    assert audits[0]["train_end"] != audits[1]["train_end"]
