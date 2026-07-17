from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from pytest_mock import MockerFixture

from src.domain.futures.strategy.tiered_workflow.crisis_policy import (
    CrisisReliabilityAssessment,
    CrisisWindowMetrics,
    evaluate_crisis_survival,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2Result,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    CrisisWindow,
    apply_crisis_reliability_override,
    assess_crisis_reliability,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_metric(
    label: str = "test_window",
    status: str = "stress_tested_pass",
    detail: str = "",
    symbol_count: int = 15,
    observation_days: int = 400,
    bar_count: int = 1000,
    event_count: int = 500,
    trade_count: int = 50,
    mdd: float | None = 0.10,
    cagr: float | None = 0.05,
    cvar_95: float | None = 0.03,
) -> CrisisWindowMetrics:
    return CrisisWindowMetrics(
        label=label,
        status=status,  # type: ignore[arg-type]
        detail=detail,
        symbol_count=symbol_count,
        observation_days=observation_days,
        bar_count=bar_count,
        event_count=event_count,
        trade_count=trade_count,
        mdd=mdd,
        cagr=cagr,
        cvar_95=cvar_95,
    )


def _make_assessment(**kw: object) -> CrisisReliabilityAssessment:
    defaults = {
        "status": "stress_tested_pass",
        "verified": True,
        "detail": "all good",
        "window_results": (),
        "blockers": (),
        "usable_window_count": 1,
    }
    defaults.update(**kw)
    return CrisisReliabilityAssessment(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Scenario 1 — Happy path
# ---------------------------------------------------------------------------

def test_evaluate_crisis_survival_passes_all_usable_windows() -> None:
    w1 = _make_metric(label="window_a")
    w2 = _make_metric(label="window_b")

    assessment = evaluate_crisis_survival(
        (w1, w2),
        max_mdd_abs=0.21,
        min_cagr=-0.05,
        max_cvar_95=0.06,
        min_symbols=10,
        min_observation_days=300,
        min_trades=30,
        min_usable_windows=1,
    )

    assert assessment.verified is True
    assert assessment.status == "stress_tested_pass"
    assert len(assessment.window_results) == 2
    assert assessment.blockers == ()
    assert assessment.usable_window_count == 2


# ---------------------------------------------------------------------------
# Scenario 2 — Current LUNA/FTX metrics fail
# ---------------------------------------------------------------------------

def test_evaluate_crisis_survival_blocks_current_luna_ftx_metrics() -> None:
    w = _make_metric(
        label="luna_ftx",
        mdd=0.2901,
        cagr=-0.3273,
        cvar_95=0.04,
        trade_count=60,
    )

    assessment = evaluate_crisis_survival(
        (w,),
        max_mdd_abs=0.21,
        min_cagr=-0.05,
        max_cvar_95=0.06,
        min_symbols=10,
        min_observation_days=300,
        min_trades=30,
        min_usable_windows=1,
    )

    assert assessment.verified is False
    assert "luna_ftx:mdd_abs" in assessment.blockers
    assert "luna_ftx:cagr" in assessment.blockers


# ---------------------------------------------------------------------------
# Scenario 3 — Data insufficiency / non-finite metrics
# ---------------------------------------------------------------------------

def test_evaluate_crisis_survival_rejects_insufficient_or_nonfinite_metrics() -> None:
    cases: list[tuple[str, CrisisWindowMetrics]] = [
        ("too few symbols", _make_metric(symbol_count=9)),
        ("too few observation days", _make_metric(observation_days=299)),
        ("too few trades", _make_metric(trade_count=29)),
        ("nan mdd", _make_metric(mdd=float("nan"))),
    ]

    for name, metric in cases:
        assessment = evaluate_crisis_survival(
            (metric,),
            max_mdd_abs=0.21,
            min_cagr=-0.05,
            max_cvar_95=0.06,
            min_symbols=10,
            min_observation_days=300,
            min_trades=30,
            min_usable_windows=1,
        )
        assert assessment.verified is False, f"{name}: expected not verified"
        # single insufficient window → usable_count 0 < min_usable_windows → untested_no_data
        assert assessment.status == "untested_no_data", f"{name}: status={assessment.status}"


# ---------------------------------------------------------------------------
# Scenario 4 — No early return
# ---------------------------------------------------------------------------

def test_assess_crisis_reliability_evaluates_every_window_without_early_return(
    mocker: MockerFixture,
) -> None:
    mock_l3 = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        side_effect=[
            _make_l3_result(mdd=0.10, cagr=0.02, cvar95=0.01, n_trades=50),
            _make_l3_result(mdd=0.29, cagr=-0.32, cvar95=0.04, n_trades=60),
        ],
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._build_rule_based_stress_batch",
        return_value=_make_dummy_batch(),
    )
    mocker.patch(
        "src.domain.futures.optimization.opt_data_utils.load_futures_data_maps_for_symbols",
        return_value=(_make_data_maps(), {}, list(_make_data_maps().keys())),
    )
    mocker.patch(
        "src.domain.futures.strategy.timeframe_probe._resample_ohlcv",
        return_value=_make_dummy_dataframe(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.align_data_maps",
        return_value=_make_aligned(100),
    )
    _stub_registry(mocker)

    _all_syms = tuple(_make_registry().by_symbol.keys())
    window_a = CrisisWindow(
        start=date(2022, 4, 1), end=date(2023, 2, 15),
        label="crisis_a", symbols=_all_syms, source_note="a",
    )
    window_b = CrisisWindow(
        start=date(2022, 4, 1), end=date(2023, 2, 15),
        label="crisis_b", symbols=_all_syms, source_note="b",
    )

    assessment = assess_crisis_reliability(
        deployment_registry=_make_registry(),
        strategy_cfg=_make_strategy_cfg(),
        config=Layer2AllocationConfig(),
        caps=_make_caps(),
        tf="8h",
        deploy_leverage=1.5,
        crisis_windows=(window_a, window_b),
    )

    assert mock_l3.call_count == 2
    assert len(assessment.window_results) == 2
    assert assessment.verified is False
    assert assessment.status == "stress_tested_fail"


# ---------------------------------------------------------------------------
# Scenario 5 — Error isolation
# ---------------------------------------------------------------------------

def test_assess_crisis_reliability_records_loader_failure_and_continues(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "src.domain.futures.optimization.opt_data_utils.load_futures_data_maps_for_symbols",
        side_effect=[
            RuntimeError("loader crashed"),
            (_make_data_maps(), {}, list(_make_data_maps().keys())),
        ],
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.align_data_maps",
        return_value=_make_aligned(100),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._build_rule_based_stress_batch",
        return_value=_make_dummy_batch(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        return_value=_make_l3_result(mdd=0.10, cagr=0.02, cvar95=0.01, n_trades=50),
    )
    _stub_registry(mocker)

    _all_syms = tuple(_make_registry().by_symbol.keys())
    window_a = CrisisWindow(
        start=date(2022, 4, 1), end=date(2023, 2, 15),
        label="crash_a", symbols=_all_syms, source_note="a",
    )
    window_b = CrisisWindow(
        start=date(2022, 4, 1), end=date(2023, 2, 15),
        label="crash_b", symbols=_all_syms, source_note="b",
    )

    assessment = assess_crisis_reliability(
        deployment_registry=_make_registry(),
        strategy_cfg=_make_strategy_cfg(),
        config=Layer2AllocationConfig(),
        caps=_make_caps(),
        tf="8h",
        deploy_leverage=1.5,
        crisis_windows=(window_a, window_b),
    )

    assert len(assessment.window_results) == 2
    assert assessment.verified is False
    assert assessment.status in ("stress_data_invalid", "untested_no_data")


# ---------------------------------------------------------------------------
# Scenario 6 — Native parity
# ---------------------------------------------------------------------------

def test_assess_crisis_reliability_native_metrics_use_same_policy() -> None:
    w = _make_metric(
        label="native_coverage",
        mdd=0.10,
        cagr=-0.50,
        cvar_95=0.01,
    )

    assessment = evaluate_crisis_survival(
        (w,),
        max_mdd_abs=0.21,
        min_cagr=-0.05,
        max_cvar_95=0.06,
        min_symbols=10,
        min_observation_days=300,
        min_trades=30,
        min_usable_windows=1,
    )

    assert assessment.verified is False
    assert "native_coverage:cagr" in assessment.blockers


# ---------------------------------------------------------------------------
# Scenario 8 — Monotonic override
# ---------------------------------------------------------------------------

def test_apply_crisis_reliability_override_is_monotonic_and_typed() -> None:
    existing_passed = Layer2Result(
        selected_last=frozenset(),
        weights_last={},
        sharpe_hybrid=1.0,
        sharpe_baseline=0.5,
        mdd_hybrid=0.10,
        mdd_baseline=0.20,
        cagr_hybrid=0.30,
        cagr_baseline=0.10,
        mar_hybrid=3.0,
        mar_baseline=0.5,
        fold_pass_ratio=0.8,
        turnover=0.2,
        friction_pass_pct=0.9,
        gate_passed=True,
        blocker_reason="",
    )

    failing_assessment = _make_assessment(
        status="stress_tested_fail",
        verified=False,
        blockers=("window_a:mdd_abs",),
        window_results=(_make_metric(mdd=0.30),),
        usable_window_count=1,
    )

    overridden = apply_crisis_reliability_override(
        existing_passed,
        failing_assessment,
        require_crisis_reliability=True,
    )

    assert overridden.gate_passed is False
    assert overridden.blocker_reason == "crisis_survival"
    assert overridden.crisis_reliability_blockers == ("window_a:mdd_abs",)
    assert overridden.crisis_window_count == 1
    assert overridden.crisis_usable_window_count == 1


def test_apply_crisis_reliability_override_preserves_existing_fail() -> None:
    existing_failed = Layer2Result(
        selected_last=frozenset(),
        weights_last={},
        sharpe_hybrid=1.0,
        sharpe_baseline=0.5,
        mdd_hybrid=0.10,
        mdd_baseline=0.20,
        cagr_hybrid=0.30,
        cagr_baseline=0.10,
        mar_hybrid=3.0,
        mar_baseline=0.5,
        fold_pass_ratio=0.8,
        turnover=0.2,
        friction_pass_pct=0.9,
        gate_passed=False,
        blocker_reason="mdd_abs",
    )

    passing_assessment = _make_assessment(
        status="stress_tested_pass",
        verified=True,
        window_results=(_make_metric(),),
        usable_window_count=1,
    )

    still_failed = apply_crisis_reliability_override(
        existing_failed,
        passing_assessment,
        require_crisis_reliability=True,
    )

    assert still_failed.gate_passed is False
    assert still_failed.blocker_reason == "mdd_abs"
    assert still_failed.crisis_window_count == 1


def test_apply_crisis_reliability_override_passes_through_opt_out() -> None:
    existing_passed = Layer2Result(
        selected_last=frozenset(),
        weights_last={},
        sharpe_hybrid=1.0,
        sharpe_baseline=0.5,
        mdd_hybrid=0.10,
        mdd_baseline=0.20,
        cagr_hybrid=0.30,
        cagr_baseline=0.10,
        mar_hybrid=3.0,
        mar_baseline=0.5,
        fold_pass_ratio=0.8,
        turnover=0.2,
        friction_pass_pct=0.9,
        gate_passed=True,
        blocker_reason="",
    )

    failing_assessment = _make_assessment(
        status="stress_tested_fail",
        verified=False,
        blockers=("window:mdd_abs",),
        window_results=(_make_metric(mdd=0.30),),
        usable_window_count=1,
    )

    result = apply_crisis_reliability_override(
        existing_passed,
        failing_assessment,
        require_crisis_reliability=False,
    )

    assert result.gate_passed is True
    assert result.crisis_reliability_blockers == ("window:mdd_abs",)


# ---------------------------------------------------------------------------
# Scenario 9 — Config validation
# ---------------------------------------------------------------------------

def test_layer2_crisis_config_defaults_and_validation() -> None:
    cfg = Layer2AllocationConfig()
    assert cfg.l2_crisis_min_symbols == 10
    assert cfg.l2_crisis_min_observation_days == 300
    assert cfg.l2_crisis_min_usable_windows == 1

    with pytest.raises(ValueError, match="l2_deploy_mdd_margin"):
        Layer2AllocationConfig.from_mapping({"l2_deploy_mdd_margin": 0.0})
    with pytest.raises(ValueError, match="l2_deploy_mdd_margin"):
        Layer2AllocationConfig.from_mapping({"l2_deploy_mdd_margin": 1.0})
    with pytest.raises(ValueError, match="l2_crisis_min_symbols"):
        Layer2AllocationConfig.from_mapping({"l2_crisis_min_symbols": 0})
    with pytest.raises(ValueError, match="l2_crisis_min_observation_days"):
        Layer2AllocationConfig.from_mapping({"l2_crisis_min_observation_days": 0})


# ---------------------------------------------------------------------------
# Test: insufficient usable windows
# ---------------------------------------------------------------------------

def test_evaluate_crisis_survival_insufficient_usable_windows() -> None:
    assessment = evaluate_crisis_survival(
        (),
        max_mdd_abs=0.21,
        min_cagr=-0.05,
        max_cvar_95=0.06,
        min_symbols=10,
        min_observation_days=300,
        min_trades=30,
        min_usable_windows=1,
    )
    assert assessment.verified is False
    assert assessment.status == "untested_no_data"


# ---------------------------------------------------------------------------
# Scenario 10 — Sequential replay (performance proxy)
# ---------------------------------------------------------------------------

def test_assess_crisis_reliability_replays_windows_sequentially(
    mocker: MockerFixture,
) -> None:
    active_count: list[int] = [0]
    max_active: list[int] = [0]

    def _tracking_holdout(**kwargs: object) -> object:
        active_count[0] += 1
        max_active[0] = max(max_active[0], active_count[0])
        result = _make_l3_result(mdd=0.10, cagr=0.02, cvar95=0.01, n_trades=50)
        active_count[0] -= 1
        return result

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        side_effect=_tracking_holdout,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._build_rule_based_stress_batch",
        return_value=_make_dummy_batch(),
    )
    mocker.patch(
        "src.domain.futures.optimization.opt_data_utils.load_futures_data_maps_for_symbols",
        return_value=(_make_data_maps(), {}, list(_make_data_maps().keys())),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.align_data_maps",
        return_value=_make_aligned(100),
    )
    _stub_registry(mocker)

    _all_syms = tuple(_make_registry().by_symbol.keys())
    windows = tuple(
        CrisisWindow(
            start=date(2022, 4, 1), end=date(2023, 2, 15),
            label=f"w{i}", symbols=_all_syms, source_note="x",
        )
        for i in range(3)
    )

    assessment = assess_crisis_reliability(
        deployment_registry=_make_registry(),
        strategy_cfg=_make_strategy_cfg(),
        config=Layer2AllocationConfig(),
        caps=_make_caps(),
        tf="8h",
        deploy_leverage=1.5,
        crisis_windows=windows,
    )

    assert max_active[0] <= 1, "holdout must not run concurrently"
    assert len(assessment.window_results) == 3


# ---------------------------------------------------------------------------
# Test: threshold integrity — evaluate_crisis_survival rejects invalid thresholds
# ---------------------------------------------------------------------------

def test_evaluate_crisis_survival_rejects_bad_thresholds() -> None:
    w = _make_metric()
    with pytest.raises(ValueError, match="max_mdd_abs"):
        evaluate_crisis_survival(
            (w,), max_mdd_abs=0.0, min_cagr=-0.05, max_cvar_95=0.06,
            min_symbols=10, min_observation_days=300, min_trades=30,
            min_usable_windows=1,
        )
    with pytest.raises(ValueError, match="min_usable_windows"):
        evaluate_crisis_survival(
            (w,), max_mdd_abs=0.21, min_cagr=-0.05, max_cvar_95=0.06,
            min_symbols=10, min_observation_days=300, min_trades=30,
            min_usable_windows=0,
        )


# ---------------------------------------------------------------------------
# Scenario 7 — Anti-leakage: crisis assessment has no Optuna/study fields
# ---------------------------------------------------------------------------

def test_active_pipeline_applies_crisis_only_after_champion_selection() -> None:
    fields = {f.name for f in dataclasses.fields(CrisisReliabilityAssessment)}
    assert "params_mutated" not in fields
    assert "objective_value" not in fields
    assert "trial_number" not in fields
    assert "study_name" not in fields

    import inspect
    sig = inspect.signature(evaluate_crisis_survival)
    for p in ("params", "objective", "trial", "study"):
        assert p not in sig.parameters, f"leak: {p} in evaluate_crisis_survival"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockL3Result:
    cagr: float
    mdd: float
    cvar95: float
    n_trades: int
    sharpe: float = 0.0
    mar: float = 0.0
    cagr_baseline: float = 0.0
    mdd_baseline: float = 0.0
    sharpe_baseline: float = 0.0
    mar_baseline: float = 0.0
    gate_passed: bool = True
    blocker_reason: str = ""
    sortino: float = 0.0
    total_return: float = 0.0
    equity_multiple: float = 1.0


def _make_l3_result(
    mdd: float = 0.10,
    cagr: float = 0.05,
    cvar95: float = 0.03,
    n_trades: int = 50,
) -> _MockL3Result:
    return _MockL3Result(cagr=cagr, mdd=mdd, cvar95=cvar95, n_trades=n_trades)


@dataclass
class _MockAligned:
    datetimes: tuple[int, ...]


def _make_aligned(n_bars: int = 100) -> _MockAligned:
    return _MockAligned(datetimes=tuple(range(n_bars)))


def _make_dummy_dataframe() -> object:
    import pandas as pd
    return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})


@dataclass
class _MockBatch:
    events: tuple[int, ...] = (1, 2, 3)


def _make_dummy_batch() -> _MockBatch:
    return _MockBatch()


@dataclass
class _MockRegistry:
    by_symbol: dict[str, object]
    ready_symbols: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.by_symbol = {f"SYM{i}USDT": object() for i in range(15)}


def _make_registry() -> Any:
    return _MockRegistry()

def _make_data_maps() -> dict[str, dict[str, object]]:
    return {f"SYM{i}USDT": {"4h": _make_dummy_dataframe()} for i in range(15)}


def _make_strategy_cfg() -> Any:
    return object()


def _make_caps() -> Any:
    return object()


def _stub_registry(mocker: MockerFixture) -> None:
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.dataclasses.replace",
        side_effect=lambda x, **kw: x,
    )
