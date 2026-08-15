import dataclasses
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.domain.futures.strategy.candidate_contracts import (
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.crisis_policy import (
    CrisisReliabilityAssessment,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2Result
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    apply_crisis_reliability_override,
    assess_crisis_reliability,
)


def _make_registry(*, symbol: str, strategy_id: str) -> QualifiedSignalRegistry:
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey(symbol=symbol, strategy_id=strategy_id, activation_context="all"),
        mean_gross_bps=50.0,
        mean_incremental_bps=20.0,
        p_value=0.02,
        q_value=0.08,
        positive_fold_ratio=0.8,
        n_obs=200,
        effective_n=150.0,
        n_folds=4,
        quality_weight=0.8,
        lcb_net_bps=15.0,
    )
    return QualifiedSignalRegistry(
        by_symbol={symbol: (evidence,)},
        ready_symbols=(symbol,),
        trade_scope_count=1,
        registry_version="test",
    )


def _make_aligned(n_bars: int = 200) -> AlignedMarketData:
    datetimes = (np.datetime64("2022-04-01T00:00", "h") + np.arange(n_bars).astype("timedelta64[h]") * 8).astype(
        "datetime64[ns]"
    )
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("ARUSDT",),
        open_2d=np.ones((n_bars, 1)),
        high_2d=np.ones((n_bars, 1)),
        low_2d=np.ones((n_bars, 1)),
        close_2d=np.ones((n_bars, 1)),
        volume_2d=np.ones((n_bars, 1)),
        funding_2d=np.zeros((n_bars, 1)),
        active_mask=np.ones(n_bars, dtype=bool),
        warm_mask=np.ones(n_bars, dtype=bool),
        entry_block_mask=np.zeros(n_bars, dtype=bool),
        kill_mask=np.zeros(n_bars, dtype=bool),
    )


def _make_l2_result(**kwargs: Any) -> Layer2Result:
    base = Layer2Result(
        selected_last=frozenset(),
        weights_last={},
        sharpe_hybrid=1.5,
        sharpe_baseline=1.0,
        mdd_hybrid=0.12,
        mdd_baseline=0.18,
        cagr_hybrid=0.40,
        cagr_baseline=0.20,
        mar_hybrid=3.3,
        mar_baseline=1.1,
        fold_pass_ratio=0.75,
        turnover=0.15,
        friction_pass_pct=0.75,
        gate_passed=True,
        blocker_reason="",
    )
    return dataclasses.replace(base, **kwargs)


def test_assess_crisis_reliability_untested_when_no_registry() -> None:
    result = assess_crisis_reliability(
        deployment_registry=None,
        strategy_cfg=MagicMock(),
        config=MagicMock(
            l2_max_mdd_abs=0.30,
            l2_deploy_mdd_margin=0.30,
            l2_min_worst_fold_cagr=-0.05,
            l2_max_cvar_95=0.06,
            l2_crisis_min_symbols=10,
            l2_crisis_min_observation_days=300,
            l2_min_trades=30,
            l2_crisis_min_usable_windows=1,
        ),
        caps=MagicMock(),
        tf="8h",
        deploy_leverage=1.5,
    )

    assert result.status == "untested_no_data"
    assert result.verified is False


def test_assess_crisis_reliability_untested_when_registry_has_no_window_overlap() -> None:
    registry = _make_registry(symbol="NOTINCRISISUSDT", strategy_id="trend_donchian:donchian_72")

    result = assess_crisis_reliability(
        deployment_registry=registry,
        strategy_cfg=MagicMock(),
        config=MagicMock(
            l2_max_mdd_abs=0.30,
            l2_deploy_mdd_margin=0.30,
            l2_min_worst_fold_cagr=-0.05,
            l2_max_cvar_95=0.06,
            l2_crisis_min_symbols=10,
            l2_crisis_min_observation_days=300,
            l2_min_trades=30,
            l2_crisis_min_usable_windows=1,
        ),
        caps=MagicMock(),
        tf="8h",
        deploy_leverage=1.5,
    )

    assert result.status == "untested_no_data"
    assert result.verified is False


def test_apply_crisis_reliability_override_blocks_when_unverified() -> None:
    l2_result = _make_l2_result()
    assessment = CrisisReliabilityAssessment(
        status="stress_tested_fail",
        verified=False,
        detail="luna_ftx_2022_collapse: mdd=0.4000 > 0.30",
        window_results=(),
        blockers=("test:mdd_abs",),
        usable_window_count=0,
    )

    updated = apply_crisis_reliability_override(
        l2_result,
        assessment,
        require_crisis_reliability=True,
    )

    assert updated.gate_passed is False
    assert updated.blocker_reason == "crisis_survival"
    assert updated.crisis_reliability_status == "stress_tested_fail"


def test_apply_crisis_reliability_override_opt_out_preserves_gate() -> None:
    l2_result = _make_l2_result()
    assessment = CrisisReliabilityAssessment(
        status="stress_tested_fail",
        verified=False,
        detail="...",
        window_results=(),
        blockers=("test:mdd_abs",),
        usable_window_count=0,
    )

    updated = apply_crisis_reliability_override(
        l2_result,
        assessment,
        require_crisis_reliability=False,
    )

    assert updated.gate_passed is True
    assert updated.crisis_reliability_status == "stress_tested_fail"
