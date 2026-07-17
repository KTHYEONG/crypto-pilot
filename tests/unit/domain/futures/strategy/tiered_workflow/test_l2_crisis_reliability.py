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
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2Result
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    CrisisReliabilityAssessment,
    CrisisWindow,
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
    datetimes = (
        np.datetime64("2022-04-01T00:00", "h") + np.arange(n_bars).astype("timedelta64[h]") * 8
    ).astype("datetime64[ns]")
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


def test_assess_crisis_reliability_native_coverage_skips_stress_check() -> None:
    result = assess_crisis_reliability(
        native_covered=True,
        native_detail="fold=0 mdd=0.180 cagr=-0.050",
        deployment_registry=None,
        strategy_cfg=MagicMock(),
        config=MagicMock(l2_max_mdd_abs=0.30),
        caps=MagicMock(),
        tf="8h",
        deploy_leverage=1.5,
    )

    assert result.status == "native_coverage"
    assert result.verified is True
    assert result.detail == "fold=0 mdd=0.180 cagr=-0.050"


def test_assess_crisis_reliability_untested_when_no_registry() -> None:
    result = assess_crisis_reliability(
        native_covered=False,
        native_detail="no_bottleneck_caliber_fold_in_window",
        deployment_registry=None,
        strategy_cfg=MagicMock(),
        config=MagicMock(l2_max_mdd_abs=0.30),
        caps=MagicMock(),
        tf="8h",
        deploy_leverage=1.5,
    )

    assert result.status == "untested_no_data"
    assert result.verified is False


def test_assess_crisis_reliability_untested_when_registry_has_no_window_overlap() -> None:
    registry = _make_registry(symbol="NOTINCRISISUSDT", strategy_id="trend_donchian:donchian_72")

    result = assess_crisis_reliability(
        native_covered=False,
        native_detail="no_bottleneck_caliber_fold_in_window",
        deployment_registry=registry,
        strategy_cfg=MagicMock(),
        config=MagicMock(l2_max_mdd_abs=0.30),
        caps=MagicMock(),
        tf="8h",
        deploy_leverage=1.5,
    )

    assert result.status == "untested_no_data"
    assert result.verified is False


def test_assess_crisis_reliability_stress_tested_fail_when_mdd_exceeds_cap(
    mocker: Any,
) -> None:
    registry = _make_registry(symbol="ARUSDT", strategy_id="trend_donchian:donchian_72")
    aligned_stress = _make_aligned()
    window = CrisisWindow(
        start=aligned_stress.datetimes[0].astype("datetime64[D]").tolist(),
        end=aligned_stress.datetimes[-1].astype("datetime64[D]").tolist(),
        label="test_crisis",
        symbols=("ARUSDT",),
        source_note="synthetic",
    )
    panel = MagicMock(
        family="trend_donchian",
        variant="donchian_72",
        expected_holding_bars=12,
        signed_score_2d=np.ones((len(aligned_stress.datetimes), 1)),
        valid_mask_2d=np.ones((len(aligned_stress.datetimes), 1), dtype=bool),
    )

    mocker.patch(
        "src.domain.futures.optimization.opt_data_utils.load_futures_data_maps_for_symbols",
        return_value=({"ARUSDT": {}}, {}, ["ARUSDT"]),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.align_data_maps",
        return_value=aligned_stress,
    )
    mocker.patch(
        "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
        return_value=(panel,),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
        return_value=MagicMock(mdd=0.40, cagr=-0.10),
    )

    result = assess_crisis_reliability(
        native_covered=False,
        native_detail="no_bottleneck_caliber_fold_in_window",
        deployment_registry=registry,
        strategy_cfg=CandidateStrategyConfig(),
        config=MagicMock(l2_max_mdd_abs=0.30),
        caps=MagicMock(),
        tf="8h",
        deploy_leverage=1.5,
        crisis_windows=(window,),
    )

    assert result.status == "stress_tested_fail"
    assert result.verified is False
    assert result.stress_mdd == 0.40
    assert result.stress_symbol_count == 1


def test_apply_crisis_reliability_override_blocks_when_unverified() -> None:
    l2_result = _make_l2_result()
    assessment = CrisisReliabilityAssessment(
        status="stress_tested_fail", verified=False,
        detail="luna_ftx_2022_collapse: mdd=0.4000 > 0.30",
    )

    updated = apply_crisis_reliability_override(
        l2_result, assessment, require_crisis_reliability=True,
    )

    assert updated.gate_passed is False
    assert updated.blocker_reason == "crisis_unverified"
    assert updated.crisis_reliability_status == "stress_tested_fail"


def test_apply_crisis_reliability_override_opt_out_preserves_gate() -> None:
    l2_result = _make_l2_result()
    assessment = CrisisReliabilityAssessment(
        status="stress_tested_fail", verified=False, detail="...",
    )

    updated = apply_crisis_reliability_override(
        l2_result, assessment, require_crisis_reliability=False,
    )

    assert updated.gate_passed is True
    assert updated.crisis_reliability_status == "stress_tested_fail"
