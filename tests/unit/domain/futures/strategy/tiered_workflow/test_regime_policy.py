from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    RegimeCellPolicy,
)
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    apply_regime_cell_policy,
    build_regime_policy_by_fold,
    build_regime_routing_plan,
)
from src.domain.futures.strategy.walk_forward import WFFold


def _make_cache(n_bars: int, *, strategy_id: str = "donchian_72_4h") -> MagicMock:
    cache = MagicMock()
    cache.signal_mask_2d = np.ones((n_bars, 1), dtype=bool)
    cache.side_2d = np.ones((n_bars, 1), dtype=np.float64)
    cache.holding_bars_2d = np.ones((n_bars, 1), dtype=np.float64)
    cache.sleeve_to_sym = np.zeros(1, dtype=np.int64)
    cache.sleeve_ids = (("BTCUSDT", strategy_id),)
    cache.sleeve_to_tf = ("4h",)
    return cache


def _make_aligned(close_1d: list[float]) -> MagicMock:
    aligned = MagicMock()
    aligned.close_2d = np.asarray(close_1d, dtype=np.float64).reshape(-1, 1)
    aligned.symbols = ("BTCUSDT",)
    return aligned


def _make_fold() -> tuple[WFFold, ...]:
    return (
        WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=5, oos_start=5, oos_end=8),
    )


def _make_symbol_signal() -> SymbolSignal:
    return SymbolSignal(
        raw_mu=20.0,
        volatility=0.2,
        n_obs=10,
        t_stat=2.0,
        valid=True,
        beta_btc=None,
        quality_weight=1.0,
    )


def test_build_regime_policy_by_fold_ignores_oos_extreme_returns() -> None:
    folds = _make_fold()
    regime = np.zeros(8, dtype=np.int8)
    cache = _make_cache(8)
    aligned_a = _make_aligned([100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 90.0, 80.0])
    aligned_b = _make_aligned([100.0, 101.0, 102.0, 103.0, 104.0, 60.0, 50.0, 40.0])

    policy_a, diag_a = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned_a,
        awf_folds=folds,
        regime_code_1d=regime,
        state_names=("bull", "bear", "crisis"),
        mode="hybrid",
        min_n=1,
        cal_min_n=1,
        min_cal_lift_bps=1.0,
        min_confidence=0.0,
    )
    policy_b, diag_b = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned_b,
        awf_folds=folds,
        regime_code_1d=regime,
        state_names=("bull", "bear", "crisis"),
        mode="hybrid",
        min_n=1,
        cal_min_n=1,
        min_cal_lift_bps=1.0,
        min_confidence=0.0,
    )

    key = (0, "donchian_72", "4h")
    assert policy_a == policy_b
    assert policy_a[0][key].action == policy_b[0][key].action
    assert diag_a.reason == diag_b.reason


def test_build_regime_policy_by_fold_when_cal_sample_too_small_returns_pooled_policy() -> None:
    folds = _make_fold()
    regime = np.zeros(8, dtype=np.int8)
    cache = _make_cache(8)
    aligned = _make_aligned([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])

    policy_by_fold, diagnostics = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        regime_code_1d=regime,
        state_names=("bull", "bear", "crisis"),
        mode="hybrid",
        min_n=1,
        cal_min_n=10,
        min_confidence=0.8,
    )

    assert diagnostics.global_reliable is False
    assert diagnostics.n_block == 0
    assert diagnostics.n_downweight == 0
    assert all(policy.action == "pooled" for policy in policy_by_fold[0].values())
    assert all(policy.edge_multiplier == pytest.approx(1.0) for policy in policy_by_fold[0].values())


def test_apply_regime_cell_policy_soft_mode_downweights_negative_cell() -> None:
    sig = _make_symbol_signal()
    policy = RegimeCellPolicy(
        state=0,
        state_name="bull",
        family="donchian_72",
        tf="4h",
        action="downweight",
        reason="negative_cal_lift",
        edge_multiplier=0.5,
        confidence=1.0,
        fit_edge_bps=10.0,
        pooled_fit_edge_bps=5.0,
        cal_edge_bps=-5.0,
        pooled_cal_edge_bps=2.0,
        cal_lift_bps=-7.0,
        n_fit=5,
        n_cal=5,
    )

    result = apply_regime_cell_policy(
        {("BTCUSDT", "donchian_72_4h"): sig},
        {("BTCUSDT", "donchian_72_4h"): 20.0},
        {(0, "donchian_72", "4h"): policy},
        0,
        mode="soft",
    )

    assert ("BTCUSDT", "donchian_72_4h") in result.sleeve_sigs
    assert result.sleeve_edges[("BTCUSDT", "donchian_72_4h")] == pytest.approx(10.0)
    assert result.n_downweight == 1
    assert result.n_block == 0


def test_apply_regime_cell_policy_hybrid_mode_blocks_negative_cell() -> None:
    sig = _make_symbol_signal()
    block_policy = RegimeCellPolicy(
        state=0,
        state_name="bull",
        family="donchian_72",
        tf="4h",
        action="block",
        reason="negative_cal_lift",
        edge_multiplier=0.0,
        confidence=1.0,
        fit_edge_bps=10.0,
        pooled_fit_edge_bps=5.0,
        cal_edge_bps=-20.0,
        pooled_cal_edge_bps=2.0,
        cal_lift_bps=-22.0,
        n_fit=5,
        n_cal=5,
    )

    result = apply_regime_cell_policy(
        {
            ("BTCUSDT", "donchian_72_4h"): sig,
            ("BTCUSDT", "trend_pullback_4h"): sig,
        },
        {
            ("BTCUSDT", "donchian_72_4h"): 20.0,
            ("BTCUSDT", "trend_pullback_4h"): 15.0,
        },
        {(0, "donchian_72", "4h"): block_policy},
        0,
        mode="hybrid",
    )

    assert ("BTCUSDT", "donchian_72_4h") not in result.sleeve_sigs
    assert ("BTCUSDT", "trend_pullback_4h") in result.sleeve_sigs
    assert result.n_block == 1


def test_apply_regime_cell_policy_observe_mode_never_changes_sleeves_or_edges() -> None:
    sig = _make_symbol_signal()
    policy = RegimeCellPolicy(
        state=0,
        state_name="bull",
        family="donchian_72",
        tf="4h",
        action="block",
        reason="observe_only",
        edge_multiplier=0.0,
        confidence=1.0,
        fit_edge_bps=10.0,
        pooled_fit_edge_bps=5.0,
        cal_edge_bps=-20.0,
        pooled_cal_edge_bps=2.0,
        cal_lift_bps=-22.0,
        n_fit=5,
        n_cal=5,
    )
    sleeve_sigs = {("BTCUSDT", "donchian_72_4h"): sig}
    sleeve_edges = {("BTCUSDT", "donchian_72_4h"): 20.0}

    result = apply_regime_cell_policy(
        sleeve_sigs,
        sleeve_edges,
        {(0, "donchian_72", "4h"): policy},
        0,
        mode="observe",
    )

    assert result.sleeve_sigs == sleeve_sigs
    assert result.sleeve_edges == sleeve_edges
    assert result.n_block == 1


def test_layer2_allocation_config_validates_regime_policy_params() -> None:
    with pytest.raises(ValueError, match="l2_regime_policy_mode"):
        Layer2AllocationConfig.from_mapping({"l2_regime_policy_mode": "bad"})

    with pytest.raises(ValueError, match="l2_regime_soft_downweight_min"):
        Layer2AllocationConfig.from_mapping({"l2_regime_soft_downweight_min": -0.1})

    with pytest.raises(ValueError, match="l2_regime_soft_downweight_max"):
        Layer2AllocationConfig.from_mapping({"l2_regime_soft_downweight_max": 1.1})

    with pytest.raises(ValueError, match="l2_regime_min_policy_confidence"):
        Layer2AllocationConfig.from_mapping({"l2_regime_min_policy_confidence": 1.1})


def test_build_regime_routing_plan_attaches_policy_by_fold_and_diagnostics() -> None:
    raw_codes = np.zeros(8, dtype=np.int8)
    cache = _make_cache(8)
    aligned = _make_aligned([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])

    plan = build_regime_routing_plan(
        cache=cache,
        aligned=aligned,
        awf_folds=_make_fold(),
        raw_regime_code_1d=raw_codes,
        compression_enabled=True,
        proof_enabled=False,
        min_n=1,
        policy_mode="hybrid",
        policy_cal_min_n=1,
        policy_min_confidence=0.0,
        debug_diagnostics_enabled=False,
    )

    assert len(plan.policy_by_fold) == 1
    assert plan.diagnostics.policy_diagnostics is not None
    assert plan.diagnostics.debug_diagnostics is None
