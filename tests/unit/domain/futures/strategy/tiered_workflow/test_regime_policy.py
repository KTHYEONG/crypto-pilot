from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow import l2_meta
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    RegimeCellPolicy,
)
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    apply_regime_cell_policy,
    apply_regime_risk_cap,
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


def _policy(
    *,
    action: str = "downweight",
    reason: str = "negative_cal_lift",
    edge_multiplier: float = 0.5,
    fit_lift_bps: float = -20.0,
    cal_lift_bps: float = -20.0,
    sign_consistent: bool = True,
    hard_block_eligible: bool = False,
) -> RegimeCellPolicy:
    return RegimeCellPolicy(
        state=0,
        state_name="bull",
        family="donchian_72",
        tf="4h",
        action=action,  # type: ignore[arg-type]
        reason=reason,  # type: ignore[arg-type]
        edge_multiplier=edge_multiplier,
        confidence=1.0,
        fit_edge_bps=10.0,
        pooled_fit_edge_bps=5.0,
        cal_edge_bps=-5.0,
        pooled_cal_edge_bps=2.0,
        fit_lift_bps=fit_lift_bps,
        cal_lift_bps=cal_lift_bps,
        sign_consistent=sign_consistent,
        hard_block_eligible=hard_block_eligible,
        n_fit=5,
        n_cal=5,
    )


def test_layer2_allocation_config_defaults_regime_soft_risk_caps() -> None:
    cfg = Layer2AllocationConfig.from_mapping({})

    assert cfg.l2_regime_policy_mode == "soft"
    assert cfg.l2_regime_soft_downweight_min == pytest.approx(0.50)
    assert cfg.l2_regime_hard_block_enabled is False
    assert cfg.l2_regime_risk_cap_enabled is True
    assert cfg.l2_regime_bull_gross_cap == pytest.approx(1.0)
    assert cfg.l2_regime_bear_gross_cap == pytest.approx(0.75)
    assert cfg.l2_regime_crisis_gross_cap == pytest.approx(0.55)


def test_layer2_allocation_config_validates_regime_risk_caps() -> None:
    with pytest.raises(ValueError, match="l2_regime_bull_gross_cap"):
        Layer2AllocationConfig.from_mapping({"l2_regime_bull_gross_cap": 0.0})

    with pytest.raises(ValueError, match="l2_regime_bear_gross_cap"):
        Layer2AllocationConfig.from_mapping({"l2_regime_bear_gross_cap": 1.1})

    with pytest.raises(ValueError, match="l2_regime_crisis_gross_cap"):
        Layer2AllocationConfig.from_mapping({"l2_regime_crisis_gross_cap": -0.1})


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
    assert diagnostics.n_unstable == 0
    assert all(policy.action == "pooled" for policy in policy_by_fold[0].values())
    assert all(policy.edge_multiplier == pytest.approx(1.0) for policy in policy_by_fold[0].values())


def test_build_regime_policy_by_fold_sign_unstable_returns_pooled(monkeypatch: pytest.MonkeyPatch) -> None:
    folds = _make_fold()
    regime = np.zeros(8, dtype=np.int8)
    cache = _make_cache(8)
    aligned = _make_aligned([100.0, 110.0, 120.0, 130.0, 140.0, 120.0, 110.0, 100.0])

    def _bucket_stats(*args: object, start: int, **kwargs: object) -> dict[tuple[int, str, str], SimpleNamespace]:
        if start == 0:
            return {(0, "donchian_72", "4h"): SimpleNamespace(edge_bps=60.0, n_obs=5)}
        return {(0, "donchian_72", "4h"): SimpleNamespace(edge_bps=-40.0, n_obs=5)}

    def _pooled_stats(*args: object, start: int, **kwargs: object) -> dict[tuple[str, str], SimpleNamespace]:
        if start == 0:
            return {("donchian_72", "4h"): SimpleNamespace(edge_bps=10.0, n_obs=5)}
        return {("donchian_72", "4h"): SimpleNamespace(edge_bps=5.0, n_obs=5)}

    monkeypatch.setattr(l2_meta, "compute_bucket_realized_edge_stats", _bucket_stats)
    monkeypatch.setattr(l2_meta, "compute_pooled_realized_edge_stats", _pooled_stats)

    policy_by_fold, diagnostics = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        regime_code_1d=regime,
        state_names=("bull", "bear", "crisis"),
        mode="soft",
        min_n=1,
        cal_min_n=1,
        min_cal_lift_bps=8.0,
        block_lift_bps=-12.0,
        min_confidence=0.0,
        require_sign_consistency=True,
    )

    key = (0, "donchian_72", "4h")
    assert policy_by_fold[0][key].action == "pooled"
    assert policy_by_fold[0][key].reason == "cal_sign_unstable"
    assert policy_by_fold[0][key].sign_consistent is False
    assert diagnostics.n_unstable >= 1
    assert diagnostics.global_reliable is False


def test_apply_regime_cell_policy_soft_mode_downweights_negative_cell() -> None:
    sig = _make_symbol_signal()
    policy = _policy(edge_multiplier=0.5, cal_lift_bps=-7.0)

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


def test_apply_regime_cell_policy_soft_never_blocks_even_when_policy_action_block() -> None:
    sig = _make_symbol_signal()
    policy = _policy(action="block", edge_multiplier=0.0, hard_block_eligible=True)

    result = apply_regime_cell_policy(
        {("BTCUSDT", "donchian_72_4h"): sig},
        {("BTCUSDT", "donchian_72_4h"): 20.0},
        {(0, "donchian_72", "4h"): policy},
        0,
        mode="soft",
    )

    assert ("BTCUSDT", "donchian_72_4h") in result.sleeve_sigs
    assert result.sleeve_edges[("BTCUSDT", "donchian_72_4h")] == pytest.approx(20.0)
    assert result.n_block == 0


def test_apply_regime_cell_policy_hybrid_mode_blocks_negative_cell() -> None:
    sig = _make_symbol_signal()
    block_policy = _policy(action="block", edge_multiplier=0.0, cal_lift_bps=-22.0, hard_block_eligible=True)

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


def test_build_regime_policy_by_fold_hybrid_requires_hard_block_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    folds = _make_fold()
    regime = np.zeros(8, dtype=np.int8)
    cache = _make_cache(8)
    aligned = _make_aligned([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0])

    def _bucket_stats(*args: object, start: int, **kwargs: object) -> dict[tuple[int, str, str], SimpleNamespace]:
        if start == 0:
            return {(0, "donchian_72", "4h"): SimpleNamespace(edge_bps=-40.0, n_obs=5)}
        return {(0, "donchian_72", "4h"): SimpleNamespace(edge_bps=-60.0, n_obs=5)}

    def _pooled_stats(*args: object, **kwargs: object) -> dict[tuple[str, str], SimpleNamespace]:
        return {("donchian_72", "4h"): SimpleNamespace(edge_bps=0.0, n_obs=5)}

    monkeypatch.setattr(l2_meta, "compute_bucket_realized_edge_stats", _bucket_stats)
    monkeypatch.setattr(l2_meta, "compute_pooled_realized_edge_stats", _pooled_stats)

    policy_soft_blocked, diag_soft_blocked = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        regime_code_1d=regime,
        state_names=("bull", "bear", "crisis"),
        mode="hybrid",
        min_n=1,
        cal_min_n=1,
        min_cal_lift_bps=8.0,
        block_lift_bps=-12.0,
        min_confidence=0.0,
        hard_block_enabled=False,
        block_min_confidence=0.8,
        require_sign_consistency=False,
    )

    key = (0, "donchian_72", "4h")
    assert policy_soft_blocked[0][key].action == "downweight"
    assert policy_soft_blocked[0][key].hard_block_eligible is False
    assert diag_soft_blocked.n_block == 0

    policy_hard_blocked, diag_hard_blocked = build_regime_policy_by_fold(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        regime_code_1d=regime,
        state_names=("bull", "bear", "crisis"),
        mode="hybrid",
        min_n=1,
        cal_min_n=1,
        min_cal_lift_bps=8.0,
        block_lift_bps=-12.0,
        min_confidence=0.0,
        hard_block_enabled=True,
        block_min_confidence=0.8,
        require_sign_consistency=False,
    )

    assert policy_hard_blocked[0][key].action == "block"
    assert policy_hard_blocked[0][key].hard_block_eligible is True
    assert diag_hard_blocked.n_block >= 1


def test_apply_regime_cell_policy_observe_mode_never_changes_sleeves_or_edges() -> None:
    sig = _make_symbol_signal()
    policy = _policy(action="block", reason="observe_only", edge_multiplier=0.0, cal_lift_bps=-22.0)
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


def test_build_regime_routing_plan_forwards_soft_policy_knobs() -> None:
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
        policy_mode="soft",
        policy_cal_min_n=1,
        policy_min_confidence=0.0,
        policy_hard_block_enabled=False,
        policy_require_sign_consistency=True,
        debug_diagnostics_enabled=False,
    )

    assert plan.diagnostics.policy_diagnostics is not None
    assert plan.diagnostics.policy_diagnostics.mode == "soft"
    assert plan.diagnostics.policy_diagnostics.hard_block_enabled is False
    assert all(policy.action != "block" for fold in plan.policy_by_fold for policy in fold.values())


def test_apply_regime_risk_cap_scales_crisis_gross() -> None:
    weights = np.asarray([0.8, -0.7], dtype=np.float64)

    scaled, mult = apply_regime_risk_cap(
        weights,
        2,
        ("bull", "bear", "crisis"),
        crisis_gross_cap=0.55,
    )

    assert mult == pytest.approx(0.55 / 1.5)
    assert np.sum(np.abs(scaled)) == pytest.approx(0.55)
    assert np.sign(scaled[0]) == np.sign(weights[0])
    assert np.sign(scaled[1]) == np.sign(weights[1])


def test_apply_regime_risk_cap_noop_when_disabled_or_under_cap() -> None:
    weights = np.asarray([0.2, -0.1], dtype=np.float64)

    scaled_disabled, mult_disabled = apply_regime_risk_cap(
        weights,
        1,
        ("bull", "bear", "crisis"),
        enabled=False,
    )
    scaled_under_cap, mult_under_cap = apply_regime_risk_cap(
        weights,
        1,
        ("bull", "bear", "crisis"),
        bear_gross_cap=0.75,
    )
    zeros = np.zeros(2, dtype=np.float64)
    scaled_zero, mult_zero = apply_regime_risk_cap(
        zeros,
        1,
        ("bull", "bear", "crisis"),
    )

    assert mult_disabled == pytest.approx(1.0)
    assert np.array_equal(scaled_disabled, weights)
    assert mult_under_cap == pytest.approx(1.0)
    assert np.array_equal(scaled_under_cap, weights)
    assert mult_zero == pytest.approx(1.0)
    assert np.array_equal(scaled_zero, zeros)


def test_apply_regime_risk_cap_invalid_cap_raises() -> None:
    weights = np.asarray([0.2, -0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="gross_cap"):
        apply_regime_risk_cap(weights, 0, ("bull", "bear", "crisis"), bull_gross_cap=0.0)
    with pytest.raises(ValueError, match="gross_cap"):
        apply_regime_risk_cap(weights, 1, ("bull", "bear", "crisis"), bear_gross_cap=1.5)
    with pytest.raises(ValueError, match="gross_cap"):
        apply_regime_risk_cap(weights, 2, ("bull", "bear", "crisis"), crisis_gross_cap=-0.2)
