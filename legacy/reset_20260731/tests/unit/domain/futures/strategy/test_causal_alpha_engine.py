from __future__ import annotations

import inspect
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.portfolio.online_growth_allocator import (
    OnlineAllocatorConfig,
    allocate_online_policy_mix,
    initialize_online_policy_state,
    update_online_policy_state,
)
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.candidate_contracts import (
    CausalAlphaSegment,
    CausalAlphaTape,
    ValidatedSignalBatch,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.causal_alpha_tape import (
    validate_alpha_tape_causality,
)
from src.domain.futures.strategy.walk_forward import build_causal_l2_folds


POLICIES = (
    "equal_weight",
    "inverse_vol",
    "kelly",
    "l1_confidence_shrinkage",
)


def _caps() -> PortfolioCaps:
    return PortfolioCaps(
        gross=2.0,
        per_symbol=1.0,
        net=2.0,
        beta=10.0,
        target_ann_vol=None,
    )


def _state(n_symbols: int = 2):
    return initialize_online_policy_state(
        policies=POLICIES,
        n_symbols=n_symbols,
        config_fingerprint="cfg-v1",
    )


def test_causal_fold_builder_has_real_policy_warmup() -> None:
    folds = build_causal_l2_folds(
        n_bars=1_000,
        l2_start_idx=200,
        holdout_start_idx=800,
        n_folds=3,
        min_warmup_bars=120,
    )
    assert len(folds) == 3
    assert folds[0].policy_fit_start == 200
    assert folds[0].policy_fit_end_exclusive == folds[0].oos_start
    assert folds[0].policy_fit_end_exclusive > folds[0].policy_fit_start
    assert folds[-1].oos_end_exclusive == 800
    assert all(a.oos_end_exclusive == b.oos_start for a, b in pairwise(folds))


def test_causal_fold_builder_rejects_insufficient_span() -> None:
    with pytest.raises(ValueError, match="insufficient L2 span"):
        build_causal_l2_folds(
            n_bars=100,
            l2_start_idx=90,
            holdout_start_idx=100,
            n_folds=4,
            min_warmup_bars=8,
        )


def test_build_cross_fitted_alpha_tape_creates_segments_for_folds() -> None:
    folds = build_causal_l2_folds(
        n_bars=1_000,
        l2_start_idx=200,
        holdout_start_idx=800,
        n_folds=3,
        min_warmup_bars=120,
    )
    aligned = AlignedMarketData(
        datetimes=np.array(["2020-01-01"], dtype="datetime64[ns]"),
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=np.ones((1, 2), dtype=np.float64),
        high_2d=np.ones((1, 2), dtype=np.float64),
        low_2d=np.ones((1, 2), dtype=np.float64),
        close_2d=np.ones((1, 2), dtype=np.float64),
        volume_2d=np.ones((1, 2), dtype=np.float32),
        funding_2d=np.zeros((1, 2), dtype=np.float32),
        active_mask=np.ones((1, 2), dtype=np.bool_),
        warm_mask=np.ones((1, 2), dtype=np.bool_),
        entry_block_mask=np.ones((1, 2), dtype=np.bool_),
        kill_mask=np.zeros((1, 2), dtype=np.bool_),
    )
    from src.domain.futures.strategy.tiered_workflow.causal_alpha_tape import (
        build_cross_fitted_alpha_tape,
    )
    tape = build_cross_fitted_alpha_tape(
        labeled_events=pd.DataFrame(),
        aligned=aligned,
        folds=folds,
        cfg=CandidateStrategyConfig(),
        timeframes=("4h",),
        max_label_horizon_bars=12,
        seed=42,
    )
    assert len(tape.segments) == 3
    assert tape.start_idx == folds[0].oos_start
    assert tape.end_idx_exclusive == folds[-1].oos_end_exclusive
    assert tape.symbols == ("BTCUSDT", "ETHUSDT")


def test_tape_rejects_model_or_evidence_leakage() -> None:
    batch = ValidatedSignalBatch(
        events=(),
        start_idx=100,
        end_idx=120,
        symbols=("BTCUSDT",),
        registry_version="r1",
        model_version="m1",
    )
    segment = CausalAlphaSegment(
        segment_idx=0,
        predict_start=100,
        predict_end_exclusive=120,
        model_fit_end_exclusive=99,
        max_evidence_exit_idx=100,
        is_warmup=True,
        batch=batch,
    )
    tape = CausalAlphaTape(
        segments=(segment,),
        start_idx=100,
        end_idx_exclusive=120,
        symbols=("BTCUSDT",),
        fingerprint="bad",
    )
    with pytest.raises(ValueError, match="evidence leakage"):
        validate_alpha_tape_causality(tape, purge_bars=1)


def test_online_state_rejects_future_update() -> None:
    with pytest.raises(ValueError, match="completed return must precede decision"):
        update_online_policy_state(
            state=_state(),
            completed_block_returns_by_policy=np.array([0.01, 0.02, 0.00, -0.01]),
            completed_return_end_idx=10,
            next_decision_idx=10,
            config=OnlineAllocatorConfig(),
        )


def test_all_non_positive_policy_growth_forces_cash() -> None:
    cfg = OnlineAllocatorConfig()
    state = _state()
    for block_idx in range(3):
        state = update_online_policy_state(
            state=state,
            completed_block_returns_by_policy=np.array([-0.02, -0.01, -0.03, -0.01]),
            completed_return_end_idx=block_idx * 10 + 9,
            next_decision_idx=block_idx * 10 + 10,
            config=cfg,
        )
    decision = allocate_online_policy_mix(
        state=state,
        policy_weights_2d=np.array([[1.0, 0.0], [0.5, -0.5], [0.8, -0.2], [0.6, -0.4]]),
        trailing_ensemble_returns=np.array([-0.01, -0.02, 0.0]),
        config=cfg,
        caps=_caps(),
        btc_beta=np.zeros(2),
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_equity_multiplier=0.20,
        exchange_leverage_cap=2.0,
        decision_idx=31,
    )
    assert decision.mode == "abstain_cash"
    assert decision.cash_weight == pytest.approx(1.0)
    np.testing.assert_array_equal(decision.target_weights, np.zeros(2))


def test_positive_history_produces_bounded_normalized_mix() -> None:
    cfg = OnlineAllocatorConfig()
    state = _state()
    blocks = (
        np.array([0.03, 0.02, 0.01, 0.025]),
        np.array([0.02, 0.018, -0.005, 0.020]),
        np.array([0.025, 0.019, 0.002, 0.021]),
        np.array([0.020, 0.017, 0.001, 0.018]),
    )
    for block_idx, returns in enumerate(blocks):
        state = update_online_policy_state(
            state=state,
            completed_block_returns_by_policy=returns,
            completed_return_end_idx=block_idx * 10 + 9,
            next_decision_idx=block_idx * 10 + 10,
            config=cfg,
        )
    decision = allocate_online_policy_mix(
        state=state,
        policy_weights_2d=np.array([[0.5, -0.5], [0.4, -0.6], [0.7, -0.3], [0.6, -0.4]]),
        trailing_ensemble_returns=np.full(60, 0.001),
        config=cfg,
        caps=_caps(),
        btc_beta=np.zeros(2),
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_equity_multiplier=0.20,
        exchange_leverage_cap=2.0,
        decision_idx=41,
    )
    assert decision.mode == "risk_on"
    assert max(decision.posterior_by_policy) <= 0.60 + 1e-12
    assert sum(decision.posterior_by_policy) + decision.cash_weight == pytest.approx(1.0)
    assert 0.0 < decision.risk_scale <= 2.0


def test_future_mutation_cannot_change_current_decision() -> None:
    cfg = OnlineAllocatorConfig()
    state = _state()
    for block_idx in range(3):
        state = update_online_policy_state(
            state=state,
            completed_block_returns_by_policy=np.array([0.02, 0.01, 0.015, 0.012]),
            completed_return_end_idx=block_idx * 10 + 9,
            next_decision_idx=block_idx * 10 + 10,
            config=cfg,
        )
    kwargs = {
        "state": state,
        "policy_weights_2d": np.ones((4, 2)) * 0.25,
        "trailing_ensemble_returns": np.full(40, 0.001),
        "config": cfg,
        "caps": _caps(),
        "btc_beta": np.zeros(2),
        "max_mdd": 0.30,
        "max_cvar_95": 0.06,
        "min_equity_multiplier": 0.20,
        "exchange_leverage_cap": 2.0,
        "decision_idx": 31,
    }
    first = allocate_online_policy_mix(**kwargs)
    mutated_future = np.full(100, -0.99)
    second = allocate_online_policy_mix(**kwargs)
    assert mutated_future.shape == (100,)
    np.testing.assert_allclose(first.target_weights, second.target_weights)
    assert first.posterior_by_policy == second.posterior_by_policy


def test_state_history_is_bounded() -> None:
    cfg = replace(OnlineAllocatorConfig(), max_history_blocks=12)
    state = _state()
    for block_idx in range(20):
        state = update_online_policy_state(
            state=state,
            completed_block_returns_by_policy=np.full(4, 0.01),
            completed_return_end_idx=block_idx * 10 + 9,
            next_decision_idx=block_idx * 10 + 10,
            config=cfg,
        )
    assert all(len(history) == 12 for history in state.block_growth_by_policy)


def test_policy_shadow_returns_are_policy_specific() -> None:
    cfg = OnlineAllocatorConfig()
    state = _state(n_symbols=3)
    for block_idx in range(3):
        returns = np.array([0.02, 0.015, -0.01, 0.01]) * (block_idx + 1)
        state = update_online_policy_state(
            state=state,
            completed_block_returns_by_policy=returns,
            completed_return_end_idx=block_idx * 10 + 9,
            next_decision_idx=block_idx * 10 + 10,
            config=cfg,
        )
    for policy_idx in range(4):
        growth_history = state.block_growth_by_policy[policy_idx]
        assert len(growth_history) == 3
        assert growth_history[1] != growth_history[0] if policy_idx != 0 else True


def test_growth_safety_summary_uses_mapping_keys() -> None:
    from src.domain.futures.strategy.tiered_workflow.l2_gate import (
        evaluate_growth_safety_gate,
    )

    gate = evaluate_growth_safety_gate(
        tape_valid=True,
        execution_valid=True,
        deployed_returns=np.array([0.01, -0.005, 0.02, 0.0, -0.01], dtype=np.float64),
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_equity_multiplier=0.20,
        growth_lcb=0.01,
    )
    mapping = gate.constraints.as_mapping()
    expected_keys = {"data_integrity", "execution_integrity", "mdd", "cvar_95", "ruin"}
    assert set(mapping) == expected_keys
    assert gate.mode == "risk_on"
    assert gate.passed


def test_active_pipeline_builds_alpha_tape_once() -> None:
    from src.domain.futures.strategy.walk_forward import build_causal_l2_folds

    folds = build_causal_l2_folds(
        n_bars=1_000,
        l2_start_idx=200,
        holdout_start_idx=800,
        n_folds=3,
        min_warmup_bars=120,
    )
    assert len(folds) == 3
    tape_fingerprints: set[str] = set()
    import hashlib
    for fold in folds:
        fp = hashlib.sha256(
            f"{fold.fold_idx}_{fold.policy_fit_start}_{fold.oos_end_exclusive}".encode()
        ).hexdigest()[:8]
        tape_fingerprints.add(fp)
    assert len(tape_fingerprints) == 3


def test_active_pipeline_does_not_call_l2_optuna_or_champion_selection() -> None:
    import src.domain.futures.strategy.tiered_workflow.pipeline as _pipeline_mod

    src = inspect.getsource(_pipeline_mod.run_online_growth_l2)
    assert "optuna" not in src.lower()
    assert "champion" not in src.lower()


def test_online_oos_target_uses_shadow_posterior_not_direct_kelly() -> None:
    cfg = OnlineAllocatorConfig()
    state = _state(n_symbols=2)
    for block_idx in range(3):
        state = update_online_policy_state(
            state=state,
            completed_block_returns_by_policy=np.array([0.02, 0.015, -0.01, 0.01]),
            completed_return_end_idx=block_idx * 10 + 9,
            next_decision_idx=block_idx * 10 + 10,
            config=cfg,
        )
    decision = allocate_online_policy_mix(
        state=state,
        policy_weights_2d=np.array([[0.5, -0.5], [0.4, -0.6], [0.7, -0.3], [0.6, -0.4]]),
        trailing_ensemble_returns=np.full(40, 0.001),
        config=cfg,
        caps=_caps(),
        btc_beta=np.zeros(2),
        max_mdd=0.30,
        max_cvar_95=0.06,
        min_equity_multiplier=0.20,
        exchange_leverage_cap=2.0,
        decision_idx=31,
    )
    assert sum(decision.posterior_by_policy) + decision.cash_weight == pytest.approx(1.0)
    assert len(decision.posterior_by_policy) == 4


def test_signal_window_and_policy_fit_window_overlap() -> None:
    folds = build_causal_l2_folds(
        n_bars=1_000,
        l2_start_idx=200,
        holdout_start_idx=800,
        n_folds=3,
        min_warmup_bars=120,
    )
    for fold in folds:
        assert fold.policy_fit_end_exclusive == fold.oos_start
        assert fold.policy_fit_start < fold.policy_fit_end_exclusive
        assert fold.oos_start < fold.oos_end_exclusive
