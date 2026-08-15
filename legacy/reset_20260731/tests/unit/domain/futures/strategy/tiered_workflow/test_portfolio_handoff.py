from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    QualifiedSignalRegistry,
    SignalSleeveKey,
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import (
    PortfolioHandoffConfig,
    _bar_level_marginal_growth_lcb,
    _kelly_proportional_weights,
    _l1_evidence_by_key,
    _moving_block_bootstrap_lcb,
    _rank_and_cap_sleeve_indices,
    evaluate_portfolio_handoff,
)


def _ev(sym: str, sid: str, qw: float) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(sym, sid, "ctx"),
        mean_gross_bps=10.0, mean_incremental_bps=5.0, block_tstat_incremental=2.0,
        probability_positive=0.85, p_value=0.01, q_value=0.05, positive_fold_ratio=1.0,
        n_obs=100, effective_n=80.0, n_folds=4, quality_weight=qw, hard_eligible=True,
        lcb_net_bps=3.0,
    )


def test_portfolio_handoff_defaults_are_conservative() -> None:
    config = PortfolioHandoffConfig()
    assert config.max_candidate_sleeves == 32
    assert config.min_calibration_windows == 3
    assert not hasattr(config, "max_sleeves_per_cluster")


def test_rank_and_cap_sleeve_indices_keeps_top_n_by_quality() -> None:
    sleeve_keys = tuple(
        SignalSleeveKey(f"SYM{i}", "4h", "strat") for i in range(5)
    )
    registry = QualifiedSignalRegistry(
        by_symbol={
            f"SYM{i}": (_ev(f"SYM{i}", "strat", qw),)
            for i, qw in enumerate([0.1, 0.9, 0.5, 0.3, 0.7])
        },
        ready_symbols=tuple(f"SYM{i}" for i in range(5)),
        trade_scope_count=5, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=3)
    assert active == (1, 2, 4)


def test_rank_and_cap_sleeve_indices_deterministic_tie_break() -> None:
    sleeve_keys = (
        SignalSleeveKey("SYMa", "4h", "s1"),
        SignalSleeveKey("SYMb", "4h", "s2"),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={
            "SYMa": (_ev("SYMa", "s1", 0.5),),
            "SYMb": (_ev("SYMb", "s2", 0.5),),
        },
        ready_symbols=("SYMa", "SYMb"),
        trade_scope_count=2, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    result1 = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=1)
    result2 = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=1)
    assert result1 == result2


def test_rank_and_cap_sleeve_indices_missing_registry_entry_defaults_zero_quality() -> None:
    sleeve_keys = (
        SignalSleeveKey("SYM_A", "4h", "s1"),
        SignalSleeveKey("SYM_B", "4h", "s2"),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"SYM_A": (_ev("SYM_A", "s1", 0.8),)},
        ready_symbols=("SYM_A",),
        trade_scope_count=1, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    result = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=2)
    assert result == (0, 1)


def test_kelly_proportional_weights_favors_lower_volatility() -> None:
    rng = np.random.default_rng(7)
    n = 200
    low_vol = 0.001 + rng.normal(0, 0.0005, n)
    high_vol = 0.001 + rng.normal(0, 0.005, n)
    window = np.column_stack([low_vol, high_vol])
    weights = _kelly_proportional_weights(window)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert weights[0] > weights[1]


def test_kelly_proportional_weights_zero_for_negative_mean_column() -> None:
    window = np.column_stack([
        np.full(50, 0.002), np.full(50, -0.001),
    ])
    weights = _kelly_proportional_weights(window)
    assert weights[1] == 0.0
    assert weights[0] == pytest.approx(1.0)


def test_kelly_proportional_weights_falls_back_to_equal_when_all_nonpositive() -> None:
    window = np.column_stack([np.full(30, -0.001), np.full(30, -0.002)])
    weights = _kelly_proportional_weights(window)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.5)


def test_kelly_proportional_weights_single_column_no_crash() -> None:
    window = np.column_stack([np.full(50, 0.001)])
    weights = _kelly_proportional_weights(window)
    assert weights[0] == pytest.approx(1.0)


def test_handoff_single_family_pool_is_no_longer_blanket_blocked() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import evaluate_portfolio_handoff
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (
        SignalSleeveKey("BTCUSDT", "4h", "family_a:strat1"),
        SignalSleeveKey("ETHUSDT", "4h", "family_a:strat2"),
    )
    n_bars = 90
    rng = np.random.default_rng(42)
    rets = np.column_stack([
        0.001 + rng.normal(0, 0.0005, n_bars),
        0.0015 + rng.normal(0, 0.0006, n_bars),
    ]).astype(np.float64)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 2)), tradeable_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 2)), funding_2d=np.zeros((n_bars, 2)),
        beta_1d=np.ones(2),
        expected_gross_bps_2d=np.ones((n_bars, 2)), expected_net_bps_2d=np.ones((n_bars, 2)),
        holding_bars_2d=np.ones((n_bars, 2)), side_2d=np.ones((n_bars, 2)),
        quality_weight_2d=np.ones((n_bars, 2)), signal_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        sleeve_to_sym=np.array([0, 1], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={
            "BTCUSDT": (_ev("BTCUSDT", "family_a:strat1", 0.9),),
            "ETHUSDT": (_ev("ETHUSDT", "family_a:strat2", 0.8),),
        },
        ready_symbols=("BTCUSDT", "ETHUSDT"),
        trade_scope_count=2, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT", "ETHUSDT"),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(min_source_families=2),
    )

    assert result.passed, f"handoff blocked: {result.blocker_reason}"
    for ev in result.evidence_by_fold[0]:
        if ev.admitted:
            assert "insufficient_family_diversity" not in ev.rejection_reasons


def test_moving_block_bootstrap_lcb_not_degenerate_at_three_windows() -> None:
    values = np.array([0.01, -0.02, 0.03])
    lcb = _moving_block_bootstrap_lcb(values, k=1.0, block_size=10, seed=42)
    assert lcb < float(np.mean(values))
    assert np.isfinite(lcb)


# ── New bar-level bootstrap statistic tests ──


def test_bar_level_marginal_growth_lcb_bounded_for_sparse_adverse_move() -> None:
    n_bars = 90
    rng = np.random.default_rng(11)
    base = 0.0005 + rng.normal(0, 0.0002, n_bars)
    sparse = np.zeros(n_bars)
    sparse[10] = -0.05
    returns_window = np.column_stack([base, sparse])
    w = np.array([0.9, 0.1])

    lcb, pos_ratio, chunk_sums = _bar_level_marginal_growth_lcb(
        returns_window, w, sleeve_s=1, n_chunks=3, seed=7,
    )

    assert np.isfinite(lcb)
    assert abs(lcb) < 1.0
    assert len(chunk_sums) == 3


def test_bar_level_marginal_growth_lcb_insufficient_bars_returns_neg_inf() -> None:
    returns_window = np.zeros((1, 2))
    lcb, pos_ratio, chunk_sums = _bar_level_marginal_growth_lcb(
        returns_window, np.array([0.5, 0.5]), sleeve_s=0, n_chunks=3,
    )
    assert lcb == float("-inf")
    assert pos_ratio == 0.0
    assert chunk_sums == ()


def test_l1_evidence_by_key_keys_by_symbol_and_strategy_id() -> None:
    registry = QualifiedSignalRegistry(
        by_symbol={
            "BTCUSDT": (_ev("BTCUSDT", "strat_a", 0.8),),
            "ETHUSDT": (_ev("ETHUSDT", "strat_b", 0.6),),
        },
        ready_symbols=("BTCUSDT", "ETHUSDT"), trade_scope_count=2, registry_version="test-v1",
    )
    lookup = _l1_evidence_by_key(registry)
    assert lookup[("BTCUSDT", "strat_a")].quality_weight == pytest.approx(0.8)
    assert ("ETHUSDT", "strat_a") not in lookup


def test_l1_evidence_by_key_empty_registry() -> None:
    registry = QualifiedSignalRegistry(
        by_symbol={}, ready_symbols=(), trade_scope_count=0, registry_version="test-v1",
    )
    lookup = _l1_evidence_by_key(registry)
    assert lookup == {}


def test_rank_and_cap_sleeve_indices_guarantees_per_tf_quota() -> None:
    tfs = ("1h", "2h", "4h", "6h", "8h", "12h", "1d")
    sleeve_keys = tuple(
        SignalSleeveKey(f"SYM{i}", tf, "strat")
        for tf in tfs
        for i in range(3)
    )
    registry = QualifiedSignalRegistry(
        by_symbol={
            f"SYM{i}": (_ev(f"SYM{i}", "strat", 0.9 - i * 0.1),)
            for i in range(3)
        },
        ready_symbols=tuple(f"SYM{i}" for i in range(3)),
        trade_scope_count=3, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=14)
    selected_tfs = [sleeve_keys[i].native_tf for i in active]
    from collections import Counter
    counts = Counter(selected_tfs)
    assert len(active) == 14
    assert all(counts[tf] == 2 for tf in tfs)


def test_rank_and_cap_sleeve_indices_backfills_from_undersized_tf_group() -> None:
    tfs = ("1h", "2h", "4h")
    sleeve_keys = tuple(
        SignalSleeveKey(f"SYM{i}", tf, "strat")
        for tf in tfs
        for i in range(3)
    )
    registry = QualifiedSignalRegistry(
        by_symbol={f"SYM{i}": (_ev(f"SYM{i}", "strat", 0.9 - i * 0.1),) for i in range(3)},
        ready_symbols=tuple(f"SYM{i}" for i in range(3)),
        trade_scope_count=3, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=5)
    selected_tfs = [sleeve_keys[i].native_tf for i in active]
    from collections import Counter
    counts = Counter(selected_tfs)
    assert len(active) == 5
    assert all(counts[tf] >= 1 for tf in tfs)


def test_rank_and_cap_sleeve_indices_single_tf_matches_legacy_flat_ranking() -> None:
    sleeve_keys = tuple(SignalSleeveKey(f"SYM{i}", "4h", "strat") for i in range(5))
    registry = QualifiedSignalRegistry(
        by_symbol={f"SYM{i}": (_ev(f"SYM{i}", "strat", qw),) for i, qw in enumerate([0.1, 0.9, 0.5, 0.3, 0.7])},
        ready_symbols=tuple(f"SYM{i}" for i in range(5)), trade_scope_count=5, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=3)
    assert active == (1, 2, 4)


def test_rank_and_cap_sleeve_indices_uneven_division_deterministic_backfill() -> None:
    tfs = ("1h", "2h", "4h")
    sleeve_keys = tuple(
        SignalSleeveKey(f"SYM{i}", tf, "strat")
        for tf in tfs
        for i in range(3)
    )
    registry = QualifiedSignalRegistry(
        by_symbol={f"SYM{i}": (_ev(f"SYM{i}", "strat", 0.95 - i * 0.1),) for i in range(3)},
        ready_symbols=tuple(f"SYM{i}" for i in range(3)),
        trade_scope_count=3, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=7)
    selected_tfs = [sleeve_keys[i].native_tf for i in active]
    from collections import Counter
    counts = Counter(selected_tfs)
    assert len(active) == 7
    assert len(counts) == len(tfs)


def test_rank_and_cap_sleeve_indices_accepts_prebuilt_evidence_lookup() -> None:
    sleeve_keys = tuple(SignalSleeveKey(f"SYM{i}", "4h", "strat") for i in range(3))
    registry = QualifiedSignalRegistry(
        by_symbol={f"SYM{i}": (_ev(f"SYM{i}", "strat", qw),) for i, qw in enumerate([0.1, 0.9, 0.5])},
        ready_symbols=tuple(f"SYM{i}" for i in range(3)), trade_scope_count=3, registry_version="test-v1",
    )
    lookup = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, lookup, max_candidate_sleeves=2)
    assert active == (1, 2)


# ── L1-override precedence tests ──


def _ev_with_lcb(sym: str, sid: str, qw: float, lcb_bps: float) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(sym, sid, "ctx"),
        mean_gross_bps=10.0, mean_incremental_bps=5.0, block_tstat_incremental=2.0,
        probability_positive=0.85, p_value=0.01, q_value=0.05, positive_fold_ratio=1.0,
        n_obs=100, effective_n=80.0, n_folds=4, quality_weight=qw, hard_eligible=True,
        lcb_net_bps=lcb_bps,
    )


def test_handoff_no_override_without_l1_match() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "4h", "unknown_strat"),)
    n_bars = 90
    rets = np.zeros((n_bars, 1), dtype=np.float64)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1)), tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1)), funding_2d=np.zeros((n_bars, 1)), beta_1d=np.ones(1),
        expected_gross_bps_2d=np.ones((n_bars, 1)), expected_net_bps_2d=np.ones((n_bars, 1)),
        holding_bars_2d=np.ones((n_bars, 1)), side_2d=np.ones((n_bars, 1)),
        quality_weight_2d=np.ones((n_bars, 1)), signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"OTHER": (_ev_with_lcb("OTHER", "strat", 0.9, 200.0),)},
        ready_symbols=("OTHER",), trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(),
    )

    ev = result.evidence_by_fold[0][0]
    assert not ev.admitted
    assert not ev.admitted_via_l1_edge_override


def test_handoff_no_override_when_l1_lcb_net_bps_nonpositive() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "4h", "strat"),)
    n_bars = 90
    rets = np.zeros((n_bars, 1), dtype=np.float64)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1)), tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1)), funding_2d=np.zeros((n_bars, 1)), beta_1d=np.ones(1),
        expected_gross_bps_2d=np.ones((n_bars, 1)), expected_net_bps_2d=np.ones((n_bars, 1)),
        holding_bars_2d=np.ones((n_bars, 1)), side_2d=np.ones((n_bars, 1)),
        quality_weight_2d=np.ones((n_bars, 1)), signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (_ev_with_lcb("BTCUSDT", "strat", 0.9, 0.0),)},
        ready_symbols=("BTCUSDT",), trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(),
    )

    ev = result.evidence_by_fold[0][0]
    assert not ev.admitted
    assert not ev.admitted_via_l1_edge_override


def test_handoff_override_admitted_sleeve_still_pruned_by_correlation() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (
        SignalSleeveKey("BTCUSDT", "4h", "strat_a"),
        SignalSleeveKey("BTCUSDT", "4h", "strat_b"),
    )
    n_bars = 90
    rng = np.random.default_rng(42)
    common = rng.normal(0, 0.001, n_bars)
    rets = np.column_stack([common, common * 0.99 + rng.normal(0, 0.00001, n_bars)])

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 2)), tradeable_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 2)), funding_2d=np.zeros((n_bars, 2)), beta_1d=np.ones(2),
        expected_gross_bps_2d=np.ones((n_bars, 2)), expected_net_bps_2d=np.ones((n_bars, 2)),
        holding_bars_2d=np.ones((n_bars, 2)), side_2d=np.ones((n_bars, 2)),
        quality_weight_2d=np.ones((n_bars, 2)), signal_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        sleeve_to_sym=np.array([0, 1], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={
            "BTCUSDT": (
                _ev_with_lcb("BTCUSDT", "strat_a", 0.8, 200.0),
                _ev_with_lcb("BTCUSDT", "strat_b", 0.6, 150.0),
            ),
        },
        ready_symbols=("BTCUSDT",), trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(),
    )

    admitted = [ev for ev in result.evidence_by_fold[0] if ev.admitted]
    assert len(admitted) <= 1


def test_handoff_admits_sleeves_with_positive_marginal_growth_only() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "4h", "strat_a"),)
    n_bars = 90
    rets = np.full((n_bars, 1), 0.003, dtype=np.float64)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1)), tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1)), funding_2d=np.zeros((n_bars, 1)),
        beta_1d=np.ones(1),
        expected_gross_bps_2d=np.ones((n_bars, 1)), expected_net_bps_2d=np.ones((n_bars, 1)),
        holding_bars_2d=np.ones((n_bars, 1)), side_2d=np.ones((n_bars, 1)),
        quality_weight_2d=np.ones((n_bars, 1)), signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (_ev_with_lcb("BTCUSDT", "strat_a", 0.9, 200.0),)},
        ready_symbols=("BTCUSDT",), trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(min_source_families=1),
    )

    ev = result.evidence_by_fold[0][0]
    assert ev.admitted_via_l1_edge_override is False


def test_handoff_invalid_handoff_weights_not_triggered_by_negative_override_weight_sum() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (
        SignalSleeveKey("BTCUSDT", "4h", "strat_a"),
        SignalSleeveKey("ETHUSDT", "4h", "strat_b"),
    )
    n_bars = 90
    rng = np.random.default_rng(42)
    rets = np.zeros((n_bars, 2), dtype=np.float64)
    rets[:, 0] = 0.002 + rng.normal(0, 0.0005, n_bars)
    rets[:, 1] = -0.001 + rng.normal(0, 0.0003, n_bars)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 2)), tradeable_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 2)), funding_2d=np.zeros((n_bars, 2)), beta_1d=np.ones(2),
        expected_gross_bps_2d=np.ones((n_bars, 2)), expected_net_bps_2d=np.ones((n_bars, 2)),
        holding_bars_2d=np.ones((n_bars, 2)), side_2d=np.ones((n_bars, 2)),
        quality_weight_2d=np.ones((n_bars, 2)), signal_mask_2d=np.ones((n_bars, 2), dtype=np.bool_),
        sleeve_to_sym=np.array([0, 1], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={
            "BTCUSDT": (_ev_with_lcb("BTCUSDT", "strat_a", 0.9, 200.0),),
            "ETHUSDT": (_ev_with_lcb("ETHUSDT", "strat_b", 0.8, 100.0),),
        },
        ready_symbols=("BTCUSDT", "ETHUSDT"), trade_scope_count=2, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT", "ETHUSDT"),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(),
    )

    assert result.blocker_reason != "invalid_handoff_weights"


def test_rank_and_cap_suffixed_sleeve_reads_registry_quality_weight() -> None:
    from src.domain.futures.strategy.tiered_workflow.portfolio_handoff import _rank_and_cap_sleeve_indices

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "8h", "tpc_50_200_8h"),)
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (_ev("BTCUSDT", "tpc_50_200", 0.9),)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test-v1",
    )
    evidence_by_key = _l1_evidence_by_key(registry)
    active = _rank_and_cap_sleeve_indices(sleeve_keys, evidence_by_key, max_candidate_sleeves=1)
    assert active == (0,), f"suffixed sleeve should match registry base id, got {active}"


def test_handoff_l1_edge_override_never_fires() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "8h", "tpc_50_200_8h"),)
    n_bars = 90
    rng = np.random.default_rng(42)
    rets = (rng.normal(-0.0005, 0.001, size=(n_bars, 1)).astype(np.float32),)

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1)), tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1)), funding_2d=np.zeros((n_bars, 1)), beta_1d=np.ones(1),
        expected_gross_bps_2d=np.ones((n_bars, 1)), expected_net_bps_2d=np.ones((n_bars, 1)),
        holding_bars_2d=np.ones((n_bars, 1)), side_2d=np.ones((n_bars, 1)),
        quality_weight_2d=np.ones((n_bars, 1)), signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (_ev_with_lcb("BTCUSDT", "tpc_50_200", 0.9, 120.0),)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=rets,
        config=PortfolioHandoffConfig(),
    )

    ev = result.evidence_by_fold[0][0]
    assert ev.admitted_via_l1_edge_override is False


def test_handoff_4h_suffixless_sleeve_admission_byte_identical() -> None:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "4h", "strat_a"),)
    n_bars = 90
    rng = np.random.default_rng(42)
    rets = np.column_stack([rng.normal(0.001, 0.0005, n_bars)])

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1)), tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1)), funding_2d=np.zeros((n_bars, 1)), beta_1d=np.ones(1),
        expected_gross_bps_2d=np.ones((n_bars, 1)), expected_net_bps_2d=np.ones((n_bars, 1)),
        holding_bars_2d=np.ones((n_bars, 1)), side_2d=np.ones((n_bars, 1)),
        quality_weight_2d=np.ones((n_bars, 1)), signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (_ev_with_lcb("BTCUSDT", "strat_a", 0.9, 200.0),)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    result = evaluate_portfolio_handoff(
        registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
        net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
        config=PortfolioHandoffConfig(),
    )

    ev = result.evidence_by_fold[0][0]
    assert ev.admitted is True
    assert ev.key.symbol == "BTCUSDT"
    assert ev.key.strategy_id == "strat_a"
    assert ev.key.native_tf == "4h"


def test_handoff_admitted_tf_breakdown_logged_at_debug_level(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from src.domain.futures.strategy.tiered_workflow.dataclasses import L2SimulationCache
    from src.domain.futures.strategy.candidate_contracts import ValidatedSignalBatch
    from src.domain.futures.strategy.walk_forward import WFFold

    caplog.set_level(logging.DEBUG, logger="opt_main_futures")
    target_logger = logging.getLogger("opt_main_futures")
    prior_propagate = target_logger.propagate
    target_logger.propagate = True

    sleeve_keys = (SignalSleeveKey("BTCUSDT", "4h", "strat_a"),)
    n_bars = 90
    rng = np.random.default_rng(42)
    rets = np.column_stack([rng.normal(0.001, 0.0005, n_bars)])

    cache = L2SimulationCache(
        vol_matrix_2d=np.ones((n_bars, 1)), tradeable_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        hurdle_2d=np.zeros((n_bars, 1)), funding_2d=np.zeros((n_bars, 1)), beta_1d=np.ones(1),
        expected_gross_bps_2d=np.ones((n_bars, 1)), expected_net_bps_2d=np.ones((n_bars, 1)),
        holding_bars_2d=np.ones((n_bars, 1)), side_2d=np.ones((n_bars, 1)),
        quality_weight_2d=np.ones((n_bars, 1)), signal_mask_2d=np.ones((n_bars, 1), dtype=np.bool_),
        sleeve_to_sym=np.array([0], dtype=np.int64), sleeve_keys=sleeve_keys,
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (_ev_with_lcb("BTCUSDT", "strat_a", 0.9, 200.0),)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test-v1",
    )
    batch = ValidatedSignalBatch(
        events=(), start_idx=0, end_idx=n_bars, symbols=("BTCUSDT",),
        registry_version="test-v1", model_version="test",
    )
    fold = WFFold(fit_start=0, fit_end=30, cal_start=30, cal_end=60, oos_start=60, oos_end=90)

    try:
        evaluate_portfolio_handoff(
            registry=registry, signal_batch=batch, cache=cache, folds=(fold,),
            net_sleeve_returns_by_fold=(np.ascontiguousarray(rets, dtype=np.float32),),
            config=PortfolioHandoffConfig(),
        )
    finally:
        target_logger.propagate = prior_propagate

    debug_msgs = [r.message for r in caplog.records if "handoff_admitted_tf_breakdown" in r.message]
    assert len(debug_msgs) == 1
    assert "fold=0" in debug_msgs[0]
    assert "'4h': 1" in debug_msgs[0]
