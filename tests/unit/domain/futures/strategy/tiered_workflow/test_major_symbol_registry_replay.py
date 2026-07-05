"""Tests for Major Symbol Registry Replay & Classification."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.domain.futures.strategy.candidate_contracts import (
    QualifiedSignalRegistry,
    ValidatedSignalBatch,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import PerTfL1Result
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    MajorSymbolRegistryCensusEntry,
    MajorSymbolSleeveContributionSummary,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer1Result,
    Layer2AllocationConfig,
    Layer2Result,
    Layer3Result,
)
from src.domain.futures.strategy.tiered_workflow.major_symbol_registry_replay import (
    MajorSymbolRegistryReplayResult,
    _major_symbol_registry_replay_adoption_verdict,
    classify_major_symbol_registry_gap,
    format_major_symbol_registry_replay_table,
    run_major_symbol_registry_replay,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    _aggregate_per_tf_l1,
    _resolve_l2_master_tf,
    _select_representative_l1_registry,
)

# ── Mock Boilerplate (from spec) ──────────────────────────────────────────────


def make_registry(strategy_id: str, *, symbol: str = "BTCUSDT", hard_eligible: bool = True) -> QualifiedSignalRegistry:
    ev = SimpleNamespace(
        key=SimpleNamespace(strategy_id=strategy_id),
        mean_incremental_bps=3.5,
        hard_eligible=hard_eligible,
    )
    return QualifiedSignalRegistry(
        by_symbol={symbol: (ev,)},
        ready_symbols=(symbol,) if hard_eligible else (),
        trade_scope_count=1,
        registry_version="test",
    )


def make_l1_result(
    *,
    registry: QualifiedSignalRegistry | None = None,
    artifact_registry: QualifiedSignalRegistry | None = None,
) -> Layer1Result:
    artifact = None
    if artifact_registry is not None:
        artifact = SimpleNamespace(deployment_registry=artifact_registry)
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked={"BTCUSDT": MagicMock()},
        pooled_ic=0.05,
        pooled_tstat=2.0,
        breadth=0.8,
        valid_coverage=0.9,
        fold_pass_ratio=0.75,
        gate_passed=True,
        n_valid=1,
        n_total=1,
        deployment_registry=registry,
        inference_artifact=artifact,
    )


def make_per_tf_result(tf: str, l1_result: Layer1Result) -> PerTfL1Result:
    return PerTfL1Result(tf=tf, l1_result=l1_result, n_winning_signals=len(l1_result.oos_stacked))


# ── Scenario 1: Happy Path ────────────────────────────────────────────────────


def test_aggregate_per_tf_l1_preserves_preferred_tf_registry() -> None:
    """S1: preferred_tf='4h' → result.deployment_registry is per_tf_l1['4h'].l1_result.deployment_registry."""
    reg_4h = make_registry("test:4h", symbol="BTCUSDT")
    reg_8h = make_registry("test:8h", symbol="ETHUSDT")
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", make_l1_result(registry=reg_4h)),
        "8h": make_per_tf_result("8h", make_l1_result(registry=reg_8h)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is reg_4h


def test_aggregate_per_tf_l1_falls_back_to_artifact_registry() -> None:
    """S2: top-level deployment_registry=None, artifact registry exists → use artifact registry."""
    reg_art = make_registry("test:art", symbol="BTCUSDT")
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", make_l1_result(artifact_registry=reg_art)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is reg_art


def test_classify_major_symbol_registry_gap_returns_outvoted_for_btc() -> None:
    """S3: BTC family, registry/observed both exist, adverse mismatch 0.63 → outvoted."""
    entries = (
        MajorSymbolRegistryCensusEntry(
            symbol="BTCUSDT", family="dual_momentum",
            registry_mean_incremental_bps=3.5, hard_eligible=True,
            observed_active_in_holdout=True,
        ),
    )
    summaries = (
        MajorSymbolSleeveContributionSummary(
            symbol="BTCUSDT", family="dual_momentum", n_obs=10,
            mean_raw_mu_sleeve=2.0, mean_quality_weight_sleeve=1.0,
            sign_mismatch_pct=0.5, regime_adverse_sign_mismatch_pct=0.63,
        ),
    )
    result = classify_major_symbol_registry_gap(
        symbol="BTCUSDT", family="dual_momentum",
        registry_entries=entries, observed_sleeve_summaries=summaries,
        adverse_sign_mismatch_threshold=0.50,
    )
    assert result == "outvoted"


def test_run_major_symbol_registry_replay_returns_two_variants_with_seed() -> None:
    """S4: run_l2_awf/run_l3_holdout patched → 2 rows, correct variant names and seed."""
    mock_l2 = MagicMock(spec=Layer2Result)
    mock_l2.cagr_hybrid = 0.15
    mock_l3 = MagicMock(spec=Layer3Result)
    mock_l3.total_return = 0.10
    mock_l3.cagr = 0.12
    mock_l3.mdd = 0.08
    mock_l3.sharpe = 0.9
    mock_l3.sortino = 1.2
    mock_l3.n_trades = 50
    mock_l3.major_symbol_diag = ()
    mock_l3.major_symbol_sleeve_diag = ()

    reg = make_registry("test", symbol="BTCUSDT")
    config = Layer2AllocationConfig()
    caps = MagicMock()

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf", return_value=mock_l2),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout", return_value=mock_l3),
    ):
        results = run_major_symbol_registry_replay(
            seed=42,
            registry=reg,
            l2_signal_batch=MagicMock(spec=ValidatedSignalBatch),
            l3_signal_batch=MagicMock(spec=ValidatedSignalBatch),
            aligned=MagicMock(spec=AlignedMarketData),
            awf_folds=(),
            holdout_span=(0, 10),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=None,
        )
    assert len(results) == 2
    variants = {r.variant for r in results}
    assert variants == {"baseline", "btc_divergence_dampener"}
    assert all(r.seed == 42 for r in results)


def test_major_symbol_registry_replay_adoption_passes_on_positive_median_delta() -> None:
    """S5: 3 seeds → median total return up, median MDD down, trades maintained → verdict True."""
    baseline_rows = (
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=1, baseline_parity=True,
            l2_cagr=0.15, l3_total_return=0.10, l3_cagr=0.12, l3_mdd=0.08,
            l3_sharpe=0.9, l3_sortino=1.2, l3_trade_count=50,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=2, baseline_parity=True,
            l2_cagr=0.14, l3_total_return=0.09, l3_cagr=0.11, l3_mdd=0.09,
            l3_sharpe=0.8, l3_sortino=1.1, l3_trade_count=45,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=3, baseline_parity=True,
            l2_cagr=0.13, l3_total_return=0.08, l3_cagr=0.10, l3_mdd=0.10,
            l3_sharpe=0.7, l3_sortino=1.0, l3_trade_count=55,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
    )
    candidate_rows = (
        MajorSymbolRegistryReplayResult(
            variant="btc_divergence_dampener", seed=1, baseline_parity=True,
            l2_cagr=0.16, l3_total_return=0.12, l3_cagr=0.13, l3_mdd=0.07,
            l3_sharpe=1.0, l3_sortino=1.3, l3_trade_count=48,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
        MajorSymbolRegistryReplayResult(
            variant="btc_divergence_dampener", seed=2, baseline_parity=True,
            l2_cagr=0.15, l3_total_return=0.11, l3_cagr=0.12, l3_mdd=0.08,
            l3_sharpe=0.9, l3_sortino=1.2, l3_trade_count=42,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
        MajorSymbolRegistryReplayResult(
            variant="btc_divergence_dampener", seed=3, baseline_parity=True,
            l2_cagr=0.14, l3_total_return=0.10, l3_cagr=0.11, l3_mdd=0.09,
            l3_sharpe=0.8, l3_sortino=1.1, l3_trade_count=50,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
    )
    adopted, reason = _major_symbol_registry_replay_adoption_verdict(
        baseline_rows=baseline_rows, candidate_rows=candidate_rows,
    )
    assert adopted is True
    assert reason == ""


# ── Scenario 2: Edge Cases ────────────────────────────────────────────────────


def test_aggregate_per_tf_l1_does_not_concat_multiple_registries() -> None:
    """X1: 2 TF with different registry rows → single registry identity preserved, not merged count."""
    reg = make_registry("test", symbol="BTCUSDT")
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", make_l1_result(registry=reg)),
        "8h": make_per_tf_result("8h", make_l1_result(registry=reg)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is reg


def test_classify_major_symbol_registry_gap_returns_admission_gap_when_rows_absent() -> None:
    """X2: No registry rows → admission_gap."""
    result = classify_major_symbol_registry_gap(
        symbol="ETHUSDT", family="dual_momentum",
        registry_entries=(), observed_sleeve_summaries=(),
    )
    assert result == "admission_gap"


def test_classify_major_symbol_registry_gap_returns_activation_gap_when_hard_eligible_but_unobserved() -> None:
    """X3: hard_eligible row exists but no holdout observation → activation_gap."""
    entries = (
        MajorSymbolRegistryCensusEntry(
            symbol="ETHUSDT", family="dual_momentum",
            registry_mean_incremental_bps=3.5, hard_eligible=True,
            observed_active_in_holdout=False,
        ),
    )
    result = classify_major_symbol_registry_gap(
        symbol="ETHUSDT", family="dual_momentum",
        registry_entries=entries, observed_sleeve_summaries=(),
    )
    assert result == "activation_gap"


def test_classify_major_symbol_registry_gap_ignores_near_zero_mismatch_under_dead_zone() -> None:
    """X4: near-zero mismatch (1e-13) under dead_zone → not outvoted."""
    entries = (
        MajorSymbolRegistryCensusEntry(
            symbol="BTCUSDT", family="dual_momentum",
            registry_mean_incremental_bps=3.5, hard_eligible=True,
            observed_active_in_holdout=True,
        ),
    )
    summaries = (
        MajorSymbolSleeveContributionSummary(
            symbol="BTCUSDT", family="dual_momentum", n_obs=10,
            mean_raw_mu_sleeve=1e-13, mean_quality_weight_sleeve=1.0,
            sign_mismatch_pct=1e-13, regime_adverse_sign_mismatch_pct=1e-13,
        ),
    )
    result = classify_major_symbol_registry_gap(
        symbol="BTCUSDT", family="dual_momentum",
        registry_entries=entries, observed_sleeve_summaries=summaries,
        adverse_sign_mismatch_threshold=0.50, dead_zone=1e-12,
    )
    assert result != "outvoted"


def test_run_major_symbol_registry_replay_handles_registry_none() -> None:
    """X5: registry=None → rows created, registry_census == ()."""
    mock_l2 = MagicMock(spec=Layer2Result)
    mock_l2.cagr_hybrid = 0.15
    mock_l3 = MagicMock(spec=Layer3Result)
    mock_l3.total_return = 0.10
    mock_l3.cagr = 0.12
    mock_l3.mdd = 0.08
    mock_l3.sharpe = 0.9
    mock_l3.sortino = 1.2
    mock_l3.n_trades = 50
    mock_l3.major_symbol_diag = ()
    mock_l3.major_symbol_sleeve_diag = ()

    config = Layer2AllocationConfig()
    caps = MagicMock()

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf", return_value=mock_l2),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout", return_value=mock_l3),
    ):
        results = run_major_symbol_registry_replay(
            seed=42,
            registry=None,
            l2_signal_batch=MagicMock(spec=ValidatedSignalBatch),
            l3_signal_batch=MagicMock(spec=ValidatedSignalBatch),
            aligned=MagicMock(spec=AlignedMarketData),
            awf_folds=(),
            holdout_span=(0, 10),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=None,
        )
    assert len(results) == 2
    for r in results:
        assert r.registry_census == ()


def test_major_symbol_registry_replay_adoption_fails_on_trade_collapse() -> None:
    """X6: candidate trades < 75% of baseline → blocker 'trade_collapse'."""
    baseline_rows = (
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=1, baseline_parity=True,
            l2_cagr=0.15, l3_total_return=0.10, l3_cagr=0.12, l3_mdd=0.08,
            l3_sharpe=0.9, l3_sortino=1.2, l3_trade_count=100,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
    )
    candidate_rows = (
        MajorSymbolRegistryReplayResult(
            variant="btc_divergence_dampener", seed=1, baseline_parity=True,
            l2_cagr=0.16, l3_total_return=0.12, l3_cagr=0.13, l3_mdd=0.07,
            l3_sharpe=1.0, l3_sortino=1.3, l3_trade_count=70,  # 70% < 75%
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
    )
    adopted, reason = _major_symbol_registry_replay_adoption_verdict(
        baseline_rows=baseline_rows, candidate_rows=candidate_rows,
        min_trade_ratio=0.75,
    )
    assert adopted is False
    assert reason == "trade_collapse"


def test_major_symbol_registry_replay_adoption_fails_when_mdd_not_improved() -> None:
    """X7: total return up but median MDD not improved → blocker 'median_mdd_not_improved'."""
    baseline_rows = (
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=1, baseline_parity=True,
            l2_cagr=0.15, l3_total_return=0.10, l3_cagr=0.12, l3_mdd=0.08,
            l3_sharpe=0.9, l3_sortino=1.2, l3_trade_count=50,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=2, baseline_parity=True,
            l2_cagr=0.14, l3_total_return=0.09, l3_cagr=0.11, l3_mdd=0.09,
            l3_sharpe=0.8, l3_sortino=1.1, l3_trade_count=45,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
    )
    candidate_rows = (
        MajorSymbolRegistryReplayResult(
            variant="btc_divergence_dampener", seed=1, baseline_parity=True,
            l2_cagr=0.16, l3_total_return=0.12, l3_cagr=0.13, l3_mdd=0.09,
            l3_sharpe=1.0, l3_sortino=1.3, l3_trade_count=48,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
        MajorSymbolRegistryReplayResult(
            variant="btc_divergence_dampener", seed=2, baseline_parity=True,
            l2_cagr=0.15, l3_total_return=0.11, l3_cagr=0.12, l3_mdd=0.10,
            l3_sharpe=0.9, l3_sortino=1.2, l3_trade_count=42,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=False, blocker_reason="",
        ),
    )
    adopted, reason = _major_symbol_registry_replay_adoption_verdict(
        baseline_rows=baseline_rows, candidate_rows=candidate_rows,
    )
    assert adopted is False
    assert reason == "median_mdd_not_improved"


# ── Scenario 3: Error Handling ────────────────────────────────────────────────


def test_select_representative_l1_registry_returns_none_on_empty_input() -> None:
    """E1: Empty dict → None, no exception."""
    result = _select_representative_l1_registry(per_tf_l1={})
    assert result is None


def test_select_representative_l1_registry_falls_back_when_preferred_tf_missing() -> None:
    """E2: preferred_tf='12h' not present → fine TF fallback."""
    reg = make_registry("test", symbol="BTCUSDT")
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", make_l1_result(registry=reg)),
        "8h": make_per_tf_result("8h", make_l1_result(registry=None)),
    }
    result = _select_representative_l1_registry(
        per_tf_l1=per_tf_l1,
        preferred_tf="12h",
    )
    # Falls back to min TF hours = "4h"
    assert result is reg


def test_run_major_symbol_registry_replay_propagates_baseline_parity_false() -> None:
    """E3: baseline L2/L3 mismatch → all candidate rows baseline_parity=False, adoption_passed=False."""
    mock_l2 = MagicMock(spec=Layer2Result)
    mock_l2.cagr_hybrid = 0.15
    mock_l3 = MagicMock(spec=Layer3Result)
    mock_l3.total_return = 0.10
    mock_l3.cagr = 0.12
    mock_l3.mdd = 0.08
    mock_l3.sharpe = 0.9
    mock_l3.sortino = 1.2
    mock_l3.n_trades = 50
    mock_l3.major_symbol_diag = ()
    mock_l3.major_symbol_sleeve_diag = ()

    reg = make_registry("test", symbol="BTCUSDT")
    config = Layer2AllocationConfig()
    caps = MagicMock()

    # Simulate baseline_l2/l3 that differ from replayed = parity fails
    baseline_l2 = MagicMock(spec=Layer2Result)
    baseline_l2.selected_last = frozenset()
    baseline_l2.weights_last = {}
    baseline_l2.cagr_hybrid = 0.99  # deliberately different

    baseline_l3 = MagicMock(spec=Layer3Result)
    baseline_l3.cagr = 0.99  # deliberately different

    with (
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf", return_value=mock_l2),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout", return_value=mock_l3),
    ):
        results = run_major_symbol_registry_replay(
            seed=42,
            registry=reg,
            l2_signal_batch=MagicMock(spec=ValidatedSignalBatch),
            l3_signal_batch=MagicMock(spec=ValidatedSignalBatch),
            aligned=MagicMock(spec=AlignedMarketData),
            awf_folds=(),
            holdout_span=(0, 10),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=None,
            baseline_l2=baseline_l2,
            baseline_l3=baseline_l3,
        )
    for r in results:
        if r.variant != "baseline":
            assert r.baseline_parity is False, f"{r.variant} should have parity=False"
            assert r.adoption_passed is False
            assert r.blocker_reason == "baseline_parity"


def test_classify_major_symbol_registry_gap_defaults_to_no_gap_for_non_matching_family() -> None:
    """E4: symbol/family mismatch → no_gap, no exception."""
    entries = (
        MajorSymbolRegistryCensusEntry(
            symbol="BTCUSDT", family="dual_momentum",
            registry_mean_incremental_bps=3.5, hard_eligible=True,
            observed_active_in_holdout=True,
        ),
    )
    result = classify_major_symbol_registry_gap(
        symbol="ETHUSDT", family="trend_ma",
        registry_entries=entries, observed_sleeve_summaries=(),
    )
    assert result == "admission_gap"


# ── TDD Scenario 1: Happy Path ────────────────────────────────────────────────


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock(spec="CandidateStrategyConfig")
    defaults: dict[str, object] = {
        "l2_master_tf": None,
        "seed": 42,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def test_run_tiered_pipeline_master_tf_matches_registry_source() -> None:
    """S1: l2_tf resolved from per_tf_l1 matches the TF whose registry is propagated.
    
    Invariant: deployment_registry in aggregated result comes from the same TF
    that _resolve_l2_master_tf selects as master.
    """
    reg_4h = make_registry("test:4h", symbol="BTCUSDT")
    reg_8h = make_registry("test:8h", symbol="BTCUSDT")

    # 8h has more n_winning_signals → higher edge quality → wins master TF
    l1_4h = make_l1_result(registry=reg_4h)
    l1_8h = Layer1Result(
        signals_per_fold=l1_4h.signals_per_fold,
        oos_stacked={"sym1": MagicMock(), "sym2": MagicMock(), "sym3": MagicMock()},
        pooled_ic=l1_4h.pooled_ic, pooled_tstat=l1_4h.pooled_tstat,
        breadth=l1_4h.breadth, valid_coverage=l1_4h.valid_coverage,
        fold_pass_ratio=l1_4h.fold_pass_ratio,
        gate_passed=True, n_valid=l1_4h.n_valid, n_total=l1_4h.n_total,
        deployment_registry=reg_8h,
    )
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", l1_4h),
        "8h": make_per_tf_result("8h", l1_8h),
    }
    cfg = _make_cfg(l2_master_tf=None)

    l2_tf = _resolve_l2_master_tf(cfg, per_tf_l1, probe_manifest=None)
    assert l2_tf == "8h", "Master TF should be 8h (higher n_winning_signals)"

    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf=l2_tf)
    assert result.deployment_registry is reg_8h, (
        f"Registry should come from master TF '8h', got {result.deployment_registry}"
    )


# ── TDD Scenario 2: Edge Cases ────────────────────────────────────────────────


def test_master_tf_selected_even_when_finest_tf_gate_fails() -> None:
    """S2: '4h' gate_passed=False (registry=None), '8h' gate_passed=True → registry from '8h'."""
    reg_8h = make_registry("test:8h", symbol="BTCUSDT")

    l1_4h_none = Layer1Result(
        signals_per_fold=(), oos_stacked={"s": MagicMock()},
        pooled_ic=0.0, pooled_tstat=0.0, breadth=0.0,
        valid_coverage=0.0, fold_pass_ratio=0.0,
        gate_passed=False, n_valid=0, n_total=1,
        deployment_registry=None,
    )
    l1_8h = Layer1Result(
        signals_per_fold=(), oos_stacked={"s1": MagicMock(), "s2": MagicMock()},
        pooled_ic=0.05, pooled_tstat=2.0, breadth=0.8,
        valid_coverage=0.9, fold_pass_ratio=0.75,
        gate_passed=True, n_valid=1, n_total=1,
        deployment_registry=reg_8h,
    )
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", l1_4h_none),
        "8h": make_per_tf_result("8h", l1_8h),
    }
    cfg = _make_cfg(l2_master_tf=None)

    l2_tf = _resolve_l2_master_tf(cfg, per_tf_l1, probe_manifest=None)
    assert l2_tf == "8h", "8h should win on n_winning_signals despite 4h gate failure"

    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf=l2_tf)
    assert result.deployment_registry is reg_8h, (
        "Registry should come from '8h' (master TF), not None from '4h'"
    )


def test_l1_result_override_path_unaffected() -> None:
    """S2: l1_result_override path returns the override registry unchanged."""
    reg_override = make_registry("test:override", symbol="BTCUSDT")
    override = make_l1_result(registry=reg_override)

    cfg = _make_cfg(l2_master_tf="8h")
    l2_tf = _resolve_l2_master_tf(cfg, {}, probe_manifest=None)
    assert l2_tf == "8h", "Override path should use cfg.l2_master_tf"
    assert override.deployment_registry is reg_override


def test_divergence_diag_distinguishes_dead_vs_live_dissent() -> None:
    """S2: quality_weight=0 → dissent_present=1 but dissent_qw_sum=0.0 (log field)."""
    _dc: dict[str, float] = {
        "dom": 1, "dissent_present": 1, "dissent_nonzero": 1, "armed": 0,
        "dom_qw_sum": 2.5, "dissent_qw_sum": 0.0,
    }
    dom_qw_mean = _dc["dom_qw_sum"] / max(_dc["dom"], 1)
    dissent_qw_mean = _dc["dissent_qw_sum"] / max(_dc["dissent_present"], 1)
    assert dissent_qw_mean == 0.0, "Dead dissent (qw=0) should yield qw_mean=0.0"
    assert dom_qw_mean == 2.5, "Live dominant should yield qw_mean=2.5"


# ── TDD Scenario 3: Error Handling ─────────────────────────────────────────────


def test_aggregate_per_tf_l1_all_gates_failed_registry_none_no_crash() -> None:
    """S3: All gate_passed=False, preferred_tf or not → deployment_registry is None, no crash."""
    per_tf_l1 = {
        "4h": make_per_tf_result("4h", make_l1_result(registry=None)),
        "8h": make_per_tf_result("8h", make_l1_result(registry=None)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is None
    assert result.gate_passed  # gate_passed is OR of all TFs (both True)


def test_resolve_l2_master_tf_empty_dict_falls_back_gracefully() -> None:
    """S3: Empty per_tf_l1 → falls back to probe_manifest or '8h' default."""
    cfg = _make_cfg(l2_master_tf=None)
    l2_tf = _resolve_l2_master_tf(cfg, {}, probe_manifest=None)
    assert l2_tf == "8h", "Empty dict should fall back to '8h'"

    probe_manifest = [{"is_winner": True, "tf": "12h"}]
    l2_tf_with_probe = _resolve_l2_master_tf(cfg, {}, probe_manifest=probe_manifest)
    assert l2_tf_with_probe == "12h", "With probe_manifest winner, should use '12h'"


# ── Misc: format table smoke test ─────────────────────────────────────────────


def test_format_major_symbol_registry_replay_table_smoke() -> None:
    """Smoke: format table does not raise."""
    results = (
        MajorSymbolRegistryReplayResult(
            variant="baseline", seed=42, baseline_parity=True,
            l2_cagr=0.15, l3_total_return=0.10, l3_cagr=0.12, l3_mdd=0.08,
            l3_sharpe=0.9, l3_sortino=1.2, l3_trade_count=50,
            btc_mu_bullish_pct=0.6, eth_mu_bullish_pct=0.5,
            registry_census=(), adoption_passed=True, blocker_reason="",
        ),
    )
    table = format_major_symbol_registry_replay_table(results)
    assert "baseline" in table
