"""Tests for adaptive t-statistic thresholding, MDES filter, and
activation floor boundary checks in signal selection.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    EdgeSource,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _candidate_output_to_signal_batch,
    _log_family_admission_diag,
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
)


def _make_cfg(**overrides: object) -> MagicMock:
    """Return a CandidateStrategyConfig-like mock stub."""
    cfg = MagicMock()
    defaults: dict[str, object] = {
        "l1_baseline_mode": "peer_exclusive",
        "l1_qualify_by_regime": False,
        "l1_pair_min_effective_obs": 5.0,
        "l1_pair_alpha": 0.05,
        "l1_pair_power": 0.80,
        "l1_pair_mdes_multiplier": 0.5,
        "l1_pair_min_folds": 1,
        "l1_pair_min_mean_gross_bps": 0.0,
        "l1_pair_min_incremental_bps": 0.0,
        "l1_pair_min_positive_fold_ratio": 0.0,
        "l1_pair_fdr_alpha": 1.0,
        "l1_bootstrap_block_bars": 1,
        "l1_bootstrap_samples": 10,
        "l1_lcb_quantile_base": 0.05,
        "l1_lcb_quantile_relaxed": 0.20,
        "l1_lcb_quantile_full_conf_blocks": 15,
        "l1_lcb_quantile_floor_blocks": 3,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_event_frame(
    gross_bps_list: list[float],
    symbol: str = "BTCUSDT",
    strategy_id: str = "trend:fast",
) -> pd.DataFrame:
    """Helper to construct an event DataFrame for evidence computation."""
    rows = [
        {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "side": 1,
            "holding_bucket": 4,
            "gross_event_bps": g,
            "incremental_bps": g,  # Will be recalculated but provides seed values
            "fold_id": 0,
            "uniqueness_weight": 1.0,
            "expected_holding_bars": 4,
        }
        for g in gross_bps_list
    ]
    df = pd.DataFrame(rows)
    return df


# ─── Scenario 1: Insufficient Effective Obs ───────────────────────────────


def test_insufficient_effective_obs_rejection() -> None:
    """If effective_n < l1_pair_min_effective_obs, reject with insufficient_effective_obs."""
    # Given: effective_obs = 3, but threshold is 5.0
    cfg = _make_cfg(l1_pair_min_effective_obs=5.0)
    df = _make_event_frame(gross_bps_list=[5.0, 4.0, 6.0])

    # When
    evidence = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    # Then
    assert len(evidence) == 1
    assert not evidence[0].qualified
    assert "insufficient_effective_obs" in evidence[0].rejection_reasons


# ─── Scenario: l1_pair_fdr_procedure wiring [ADR pending: L1_REGISTRY_ADMISSION_RECALIBRATION Phase B] ──


def test_compute_symbol_strategy_evidence_forwards_bh_procedure_to_by_q_values(mocker: MockerFixture) -> None:
    """Integration (Scenario 4): cfg.l1_pair_fdr_procedure='bh' is forwarded as
    harmonic_override=1.0 into _by_q_values; default 'by' forwards None
    (zero behavior change)."""
    import src.domain.futures.strategy.tiered_workflow.signal_selection as sel_mod

    spy = mocker.spy(sel_mod, "_by_q_values")
    df = _make_event_frame(gross_bps_list=[5.0, 4.0, 6.0, 5.5, 4.5])

    cfg_bh = _make_cfg(l1_pair_fdr_procedure="bh")
    compute_symbol_strategy_evidence(event_results=df, cfg=cfg_bh, seed=0, registry_as_of_idx=999)
    assert spy.call_args.kwargs["harmonic_override"] == 1.0

    cfg_by = _make_cfg(l1_pair_fdr_procedure="by")
    compute_symbol_strategy_evidence(event_results=df, cfg=cfg_by, seed=0, registry_as_of_idx=999)
    assert spy.call_args.kwargs["harmonic_override"] is None


# ─── Scenario 2: Adaptive t-statistic and MDES Filtering ──────────────────


def test_adaptive_tstat_and_mdes_filtering() -> None:
    """Verify adaptive t-threshold and MDES filtering."""
    # N=10, df=9. For alpha=0.05, t_crit is approx 1.833.
    # Mean = 0.15, Std = 0.0527. t_stat = 0.15 / (0.0527 / sqrt(10)) = 9.0 (Far above 1.833).
    gross_values = [0.1, 0.2] * 5
    df = _make_event_frame(gross_bps_list=gross_values)

    # Case A: MDES multiplier is small (0.1), should pass since mean (0.15) > mdes_bps * multiplier
    cfg_pass = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_alpha=0.05,
        l1_pair_power=0.80,
        l1_pair_mdes_multiplier=0.1,
    )
    evidence_pass = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg_pass,
        seed=0,
        registry_as_of_idx=999,
    )
    assert len(evidence_pass) == 1
    assert "insufficient_effect_size" not in evidence_pass[0].rejection_reasons
    assert "weak_tstat" not in evidence_pass[0].rejection_reasons

    # Case B: MDES multiplier is large (10.0), should fail with insufficient_effect_size
    cfg_fail = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_alpha=0.05,
        l1_pair_power=0.80,
        l1_pair_mdes_multiplier=10.0,
    )
    evidence_fail = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg_fail,
        seed=0,
        registry_as_of_idx=999,
    )
    assert len(evidence_fail) == 1
    assert "insufficient_effect_size" in evidence_fail[0].rejection_reasons
    assert evidence_fail[0].qualified


# ─── Scenario 3 & 4: Activation Floor Boundary Condition and Zero-Prediction Fallback ─


def test_candidate_output_to_signal_batch_zero_bps_retained() -> None:
    """Verify that pred == 0.0 is not discarded when activation_floor_bps == 0.0 (Strict comparison <)."""
    # Arrange
    events = pd.DataFrame(
        [
            {
                "entry_idx": 1,
                "symbol": "BTCUSDT",
                "family": "trend",
                "variant": "fast",
                "entry_regime": "bull",
                "side": 1,
                "expected_holding_bars": 3,
            }
        ]
    )
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.asarray([1.0], dtype=np.float64),
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_gross_bps=np.asarray([0.0], dtype=np.float64),  # 0.0 prediction
        q10_gross_bps=np.asarray([-2.0], dtype=np.float64),
        q90_gross_bps=np.asarray([2.0], dtype=np.float64),
    )
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=5.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.1,
        p_value=0.02,
        q_value=0.03,
        positive_fold_ratio=1.0,
        n_obs=12,
        effective_n=12.0,
        n_folds=3,
        reliability=0.8,
        qualified=True,
        rejection_reasons=(),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (evidence,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="deployment",
    )
    datetimes = np.array(
        [
            np.datetime64("2026-06-13T00:00:00"),
            np.datetime64("2026-06-13T04:00:00"),
            np.datetime64("2026-06-13T08:00:00"),
        ],
        dtype="datetime64[ns]",
    )

    # Act: activation_floor_bps = 0.0. Under BEFORE(<=), pred=0.0 is dropped. Under AFTER(<), it must be retained.
    batch = _candidate_output_to_signal_batch(
        model_output=model_output,
        registry=registry,
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        model_version="m1",
        activation_floor_bps=0.0,
    )

    # Assert
    assert len(batch.events) == 1
    assert batch.events[0].symbol == "BTCUSDT"


def test_build_qualified_signal_registry_prefers_quality_weight_over_input_order() -> None:
    strong = SimpleNamespace(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=4.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.5,
        p_value=0.01,
        q_value=0.02,
        positive_fold_ratio=1.0,
        n_obs=10,
        effective_n=10.0,
        n_folds=3,
        reliability=0.7,
        quality_weight=0.95,
        qualified=True,
        rejection_reasons=(),
    )
    weak = SimpleNamespace(
        key=SignalSourceKey("BTCUSDT", "trend:slow", "bull"),
        mean_gross_bps=4.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.5,
        p_value=0.01,
        q_value=0.02,
        positive_fold_ratio=1.0,
        n_obs=10,
        effective_n=10.0,
        n_folds=3,
        reliability=0.7,
        quality_weight=0.45,
        qualified=True,
        rejection_reasons=(),
    )

    builder: Any = build_qualified_signal_registry
    registry: Any = builder(
        evidence=(weak, strong),
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="v1",
    )

    assert registry.by_symbol["BTCUSDT"][0].quality_weight == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# P3: Adaptive Evidence Gate (snapshot_index)
# ---------------------------------------------------------------------------


def _make_event_frame_3events() -> pd.DataFrame:
    """Return a 3-row event DataFrame with 1 fold, effective_n ≈ 3.0.

    Uses the same columns as the existing `_make_event_frame` helper (no exit_idx
    so maturity filter is skipped). entry_idx is included for bootstrap grouping.
    """
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "family": ["trend_ma"] * 3,
            "variant": ["ema_12_72"] * 3,
            "strategy_id": ["trend_ma:ema_12_72"] * 3,
            "activation_context": ["all"] * 3,
            "fold_id": [0] * 3,
            "gross_event_bps": [5.0, 6.0, 4.0],
            "side": [1] * 3,
            "expected_holding_bars": [24] * 3,
            "uniqueness_weight": [1.0] * 3,
            "entry_idx": [0, 1, 2],
        }
    )


def test_adaptive_gate_early_snapshot_relaxed() -> None:
    """P3.1: Early snapshot (index=1 < early_snapshots=2) uses relaxed thresholds."""
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,  # strict
        l1_pair_min_folds=2,  # strict
        l1_evidence_early_snapshots=2,
        l1_pair_min_effective_obs_early=2.0,  # relaxed
        l1_pair_min_folds_early=1,  # relaxed
        l1_pair_min_mean_gross_bps=0.0,
        l1_pair_min_incremental_bps=0.0,
        l1_quality_weight_enabled=False,  # simplify: quality_weight controlled by hard_eligible only
    )
    df = _make_event_frame_3events()

    evidence = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=1,
    )
    # With relaxed gates (eff_obs>=2.0, folds>=1), all 3 events pass
    assert len(evidence) == 1
    assert evidence[0].hard_eligible, "early snapshot with relaxed gates should be hard_eligible"


def test_adaptive_gate_late_snapshot_strict() -> None:
    """P3.2: Late snapshot (index=3 >= early_snapshots=2) uses strict thresholds."""
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,  # strict — 3 < 5
        l1_pair_min_folds=2,  # strict — 1 < 2
        l1_evidence_early_snapshots=2,
        l1_pair_min_effective_obs_early=2.0,
        l1_pair_min_folds_early=1,
    )
    df = _make_event_frame_3events()

    evidence = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=3,
    )
    assert len(evidence) == 1
    assert not evidence[0].hard_eligible
    # Should fail on both insufficient_effective_obs (3<5) AND insufficient_folds (1<2)
    reasons = set(evidence[0].structural_reasons)
    assert "insufficient_effective_obs" in reasons
    assert "insufficient_folds" in reasons


def test_adaptive_gate_disabled_always_strict() -> None:
    """P3.3: l1_evidence_early_snapshots=0 → strict gates regardless of snapshot_index."""
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_min_folds=2,
        l1_evidence_early_snapshots=0,  # disabled
        l1_pair_min_effective_obs_early=2.0,  # would be relaxed but never used
        l1_pair_min_folds_early=1,
    )
    df = _make_event_frame_3events()

    # Even with snapshot_index=0, fails because early_snapshots=0 means never relaxed
    evidence_early = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=0,
    )
    assert len(evidence_early) == 1
    assert not evidence_early[0].hard_eligible

    # snapshot_index=1 also fails
    evidence_late = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
        snapshot_index=1,
    )
    assert not evidence_late[0].hard_eligible


def _make_evidence(
    family: str,
    n_obs: int,
    effective_n: float,
    structural_reasons: tuple[str, ...],
    variant: str = "v1",
    symbol: str = "BTCUSDT",
) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(symbol=symbol, strategy_id=f"{family}:{variant}", activation_context="all"),
        mean_gross_bps=10.0,
        mean_incremental_bps=5.0,
        block_tstat_incremental=2.0,
        probability_positive=0.6,
        p_value=0.03,
        q_value=0.05,
        positive_fold_ratio=0.8,
        n_obs=n_obs,
        effective_n=effective_n,
        n_folds=3,
        quality_weight=0.5,
        hard_eligible=not structural_reasons,
        structural_reasons=structural_reasons,
    )


# ─── Family Admission Diagnostic Logging ─────────────────────────────


def test_family_admission_diag_groups_by_family(caplog: pytest.LogCaptureFixture) -> None:
    """Scenario 1 (Happy Path) & Scenario 2 (Edge Cases): families aggregate correctly."""
    caplog.set_level(logging.DEBUG)
    evidence = (
        _make_evidence(
            family="dual_momentum", n_obs=8000, effective_n=2.1,
            structural_reasons=("insufficient_effective_obs",),
        ),
        _make_evidence(
            family="dual_momentum", n_obs=6000, effective_n=1.8,
            structural_reasons=("insufficient_effective_obs",),
        ),
        _make_evidence(
            family="trend_ma", n_obs=500, effective_n=6.2,
            structural_reasons=(),
        ),
    )

    _log_family_admission_diag(evidence)

    assert "family=dual_momentum" in caplog.text
    assert "n_pairs=2" in caplog.text
    assert "top_reasons=insufficient_effective_obs=2" in caplog.text
    assert "family=trend_ma" in caplog.text
    assert "none(all_hard_eligible)" in caplog.text


def test_family_admission_diag_empty_evidence_skips_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Scenario 3 (Error Handling): empty evidence → no exception, no log output."""
    caplog.set_level(logging.DEBUG)

    _log_family_admission_diag(())

    assert caplog.text == ""


# ─── Adaptive LCB Quantile in compute_symbol_strategy_evidence (Tier 2 fix) ─


def test_compute_symbol_strategy_evidence_large_n_quantile_bit_identical() -> None:
    """S1: large N (effective_n=30, block_bars_eff=2 → num_blocks>=15) → quantile stays 0.05, bit-identical."""
    cfg = _make_cfg(l1_bootstrap_block_bars=1, l1_pair_min_effective_obs=4.0)
    df = _make_event_frame(gross_bps_list=[15.0] * 30)
    evidence = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=42, registry_as_of_idx=999,
    )
    assert len(evidence) == 1
    assert evidence[0].lcb_net_bps is not None
    assert np.isfinite(evidence[0].lcb_net_bps)


def test_compute_symbol_strategy_evidence_small_n_relaxes_lcb_net_quantile() -> None:
    """S2: small N (effective_n=4, block_bars_eff=6 → num_blocks<=3) → adaptive quantile raises lcb_net."""
    cfg = _make_cfg(
        l1_bootstrap_block_bars=6, l1_pair_min_effective_obs=4.0,
        l1_pair_min_mean_gross_bps=0.0, l1_pair_min_incremental_bps=0.0,
        l1_pair_min_positive_fold_ratio=0.0, l1_quality_weight_enabled=False,
    )
    df = _make_event_frame(gross_bps_list=[15.0, 12.0, 18.0, 10.0])

    baseline_cfg = _make_cfg(
        l1_bootstrap_block_bars=6, l1_pair_min_effective_obs=4.0,
        l1_pair_min_mean_gross_bps=0.0, l1_pair_min_incremental_bps=0.0,
        l1_pair_min_positive_fold_ratio=0.0, l1_quality_weight_enabled=False,
        l1_lcb_quantile_floor_blocks=0,
    )
    adaptive = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=42, registry_as_of_idx=999,
    )
    baseline = compute_symbol_strategy_evidence(
        event_results=df, cfg=baseline_cfg, seed=42, registry_as_of_idx=999,
    )

    assert adaptive[0].lcb_net_bps >= baseline[0].lcb_net_bps


# ─── L1 Baseline Family-Scoped Admission Integration (Scenario 4) ──────────


def test_compute_symbol_strategy_evidence_lone_family_member_not_washed_out_by_unrelated_peers() -> None:
    """Integration test: lone-family candidate not washed out under new default
    (peer_exclusive_family), but IS washed out under legacy peer_exclusive.

    The fixture models the measured 12h washout pattern:
    xs_momentum:v1 (lone family member, gross ~10bps) shares a bucket with
    trend_donchian:donchian_72 (unrelated family, gross ~25bps, higher
    positive returns). Under legacy peer_exclusive the peer mean drags
    xs_momentum's incremental below 0 → false no_incremental_edge rejection.
    Under peer_exclusive_family family-of-one → peer_count=0 → baseline=0
    → incremental == gross → no rejection.
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for g in rng.normal(10.0, 2.0, size=100).tolist():
        rows.append({
            "symbol": "BTCUSDT", "side": 1, "strategy_id": "xs_momentum:v1",
            "gross_event_bps": g, "expected_holding_bars": 4,
            "fold_id": (len(rows) // 20) % 5,
        })
    for g in rng.normal(25.0, 3.0, size=100).tolist():
        rows.append({
            "symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_donchian:donchian_72",
            "gross_event_bps": g, "expected_holding_bars": 4,
            "fold_id": (len(rows) // 20) % 5,
        })
    df = pd.DataFrame(rows)

    cfg_family = _make_cfg(l1_baseline_mode="peer_exclusive_family", l1_pair_min_incremental_bps=0.0,
                            l1_pair_min_mean_gross_bps=0.0, l1_pair_min_positive_fold_ratio=0.0,
                            l1_quality_weight_enabled=False, l1_pair_min_effective_obs=1.0,
                            l1_pair_min_folds=1)
    cfg_legacy = _make_cfg(l1_baseline_mode="peer_exclusive", l1_pair_min_incremental_bps=0.0,
                           l1_pair_min_mean_gross_bps=0.0, l1_pair_min_positive_fold_ratio=0.0,
                           l1_quality_weight_enabled=False, l1_pair_min_effective_obs=1.0,
                           l1_pair_min_folds=1)

    ev_family = compute_symbol_strategy_evidence(event_results=df, cfg=cfg_family, seed=42, registry_as_of_idx=999)
    ev_legacy = compute_symbol_strategy_evidence(event_results=df, cfg=cfg_legacy, seed=42, registry_as_of_idx=999)

    xs_family = next(e for e in ev_family if e.key.strategy_id == "xs_momentum:v1")
    xs_legacy = next(e for e in ev_legacy if e.key.strategy_id == "xs_momentum:v1")

    assert "no_incremental_edge" not in xs_family.structural_reasons
    assert xs_family.mean_incremental_bps > 0
    assert xs_legacy.mean_incremental_bps < 0  # washed out by unrelated peers
    assert "no_incremental_edge" in xs_legacy.structural_reasons


def test_compute_symbol_strategy_evidence_baseline_mode_override_takes_precedence_over_cfg() -> None:
    """[ADR pending: L1_BASELINE_FAMILY_SCOPED_ADMISSION regression fix]

    baseline_mode_override, when passed, must win over cfg.l1_baseline_mode --
    this is how the deployment call site applies family-scoped admission without
    touching walk-forward snapshot calls (which omit the override and therefore
    keep using cfg.l1_baseline_mode's plain 'peer_exclusive' default).
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for g in rng.normal(10.0, 2.0, size=100).tolist():
        rows.append({
            "symbol": "BTCUSDT", "side": 1, "strategy_id": "xs_momentum:v1",
            "gross_event_bps": g, "expected_holding_bars": 4,
            "fold_id": (len(rows) // 20) % 5,
        })
    for g in rng.normal(25.0, 3.0, size=100).tolist():
        rows.append({
            "symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_donchian:donchian_72",
            "gross_event_bps": g, "expected_holding_bars": 4,
            "fold_id": (len(rows) // 20) % 5,
        })
    df = pd.DataFrame(rows)

    # cfg default is now plain "peer_exclusive" (post-regression-fix) -- override must
    # still force family-scoped behavior for this call.
    cfg_default = _make_cfg(l1_baseline_mode="peer_exclusive", l1_pair_min_incremental_bps=0.0,
                             l1_pair_min_mean_gross_bps=0.0, l1_pair_min_positive_fold_ratio=0.0,
                             l1_quality_weight_enabled=False, l1_pair_min_effective_obs=1.0,
                             l1_pair_min_folds=1)

    ev_no_override = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg_default, seed=42, registry_as_of_idx=999,
    )
    ev_with_override = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg_default, seed=42, registry_as_of_idx=999,
        baseline_mode_override="peer_exclusive_family",
    )

    xs_no_override = next(e for e in ev_no_override if e.key.strategy_id == "xs_momentum:v1")
    xs_with_override = next(e for e in ev_with_override if e.key.strategy_id == "xs_momentum:v1")

    assert "no_incremental_edge" in xs_no_override.structural_reasons  # cfg default: washed out
    assert "no_incremental_edge" not in xs_with_override.structural_reasons  # override: rescued


# ─── FDR Hard-Reject Override Scenarios (L1_SNAPSHOT_FDR_DECOUPLING) ───────


def test_fdr_hard_reject_override_soft_scales_instead_of_zeroing() -> None:
    """Scenario 1 (Happy Path): fdr_hard_reject_override=False keeps qw positive
    (same as no-FDR baseline) instead of zeroing it when q_value > l1_pair_fdr_alpha.
    """
    rng = np.random.default_rng(1)
    rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in rng.normal(1.0, 8.0, size=4).tolist()
    ]
    df = pd.DataFrame(rows)

    cfg_no_fdr = _make_cfg(l1_pair_fdr_alpha=1.0, l1_fdr_hard_reject=True,
                            l1_pair_min_effective_obs=1.0, l1_pair_min_folds=1)
    cfg_hard = _make_cfg(l1_pair_fdr_alpha=0.15, l1_fdr_hard_reject=True,
                          l1_pair_min_effective_obs=1.0, l1_pair_min_folds=1)

    ev_no_fdr = compute_symbol_strategy_evidence(event_results=df, cfg=cfg_no_fdr, seed=1, registry_as_of_idx=999)
    ev_hard = compute_symbol_strategy_evidence(event_results=df, cfg=cfg_hard, seed=1, registry_as_of_idx=999)
    ev_soft = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg_hard, seed=1, registry_as_of_idx=999,
        fdr_hard_reject_override=False,
    )

    ev0_nofdr, ev0_hard, ev0_soft = ev_no_fdr[0], ev_hard[0], ev_soft[0]

    # Fixture must trigger FDR (q > alpha)
    assert ev0_nofdr.q_value > cfg_hard.l1_pair_fdr_alpha, (
        f"fixture q_value={ev0_nofdr.q_value:.4f} not > 0.15"
    )
    assert ev0_hard.quality_weight == 0.0  # hard reject zeros
    # Soft reject preserves qw (same scaling as no-FDR unconditional path)
    assert ev0_soft.quality_weight == pytest.approx(ev0_nofdr.quality_weight)


def test_fdr_hard_reject_override_none_is_bit_identical() -> None:
    """Scenario 2 (Edge, LIMIT-01): Explicit fdr_hard_reject_override=None produces
    bit-identical result to omitting the parameter entirely.
    """
    rng = np.random.default_rng(1)
    rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in rng.normal(1.0, 8.0, size=4).tolist()
    ]
    df = pd.DataFrame(rows)
    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_fdr_hard_reject=True,
                     l1_pair_min_effective_obs=1.0, l1_pair_min_folds=1)

    ev_no_param = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999,
    )
    ev_none = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999,
        fdr_hard_reject_override=None,
    )

    for a, b in zip(ev_no_param, ev_none, strict=True):
        assert a.quality_weight == b.quality_weight
        assert a.hard_eligible == b.hard_eligible
        assert a.structural_reasons == b.structural_reasons


def test_fdr_hard_reject_override_does_not_rescue_zero_raw_qw() -> None:
    """Scenario 3 (Boundary): fdr_hard_reject_override=False does NOT rescue a
    candidate whose raw quality_weight is already 0.0 (probability_positive <= 0.5).
    """
    rng = np.random.default_rng(2)
    rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in rng.normal(0.0, 10.0, size=4).tolist()
    ]
    df = pd.DataFrame(rows)
    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_fdr_hard_reject=True,
                     l1_pair_min_effective_obs=1.0, l1_pair_min_folds=1)

    # Verify raw qw is already 0.0 (no-FDR reference)
    ev_ref = compute_symbol_strategy_evidence(
        event_results=df, cfg=_make_cfg(l1_pair_fdr_alpha=1.0, l1_fdr_hard_reject=True,
                                         l1_pair_min_effective_obs=1.0, l1_pair_min_folds=1),
        seed=2, registry_as_of_idx=999,
    )
    assert ev_ref[0].quality_weight == 0.0, "fixture must have zero raw qw"

    ev_soft = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=2, registry_as_of_idx=999,
        fdr_hard_reject_override=False,
    )

    for ev in ev_soft:
        if ev.q_value > cfg.l1_pair_fdr_alpha:
            assert ev.quality_weight == 0.0, (
                f"symbol={ev.key.symbol} q_value={ev.q_value:.4f} qw={ev.quality_weight:.6f}"
            )


def test_fdr_hard_reject_override_unblocks_qualified_registry() -> None:
    """Scenario 4 (Integration): thin-early-fold fixture where hard-reject
    produces registry.ready_symbols == () and soft override produces non-empty.
    """
    rng = np.random.default_rng(1)
    rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in rng.normal(1.0, 8.0, size=4).tolist()
    ]
    df = pd.DataFrame(rows)

    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_fdr_hard_reject=True,
                     l1_pair_min_effective_obs=1.0, l1_pair_min_folds=1,
                     l1_breakeven_floor_bps=0.0)

    ev_hard = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999,
    )
    ev_soft = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999,
        fdr_hard_reject_override=False,
    )

    registry_hard = build_qualified_signal_registry(
        evidence=ev_hard,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )
    registry_soft = build_qualified_signal_registry(
        evidence=ev_soft,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )

    assert not registry_hard.ready_symbols, "fixture must produce empty hard registry"
    assert registry_soft.ready_symbols, (
        f"hard registry empty, soft also empty (n_evidence={len(ev_hard)})"
    )
    assert len(registry_soft.ready_symbols) >= len(registry_hard.ready_symbols)


# ─── FDR Hard-Eligible Scoping (L1_FDR_HARD_ELIGIBLE_SCOPING) ──────────────


def test_fdr_restricted_to_hard_eligible_reduces_m() -> None:
    """Scenario 1 (Happy Path): FDR correction restricted to hard_eligible subset
    produces q_values computed over m=hard_eligible count, not m=full pool count.
    """
    real_rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": i % 2}
        for i, g in enumerate([8.0, 9.0, 7.5, 8.5, 9.5, 7.0] * 2)
    ]
    padding_rows = [
        {"symbol": "ETHUSDT", "side": 1, "strategy_id": f"dead_family_{k}:v1",
         "gross_event_bps": 3.0, "expected_holding_bars": 4, "fold_id": 0}
        for k in range(8)
    ]
    df = pd.DataFrame(real_rows + padding_rows)
    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_pair_min_effective_obs=4.0, l1_pair_min_folds=1)

    evidence = compute_symbol_strategy_evidence(event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999)

    real_ev = next(e for e in evidence if e.key.symbol == "BTCUSDT")
    dead_evs = [e for e in evidence if e.key.symbol == "ETHUSDT"]
    assert real_ev.hard_eligible is True
    assert all(not e.hard_eligible for e in dead_evs)
    assert all(e.q_value == 1.0 for e in dead_evs)
    from src.domain.futures.strategy.tiered_workflow.signal_selection import _by_q_values
    expected_q = float(_by_q_values(np.asarray([real_ev.p_value]), harmonic_override=None)[0])
    assert real_ev.q_value == pytest.approx(expected_q)


def test_fdr_zero_hard_eligible_does_not_crash() -> None:
    """Scenario 2 (Edge, LIMIT-02): Zero hard_eligible candidates -> no crash,
    all q_values=1.0 sentinel, quality_weight=0.0.
    """
    rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in [5.0, 6.0, 7.0, 5.5, 6.5]
    ]
    df = pd.DataFrame(rows)
    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_pair_min_effective_obs=100.0, l1_pair_min_folds=1)

    evidence = compute_symbol_strategy_evidence(event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999)

    assert len(evidence) > 0
    for ev in evidence:
        assert not ev.hard_eligible
        assert ev.q_value == 1.0
        assert ev.quality_weight == 0.0


def test_fdr_non_hard_eligible_sentinel_qvalue_one() -> None:
    """Scenario 3 (Boundary, LIMIT-01): non-hard-eligible candidate with
    positive gross evidence gets q_value=1.0 sentinel, quality_weight=0.0,
    and is NOT admitted by build_qualified_signal_registry.
    """
    strong_rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:v1",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in [10.0, 12.0, 11.0, 9.0, 10.5] * 2
    ]
    weak_rows = [
        {"symbol": "ETHUSDT", "side": 1, "strategy_id": "dead:v2",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": 0}
        for g in [3.0, 2.5]
    ]
    df = pd.DataFrame(strong_rows + weak_rows)
    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_pair_min_effective_obs=5.0, l1_pair_min_folds=1,
                     l1_breakeven_floor_bps=0.0)

    evidence = compute_symbol_strategy_evidence(event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999)

    strong_ev = next(e for e in evidence if e.key.symbol == "BTCUSDT")
    weak_ev = next(e for e in evidence if e.key.symbol == "ETHUSDT")
    assert strong_ev.hard_eligible is True
    assert weak_ev.hard_eligible is False
    assert weak_ev.q_value == 1.0
    assert weak_ev.quality_weight == 0.0

    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT", "ETHUSDT"),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )
    assert "BTCUSDT" in registry.by_symbol
    assert "ETHUSDT" not in registry.by_symbol


def test_fdr_restricted_to_hard_eligible_unblocks_registry() -> None:
    """Scenario 4 (Integration): thin-pool fixture where unrestricted FDR
    (m=full pool) would produce empty registry, but restricted FDR (m=hard_eligible
    subset) unblocks at least one candidate.
    """
    real_rows = [
        {"symbol": "BTCUSDT", "side": 1, "strategy_id": "trend_ma:ema_12_72",
         "gross_event_bps": g, "expected_holding_bars": 4, "fold_id": i % 2}
        for i, g in enumerate([8.0, 9.0, 7.5, 8.5, 9.5, 7.0] * 2)
    ]
    padding_rows = [
        {"symbol": "ETHUSDT", "side": 1, "strategy_id": f"dead_family_{k}:v1",
         "gross_event_bps": 3.0, "expected_holding_bars": 4, "fold_id": 0}
        for k in range(8)
    ]
    df = pd.DataFrame(real_rows + padding_rows)

    # Compute old unrestricted-FDR q-values by calling _by_q_values on ALL p-values
    cfg = _make_cfg(l1_pair_fdr_alpha=0.15, l1_pair_min_effective_obs=4.0, l1_pair_min_folds=1,
                     l1_breakeven_floor_bps=0.0)
    evidence = compute_symbol_strategy_evidence(event_results=df, cfg=cfg, seed=1, registry_as_of_idx=999)

    old_p_values = np.asarray([e.p_value for e in evidence], dtype=np.float64)
    from src.domain.futures.strategy.tiered_workflow.signal_selection import _by_q_values
    old_q_all = _by_q_values(old_p_values, harmonic_override=None)

    old_evidence_overrides = []
    for ev, old_q in zip(evidence, old_q_all, strict=True):
        old_evidence_overrides.append(SymbolStrategyEvidence(
            key=ev.key,
            mean_gross_bps=ev.mean_gross_bps,
            mean_incremental_bps=ev.mean_incremental_bps,
            block_tstat_incremental=ev.block_tstat_incremental,
            probability_positive=ev.probability_positive,
            p_value=ev.p_value,
            q_value=float(old_q),
            positive_fold_ratio=ev.positive_fold_ratio,
            n_obs=ev.n_obs,
            effective_n=ev.effective_n,
            n_folds=ev.n_folds,
            quality_weight=ev.quality_weight,
            hard_eligible=ev.hard_eligible,
            structural_reasons=ev.structural_reasons,
            diagnostic_flags=ev.diagnostic_flags,
            lcb_net_bps=ev.lcb_net_bps,
            adverse_regime_lcb_bps=ev.adverse_regime_lcb_bps,
            adverse_regime_n_obs=ev.adverse_regime_n_obs,
            adverse_regime_defended=ev.adverse_regime_defended,
        ))

    registry_old = build_qualified_signal_registry(
        evidence=tuple(old_evidence_overrides),
        symbols=("BTCUSDT", "ETHUSDT"),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )
    registry_new = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTCUSDT", "ETHUSDT"),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg,
    )

    if not registry_old.ready_symbols:
        assert registry_new.ready_symbols, (
            "old (unrestricted) registry empty, new (restricted) also empty"
        )
    assert len(registry_new.ready_symbols) >= len(registry_old.ready_symbols)
