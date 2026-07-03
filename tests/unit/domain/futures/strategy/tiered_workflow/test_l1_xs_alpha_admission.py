"""TDD tests for L1 XS Alpha Portfolio Admission (l1-xs-alpha-portfolio-admission.md).

Scenarios 1-5: resolve_xs_alpha_admission
Scenarios 6-8: compute_symbol_strategy_evidence xs_admission substitution
Scenario 9: compute_xs_factor_spread_diagnostics 10-tuple regression
Scenario 10: build_qualified_signal_registry gate-2 regression
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    XsAdmissionBasis,
    XsFactorSpreadDiagnostics,
    build_qualified_signal_registry,
    compute_symbol_strategy_evidence,
    compute_xs_factor_spread_diagnostics,
    resolve_xs_alpha_admission,
)


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock()
    defaults: dict[str, object] = {
        "l1_xs_alpha_admission_enabled": False,
        "l1_xs_admission_min_sharpe": 0.15,
        "l1_breakeven_floor_bps": 7.5,
        "l1_baseline_mode": "peer_exclusive",
        "l1_qualify_by_regime": False,
        "l1_pair_min_effective_obs": 5.0,
        "l1_pair_min_effective_obs_early": 3.0,
        "l1_pair_alpha": 0.05,
        "l1_pair_power": 0.80,
        "l1_pair_mdes_multiplier": 0.5,
        "l1_pair_min_folds": 1,
        "l1_pair_min_folds_early": 1,
        "l1_pair_min_mean_gross_bps": 0.0,
        "l1_pair_min_incremental_bps": 0.0,
        "l1_pair_min_positive_fold_ratio": 0.0,
        "l1_pair_fdr_alpha": 1.0,
        "l1_bootstrap_block_bars": 1,
        "l1_bootstrap_samples": 10,
        "l1_quality_weight_enabled": True,
        "l1_qw_floor": 0.0,
        "l1_fdr_hard_reject": False,
        "l1_evidence_early_snapshots": 0,
        "l1_evidence_lookback_bars": None,
        "l1_signal_activation_floor_bps": 0.0,
        "l1_conviction_metric": "prob_positive",
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _xs_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["strategy_id"] = df["family"] + ":" + df["variant"]
    return df


def _positive_factor_rows(
    n_bars: int, family: str, variant: str, archetype: str = "xs_alpha",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for b in range(n_bars):
        rows.append({
            "decision_idx": b,
            "symbol": "A",
            "family": family,
            "variant": variant,
            "archetype": archetype,
            "side": 1,
            "score_z": 1.0,
            "realized_side_adjusted_gross_bps": 40.0,
        })
        rows.append({
            "decision_idx": b,
            "symbol": "B",
            "family": family,
            "variant": variant,
            "archetype": archetype,
            "side": -1,
            "score_z": -1.0,
            "realized_side_adjusted_gross_bps": 35.0,
        })
    return rows


def _make_event_frame(
    target_gross_high: float = 10.0,
    target_gross_low: float = 5.0,
    *,
    symbol: str = "BTCUSDT",
    strategy_id: str = "xs_momentum:v1",
    n_obs: int = 30,
    n_folds: int = 3,
    peer_gross: float = 100.0,
) -> pd.DataFrame:
    """Event DataFrame for evidence tests under peer_exclusive baseline.

    Two strategies share the same (symbol, side, holding_bucket). The target
    strategy uses *target_gross_high* for fold-0 events and *target_gross_low*
    for the remaining folds. The peer strategy always uses *peer_gross*.

    This layout ensures some folds have positive incremental (fold 0 when
    target_gross_high >> peer_gross) while the overall mean stays negative
    (target_gross_low << peer_gross in most folds).
    """
    rows: list[dict[str, object]] = []
    for i in range(n_obs):
        fold = i % max(n_folds, 1)
        t_g = target_gross_high if fold == 0 else target_gross_low
        rows.append({
            "symbol": symbol,
            "strategy_id": strategy_id,
            "activation_context": "all",
            "side": 1,
            "holding_bucket": 4,
            "gross_event_bps": t_g,
            "fold_id": fold,
            "uniqueness_weight": 1.0,
            "expected_holding_bars": 4,
            "decision_idx": i,
        })
        rows.append({
            "symbol": symbol,
            "strategy_id": "xs_peer:high_gross",
            "activation_context": "all",
            "side": 1,
            "holding_bucket": 4,
            "gross_event_bps": peer_gross,
            "fold_id": fold,
            "uniqueness_weight": 1.0,
            "expected_holding_bars": 4,
            "decision_idx": i,
        })
    return pd.DataFrame(rows)


# ─── Scenario 1: resolve_xs_alpha_admission — Happy Path ──────────────────

def test_resolve_xs_alpha_admission_passes_when_lcb_and_sharpe_exceed_floor() -> None:
    diag = XsFactorSpreadDiagnostics(
        fold_id=0,
        by_factor={"xs_momentum:v1": (50, 500, 60.0, 40.0, 0.55, 25.0, -0.03, -2.0, 0.52, 0.97)},
    )
    cfg = _make_cfg(
        l1_xs_alpha_admission_enabled=True,
        l1_breakeven_floor_bps=7.5,
        l1_xs_admission_min_sharpe=0.15,
    )
    result = resolve_xs_alpha_admission(diag, cfg)
    assert "xs_momentum:v1" in result
    basis = result["xs_momentum:v1"]
    assert basis.mean_bps == pytest.approx(60.0)
    assert basis.lcb_bps == pytest.approx(25.0)
    assert basis.sharpe == pytest.approx(0.55)
    assert basis.probability_positive == pytest.approx(0.97)
    assert basis.n_bars == 50


# ─── Scenario 2: Feature flag off ─────────────────────────────────────────

def test_resolve_xs_alpha_admission_returns_empty_when_disabled() -> None:
    diag = XsFactorSpreadDiagnostics(
        fold_id=0,
        by_factor={"xs_momentum:v1": (50, 500, 60.0, 40.0, 0.55, 25.0, -0.03, -2.0, 0.52, 0.97)},
    )
    cfg = _make_cfg(l1_xs_alpha_admission_enabled=False)
    result = resolve_xs_alpha_admission(diag, cfg)
    assert result == {}


# ─── Scenario 3: LCB below breakeven floor ────────────────────────────────

def test_resolve_xs_alpha_admission_rejects_when_lcb_below_breakeven_floor() -> None:
    diag = XsFactorSpreadDiagnostics(
        fold_id=0,
        by_factor={"xs_flow:v1": (50, 500, 10.0, 40.0, 0.20, 5.0, 0.01, 0.5, 0.50, 0.80)},
    )
    cfg = _make_cfg(
        l1_xs_alpha_admission_enabled=True,
        l1_breakeven_floor_bps=7.5,
        l1_xs_admission_min_sharpe=0.15,
    )
    result = resolve_xs_alpha_admission(diag, cfg)
    assert result == {}


# ─── Scenario 4: Sharpe below minimum ─────────────────────────────────────

def test_resolve_xs_alpha_admission_rejects_when_sharpe_below_minimum() -> None:
    diag = XsFactorSpreadDiagnostics(
        fold_id=0,
        by_factor={"xs_carry:v1": (50, 500, 30.0, 40.0, 0.05, 20.0, 0.01, 0.5, 0.50, 0.70)},
    )
    cfg = _make_cfg(
        l1_xs_alpha_admission_enabled=True,
        l1_breakeven_floor_bps=7.5,
        l1_xs_admission_min_sharpe=0.15,
    )
    result = resolve_xs_alpha_admission(diag, cfg)
    assert result == {}


# ─── Scenario 5: None / empty diag ────────────────────────────────────────

def test_resolve_xs_alpha_admission_handles_none_and_empty_diag() -> None:
    cfg = _make_cfg(l1_xs_alpha_admission_enabled=True)
    assert resolve_xs_alpha_admission(None, cfg) == {}
    diag_empty = XsFactorSpreadDiagnostics(fold_id=0, by_factor={})
    assert resolve_xs_alpha_admission(diag_empty, cfg) == {}


# ─── Scenario 6: xs_admission substitution integration ────────────────────

def test_compute_symbol_strategy_evidence_xs_admission_substitutes_gate_inputs() -> None:
    """With xs_admission, factor-level values override pair-level negative incremental."""
    # fold 0: target_high=120 >> peer=100 → positive incremental → positive_fold_ratio > 0
    # overall mean_incremental stays negative (folds 1,2 have target_low=10 << peer=100)
    df = _make_event_frame(target_gross_high=120, target_gross_low=10, n_obs=30)
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_min_folds=1,
        l1_pair_min_incremental_bps=0.0,
    )
    admission_map = {
        "xs_momentum:v1": XsAdmissionBasis(
            mean_bps=54.0, lcb_bps=41.8, sharpe=0.47,
            probability_positive=0.9, n_bars=544,
        ),
    }
    evidence = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=0, registry_as_of_idx=999,
        xs_admission=admission_map,
    )
    evs = {e.key.strategy_id: e for e in evidence}
    assert "xs_momentum:v1" in evs
    ev = evs["xs_momentum:v1"]
    assert ev.hard_eligible is True
    assert "no_incremental_edge" not in ev.structural_reasons
    assert "negative_gross_edge" not in ev.structural_reasons
    assert ev.quality_weight > 0


# ─── Scenario 7: Regression — xs_admission=None keeps existing gate ───────

def test_compute_symbol_strategy_evidence_without_xs_admission_keeps_existing_gate() -> None:
    """Without xs_admission, pair-level negative incremental triggers no_incremental_edge."""
    df = _make_event_frame(target_gross_high=10, target_gross_low=5, n_obs=30)
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_min_folds=1,
        l1_pair_min_incremental_bps=0.0,
        l1_pair_min_mean_gross_bps=0.0,
    )
    evidence = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=0, registry_as_of_idx=999,
        xs_admission=None,
    )
    evs = {e.key.strategy_id: e for e in evidence}
    assert "xs_momentum:v1" in evs
    ev = evs["xs_momentum:v1"]
    assert ev.hard_eligible is False


# ─── Scenario 8: Sample size gate still blocks ────────────────────────────

def test_compute_symbol_strategy_evidence_xs_admission_does_not_bypass_sample_size_gate() -> None:
    """Even with xs_admission, insufficient effective_obs still blocks."""
    df = _make_event_frame(target_gross_high=10, target_gross_low=5, n_obs=3)
    cfg = _make_cfg(
        l1_pair_min_effective_obs=5.0,
        l1_pair_min_folds=1,
        l1_pair_min_incremental_bps=0.0,
    )
    admission_map = {
        "xs_momentum:v1": XsAdmissionBasis(
            mean_bps=54.0, lcb_bps=41.8, sharpe=0.47,
            probability_positive=0.9, n_bars=544,
        ),
    }
    evidence = compute_symbol_strategy_evidence(
        event_results=df, cfg=cfg, seed=0, registry_as_of_idx=999,
        xs_admission=admission_map,
    )
    evs = {e.key.strategy_id: e for e in evidence}
    assert "xs_momentum:v1" in evs
    ev = evs["xs_momentum:v1"]
    assert "insufficient_effective_obs" in ev.structural_reasons
    assert ev.hard_eligible is False


# ─── Scenario 9: compute_xs_factor_spread_diagnostics 10-tuple regression ─

def test_compute_xs_factor_spread_diagnostics_includes_probability_positive() -> None:
    rows = _positive_factor_rows(12, "xs_momentum", "xs_momentum_48")
    frame = _xs_frame(rows)
    cfg = _make_cfg()
    result = compute_xs_factor_spread_diagnostics(
        realized_event_results=frame, cfg=cfg, fold_id=0, seed=0,
    )
    assert result is not None
    vals = result.by_factor.get("xs_momentum:xs_momentum_48")
    assert vals is not None
    assert len(vals) == 10
    _nbars, _nevents, _mean, _std, _sharpe, _lcb, _ic, _ict, _lf, prob_positive = vals
    assert 0.0 <= prob_positive <= 1.0
    assert prob_positive > 0.5  # positive factor → high prob_positive


# ─── Scenario 10: build_qualified_signal_registry gate-2 regression ───────

def test_build_qualified_signal_registry_admits_xs_pair_via_substituted_lcb() -> None:
    """Gate-2 (lcb_net_bps > breakeven) must pass when xs_admission substitutes lcb."""
    df = _make_event_frame(target_gross_high=120, target_gross_low=10, n_obs=30)

    # Positive lcb after substitution → should be admitted by registry
    ev_pass = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=_make_cfg(l1_pair_min_effective_obs=5.0, l1_pair_min_folds=1, l1_pair_min_incremental_bps=0.0),
        seed=0, registry_as_of_idx=999,
        xs_admission={
            "xs_momentum:v1": XsAdmissionBasis(
                mean_bps=54.0, lcb_bps=41.8, sharpe=0.47,
                probability_positive=0.9, n_bars=544,
            ),
        },
    )
    cfg_reg = _make_cfg(l1_breakeven_floor_bps=7.5)
    registry = build_qualified_signal_registry(
        evidence=ev_pass,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=1,
        registry_version="test",
        cfg=cfg_reg,
    )
    assert "BTCUSDT" in registry.by_symbol
    assert any(e.key.strategy_id == "xs_momentum:v1" for e in registry.by_symbol["BTCUSDT"])

    # Without xs_admission → pair-level lcb is negative (incremental < 0 due to peer baseline)
    ev_fail = compute_symbol_strategy_evidence(
        event_results=df,
        cfg=_make_cfg(l1_pair_min_effective_obs=5.0, l1_pair_min_folds=1, l1_pair_min_incremental_bps=0.0),
        seed=0, registry_as_of_idx=999,
        xs_admission=None,
    )
    registry_fail = build_qualified_signal_registry(
        evidence=ev_fail,
        symbols=("BTCUSDT",),
        min_signals_per_symbol=2,  # require both strategies → low-lcb pair absent → symbol absent
        registry_version="test",
        cfg=cfg_reg,
    )
    assert "BTCUSDT" not in registry_fail.by_symbol
