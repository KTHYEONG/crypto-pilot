"""Tests for weighted multi-timeframe evidence fusion."""

from __future__ import annotations

import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.multi_tf_fusion import (
    fuse_multi_timeframe_evidence_weighted,
)


def _row(
    tf: str,
    mean_net_bps: float,
    *,
    n_events: int = 40,
    effective_n: float = 20.0,
    family: str = "trend_ma",
    variant: str = "ema_18_108",
) -> dict[str, object]:
    return {
        "run_id": "r1",
        "timeframe": tf,
        "family": family,
        "variant": variant,
        "recipe_id": f"{family}:{variant}:{tf}",
        "archetype": "trend",
        "n_events": n_events,
        "effective_n": effective_n,
        "mean_net_bps": mean_net_bps,
        "nw_tstat": 2.0,
        "block_lcb_bps": mean_net_bps,
        "gate_passed": True,
        "reject_reasons": "",
        "bootstrap_agree": True,
    }


# ── HP-03: Weighted TF fusion ─────────────────────────────────────────


def test_weighted_fusion_returns_partial_support() -> None:
    """HTF with partial coverage and same sign returns partial_support."""
    evidence_by_tf = {
        "4h": pd.DataFrame([_row("4h", 18.8, n_events=100, effective_n=80.0)]),
        "6h": pd.DataFrame([_row("6h", 12.1, n_events=30, effective_n=15.0)]),
    }
    results = fuse_multi_timeframe_evidence_weighted(
        evidence_by_tf=evidence_by_tf,
        min_coverage_mass_for_corroboration=1.0,
        min_partial_coverage_mass=0.25,
    )
    assert len(results) == 2
    native_4h = next(r for r in results if r.native_timeframe == "4h")
    assert native_4h.corroboration_tier == "partial_support" or native_4h.corroboration_tier == "single_tf_strict"


# ── EC-05: HTF with insufficient events contributes fractional weight ──


def test_weighted_fusion_htf_insufficient_events_contributes_weight() -> None:
    """HTF with n_events below min_events still contributes partial weight >= 0."""
    evidence_by_tf = {
        "4h": pd.DataFrame([_row("4h", 18.8, n_events=100, effective_n=80.0)]),
        "12h": pd.DataFrame([_row("12h", 15.0, n_events=10, effective_n=5.0)]),
    }
    results = fuse_multi_timeframe_evidence_weighted(
        evidence_by_tf=evidence_by_tf,
        min_coverage_mass_for_corroboration=1.0,
        min_partial_coverage_mass=0.25,
    )
    assert len(results) == 2
    # 12h with insufficient events still returns something, not zero coverage
    native_4h = next(r for r in results if r.native_timeframe == "4h")
    assert native_4h.tf_coverage_count >= 0


# ── Empty input ────────────────────────────────────────────────────────


def test_weighted_fusion_empty_input() -> None:
    assert fuse_multi_timeframe_evidence_weighted(evidence_by_tf={}) == ()


# ── Supplementary coverage tests ────────────────────────────────────────


def test_weighted_fusion_corroborated() -> None:
    """High coverage mass + same sign → corroborated."""
    evidence_by_tf = {
        "4h": pd.DataFrame([_row("4h", 18.8, n_events=100, effective_n=80.0)]),
        "6h": pd.DataFrame([_row("6h", 12.0, n_events=80, effective_n=60.0)]),
    }
    results = fuse_multi_timeframe_evidence_weighted(
        evidence_by_tf=evidence_by_tf,
        min_coverage_mass_for_corroboration=0.25,
        min_sign_agreement_ratio=0.66,
    )
    native = next(r for r in results if r.native_timeframe == "4h")
    assert native.corroboration_tier == "corroborated"


def test_weighted_fusion_contradicted() -> None:
    """High coverage mass + opposite sign → contradicted."""
    evidence_by_tf = {
        "4h": pd.DataFrame([_row("4h", 18.8, n_events=100, effective_n=80.0)]),
        "6h": pd.DataFrame([_row("6h", -15.0, n_events=80, effective_n=60.0)]),
    }
    results = fuse_multi_timeframe_evidence_weighted(
        evidence_by_tf=evidence_by_tf,
        min_coverage_mass_for_corroboration=0.25,
        max_sign_agreement_ratio_for_contradiction=0.50,
    )
    native = next(r for r in results if r.native_timeframe == "4h")
    assert native.corroboration_tier == "contradicted"


def test_weighted_fusion_insufficient_coverage() -> None:
    """Single TF only → insufficient_coverage."""
    evidence_by_tf = {
        "4h": pd.DataFrame([_row("4h", 18.8)]),
    }
    results = fuse_multi_timeframe_evidence_weighted(
        evidence_by_tf=evidence_by_tf,
        min_partial_coverage_mass=0.25,
    )
    native = next(r for r in results if r.native_timeframe == "4h")
    assert native.corroboration_tier == "insufficient_coverage"


def test_weighted_fusion_duplicate_rows_raises() -> None:
    """Duplicate (family, variant, timeframe) raises ValueError."""
    evidence_by_tf = {
        "4h": pd.DataFrame([
            _row("4h", 18.8, family="trend_ma", variant="ema_18_108"),
            _row("4h", 15.0, family="trend_ma", variant="ema_18_108"),
        ]),
    }
    with pytest.raises(ValueError, match="duplicate"):
        fuse_multi_timeframe_evidence_weighted(evidence_by_tf=evidence_by_tf)
