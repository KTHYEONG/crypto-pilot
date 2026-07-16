"""TDD tests for Phase 0 - L1 per-symbol atomization diagnostic.

Scenarios:
  1a: Happy path — pooled gross reconstruction from 3 cells
  2a: [LIMIT-01] _compute_incremental_bps peer-exclusive convergence
  2b: [LIMIT-02] sign_flip_ratio vs weighted divergence
  2d: Empty evidence
  3a: Degenerate (zero n_obs)
  3b: Boundary (min_effective_obs=0.0)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.tiered_workflow.atomization_diagnostics import (
    diagnose_strategy_atomization,
)
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    _compute_incremental_bps,
)


def _make_evidence(
    *,
    symbol: str,
    strategy_id: str,
    n_obs: int,
    mean_gross_bps: float,
    effective_n: float | None = None,
) -> SymbolStrategyEvidence:
    return SymbolStrategyEvidence(
        key=SignalSourceKey(symbol=symbol, strategy_id=strategy_id, activation_context="all"),
        mean_gross_bps=mean_gross_bps,
        mean_incremental_bps=mean_gross_bps,
        block_tstat_incremental=0.0,
        probability_positive=0.5,
        p_value=0.5,
        q_value=0.5,
        positive_fold_ratio=0.5,
        n_obs=n_obs,
        effective_n=effective_n if effective_n is not None else float(n_obs),
        n_folds=2,
        quality_weight=0.0,
        hard_eligible=False,
        structural_reasons=("no_incremental_edge",) if mean_gross_bps <= 0 else (),
        diagnostic_flags=(),
        lcb_net_bps=mean_gross_bps,
    )


# ─── Scenario 1a: Happy path — pooled gross reconstruction ────────────────


def test_diagnose_strategy_atomization_reconstructs_pooled_gross_from_cells() -> None:
    evidence = (
        _make_evidence(symbol="BTCUSDT", strategy_id="tpc:v1", n_obs=500, mean_gross_bps=80.0),
        _make_evidence(symbol="ETHUSDT", strategy_id="tpc:v1", n_obs=300, mean_gross_bps=60.0),
        _make_evidence(symbol="DOGEUSDT", strategy_id="tpc:v1", n_obs=10, mean_gross_bps=-20.0),
    )

    reports = diagnose_strategy_atomization(evidence, min_effective_obs=5.0)

    assert len(reports) == 1
    rep = reports[0]
    assert rep.strategy_id == "tpc:v1"
    assert rep.pooled_mean_gross_bps == pytest.approx(71.358, rel=1e-3)
    assert rep.sign_flip_ratio == pytest.approx(1.0 / 3.0)
    assert rep.sign_flip_ratio_weighted < 0.02


# ─── Scenario 2a: [LIMIT-01] Peer-cannibalization convergence ────────────


def test_compute_incremental_bps_peer_exclusive_converges_to_zero() -> None:
    """3 strategies sharing same (symbol,side,holding_bucket) with equal
    mean gross → each strategy's mean incremental converges to 0 (peer-
    cannibalization mathematical property). [LIMIT-01]
    """
    rows: list[dict[str, object]] = [
        {
            "gross_event_bps": 100.0,
            "symbol": "BTCUSDT",
            "side": 1,
            "holding_bucket": 4,
            "strategy_id": sid,
        }
        for sid in ("strat_a", "strat_b", "strat_c")
        for _ in range(20)
    ]
    frame = pd.DataFrame(rows)
    inc = _compute_incremental_bps(frame, mode="peer_exclusive")
    means = frame.assign(inc=inc).groupby("strategy_id")["inc"].mean()
    for sid in ("strat_a", "strat_b", "strat_c"):
        assert means[sid] == pytest.approx(0.0, abs=0.1), f"{sid} mean_incremental not ~0"


# ─── Scenario 2b: [LIMIT-02] Sign-flip ratio divergence ─────────────────


def test_diagnose_strategy_atomization_sign_flip_weighted_diverges_from_raw() -> None:
    """1 large positive cell + 5 small negative cells → raw flip ratio (5/6)
    is high while weighted flip ratio (~15/1015≈0.015) is low. [LIMIT-02]
    """
    evidence = (
        _make_evidence(symbol="BTCUSDT", strategy_id="flip:v1", n_obs=1000, mean_gross_bps=50.0),
        _make_evidence(symbol="ETHUSDT", strategy_id="flip:v1", n_obs=3, mean_gross_bps=-10.0),
        _make_evidence(symbol="ADAUSDT", strategy_id="flip:v1", n_obs=3, mean_gross_bps=-10.0),
        _make_evidence(symbol="SOLUSDT", strategy_id="flip:v1", n_obs=3, mean_gross_bps=-10.0),
        _make_evidence(symbol="DOTUSDT", strategy_id="flip:v1", n_obs=3, mean_gross_bps=-10.0),
        _make_evidence(symbol="LINKUSDT", strategy_id="flip:v1", n_obs=3, mean_gross_bps=-10.0),
    )

    reports = diagnose_strategy_atomization(evidence, min_effective_obs=5.0)

    assert len(reports) == 1
    rep = reports[0]
    assert rep.sign_flip_ratio == pytest.approx(5.0 / 6.0)
    assert rep.sign_flip_ratio_weighted < 0.02


# ─── Scenario 2d: Empty evidence ────────────────────────────────────────


def test_diagnose_strategy_atomization_empty_evidence_returns_empty_tuple() -> None:
    reports = diagnose_strategy_atomization((), min_effective_obs=5.0)
    assert reports == ()


# ─── Scenario 3a: Degenerate (zero n_obs) ────────────────────────────────


def test_diagnose_strategy_atomization_degenerate_zero_nobs() -> None:
    """All cells have n_obs=0 for a strategy_id → pooled_mean_gross_bps=0.0,
    no ZeroDivisionError."""
    evidence = (
        _make_evidence(symbol="BTCUSDT", strategy_id="zero:v1", n_obs=0, mean_gross_bps=50.0),
        _make_evidence(symbol="ETHUSDT", strategy_id="zero:v1", n_obs=0, mean_gross_bps=30.0),
    )
    reports = diagnose_strategy_atomization(evidence, min_effective_obs=5.0)
    assert len(reports) == 1
    assert reports[0].pooled_mean_gross_bps == pytest.approx(0.0)


# ─── Scenario 3b: Boundary (min_effective_obs=0.0) ──────────────────────


def test_diagnose_strategy_atomization_zero_min_effective_obs() -> None:
    evidence = (
        _make_evidence(symbol="BTCUSDT", strategy_id="bnd:v1", n_obs=10, mean_gross_bps=10.0),
    )
    reports = diagnose_strategy_atomization(evidence, min_effective_obs=0.0)
    assert len(reports) == 1
    assert reports[0].n_cells_below_min_effective_obs == 0


# ─── L1 Baseline Family-Scoped Admission (l1-baseline-family-scoped-admission.md) ──


def _make_two_family_frame(
    *,
    family_a_gross: list[float],
    family_b_gross: list[float],
) -> pd.DataFrame:
    _base = {"symbol": "BTCUSDT", "side": 1, "holding_bucket": 4,
             "expected_holding_bars": 4, "fold_id": 0, "activation_context": "all"}
    rows = [
        {**_base, "strategy_id": "xs_momentum:v1", "family": "xs_momentum", "gross_event_bps": g}
        for g in family_a_gross
    ]
    rows.extend(
        {**_base, "strategy_id": "trend_donchian:donchian_72", "family": "trend_donchian", "gross_event_bps": g}
        for g in family_b_gross
    )
    return pd.DataFrame(rows)


def _make_same_family_frame(
    *,
    strat_a_gross: list[float],
    strat_b_gross: list[float],
) -> pd.DataFrame:
    _base = {"symbol": "BTCUSDT", "side": 1, "holding_bucket": 4,
             "expected_holding_bars": 4, "fold_id": 0, "activation_context": "all"}
    rows = [
        {**_base, "strategy_id": "trend_ma:ema_12_72", "family": "trend_ma", "gross_event_bps": g}
        for g in strat_a_gross
    ]
    rows.extend(
        {**_base, "strategy_id": "trend_ma:ema_6_36", "family": "trend_ma", "gross_event_bps": g}
        for g in strat_b_gross
    )
    return pd.DataFrame(rows)


# --- Scenario 1 (Happy Path): same-family variants produce identical
#     incremental_bps under peer_exclusive_family and legacy peer_exclusive.
def test_compute_incremental_bps_single_family_bucket_matches_legacy_peer_exclusive() -> None:
    df = _make_same_family_frame(
        strat_a_gross=[100.0] * 20,
        strat_b_gross=[80.0] * 20,
    )
    inc_family = _compute_incremental_bps(df, mode="peer_exclusive_family")
    inc_legacy = _compute_incremental_bps(df, mode="peer_exclusive")
    pd.testing.assert_series_equal(inc_family, inc_legacy)


# --- Scenario 2 (Edge Case, [LIMIT-01]/[LIMIT-02]): lone-family member
#     gets no peer subtraction under peer_exclusive_family (peer_count==0
#     fallback to absolute), but DOES get subtracted under legacy
#     peer_exclusive which includes unrelated-family peers.
def test_compute_incremental_bps_lone_family_member_falls_back_to_absolute_baseline() -> None:
    """Lone-family member gets no peer subtraction under family mode
    (peer_count==0 → absolute fallback). Legacy mode subtracts unrelated
    higher-return peer mean → incremental washed down.
    """
    df = _make_two_family_frame(
        family_a_gross=[15.0, 15.0, 15.0],
        family_b_gross=[30.0, 30.0, 30.0],
    )
    inc_family = _compute_incremental_bps(df, mode="peer_exclusive_family")
    inc_legacy = _compute_incremental_bps(df, mode="peer_exclusive")

    # lone family member (xs_momentum): family-scoped → peer_count==0 → baseline=0
    idx_a = df["strategy_id"] == "xs_momentum:v1"
    assert (inc_family[idx_a] == df.loc[idx_a, "gross_event_bps"]).all()
    # legacy: unrelated higher-return peers push baseline above gross → incremental < 0
    assert (inc_legacy[idx_a] < 0).all()


# --- Scenario 3 (Boundary, [LIMIT-02]): strategy_id without ':' separator
#     degrades family to full strategy_id (no crash).
def test_compute_incremental_bps_strategy_id_without_colon_uses_full_id_as_family() -> None:
    rows = [
        {"symbol": "BTCUSDT", "side": 1, "holding_bucket": 4,
         "strategy_id": "legacyname", "family": "legacyname",
         "gross_event_bps": 50.0},
    ]
    df = pd.DataFrame(rows)
    inc = _compute_incremental_bps(df, mode="peer_exclusive_family")
    assert not inc.empty
    assert inc.iloc[0] == 50.0  # peer_count==0 → baseline=0
