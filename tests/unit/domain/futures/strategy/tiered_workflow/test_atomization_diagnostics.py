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
    rows: list[dict] = [
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
