from __future__ import annotations

import time

import pytest

import numpy as np

from src.domain.futures.compound.contracts import (
    CandidateTrial,
    CandidateTrialLedger,
    ExitPolicyKind,
    ExitPolicySpec,
    L1SleevePosterior,
    PortfolioAdmissionEvidence,
)


def _mask(n: int = 2) -> np.ndarray:
    m = np.zeros(n, dtype=np.bool_)
    m[0] = True
    return m


def test_exit_policy_spec_requires_calibration_hash() -> None:
    policy = ExitPolicySpec("time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    assert policy.kind is ExitPolicyKind.TIME
    with pytest.raises(ValueError, match="calibration_hash"):
        ExitPolicySpec("time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "")
    with pytest.raises(ValueError, match="multipliers"):
        ExitPolicySpec("bad", ExitPolicyKind.ASYMMETRIC_ATR, -1.0, 2.0, None, 0, 4, -1, "hash")
    with pytest.raises(ValueError, match="holding"):
        ExitPolicySpec("bad_hold", ExitPolicyKind.TIME, None, None, None, -1, 0, -1, "hash")


def _dummy_trial(candidate_hash: str = "ch1", cutoff: int = 1000) -> CandidateTrial:
    return CandidateTrial(
        candidate_hash=candidate_hash,
        strategy_spec_hash="spec1",
        descriptor_ids=("sig1",),
        risk_policy_hash="risk1",
        cutoff_time_ns=cutoff,
    )


class TestCandidateTrialLedger:
    def test_register_same_candidate_twice_is_idempotent(self, tmp_path) -> None:
        ledger = CandidateTrialLedger(tmp_path / "ledger.sqlite3")
        trial = _dummy_trial()
        assert ledger.register(trial) == 1
        assert ledger.register(trial) == 0
        assert ledger.distinct_count(cutoff_time_ns=1000) >= 1

    def test_load_trial_returns_drops_short_rows_and_truncates_to_common_length(self, tmp_path) -> None:
        ledger = CandidateTrialLedger(tmp_path / "load_test.sqlite3")
        rng = np.random.default_rng(42)
        now_ns = int(time.time_ns())
        for i in range(3):
            n_days = [400, 365, 20][i]
            rets = rng.normal(0.001, 0.01, n_days).astype(np.float64)
            trial = _dummy_trial(candidate_hash=f"ch{i}", cutoff=now_ns)
            ledger.register(trial, l2_daily_returns=rets)
        result = ledger.load_trial_returns(cutoff_time_ns=now_ns, min_days=30)
        assert result.shape == (2, 365)

    def test_distinct_count_floor(self, tmp_path) -> None:
        ledger = CandidateTrialLedger(tmp_path / "floor.sqlite3")
        count = ledger.distinct_count(cutoff_time_ns=9999, floor=27)
        assert count == 27


def test_l1_sleeve_posterior_rejects_invalid_range() -> None:
    policy = ExitPolicySpec("time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    with pytest.raises(ValueError, match="posterior range"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, _mask(), "h", policy, 0.0, 0.0, -1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="finite"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, _mask(), "h", policy, 0.0, float("nan"), 1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="member_hash is required"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, _mask(), "", policy, 0.0, 0.0, 1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="at least one True"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, np.zeros(2, dtype=bool), "h", policy, 0.0, 0.0, 1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="member_mask_1d must be 1-D"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, np.ones((1, 2), dtype=bool), "h", policy, 0.0, 0.0, 1.0, 0.5, 1.0, (), 0, False, ())


def test_portfolio_admission_evidence_handoff_scale_bounds() -> None:
    ev = PortfolioAdmissionEvidence(
        admitted=False, reasons=("posterior_0.793_below_0.9",),
        net_alpha_ann=0.1229, stressed_net_alpha_ann=0.0322,
        posterior_positive=0.793, positive_folds=4, n_folds=7,
        n_traded_bars=1260, handoff_scale=0.7325,
    )
    assert ev.handoff_scale > 0.0
    assert ev.handoff_scale < 1.0
