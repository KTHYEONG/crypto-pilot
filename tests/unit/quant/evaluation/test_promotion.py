from __future__ import annotations

import pytest

from src.quant.evaluation.promotion import (
    CandidateIdentity,
    PromotionResult,
    compose_promotion_verdict,
)
from src.quant.evaluation.reliability import FoldDistributionResult, ReliabilityGateResult


def _gate(verdict: str) -> ReliabilityGateResult:
    return ReliabilityGateResult(
        lcb90_cagr=0.20 if verdict == "PASS" else 0.04,
        lcb95_cagr=0.15 if verdict == "PASS" else 0.02,
        p_negative=0.0,
        point_cagr=0.25 if verdict == "PASS" else 0.10,
        t_stat=3.0 if verdict == "PASS" else 1.4,
        trade_count=0 if verdict == "PENDING" else 50,
        block_size_used=1,
        verdict=verdict,
    )


def _fold(gate_pass: bool) -> FoldDistributionResult:
    return FoldDistributionResult(
        n_folds=4, median_fold_cagr=0.05, worst_fold_cagr=0.01,
        median_fold_calmar=0.5,
        max_period_contribution=0.30 if gate_pass else 0.80,
        gate_pass=gate_pass,
    )


def _pass_set() -> tuple[ReliabilityGateResult, FoldDistributionResult, ReliabilityGateResult]:
    return _gate("PASS"), _fold(True), _gate("PASS")


class TestComposePromotionVerdict:
    def test_all_observation_components_pass_without_holdout_is_observation_pass(self) -> None:
        # SC-PROMO-01: observation+folds+stress pass, no holdout -> OBSERVATION_PASS
        observation, folds, stress = _pass_set()
        result = compose_promotion_verdict(observation, folds, stress, None)

        assert isinstance(result, PromotionResult)
        assert result.status == "OBSERVATION_PASS"
        assert result.observation_verdict == "PASS"
        assert result.fold_gate_pass is True
        assert result.stress_verdict == "PASS"
        assert result.holdout_verdict is None

    def test_observation_fail_rejects_regardless_of_fold_and_stress(self) -> None:
        # SC-PROMO-02: observation FAIL, fold true -> REJECTED
        observation = _gate("FAIL")
        result = compose_promotion_verdict(observation, _fold(True), _gate("PASS"), None)
        assert result.status == "REJECTED"

    def test_fold_false_rejects(self) -> None:
        # SC-PROMO-02: observation PASS, fold false -> REJECTED
        observation, _, stress = _pass_set()
        result = compose_promotion_verdict(observation, _fold(False), stress, None)
        assert result.status == "REJECTED"

    def test_stress_fail_rejects(self) -> None:
        # SC-PROMO-03: fold true, stress FAIL -> REJECTED
        observation, folds, _ = _pass_set()
        result = compose_promotion_verdict(observation, folds, _gate("FAIL"), None)
        assert result.status == "REJECTED"

    def test_all_observation_components_and_holdout_pass_is_holdout_pass(self) -> None:
        # SC-PROMO-04: observation+folds+stress pass and holdout PASS -> HOLDOUT_PASS
        observation, folds, stress = _pass_set()
        result = compose_promotion_verdict(observation, folds, stress, _gate("PASS"))
        assert result.status == "HOLDOUT_PASS"
        assert result.holdout_verdict == "PASS"

    def test_pending_or_missing_evidence_is_rejected(self) -> None:
        # SC-PROMO-05: any PENDING observation/fold/stress evidence -> REJECTED
        observation = _gate("PENDING")
        result = compose_promotion_verdict(observation, _fold(True), _gate("PASS"), None)
        assert result.status == "REJECTED"

        observation, _, _ = _pass_set()
        result = compose_promotion_verdict(observation, _fold(True), _gate("PENDING"), None)
        assert result.status == "REJECTED"

    def test_holdout_pending_or_absent_never_yields_holdout_pass(self) -> None:
        # SC-PROMO-05: PENDING/absent holdout can never produce a stronger status
        observation, folds, stress = _pass_set()
        for holdout in (None, _gate("PENDING"), _gate("FAIL")):
            result = compose_promotion_verdict(observation, folds, stress, holdout)
            assert result.status == "OBSERVATION_PASS"
            assert result.status != "HOLDOUT_PASS"

    def test_evidence_is_preserved_verbatim_in_result(self) -> None:
        observation, folds, stress = _pass_set()
        result = compose_promotion_verdict(observation, folds, stress, None)
        assert result.observation_verdict == observation.verdict
        assert result.fold_gate_pass == folds.gate_pass
        assert result.stress_verdict == stress.verdict
        assert result.candidate is None


class TestCandidateIdentity:
    def test_candidate_identity_rejects_missing_fields(self) -> None:
        valid = {
            "hypothesis_id": "hyp-001", "code_hash": "sha-abc",
            "parameters": {"period": 20}, "data_start": "2019-01-01",
            "data_end": "2025-12-31", "return_source": "breakout",
        }
        with pytest.raises(ValueError, match="hypothesis_id"):
            CandidateIdentity(**{**valid, "hypothesis_id": ""})
        with pytest.raises(ValueError, match="code_hash"):
            CandidateIdentity(**{**valid, "code_hash": ""})
        with pytest.raises(ValueError, match="return_source"):
            CandidateIdentity(**{**valid, "return_source": ""})

    def test_candidate_identity_accepts_valid(self) -> None:
        identity = CandidateIdentity(
            hypothesis_id="hyp-001", code_hash="sha-abc", parameters={"period": 20},
            data_start="2019-01-01", data_end="2025-12-31", return_source="breakout",
        )
        assert identity.hypothesis_id == "hyp-001"
        assert identity.parameters == {"period": 20}
