"""Tests for L1 causal feedback contract: resolve_l1_feedback_multiplier."""

from __future__ import annotations

import pytest

from src.domain.futures.alpha_foundry.contracts import (
    CausalFeedbackError,
    L1CausalFeedback,
    SignalHypothesisKey,
    resolve_l1_feedback_multiplier,
)


class TestCausalFeedbackStrictlyPrior:
    """[LIMIT-01][LIMIT-03] Feedback must be strictly prior — equality/future raises."""

    def test_equality_raises_causal_feedback_error(self) -> None:
        feedback = L1CausalFeedback(
            key=SignalHypothesisKey("trend_ma", "ema_12_72", "4h"),
            outcome="deployable",
            evidence_end_ns=200,
            effective_n=20.0,
            survival_successes=3,
            survival_trials=4,
            pooled_net_lcb_bps=12.0,
            positive_fold_ratio=0.75,
        )
        with pytest.raises(CausalFeedbackError, match="evidence_end_ns"):
            resolve_l1_feedback_multiplier(
                feedback=feedback,
                current_evidence_start_ns=200,
            )

    def test_future_feedback_raises(self) -> None:
        feedback = L1CausalFeedback(
            key=SignalHypothesisKey("trend_ma", "ema_12_72", "4h"),
            outcome="deployable",
            evidence_end_ns=300,
            effective_n=20.0,
            survival_successes=3,
            survival_trials=4,
            pooled_net_lcb_bps=12.0,
            positive_fold_ratio=0.75,
        )
        with pytest.raises(CausalFeedbackError):
            resolve_l1_feedback_multiplier(
                feedback=feedback,
                current_evidence_start_ns=200,
            )


class TestFeedbackMultiplier:
    """S1.2: causal feedback with valid prior"""

    def test_prior_survival_boosts_multiplier(self) -> None:
        """Strong survival (3/4) with high effective_n → multiplier in (1.0, 1.5]"""
        feedback = L1CausalFeedback(
            key=SignalHypothesisKey("trend_ma", "ema_12_72", "4h"),
            outcome="deployable",
            evidence_end_ns=100,
            effective_n=60.0,
            survival_successes=3,
            survival_trials=4,
            pooled_net_lcb_bps=12.0,
            positive_fold_ratio=0.75,
        )
        mult = resolve_l1_feedback_multiplier(
            feedback=feedback,
            current_evidence_start_ns=200,
        )
        assert 1.0 < mult <= 1.5

    def test_missing_feedback_returns_neutral(self) -> None:
        mult = resolve_l1_feedback_multiplier(
            feedback=None,
            current_evidence_start_ns=200,
        )
        assert mult == 1.0

    def test_zero_effective_n_fallback(self) -> None:
        """[LIMIT-01] Zero effective_n returns neutral multiplier"""
        feedback = L1CausalFeedback(
            key=SignalHypothesisKey("trend_ma", "ema_12_72", "4h"),
            outcome="net_edge_negative",
            evidence_end_ns=100,
            effective_n=0.0,
            survival_successes=0,
            survival_trials=1,
            pooled_net_lcb_bps=-15.0,
            positive_fold_ratio=0.25,
        )
        mult = resolve_l1_feedback_multiplier(
            feedback=feedback,
            current_evidence_start_ns=200,
        )
        assert mult == 1.0

    def test_negative_outcome_reduces_multiplier(self) -> None:
        """net_edge_negative with poor survival → multiplier near floor 0.5"""
        feedback = L1CausalFeedback(
            key=SignalHypothesisKey("trend_ma", "ema_12_72", "4h"),
            outcome="net_edge_negative",
            evidence_end_ns=100,
            effective_n=50.0,
            survival_successes=0,
            survival_trials=4,
            pooled_net_lcb_bps=-27.0,
            positive_fold_ratio=0.10,
        )
        mult = resolve_l1_feedback_multiplier(
            feedback=feedback,
            current_evidence_start_ns=200,
        )
        assert 0.5 <= mult < 0.7
