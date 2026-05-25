"""Tests for Dual Decay Gate risk control logic."""

from __future__ import annotations

import pytest

from src.domain.futures.portfolio.risk_controls import (
    DualDecayConfig,
    evaluate_dual_decay,
)


def test_dual_decay_percent_success_and_failure() -> None:
    """Test percent-based decay evaluation when coarse CAGR > 0."""
    cfg = DualDecayConfig()

    # Case A: Failure
    # percent_decay = (-0.05 - 0.20) / 0.20 = -1.25 (-125%) < -0.15 (-15%) -> FAIL
    res_fail = evaluate_dual_decay(intrabar_cagr=-0.05, coarse_cagr=0.20, cfg=cfg)
    assert res_fail.passed is False
    assert "DUAL_DECAY_PERCENT" in res_fail.failures
    assert res_fail.percent_decay is not None
    assert abs(res_fail.percent_decay - (-1.25)) < 1e-9

    # Case B: Success
    # percent_decay = (0.18 - 0.20) / 0.20 = -0.10 (-10%) > -0.15 (-15%) -> PASS
    res_pass = evaluate_dual_decay(intrabar_cagr=0.18, coarse_cagr=0.20, cfg=cfg)
    assert res_pass.passed is True
    assert len(res_pass.failures) == 0
    assert res_pass.percent_decay is not None
    assert abs(res_pass.percent_decay - (-0.10)) < 1e-9


def test_dual_decay_absolute_success_and_failure() -> None:
    """Test absolute bps decay evaluation."""
    cfg = DualDecayConfig()

    # Case A: Failure
    # abs_decay = (-0.10 - 0.05) * 10000 = -1500 bps < -500 bps -> FAIL
    res_fail = evaluate_dual_decay(intrabar_cagr=-0.10, coarse_cagr=0.05, cfg=cfg)
    assert res_fail.passed is False
    assert "DUAL_DECAY_ABSOLUTE" in res_fail.failures

    # Case B: Success
    # abs_decay = (0.045 - 0.05) * 10000 = -50 bps > -500 bps -> PASS
    # percent_decay = (0.045 - 0.05) / 0.05 = -0.10 (-10%) > -0.15 (-15%) -> PASS
    res_pass = evaluate_dual_decay(intrabar_cagr=0.045, coarse_cagr=0.05, cfg=cfg)
    assert res_pass.passed is True
    assert len(res_pass.failures) == 0


def test_dual_decay_coarse_negative_skips_percent() -> None:
    """Test that percent-based decay is skipped when coarse CAGR <= 0."""
    cfg = DualDecayConfig()

    # coarse <= 0, percent_decay should be None.
    # abs_decay = (-0.04 - -0.05) * 10000 = +100 bps -> PASS
    res = evaluate_dual_decay(intrabar_cagr=-0.04, coarse_cagr=-0.05, cfg=cfg)
    assert res.passed is True
    assert res.percent_decay is None
    assert res.absolute_decay_bps == pytest.approx(100.0)
    assert len(res.failures) == 0


def test_dual_decay_both_failures() -> None:
    """Test that both failures are collected when thresholds are violated."""
    cfg = DualDecayConfig()

    # percent_decay = (-0.10 - 0.10) / 0.10 = -2.00 (-200%) < -0.15 -> FAIL
    # abs_decay = (-0.10 - 0.10) * 10000 = -2000 bps < -500 bps -> FAIL
    res = evaluate_dual_decay(intrabar_cagr=-0.10, coarse_cagr=0.10, cfg=cfg)
    assert res.passed is False
    assert "DUAL_DECAY_PERCENT" in res.failures
    assert "DUAL_DECAY_ABSOLUTE" in res.failures
    assert len(res.failures) == 2
