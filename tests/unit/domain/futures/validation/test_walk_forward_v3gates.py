"""Tests for Walk-Forward v3.0 quality gates (DSR, funding drag)."""

from __future__ import annotations

from src.domain.futures.validation.walk_forward import (
    WalkForwardConfig,
    mirror_walk_forward_result_from_awf_user_attrs,
)


def test_walk_forward_v3gates_success() -> None:
    """Test successful walk-forward validation when all gates pass."""
    cfg = WalkForwardConfig()
    # 8 legs, all healthy (>1.0)
    leg_log_tw = [0.05] * 8
    user_attrs = {
        "awf_path_leg_log_tw": leg_log_tw,
        "dsr": 0.75,
        "funding_drag_ratio": 0.15,
    }

    result = mirror_walk_forward_result_from_awf_user_attrs(user_attrs, cfg)
    assert result.passed is True
    assert len(result.failures) == 0
    assert result.dsr == 0.75
    assert result.funding_drag_ratio == 0.15


def test_walk_forward_v3gates_dsr_failure() -> None:
    """Test walk-forward validation failure due to low DSR."""
    cfg = WalkForwardConfig()
    leg_log_tw = [0.05] * 8
    # DSR under floor (0.60)
    user_attrs = {
        "awf_path_leg_log_tw": leg_log_tw,
        "dsr": 0.58,
        "funding_drag_ratio": 0.15,
    }

    result = mirror_walk_forward_result_from_awf_user_attrs(user_attrs, cfg)
    assert result.passed is False
    assert "WF_DSR_FLOOR" in result.failures


def test_walk_forward_v3gates_funding_drag_failure() -> None:
    """Test walk-forward validation failure due to high funding drag."""
    cfg = WalkForwardConfig()
    leg_log_tw = [0.05] * 8
    # Funding drag above ceiling (0.30)
    user_attrs = {
        "awf_path_leg_log_tw": leg_log_tw,
        "dsr": 0.75,
        "funding_drag_ratio": 0.35,
    }

    result = mirror_walk_forward_result_from_awf_user_attrs(user_attrs, cfg)
    assert result.passed is False
    assert "WF_FUNDING_DRAG" in result.failures


def test_walk_forward_v3gates_multiple_failures() -> None:
    """Test multiple gate failures detected simultaneously."""
    cfg = WalkForwardConfig()
    # Bad legs, bad DSR, bad funding
    leg_log_tw = [-0.2] * 8  # all negative
    user_attrs = {
        "awf_path_leg_log_tw": leg_log_tw,
        "dsr": 0.40,
        "funding_drag_ratio": 0.50,
    }

    result = mirror_walk_forward_result_from_awf_user_attrs(user_attrs, cfg)
    assert result.passed is False
    assert "WF_POSITIVE_LEG_RATIO" in result.failures
    assert "WF_DSR_FLOOR" in result.failures
    assert "WF_FUNDING_DRAG" in result.failures
