from __future__ import annotations

from src.domain.futures.optimization.run_tracker import build_p7_ops_summary


def test_build_p7_ops_summary_fail_on_no_candidate_and_no_elite() -> None:
    summary = build_p7_ops_summary(
        mode="optimization",
        ml_integrity_report={
            "panel": {"nan_pct": 0.0},
            "panel_pre_fillna_nan_pct": 0.01,
            "stages": [{"stage": "raw"}],
            "feature_group_coverage": [{"group": "price"}],
        },
        alpha_filter_meta={"n_surviving": 0, "n_components": 48},
        alpha_goal_meta={"verdict": "fail", "reason_codes": ["no_elite_components"]},
        hmm_goal_meta={"verdict": "pass", "reason_codes": []},
        alpha_cache_meta={"cache_state": "hit", "cache_schema": "v1"},
        study_user_attrs={"obs_no_valid_candidates_reason": "gate_reject_all"},
        selection_summary={"selection_reject_reason_count": {"mdd_hard": 4}},
    )
    assert summary["health_status"] == "fail"
    assert "no_elite_components" in summary["reason_codes"]
    assert "no_candidate:gate_reject_all" in summary["reason_codes"]


def test_build_p7_ops_summary_warn_on_high_nan_without_hard_fail() -> None:
    summary = build_p7_ops_summary(
        mode="alpha-only",
        ml_integrity_report={"panel": {"nan_pct": 0.12}, "panel_pre_fillna_nan_pct": 0.2},
        alpha_filter_meta={"n_surviving": 3, "n_components": 24},
        alpha_goal_meta={"verdict": "warn", "reason_codes": []},
        hmm_goal_meta={"verdict": "pass", "reason_codes": []},
        alpha_cache_meta={"cache_state": "miss", "cache_schema": "v1"},
        study_user_attrs={},
        selection_summary={},
    )
    assert summary["health_status"] == "warn"
    assert summary["alpha_cache"]["enabled"] is True
    assert summary["alpha_cache"]["state"] == "miss"
