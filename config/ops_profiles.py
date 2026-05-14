"""Operational run profiles for futures-opt (trials/seeds/revalidation expectations).

Use with `--ops-profile` on opt_main_futures or environment `FUTURES_OPS_PROFILE`.
"""

from __future__ import annotations

from typing import Any

# Profile name -> settings (extend without breaking older runners).
OPS_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "description": "Fast sanity: single seed, few trials, minimal strict/expensive gates",
        "trials": 50,
        "seeds": [42],
        "require_revalidation": False,
        "expected_min_complete_trials_ratio": 0.0,
        "config_overrides": {
            "FUTURES_USE_META_LABELER": False,
            "FUTURES_WF_HMM_LEG_REFIT": False,
            "FUTURES_TMP_MD_CHAMPION_GATES_ENABLED": False,
            "FUTURES_TMP_LAYER3_HARD_GATE": False,
            "FUTURES_CHAMP_STABILITY_HARD_GATE": False,
            "FUTURES_STABILITY_RUNNER_UP_RETRY": False,
            "FUTURES_STABILITY_SEEDS": [42],
            "FUTURES_ERGODICITY_HARD_GATE_ENABLED": False,
            "FUTURES_VALIDATION_CONFIG": {
                "wf_ergodicity_hard_gate_enabled": False,
            },
        },
    },
    "candidate": {
        "description": "Exploration batch: moderate strictness and runtime depth",
        "trials": 1000,
        "seeds": [42, 7, 13],
        "require_revalidation": False,
        "expected_min_complete_trials_ratio": 0.0,
        "config_overrides": {
            "FUTURES_USE_META_LABELER": True,
            "FUTURES_WF_HMM_LEG_REFIT": False,
            "FUTURES_TMP_MD_CHAMPION_GATES_ENABLED": True,
            "FUTURES_TMP_LAYER3_HARD_GATE": False,
            "FUTURES_CHAMP_STABILITY_HARD_GATE": False,
            "FUTURES_STABILITY_RUNNER_UP_RETRY": True,
            "FUTURES_STABILITY_SEEDS": [42, 7],
            "FUTURES_ERGODICITY_HARD_GATE_ENABLED": False,
            "FUTURES_VALIDATION_CONFIG": {
                "wf_ergodicity_hard_gate_enabled": False,
            },
        },
    },
    "promotion": {
        "description": "High-Precision Promotion: strict gates on, deeper stability checks",
        "trials": 2000,
        "seeds": [42, 7, 13, 21, 55],
        "require_revalidation": True,
        "expected_min_complete_trials_ratio": 0.0,
        "config_overrides": {
            "FUTURES_USE_META_LABELER": True,
            "FUTURES_WF_HMM_LEG_REFIT": True,
            "FUTURES_TMP_MD_CHAMPION_GATES_ENABLED": True,
            "FUTURES_TMP_LAYER3_HARD_GATE": True,
            "FUTURES_CHAMP_STABILITY_HARD_GATE": True,
            "FUTURES_STABILITY_RUNNER_UP_RETRY": True,
            "FUTURES_STABILITY_SEEDS": [42, 7, 13, 21, 55],
            "FUTURES_STEP2_REGIME_DEPLOY_ENABLED": True,
            "FUTURES_STEP4_DEPLOYABILITY_ENABLED": True,
            "FUTURES_ERGODICITY_HARD_GATE_ENABLED": True,
            "FUTURES_VALIDATION_CONFIG": {
                "wf_ergodicity_hard_gate_enabled": True,
            },
        },
    },
}


def resolve_ops_profile(name: str | None) -> dict[str, Any] | None:
    """Return merged profile dict or None for custom (no preset)."""
    if not name or name == "custom":
        return None
    key = str(name).strip().lower()
    base = OPS_PROFILES.get(key)
    if base is None:
        return None
    out = dict(base)
    out["id"] = key
    return out


def check_run_summary_against_profile(
    profile_name: str,
    summary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Lightweight auto-check after a run (pass counts, gate flag, optional revalidate).

    summary keys (best-effort):
      - gate_ok: bool
      - n_seeds, trials_per_seed, n_complete, n_pass_trials (optional)
    """
    prof = resolve_ops_profile(profile_name)
    if prof is None:
        return True, []
    issues: list[str] = []
    if prof.get("require_revalidation") and not summary.get("revalidation_ok", True):
        issues.append("revalidation_missing_or_failed")
    if "gate_ok" in summary and summary["gate_ok"] is not True and profile_name == "promotion":
        issues.append("promotion_profile_expected_gate_ok")
    exp_seeds = len(prof.get("seeds") or [])
    if exp_seeds and summary.get("n_seeds") not in (None, exp_seeds):
        if int(summary.get("n_seeds", 0)) != exp_seeds:
            issues.append("seed_count_mismatch")
    exp_trials = prof.get("trials")
    got_trials = summary.get("trials_per_seed")
    if exp_trials is not None and got_trials is not None:
        if int(got_trials) != int(exp_trials):
            issues.append("trials_per_seed_mismatch")
    return len(issues) == 0, issues
