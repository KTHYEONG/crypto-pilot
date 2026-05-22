from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.optimization.observability.dashboard import (
    _build_alpha_goal_eval_meta,
    _build_hmm_goal_eval_meta,
    log_hmm_report_summary,
)


def test_build_alpha_goal_eval_meta_reports_missing_reasons() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"), ["BTCUSDT"]],
        names=["datetime", "symbol"],
    )
    panel = pd.DataFrame({"alpha_long_00": [0.5, 0.5], "target": [0.1, 0.2]}, index=idx)
    panel.attrs["alpha_component_filter"] = {
        "n_surviving": 1.0,
        "n_components": 1.0,
        "fail_fdr": 0.0,
        "fail_dsr": 0.0,
        "fail_half_life": 0.0,
        "fail_tail": 0.0,
        "fail_oos": 0.0,
        "fail_short": 0.0,
        "fail_sym_bal": 0.0,
    }

    meta = _build_alpha_goal_eval_meta(panel, is_end_date="2026-01-02")
    assert meta["framework"] == "g-alpha.v8"
    assert meta["verdict"] == "warn"
    assert "missing_icir_oos" in meta["reason_codes"]
    assert "missing_gate_status" in meta["reason_codes"]


def test_build_hmm_goal_eval_meta_captures_gate_fail_and_missing_groups() -> None:
    rep = {
        "hmm_lead_lag_tail_capture_8bar": 41.0,
        "hmm_avg_duration": 20.0,
        "hmm_crisis_precision": 12.0,
        "hmm_prob_bull_calm_vol_scale": 0.7,
        "hmm_execution_damp_tail_capture": 79.0,  # fail
        "hmm_execution_damp_crisis_cap": 91.0,
        "hmm_execution_protected_exposure_share": 35.0,
        "hmm_false_flat_cost": 10.0,
        "hmm_execution_damp_active_rate": 0.0,  # missing execution damp gates
    }
    meta = _build_hmm_goal_eval_meta(rep)
    assert meta["framework"] == "h-hmm.v11"
    assert meta["verdict"] == "fail"
    assert "gate_fail:damp_tail_capture" in meta["reason_codes"]
    assert "missing_feature_group:supervised_q_scores" in meta["reason_codes"]
    assert "missing_feature_group:execution_damp_gates" in meta["reason_codes"]


def test_log_hmm_report_summary_attaches_structured_meta() -> None:
    rep: dict[str, object] = {
        "hmm_lead_lag_tail_capture_8bar": 45.0,
        "hmm_avg_duration": 24.0,
        "hmm_crisis_precision": 20.0,
        "hmm_prob_bull_calm_vol_scale": 0.6,
        "hmm_execution_damp_tail_capture": 85.0,
        "hmm_execution_damp_crisis_cap": 93.0,
        "hmm_execution_protected_exposure_share": 45.0,
        "hmm_false_flat_cost": 12.0,
        "hmm_execution_damp_active_rate": 10.0,
        "hmm_sup_q10_h8_top_decile_hit": 30.0,
    }
    log_hmm_report_summary(rep)
    assert "hmm_goal_eval_meta" in rep
    meta = cast(dict[str, Any], rep["hmm_goal_eval_meta"])
    assert meta["verdict"] in {"pass", "warn", "fail"}
