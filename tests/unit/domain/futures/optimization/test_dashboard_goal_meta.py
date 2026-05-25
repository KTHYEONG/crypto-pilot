from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = str(Path(__file__).resolve().parents[5])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.optimization.observability.dashboard import (
    _build_alpha_goal_eval_meta,
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
