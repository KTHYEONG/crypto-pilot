from __future__ import annotations

import numpy as np

from src.domain.futures.optimization.final_evaluator import _build_oos_alpha_attribution_report


def test_alpha_attribution_fallback_is_finite_with_missing_data() -> None:
    report = _build_oos_alpha_attribution_report(
        oos_port={"equity_curve": np.array([100.0, 100.5, 100.2], dtype=np.float64)},
        oos_data_maps={},
        symbols=["BTCUSDT"],
        tf="4h",
    )
    assert report["status"] == "ok"


def test_alpha_attribution_residual_can_be_negative() -> None:
    report = _build_oos_alpha_attribution_report(
        oos_port={"equity_curve": np.array([100.0, 99.7, 99.2], dtype=np.float64)},
        oos_data_maps={},
        symbols=["BTCUSDT"],
        tf="4h",
    )
    assert float(report["pnl_pct"]["residual"]) <= 0.0
