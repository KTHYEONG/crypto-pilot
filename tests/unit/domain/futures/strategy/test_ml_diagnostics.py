from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.diagnostics import (
    build_quality_report,
    ndcg_proxy_at_k,
    passes_quality_gate,
)


def test_ndcg_proxy_at_k_range() -> None:
    score = np.array([[0.9, 0.5, 0.1, -0.1, -0.5]], dtype=np.float64)
    rel = np.array([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=np.float64)
    val = ndcg_proxy_at_k(score, rel, k=3)
    assert 0.0 <= val <= 1.0
    assert val > 0.95


def test_build_quality_report_and_gate_pass() -> None:
    t, n, f = 8, 6, 4
    rng = np.random.default_rng(42)
    feature_values = rng.normal(size=(t, n, f)).astype(np.float32)
    feature_valid_mask = np.ones((t, n), dtype=bool)
    label_eligible_mask = np.ones((t, n), dtype=bool)
    signed_ret = rng.normal(scale=1e-3, size=(t, n)).astype(np.float64)
    score = signed_ret + rng.normal(scale=1e-5, size=(t, n)).astype(np.float64)
    relevance = np.tile(np.array([4, 3, 2, 1, 0, 2], dtype=np.float64), (t, 1))
    q50 = score.copy()
    q10 = q50 - 2e-4
    q90 = q50 + 2e-4
    alpha_long = np.maximum(score, 0.0)
    alpha_short = np.maximum(-score, 0.0)
    cost = np.full((t, n), 1e-4, dtype=np.float64)

    report = build_quality_report(
        feature_values=feature_values,
        feature_valid_mask=feature_valid_mask,
        label_eligible_mask=label_eligible_mask,
        score_2d=score,
        signed_ret_2d=signed_ret,
        relevance_2d=relevance,
        q10_2d=q10,
        q50_2d=q50,
        q90_2d=q90,
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        cost_2d=cost,
    )

    assert report["feature_finite_ratio"] == 1.0
    assert report["label_valid_ratio"] == 1.0
    assert report["ranker_valid_ndcg_at_5"] > 0.0
    assert "ev_cost_ratio_proxy" in report
    assert passes_quality_gate(report) is True
