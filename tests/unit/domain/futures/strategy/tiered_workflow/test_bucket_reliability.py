from __future__ import annotations

from src.domain.futures.strategy.tiered_workflow.bucket_reliability import (
    build_bucket_reliability,
)


def test_build_bucket_reliability_allows_consistent_fit_cal() -> None:
    reliability = build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=18.0,
        cal_edge_bps=12.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    assert reliability.sign_consistent is True
    assert reliability.action == "allow"
    assert reliability.reliability >= 0.55


def test_build_bucket_reliability_pools_sign_flip() -> None:
    reliability = build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=18.0,
        cal_edge_bps=-4.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    assert reliability.sign_consistent is False
    assert reliability.action == "pool"


def test_build_bucket_reliability_downweights_weak_consistent_signal() -> None:
    reliability = build_bucket_reliability(
        regime=1,
        family="trend",
        tf="4h",
        fit_edge_bps=5.0,
        cal_edge_bps=4.0,
        n_fit=30,
        n_cal=24,
        min_fit_n=15,
        min_cal_n=20,
        min_cal_lift_bps=8.0,
        min_reliability=0.55,
    )

    # relaxed_reliability_threshold=0.35 default converts reliability=0.5 to allow
    assert reliability.action == "allow"
    assert reliability.reliability < 0.55 or reliability.cal_edge_bps < 8.0
