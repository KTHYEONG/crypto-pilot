from __future__ import annotations

import math

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    Layer2EdgeWaterfall,
    _assemble_edge_waterfall,
)


def _call_waterfall(
    *,
    fold_idx: int = 0,
    admitted_contrib: float = 0.0,
    weighted_contrib: float = 0.0,
    capped_contrib: float = 0.0,
    realized_total_bps: float = 0.0,
    cap_binding_bars: int = 0,
    n_rebal: int = 0,
    sleeves_admitted_sum: int = 0,
) -> Layer2EdgeWaterfall:
    return _assemble_edge_waterfall(
        fold_idx=fold_idx,
        admitted_contrib=admitted_contrib,
        weighted_contrib=weighted_contrib,
        capped_contrib=capped_contrib,
        realized_total_bps=realized_total_bps,
        cap_binding_bars=cap_binding_bars,
        n_rebal=n_rebal,
        sleeves_admitted_sum=sleeves_admitted_sum,
    )


def test_assemble_edge_waterfall_tracks_stage_losses() -> None:
    wf = _call_waterfall(
        fold_idx=0,
        admitted_contrib=100.0,
        weighted_contrib=70.0,
        capped_contrib=55.0,
        realized_total_bps=40.0,
        cap_binding_bars=3,
        n_rebal=10,
        sleeves_admitted_sum=50,
    )
    assert wf.loss_weighting == 30.0
    assert wf.loss_capping == 15.0
    assert wf.loss_friction == 15.0
    assert wf.n_sleeves_admitted_mean == 5.0
    assert wf.cap_binding_ratio == 0.3
    assert all(math.isfinite(v) for v in (
        wf.admitted_contrib, wf.weighted_contrib, wf.capped_contrib,
        wf.realized_contrib, wf.loss_weighting, wf.loss_capping,
        wf.loss_friction, wf.n_sleeves_admitted_mean, wf.cap_binding_ratio,
    ))


def test_assemble_edge_waterfall_preserves_negative_edge_sign() -> None:
    wf = _call_waterfall(
        fold_idx=0,
        admitted_contrib=50.0,
        weighted_contrib=-20.0,
        capped_contrib=-15.0,
        realized_total_bps=-10.0,
        cap_binding_bars=0,
        n_rebal=5,
        sleeves_admitted_sum=10,
    )
    assert wf.loss_weighting == 70.0
    assert wf.loss_capping == -5.0
    assert wf.loss_friction == -5.0
    assert wf.admitted_contrib == 50.0
    assert wf.weighted_contrib == -20.0
    assert wf.capped_contrib == -15.0
    assert wf.realized_contrib == -10.0
    assert all(math.isfinite(v) for v in (
        wf.loss_weighting, wf.loss_capping, wf.loss_friction,
    ))


def test_assemble_edge_waterfall_zero_rebal_no_zerodiv() -> None:
    wf = _call_waterfall(
        fold_idx=0,
        admitted_contrib=0.0,
        weighted_contrib=0.0,
        capped_contrib=0.0,
        realized_total_bps=0.0,
        cap_binding_bars=0,
        n_rebal=0,
        sleeves_admitted_sum=0,
    )
    assert wf.n_sleeves_admitted_mean == 0.0
    assert wf.cap_binding_ratio == 0.0
    assert not math.isnan(wf.n_sleeves_admitted_mean)
    assert not math.isinf(wf.n_sleeves_admitted_mean)
    assert not math.isnan(wf.cap_binding_ratio)
    assert not math.isinf(wf.cap_binding_ratio)
