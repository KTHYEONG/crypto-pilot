from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig
from src.domain.futures.strategy.tiered_workflow.l2_meta import (
    build_regime_routing_plan,
    compute_pooled_realized_edges,
    replicate_pooled_edges_by_regime,
)
from src.domain.futures.strategy.walk_forward import WFFold


def _make_cache(n_bars: int, *, n_sleeve: int = 1) -> MagicMock:
    cache = MagicMock()
    cache.signal_mask_2d = np.ones((n_bars, n_sleeve), dtype=bool)
    cache.side_2d = np.ones((n_bars, n_sleeve), dtype=np.float64)
    cache.holding_bars_2d = np.ones((n_bars, n_sleeve), dtype=np.float64)
    cache.sleeve_to_sym = np.zeros(n_sleeve, dtype=np.int64)
    cache.sleeve_ids = (("BTCUSDT", "trend_4h"),)
    cache.sleeve_to_tf = ("4h",)
    return cache


def _make_aligned(close_1d: list[float]) -> MagicMock:
    aligned = MagicMock()
    aligned.close_2d = np.asarray(close_1d, dtype=np.float64).reshape(-1, 1)
    aligned.symbols = ("BTCUSDT",)
    return aligned


def test_build_regime_routing_plan_when_compression_enabled_uses_three_states() -> None:
    raw_codes = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int8)
    cache = _make_cache(len(raw_codes))
    aligned = _make_aligned([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0])
    folds = (
        WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=4, oos_start=4, oos_end=8),
    )

    plan = build_regime_routing_plan(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        raw_regime_code_1d=raw_codes,
        compression_enabled=True,
        min_n=1,
    )

    assert int(plan.effective_regime_code_1d.max()) <= 2
    assert plan.diagnostics.active_state_count == 3
    assert plan.diagnostics.active_state_names == ("bull", "bear", "crisis")


def test_build_regime_routing_plan_when_proof_fails_replicates_pooled_edges() -> None:
    raw_codes = np.array([0] * 12, dtype=np.int8)
    cache = _make_cache(len(raw_codes))
    aligned = _make_aligned([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0])
    folds = (
        WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=4, oos_start=4, oos_end=8),
        WFFold(fit_start=0, fit_end=7, cal_start=7, cal_end=8, oos_start=8, oos_end=12),
    )

    plan = build_regime_routing_plan(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        raw_regime_code_1d=raw_codes,
        compression_enabled=True,
        min_n=1,
        fallback_mode="pooled",
    )

    expected_fold0 = replicate_pooled_edges_by_regime(plan.pooled_edges_by_fold[0], state_count=3)
    expected_fold1 = replicate_pooled_edges_by_regime(plan.pooled_edges_by_fold[1], state_count=3)

    assert plan.diagnostics.proof_passed is False
    assert plan.diagnostics.conditioning_path == "pooled_fallback"
    assert plan.effective_bucket_edges_by_fold[0] == expected_fold0
    assert plan.effective_bucket_edges_by_fold[1] == expected_fold1


def test_build_regime_routing_plan_when_lift_is_consistent_uses_conditioned_edges() -> None:
    n_bars = 40
    cache = _make_cache(n_bars)
    aligned = _make_aligned([100.0] * n_bars)
    close_values: list[float] = []
    raw_regime_list: list[int] = []
    price = 100.0
    for bar_idx in range(n_bars):
        close_values.append(price)
        if bar_idx < n_bars - 1:
            if bar_idx % 2 == 0:
                price *= 1.04
                raw_regime_list.append(0)
            else:
                price /= 1.04
                raw_regime_list.append(2)
    raw_regime_list.append(0)
    aligned = _make_aligned(close_values)
    raw_codes = np.array(raw_regime_list, dtype=np.int8)
    folds = (
        WFFold(fit_start=0, fit_end=9, cal_start=9, cal_end=10, oos_start=10, oos_end=20),
        WFFold(fit_start=0, fit_end=19, cal_start=19, cal_end=20, oos_start=20, oos_end=30),
        WFFold(fit_start=0, fit_end=29, cal_start=29, cal_end=30, oos_start=30, oos_end=40),
    )

    plan = build_regime_routing_plan(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        raw_regime_code_1d=raw_codes,
        compression_enabled=True,
        min_n=1,
        proof_nw_tstat_threshold=1.0,
        proof_fold_pass_ratio_threshold=0.5,
    )

    assert plan.diagnostics.proof_passed is True
    assert plan.diagnostics.conditioning_path == "regime_conditioned"
    assert plan.effective_bucket_edges_by_fold == plan.raw_bucket_edges_by_fold
    assert plan.diagnostics.mean_lift_bps > 0.0


def test_compute_pooled_realized_edges_respects_holding_bars() -> None:
    cache = _make_cache(5)
    cache.holding_bars_2d[:, :] = 2.0
    aligned = _make_aligned([100.0, 101.0, 103.0, 104.0, 105.0])

    pooled = compute_pooled_realized_edges(
        cache=cache,
        aligned=aligned,
        fit_start=0,
        fit_end=3,
        cost_bps=0.0,
        min_n=1,
    )

    expected = (
        ((103.0 - 100.0) / 100.0) * 10000.0
        + ((103.0 - 101.0) / 101.0) * 10000.0
        + 0.0
    ) / 3.0
    assert pooled[("trend", "4h")] == pytest.approx(expected)


def test_build_regime_routing_plan_passes_regime_and_pooled_edges_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def _fake_evaluate_regime_lift_proof(**kwargs: object) -> SimpleNamespace:
        captured["regime_cond_edges"] = np.asarray(kwargs["regime_cond_edges"], dtype=np.float64)
        captured["pooled_edges"] = np.asarray(kwargs["pooled_edges"], dtype=np.float64)
        return SimpleNamespace(
            proof_passed=False,
            conditioning_path="pooled_fallback",
            mean_lift_bps=0.0,
            n_eff=0.0,
            nw_tstat=0.0,
            deflated_sharpe=0.0,
            fold_pass_ratio=0.0,
            n_folds_evaluated=0,
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.evaluate_regime_lift_proof",
        _fake_evaluate_regime_lift_proof,
    )

    raw_codes = np.array([0] * 8, dtype=np.int8)
    cache = _make_cache(len(raw_codes))
    aligned = _make_aligned([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0])
    folds = (
        WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=4, oos_start=4, oos_end=8),
    )

    plan = build_regime_routing_plan(
        cache=cache,
        aligned=aligned,
        awf_folds=folds,
        raw_regime_code_1d=raw_codes,
        compression_enabled=True,
        min_n=1,
        fallback_mode="pooled",
    )

    assert plan.diagnostics.proof_passed is False
    assert captured["regime_cond_edges"].size > 0
    assert captured["pooled_edges"].size > 0
    assert not np.allclose(captured["regime_cond_edges"], captured["pooled_edges"])


def test_layer2_allocation_config_regime_proof_defaults() -> None:
    cfg = Layer2AllocationConfig.from_mapping({})

    assert cfg.l2_regime_compression_enabled is True
    assert cfg.l2_regime_proof_enabled is True
    assert cfg.l2_regime_fallback_mode == "pooled"
    assert cfg.l2_regime_proof_nw_tstat == pytest.approx(1.5)
    assert cfg.l2_regime_proof_fold_pass_ratio == pytest.approx(0.60)
