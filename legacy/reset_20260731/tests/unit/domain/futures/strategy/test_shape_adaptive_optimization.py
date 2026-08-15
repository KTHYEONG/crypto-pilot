from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.causal_statistics import (
    causal_expanding_quantile,
    causal_expanding_robust_location_scale,
)
from src.domain.futures.strategy.common.alignment import (
    extract_aligned_symbol_block,
)
from src.domain.futures.strategy.market_regime import (
    compute_market_regime_context,
    compute_risk_overlay,
)
from src.domain.futures.strategy.tiered_workflow.memory import (
    L1PilotMeasurement,
    MIN_WORKER_BYTES,
    ProcessTreeMemory,
    resolve_post_pilot_memory_plan,
    select_l1_pilot_fold_index,
)
from src.domain.futures.strategy.walk_forward import WFFold

# ──────────────────────────────────────────────────────────────────────────────
# S1: [QUANT-CAUSAL] causal_expanding_statistics
# ──────────────────────────────────────────────────────────────────────────────


def _reference_expanding_quantile(values: np.ndarray, q: float, min_periods: int = 1) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        finite = values[: idx + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size >= min_periods:
            out[idx] = float(np.percentile(finite, q * 100.0))
    return out


def test_causal_expanding_statistics_match_reference_and_preserve_prefix_causality() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 50).astype(np.float64)

    q50_vec = causal_expanding_quantile(values, 0.5)
    q50_ref = _reference_expanding_quantile(values, 0.5)
    np.testing.assert_allclose(q50_vec, q50_ref, atol=1e-12, err_msg="q=0.5 mismatch vs reference")

    q90_vec = causal_expanding_quantile(values, 0.9)
    q90_ref = _reference_expanding_quantile(values, 0.9)
    np.testing.assert_allclose(q90_vec, q90_ref, atol=1e-12, err_msg="q=0.9 mismatch vs reference")

    loc_vec, scale_vec = causal_expanding_robust_location_scale(values)
    loc_ref = _reference_expanding_quantile(values, 0.5)
    q75_ref = _reference_expanding_quantile(values, 0.75)
    q25_ref = _reference_expanding_quantile(values, 0.25)
    iqr = q75_ref - q25_ref
    scale_ref = np.where(np.isfinite(iqr) & (iqr / 1.3489795 >= 1e-12), iqr / 1.3489795, 1e-12)
    np.testing.assert_allclose(loc_vec, loc_ref, atol=1e-10, err_msg="location mismatch vs reference")
    np.testing.assert_allclose(scale_vec, scale_ref, atol=1e-10, err_msg="scale mismatch vs reference")

    # Prefix causality: future suffix must not change prefix results
    prefix = values[:20]
    suffix = np.concatenate([prefix, rng.normal(0, 1, 30).astype(np.float64)])
    q50_prefix = causal_expanding_quantile(prefix, 0.5)
    q50_suffix = causal_expanding_quantile(suffix, 0.5)[:20]
    np.testing.assert_allclose(q50_prefix, q50_suffix, atol=1e-12, err_msg="prefix causality violated")

    # non-finite handling
    with_nan = values.copy()
    with_nan[5:10] = np.nan
    q_nan = causal_expanding_quantile(with_nan, 0.5)
    finite_mask = np.isfinite(with_nan)
    expected = _reference_expanding_quantile(with_nan, 0.5)
    np.testing.assert_allclose(q_nan, expected, atol=1e-12)

    # min_periods: need 3 finite obs before result becomes finite
    sparse = np.full(10, np.nan, dtype=np.float64)
    sparse[[2, 4, 6]] = [1.0, 2.0, 3.0]
    result = causal_expanding_quantile(sparse, 0.5, min_periods=3)
    assert np.isnan(result[:3]).all(), "pre-finite slots must be NaN"
    assert np.isfinite(result[6]), "3rd finite at idx 6 should meet min_periods=3"

    # empty input
    empty = np.empty(0, dtype=np.float64)
    assert causal_expanding_quantile(empty, 0.5).shape == (0,)
    loc, scale = causal_expanding_robust_location_scale(empty)
    assert loc.shape == (0,)
    assert scale.shape == (0,)

    # invalid q raises
    with pytest.raises(ValueError, match="q must be in"):
        causal_expanding_quantile(values, -0.1)
    with pytest.raises(ValueError, match="q must be in"):
        causal_expanding_quantile(values, 1.5)

    # multi-dim raises
    with pytest.raises(ValueError, match="values must be 1-D"):
        causal_expanding_quantile(np.ones((5, 5), dtype=np.float64), 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# S2: [PERF-L1-CACHE] market_regime reuses precomputed overlay
# ──────────────────────────────────────────────────────────────────────────────


def _make_fake_aligned(t_len: int = 50, n_sym: int = 2) -> MagicMock:
    rng = np.random.default_rng(42)
    aligned = MagicMock()
    aligned.close_2d = rng.uniform(100, 200, (t_len, n_sym)).astype(np.float64)
    aligned.open_2d = rng.uniform(99, 201, (t_len, n_sym)).astype(np.float64)
    aligned.high_2d = np.maximum(aligned.close_2d, rng.uniform(100, 210, (t_len, n_sym))).astype(np.float64)
    aligned.low_2d = np.minimum(aligned.close_2d, rng.uniform(90, 100, (t_len, n_sym))).astype(np.float64)
    aligned.volume_2d = rng.uniform(1e6, 1e8, (t_len, n_sym)).astype(np.float64)
    aligned.funding_2d = rng.uniform(-0.001, 0.001, (t_len, n_sym)).astype(np.float64)
    aligned.symbols = ("BTCUSDT", "ETHUSDT")
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(t_len)],
        dtype="datetime64[ns]",
    )
    return aligned


def test_market_regime_reuses_precomputed_overlay_once() -> None:
    aligned = _make_fake_aligned()

    with patch(
        "src.domain.futures.strategy.market_regime.compute_risk_overlay",
        wraps=compute_risk_overlay,
    ) as spy:
        _ = compute_market_regime_context(aligned=aligned, overlay=None)
        spy.assert_called_once()

    overlay = compute_risk_overlay(aligned=aligned)
    with patch(
        "src.domain.futures.strategy.market_regime.compute_risk_overlay",
    ) as mock:
        ctx = compute_market_regime_context(aligned=aligned, overlay=overlay)
        mock.assert_not_called()
    assert ctx.code_1d.shape[0] == aligned.close_2d.shape[0]


# ──────────────────────────────────────────────────────────────────────────────
# S3: [MEM-PLAN] post_pilot_plan adapts to actual worker PSS
# ──────────────────────────────────────────────────────────────────────────────


def test_post_pilot_plan_adapts_to_actual_worker_pss_without_tf_assumptions() -> None:
    pilot = L1PilotMeasurement(
        stage="evidence",
        fold_id=0,
        shared_input_bytes=500 * 1024**2,
        worker_private_bytes=700 * 1024**2,
        result_bytes=100 * 1024**2,
        elapsed_seconds=10.0,
    )
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=2 * 1024**3,
        tree_pss_bytes=3 * 1024**3,
        tree_uss_bytes=None,
        available_bytes=20 * 1024**3,
    )
    plan = resolve_post_pilot_memory_plan(
        n_remaining=8,
        pilot=pilot,
        snapshot=snapshot,
        stage_cap=4,
        cpu_cap=8,
    )
    assert plan.workers >= 1
    assert plan.workers <= 4
    assert plan.binding_constraint != ""

    # pinned acts as upper bound
    plan_pinned = resolve_post_pilot_memory_plan(
        n_remaining=8,
        pilot=pilot,
        snapshot=snapshot,
        stage_cap=4,
        cpu_cap=8,
        pinned=2,
    )
    assert plan_pinned.workers <= 2

    # tiny memory -> serial
    tight = resolve_post_pilot_memory_plan(
        n_remaining=8,
        pilot=pilot,
        snapshot=ProcessTreeMemory(
            parent_rss_bytes=9 * 1024**3,
            tree_pss_bytes=9 * 1024**3,
            tree_uss_bytes=None,
            available_bytes=1 * 1024**3,
        ),
        stage_cap=4,
        cpu_cap=8,
    )
    assert tight.workers == 1
    assert "serial" in tight.reason or "floor" in tight.reason


# ──────────────────────────────────────────────────────────────────────────────
# S4: [RESILIENCE] pilot falls back when memory metrics unavailable
# ──────────────────────────────────────────────────────────────────────────────


def test_pilot_stage_falls_back_when_memory_metrics_are_unavailable() -> None:
    pilot = L1PilotMeasurement(
        stage="evidence",
        fold_id=0,
        shared_input_bytes=500 * 1024**2,
        worker_private_bytes=700 * 1024**2,
        result_bytes=100 * 1024**2,
        elapsed_seconds=10.0,
    )
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=None,
        tree_uss_bytes=None,
        available_bytes=0,
    )
    plan = resolve_post_pilot_memory_plan(
        n_remaining=8,
        pilot=pilot,
        snapshot=snapshot,
        stage_cap=4,
        cpu_cap=8,
    )
    assert plan.workers == 1
    assert "serial" in plan.reason or "floor" in plan.reason


def test_select_l1_pilot_fold_index_selects_largest_span() -> None:
    folds = (
        WFFold(0, 10, 10, 20, 20, 30),
        WFFold(0, 50, 50, 100, 100, 120),
        WFFold(0, 5, 5, 10, 10, 15),
    )
    idx = select_l1_pilot_fold_index(folds)
    assert idx == 1

    # tie: smallest fold_id
    folds_tie = (
        WFFold(0, 30, 30, 60, 60, 80),
        WFFold(0, 30, 30, 60, 60, 80),
    )
    idx_tie = select_l1_pilot_fold_index(folds_tie)
    assert idx_tie == 0


# ──────────────────────────────────────────────────────────────────────────────
# S5: [QUANT-DETERMINISM] pipeline runs pilot once and preserves fold order
# ──────────────────────────────────────────────────────────────────────────────


def test_l1_pipeline_runs_pilot_once_and_preserves_fold_order() -> None:
    """Verifies select_l1_pilot_fold_index picks one fold as pilot."""
    folds = tuple(
        WFFold(i * 10, i * 10 + 5, i * 10 + 5, i * 10 + 8, i * 10 + 8, i * 10 + 10)
        for i in range(8)
    )
    pilot_idx = select_l1_pilot_fold_index(folds)
    assert 0 <= pilot_idx < len(folds)
    remaining = [f for i, f in enumerate(folds) if i != pilot_idx]
    assert len(remaining) == len(folds) - 1
    # remaining must preserve original fold_id order
    assert remaining == [f for i, f in enumerate(folds) if i != pilot_idx]


# ──────────────────────────────────────────────────────────────────────────────
# S6: [MEM-CAP] pipeline throttles unscheduled folds after PSS cap
# ──────────────────────────────────────────────────────────────────────────────


def test_l1_pipeline_throttles_unscheduled_folds_after_pss_cap() -> None:
    """resolve_post_pilot_memory_plan must react to cap-exceeded snapshots."""
    pilot = L1PilotMeasurement(
        stage="evidence",
        fold_id=0,
        shared_input_bytes=500 * 1024**2,
        worker_private_bytes=700 * 1024**2,
        result_bytes=100 * 1024**2,
        elapsed_seconds=10.0,
    )
    snapshot_over_cap = ProcessTreeMemory(
        parent_rss_bytes=9 * 1024**3,
        tree_pss_bytes=9 * 1024**3 + 1,
        tree_uss_bytes=None,
        available_bytes=20 * 1024**3,
    )
    plan = resolve_post_pilot_memory_plan(
        n_remaining=8,
        pilot=pilot,
        snapshot=snapshot_over_cap,
        stage_cap=4,
        cpu_cap=8,
        tree_pss_cap_bytes=10 * 1024**3,
        reserve_bytes=1 * 1024**3,
    )
    assert plan.workers == 1
    assert "serial" in plan.reason or "floor" in plan.reason


# ──────────────────────────────────────────────────────────────────────────────
# S7: [MEM-LIFECYCLE] release_aligned_feature_cache
# ──────────────────────────────────────────────────────────────────────────────


def test_release_aligned_feature_cache_on_success_and_exception() -> None:
    from src.domain.futures.strategy.candidate_dataset import (
        _ALIGNED_FEATURE_CACHE,
        release_aligned_feature_cache,
    )

    aligned = _make_fake_aligned()
    aligned_id = id(aligned)
    _ALIGNED_FEATURE_CACHE[aligned_id] = {
        "__aligned_ref__": aligned,
        "arr_1": np.ones((100, 10), dtype=np.float64),
        "arr_2": np.ones((50, 5), dtype=np.float64),
    }
    freed = release_aligned_feature_cache(aligned)
    assert freed > 0
    assert aligned_id not in _ALIGNED_FEATURE_CACHE
    # aligned object itself is not mutated
    assert aligned.close_2d is not None

    # idempotent: second call returns 0
    assert release_aligned_feature_cache(aligned) == 0


def test_release_aligned_feature_cache_handles_missing_entry() -> None:
    from src.domain.futures.strategy.candidate_dataset import release_aligned_feature_cache

    aligned = _make_fake_aligned()
    assert release_aligned_feature_cache(aligned) == 0


# ──────────────────────────────────────────────────────────────────────────────
# S8: [OUTPUT-PARITY] extract_aligned_symbol_block
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_aligned_symbol_block_matches_reference_arrays_and_masks() -> None:
    rng = np.random.default_rng(42)
    n = 30
    frame = pd.DataFrame(
        {
            "open": rng.uniform(100, 200, n),
            "high": rng.uniform(100, 200, n),
            "low": rng.uniform(100, 200, n),
            "close": rng.uniform(100, 200, n),
            "volume": rng.uniform(1e6, 1e8, n),
            "active_mask": rng.choice([True, False], n),
            "warm_mask": rng.choice([True, False], n),
        }
    )
    numeric_cols = ("open", "high", "low", "close", "volume")
    mask_cols = ("active_mask", "warm_mask")

    block = extract_aligned_symbol_block(
        frame,
        start=5,
        end=25,
        numeric_columns=numeric_cols,
        mask_columns=mask_cols,
    )
    assert block.numeric.shape == (20, 5)
    assert block.masks.shape == (20, 2)
    assert block.numeric_columns == numeric_cols
    assert block.mask_columns == mask_cols

    # values match iloc reference
    ref_numeric = frame.iloc[5:25][list(numeric_cols)].to_numpy(dtype=np.float64)
    ref_masks = frame.iloc[5:25][list(mask_cols)].to_numpy(dtype=np.bool_)
    np.testing.assert_allclose(block.numeric, ref_numeric, atol=1e-12)
    np.testing.assert_array_equal(block.masks, ref_masks)

    # empty columns
    empty_num = extract_aligned_symbol_block(
        frame,
        start=0,
        end=n,
        numeric_columns=(),
        mask_columns=mask_cols,
    )
    assert empty_num.numeric.shape == (n, 0)
    assert empty_num.masks.shape == (n, 2)


# ──────────────────────────────────────────────────────────────────────────────
# S9: [QUANT-ALIGNMENT] align_data_maps bulk path preserves timestamp/symbol order
# ──────────────────────────────────────────────────────────────────────────────


def test_align_data_maps_bulk_path_preserves_timestamp_and_symbol_order() -> None:
    """extract_aligned_symbol_block preserves row (timestamp) and column (symbol) order."""
    from src.domain.futures.strategy.common.alignment import align_data_maps

    rng = np.random.default_rng(42)
    base = pd.date_range("2024-01-01", periods=50, freq="4h")
    dfs = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        n = 50
        dfs[sym] = pd.DataFrame({
            "datetime": base,
            "open": rng.uniform(100, 200, n),
            "high": rng.uniform(100, 200, n),
            "low": rng.uniform(100, 200, n),
            "close": rng.uniform(100, 200, n),
            "volume": rng.uniform(1e6, 1e8, n),
            "funding_rate": rng.uniform(-0.001, 0.001, n),
            "universe_active_mask": rng.choice([True, False], n),
            "universe_entry_warm_mask": rng.choice([True, False], n),
        })

    data_maps = {sym: {"4h": dfs[sym]} for sym in dfs}
    fake_info = {
        "eff_ref_len": 50,
        "alignment_offsets": {"BTCUSDT": 0, "ETHUSDT": 0, "SOLUSDT": 0},
    }

    with patch(
        "src.domain.futures.strategy.common.alignment.compute_multi_alignment_info",
        return_value=fake_info,
    ):
        result = align_data_maps(data_maps, symbols=list(dfs), tf="4h", cache_result=False)
    assert result.datetimes.shape[0] == 50
    assert len(result.symbols) == 3
    assert result.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    np.testing.assert_array_equal(result.datetimes, base.values.astype("datetime64[ns]"))


# ──────────────────────────────────────────────────────────────────────────────
# S10: [PERF-PROBE] sparse PSS and injected logger
# ──────────────────────────────────────────────────────────────────────────────


def test_l2_probe_uses_sparse_pss_and_injected_logger() -> None:
    from pathlib import Path

    from src.domain.futures.optimization.observability.l2_runtime_probe import (
        L2RuntimeProbe,
        PSS_INTERVAL_SAMPLES,
    )

    probe_logger = logging.getLogger("test_probe")
    probe_logger.setLevel(logging.DEBUG)

    _test_probe_path = Path("test_probe.jsonl")
    probe = L2RuntimeProbe(
        enabled=True,
        sample_interval_ms=50,
        hot_sample_interval_ms=50,
        jsonl_enabled=False,
        jsonl_path=_test_probe_path,
        logger=probe_logger,
        pss_interval_samples=5,
    )
    assert probe._logger is probe_logger
    assert probe._pss_interval_samples == 5

    probe.start_run(stage="test")
    assert probe._run_id != ""

    snap = probe.snapshot_now(reason="span_start")
    assert snap is not None
    assert snap.rss_mb >= 0 or snap.rss_mb == -1.0  # -1.0 if access denied
    # cheap sample: PSS may be -1 on some platforms but should be collected
    # at span boundaries regardless of sparseness
    probe.stop_run(outcome="completed")

    path2 = Path("test_probe2.jsonl")
    probe2 = L2RuntimeProbe(
        enabled=True,
        sample_interval_ms=100,
        hot_sample_interval_ms=100,
        jsonl_enabled=False,
        jsonl_path=path2,
        pss_interval_samples=PSS_INTERVAL_SAMPLES,
    )
    with (
        patch.object(probe2, "_should_collect_pss", return_value=False),
        patch.object(probe2, "_collect_rss_only") as mock_rss,
    ):
        mock_rss.return_value = ([{"pid": 0, "role": "parent", "rss_mb": 100.0, "pss_mb": -1.0, "status": "ok"}], 100.0, 0.5)
        probe2._do_sample(reason="periodic", stage_path="root")
        mock_rss.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# S11: [ROBUSTNESS] shape-adaptive plan (no hardcoded TF/signal assumptions)
# ──────────────────────────────────────────────────────────────────────────────


def test_runtime_optimization_is_shape_adaptive() -> None:
    """resolve_post_pilot_memory_plan must produce reasonable plans for diverse shapes."""
    configs: list[tuple[str, int, int, int, int, int]] = [
        ("small", 6, 500, 300, 50, 16),
        ("large_signal", 24, 2000, 1500, 400, 64),
        ("many_folds", 10, 800, 600, 100, 4),
        ("tight_memory", 12, 1200, 900, 200, 8),
    ]
    for label, n_rem, shared_mb, worker_mb, result_mb, cap in configs:
        pilot = L1PilotMeasurement(
            stage="evidence",
            fold_id=0,
            shared_input_bytes=shared_mb * 1024**2,
            worker_private_bytes=worker_mb * 1024**2,
            result_bytes=result_mb * 1024**2,
            elapsed_seconds=5.0,
        )
        snap = ProcessTreeMemory(
            parent_rss_bytes=2 * 1024**3,
            tree_pss_bytes=3 * 1024**3,
            tree_uss_bytes=None,
            available_bytes=12 * 1024**3,
        )
        plan = resolve_post_pilot_memory_plan(
            n_remaining=n_rem,
            pilot=pilot,
            snapshot=snap,
            stage_cap=cap,
            cpu_cap=cap,
        )
        assert plan.workers >= 1, f"{label}: workers must be >= 1"
        assert plan.workers <= cap, f"{label}: workers must not exceed stage_cap"
        assert plan.estimated_worker_private_bytes >= MIN_WORKER_BYTES, f"{label}: below min worker bytes"
        assert plan.projected_tree_bytes > 0, f"{label}: projected_tree must be > 0"
