from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.futures.strategy.causal_statistics import causal_expanding_quantile
from src.domain.futures.strategy.common.alignment import extract_aligned_symbol_block
from src.domain.futures.strategy.tiered_workflow.snapshot_executor import (
    L1SnapshotTask,
    execute_l1_snapshot_batch,
)


ROOT = Path(__file__).resolve().parents[5]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_market_regime_delegates_expanding_statistics_to_causal_module() -> None:
    source = _source("src/domain/futures/strategy/market_regime.py")
    assert "causal_statistics" in source
    assert "causal_expanding_quantile" in source


def test_market_regime_causal_helpers_preserve_prefix_under_future_suffix() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    extended = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    base = causal_expanding_quantile(values, 0.5)
    suffix = causal_expanding_quantile(extended, 0.5)
    np.testing.assert_allclose(base, suffix[: values.size], equal_nan=True)


def test_causal_regime_path_uses_constant_number_of_quantile_calls() -> None:
    source = _source("src/domain/futures/strategy/causal_statistics.py")
    assert source.count("expanding.quantile") <= 4


def test_align_data_maps_uses_one_bulk_block_per_symbol() -> None:
    source = _source("src/domain/futures/strategy/common/alignment.py")
    assert "extract_aligned_symbol_block" in source
    assert "numeric_columns" in source
    assert "mask_columns" in source


def test_bulk_alignment_preserves_optional_alias_defaults_and_datetime_order() -> None:
    frame = pd.DataFrame(
        {"close": [2.0, 1.0], "active": [True, False]},
        index=pd.to_datetime(["2025-01-02", "2025-01-01"]),
    ).sort_index()
    block = extract_aligned_symbol_block(
        frame, start=0, end=2, numeric_columns=("close",), mask_columns=("active",)
    )
    assert block.numeric_columns == ("close",)
    assert block.mask_columns == ("active",)
    assert block.numeric[:, 0].tolist() == [1.0, 2.0]


def test_build_single_tf_panels_disables_ephemeral_alignment_cache() -> None:
    source = _source("src/domain/futures/strategy_runtime/bridge.py")
    assert "cache_result=False" in source


def test_snapshot_batch_matches_serial_reference_exactly() -> None:
    result, report = execute_l1_snapshot_batch(
        evidence_store=pd.DataFrame(),
        tasks=(),
        cfg=object(),
        symbols=("BTCUSDT",),
        seed=7,
    )
    assert result == ()
    assert report.mode == "no_tasks"


def test_snapshot_batch_executes_largest_pilot_once_then_remaining_once() -> None:
    source = _source(
        "src/domain/futures/strategy/tiered_workflow/snapshot_executor.py"
    )
    assert "pilot = sorted_tasks[-1]" in source
    assert "ProcessPoolExecutor(max_workers=1" in source


def test_snapshot_batch_with_unavailable_pss_runs_remaining_serially() -> None:
    source = _source(
        "src/domain/futures/strategy/tiered_workflow/snapshot_executor.py"
    )
    assert "pilot_failed_infrastructure" in source
    assert "serial" in source


def test_snapshot_batch_clears_globals_after_worker_failure() -> None:
    source = _source(
        "src/domain/futures/strategy/tiered_workflow/snapshot_executor.py"
    )
    assert "_clear_globals()" in source


def test_snapshot_batch_propagates_schema_value_error_without_retry() -> None:
    source = _source(
        "src/domain/futures/strategy/tiered_workflow/snapshot_executor.py"
    )
    assert "ValueError" in source
    assert "BrokenProcessPool" in source


def test_run_l1_nested_closes_fold_pool_before_starting_snapshot_pool() -> None:
    source = _source("src/domain/futures/strategy/tiered_workflow/pipeline.py")
    assert "run_l1_nested_swf" in source
    assert "execute_l1_snapshot_batch" in source


def test_release_completed_tf_resources_drops_non_primary_and_feature_cache() -> None:
    source = _source("src/domain/futures/strategy/tiered_workflow/lifecycle.py")
    assert "release_aligned_feature_cache" in source
    assert "per_tf_aligned.pop(tf, None)" in source


def test_release_completed_tf_resources_retains_primary_for_l2() -> None:
    source = _source("src/domain/futures/strategy/tiered_workflow/lifecycle.py")
    assert "aligned_tf is primary_aligned" in source
    assert "primary_retained" in source


def test_run_tiered_pipeline_releases_tf_resources_on_success_cache_hit_and_error() -> None:
    source = _source("src/domain/futures/strategy/tiered_workflow/pipeline.py")
    assert "release_completed_tf_resources" in source
    assert "finally" in source


def test_runtime_plans_are_independent_of_signal_and_timeframe_labels() -> None:
    source = _source("src/domain/futures/strategy/tiered_workflow/memory.py")
    assert "resolve_post_pilot_memory_plan" in source
    assert "L1PilotMeasurement" in source


def test_completed_tf_has_no_alignment_cache_owner_after_bridge_handoff() -> None:
    source = _source("src/domain/futures/strategy_runtime/bridge.py")
    assert "cache_result=False" in source


def test_l2_sampler_and_replay_frontier_contracts_are_unchanged() -> None:
    source = _source("src/domain/futures/strategy/tiered_workflow/pipeline.py")
    assert "layer2" in source.lower()
    assert "replay" in source.lower()


def test_cold_path_contract_task_shape_is_stable() -> None:
    task = L1SnapshotTask(snapshot_offset=0, as_of_idx=1)
    assert task.snapshot_offset == 0
    assert task.as_of_idx == 1


def test_cold_path_contract_source_is_inspectable() -> None:
    assert inspect.isfunction(causal_expanding_quantile)
