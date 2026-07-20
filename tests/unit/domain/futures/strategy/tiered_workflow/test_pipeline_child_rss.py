from __future__ import annotations

import multiprocessing
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.pipeline import _get_child_peak_rss_mb, run_l1_nested_swf
from src.domain.futures.strategy.walk_forward import WFFold


def _spawn_and_exit_child() -> None:
    pass


@pytest.fixture
def minimal_aligned() -> Any:
    aligned = SimpleNamespace()
    aligned.close_2d = np.ones((16, 1), dtype=np.float64)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(16)],
        dtype="datetime64[ns]",
    )
    aligned.symbols = ("BTC",)
    aligned.beta_vs_market_1d = None
    aligned.active_mask = None
    return aligned


@pytest.fixture
def minimal_cfg() -> CandidateStrategyConfig:
    return CandidateStrategyConfig(
        wf_n_folds=2,
        l1_min_signals_per_symbol=1,
        l1_signal_activation_floor_bps=0.0,
        l1_bootstrap_block_bars=6,
        l1_bootstrap_samples=200,
        l1_pair_alpha=0.05,
        l1_pair_power=0.80,
    )


@pytest.fixture
def empty_fold_out() -> SimpleNamespace:
    return SimpleNamespace(
        fit_status="trained",
        timing_profile={
            "schema": 0.01,
            "dataset_fit": 0.05,
            "dataset_early_stop": 0.02,
            "dataset_calibration_fit": 0.03,
            "dataset_calibration_eval": 0.01,
            "dataset_oos": 0.02,
            "edge_fit": 0.10,
            "inference": 0.08,
            "selection": 0.04,
            "total": 0.36,
        },
        model_output=SimpleNamespace(
            events=pd.DataFrame(),
            expected_gross_bps=np.zeros((0,), dtype=np.float64),
            expected_net_bps=np.zeros((0,), dtype=np.float64),
            q10_gross_bps=np.zeros((0,), dtype=np.float64),
            q90_gross_bps=np.zeros((0,), dtype=np.float64),
            q10_net_bps=np.zeros((0,), dtype=np.float64),
            q90_net_bps=np.zeros((0,), dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            edge_weight=np.zeros((0,), dtype=np.float64),
            y_return_bps=np.zeros((0,), dtype=np.float64),
        ),
    )


def _run_l1_nested(aligned: Any, cfg: Any, empty_fold_out: Any) -> Any:
    import concurrent.futures

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch(
            "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold",
            return_value=empty_fold_out,
        ),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table",
            return_value="gate",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table",
            return_value="outer",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table",
            return_value="reg",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline._prefit_layer1_from_globals",
            return_value=None,
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.snapshot_executor.execute_l1_snapshot_batch",
            return_value=([], SimpleNamespace(mode="mock", resolved_workers=1)),
        ),
    ):
        return run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            outer_folds=outer_folds,
            cfg=cfg,
            seed=3,
        )


def test_get_child_peak_rss_mb_returns_positive_after_child_exits() -> None:
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_spawn_and_exit_child)
    proc.start()
    proc.join()
    result = _get_child_peak_rss_mb()
    assert result >= 0.0


def test_get_child_peak_rss_mb_returns_nonnegative_with_no_children() -> None:
    result = _get_child_peak_rss_mb()
    assert result >= 0.0


def test_get_child_peak_rss_mb_returns_negative_one_on_exception() -> None:
    with patch("resource.getrusage", side_effect=OSError("boom")):
        result = _get_child_peak_rss_mb()
    assert result == -1.0


def test_get_child_peak_rss_mb_returns_positive_after_child_exits_active_pipeline() -> None:
    from src.application.futures.runner.active_pipeline import _get_child_peak_rss_mb as _ap_func

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_spawn_and_exit_child)
    proc.start()
    proc.join()
    result = _ap_func()
    assert result >= 0.0


def test_get_child_peak_rss_mb_returns_nonnegative_with_no_children_active_pipeline() -> None:
    from src.application.futures.runner.active_pipeline import _get_child_peak_rss_mb as _ap_func

    result = _ap_func()
    assert result >= 0.0


def test_get_child_peak_rss_mb_returns_negative_one_on_exception_active_pipeline() -> None:
    from src.application.futures.runner.active_pipeline import _get_child_peak_rss_mb as _ap_func

    with patch("resource.getrusage", side_effect=OSError("boom")):
        result = _ap_func()
    assert result == -1.0


def test_run_l1_nested_swf_wires_child_peak_rss(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, mocker: Any,
) -> None:
    import src.domain.futures.strategy.tiered_workflow.pipeline as _pl_mod

    spy = mocker.spy(_pl_mod, "_get_child_peak_rss_mb")
    _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    assert spy.call_count == 2
    for ret in spy.spy_return_list:
        assert isinstance(ret, float)
        assert ret >= 0.0 or ret == -1.0
