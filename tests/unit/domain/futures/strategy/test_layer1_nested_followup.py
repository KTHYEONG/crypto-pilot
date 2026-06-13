from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput, EdgeSource
from src.domain.futures.strategy.tiered_workflow import run_l1_nested_swf
from src.domain.futures.strategy.walk_forward import WFFold


def test_run_l1_nested_swf_emits_new_runtime_tables() -> None:
    aligned = MagicMock()
    aligned.close_2d = np.ones((16, 1), dtype=np.float64)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(16)],
        dtype="datetime64[ns]",
    )
    aligned.symbols = ("BTC",)

    cfg = MagicMock()
    cfg.wf_n_folds = 2
    cfg.l1_min_signals_per_symbol = 1
    cfg.l1_signal_activation_floor_bps = 0.0

    empty_out = SimpleNamespace(
        fit_status="trained",
        model_output=SimpleNamespace(
            events=pd.DataFrame(),
            expected_gross_bps=np.zeros((0,), dtype=np.float64),
            q10_gross_bps=np.zeros((0,), dtype=np.float64),
            q90_gross_bps=np.zeros((0,), dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            edge_weight=np.zeros((0,), dtype=np.float64),
            y_return_bps=np.zeros((0,), dtype=np.float64),
        ),
    )
    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)

    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch("src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold", return_value=empty_out),
        patch("src.domain.futures.strategy.tiered_workflow.format_layer1_gate_table", return_value="gate-table"),
        patch("src.domain.futures.strategy.tiered_workflow.format_layer1_outer_fold_table", return_value="outer-table"),
        patch(
            "src.domain.futures.strategy.tiered_workflow.format_layer1_deployment_registry_table",
            return_value="registry-table",
        ),
        patch("src.domain.futures.strategy.tiered_workflow.logger.info") as mock_log,
    ):
        result = run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            outer_folds=outer_folds,
            cfg=cfg,
            seed=3,
        )

    assert result.gate_passed is False
    logged = [call.args[0] for call in mock_log.call_args_list]
    assert "gate-table" in logged
    assert "outer-table" in logged
    assert "registry-table" not in logged


def test_candidate_model_output_gross_fields_do_not_fallback_from_net() -> None:
    output = CandidateModelOutput(
        events=pd.DataFrame({"symbol": ["BTCUSDT"]}),
        p_pass=np.asarray([1.0], dtype=np.float64),
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_net_bps=np.asarray([12.0], dtype=np.float64),
        q10_net_bps=np.asarray([-4.0], dtype=np.float64),
        q90_net_bps=np.asarray([20.0], dtype=np.float64),
    )

    assert output.expected_gross_bps[0] == pytest.approx(0.0)
    assert output.q10_gross_bps[0] == pytest.approx(0.0)
    assert output.q90_gross_bps[0] == pytest.approx(0.0)


def test_candidate_model_output_net_fields_do_not_fallback_from_gross() -> None:
    output = CandidateModelOutput(
        events=pd.DataFrame({"symbol": ["BTCUSDT"]}),
        p_pass=np.asarray([1.0], dtype=np.float64),
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_gross_bps=np.asarray([7.0], dtype=np.float64),
        q10_gross_bps=np.asarray([2.0], dtype=np.float64),
        q90_gross_bps=np.asarray([10.0], dtype=np.float64),
    )

    assert output.expected_net_bps[0] == pytest.approx(0.0)
    assert output.q10_net_bps[0] == pytest.approx(0.0)
    assert output.q90_net_bps[0] == pytest.approx(0.0)
