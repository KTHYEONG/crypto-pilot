from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2GateEvaluation,
    Layer2TrialEvaluation,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import run_l2_awf


def test_awf_fold_diags_reports_real_mdd_not_zero() -> None:
    signal_batch = MagicMock()
    aligned = MagicMock()
    aligned.symbols = ("BTC",)
    aligned.close_2d = np.ones((10, 1), dtype=float)
    aligned.datetimes = np.array(
        [f"2024-01-{i:02d}" for i in range(1, 11)], dtype="datetime64[ns]"
    )

    from src.domain.futures.strategy.walk_forward import WFFold

    awf_folds = (
        WFFold(fit_start=0, fit_end=3, cal_start=0, cal_end=3, oos_start=3, oos_end=6),
        WFFold(fit_start=3, fit_end=6, cal_start=3, cal_end=6, oos_start=6, oos_end=9),
    )

    fake_eval = Layer2TrialEvaluation(
        objective_value=0.0,
        constraint_values=(),
        cagr_hybrid=0.1,
        cagr_baseline=0.0,
        growth_lcb_hybrid=0.0,
        growth_lcb_baseline=0.0,
        sharpe_hac_hybrid=1.0,
        sharpe_hac_baseline=0.0,
        psr_hybrid=0.0,
        mdd_hybrid=0.15,
        cvar_95_hybrid=0.0,
        fold_pass_ratio=1.0,
        break_even_pass_pct=1.0,
        average_gross_exposure=0.5,
        cap_saturation_ratio=0.0,
        total_cost_bps=0.0,
        block_metrics=(),
        returns_hybrid=tuple([0.01] * 10),
        returns_baseline=tuple([0.005] * 10),
        rets_baseline_ew=tuple([0.005] * 10),
        last_selected_symbols=("BTC",),
        last_weights=(1.0,),
        all_turnovers=tuple([0.1] * 10),
        rebalance_count=10,
        all_net_exposures=tuple([0.5] * 10),
        fold_deployed_cagrs=(0.2, 0.15),
        fold_deployed_mdds=(0.11, 0.17),
        fold_selected_symbols=(("BTC",), ("BTC",)),
        gate=Layer2GateEvaluation(
            optuna_constraint_values=(),
            promotion_passed=True,
            promotion_blocker="",
            promotion_constraint_values=(),
        ),
    )

    with (
        patch(
            "src.domain.futures.optimization.workflow.evaluate_l2_trial",
            return_value=fake_eval,
        ) as _mock_eval,
        patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        ) as mock_cache,
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer2_table",
        ) as spy_format,
    ):
        mock_cache.return_value = MagicMock()

        run_l2_awf(
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=Layer2AllocationConfig(),
            caps=PortfolioCaps(),
            tf="4h",
            verbose=True,
        )

    assert spy_format.called
    awf_folds_arg = spy_format.call_args.kwargs["awf_folds"]
    assert awf_folds_arg is not None
    assert len(awf_folds_arg) == 2
    assert awf_folds_arg[0]["mdd"] == pytest.approx(0.11)
    assert awf_folds_arg[1]["mdd"] == pytest.approx(0.17)
