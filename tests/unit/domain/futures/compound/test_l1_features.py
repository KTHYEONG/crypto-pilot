from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.domain.futures.compound.contracts import (
    CausalFold,
    EdgeEvidence,
    ForecastFrame,
    MarketFeatureCube,
    MultiscaleAlphaDefinition,
)
from src.domain.futures.compound.config import L1MultiscaleConfig
from src.domain.futures.compound.l1_features import (
    InsufficientCoverageError,
    build_causal_forecasts,
)
from src.domain.futures.compound.l1_multiscale import (
    NoAdmissibleAlphaError,
    _build_alpha_events,
    run_l1_multiscale,
)
from src.domain.futures.compound.contracts import AlphaCandidateState


def _market() -> MarketFeatureCube:
    n = 40
    close = np.linspace(100.0, 110.0, n, dtype=np.float64).reshape(-1, 1)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n, dtype=np.int64),
        symbols=("BTCUSDT",),
        fields_2d={
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "quote_volume": np.ones((n, 1), dtype=np.float64),
        },
        available_2d={},
        eligible_2d=np.ones((n, 1), dtype=np.bool_),
        entry_block_2d=np.zeros((n, 1), dtype=np.bool_),
        exit_required_2d=np.zeros((n, 1), dtype=np.bool_),
        capacity_usdt_2d=np.ones((n, 1), dtype=np.float64),
        execution_cost_bps_2d=np.ones((n, 1), dtype=np.float32),
        data_manifest_hash="data-v1",
    )


def _recipe() -> MultiscaleAlphaDefinition:
    return MultiscaleAlphaDefinition(
        recipe_id="trend-test",
        family="trend",
        native_timeframe="4h",
        lookback_hours=(4,),
        horizon_hours=1,
        required_fields=("close",),
        initial_state=AlphaCandidateState.CORE_CANDIDATE,
        max_half_life_hours=1.0,
    )


def test_build_causal_forecasts_returns_one_frame() -> None:
    folds = (CausalFold(1, 0, 10, 8, 9, 10, 20, 1, 1),)
    frames = build_causal_forecasts(
        market=_market(), catalog=(_recipe(),), folds=folds
    )
    assert len(frames) == 1
    assert frames[0].recipe_id == "trend-test"
    assert frames[0].valid_2d.shape == (40, 1)


def test_build_causal_forecasts_rejects_overlapping_fold() -> None:
    folds = (CausalFold(1, 0, 11, 8, 9, 10, 20, 1, 1),)
    with pytest.raises(InsufficientCoverageError):
        build_causal_forecasts(
            market=_market(), catalog=(_recipe(),), folds=folds
        )


def test_build_alpha_events_skips_empty_decision_rows() -> None:
    forecast = ForecastFrame(
        timestamps_ns=np.array([1, 2], dtype=np.int64),
        symbols=("BTCUSDT",),
        recipe_id="trend-test",
        scores_2d=np.array([[np.nan], [1.0]], dtype=np.float32),
        valid_2d=np.array([[False], [True]], dtype=np.bool_),
    )
    evidence = EdgeEvidence(
        recipe_id="trend-test",
        outer_folds=5,
        positive_folds=5,
        effective_days=200.0,
        effective_events=1000,
        net_growth_lcb90=0.01,
        doubled_cost_growth=0.005,
        probability_positive=0.9,
        sign_consistency=0.9,
        fdr_q_value=0.01,
        max_residual_correlation=0.0,
        incremental_growth_lcb90=0.01,
        capacity_feasible=True,
        admitted=True,
        reasons=(),
    )
    table = _build_alpha_events(
        (forecast,), [evidence], (_recipe(),), L1MultiscaleConfig()
    )
    assert table.num_rows == 1


def test_run_l1_rejects_admitted_recipe_without_valid_events() -> None:
    base = _market()
    market = replace(
        base,
        timestamps_ns=np.arange(320, dtype=np.int64),
        fields_2d={
            key: np.repeat(value, 8, axis=0)
            for key, value in base.fields_2d.items()
        },
        eligible_2d=np.zeros((320, 1), dtype=np.bool_),
    )
    with pytest.raises(NoAdmissibleAlphaError):
        run_l1_multiscale(
            market=market,
            universe=object(),
            catalog=(_recipe(),),
            config=L1MultiscaleConfig(),
        )


def test_run_l1_rejects_short_history() -> None:
    with pytest.raises(InsufficientCoverageError):
        run_l1_multiscale(
            market=_market(),
            universe=object(),
            catalog=(_recipe(),),
            config=L1MultiscaleConfig(),
        )
