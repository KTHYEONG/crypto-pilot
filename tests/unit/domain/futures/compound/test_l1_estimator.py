from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.config import L1EstimatorConfig
from src.domain.futures.compound.contracts import (
    AlphaLifecycle,
    AlphaLifecycleEvidence,
    MarketFeatureCube,
    RawAlphaTape,
)
from src.domain.futures.compound.l1_estimator import (
    build_causal_alpha_folds,
    estimate_cross_fitted_alpha_tape,
    update_alpha_lifecycle,
)


class TestBuildCausalAlphaFolds:
    def test_returns_correct_number_of_folds(self) -> None:
        folds = build_causal_alpha_folds(
            n_bars=1000, fit_start=100, holdout_start=800,
            n_folds=4, purge_bars=25, embargo_bars=1,
        )
        assert len(folds) == 4

    def test_purges_all_overlapping_labels(self) -> None:
        folds = build_causal_alpha_folds(
            n_bars=1000, fit_start=100, holdout_start=800,
            n_folds=4, purge_bars=25, embargo_bars=1,
        )
        for fold in folds:
            assert fold.oos_start >= fold.fit_end_exclusive


class TestEstimateCrossFittedAlphaTape:
    @pytest.fixture
    def raw_and_cube(self) -> tuple[RawAlphaTape, MarketFeatureCube]:
        n_bars, n_syms, n_recipes = 500, 2, 2
        timestamps = np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000
        symbols = ("BTCUSDT", "ETHUSDT")
        recipe_ids = ("trend_h4", "carry_h12")
        np.random.seed(42)
        scores = np.random.randn(n_bars, n_syms, n_recipes).astype(np.float32)
        raw = RawAlphaTape(
            timestamps_ns=timestamps,
            symbols=symbols,
            recipe_ids=recipe_ids,
            scores_3d=scores,
            valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
            horizon_bars_1d=np.array([4, 12], dtype=np.int16),
        )
        cube = MarketFeatureCube(
            timestamps_ns=timestamps,
            symbols=symbols,
            fields_2d={
                "open": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
                "high": np.ones((n_bars, n_syms), dtype=np.float64) * 101,
                "low": np.ones((n_bars, n_syms), dtype=np.float64) * 99,
                "close": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
                "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32),
                "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
                "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            },
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="h1",
        )
        return raw, cube

    def test_does_not_use_oos_outcome(self, raw_and_cube: tuple[RawAlphaTape, MarketFeatureCube]) -> None:
        raw, cube = raw_and_cube
        folds = build_causal_alpha_folds(
            n_bars=raw.timestamps_ns.size, fit_start=50, holdout_start=400,
            n_folds=2, purge_bars=25, embargo_bars=1,
        )
        config = L1EstimatorConfig()
        tape = estimate_cross_fitted_alpha_tape(raw=raw, cube=cube, folds=folds, config=config)
        assert tape.gross_mu_3d.shape == raw.scores_3d.shape
        assert tape.forecast_var_3d.shape == raw.scores_3d.shape


class TestUpdateAlphaLifecycle:
    def test_when_evidence_uncertain_keeps_active(self) -> None:
        current = (AlphaLifecycle.ACTIVE,)
        evidence = (AlphaLifecycleEvidence(
            recipe_id="0", effective_n=30, probability_net_positive=0.55,
            consecutive_negative_versions=1, data_valid=True,
        ),)
        config = L1EstimatorConfig(active_effective_n=20)
        result = update_alpha_lifecycle(current=current, evidence=evidence, config=config)
        assert result[0] is AlphaLifecycle.ACTIVE

    def test_when_three_versions_negative_retires(self) -> None:
        current = (AlphaLifecycle.ACTIVE,)
        evidence = (AlphaLifecycleEvidence(
            recipe_id="0", effective_n=60, probability_net_positive=0.15,
            consecutive_negative_versions=3, data_valid=True,
        ),)
        config = L1EstimatorConfig(active_effective_n=20, retire_effective_n=60)
        result = update_alpha_lifecycle(current=current, evidence=evidence, config=config)
        assert result[0] is AlphaLifecycle.RETIRED

    def test_when_data_invalid_retires(self) -> None:
        current = (AlphaLifecycle.ACTIVE,)
        evidence = (AlphaLifecycleEvidence(
            recipe_id="0", effective_n=50, probability_net_positive=0.5,
            consecutive_negative_versions=0, data_valid=False,
        ),)
        config = L1EstimatorConfig()
        result = update_alpha_lifecycle(current=current, evidence=evidence, config=config)
        assert result[0] is AlphaLifecycle.RETIRED


def test_build_causal_alpha_folds_purges_all_overlapping_labels() -> None:
    folds = build_causal_alpha_folds(
        n_bars=1000, fit_start=100, holdout_start=800,
        n_folds=4, purge_bars=25, embargo_bars=1,
    )
    assert len(folds) == 4
    for fold in folds:
        assert fold.oos_start >= fold.fit_end_exclusive


def test_estimate_cross_fitted_alpha_tape_does_not_use_oos_outcome() -> None:
    n_bars, n_syms, n_recipes = 500, 2, 2
    timestamps = np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000
    symbols = ("BTCUSDT", "ETHUSDT")
    recipe_ids = ("trend_h4", "carry_h12")
    np.random.seed(42)
    scores = np.random.randn(n_bars, n_syms, n_recipes).astype(np.float32)
    raw = RawAlphaTape(
        timestamps_ns=timestamps, symbols=symbols, recipe_ids=recipe_ids,
        scores_3d=scores, valid_3d=np.ones((n_bars, n_syms, n_recipes), dtype=np.bool_),
        horizon_bars_1d=np.array([4, 12], dtype=np.int16),
    )
    cube = MarketFeatureCube(
        timestamps_ns=timestamps, symbols=symbols,
        fields_2d={
            "open": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
            "high": np.ones((n_bars, n_syms), dtype=np.float64) * 101,
            "low": np.ones((n_bars, n_syms), dtype=np.float64) * 99,
            "close": np.ones((n_bars, n_syms), dtype=np.float64) * 100,
            "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32),
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="h1",
    )
    folds = build_causal_alpha_folds(
        n_bars=n_bars, fit_start=50, holdout_start=400, n_folds=2, purge_bars=25, embargo_bars=1,
    )
    config = L1EstimatorConfig()
    tape = estimate_cross_fitted_alpha_tape(raw=raw, cube=cube, folds=folds, config=config)
    assert tape.gross_mu_3d.shape == raw.scores_3d.shape
    assert tape.forecast_var_3d.shape == raw.scores_3d.shape


def test_update_alpha_lifecycle_when_evidence_uncertain_keeps_active() -> None:
    current = (AlphaLifecycle.ACTIVE,)
    evidence = (AlphaLifecycleEvidence(
        recipe_id="0", effective_n=30, probability_net_positive=0.55,
        consecutive_negative_versions=1, data_valid=True,
    ),)
    config = L1EstimatorConfig(active_effective_n=20)
    result = update_alpha_lifecycle(current=current, evidence=evidence, config=config)
    assert result[0] is AlphaLifecycle.ACTIVE


def test_update_alpha_lifecycle_when_three_versions_negative_retires() -> None:
    current = (AlphaLifecycle.ACTIVE,)
    evidence = (AlphaLifecycleEvidence(
        recipe_id="0", effective_n=60, probability_net_positive=0.15,
        consecutive_negative_versions=3, data_valid=True,
    ),)
    config = L1EstimatorConfig(active_effective_n=20, retire_effective_n=60)
    result = update_alpha_lifecycle(current=current, evidence=evidence, config=config)
    assert result[0] is AlphaLifecycle.RETIRED
