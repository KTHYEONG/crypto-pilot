from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    CandidateFoldOutput,
    EdgeSource,
)
from src.domain.futures.strategy.candidate_workflow import run_candidate_walk_forward
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.walk_forward import WFFold


class DummyDataset:
    def __init__(self, size: int):
        self.X = np.random.randn(size, 5)
        self.y_edge_bps = np.random.randn(size)
        self.edge_weight = np.ones(size)
        self.event_index = pd.DataFrame({
            "datetime": pd.date_range("2026-01-01", periods=size, freq="4h"),
            "symbol": ["BTC"] * size,
            "entry_idx": np.arange(size),
            "family": ["trend_ma"] * size,
            "variant": ["fast"] * size,
            "side": [1.0] * size,
            "expected_holding_bars": [6] * size,
        })
        self.y_gate = np.ones(size, dtype=np.int8)
        self.gate_weight = np.ones(size)
        self.feature_names = ["f1", "f2", "f3", "f4", "f5"]
        self.y_return_r = np.random.randn(size)
        self.y_return_bps = np.random.randn(size)
        self.y_q10_bps = np.random.randn(size)
        self.y_mfe_bps = np.random.randn(size)
        self.risk_unit_bps = np.full(size, 25.0)
        self.groups = np.zeros(size, dtype=np.int32)
        self.effective_sample_size = float(size)


def test_run_candidate_walk_forward_prior_only_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup dummy AlignedMarketData
    size = 10
    dummy_aligned = AlignedMarketData(
        symbols=("BTC",),
        datetimes=np.array(pd.date_range("2026-01-01", periods=size, freq="4h")),
        open_2d=np.ones((size, 1)),
        high_2d=np.ones((size, 1)),
        low_2d=np.ones((size, 1)),
        close_2d=np.ones((size, 1)),
        volume_2d=np.ones((size, 1)),
        funding_2d=np.zeros((size, 1)),
        active_mask=np.ones((size, 1), dtype=bool),
        warm_mask=np.ones((size, 1), dtype=bool),
        entry_block_mask=np.zeros((size, 1), dtype=bool),
        kill_mask=np.zeros((size, 1), dtype=bool),
    )

    cfg = CandidateStrategyConfig(
        min_fit_obs=100,  # force fallback since size < 100
    )

    # Monkeypatch dataset builders to return DummyDataset
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.build_candidate_dataset",
        lambda *args, **kwargs: DummyDataset(10)
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.fit_candidate_feature_schema",
        lambda *args, **kwargs: {}
    )

    labeled_events = pd.DataFrame()
    folds = (
        WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=7, oos_start=7, oos_end=10),
    )

    outputs = run_candidate_walk_forward(
        labeled_events=labeled_events,
        aligned=dummy_aligned,
        cfg=cfg,
        folds=folds,
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], CandidateFoldOutput)
    assert outputs[0].fit_status == "insufficient_fit"
    assert outputs[0].n_fit == 10
    assert outputs[0].skip_reason == "insufficient_observations"
    assert outputs[0].model_output.edge_source == EdgeSource.DISABLED
    assert not outputs[0].gate_report.enabled
    assert not outputs[0].edge_report.selected
    assert outputs[0].timing_profile is not None
    assert set(outputs[0].timing_profile) >= {
        "schema",
        "dataset_fit",
        "dataset_early_stop",
        "dataset_calibration_fit",
        "dataset_calibration_eval",
        "dataset_oos",
        "gate_fit",
        "edge_fit",
        "inference",
        "selection",
        "total",
    }


def test_run_candidate_walk_forward_parallel_path_preserves_fold_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 12
    dummy_aligned = AlignedMarketData(
        symbols=("BTC",),
        datetimes=np.array(pd.date_range("2026-01-01", periods=size, freq="4h")),
        open_2d=np.ones((size, 1)),
        high_2d=np.ones((size, 1)),
        low_2d=np.ones((size, 1)),
        close_2d=np.ones((size, 1)),
        volume_2d=np.ones((size, 1)),
        funding_2d=np.zeros((size, 1)),
        active_mask=np.ones((size, 1), dtype=bool),
        warm_mask=np.ones((size, 1), dtype=bool),
        entry_block_mask=np.zeros((size, 1), dtype=bool),
        kill_mask=np.zeros((size, 1), dtype=bool),
    )
    cfg = CandidateStrategyConfig(min_fit_obs=100)

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.build_candidate_dataset",
        lambda *args, **kwargs: DummyDataset(6),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.fit_candidate_feature_schema",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr("src.domain.futures.strategy.candidate_workflow.os.cpu_count", lambda: 8)

    class _ImmediateFuture:
        def __init__(self, fn: Callable[..., CandidateFoldOutput], *args: object) -> None:
            self._result = fn(*args)

        def result(self) -> CandidateFoldOutput:
            return self._result

    class _FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self) -> _FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(
            self,
            fn: Callable[..., CandidateFoldOutput],
            *args: object,
        ) -> _ImmediateFuture:
            return _ImmediateFuture(fn, *args)

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", _FakeExecutor)

    folds = (
        WFFold(fit_start=0, fit_end=4, cal_start=4, cal_end=6, oos_start=6, oos_end=8),
        WFFold(fit_start=1, fit_end=5, cal_start=5, cal_end=7, oos_start=7, oos_end=10),
    )

    outputs = run_candidate_walk_forward(
        labeled_events=pd.DataFrame(),
        aligned=dummy_aligned,
        cfg=cfg,
        folds=folds,
    )

    assert [out.fold_id for out in outputs] == [0, 1]
    assert all(out.timing_profile is not None for out in outputs)


def test_allocation_backend_ensemble_skips_lgbm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    size = 12
    dummy_aligned = AlignedMarketData(
        symbols=("BTC",),
        datetimes=np.array(pd.date_range("2026-01-01", periods=size, freq="4h")),
        open_2d=np.ones((size, 1)),
        high_2d=np.ones((size, 1)),
        low_2d=np.ones((size, 1)),
        close_2d=np.ones((size, 1)),
        volume_2d=np.ones((size, 1)),
        funding_2d=np.zeros((size, 1)),
        active_mask=np.ones((size, 1), dtype=bool),
        warm_mask=np.ones((size, 1), dtype=bool),
        entry_block_mask=np.zeros((size, 1), dtype=bool),
        kill_mask=np.zeros((size, 1), dtype=bool),
    )
    cfg = CandidateStrategyConfig(min_fit_obs=2, allocation_backend="ensemble_b0")

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.build_candidate_dataset",
        lambda *args, **kwargs: DummyDataset(6),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.fit_candidate_feature_schema",
        lambda *args, **kwargs: {},
    )
    calls = {"ensemble_fit": 0, "ensemble_predict": 0}

    def _fit_stub(*args: object, **kwargs: object) -> object:
        calls["ensemble_fit"] += 1
        proof_events = kwargs["oos_proof_events"]
        proof_fold_ids = kwargs["fold_ids"]
        assert isinstance(proof_events, pd.DataFrame)
        assert isinstance(proof_fold_ids, np.ndarray)
        assert len(proof_events) == len(proof_fold_ids)
        return object()

    def _predict_stub(*args: object, **kwargs: object) -> object:
        calls["ensemble_predict"] += 1
        event_index = kwargs["oos_events"]
        assert isinstance(event_index, pd.DataFrame)
        from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput

        return CandidateModelOutput(
            events=event_index,
            p_pass=np.ones(len(event_index), dtype=np.float64),
            expected_net_bps=np.full(len(event_index), 3.0, dtype=np.float64),
            q10_net_bps=np.full(len(event_index), -2.0, dtype=np.float64),
            selection_score=np.full(len(event_index), 3.0, dtype=np.float64),
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.fit_regime_conditional_ensemble",
        _fit_stub,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.predict_regime_conditional_ensemble",
        _predict_stub,
    )

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("LGBM path must not be called for ensemble_b0")

    monkeypatch.setattr("src.domain.futures.strategy.candidate_edge.fit_candidate_edge_models", _forbidden)
    monkeypatch.setattr("src.domain.futures.strategy.candidate_edge.predict_candidate_edges", _forbidden)
    monkeypatch.setattr("src.domain.futures.strategy.candidate_gate.fit_candidate_gate", _forbidden)
    monkeypatch.setattr("src.domain.futures.strategy.candidate_gate.predict_candidate_gate", _forbidden)

    outputs = run_candidate_walk_forward(
        labeled_events=pd.DataFrame(),
        aligned=dummy_aligned,
        cfg=cfg,
        folds=(WFFold(fit_start=0, fit_end=4, cal_start=4, cal_end=6, oos_start=6, oos_end=10),),
    )

    assert calls == {"ensemble_fit": 1, "ensemble_predict": 1}
    assert outputs[0].model_output.expected_net_bps.shape[0] == 6


def test_empty_dataset_fallback_accepts_dict_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    size = 8
    dummy_aligned = AlignedMarketData(
        symbols=("BTC",),
        datetimes=np.array(pd.date_range("2026-01-01", periods=size, freq="4h")),
        open_2d=np.ones((size, 1)),
        high_2d=np.ones((size, 1)),
        low_2d=np.ones((size, 1)),
        close_2d=np.ones((size, 1)),
        volume_2d=np.ones((size, 1)),
        funding_2d=np.zeros((size, 1)),
        active_mask=np.ones((size, 1), dtype=bool),
        warm_mask=np.ones((size, 1), dtype=bool),
        entry_block_mask=np.zeros((size, 1), dtype=bool),
        kill_mask=np.zeros((size, 1), dtype=bool),
    )
    cfg = CandidateStrategyConfig(min_fit_obs=100, allocation_backend="ensemble_b0")

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.build_candidate_dataset",
        lambda *args, **kwargs: DummyDataset(4),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.fit_candidate_feature_schema",
        lambda *args, **kwargs: {},
    )

    outputs = run_candidate_walk_forward(
        labeled_events=pd.DataFrame(),
        aligned=dummy_aligned,
        cfg=cfg,
        folds=(WFFold(fit_start=0, fit_end=2, cal_start=2, cal_end=2, oos_start=2, oos_end=6),),
    )

    assert len(outputs) == 1
    assert outputs[0].fit_status == "insufficient_fit"
