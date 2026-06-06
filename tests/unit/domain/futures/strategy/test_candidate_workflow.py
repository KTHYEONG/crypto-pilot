from __future__ import annotations

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
    assert outputs[0].model_output.edge_source == EdgeSource.DISABLED
    assert not outputs[0].gate_report.enabled
    assert not outputs[0].edge_report.selected
