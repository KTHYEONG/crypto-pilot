from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import run_candidate_strategy_for_universe


def test_bridge_passes_no_leak_recommendation_window(monkeypatch: Any) -> None:
    datetimes = np.asarray(
        [np.datetime64("2026-01-01T00:00:00") + np.timedelta64(i, "h") for i in range(20)],
        dtype="datetime64[ns]",
    )
    aligned = SimpleNamespace(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        close_2d=np.linspace(100.0, 110.0, 20, dtype=np.float64).reshape(20, 1),
        open_2d=np.linspace(100.0, 110.0, 20, dtype=np.float64).reshape(20, 1),
        high_2d=np.linspace(101.0, 111.0, 20, dtype=np.float64).reshape(20, 1),
        low_2d=np.linspace(99.0, 109.0, 20, dtype=np.float64).reshape(20, 1),
        volume_2d=np.full((20, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((20, 1), dtype=np.float64),
        active_mask=np.ones((20, 1), dtype=bool),
        warm_mask=np.ones((20, 1), dtype=bool),
        entry_block_mask=np.zeros((20, 1), dtype=bool),
        kill_mask=np.zeros((20, 1), dtype=bool),
        execution_cost_bps_2d=np.zeros((20, 1), dtype=np.float64),
    )
    raw_events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[0]],
            "symbol": ["BTCUSDT"],
            "family": ["trend_ma"],
            "variant": ["ema_12_72"],
            "side": [1],
            "raw_score": [0.9],
            "score_z": [0.9],
            "entry_idx": [0],
            "expected_holding_bars": [1],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "turnover_proxy": [0.1],
            "cost_floor_bps": [0.0],
            "hurdle_bps": [0.0],
        }
    )
    labeled = raw_events.copy()

    captured: dict[str, Any] = {}

    def fake_align_data_maps(*_: Any, **__: Any) -> Any:
        return aligned

    def fake_build_rule_signal_panels(*_: Any, **__: Any) -> dict[str, Any]:
        return {"panel": "dummy"}

    def fake_candidate_panels_to_events(*_: Any, **__: Any) -> pd.DataFrame:
        return raw_events.copy()

    def fake_label_candidate_events(*_: Any, **__: Any) -> pd.DataFrame:
        return labeled.copy()

    def fake_compute_rule_diagnostics(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0),
            report_split=(0, 0),
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.common.alignment.align_data_maps",
        fake_align_data_maps,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
        fake_build_rule_signal_panels,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        fake_candidate_panels_to_events,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_labels.label_candidate_events",
        fake_label_candidate_events,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        fake_compute_rule_diagnostics,
    )

    strategy_cfg = StrategyConfig()
    object.__setattr__(
        strategy_cfg,
        "candidate",
        replace(
            strategy_cfg.candidate,
            ml_fit_fraction=0.5,
            ml_calibration_fraction=0.2,
            purge_bars=0,
            embargo_bars=0,
        ),
    )

    result = run_candidate_strategy_for_universe(
        ["BTCUSDT"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"BTCUSDT": {"4h": pd.DataFrame()}},
    )

    assert captured["recommendation_start"] == 0
    assert captured["recommendation_end"] == 14
    assert captured["report_start"] == 14
    assert captured["report_end"] == 20
    assert result.rule_report is not None
    assert result.rule_report["recommended_keep_variants"] == ()
    assert result.rule_report["recommended_flip_variants"] == ()
