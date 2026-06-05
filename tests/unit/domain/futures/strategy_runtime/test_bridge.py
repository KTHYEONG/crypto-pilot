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


def test_bridge_wf_fold_pass_ratio_blocks_when_all_folds_fail(monkeypatch: Any) -> None:
    """min_wf_fold_pass_ratio gate: 모든 폴드 cost survival 실패 시 zero weights 반환."""
    import numpy as np
    import pandas as pd
    from dataclasses import replace as dc_replace
    from types import SimpleNamespace
    from src.domain.futures.strategy.config import StrategyConfig
    from src.domain.futures.strategy_runtime.bridge import run_candidate_strategy_for_universe

    t = 200
    datetimes = np.asarray(
        [np.datetime64("2026-01-01T00:00:00") + np.timedelta64(i, "h") for i in range(t)],
        dtype="datetime64[ns]",
    )
    aligned = SimpleNamespace(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        close_2d=np.linspace(100.0, 110.0, t, dtype=np.float64).reshape(t, 1),
        open_2d=np.linspace(100.0, 110.0, t, dtype=np.float64).reshape(t, 1),
        high_2d=np.linspace(101.0, 111.0, t, dtype=np.float64).reshape(t, 1),
        low_2d=np.linspace(99.0, 109.0, t, dtype=np.float64).reshape(t, 1),
        volume_2d=np.full((t, 1), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=np.ones((t, 1), dtype=bool),
        warm_mask=np.ones((t, 1), dtype=bool),
        entry_block_mask=np.zeros((t, 1), dtype=bool),
        kill_mask=np.zeros((t, 1), dtype=bool),
        execution_cost_bps_2d=np.zeros((t, 1), dtype=np.float64),
    )
    raw_events = pd.DataFrame({
        "datetime": [datetimes[0]],
        "symbol": ["BTCUSDT"],
        "family": ["trend_ma"],
        "variant": ["ema_12_72"],
        "side": [1],
        "raw_score": [0.9],
        "score_z": [0.9],
        "entry_idx": [0],
        "expected_holding_bars": [1],
        "min_holding_bars": [0],
        "stop_atr_mult": [50.0],
        "take_profit_atr_mult": [50.0],
        "turnover_proxy": [0.1],
        "cost_floor_bps": [0.0],
        "hurdle_bps": [0.0],
    })

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", lambda *_, **__: aligned)
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.build_rule_signal_panels", lambda *_, **__: {})
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.candidate_panels_to_events", lambda *_, **__: raw_events.copy())
    monkeypatch.setattr("src.domain.futures.strategy.candidate_labels.label_candidate_events", lambda *_, **__: raw_events.copy())
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        lambda *_, **__: SimpleNamespace(
            recommended_keep_variants=(), recommended_flip_variants=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0), report_split=(0, 0),
        ),
    )

    # Patch build_candidate_dataset to return a minimal dataset stub
    from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput

    class _FakeDataset:
        X = np.zeros((5, 2), dtype=np.float64)
        y_gate = np.zeros(5, dtype=np.float64)
        y_edge = np.zeros(5, dtype=np.float64)
        event_index = pd.DataFrame({"family": ["f"] * 5, "variant": ["v"] * 5})
        feature_names: list[str] = []

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_dataset.build_candidate_dataset",
        lambda *_, **__: _FakeDataset(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_gate.fit_candidate_gate",
        lambda *_, **__: SimpleNamespace(calibration_used=False, calibration_reason="test"),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_gate.predict_candidate_gate",
        lambda *_, **__: np.full(5, 0.5, dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_edge.fit_candidate_edge_models",
        lambda *_, **__: None,
    )

    # Force all fold mu = -999bps (fails cost survival) by patching predict_candidate_edges
    def fake_predict_edges(*, models: object, dataset: object, p_pass: np.ndarray, cfg: object) -> CandidateModelOutput:
        n = p_pass.shape[0]
        neg_mu = np.full(n, -999.0, dtype=np.float64)
        return CandidateModelOutput(
            events=getattr(dataset, "event_index", pd.DataFrame()),
            p_pass=p_pass,
            mu_gross_bps=neg_mu,
            mu_net_decision_bps=neg_mu,  # all negative → cost survival fails
            q10_net_bps=neg_mu,
            q90_net_bps=neg_mu,
            utility_score=neg_mu,
            selection_thresholds={},
        )
    monkeypatch.setattr("src.domain.futures.strategy.candidate_edge.predict_candidate_edges", fake_predict_edges)

    strategy_cfg = StrategyConfig()
    object.__setattr__(
        strategy_cfg, "candidate",
        dc_replace(
            strategy_cfg.candidate,
            ml_fit_fraction=0.5,
            ml_calibration_fraction=0.2,
            purge_bars=0,
            embargo_bars=0,
            wf_scheme="anchored",
            wf_n_folds=2,
            min_fit_obs=1,
            min_wf_fold_pass_ratio=0.6,  # requires 60% folds to pass
            promotion_filter_enabled=False,  # bypass promo filter to reach WF gate
        ),
    )

    result = run_candidate_strategy_for_universe(
        ["BTCUSDT"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"BTCUSDT": {"4h": pd.DataFrame()}},
    )

    # All folds fail cost survival → gate blocks → zero weights
    assert result.rule_report is not None
    assert result.rule_report.get("zero_reason") == "wf_fold_pass_ratio_fail"
    assert result.target_weights is not None
    assert np.all(result.target_weights == 0.0)
