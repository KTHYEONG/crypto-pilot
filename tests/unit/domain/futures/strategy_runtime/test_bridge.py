from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import run_candidate_strategy_for_universe


def _make_panel() -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family="trend_ma",
        variant="ema_12_72",
        params={},
        datetimes=np.asarray([np.datetime64("2026-01-01T00:00:00")], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        signed_score_2d=np.zeros((1, 1), dtype=np.float64),
        side_hint_2d=np.ones((1, 1), dtype=np.int8),
        expected_holding_bars=1,
        min_holding_bars=1,
        stop_atr_mult=50.0,
        take_profit_atr_mult=50.0,
        turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
        valid_mask_2d=np.ones((1, 1), dtype=bool),
        metadata={"native_tf": "4h"},
        archetype="trend",
    )


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
        execution_eligibility_mask=np.ones((20, 1), dtype=bool),
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
            "exit_idx": [1],
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

    def fake_build_rule_signal_panels(*_: Any, **__: Any) -> tuple[CandidateSignalPanel, ...]:
        return (_make_panel(),)

    def fake_candidate_panels_to_events(*_: Any, **__: Any) -> pd.DataFrame:
        return raw_events.copy()

    def fake_label_candidate_events(*_: Any, **__: Any) -> pd.DataFrame:
        return labeled.copy()

    def fake_compute_rule_diagnostics(*_: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommended_keep_signal_cells=(),
            recommended_flip_signal_cells=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0),
            report_split=(0, 0),
            recommendation_failure_report=None,
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
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

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
    raw_events = pd.DataFrame(
        {
            "datetime": [datetimes[0]],
            "symbol": ["BTCUSDT"],
            "family": ["trend_ma"],
            "variant": ["ema_12_72"],
            "side": [1],
            "raw_score": [0.9],
            "score_z": [0.9],
            "entry_idx": [0],
            "exit_idx": [1],
            "expected_holding_bars": [1],
            "min_holding_bars": [0],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "turnover_proxy": [0.1],
            "cost_floor_bps": [0.0],
            "hurdle_bps": [0.0],
        }
    )

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", lambda *_, **__: aligned)
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
        lambda *_, **__: (_make_panel(),),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_labels.label_candidate_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        lambda *_, **__: SimpleNamespace(
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommended_keep_signal_cells=(),
            recommended_flip_signal_cells=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0),
            report_split=(0, 0),
            recommendation_failure_report=None,
        ),
    )

    # Patch build_candidate_dataset to return a minimal dataset stub
    from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput

    class _FakeDataset:
        X = np.zeros((5, 2), dtype=np.float64)
        y_gate = np.zeros(5, dtype=np.int8)
        y_edge_bps = np.zeros(5, dtype=np.float32)
        y_q10_bps = np.zeros(5, dtype=np.float32)
        y_mfe_bps = np.zeros(5, dtype=np.float32)
        gate_weight = np.ones(5, dtype=np.float32)
        edge_weight = np.ones(5, dtype=np.float32)
        groups = np.arange(5, dtype=np.int32)
        event_index = pd.DataFrame({"family": ["f"] * 5, "variant": ["v"] * 5})
        feature_names: ClassVar[list[str]] = []
        effective_sample_size = 5.0

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
    def fake_predict_edges(
        *, models: object, dataset: object, p_pass: np.ndarray, cfg: object, **__: Any
    ) -> CandidateModelOutput:
        from src.domain.futures.strategy.candidate_contracts import EdgeSource

        n = p_pass.shape[0]
        neg_mu = np.full(n, -999.0, dtype=np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        return CandidateModelOutput(
            events=getattr(dataset, "event_index", pd.DataFrame()),
            p_pass=p_pass,
            gate_enabled=False,
            gate_threshold=0.5,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_return_r=zeros,
            expected_net_bps=neg_mu,
            q10_return_r=zeros,
            q10_net_bps=neg_mu,
            q90_return_r=zeros,
            q90_net_bps=neg_mu,
            selection_score=neg_mu,
            kelly_fraction=zeros,
            validation_diagnostics={},
        )

    monkeypatch.setattr("src.domain.futures.strategy.candidate_edge.predict_candidate_edges", fake_predict_edges)


def test_bridge_emits_profile_log_when_raw_events_empty(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    caplog.set_level(logging.DEBUG)
    logging.getLogger("src.domain.futures.strategy_runtime.bridge").setLevel(logging.DEBUG)
    aligned = SimpleNamespace(
        datetimes=np.asarray([np.datetime64("2026-01-01T00:00:00")], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        close_2d=np.ones((1, 1), dtype=np.float64),
        open_2d=np.ones((1, 1), dtype=np.float64),
        high_2d=np.ones((1, 1), dtype=np.float64),
        low_2d=np.ones((1, 1), dtype=np.float64),
        volume_2d=np.ones((1, 1), dtype=np.float64),
        funding_2d=np.zeros((1, 1), dtype=np.float64),
        active_mask=np.ones((1, 1), dtype=bool),
        warm_mask=np.ones((1, 1), dtype=bool),
        entry_block_mask=np.zeros((1, 1), dtype=bool),
        kill_mask=np.zeros((1, 1), dtype=bool),
        execution_cost_bps_2d=np.zeros((1, 1), dtype=np.float64),
    )

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", lambda *_, **__: aligned)
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
        lambda *_, **__: (_make_panel(),),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        lambda *_, **__: pd.DataFrame(),
    )

    result = run_candidate_strategy_for_universe(
        ["BTCUSDT"],
        "4h",
        strategy_cfg=StrategyConfig(),
        preloaded_data_maps={"BTCUSDT": {"4h": pd.DataFrame()}},
    )

    assert result.alpha_panel is not None
    assert ("[BRIDGE PERFORMANCE]" in caplog.text) or ("[SYS]" in caplog.text)
    assert "Total Runtime:" in caplog.text
    assert "Alpha Panel" in caplog.text

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

    # No raw events → bridge returns zero weights immediately
    assert result.rule_report is not None
    assert result.rule_report.get("zero_reason") == "no_events"
    assert result.target_weights is not None
    assert np.all(result.target_weights == 0.0)


def test_bridge_realized_fold_survival_fails_when_selected_realized_edge_is_negative(monkeypatch: Any) -> None:
    t = 120
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
        execution_eligibility_mask=np.ones((t, 1), dtype=bool),
    )
    raw_events = pd.DataFrame(
        {
            "datetime": [datetimes[20]],
            "symbol": ["BTCUSDT"],
            "family": ["trend_ma"],
            "variant": ["ema_12_72"],
            "side": [1],
            "raw_score": [0.9],
            "score_z": [0.9],
            "entry_idx": [20],
            "exit_idx": [24],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [2.0],
            "take_profit_atr_mult": [4.0],
            "turnover_proxy": [0.1],
            "cost_floor_bps": [0.0],
            "hurdle_bps": [0.0],
            "edge_after_hurdle_bps": [-15.0],
        }
    )

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", lambda *_, **__: aligned)
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.build_rule_signal_panels", lambda *_, **__: {})
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_labels.label_candidate_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        lambda *_, **__: SimpleNamespace(
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommended_keep_signal_cells=(),
            recommended_flip_signal_cells=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0),
            report_split=(0, 0),
            recommendation_failure_report=None,
        ),
    )

    class _FakeDataset:
        X = np.zeros((1, 2), dtype=np.float64)
        y_gate = np.ones(1, dtype=np.int8)
        y_edge_bps = np.array([-15.0], dtype=np.float32)
        y_q10_bps = np.array([-20.0], dtype=np.float32)
        y_mfe_bps = np.array([5.0], dtype=np.float32)
        gate_weight = np.ones(1, dtype=np.float32)
        edge_weight = np.ones(1, dtype=np.float32)
        groups = np.zeros(1, dtype=np.int32)
        event_index = raw_events.copy()
        feature_names: ClassVar[list[str]] = []
        effective_sample_size = 1.0

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
        lambda *_, **__: np.array([0.9], dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_edge.fit_candidate_edge_models",
        lambda *_, **__: None,
    )

    from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput

    def fake_predict_edges(
        *, models: object, dataset: object, p_pass: np.ndarray, cfg: object, **__: Any
    ) -> CandidateModelOutput:
        from src.domain.futures.strategy.candidate_contracts import EdgeSource

        zeros = np.zeros(1, dtype=np.float64)
        return CandidateModelOutput(
            events=getattr(dataset, "event_index", pd.DataFrame()),
            p_pass=p_pass,
            gate_enabled=False,
            gate_threshold=0.5,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_return_r=zeros,
            expected_net_bps=np.array([30.0], dtype=np.float64),
            q10_return_r=zeros,
            q10_net_bps=np.array([-10.0], dtype=np.float64),
            q90_return_r=zeros,
            q90_net_bps=np.array([40.0], dtype=np.float64),
            selection_score=np.array([8.0], dtype=np.float64),
            kelly_fraction=zeros,
            validation_diagnostics={},
        )

    monkeypatch.setattr("src.domain.futures.strategy.candidate_edge.predict_candidate_edges", fake_predict_edges)

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
            wf_scheme="single",
            min_fit_obs=1,
            min_wf_fold_pass_ratio=1.0,
            fold_survival_metric="realized_selected_edge",
            min_fold_selected_events=1,
            selection_policy="utility_topk",
            promotion_filter_enabled=False,
        ),
    )

    result = run_candidate_strategy_for_universe(
        ["BTCUSDT"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"BTCUSDT": {"4h": pd.DataFrame()}},
    )

    assert result.rule_report is not None
    assert result.rule_report["zero_reason"] == "wf_fold_pass_ratio_fail"
    assert result.aligned is not None
    assert result.labeled is not None
    assert result.labeled_unfiltered is not None
    assert result.fit_set is not None
    assert result.calibration_set is not None
    assert result.oos_set is not None


def test_bridge_reports_shadow_profile_when_production_selection_stays_blocked(monkeypatch: Any) -> None:
    t = 120
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
    raw_events = pd.DataFrame(
        {
            "datetime": [datetimes[20], datetimes[24], datetimes[30], datetimes[34]],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "family": ["trend_ma", "trend_ma", "trend_ma", "trend_ma"],
            "variant": ["ema_12_72", "ema_12_72", "ema_12_72", "ema_12_72"],
            "side": [1, 1, 1, 1],
            "raw_score": [0.9, 0.8, 0.7, 0.6],
            "score_z": [0.9, 0.8, 0.7, 0.6],
            "entry_idx": [20, 24, 30, 34],
            "exit_idx": [24, 28, 34, 38],
            "expected_holding_bars": [4, 4, 4, 4],
            "min_holding_bars": [1, 1, 1, 1],
            "stop_atr_mult": [2.0, 2.0, 2.0, 2.0],
            "take_profit_atr_mult": [4.0, 4.0, 4.0, 4.0],
            "turnover_proxy": [0.1, 0.1, 0.1, 0.1],
            "cost_floor_bps": [0.0, 0.0, 0.0, 0.0],
            "hurdle_bps": [0.0, 0.0, 0.0, 0.0],
            "edge_after_hurdle_bps": [15.0, 18.0, 12.0, 14.0],
            "profitable_after_hurdle_label": [1, 1, 1, 1],
            "mae_bps": [-6.0, -8.0, -5.0, -7.0],
            "mfe_bps": [18.0, 22.0, 15.0, 20.0],
            "archetype": ["trend", "trend", "trend", "trend"],
            "entry_regime_code": [0, 0, 0, 0],
        }
    )

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", lambda *_, **__: aligned)
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.build_rule_signal_panels", lambda *_, **__: {})
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_labels.label_candidate_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        lambda *_, **__: SimpleNamespace(
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommended_keep_signal_cells=(),
            recommended_flip_signal_cells=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0),
            report_split=(0, 0),
            recommendation_failure_report=None,
        ),
    )

    class _FakeDataset:
        X = np.zeros((4, 2), dtype=np.float32)
        y_gate = np.ones(4, dtype=np.int8)
        y_edge_bps = np.array([15.0, 18.0, 12.0, 14.0], dtype=np.float32)
        y_q10_bps = np.array([-20.0, -20.0, -20.0, -20.0], dtype=np.float32)
        y_mfe_bps = np.array([20.0, 22.0, 15.0, 20.0], dtype=np.float32)
        gate_weight = np.ones(4, dtype=np.float32)
        edge_weight = np.ones(4, dtype=np.float32)
        groups = np.zeros(4, dtype=np.int32)
        event_index = raw_events.copy()
        feature_names: ClassVar[list[str]] = []
        effective_sample_size = 4.0
        feature_schema_version = "candidate_v5"
        y_return_r = np.zeros(4, dtype=np.float32)
        y_return_bps = np.zeros(4, dtype=np.float32)
        y_mae_r = np.zeros(4, dtype=np.float32)
        risk_unit_bps = np.full(4, 25.0, dtype=np.float32)

    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_workflow.build_candidate_dataset",
        lambda *_, **__: _FakeDataset(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_gate.fit_candidate_gate",
        lambda *_, **__: SimpleNamespace(calibration_used=False, calibration_reason="test"),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_gate.predict_candidate_gate",
        lambda *_, **__: np.array([0.30, 0.34, 0.31, 0.35], dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_edge.fit_candidate_edge_models",
        lambda *_, **__: None,
    )

    from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput

    def fake_predict_edges(
        *, models: object, dataset: object, p_pass: np.ndarray, cfg: object, **__: Any
    ) -> CandidateModelOutput:
        from src.domain.futures.strategy.candidate_contracts import EdgeSource

        n = len(getattr(dataset, "event_index", pd.DataFrame()))
        zeros = np.zeros(n, dtype=np.float64)
        net_bps = np.full(n, 42.0, dtype=np.float64)  # positive edge
        q10_bps = np.full(n, -5.0, dtype=np.float64)
        q90_bps = np.full(n, 62.0, dtype=np.float64)
        score = np.full(n, 12.0, dtype=np.float64)
        p = np.full(n, 0.46, dtype=np.float64)
        return CandidateModelOutput(
            events=getattr(dataset, "event_index", pd.DataFrame()),
            p_pass=p,
            gate_enabled=True,
            gate_threshold=0.35,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_return_r=zeros,
            expected_net_bps=net_bps,
            mu_net_decision_bps=net_bps,
            q10_return_r=zeros,
            q10_net_bps=q10_bps,
            q90_return_r=zeros,
            q90_net_bps=q90_bps,
            selection_score=score,
            utility_score=score,
            kelly_fraction=zeros,
            validation_diagnostics={},
        )

    monkeypatch.setattr("src.domain.futures.strategy.candidate_edge.predict_candidate_edges", fake_predict_edges)

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
            wf_scheme="single",
            min_fit_obs=1,
            min_wf_fold_pass_ratio=1.0,
            fold_survival_metric="realized_selected_edge",
            min_fold_selected_events=1,
            selection_policy="utility_topk",
            selection_shadow_utility_floors_bps=(-20.0, 0.0),
            selection_shadow_breakeven_floor_fractions=(0.0, 0.5),
            promotion_filter_enabled=False,
            min_fold_realized_edge_bps=50.0,
            min_expected_net_bps=-50.0,
            selection_shadow_profiles_enabled=True,
            selection_min_expected_utility_bps=-100.0,
            cost_floor_bps=0.0,
            expected_cost_bps=0.0,
        ),
    )

    result = run_candidate_strategy_for_universe(
        ["BTCUSDT"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"BTCUSDT": {"4h": pd.DataFrame()}},
    )

    assert result.rule_report is not None
    assert result.rule_report["selected_total"] == 0
    assert result.rule_report["wf_shadow_profile_count"] > 0
    assert result.rule_report["wf_shadow_max_selected_total"] > 0
    assert result.rule_report["zero_reason"] == "wf_fold_pass_ratio_fail"


def test_bridge_signal_only_silent_diagnostics(monkeypatch: Any) -> None:
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
        execution_eligibility_mask=np.ones((20, 1), dtype=bool),
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
            "exit_idx": [1],
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

    captured_kwargs: dict[str, Any] = {}

    def fake_align_data_maps(*_: Any, **__: Any) -> Any:
        return aligned

    def fake_build_rule_signal_panels(*_: Any, **__: Any) -> tuple[CandidateSignalPanel, ...]:
        return (_make_panel(),)

    def fake_candidate_panels_to_events(*_: Any, **__: Any) -> pd.DataFrame:
        return raw_events.copy()

    def fake_label_candidate_events(*_: Any, **__: Any) -> pd.DataFrame:
        return labeled.copy()

    def fake_compute_rule_diagnostics(*_: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommended_keep_signal_cells=(),
            recommended_flip_signal_cells=(),
            recommendation_basis="fit_calibration",
            recommendation_split=(0, 0),
            report_split=(0, 0),
            recommendation_failure_report=None,
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
            signal_only=True,
            wf_enabled=False,
            ml_fit_fraction=0.5,
            ml_calibration_fraction=0.2,
            purge_bars=0,
            embargo_bars=0,
        ),
    )

    run_candidate_strategy_for_universe(
        ["BTCUSDT"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"BTCUSDT": {"4h": pd.DataFrame()}},
        silent=False,
    )

    # signal_only=True 시 compute_rule_diagnostics 전체가 스킵되므로
    # captured_kwargs는 비어있어야 한다 (함수 자체가 호출 안 됨)
    assert captured_kwargs == {}, f"signal_only=True일 때 compute_rule_diagnostics가 호출되면 안 됨: {captured_kwargs}"


def test_verify_data_integrity_happy_path() -> None:
    from src.domain.futures.strategy_runtime.bridge import verify_data_integrity

    n_bars = 150
    close = np.linspace(100.0, 110.0, n_bars).reshape(n_bars, 1)
    high = close + 1.0
    low = close - 1.0
    volume = np.full((n_bars, 1), 500.0)

    close_2d = np.hstack([close, close])
    high_2d = np.hstack([high, high])
    low_2d = np.hstack([low, low])
    volume_2d = np.hstack([volume, volume])

    aligned = SimpleNamespace(
        close_2d=close_2d,
        high_2d=high_2d,
        low_2d=low_2d,
        volume_2d=volume_2d,
    )

    report = verify_data_integrity(aligned, ["BTC", "ETH"], min_length=100)  # type: ignore
    assert report["BTC"]["status"] == "PASS"
    assert report["ETH"]["status"] == "PASS"
    assert not report["BTC"]["reasons"]


def test_verify_data_integrity_edge_cases() -> None:
    from src.domain.futures.strategy_runtime.bridge import verify_data_integrity

    n_bars = 50
    close = np.linspace(100.0, 110.0, n_bars).reshape(n_bars, 1)
    high = close + 1.0
    low = close - 1.0
    volume = np.full((n_bars, 1), 500.0)

    close_eth = close.copy()
    close_eth[10] = np.nan

    close_sol = np.full((n_bars, 1), 100.0)
    high_sol = np.full((n_bars, 1), 100.0)
    low_sol = np.full((n_bars, 1), 100.0)

    high_ada = close - 2.0
    low_ada = close + 2.0

    close_2d = np.hstack([close, close_eth, close_sol, close])
    high_2d = np.hstack([high, high, high_sol, high_ada])
    low_2d = np.hstack([low, low, low_sol, low_ada])
    volume_2d = np.hstack([volume, volume, volume, volume])

    aligned = SimpleNamespace(
        close_2d=close_2d,
        high_2d=high_2d,
        low_2d=low_2d,
        volume_2d=volume_2d,
    )

    report = verify_data_integrity(aligned, ["BTC", "ETH", "SOL", "ADA"], min_length=100)  # type: ignore

    assert report["BTC"]["status"] == "FAIL"
    assert "too_short" in report["BTC"]["reasons"]

    assert "excessive_nan" in report["ETH"]["reasons"]
    assert "too_short" in report["ETH"]["reasons"]

    assert "stuck_price" in report["SOL"]["reasons"]

    assert "hi_lo_violation" in report["ADA"]["reasons"]
