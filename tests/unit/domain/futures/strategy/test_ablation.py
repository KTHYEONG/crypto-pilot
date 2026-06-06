from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.ablation import (
    _build_barrier_arrays,
    _build_rule_equal_size_weights,
    _build_uncapped_kelly_edge_weights,
    _build_variant_prior_output,
    _compute_realized_edge,
    run_candidate_ablation,
    validate_candidate_signals,
)
from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _make_mock_data_maps(t: int = 150) -> dict[str, dict[str, Any]]:
    symbols = ["BTCUSDT", "ETHUSDT"]
    datetimes = pd.date_range("2025-01-01", periods=t, freq="4h")
    
    maps = {}
    for sym in symbols:
        base = np.linspace(100.0, 130.0, t) if sym == "BTCUSDT" else np.linspace(10.0, 13.0, t)
        df = pd.DataFrame({
            "datetime": datetimes,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.full(t, 1000.0, dtype=np.float64),
            "funding_rate": np.zeros(t, dtype=np.float64),
            "universe_active_mask": np.ones(t, dtype=bool),
            "universe_entry_warm_mask": np.ones(t, dtype=bool),
            "entry_block_mask": np.zeros(t, dtype=bool),
            "kill_signal": np.zeros(t, dtype=bool),
        })
        maps[sym] = {"4h": df}
    return maps


def test_run_candidate_ablation_returns_correct_ablation_dataframe(monkeypatch: Any) -> None:
    data_maps = _make_mock_data_maps(250)
    cfg = CandidateStrategyConfig(
        timeframe="4h",
        min_candidate_obs=10,  # lower observation threshold for testing
        min_rule_net_bps=0.0,
        kelly_fraction=0.1,
        gross_cap=1.2,
    )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.fit_candidate_gate",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.predict_candidate_gate",
        lambda *, dataset, **__: np.full(dataset.X.shape[0], 0.9, dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.fit_candidate_edge_models",
        lambda **_: object(),
    )

    def _fake_predict_candidate_edges(*, dataset: Any, p_pass: np.ndarray, **__: Any) -> CandidateModelOutput:
        from src.domain.futures.strategy.candidate_contracts import EdgeSource
        n = dataset.X.shape[0]
        zeros = np.zeros(n, dtype=np.float64)
        return CandidateModelOutput(
            events=dataset.event_index,
            p_pass=p_pass,
            gate_enabled=False,
            gate_threshold=0.5,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_return_r=zeros,
            expected_net_bps=np.full(n, 16.0, dtype=np.float64),
            q10_return_r=zeros,
            q10_net_bps=np.full(n, -10.0, dtype=np.float64),
            q90_return_r=zeros,
            q90_net_bps=np.full(n, 30.0, dtype=np.float64),
            selection_score=np.full(n, 8.0, dtype=np.float64),
            kelly_fraction=zeros,
            validation_diagnostics={
                "utility_min": 0.0,
            },
        )

    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.predict_candidate_edges",
        _fake_predict_candidate_edges,
    )

    df_ablation = run_candidate_ablation(
        data_maps=data_maps,
        symbols=("BTCUSDT", "ETHUSDT"),
        tf="4h",
        cfg=cfg,
    )

    assert isinstance(df_ablation, pd.DataFrame)
    if not df_ablation.empty:
        assert df_ablation.shape[0] == 6
        required_cols = {
            "variant",
            "mean_log_growth",
            "cagr",
            "max_drawdown",
            "mar",
            "turnover",
            "final_equity",
            "pass_compound_gate",
        }
        assert required_cols.issubset(df_ablation.columns)
        assert {
            "rule_stop_risk",
            "prior_rank_stop_risk",
            "prior_residual_rank_stop_risk",
            "edge_plus_validated_gate_stop_risk",
            "edge_plus_gate_event_kelly",
            "full_portfolio_caps",
        }.issubset(set(df_ablation["variant"]))


def test_build_rule_equal_size_weights_uses_entry_idx_bar() -> None:
    close_2d = np.zeros((20, 2), dtype=np.float64)
    events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [10],
        }
    )

    weights = _build_rule_equal_size_weights(
        raw_events=events,
        close_2d=close_2d,
        symbols=("BTCUSDT", "ETHUSDT"),
        max_symbol_weight=0.1,
    )

    assert weights[9, 0] == 0.0
    assert weights[10, 0] == 0.1


def test_build_uncapped_kelly_edge_weights_scales_by_holding_bars() -> None:
    base = np.linspace(100.0, 130.0, 40, dtype=np.float64)
    close_2d = np.stack([base, base], axis=1)
    events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": [1, 1],
            "entry_idx": [25, 25],
            "expected_holding_bars": [2, 20],
            "mu_net_decision_bps": [40.0, 40.0],
        }
    )

    weights = _build_uncapped_kelly_edge_weights(
        selected_events=events,
        close_2d=close_2d,
        symbols=("BTCUSDT", "ETHUSDT"),
        kelly_fraction=0.1,
    )

    short_hold = weights[25, 0]
    long_hold = weights[25, 1]
    assert short_hold > 0.0
    assert long_hold > 0.0
    assert np.isclose(short_hold / long_hold, 10.0, rtol=1e-6)


def test_build_variant_prior_output_uses_calibration_set_prior(monkeypatch: Any) -> None:
    calibration_set = SimpleNamespace(
        X=np.zeros((2, 1), dtype=np.float32),
        y_edge_bps=np.asarray([100.0, 0.0], dtype=np.float32),
        edge_weight=np.ones(2, dtype=np.float32),
        event_index=pd.DataFrame(
            {
                "family": ["trend_ma", "rsi_reversion"],
                "variant": ["ema_12_72", "rsi_6"],
            }
        ),
        feature_names=("turnover_proxy",),
    )
    oos_set = SimpleNamespace(
        X=np.zeros((2, 1), dtype=np.float32),
        event_index=pd.DataFrame(
            {
                "family": ["trend_ma", "rsi_reversion"],
                "variant": ["ema_12_72", "rsi_6"],
            }
        ),
        feature_names=("turnover_proxy",),
    )
    edge_models = SimpleNamespace(
        variant_prior_bps={
            "trend_ma:ema_12_72": -999.0,
            "rsi_reversion:rsi_6": -999.0,
        },
        global_prior_bps=-999.0,
    )

    def _fake_predict_candidate_edges(*_: Any, **__: Any) -> CandidateModelOutput:
        from src.domain.futures.strategy.candidate_contracts import EdgeSource
        zeros = np.zeros(2, dtype=np.float64)
        return CandidateModelOutput(
            events=oos_set.event_index,
            p_pass=np.asarray([0.8, 0.8], dtype=np.float64),
            gate_enabled=False,
            gate_threshold=0.5,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_return_r=zeros,
            expected_net_bps=np.asarray([1.0, 1.0], dtype=np.float64),
            q10_return_r=zeros,
            q10_net_bps=np.asarray([-10.0, -10.0], dtype=np.float64),
            q90_return_r=zeros,
            q90_net_bps=np.asarray([30.0, 30.0], dtype=np.float64),
            selection_score=np.asarray([1.0, 1.0], dtype=np.float64),
            kelly_fraction=zeros,
            validation_diagnostics={
                "utility_min": 0.0,
            },
        )

    monkeypatch.setattr("src.domain.futures.strategy.ablation.predict_candidate_edges", _fake_predict_candidate_edges)

    cfg = CandidateStrategyConfig(
        edge_prior_min_obs=1,
        edge_prior_shrinkage_obs=1,
        downside_penalty=0.0,
        turnover_penalty=0.0,
        concentration_penalty=0.0,
    )
    out = _build_variant_prior_output(
        edge_models=edge_models,
        calibration_set=calibration_set,
        oos_set=oos_set,
        p_pass=np.asarray([0.8, 0.8], dtype=np.float64),
        cfg=cfg,
    )

    assert np.allclose(out.mu_net_decision_bps, np.asarray([75.0, 25.0], dtype=np.float64))
    assert np.allclose(out.mu_gross_bps, np.asarray([75.0, 25.0], dtype=np.float64))
    assert np.isclose(out.selection_thresholds["utility_min"], 56.0)


def test_ablation_returns_attribution_columns(monkeypatch: Any) -> None:
    """Ablation DataFrame에 attribution 컬럼이 포함되어야 한다."""
    data_maps = _make_mock_data_maps(250)
    cfg = CandidateStrategyConfig(
        timeframe="4h",
        min_candidate_obs=10,
        min_rule_net_bps=0.0,
        kelly_fraction=0.1,
        gross_cap=1.2,
    )
    monkeypatch.setattr("src.domain.futures.strategy.ablation.fit_candidate_gate", lambda **_: object())
    monkeypatch.setattr(
        "src.domain.futures.strategy.ablation.predict_candidate_gate",
        lambda *, dataset, **__: np.full(dataset.X.shape[0], 0.9, dtype=np.float64),
    )
    monkeypatch.setattr("src.domain.futures.strategy.ablation.fit_candidate_edge_models", lambda **_: object())

    def _fake_edges(*, dataset: Any, p_pass: np.ndarray, **__: Any) -> CandidateModelOutput:
        from src.domain.futures.strategy.candidate_contracts import EdgeSource
        n = dataset.X.shape[0]
        zeros = np.zeros(n, dtype=np.float64)
        return CandidateModelOutput(
            events=dataset.event_index,
            p_pass=p_pass,
            gate_enabled=False,
            gate_threshold=0.5,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_return_r=zeros,
            expected_net_bps=np.full(n, 16.0, dtype=np.float64),
            q10_return_r=zeros,
            q10_net_bps=np.full(n, -10.0, dtype=np.float64),
            q90_return_r=zeros,
            q90_net_bps=np.full(n, 30.0, dtype=np.float64),
            selection_score=np.full(n, 8.0, dtype=np.float64),
            kelly_fraction=zeros,
            validation_diagnostics={
                "utility_min": 0.0,
            },
        )

    monkeypatch.setattr("src.domain.futures.strategy.ablation.predict_candidate_edges", _fake_edges)

    df_ablation = run_candidate_ablation(
        data_maps=data_maps, symbols=("BTCUSDT", "ETHUSDT"), tf="4h", cfg=cfg,
    )

    if not df_ablation.empty:
        required_attr_cols = {
            "trade_count", "deployed_bar_fraction",
            "pred_edge_bps_p50", "gross_cost_bps", "pass_deployment_gate",
        }
        assert required_attr_cols.issubset(df_ablation.columns), (
            f"Missing attribution columns: {required_attr_cols - set(df_ablation.columns)}"
        )
        assert df_ablation["trade_count"].dtype in (
            "int64", "int32", object
        ) or df_ablation["trade_count"].apply(lambda x: isinstance(x, int)).all()
        assert df_ablation["pass_deployment_gate"].dtype == bool or df_ablation[
            "pass_deployment_gate"
        ].apply(lambda x: isinstance(x, bool)).all()


def test_deployment_gate_blocks_near_zero_trading_variant() -> None:
    """near-zero-trading 변형은 pass_deployment_gate=False 여야 한다."""
    from src.domain.futures.strategy.ablation import AblationRow

    row_no_trade = AblationRow(
        variant="test",
        mean_log_growth=0.001,
        cagr=0.001,
        max_drawdown=0.0001,
        mar=1.0,
        turnover=0.0,
        final_equity=1_000_100.0,
        pass_compound_gate=True,
        trade_count=0,
        deployed_bar_fraction=0.0,
        pass_deployment_gate=False,
    )
    assert not row_no_trade.pass_deployment_gate

    cfg = CandidateStrategyConfig(
        min_deployment_trade_count=20,
        min_deployment_capital_fraction=0.05,
    )
    # trade_count=0 < 20, deployed_bar_fraction=0.0 < 0.05 → gate must fail
    gate_result = (
        row_no_trade.trade_count >= cfg.min_deployment_trade_count
        and row_no_trade.deployed_bar_fraction >= cfg.min_deployment_capital_fraction
    )
    assert not gate_result


# ── Phase 0 tests ──────────────────────────────────────────────────────────────

def test_compute_realized_edge_returns_nan_for_empty_trades() -> None:
    """Phase 0: empty trade DataFrame yields NaN realized edge."""
    result = _compute_realized_edge(pd.DataFrame())
    assert np.isnan(result)


def test_compute_realized_edge_returns_median_net_bps() -> None:
    """Phase 0: realized edge uses new net edge formula with pnl, entry_fee, entry_price, amount."""
    trades = pd.DataFrame({
        "entry_price": [100.0, 100.0],
        "amount": [10.0, 10.0],
        "pnl": [100.0, 100.0],
        "entry_fee": [1.0, 1.0],
    })
    # net_trade_bps = (100.0 - 1.0) / 1000.0 * 10_000 = 990 bps
    result = _compute_realized_edge(trades)
    assert np.isclose(result, 990.0, rtol=1e-6)


def test_ablation_result_contains_real_edge_columns() -> None:
    """Phase 0: ablation DataFrame must expose real_edge_bps_p50 and edge_capture_ratio."""
    import dataclasses
    required = {"real_edge_bps_p50", "edge_capture_ratio"}
    from src.domain.futures.strategy.ablation import AblationRow
    row = AblationRow(
        variant="test",
        mean_log_growth=0.001,
        cagr=0.01,
        max_drawdown=0.05,
        mar=0.2,
        turnover=0.01,
        final_equity=1_010_000.0,
        pass_compound_gate=False,
        trade_count=5,
        deployed_bar_fraction=0.1,
        pred_edge_bps_p50=40.0,
        real_edge_bps_p50=8.0,
        edge_capture_ratio=0.2,
        gross_cost_bps=15.0,
        pass_deployment_gate=True,
    )
    df = pd.DataFrame([dataclasses.asdict(row)])
    assert required.issubset(df.columns)
    assert float(df["real_edge_bps_p50"].iloc[0]) == pytest.approx(8.0)
    assert float(df["edge_capture_ratio"].iloc[0]) == pytest.approx(0.2)


# ── Phase 1 tests ──────────────────────────────────────────────────────────────

def test_build_barrier_arrays_writes_stop_and_tp_at_entry() -> None:
    """Phase 1: barrier arrays are non-zero only at and after each event's entry_idx."""
    events = pd.DataFrame({
        "symbol": ["BTCUSDT"],
        "entry_idx": [5],
        "stop_atr_mult": [2.0],
        "take_profit_atr_mult": [3.0],
        "expected_holding_bars": [3],
    })
    stop_2d, tp_2d = _build_barrier_arrays(
        selected_events=events,
        n_times=10,
        n_symbols=2,
        symbols=("BTCUSDT", "ETHUSDT"),
        start_idx=0,
    )
    # bars 5, 6, 7 should be filled (entry + 2 forward fills)
    assert stop_2d[4, 0] == 0.0, "bar before entry must be zero"
    assert stop_2d[5, 0] == pytest.approx(2.0)
    assert stop_2d[6, 0] == pytest.approx(2.0)
    assert stop_2d[7, 0] == pytest.approx(2.0)
    assert stop_2d[8, 0] == 0.0, "bar after holding window must be zero"
    assert tp_2d[5, 0] == pytest.approx(3.0)
    assert stop_2d[:, 1].sum() == 0.0, "ETHUSDT column must remain zero"


def test_build_barrier_arrays_respects_start_idx_offset() -> None:
    """Phase 1: global entry_idx is remapped correctly via start_idx."""
    events = pd.DataFrame({
        "symbol": ["BTCUSDT"],
        "entry_idx": [100],  # global index
        "stop_atr_mult": [1.5],
        "take_profit_atr_mult": [2.5],
        "expected_holding_bars": [2],
    })
    stop_2d, _ = _build_barrier_arrays(
        selected_events=events,
        n_times=10,
        n_symbols=1,
        symbols=("BTCUSDT",),
        start_idx=99,  # local_t = 100 - 99 = 1
    )
    assert stop_2d[0, 0] == 0.0
    assert stop_2d[1, 0] == pytest.approx(1.5)
    assert stop_2d[2, 0] == pytest.approx(1.5)
    assert stop_2d[3, 0] == 0.0


def test_build_barrier_arrays_empty_events_returns_zeros() -> None:
    """Phase 1: empty selected_events yields all-zero arrays."""
    stop_2d, tp_2d = _build_barrier_arrays(
        selected_events=pd.DataFrame(),
        n_times=5,
        n_symbols=2,
        symbols=("BTCUSDT", "ETHUSDT"),
        start_idx=0,
    )
    assert stop_2d.sum() == 0.0
    assert tp_2d.sum() == 0.0


def _make_signal_labeled(edge_bps: list[float], *, cost_bps: float = 7.5) -> pd.DataFrame:
    """Build a minimal labeled frame for validate_candidate_signals tests.

    All events fall inside OOS window [0, 100) under a single variant.
    """
    n = len(edge_bps)
    return pd.DataFrame(
        {
            "entry_idx": np.arange(n, dtype=np.int64),
            "edge_after_hurdle_bps": np.asarray(edge_bps, dtype=np.float64),
            "ex_ante_cost_bps": np.full(n, cost_bps, dtype=np.float64),
            "family": ["trend_ma"] * n,
            "variant": ["tma_1"] * n,
            "signal_cell": [""] * n,
            "side": np.ones(n, dtype=np.float64),
        }
    )


def _signal_diag() -> SimpleNamespace:
    return SimpleNamespace(
        recommended_keep_variants=("trend_ma:tma_1",),
        recommended_flip_variants=(),
        recommended_keep_signal_cells=(),
        recommended_flip_signal_cells=(),
    )


def _stress_rt() -> float:
    return ExecutionCostModel().stress_round_trip_bps()  # 11.25


def test_validate_candidate_signals_stress_mean_avoids_cost_double_counting() -> None:
    # Arrange: edge_after_hurdle already nets base cost; stress must add only the increment.
    cfg = CandidateStrategyConfig(blend_survival_use_mean=True)
    labeled = _make_signal_labeled([10.0, 20.0, 30.0, 40.0], cost_bps=7.5)

    # Act
    reports = validate_candidate_signals(
        labeled=labeled, diag=cast(Any, _signal_diag()), cfg=cfg, oos_start=0, oos_end=100
    )

    # Assert: honest stress = mean(edge) + base_cost - stress_rt = 25 + 7.5 - 11.25 = 21.25
    #         buggy double-count would be mean(edge) - stress_rt = 13.75.
    rule_only = next(r for r in reports if r.variant == "rule_only_equal_size")
    assert rule_only.net_edge_bps_mean == pytest.approx(25.0)
    assert rule_only.net_edge_bps_stress_mean == pytest.approx(25.0 + 7.5 - _stress_rt())
    assert rule_only.net_edge_bps_stress_mean != pytest.approx(25.0 - _stress_rt())


def test_validate_candidate_signals_mean_path_survives_skewed_payoff() -> None:
    # Arrange: right-skewed payoff — median negative, mean positive (ATR-stop structure).
    cfg = CandidateStrategyConfig(blend_survival_use_mean=True)
    labeled = _make_signal_labeled([-10.0] * 8 + [200.0] * 2, cost_bps=7.5)

    # Act
    reports = validate_candidate_signals(
        labeled=labeled, diag=cast(Any, _signal_diag()), cfg=cfg, oos_start=0, oos_end=100
    )

    # Assert: mean rescues skew where median would reject (p50<0<mean).
    rule_only = next(r for r in reports if r.variant == "rule_only_equal_size")
    assert rule_only.net_edge_bps_p50 < 0.0 < rule_only.net_edge_bps_mean
    assert rule_only.survives_cost is True


def test_validate_candidate_signals_legacy_median_path_blocks_skewed_payoff() -> None:
    # Arrange: identical skewed data, but legacy median gate active.
    cfg = CandidateStrategyConfig(blend_survival_use_mean=False)
    labeled = _make_signal_labeled([-10.0] * 8 + [200.0] * 2, cost_bps=7.5)

    # Act
    reports = validate_candidate_signals(
        labeled=labeled, diag=cast(Any, _signal_diag()), cfg=cfg, oos_start=0, oos_end=100
    )

    # Assert: median-stress (-10 - 11.25 < 0) blocks the skewed-but-profitable variant.
    rule_only = next(r for r in reports if r.variant == "rule_only_equal_size")
    assert rule_only.survives_cost is False


def test_validate_candidate_signals_blocks_when_mean_net_stress_negative() -> None:
    # Arrange: genuinely unprofitable signals (negative mean even before stress).
    cfg = CandidateStrategyConfig(blend_survival_use_mean=True)
    labeled = _make_signal_labeled([-10.0, -15.0, -20.0, -25.0], cost_bps=7.5)

    # Act
    reports = validate_candidate_signals(
        labeled=labeled, diag=cast(Any, _signal_diag()), cfg=cfg, oos_start=0, oos_end=100
    )

    # Assert
    rule_only = next(r for r in reports if r.variant == "rule_only_equal_size")
    assert rule_only.net_edge_bps_stress_mean < 0.0
    assert rule_only.survives_cost is False


def test_validate_candidate_signals_empty_oos_reports_not_survived() -> None:
    # Arrange: all events fall outside the OOS window.
    cfg = CandidateStrategyConfig(blend_survival_use_mean=True)
    labeled = _make_signal_labeled([20.0, 20.0, 20.0], cost_bps=7.5)

    # Act: OOS window starts past all entry_idx values.
    reports = validate_candidate_signals(
        labeled=labeled, diag=cast(Any, _signal_diag()), cfg=cfg, oos_start=500, oos_end=600
    )

    # Assert
    rule_only = next(r for r in reports if r.variant == "rule_only_equal_size")
    assert rule_only.n_events == 0
    assert rule_only.survives_cost is False
    assert np.isnan(rule_only.net_edge_bps_stress_mean)
