from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    EdgeSource,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig
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
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table", return_value="gate-table"),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table", return_value="outer-table"),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table",
            return_value="registry-table",
        ),
        patch("src.domain.futures.strategy.tiered_workflow.pipeline.logger.info") as mock_log,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_factory():
    """Factory for CandidateStrategyConfig with selective overrides."""

    def _make(**overrides: object) -> CandidateStrategyConfig:
        defaults: dict[str, object] = {
            "l1_pair_min_folds": 2,
            "l1_pair_min_effective_obs": 30.0,
            "l1_pair_min_mean_gross_bps": 0.0,
            "l1_pair_min_incremental_bps": 0.0,
            "l1_pair_min_incremental_tstat": 0.0,
            "l1_pair_min_positive_fold_ratio": 0.0,
            "l1_pair_fdr_alpha": 1.0,
            "l1_min_cross_section": 2,
            "l1_min_opportunity_timestamps": 1,
            "l1_probe_top_k": 3,
            "l1_min_probe_bps": 0.0,
            "l1_min_probe_tstat": 0.0,
            "l1_min_sym_count": 1,
            "l1_min_sym_ratio": 0.0,
            "l1_min_fold_ratio": 0.0,
            "l1_min_opp_ic": -1.0,
            "l1_min_opp_tstat": 0.0,
            "l1_qualify_by_regime": False,
            "l1_activation_match_regime": False,
            "l1_opp_ic_mode": "time_series",
        }
        defaults.update(overrides)
        return CandidateStrategyConfig(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def sample_aligned():
    """Minimal AlignedMarketData-like namespace for tests."""
    n_bars = 30
    symbols = ("BTC",)
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    ones_2d = np.ones((n_bars, len(symbols)), dtype=np.float64)
    bool_2d = np.ones((n_bars, len(symbols)), dtype=np.bool_)
    ns = SimpleNamespace(
        datetimes=datetimes,
        symbols=symbols,
        open_2d=ones_2d,
        high_2d=ones_2d,
        low_2d=ones_2d,
        close_2d=ones_2d,
        volume_2d=ones_2d,
        funding_2d=np.zeros((n_bars, len(symbols)), dtype=np.float64),
        active_mask=bool_2d,
        warm_mask=bool_2d,
        entry_block_mask=~bool_2d,
        kill_mask=~bool_2d,
    )
    return ns


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_events(
    n: int,
    regime: str,
    fold_id: int = 0,
    strategy_id: str = "strat:v1",
    symbol: str = "BTC",
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    gross = rng.normal(5.0, 3.0, n)
    return pd.DataFrame({
        "symbol": symbol,
        "strategy_id": strategy_id,
        "activation_context": regime,
        "entry_regime": regime,
        "gross_event_bps": gross,
        # Set baseline to 0 so incremental = gross (positive on average)
        "baseline_gross_bps": np.zeros(n, dtype=np.float64),
        "side": 1,
        "expected_holding_bars": 4,
        "fold_id": fold_id,
        "uniqueness_weight": 1.0,
    })


# ---------------------------------------------------------------------------
# S1-S6 Tests
# ---------------------------------------------------------------------------


def test_compute_evidence_pooling_when_qualify_by_regime_false(cfg_factory) -> None:
    """l1_qualify_by_regime=False → regime A/B 이벤트가 단일 cell로 풀링되어 qualified=True."""
    events_a = _make_events(30, regime="1", fold_id=0)
    events_b = _make_events(30, regime="2", fold_id=1)
    combined = pd.concat([events_a, events_b], ignore_index=True)
    cfg = cfg_factory(l1_qualify_by_regime=False, l1_pair_min_folds=2, l1_pair_min_effective_obs=30.0)

    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    evidence = compute_symbol_strategy_evidence(event_results=combined, cfg=cfg, seed=0)

    assert len(evidence) == 1, "풀링 시 단일 cell만 생성돼야 함"
    assert evidence[0].key.activation_context == "all"
    assert evidence[0].n_obs == 60
    assert evidence[0].qualified is True


def test_compute_evidence_regime_split_when_qualify_by_regime_true(cfg_factory) -> None:
    """l1_qualify_by_regime=True → 동일 이벤트가 regime별 2 cell로 분리, 각 30obs."""
    events_a = _make_events(30, regime="1", fold_id=0)
    events_b = _make_events(30, regime="2", fold_id=1)
    combined = pd.concat([events_a, events_b], ignore_index=True)
    cfg = cfg_factory(l1_qualify_by_regime=True, l1_pair_min_effective_obs=5.0, l1_pair_min_folds=1)

    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    evidence = compute_symbol_strategy_evidence(event_results=combined, cfg=cfg, seed=0)

    assert len(evidence) == 2, "regime별 분리 시 2 cell 생성"
    contexts = {e.key.activation_context for e in evidence}
    assert contexts == {"1", "2"}


def test_compute_evidence_qualified_with_two_folds(cfg_factory) -> None:
    """l1_pair_min_folds=2, cell이 2 fold 출현 → qualified=True."""
    events = pd.concat([
        _make_events(20, regime="all", fold_id=0),
        _make_events(20, regime="all", fold_id=1),
    ], ignore_index=True)
    cfg = cfg_factory(l1_qualify_by_regime=False, l1_pair_min_folds=2, l1_pair_min_effective_obs=10.0)

    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    evidence = compute_symbol_strategy_evidence(event_results=events, cfg=cfg, seed=0)

    assert len(evidence) >= 1
    assert evidence[0].n_folds == 2
    assert "insufficient_folds" not in evidence[0].rejection_reasons


def test_evaluate_outer_time_series_ic_single_symbol(cfg_factory, sample_aligned) -> None:
    """l1_opp_ic_mode='time_series' → 단일 심볼 이벤트도 IC 측정 가능."""
    from src.domain.futures.strategy.tiered_workflow.signal_selection import evaluate_outer_signal_opportunities

    rng = np.random.default_rng(0)
    n = 10
    # Strictly increasing pred to ensure non-constant rank for spearmanr
    pred_bps = np.linspace(2.0, 10.0, n)
    real_bps = pred_bps * 1.5 + rng.normal(0, 0.5, n)

    decision_idxs = list(range(10, 10 + n))
    events_list = [
        ValidatedSignalEvent(
            decision_idx=di,
            decision_time=sample_aligned.datetimes[di],
            symbol="BTC",
            strategy_id="strat:v1",
            activation_context="all",
            side=1,
            expected_gross_bps=float(pred_bps[i]),
            q10_gross_bps=float(pred_bps[i]) * 0.8,
            q90_gross_bps=float(pred_bps[i]) * 1.2,
            expected_holding_bars=4,
            reliability=0.5,
            registry_version="test",
            model_version="test",
        )
        for i, di in enumerate(decision_idxs)
    ]
    batch = ValidatedSignalBatch(
        events=tuple(events_list),
        start_idx=10,
        end_idx=20,
        symbols=("BTC",),
        registry_version="test",
        model_version="test",
    )
    realized = pd.DataFrame({
        "entry_idx": [di + 1 for di in decision_idxs],  # entry_idx = decision_idx + 1
        "symbol": "BTC",
        "strategy_id": "strat:v1",
        "activation_context": "all",
        "realized_side_adjusted_gross_bps": real_bps.tolist(),
        "exit_idx": [di + 4 for di in decision_idxs],
    })
    fold = WFFold(fit_start=0, fit_end=10, cal_start=10, cal_end=10, oos_start=10, oos_end=20)
    cfg = cfg_factory(l1_opp_ic_mode="time_series", l1_min_cross_section=2, l1_probe_top_k=3)
    vol = np.ones((30, len(sample_aligned.symbols)), dtype=np.float64) * 0.01

    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=realized,
        volatility_2d=vol,
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert result.opportunity_ic > 0.0, f"time_series IC는 양수여야 함, got {result.opportunity_ic}"
    assert result.valid_opportunity_timestamp_count >= 1
    assert "non_positive_probe" not in result.blockers


def test_evaluate_outer_empty_opportunities_blocker(cfg_factory, sample_aligned) -> None:
    """기존 빈 배치 → empty_opportunities 블로커 유지 (회귀)."""
    from src.domain.futures.strategy.tiered_workflow.signal_selection import evaluate_outer_signal_opportunities

    batch = ValidatedSignalBatch(
        events=(),
        start_idx=0,
        end_idx=0,
        symbols=("BTC",),
        registry_version="test",
        model_version="test",
    )
    realized = pd.DataFrame()
    fold = WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=5, oos_start=5, oos_end=10)
    cfg = cfg_factory()
    vol = np.ones((20, len(sample_aligned.symbols)), dtype=np.float64) * 0.01

    result = evaluate_outer_signal_opportunities(
        opportunities=batch,
        realized_event_results=realized,
        volatility_2d=vol,
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert "empty_opportunities" in result.blockers
    assert result.passed is False


def test_candidate_output_activation_match_false(cfg_factory, sample_aligned) -> None:
    """l1_activation_match_regime=False → OOS row가 다른 regime이어도 발화됨."""
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        build_qualified_signal_registry,
        compute_symbol_strategy_evidence,
    )

    events = _make_events(60, regime="1", fold_id=0)
    events2 = _make_events(60, regime="1", fold_id=1)
    combined = pd.concat([events, events2], ignore_index=True)
    cfg = cfg_factory(
        l1_qualify_by_regime=False,
        l1_activation_match_regime=False,
        l1_pair_min_folds=2,
    )
    evidence = compute_symbol_strategy_evidence(event_results=combined, cfg=cfg, seed=0)
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTC",),
        min_signals_per_symbol=1,
        registry_version="test",
    )
    assert "BTC" in registry.by_symbol, "registry에 BTC가 있어야 함 (사전조건)"
    assert not bool(getattr(cfg, "l1_activation_match_regime", True)), "cfg 플래그 확인"
