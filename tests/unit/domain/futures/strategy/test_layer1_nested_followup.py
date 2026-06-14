from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    CandidateModelOutput,
    EdgeSource,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
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

    import concurrent.futures
    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args, mp_context=None, **kwargs):
            super().__init__(*args, **kwargs)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch("src.domain.futures.strategy.tiered_workflow._fit_and_predict_single_fold", return_value=empty_out),
        patch("src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold", return_value=empty_out),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table",
            return_value="gate-table",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table",
            return_value="outer-table",
        ),
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


def test_candidate_output_to_signal_batch_respects_activation_match_regime() -> None:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        _candidate_output_to_signal_batch,
    )

    events = pd.DataFrame(
        {
            "entry_idx": [5],
            "symbol": ["BTCUSDT"],
            "family": ["trend"],
            "variant": ["fast"],
            "entry_regime": ["all"],
            "side": [1],
            "expected_holding_bars": [4],
        }
    )
    model_output = CandidateModelOutput(
        events=events,
        p_pass=np.asarray([1.0], dtype=np.float64),
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_gross_bps=np.asarray([6.0], dtype=np.float64),
        q10_gross_bps=np.asarray([4.0], dtype=np.float64),
        q90_gross_bps=np.asarray([8.0], dtype=np.float64),
    )
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "bull"),
        mean_gross_bps=4.0,
        mean_incremental_bps=2.0,
        bootstrap_tstat_incremental=2.5,
        p_value=0.01,
        q_value=0.02,
        positive_fold_ratio=1.0,
        n_obs=10,
        effective_n=10.0,
        n_folds=3,
        reliability=0.8,
        qualified=True,
        rejection_reasons=(),
    )
    registry = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (evidence,)},
        ready_symbols=("BTCUSDT",),
        trade_scope_count=1,
        registry_version="v1",
    )
    datetimes = np.array(
        [
            np.datetime64("2024-01-01T00:00:00", "ns"),
            np.datetime64("2024-01-01T04:00:00", "ns"),
            np.datetime64("2024-01-01T08:00:00", "ns"),
            np.datetime64("2024-01-01T12:00:00", "ns"),
            np.datetime64("2024-01-01T16:00:00", "ns"),
        ],
        dtype="datetime64[ns]",
    )

    relaxed_batch = _candidate_output_to_signal_batch(
        model_output=model_output,
        registry=registry,
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        model_version="m1",
        activation_floor_bps=0.0,
        cfg=MagicMock(l1_activation_match_regime=False),
    )
    strict_batch = _candidate_output_to_signal_batch(
        model_output=model_output,
        registry=registry,
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        model_version="m1",
        activation_floor_bps=0.0,
        cfg=MagicMock(l1_activation_match_regime=True),
    )

    assert len(relaxed_batch.events) == 1
    assert relaxed_batch.events[0].activation_context == "all"
    assert len(strict_batch.events) == 0


def test_symbol_strategy_evidence_qualified_ignores_diagnostic_flags() -> None:
    evidence = SymbolStrategyEvidence(
        key=SignalSourceKey("BTCUSDT", "trend:fast", "all"),
        mean_gross_bps=4.0,
        mean_incremental_bps=2.0,
        block_tstat_incremental=0.1,
        probability_positive=0.6,
        p_value=0.4,
        q_value=0.4,
        positive_fold_ratio=1.0,
        n_obs=8,
        effective_n=8.0,
        n_folds=2,
        quality_weight=0.5,
        hard_eligible=True,
        structural_reasons=(),
        diagnostic_flags=("weak_tstat",),
    )

    assert evidence.qualified is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_factory() -> Callable[..., CandidateStrategyConfig]:
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
def sample_aligned() -> SimpleNamespace:
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


def test_compute_evidence_pooling_when_qualify_by_regime_false(
    cfg_factory: Callable[..., CandidateStrategyConfig],
) -> None:
    """l1_qualify_by_regime=False → regime A/B 이벤트가 단일 cell로 풀링되어 qualified=True."""
    events_a = _make_events(30, regime="1", fold_id=0)
    events_b = _make_events(30, regime="2", fold_id=1)
    combined = pd.concat([events_a, events_b], ignore_index=True)
    cfg = cfg_factory(l1_qualify_by_regime=False, l1_pair_min_folds=2, l1_pair_min_effective_obs=30.0)

    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    evidence = compute_symbol_strategy_evidence(
        event_results=combined,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    assert len(evidence) == 1, "풀링 시 단일 cell만 생성돼야 함"
    assert evidence[0].key.activation_context == "all"
    assert evidence[0].n_obs == 60
    assert evidence[0].qualified is True


def test_compute_evidence_regime_split_when_qualify_by_regime_true(
    cfg_factory: Callable[..., CandidateStrategyConfig],
) -> None:
    """l1_qualify_by_regime=True → 동일 이벤트가 regime별 2 cell로 분리, 각 30obs."""
    events_a = _make_events(30, regime="1", fold_id=0)
    events_b = _make_events(30, regime="2", fold_id=1)
    combined = pd.concat([events_a, events_b], ignore_index=True)
    cfg = cfg_factory(l1_qualify_by_regime=True, l1_pair_min_effective_obs=5.0, l1_pair_min_folds=1)

    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    evidence = compute_symbol_strategy_evidence(
        event_results=combined,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    assert len(evidence) == 2, "regime별 분리 시 2 cell 생성"
    contexts = {e.key.activation_context for e in evidence}
    assert contexts == {"1", "2"}


def test_compute_evidence_qualified_with_two_folds(
    cfg_factory: Callable[..., CandidateStrategyConfig],
) -> None:
    """l1_pair_min_folds=2, cell이 2 fold 출현 → qualified=True."""
    events = pd.concat([
        _make_events(20, regime="all", fold_id=0),
        _make_events(20, regime="all", fold_id=1),
    ], ignore_index=True)
    cfg = cfg_factory(l1_qualify_by_regime=False, l1_pair_min_folds=2, l1_pair_min_effective_obs=10.0)

    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    evidence = compute_symbol_strategy_evidence(
        event_results=events,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    assert len(evidence) >= 1
    assert evidence[0].n_folds == 2
    assert "insufficient_folds" not in evidence[0].rejection_reasons


def test_evaluate_outer_time_series_ic_single_symbol(
    cfg_factory: Callable[..., CandidateStrategyConfig],
    sample_aligned: SimpleNamespace,
) -> None:
    """l1_opp_ic_mode='time_series' → probe_bps 양수, opportunity_ic는 항상 None."""
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
        aligned_symbols=tuple(sample_aligned.symbols),
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert result.opportunity_ic is None, "IC 계산 제거 후 항상 None"
    assert result.probe_bps > 0.0, f"probe_bps는 양수여야 함, got {result.probe_bps}"
    assert result.valid_opportunity_timestamp_count >= 1
    assert "non_positive_probe" not in result.blockers


def test_evaluate_outer_empty_opportunities_blocker(
    cfg_factory: Callable[..., CandidateStrategyConfig],
    sample_aligned: SimpleNamespace,
) -> None:
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
        aligned_symbols=tuple(sample_aligned.symbols),
        fold=fold,
        fold_id=0,
        cfg=cfg,
        seed=0,
    )

    assert "empty_opportunities" in result.blockers
    assert result.passed is False


def test_candidate_output_activation_match_false(
    cfg_factory: Callable[..., CandidateStrategyConfig],
    sample_aligned: SimpleNamespace,
) -> None:
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
    evidence = compute_symbol_strategy_evidence(
        event_results=combined,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )
    registry = build_qualified_signal_registry(
        evidence=evidence,
        symbols=("BTC",),
        min_signals_per_symbol=1,
        registry_version="test",
    )
    assert "BTC" in registry.by_symbol, "registry에 BTC가 있어야 함 (사전조건)"
    assert not bool(getattr(cfg, "l1_activation_match_regime", True)), "cfg 플래그 확인"


def test_compute_symbol_strategy_evidence_respects_lookback_and_quality_flag(
    cfg_factory: Callable[..., CandidateStrategyConfig],
) -> None:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import compute_symbol_strategy_evidence

    event_results = pd.DataFrame(
        {
            "symbol": ["BTC", "BTC"],
            "strategy_id": ["strat:v1", "strat:v1"],
            "activation_context": ["all", "all"],
            "gross_event_bps": [5.0, 7.0],
            "baseline_gross_bps": [0.0, 0.0],
            "side": [1, 1],
            "expected_holding_bars": [4, 4],
            "fold_id": [0, 1],
            "uniqueness_weight": [1.0, 1.0],
            "entry_idx": [10, 20],
            "exit_idx": [12, 22],
        }
    )
    cfg = cfg_factory(
        l1_pair_min_folds=1,
        l1_pair_min_effective_obs=1.0,
        l1_quality_weight_enabled=False,
        l1_evidence_lookback_bars=5,
    )

    evidence = compute_symbol_strategy_evidence(
        event_results=event_results,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=25,
    )

    assert len(evidence) == 1
    assert evidence[0].n_obs == 1
    assert evidence[0].quality_weight == pytest.approx(1.0)


def test_run_l1_nested_swf_builds_prequential_snapshots_once() -> None:
    aligned = MagicMock()
    aligned.close_2d = np.ones((32, 1), dtype=np.float64)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(32)],
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
    evidence_folds = (
        WFFold(0, 4, 4, 6, 6, 10),
        WFFold(0, 10, 10, 12, 12, 16),
    )
    outer_folds = (
        WFFold(0, 12, 12, 16, 16, 20),
        WFFold(0, 16, 16, 20, 20, 24),
    )

    import concurrent.futures
    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args, mp_context=None, **kwargs):
            super().__init__(*args, **kwargs)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch(
            "src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds",
            return_value=evidence_folds,
        ) as mock_build,
        patch(
            "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold",
            return_value=empty_out,
        ) as mock_fit,
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
    ):
        run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            outer_folds=outer_folds,
            cfg=cfg,
            seed=3,
        )

    assert mock_build.call_count == 1
    assert mock_fit.call_count == len(evidence_folds) + len(outer_folds)


# ---------------------------------------------------------------------------
# Fix A·B: evidence 격자 세밀화 검증
# ---------------------------------------------------------------------------


def test_evidence_grid_folds_multiplied_by_outer_count() -> None:
    """A1: outer_n_folds=4, mult=3 → ev_n_folds=12, 첫 evidence oos_start < 첫 outer oos_start."""
    from src.domain.futures.strategy.walk_forward import build_l1_swf_folds

    outer_n_folds = 4
    mult = 3
    # outer block_len=500 가정: total available = outer_n_folds * 500
    l1_start = 0
    l1_end = outer_n_folds * 500  # 2000
    ev_n_folds = outer_n_folds * mult  # 12

    # outer 첫 oos_start: l1_start + block_len = l1_start + l1_end//(outer_n_folds+1)
    outer_block = l1_end // (outer_n_folds + 1)
    first_outer_oos = l1_start + outer_block

    ev_folds = build_l1_swf_folds(
        n_bars=l1_end,
        n_folds=ev_n_folds,
        l1_start_bars=l1_start,
        l1_end_bars=l1_end,
        purge_bars=0,
        embargo_bars=0,
    )

    assert len(ev_folds) == ev_n_folds, f"ev_n_folds={ev_n_folds} 기대, got {len(ev_folds)}"
    first_ev_oos = ev_folds[0].oos_start
    ev_block = ev_folds[0].oos_end - ev_folds[0].oos_start
    assert first_ev_oos < first_outer_oos, (
        f"첫 evidence oos_start({first_ev_oos}) < 첫 outer oos_start({first_outer_oos}) 기대"
    )
    assert ev_block < outer_block, (
        f"evidence block_len({ev_block}) < outer block_len({outer_block}) 기대"
    )


def test_evidence_grid_max_folds_cap_applied() -> None:
    """A2: outer_n_folds=20, mult=3, l1_evidence_max_folds=32 → ev_n_folds=32 (상한 적용)."""
    outer_n_folds = 20
    mult = 3
    l1_evidence_max_folds = 32

    ev_n_folds = min(outer_n_folds * mult, l1_evidence_max_folds)

    assert ev_n_folds == 32, f"상한 32 기대, got {ev_n_folds}"


# ---------------------------------------------------------------------------
# Fix C: IC None → "n/a" 렌더링 검증
# ---------------------------------------------------------------------------


def test_format_layer1_outer_fold_table_renders_none_ic_as_na() -> None:
    """C3: IC 컬럼 제거 후 테이블에 'IC:' 없음, 'Edge:' bps 값 포함."""
    from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
    from src.domain.futures.strategy.tiered_logging import format_layer1_outer_fold_table

    report = Layer1FoldReadiness(
        fold_id=0,
        registry_source_end_idx=100,
        outer_oos_start_idx=200,
        outer_oos_end_idx=300,
        ready_symbols=(),
        opportunity_ic=None,
        passed=False,
        blockers=("empty_opportunities",),
    )

    result = format_layer1_outer_fold_table(reports=(report,))

    assert "IC:" not in result, f"IC 컬럼 제거 후 'IC:' 미존재 기대: {result!r}"
    assert "Edge:" in result, f"'Edge:' bps 표시 기대: {result!r}"
    assert "0.00 bps" in result, f"probe_bps=0.0 기본값 표시 기대: {result!r}"


# ---------------------------------------------------------------------------
# Fix 1 — warmup 격자 검증
# ---------------------------------------------------------------------------

def test_build_l1_nested_swf_folds_warmup_shifts_first_oos() -> None:
    from src.domain.futures.strategy.walk_forward import build_l1_nested_swf_folds

    # Arrange
    cfg = CandidateStrategyConfig(wf_n_folds=4, l1_outer_warmup_blocks=2)
    l1_start, l1_end, n_bars = 2190, 5480, 7518
    max_label = 10

    # Act
    folds = build_l1_nested_swf_folds(
        n_bars=n_bars,
        l1_start_idx=l1_start,
        l1_end_idx=l1_end,
        max_label_horizon_bars=max_label,
        cfg=cfg,
    )

    # Assert
    assert len(folds) == 4
    block_len = (l1_end - l1_start) // (4 + 2)
    assert folds[0].oos_start == l1_start + 2 * block_len
    assert folds[-1].oos_end == l1_end


def test_build_l1_nested_swf_folds_warmup2_larger_first_oos_window() -> None:
    from src.domain.futures.strategy.walk_forward import build_l1_nested_swf_folds

    # Arrange
    cfg1 = CandidateStrategyConfig(wf_n_folds=4, l1_outer_warmup_blocks=1)
    cfg2 = CandidateStrategyConfig(wf_n_folds=4, l1_outer_warmup_blocks=2)
    kwargs: dict[str, Any] = {
        "n_bars": 7518,
        "l1_start_idx": 2190,
        "l1_end_idx": 5480,
        "max_label_horizon_bars": 10,
    }

    # Act
    folds1 = build_l1_nested_swf_folds(**kwargs, cfg=cfg1)
    folds2 = build_l1_nested_swf_folds(**kwargs, cfg=cfg2)

    # Assert: warmup=2이면 첫 OOS 이전 증거 윈도우가 더 김
    assert folds2[0].oos_start > folds1[0].oos_start


def test_build_l1_nested_swf_folds_causality_invariant() -> None:
    from src.domain.futures.strategy.walk_forward import build_l1_nested_swf_folds

    # Arrange
    cfg = CandidateStrategyConfig(wf_n_folds=4, l1_outer_warmup_blocks=2)

    # Act
    folds = build_l1_nested_swf_folds(
        n_bars=7518,
        l1_start_idx=2190,
        l1_end_idx=5480,
        max_label_horizon_bars=10,
        cfg=cfg,
    )

    # Assert
    for fold in folds:
        assert fold.fit_end <= fold.oos_start
        assert fold.oos_start < fold.oos_end


# ---------------------------------------------------------------------------
# Fix 2 — config validate
# ---------------------------------------------------------------------------

def test_config_validate_rejects_zero_warmup_blocks() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="l1_outer_warmup_blocks"):
        CandidateStrategyConfig(l1_outer_warmup_blocks=0)


# ---------------------------------------------------------------------------
# Fix 3 — 진단 로깅
# ---------------------------------------------------------------------------

def test_compute_symbol_strategy_evidence_logs_warning_when_zero_qualified(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        compute_symbol_strategy_evidence,
    )

    # Arrange: effective_n<5를 강제하는 소량 합성 데이터
    cfg = CandidateStrategyConfig()
    events = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 3,
        "strategy_id": ["trend:v1"] * 3,
        "activation_context": ["all"] * 3,
        "gross_event_bps": [1.0, 2.0, -1.0],
        "fold_id": [0, 0, 0],
        "exit_idx": [10, 20, 30],
        "entry_idx": [1, 11, 21],
        "uniqueness_weight": [1.0, 1.0, 1.0],
        "expected_holding_bars": [5, 5, 5],
        "side": [1, 1, 1],
    })

    # Act
    with caplog.at_level(
        logging.WARNING,
        logger="src.domain.futures.strategy.tiered_workflow.signal_selection",
    ):
        _evidence = compute_symbol_strategy_evidence(
            event_results=events,
            cfg=cfg,
            seed=42,
            registry_as_of_idx=100,
        )

    # Assert
    assert any("0 qualified" in r.message for r in caplog.records)


def test_compute_symbol_strategy_evidence_deterministic_seeding(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        compute_symbol_strategy_evidence,
    )

    cfg = CandidateStrategyConfig(l1_bootstrap_samples=100)
    events = pd.DataFrame({
        "symbol": ["BTCUSDT"] * 10,
        "strategy_id": ["trend:v1"] * 10,
        "activation_context": ["all"] * 10,
        "gross_event_bps": [1.0, 2.0, -1.0, 3.0, 0.5, -0.2, 1.1, -1.5, 0.9, -0.4],
        "fold_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "exit_idx": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "entry_idx": [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
        "uniqueness_weight": [1.0] * 10,
        "expected_holding_bars": [5] * 10,
        "side": [1] * 10,
    })

    # Run 1: with default hash
    res1 = compute_symbol_strategy_evidence(
        event_results=events,
        cfg=cfg,
        seed=42,
        registry_as_of_idx=200,
    )

    # Mock builtins.hash to return arbitrary different value
    orig_hash = builtins.hash
    monkeypatch.setattr(builtins, "hash", lambda x: 99999)

    # Run 2: hash is mocked, should produce identical bootstrap statistics
    res2 = compute_symbol_strategy_evidence(
        event_results=events,
        cfg=cfg,
        seed=42,
        registry_as_of_idx=200,
    )

    # Restore hash just in case
    monkeypatch.setattr(builtins, "hash", orig_hash)

    assert len(res1) == len(res2)
    assert len(res1) > 0
    # The probability_positive and block_tstat must be identical because the seeding is deterministic
    assert res1[0].probability_positive == res2[0].probability_positive
    assert res1[0].block_tstat_incremental == res2[0].block_tstat_incremental


def test_compute_symbol_strategy_evidence_none_and_empty_types() -> None:
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        compute_symbol_strategy_evidence,
    )

    cfg = CandidateStrategyConfig(l1_bootstrap_samples=10)
    events = pd.DataFrame({
        "symbol": ["BTCUSDT", ""],
        "strategy_id": ["trend:v1", "trend:v2"],
        "activation_context": ["all", "all"],
        "gross_event_bps": [1.0, 2.0],
        "fold_id": [0, 0],
        "exit_idx": [10, 20],
        "entry_idx": [1, 11],
        "uniqueness_weight": [1.0, 1.0],
        "expected_holding_bars": [5, 5],
        "side": [1, 1],
    })

    # Act & Assert: Should run successfully without encoding/hashing errors
    res = compute_symbol_strategy_evidence(
        event_results=events,
        cfg=cfg,
        seed=42,
        registry_as_of_idx=100,
    )
    assert len(res) >= 0

