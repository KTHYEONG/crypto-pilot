"""L1 PERF 최적화 검증: P1 캐시워밍·P3 진단게이팅·P4 로깅·P5 타이머분리."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.utils.utils import PERF
from src.domain.futures.strategy.candidate_dataset import (
    _ALIGNED_FEATURE_CACHE,
    _warm_aligned_2d_cache,
)
from src.domain.futures.strategy.tiered_workflow import run_l1_nested_swf
from src.domain.futures.strategy.walk_forward import WFFold

# ─── Shared Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def aligned_mock() -> MagicMock:
    rng = np.random.default_rng(42)
    aligned = MagicMock()
    aligned.close_2d = rng.uniform(100, 200, (30, 2)).astype(np.float64)
    aligned.open_2d = rng.uniform(99, 201, (30, 2)).astype(np.float64)
    aligned.high_2d = np.maximum(aligned.close_2d, rng.uniform(100, 210, (30, 2))).astype(np.float64)
    aligned.low_2d = np.minimum(aligned.close_2d, rng.uniform(90, 100, (30, 2))).astype(np.float64)
    aligned.volume_2d = rng.uniform(1e6, 1e8, (30, 2)).astype(np.float64)
    aligned.funding_2d = rng.uniform(-0.001, 0.001, (30, 2)).astype(np.float64)
    aligned.symbols = ("BTCUSDT", "ETHUSDT")
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(30)],
        dtype="datetime64[ns]",
    )
    aligned.beta_vs_market_1d = None
    aligned.active_mask = None
    aligned.execution_cost_bps_2d = np.zeros((30, 2), dtype=np.float64)
    return aligned


@pytest.fixture
def cfg_mock() -> MagicMock:
    cfg = MagicMock()
    cfg.market_state_features_enabled = False
    cfg.wf_n_folds = 2
    cfg.l1_min_signals_per_symbol = 1
    cfg.l1_signal_activation_floor_bps = 0.0
    cfg.l1_bootstrap_block_bars = 6
    cfg.l1_bootstrap_samples = 200
    cfg.l1_pair_alpha = 0.05
    cfg.l1_pair_power = 0.80
    cfg.l1_compact_ipc_enabled = True
    cfg.l1_nested_result_soft_cap_mb = 512
    cfg.l1_nested_workers = None
    return cfg


# ─── P1: _warm_aligned_2d_cache ──────────────────────────────────────────────


def test_warm_2d_cache_populates_base_keys(aligned_mock: Any, cfg_mock: Any) -> None:
    # Arrange
    aligned_id = id(aligned_mock)
    _ALIGNED_FEATURE_CACHE.pop(aligned_id, None)

    # Act
    _warm_aligned_2d_cache(aligned_mock, cfg_mock)

    # Assert
    cache = _ALIGNED_FEATURE_CACHE.get(aligned_id, {})
    for key in (
        "sym_ret_1",
        "sym_ret_5",
        "sym_vol_20",
        "sym_volume_z20",
        "funding_z20",
        "mkt_ret_1_padded",
        "mkt_vol_20",
        "mkt_dispersion_20",
        "market_breadth_20",
        "overlay_ctx",
        "regime_ctx",
    ):
        assert key in cache, f"캐시 키 누락: {key}"


def test_warm_2d_cache_idempotent(aligned_mock: Any, cfg_mock: Any) -> None:
    # Arrange — 두 번 호출해도 __aligned_ref__ 동일하면 재계산 없음
    aligned_id = id(aligned_mock)
    _ALIGNED_FEATURE_CACHE.pop(aligned_id, None)
    _warm_aligned_2d_cache(aligned_mock, cfg_mock)
    ref_arr = _ALIGNED_FEATURE_CACHE[aligned_id]["sym_ret_1"]

    # Act
    _warm_aligned_2d_cache(aligned_mock, cfg_mock)

    # Assert — 동일 배열 객체 (재할당 없음)
    assert _ALIGNED_FEATURE_CACHE[aligned_id]["sym_ret_1"] is ref_arr


def test_prime_cache_does_not_call_build_candidate_dataset(aligned_mock: Any, cfg_mock: Any) -> None:
    from src.domain.futures.strategy.candidate_dataset import prime_aligned_feature_cache

    # Act
    with patch(
        "src.domain.futures.strategy.candidate_dataset.build_candidate_dataset",
        autospec=True,
    ) as mock_build:
        prime_aligned_feature_cache(
            labeled_events=pd.DataFrame(),
            aligned=aligned_mock,
            cfg=cfg_mock,
        )

    # Assert — per-event 조립(build_candidate_dataset) 호출 없음
    mock_build.assert_not_called()


def test_warm_cache_empty_close_returns_early(cfg_mock: Any) -> None:
    # Arrange
    empty_aligned = MagicMock()
    empty_aligned.close_2d = np.zeros((0, 0), dtype=np.float64)
    aligned_id = id(empty_aligned)
    _ALIGNED_FEATURE_CACHE.pop(aligned_id, None)

    # Act
    _warm_aligned_2d_cache(empty_aligned, cfg_mock)

    # Assert — 캐시 미생성
    assert aligned_id not in _ALIGNED_FEATURE_CACHE


# ─── P3: selection 진단 게이팅 ───────────────────────────────────────────────


def _make_model_output(n: int = 5) -> Any:
    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "symbol": ["BTCUSDT"] * n,
            "family": ["trend"] * n,
            "variant": ["v1"] * n,
            "side": [1] * n,
            "raw_score": rng.uniform(0.1, 1.0, n),
            "score_z": rng.uniform(-1, 1, n),
            "expected_holding_bars": [12] * n,
            "min_holding_bars": [4] * n,
            "stop_atr_mult": [1.0] * n,
            "take_profit_atr_mult": [2.0] * n,
            "turnover_proxy": [1.0] * n,
            "cost_floor_bps": [7.5] * n,
            "entry_idx": list(range(100, 100 + n)),
            "side_flipped": [False] * n,
        }
    )
    return SimpleNamespace(
        events=events,
        p_pass=rng.uniform(0.5, 1.0, n),
        mu_net_decision_bps=rng.uniform(-5, 10, n),
        q10_net_bps=rng.uniform(-10, 5, n),
        q90_net_bps=rng.uniform(5, 20, n),
        expected_return_r=rng.uniform(0, 0.1, n),
        expected_net_bps=rng.uniform(0, 10, n),
        expected_gross_bps=rng.uniform(0, 15, n),
        q10_return_r=rng.uniform(-0.01, 0, n),
        q10_gross_bps=rng.uniform(-5, 5, n),
        q90_return_r=rng.uniform(0.01, 0.1, n),
        q90_gross_bps=rng.uniform(5, 15, n),
        selection_score=rng.uniform(0, 1, n),
        kelly_fraction=np.full(n, 0.25),
        utility_score=rng.uniform(0, 1, n),
        selection_thresholds={},
        validation_diagnostics={},
    )


def _make_selection_cfg(*, diagnostics_enabled: bool) -> MagicMock:
    cfg = MagicMock()
    cfg.l1_selection_diagnostics_enabled = diagnostics_enabled
    cfg.selection_sensitivity_enabled = False
    cfg.selection_policy = "hard"
    cfg.min_expected_net_bps = -999.0
    cfg.selection_q10_mode = "off"
    cfg.shortfall_threshold_basis = "q10_net"
    cfg.shortfall_quantile = 0.1
    cfg.shortfall_bps_multiplier = 1.0
    cfg.selection_utility_mode = "expected_edge_direct"
    cfg.selection_min_expected_utility_bps = -999.0
    cfg.l1_breakeven_floor_bps = 0.0
    cfg.selection_top_quantile = 1.0
    cfg.max_variant_selection_fraction = 1.0
    cfg.selection_max_events_per_bar = None
    cfg.downside_penalty = 0.0
    cfg.turnover_penalty = 0.0
    cfg.selection_thresholds = {}
    return cfg


def test_selection_diagnostics_off_returns_same_selected(caplog: Any) -> None:
    from src.domain.futures.strategy.candidate_portfolio import select_candidate_events_for_portfolio

    model_output = _make_model_output(n=10)

    result_on = select_candidate_events_for_portfolio(
        model_output=model_output, cfg=_make_selection_cfg(diagnostics_enabled=True)
    )
    result_off = select_candidate_events_for_portfolio(
        model_output=model_output, cfg=_make_selection_cfg(diagnostics_enabled=False)
    )

    # 선택 결과(index)가 동일해야 함 — 진단은 read-only
    assert set(result_on.index.tolist()) == set(result_off.index.tolist()), (
        "진단 활성화 여부에 따라 selected 결과가 달라짐 → P3 적용 불가"
    )


# ─── P4: signal_batch_convert PERF 로그 ──────────────────────────────────────


def test_signal_batch_convert_emits_perf_log(caplog: Any) -> None:
    from src.domain.futures.strategy.candidate_contracts import QualifiedSignalRegistry
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        _candidate_output_to_signal_batch,
    )

    model_output = _make_model_output(n=3)
    registry = QualifiedSignalRegistry(by_symbol={}, ready_symbols=(), trade_scope_count=0, registry_version="v0")
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(200)],
        dtype="datetime64[ns]",
    )

    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow.signal_selection"):
        _candidate_output_to_signal_batch(
            model_output=model_output,
            registry=registry,
            datetimes=datetimes,
            symbols=("BTCUSDT",),
            model_version="v1",
            activation_floor_bps=0.0,
        )

    logs = [r.message for r in caplog.records if "signal_batch_convert" in r.message]
    assert len(logs) == 1, f"signal_batch_convert PERF 로그 1개 필요, got {len(logs)}"
    log = logs[0]
    assert "[PERF]" in log
    assert "pred=" in log
    assert "keys=" in log
    assert "loop=" in log
    assert "sort=" in log
    assert "total=" in log


# ─── P5: audit_tables 타이머가 inference보다 짧아야 함 ────────────────────────


@pytest.fixture
def empty_fold_out_p5() -> SimpleNamespace:
    return SimpleNamespace(
        fit_status="trained",
        timing_profile={
            "schema": 0.01,
            "dataset_fit": 0.05,
            "dataset_early_stop": 0.0,
            "dataset_calibration_fit": 0.0,
            "dataset_calibration_eval": 0.0,
            "dataset_oos": 0.02,
            "edge_fit": 0.10,
            "inference": 0.08,
            "selection": 0.04,
            "total": 0.30,
        },
        model_output=SimpleNamespace(
            events=pd.DataFrame(),
            expected_gross_bps=np.zeros((0,), dtype=np.float64),
            expected_net_bps=np.zeros((0,), dtype=np.float64),
            q10_gross_bps=np.zeros((0,), dtype=np.float64),
            q90_gross_bps=np.zeros((0,), dtype=np.float64),
            q10_net_bps=np.zeros((0,), dtype=np.float64),
            q90_net_bps=np.zeros((0,), dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            edge_weight=np.zeros((0,), dtype=np.float64),
            y_return_bps=np.zeros((0,), dtype=np.float64),
        ),
    )


@pytest.fixture
def minimal_aligned_p5() -> MagicMock:
    aligned = MagicMock()
    aligned.close_2d = np.ones((16, 1), dtype=np.float64)
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(16)],
        dtype="datetime64[ns]",
    )
    aligned.symbols = ("BTC",)
    aligned.beta_vs_market_1d = None
    aligned.active_mask = None
    return aligned


@pytest.fixture
def minimal_cfg_p5() -> MagicMock:
    cfg = MagicMock()
    cfg.wf_n_folds = 2
    cfg.l1_min_signals_per_symbol = 1
    cfg.l1_signal_activation_floor_bps = 0.0
    cfg.l1_bootstrap_block_bars = 6
    cfg.l1_bootstrap_samples = 200
    cfg.l1_pair_alpha = 0.05
    cfg.l1_pair_power = 0.80
    return cfg


def test_audit_tables_timer_excludes_inference(
    minimal_aligned_p5: Any, minimal_cfg_p5: Any, empty_fold_out_p5: Any, caplog: Any
) -> None:
    import concurrent.futures

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    with (
        caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"),
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch(
            "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold",
            return_value=empty_fold_out_p5,
        ),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table",
            return_value="gate",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table",
            return_value="outer",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table",
            return_value="reg",
        ),
    ):
        run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=minimal_aligned_p5,
            outer_folds=outer_folds,
            cfg=minimal_cfg_p5,
            seed=3,
        )

    def _took(msg: str) -> float:
        import re

        m = re.search(r"took=([\d.]+)s", msg)
        return float(m.group(1)) if m else -1.0

    audit_logs = [r.message for r in caplog.records if "l1_nested_audit_tables" in r.message]
    inference_logs = [r.message for r in caplog.records if "l1_fit_inference_artifact" in r.message]

    assert audit_logs, "l1_nested_audit_tables 로그 없음"
    audit_took = _took(audit_logs[0])

    if inference_logs:
        inference_took = _took(inference_logs[0])
        # audit_tables는 테이블 포맷팅만 → inference보다 작거나 같아야 함
        assert audit_took <= inference_took + 0.1, (
            f"audit_tables({audit_took:.4f}s) ≥ inference({inference_took:.4f}s) — 타이머 여전히 포함됨"
        )
    else:
        # gate_passed=False 경로: inference 없음, audit_tables는 테이블 포맷 시간만(≈0s)
        assert audit_took < 1.0, f"audit_tables 시간이 너무 큼: {audit_took:.4f}s"


# ─── P0: resolve_safe_nested_workers worker 계산 검증 ──────────────────────────


def test_resolve_safe_nested_workers_memory_estimate() -> None:
    """P0: 500MB frame, 8GB available → workers >= 2 (was 1 with old formula)."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 8 * 1024**3  # 8 GB
        result = resolve_safe_nested_workers(n_tasks=16, frame_memory_bytes=500 * 1024**2)
    assert result >= 2, f"Expected >=2 workers with 8GB/500MB frame, got {result}"
    assert result <= 16, "Should not exceed n_tasks"


def test_resolve_safe_nested_workers_pinned() -> None:
    """P0: pinned=2 → 정확히 2 worker 반환."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    result = resolve_safe_nested_workers(n_tasks=16, frame_memory_bytes=1_000_000_000, pinned=2)
    assert result == 2, f"Expected 2 workers (pinned), got {result}"


def test_resolve_safe_nested_workers_applies_result_soft_cap_and_logs_fields(
    caplog: Any,
) -> None:
    """soft cap은 worker 상한과 PERF 로그 필드에 반영되어야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 16 * 1024**3

        with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
            workers = resolve_safe_nested_workers(
                n_tasks=16,
                frame_memory_bytes=200 * 1024**2,
                compact_result=True,
                result_soft_cap_mb=256,
            )

    assert workers == 2
    logs = [r.message for r in caplog.records if "worker_calc" in r.message]
    assert logs
    assert "result_soft_cap_mb=256" in logs[0]
    assert "predicted_result_mb=100" in logs[0]
    assert "result_mem_limit=2" in logs[0]


def test_resolve_safe_nested_workers_pinned_does_not_bypass_safety_guards() -> None:
    """pinned는 hard override가 아니라 upper bound여야 한다."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 4 * 1024**3
        workers = resolve_safe_nested_workers(
            n_tasks=16,
            frame_memory_bytes=200 * 1024**2,
            pinned=8,
            compact_result=True,
            result_soft_cap_mb=128,
        )

    assert workers == 1


# ─── P3: defer_artifact=True 경로 검증 ─────────────────────────────────────────


def test_run_l1_nested_swf_defer_artifact_skips_inference(aligned_mock: Any, cfg_mock: Any, caplog: Any) -> None:
    """P3: defer_artifact=True 시 inference_artifact=None."""
    import concurrent.futures

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l1_nested_swf

    def empty_fold_out_fn(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            fit_status="trained",
            timing_profile=dict.fromkeys(
                (
                    "schema",
                    "dataset_fit",
                    "dataset_early_stop",
                    "dataset_calibration_fit",
                    "dataset_calibration_eval",
                    "dataset_oos",
                    "edge_fit",
                    "inference",
                    "selection",
                ),
                0.01,
            ),
            model_output=SimpleNamespace(
                events=pd.DataFrame(),
                expected_gross_bps=np.zeros((0,), dtype=np.float64),
                expected_net_bps=np.zeros((0,), dtype=np.float64),
                q10_gross_bps=np.zeros((0,), dtype=np.float64),
                q90_gross_bps=np.zeros((0,), dtype=np.float64),
                q10_net_bps=np.zeros((0,), dtype=np.float64),
                q90_net_bps=np.zeros((0,), dtype=np.float64),
            ),
            oos_set=SimpleNamespace(
                edge_weight=np.zeros((0,), dtype=np.float64),
                y_return_bps=np.zeros((0,), dtype=np.float64),
            ),
        )

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    _fit = "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold"
    _gate = "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table"
    _outer_fmt = "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table"
    _deploy_fmt = "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table"
    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch(_fit, side_effect=empty_fold_out_fn),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(_gate, return_value="gate"),
        patch(_outer_fmt, return_value="outer"),
        patch(_deploy_fmt, return_value="reg"),
        caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"),
    ):
        result = run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned_mock,
            outer_folds=outer_folds,
            cfg=cfg_mock,
            seed=3,
            defer_artifact=True,
        )

    assert result.inference_artifact is None, "defer_artifact=True → inference_artifact must be None"
    perf_logs = [r.message for r in caplog.records if "l1_fit_inference_artifact" in r.message]
    assert len(perf_logs) == 0, "defer_artifact=True → no l1_fit_inference_artifact PERF log"


# ─── PERF-GAP1: worker_calc PERF 로그 포맷 검증 ──────────────────────────────


def test_resolve_safe_nested_workers_emits_worker_calc_perf_log(caplog: Any) -> None:
    """PERF-GAP1: resolve_safe_nested_workers → [PERF] worker_calc 로그."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 16 * 1024**3
        with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
            resolve_safe_nested_workers(n_tasks=16, frame_memory_bytes=200 * 1024**2)

    logs = [r.message for r in caplog.records if "worker_calc" in r.message]
    assert len(logs) >= 1, "worker_calc PERF log must be emitted"
    assert "[PERF]" in logs[0]
    assert "workers=" in logs[0]
    assert "n_tasks=16" in logs[0]


def test_resolve_safe_nested_workers_applies_soft_cap_and_pinned_guard(caplog: Any) -> None:
    """soft cap and pinned must both respect safety clamps."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers

    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 4 * 1024**3
        with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
            workers = resolve_safe_nested_workers(
                n_tasks=16,
                frame_memory_bytes=200 * 1024**2,
                compact_result=True,
                pinned=8,
                result_soft_cap_mb=128,
            )

    assert workers == 1
    logs = [r.message for r in caplog.records if "worker_calc" in r.message]
    assert logs, "worker_calc PERF log must be emitted"
    assert "requested_workers=8" in logs[0]
    assert "result_soft_cap_mb=128" in logs[0]
    assert "predicted_result_mb=100" in logs[0]
    assert "result_mem_limit=1" in logs[0]
    assert "pinned_applied=True" in logs[0]


# ─── PERF-GAP2: l1_nested_ipc_collect PERF 로그 검증 ──────────────────────────


def test_l1_nested_ipc_collect_log_emitted(aligned_mock: Any, cfg_mock: Any, caplog: Any) -> None:
    """PERF-GAP2: run_l1_nested_swf → [PERF] l1_nested_ipc_collect 로그."""
    import concurrent.futures

    from src.domain.futures.strategy.tiered_workflow import run_l1_nested_swf
    from src.domain.futures.strategy.walk_forward import WFFold

    _empty = SimpleNamespace(
        fit_status="trained",
        timing_profile=dict.fromkeys(
            (
                "schema",
                "dataset_fit",
                "dataset_early_stop",
                "dataset_calibration_fit",
                "dataset_calibration_eval",
                "dataset_oos",
                "edge_fit",
                "inference",
                "selection",
            ),
            0.01,
        ),
        model_output=SimpleNamespace(
            events=pd.DataFrame(),
            expected_gross_bps=np.zeros((0,), dtype=np.float64),
            expected_net_bps=np.zeros((0,), dtype=np.float64),
            q10_gross_bps=np.zeros((0,), dtype=np.float64),
            q90_gross_bps=np.zeros((0,), dtype=np.float64),
            q10_net_bps=np.zeros((0,), dtype=np.float64),
            q90_net_bps=np.zeros((0,), dtype=np.float64),
        ),
        oos_set=SimpleNamespace(
            edge_weight=np.zeros((0,), dtype=np.float64),
            y_return_bps=np.zeros((0,), dtype=np.float64),
        ),
    )

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    _fit2 = "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold"
    _gate2 = "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table"
    _outer_fmt2 = "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table"
    _deploy_fmt2 = "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table"
    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch(_fit2, return_value=_empty),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(_gate2, return_value="gate"),
        patch(_outer_fmt2, return_value="outer"),
        patch(_deploy_fmt2, return_value="reg"),
        caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"),
    ):
        run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned_mock,
            outer_folds=outer_folds,
            cfg=cfg_mock,
            seed=3,
        )

    ev_logs = [r.message for r in caplog.records if "l1_evidence_ipc_collect" in r.message]
    out_logs = [r.message for r in caplog.records if "l1_outer_ipc_collect" in r.message]
    assert len(ev_logs) + len(out_logs) >= 1, "l1_*_ipc_collect PERF log must be emitted"
    for log in ev_logs + out_logs:
        assert "[PERF]" in log
        assert "n=" in log
        assert "took=" in log


def test_run_l1_nested_swf_soft_cap_forces_compact_submit(aligned_mock: Any, cfg_mock: Any, caplog: Any) -> None:
    """nested soft cap은 compact preference보다 우선하며 submit 인자에도 반영돼야 한다."""
    import concurrent.futures

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l1_nested_swf

    submitted_compact_flags: list[bool] = []

    def record_fold_submit(*args: Any, **kwargs: Any) -> Any:
        submitted_compact_flags.append(bool(args[3]))
        return SimpleNamespace(
            fold_id=args[0],
            fit_status="trained",
            n_fit=10,
            timing_profile=dict.fromkeys(
                (
                    "schema",
                    "dataset_fit",
                    "dataset_early_stop",
                    "dataset_calibration_fit",
                    "dataset_calibration_eval",
                    "dataset_oos",
                    "edge_fit",
                    "inference",
                    "selection",
                ),
                0.01,
            ),
            model_output=SimpleNamespace(
                events=pd.DataFrame(),
                expected_gross_bps=np.zeros((0,), dtype=np.float64),
                expected_net_bps=np.zeros((0,), dtype=np.float64),
                q10_gross_bps=np.zeros((0,), dtype=np.float64),
                q90_gross_bps=np.zeros((0,), dtype=np.float64),
                q10_net_bps=np.zeros((0,), dtype=np.float64),
                q90_net_bps=np.zeros((0,), dtype=np.float64),
            ),
            oos_set=SimpleNamespace(
                edge_weight=np.zeros((0,), dtype=np.float64),
                y_return_bps=np.zeros((0,), dtype=np.float64),
            ),
            selected_events=pd.DataFrame(),
            skip_reason=None,
        )

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    cfg_mock.l1_compact_ipc_enabled = False
    cfg_mock.l1_nested_result_soft_cap_mb = 256
    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    evidence_folds = (WFFold(0, 2, 2, 4, 4, 6),)

    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=evidence_folds),
        patch(
            "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold_from_globals",
            side_effect=record_fold_submit,
        ),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table",
            return_value="gate",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table",
            return_value="outer",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table",
            return_value="reg",
        ),
        caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"),
    ):
        run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned_mock,
            outer_folds=outer_folds,
            cfg=cfg_mock,
            seed=7,
        )

    assert submitted_compact_flags == [True, True]
    override_logs = [r.message for r in caplog.records if "soft_cap_force_compact" in r.message]
    assert override_logs
    assert "soft_cap_mb=256" in override_logs[0]


def test_event_results_from_fold_output_prefers_inline_compact_payload() -> None:
    """compact inline payload와 legacy fallback은 동일 event evidence를 제공해야 한다."""
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        _event_results_from_fold_output,
    )

    base_events = pd.DataFrame(
        {
            "entry_idx": [11, 12],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "family": ["trend", "trend"],
            "variant": ["v1", "v1"],
            "side": [1, -1],
            "gross_event_bps": [1.5, -2.0],
            "uniqueness_weight": [0.4, 0.8],
        }
    )
    legacy_events = base_events.drop(columns=["gross_event_bps", "uniqueness_weight"])

    inline_result = _event_results_from_fold_output(
        fold_id=3,
        fold_out=SimpleNamespace(
            model_output=SimpleNamespace(
                events=base_events,
                expected_gross_bps=np.array([5.0, 6.0], dtype=np.float64),
                q10_gross_bps=np.array([1.0, 2.0], dtype=np.float64),
                q90_gross_bps=np.array([9.0, 10.0], dtype=np.float64),
            ),
            oos_set=None,
        ),
    )
    legacy_result = _event_results_from_fold_output(
        fold_id=3,
        fold_out=SimpleNamespace(
            model_output=SimpleNamespace(
                events=legacy_events,
                expected_gross_bps=np.array([5.0, 6.0], dtype=np.float64),
                q10_gross_bps=np.array([1.0, 2.0], dtype=np.float64),
                q90_gross_bps=np.array([9.0, 10.0], dtype=np.float64),
            ),
            oos_set=SimpleNamespace(
                y_return_bps=np.array([1.5, -2.0], dtype=np.float64),
                edge_weight=np.array([0.4, 0.8], dtype=np.float64),
            ),
        ),
    )

    assert inline_result["gross_event_bps"].tolist() == legacy_result["gross_event_bps"].tolist()
    assert inline_result["uniqueness_weight"].tolist() == legacy_result["uniqueness_weight"].tolist()
    assert inline_result["decision_idx"].tolist() == legacy_result["decision_idx"].tolist()
    assert inline_result["fold_id"].tolist() == legacy_result["fold_id"].tolist()
    assert len(inline_result) == len(legacy_result)


def test_l1_nested_soft_cap_forces_compact_submit(aligned_mock: Any, cfg_mock: Any, caplog: Any) -> None:
    """Nested path should force compact IPC when full payload exceeds soft cap."""
    import concurrent.futures

    submitted_modes: list[tuple[bool, bool]] = []

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    def _fake_fold_from_globals(
        fold_idx: int,
        fold: WFFold,
        is_evidence_fold: bool = False,
        compact_result: bool = False,
    ) -> SimpleNamespace:
        del fold
        submitted_modes.append((is_evidence_fold, compact_result))
        return SimpleNamespace(
            fold_id=fold_idx,
            fit_status="trained",
            timing_profile=dict.fromkeys(
                (
                    "schema",
                    "dataset_fit",
                    "dataset_early_stop",
                    "dataset_calibration_fit",
                    "dataset_calibration_eval",
                    "dataset_oos",
                    "edge_fit",
                    "inference",
                    "selection",
                ),
                0.01,
            ),
            model_output=SimpleNamespace(
                events=pd.DataFrame(),
                expected_gross_bps=np.zeros((0,), dtype=np.float64),
                expected_net_bps=np.zeros((0,), dtype=np.float64),
                q10_gross_bps=np.zeros((0,), dtype=np.float64),
                q90_gross_bps=np.zeros((0,), dtype=np.float64),
                q10_net_bps=np.zeros((0,), dtype=np.float64),
                q90_net_bps=np.zeros((0,), dtype=np.float64),
            ),
            oos_set=SimpleNamespace(
                edge_weight=np.zeros((0,), dtype=np.float64),
                y_return_bps=np.zeros((0,), dtype=np.float64),
            ),
        )

    cfg_mock.l1_compact_ipc_enabled = False
    cfg_mock.l1_nested_result_soft_cap_mb = 256
    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    evidence_folds = (WFFold(0, 2, 2, 4, 4, 6),)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=evidence_folds),
        patch(
            "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold_from_globals",
            side_effect=_fake_fold_from_globals,
        ),
        patch("concurrent.futures.ProcessPoolExecutor", new=SafeThreadPoolExecutor),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_gate_table",
            return_value="gate",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_outer_fold_table",
            return_value="outer",
        ),
        patch(
            "src.domain.futures.strategy.tiered_workflow.pipeline.format_layer1_deployment_registry_table",
            return_value="reg",
        ),
        caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"),
    ):
        run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned_mock,
            outer_folds=outer_folds,
            cfg=cfg_mock,
            seed=3,
        )

    assert submitted_modes == [(True, True), (False, True)]
    override_logs = [r.message for r in caplog.records if "soft_cap_force_compact" in r.message]
    assert override_logs, "soft cap compact override log must be emitted"


def test_event_results_from_fold_output_compact_payload_matches_legacy() -> None:
    """Compact inline payload and legacy oos_set fallback must produce equivalent evidence rows."""
    from src.domain.futures.strategy.tiered_workflow.signal_selection import _event_results_from_fold_output

    base_events = pd.DataFrame(
        {
            "entry_idx": [10, 14],
            "family": ["mom", "rev"],
            "variant": ["a", "b"],
            "signal_cell": ["trend", "range"],
        }
    )
    gross = np.array([12.5, -3.0], dtype=np.float64)
    weights = np.array([0.7, 1.2], dtype=np.float64)
    expected = np.array([13.0, -2.5], dtype=np.float64)
    q10 = np.array([2.0, -7.5], dtype=np.float64)
    q90 = np.array([20.0, 4.5], dtype=np.float64)

    legacy = SimpleNamespace(
        model_output=SimpleNamespace(
            events=base_events.copy(),
            expected_gross_bps=expected,
            q10_gross_bps=q10,
            q90_gross_bps=q90,
        ),
        oos_set=SimpleNamespace(edge_weight=weights, y_return_bps=gross),
    )
    compact = SimpleNamespace(
        model_output=SimpleNamespace(
            events=base_events.assign(
                gross_event_bps=gross,
                uniqueness_weight=weights,
            ),
            expected_gross_bps=expected,
            q10_gross_bps=q10,
            q90_gross_bps=q90,
        ),
        oos_set=None,
    )

    legacy_frame = _event_results_from_fold_output(fold_id=3, fold_out=legacy)
    compact_frame = _event_results_from_fold_output(fold_id=3, fold_out=compact)

    pd.testing.assert_series_equal(legacy_frame["gross_event_bps"], compact_frame["gross_event_bps"])
    pd.testing.assert_series_equal(
        legacy_frame["uniqueness_weight"],
        compact_frame["uniqueness_weight"],
    )
    pd.testing.assert_series_equal(legacy_frame["decision_idx"], compact_frame["decision_idx"])
    pd.testing.assert_series_equal(legacy_frame["fold_id"], compact_frame["fold_id"])
    assert len(legacy_frame) == len(compact_frame) == 2


# ─── OPT-0: pipeline.py l1_tfs default alignment ────────────────────────────


def test_run_tiered_pipeline_default_l1_tfs_matches_config() -> None:
    """S3: run_tiered_pipeline default l1_tfs must match CandidateStrategyConfig.l1_tfs."""
    import inspect

    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

    sig = inspect.signature(run_tiered_pipeline)
    default_l1_tfs = sig.parameters["l1_tfs"].default
    ref_cfg = CandidateStrategyConfig()
    assert default_l1_tfs == ref_cfg.l1_tfs, f"Default l1_tfs={default_l1_tfs} != config l1_tfs={ref_cfg.l1_tfs}"


# ─── OPT-3: _t_ppf_cached equivalence ──────────────────────────────────────


def test_t_ppf_cached_equivalence() -> None:
    """S1: _t_ppf_cached must produce bit-identical results vs direct stats.t.ppf."""
    import scipy.stats as stats

    from src.domain.futures.strategy.tiered_workflow.signal_selection import _t_ppf_cached

    alpha = 0.05
    power = 0.80
    df_test_values = [2.0, 5.0, 10.0, 30.0, 100.0, 500.0]

    for df in df_test_values:
        expected_t_crit = float(stats.t.ppf(1.0 - alpha, float(df)))
        expected_t_power = float(stats.t.ppf(power, float(df)))

        actual_t_crit = _t_ppf_cached(round((1.0 - alpha) * 1000), round(df))
        actual_t_power = _t_ppf_cached(round(power * 1000), round(df))

        assert actual_t_crit == pytest.approx(expected_t_crit, abs=1e-12), f"t_crit mismatch df={df}"
        assert actual_t_power == pytest.approx(expected_t_power, abs=1e-12), f"t_power mismatch df={df}"


# ─── OPT-4: prefit_layer1_model + assemble_layer1_artifact equivalence ────


@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.fit_regime_conditional_ensemble")
@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.build_candidate_dataset")
@patch("src.domain.futures.strategy.tiered_workflow.signal_selection.fit_candidate_feature_schema")
def test_prefit_artifact_equivalence(
    mock_schema: MagicMock,
    mock_dataset: MagicMock,
    mock_ensemble: MagicMock,
) -> None:
    """S1: prefit_layer1_model + assemble_layer1_artifact === fit_layer1_inference_artifact."""
    from src.domain.futures.strategy.candidate_contracts import QualifiedSignalRegistry
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.tiered_workflow.signal_selection import (
        assemble_layer1_artifact,
        fit_layer1_inference_artifact,
        prefit_layer1_model,
    )

    mock_schema.return_value = SimpleNamespace(version="test_v1", feature_names=("f1", "f2"))
    mock_dataset.return_value = SimpleNamespace(
        event_index=pd.DataFrame({"symbol": ["BTCUSDT"], "side": [1], "expected_holding_bars": [12]}),
        y_return_bps=np.array([10.0], dtype=np.float64),
        y_gross_return_bps=np.array([12.0], dtype=np.float64),
    )
    mock_ensemble.return_value = SimpleNamespace(predict=lambda x: x)

    aligned = MagicMock()
    aligned.symbols = ("BTCUSDT",)
    aligned.datetimes = np.array([np.datetime64("2024-01-01", "ns")])

    registry = QualifiedSignalRegistry(
        by_symbol={},
        ready_symbols=(),
        trade_scope_count=1,
        registry_version="test",
    )
    labeled_events = pd.DataFrame({"entry_idx": [0], "family": ["mom"], "variant": ["a"], "signal_cell": ["trend"]})
    cfg = CandidateStrategyConfig()

    # Act: prefit + assemble
    core = prefit_layer1_model(
        labeled_events=labeled_events,
        aligned=aligned,
        fit_start_idx=0,
        fit_end_idx=10,
        cfg=cfg,
    )
    assembled = assemble_layer1_artifact(core=core, deployment_registry=registry, fit_end_idx=10)

    # Act: full fit
    full = fit_layer1_inference_artifact(
        labeled_events=labeled_events,
        aligned=aligned,
        deployment_registry=registry,
        fit_start_idx=0,
        fit_end_idx=10,
        cfg=cfg,
        seed=42,
    )

    # Assert: all registry-independent fields equal
    assert assembled.feature_schema is full.feature_schema
    assert assembled.model is full.model
    assert assembled.baseline_by_key == full.baseline_by_key
    assert assembled.config_hash == full.config_hash
    assert assembled.model_version == full.model_version
    assert assembled.l1_fit_end_idx == full.l1_fit_end_idx
    # Registry is present in both
    assert assembled.deployment_registry is full.deployment_registry is registry


# ─── HTF 병목 Fix 1: ATR 캐시 재사용 ─────────────────────────────────────────


def _make_label_events(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "side": [1] * n,
            "entry_idx": list(range(10, 10 + n)),
            "expected_holding_bars": [12] * n,
            "stop_atr_mult": [1.0] * n,
            "take_profit_atr_mult": [2.0] * n,
        }
    )


def _make_label_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.maker_fee_bps = 2.0
    cfg.taker_fee_bps = 5.0
    cfg.maker_ratio = 0.75
    cfg.slippage_bps = 1.0
    cfg.impact_coeff_bps = 0.0
    cfg.cost_stress_multiplier = 1.5
    return cfg


def test_label_candidate_events_atr_cache_skips_computation(aligned_mock: Any) -> None:
    """Scenario 1: precomputed_atr_2d 제공 시 _compute_yang_zhang_vol_2d 호출 없음."""
    from src.domain.futures.strategy.candidate_labels import (
        _compute_yang_zhang_vol_2d,
        label_candidate_events,
    )

    events = _make_label_events()
    cfg = _make_label_cfg()
    precomputed = _compute_yang_zhang_vol_2d(aligned_mock)

    with patch(
        "src.domain.futures.strategy.candidate_labels._compute_yang_zhang_vol_2d",
        wraps=_compute_yang_zhang_vol_2d,
    ) as mock_atr:
        result = label_candidate_events(
            events=events,
            aligned=aligned_mock,
            cfg=cfg,
            precomputed_atr_2d=precomputed,
        )

    mock_atr.assert_not_called()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    expected_cols = {"symbol", "side", "entry_idx", "expected_holding_bars", "stop_atr_mult", "take_profit_atr_mult"}
    assert expected_cols.issubset(result.columns)


def test_label_candidate_events_atr_default_computes_once(aligned_mock: Any) -> None:
    """Scenario 2: precomputed_atr_2d=None → _compute_yang_zhang_vol_2d 1회 호출."""
    from src.domain.futures.strategy.candidate_labels import (
        _compute_yang_zhang_vol_2d,
        label_candidate_events,
    )

    events = _make_label_events()
    cfg = _make_label_cfg()

    with patch(
        "src.domain.futures.strategy.candidate_labels._compute_yang_zhang_vol_2d",
        wraps=_compute_yang_zhang_vol_2d,
    ) as mock_atr:
        result = label_candidate_events(
            events=events,
            aligned=aligned_mock,
            cfg=cfg,
        )

    mock_atr.assert_called_once()
    assert isinstance(result, pd.DataFrame)


def test_label_candidate_events_atr_shape_mismatch_raises(aligned_mock: Any) -> None:
    """Scenario 3: precomputed_atr_2d 형상 불일치 → ValueError."""
    from src.domain.futures.strategy.candidate_labels import label_candidate_events

    events = _make_label_events()
    cfg = _make_label_cfg()
    wrong_shape = np.ones((100, 5), dtype=np.float64)

    with pytest.raises(ValueError, match="precomputed_atr_2d shape"):
        label_candidate_events(
            events=events,
            aligned=aligned_mock,
            cfg=cfg,
            precomputed_atr_2d=wrong_shape,
        )


# ─── HTF 병목 Fix 2: signal_only fast-path ──────────────────────────────────


@pytest.fixture
def signal_only_cfg() -> Any:
    from src.domain.futures.strategy.config import CandidateStrategyConfig

    return CandidateStrategyConfig(
        signal_only=True,
        promotion_filter_enabled=True,
        wf_enabled=False,
        wf_scheme="single",
        min_candidate_obs=3,
        ml_fit_fraction=0.5,
        ml_calibration_fraction=0.2,
        max_holding_bars=5,
        purge_bars=0,
        embargo_bars=0,
        side_flip_candidate_variants=(),
    )


@pytest.fixture
def non_signal_only_cfg() -> Any:
    from src.domain.futures.strategy.config import CandidateStrategyConfig

    return CandidateStrategyConfig(
        signal_only=False,
        promotion_filter_enabled=True,
        wf_enabled=False,
        wf_scheme="single",
        min_candidate_obs=3,
        ml_fit_fraction=0.5,
        ml_calibration_fraction=0.2,
        max_holding_bars=5,
        purge_bars=0,
        embargo_bars=0,
        side_flip_candidate_variants=(),
    )


def _make_strategy_cfg(candidate_cfg: Any) -> Any:
    from src.domain.futures.strategy.config import StrategyConfig

    return StrategyConfig(candidate=candidate_cfg)


def test_run_candidate_signal_only_skips_diagnostics(
    aligned_mock: Any, signal_only_cfg: MagicMock, caplog: Any
) -> None:
    """Scenario 4: signal_only=True → compute_rule_diagnostics 호출 0회, 빈 diag."""
    from src.domain.futures.strategy_runtime.bridge import (
        run_candidate_strategy_for_universe,
    )

    strategy_cfg = _make_strategy_cfg(signal_only_cfg)

    non_empty_events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [10],
            "expected_holding_bars": [12],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [2.0],
        }
    )
    with (
        patch(
            "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
            autospec=True,
        ) as mock_diag,
        patch(
            "src.domain.futures.strategy_runtime.bridge.verify_data_integrity",
            return_value={},
        ),
        patch(
            "src.domain.futures.strategy.candidate_portfolio.build_candidate_alpha_panel",
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            return_value=aligned_mock,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
            return_value=non_empty_events,
        ),
        patch(
            "src.domain.futures.strategy_runtime.bridge.build_multi_tf_panels",
            return_value=(),
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            return_value=(),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        result = run_candidate_strategy_for_universe(
            symbols=["BTCUSDT"],
            tf="4h",
            strategy_cfg=strategy_cfg,
            preloaded_data_maps={},
        )

    mock_diag.assert_not_called()
    assert result.rule_report is not None
    assert result.rule_report["recommended_keep_variants"] == ()
    assert result.rule_report["recommended_flip_variants"] == ()


def test_run_candidate_non_signal_only_calls_diagnostics(aligned_mock: Any, non_signal_only_cfg: MagicMock) -> None:
    """Scenario 5: signal_only=False → compute_rule_diagnostics 1회 호출."""
    from src.domain.futures.strategy.rule_diagnostics import RuleDiagnosticsResult
    from src.domain.futures.strategy_runtime.bridge import (
        run_candidate_strategy_for_universe,
    )

    strategy_cfg = _make_strategy_cfg(non_signal_only_cfg)
    fake_diag = RuleDiagnosticsResult(
        by_family=pd.DataFrame(),
        by_variant=pd.DataFrame(),
        by_family_side=pd.DataFrame(),
        side_flip=pd.DataFrame(),
        decision={},
        recommended_keep_variants=("v1",),
        recommended_flip_variants=(),
        recommended_keep_signal_cells=(),
        recommended_flip_signal_cells=(),
        recommendation_basis="test",
        recommendation_split=(0, 10),
        report_split=(0, 10),
    )

    non_empty_events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [10],
            "expected_holding_bars": [5],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [2.0],
        }
    )
    with (
        patch(
            "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
            return_value=fake_diag,
            autospec=True,
        ) as mock_diag,
        patch(
            "src.domain.futures.strategy.ablation.apply_variant_promotions",
            return_value=non_empty_events,
        ),
        patch(
            "src.domain.futures.strategy_runtime.bridge.verify_data_integrity",
            return_value={},
        ),
        patch(
            "src.domain.futures.strategy.candidate_portfolio.build_candidate_alpha_panel",
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            return_value=aligned_mock,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
            return_value=non_empty_events,
        ),
        patch(
            "src.domain.futures.strategy_runtime.bridge.build_multi_tf_panels",
            return_value=(),
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            return_value=(),
        ),
    ):
        _ = run_candidate_strategy_for_universe(
            symbols=["BTCUSDT"],
            tf="4h",
            strategy_cfg=strategy_cfg,
            preloaded_data_maps={},
        )

    mock_diag.assert_called_once()


def test_run_candidate_signal_only_skips_promotion(aligned_mock: Any, signal_only_cfg: MagicMock) -> None:
    """Scenario 6: signal_only=True + promotion_filter_enabled → apply_variant_promotions 호출 0회."""
    from src.domain.futures.strategy_runtime.bridge import (
        run_candidate_strategy_for_universe,
    )

    strategy_cfg = _make_strategy_cfg(signal_only_cfg)
    non_empty_events = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [10],
            "expected_holding_bars": [5],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [2.0],
        }
    )

    with (
        patch(
            "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
            autospec=True,
        ) as mock_diag,
        patch(
            "src.domain.futures.strategy.ablation.apply_variant_promotions",
            autospec=True,
        ) as mock_promo,
        patch(
            "src.domain.futures.strategy_runtime.bridge.verify_data_integrity",
            return_value={},
        ),
        patch(
            "src.domain.futures.strategy.candidate_portfolio.build_candidate_alpha_panel",
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            return_value=aligned_mock,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
            return_value=non_empty_events,
        ),
        patch(
            "src.domain.futures.strategy_runtime.bridge.build_multi_tf_panels",
            return_value=(),
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            return_value=(),
        ),
    ):
        run_candidate_strategy_for_universe(
            symbols=["BTCUSDT"],
            tf="4h",
            strategy_cfg=strategy_cfg,
            preloaded_data_maps={},
        )

    mock_diag.assert_not_called()
    mock_promo.assert_not_called()
