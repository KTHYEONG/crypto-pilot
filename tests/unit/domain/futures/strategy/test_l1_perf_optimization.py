"""L1 PERF 최적화 검증: P1 캐시워밍·P3 진단게이팅·P4 로깅·P5 타이머분리."""
from __future__ import annotations

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
    aligned.volume_2d = rng.uniform(1e6, 1e8, (30, 2)).astype(np.float64)
    aligned.funding_2d = rng.uniform(-0.001, 0.001, (30, 2)).astype(np.float64)
    aligned.symbols = ("BTCUSDT", "ETHUSDT")
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(30)],
        dtype="datetime64[ns]",
    )
    aligned.beta_vs_market_1d = None
    aligned.active_mask = None
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
    for key in ("sym_ret_1", "sym_ret_5", "sym_vol_20", "sym_volume_z20",
                "funding_z20", "mkt_ret_1_padded", "mkt_vol_20",
                "mkt_dispersion_20", "market_breadth_20", "overlay_ctx", "regime_ctx"):
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


def test_prime_cache_does_not_call_build_candidate_dataset(
    aligned_mock: Any, cfg_mock: Any
) -> None:
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
    events = pd.DataFrame({
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
    })
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
    registry = QualifiedSignalRegistry(
        by_symbol={}, ready_symbols=(), trade_scope_count=0, registry_version="v0"
    )
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
            "schema": 0.01, "dataset_fit": 0.05, "dataset_early_stop": 0.0,
            "dataset_calibration_fit": 0.0, "dataset_calibration_eval": 0.0,
            "dataset_oos": 0.02, "edge_fit": 0.10, "inference": 0.08,
            "selection": 0.04, "total": 0.30,
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
            "src.domain.futures.strategy.tiered_workflow.pipeline"
            ".format_layer1_deployment_registry_table",
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
        mock_mem.return_value.available = 8 * 1024 ** 3  # 8 GB
        result = resolve_safe_nested_workers(n_tasks=16, frame_memory_bytes=500 * 1024 ** 2)
    assert result >= 2, f"Expected >=2 workers with 8GB/500MB frame, got {result}"
    assert result <= 16, "Should not exceed n_tasks"


def test_resolve_safe_nested_workers_pinned() -> None:
    """P0: pinned=2 → 정확히 2 worker 반환."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import resolve_safe_nested_workers
    result = resolve_safe_nested_workers(n_tasks=16, frame_memory_bytes=1_000_000_000, pinned=2)
    assert result == 2, f"Expected 2 workers (pinned), got {result}"


# ─── P3: defer_artifact=True 경로 검증 ─────────────────────────────────────────

def test_run_l1_nested_swf_defer_artifact_skips_inference(
    aligned_mock: Any, cfg_mock: Any, caplog: Any
) -> None:
    """P3: defer_artifact=True 시 inference_artifact=None."""
    import concurrent.futures

    from src.domain.futures.strategy.tiered_workflow.pipeline import run_l1_nested_swf

    def empty_fold_out_fn(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            fit_status="trained",
            timing_profile=dict.fromkeys(
                ("schema", "dataset_fit", "dataset_early_stop", "dataset_calibration_fit",
                 "dataset_calibration_eval", "dataset_oos", "edge_fit", "inference", "selection"), 0.01
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
        mock_mem.return_value.available = 16 * 1024 ** 3
        with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
            resolve_safe_nested_workers(n_tasks=16, frame_memory_bytes=200 * 1024 ** 2)

    logs = [r.message for r in caplog.records if "worker_calc" in r.message]
    assert len(logs) >= 1, "worker_calc PERF log must be emitted"
    assert "[PERF]" in logs[0]
    assert "workers=" in logs[0]
    assert "n_tasks=16" in logs[0]


# ─── PERF-GAP2: l1_nested_ipc_collect PERF 로그 검증 ──────────────────────────

def test_l1_nested_ipc_collect_log_emitted(
    aligned_mock: Any, cfg_mock: Any, caplog: Any
) -> None:
    """PERF-GAP2: run_l1_nested_swf → [PERF] l1_nested_ipc_collect 로그."""
    import concurrent.futures

    from src.domain.futures.strategy.tiered_workflow import run_l1_nested_swf
    from src.domain.futures.strategy.walk_forward import WFFold

    _empty = SimpleNamespace(
        fit_status="trained",
        timing_profile=dict.fromkeys(("schema", "dataset_fit", "dataset_early_stop",
                                      "dataset_calibration_fit", "dataset_calibration_eval",
                                      "dataset_oos", "edge_fit", "inference", "selection"), 0.01),
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
