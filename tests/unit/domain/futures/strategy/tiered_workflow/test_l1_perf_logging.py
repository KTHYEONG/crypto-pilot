"""L1 PERF 로그 구조 검증: [PERF] 접두사 일관성 + 새 타이밍 태그 존재 여부."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.utils.utils import PERF
from src.domain.futures.strategy.tiered_workflow import run_l1_nested_swf
from src.domain.futures.strategy.walk_forward import WFFold


@pytest.fixture
def minimal_aligned() -> MagicMock:
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
def minimal_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.wf_n_folds = 2
    cfg.l1_min_signals_per_symbol = 1
    cfg.l1_signal_activation_floor_bps = 0.0
    cfg.l1_bootstrap_block_bars = 6
    cfg.l1_bootstrap_samples = 200
    cfg.l1_pair_alpha = 0.05
    cfg.l1_pair_power = 0.80
    return cfg


@pytest.fixture
def empty_fold_out() -> SimpleNamespace:
    return SimpleNamespace(
        fit_status="trained",
        timing_profile={
            "schema": 0.01,
            "dataset_fit": 0.05,
            "dataset_early_stop": 0.02,
            "dataset_calibration_fit": 0.03,
            "dataset_calibration_eval": 0.01,
            "dataset_oos": 0.02,
            "edge_fit": 0.10,
            "inference": 0.08,
            "selection": 0.04,
            "total": 0.36,
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


def _run_l1_nested(aligned: Any, cfg: Any, empty_fold_out: Any) -> Any:
    """run_l1_nested_swf 최소 실행 헬퍼."""
    import concurrent.futures

    class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, *args: Any, mp_context: Any = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    outer_folds = (WFFold(0, 4, 4, 6, 6, 10),)
    with (
        patch("src.domain.futures.strategy.config.resolve_purge_and_embargo_bars", return_value=(1, 0)),
        patch("src.domain.futures.strategy.tiered_workflow.build_l1_swf_folds", return_value=()),
        patch(
            "src.domain.futures.strategy.candidate_workflow._fit_and_predict_single_fold",
            return_value=empty_fold_out,
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
        return run_l1_nested_swf(
            labeled_events=pd.DataFrame(),
            aligned=aligned,
            outer_folds=outer_folds,
            cfg=cfg,
            seed=3,
        )


# ─── Scenario 1: L1-NESTED-PROFILE 로그 출력 확인 ─────────────────────────────


def test_l1_nested_profile_log_emitted(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    profile_lines = [r.message for r in caplog.records if "fold_avg_profile" in r.message]
    assert len(profile_lines) >= 1, "Expected at least 1 fold_avg_profile log"
    for log in profile_lines:
        assert "[PERF]" in log
        assert "edge_fit=" in log
        assert "inference=" in log
        assert "n=" in log
    has_evidence = any("l1_evidence_fold_avg_profile" in m for m in profile_lines)
    has_outer = any("l1_outer_fold_avg_profile" in m for m in profile_lines)
    assert has_evidence or has_outer, "Expected evidence or outer profile log"


# ─── Scenario 2: L1-FOLD 로그에 batch/sel/eval 세분화 확인 ────────────────────


def test_l1_outer_fold_log_contains_substep_timing(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    fold_logs = [r.message for r in caplog.records if "l1_outer_fold fold=" in r.message]
    assert fold_logs, "l1_outer_fold log must be emitted"
    for log in fold_logs:
        assert "[PERF]" in log
        assert "batch=" in log
        assert "sel=" in log
        assert "eval=" in log
        assert "total=" in log


# ─── Scenario 3: 타이밍 PERF 로그는 모두 [PERF] 접두사 ───────────────────────


def test_all_timing_perf_logs_have_perf_prefix(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    """PERF 레벨 로그 중 'took=' 또는 timing 관련 키워드를 포함하는 로그는 [PERF] 접두사 필수."""
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    perf_level_logs = [r for r in caplog.records if r.levelno == PERF]
    timing_logs = [r for r in perf_level_logs if "took=" in r.message or "took" in r.message.lower()]

    violations = [r.message for r in timing_logs if not r.message.startswith("[PERF]")]
    assert not violations, "다음 타이밍 PERF 로그가 [PERF] 접두사 없음:\n" + "\n".join(violations)


# ─── Scenario 4: portfolio_constructor solve_constrained_weights PERF 로그 ────


def test_solve_constrained_weights_emits_perf_log(caplog: Any) -> None:
    import numpy as np

    from src.domain.futures.portfolio.portfolio_constructor import solve_constrained_weights

    mu = np.array([10.0, -5.0, 7.0], dtype=np.float64)
    sigma = np.eye(3, dtype=np.float64) * 0.01

    with caplog.at_level(PERF, logger="src.domain.futures.portfolio.portfolio_constructor"):
        solve_constrained_weights(
            mu,
            sigma,
            kappa=0.3,
            f_kelly_max=0.5,
            sigma_target_ann=0.20,
            bars_per_year=2190.0,
            gross_cap=1.0,
            per_symbol_cap=0.2,
            current_dd=0.0,
        )

    logs = [r.message for r in caplog.records if "solve_constrained_weights" in r.message]
    assert len(logs) == 1
    assert "[PERF]" in logs[0]
    assert "took=" in logs[0]
    assert "n=3" in logs[0]


# ─── Scenario 5: diagonal_kelly_weights PERF 로그 ─────────────────────────────


def test_diagonal_kelly_weights_emits_perf_log(caplog: Any) -> None:
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps, diagonal_kelly_weights

    n = 4
    mu_bps = np.array([5.0, -3.0, 8.0, 2.0], dtype=np.float64)
    sigma = np.array([0.01, 0.012, 0.009, 0.011], dtype=np.float64)
    caps = PortfolioCaps(gross=1.0, per_symbol=0.3, net=0.5, beta=1.0, target_ann_vol=0.20)

    with caplog.at_level(PERF, logger="src.domain.futures.portfolio.portfolio_constructor"):
        diagonal_kelly_weights(
            mu_bps=mu_bps,
            sigma=sigma,
            kelly_fraction=0.25,
            vol_target=0.20,
            caps=caps,
            prev_w=np.zeros(n, dtype=np.float64),
            no_trade_band=0.01,
        )

    logs = [r.message for r in caplog.records if "diagonal_kelly_weights" in r.message]
    assert len(logs) == 1
    assert "[PERF]" in logs[0]
    assert "took=" in logs[0]
    assert f"n={n}" in logs[0]


# ─── Scenario 6: perf-tiered 구 접두사 완전 제거 확인 ────────────────────────

# ─── Scenario 7: evaluate_layer1_readiness PERF 로그 (Gap 4) ──────────────────


def test_evaluate_layer1_readiness_emits_perf_log(caplog: Any) -> None:
    from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.tiered_workflow.signal_selection import evaluate_layer1_readiness

    cfg = MagicMock(spec=CandidateStrategyConfig)
    cfg.l1_min_fold_cov = 0.8
    cfg.l1_min_realized_match_ratio = 0.90
    cfg.l1_min_fold_ratio = 0.5
    cfg.l1_min_probe_bps = 0.0
    cfg.l1_probe_lcb_pooled = True
    cfg.l1_sym_count_mode = "effective_n"
    cfg.l1_min_effective_sym_n = 3.0
    cfg.l1_min_sym_count = 3
    cfg.l1_min_sym_ratio = 0.0
    cfg.l1_bootstrap_block_bars = 6
    cfg.l1_bootstrap_samples = 200

    fold_reports = (
        Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=10,
            outer_oos_start_idx=10,
            outer_oos_end_idx=20,
            ready_symbols=("BTC", "ETH"),
            matched_event_count=5,
            unmatched_event_count=1,
            realized_match_ratio=0.83,
            unique_decision_count=3,
            prediction_unique_count=3,
            opportunity_ic=0.05,
            opportunity_ic_tstat=1.2,
            probe_bps=2.0,
            probe_lcb_bps=1.5,
            probe_series_bps=(2.0, 1.5, 1.8),
            effective_symbol_count=2.0,
            passed=True,
            blockers=(),
        ),
    )

    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        evaluate_layer1_readiness(
            fold_reports=fold_reports,
            fold_cov=1.0,
            trade_scope_count=5,
            cfg=cfg,
            seed=0,
        )

    logs = [r.message for r in caplog.records if "l1_gate_eval" in r.message]
    assert len(logs) == 1, f"Expected 1 l1_gate_eval log, got {len(logs)}"
    assert "[PERF]" in logs[0]
    assert "n_folds=1" in logs[0]
    assert "n_passed=1" in logs[0]
    assert "took=" in logs[0]


# ─── Scenario 8: build_qualified_signal_registry PERF 로그 (Gap 5) ─────────────


def test_build_qualified_signal_registry_emits_perf_log(caplog: Any) -> None:
    from src.domain.futures.strategy.candidate_contracts import SignalSourceKey, SymbolStrategyEvidence
    from src.domain.futures.strategy.tiered_workflow.signal_selection import build_qualified_signal_registry

    evidence = (
        SymbolStrategyEvidence(
            key=SignalSourceKey(symbol="BTC", strategy_id="ma", activation_context="all"),
            mean_gross_bps=10.0,
            mean_incremental_bps=8.0,
            block_tstat_incremental=1.5,
            probability_positive=0.8,
            p_value=0.05,
            q_value=0.05,
            positive_fold_ratio=0.8,
            n_obs=100,
            effective_n=80.0,
            n_folds=5,
            quality_weight=0.9,
            hard_eligible=True,
            structural_reasons=(),
            diagnostic_flags=(),
            lcb_net_bps=2.0,
        ),
    )

    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        build_qualified_signal_registry(
            evidence=evidence,
            symbols=("BTC", "ETH"),
            min_signals_per_symbol=1,
            registry_version="test",
        )

    logs = [r.message for r in caplog.records if "l1_build_registry" in r.message]
    assert len(logs) == 1, f"Expected 1 l1_build_registry log, got {len(logs)}"
    assert "[PERF]" in logs[0]
    assert "n_evidence=1" in logs[0]
    assert "n_ready=1" in logs[0]
    assert "n_symbols=2" in logs[0]
    assert "took=" in logs[0]


# ─── Scenario 9: l1_lifecycle PERF 로그 (Gap 6) ─────────────────────────────


def test_l1_lifecycle_perf_log_emitted(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    logs = [r.message for r in caplog.records if "l1_lifecycle" in r.message]
    assert len(logs) >= 1, "l1_lifecycle PERF log must be emitted"
    for log in logs:
        assert "[PERF]" in log
        assert "n_syms=" in log
        assert "l1_T=" in log
        assert "took=" in log


# ─── Scenario 10: [MEM] 로그 출력 확인 ────────────────────────────────────────


def test_l1_nested_mem_log_emitted(minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any) -> None:
    with caplog.at_level(logging.DEBUG, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    mem_logs = [r.message for r in caplog.records if "[MEM]" in r.message]
    assert len(mem_logs) >= 1, "At least 1 [MEM] log must be emitted"
    assert any("stage=volatility_2d" in m for m in mem_logs), "volatility_2d mem log missing"
    assert any("stage=nested_prime_cache" in m for m in mem_logs), "nested_prime_cache mem log missing"
    assert any("stage=pre_fork" in m for m in mem_logs), "pre_fork mem log missing"
    has_evidence = any("stage=evidence_ipc" in m for m in mem_logs)
    has_outer = any("stage=outer_ipc" in m for m in mem_logs)
    assert has_evidence or has_outer, "evidence_ipc or outer_ipc mem log missing"


def test_no_legacy_perf_tiered_prefix(minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    legacy_logs = [r.message for r in caplog.records if "perf-tiered" in r.message]
    assert not legacy_logs, "구 [perf-tiered] 접두사 로그가 아직 남아있음:\n" + "\n".join(legacy_logs)


# ─── Scenario 11: Evidence phase runs before outer phase ─────────────────────


def test_two_phase_evidence_before_outer(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    all_perf = [(r.message, r.created) for r in caplog.records if "l1_" in r.message]
    evidence_logs = [msg for msg, _ in all_perf if "l1_evidence_ipc_collect" in msg]
    outer_logs = [msg for msg, _ in all_perf if "l1_outer_ipc_collect" in msg]
    snap_logs = [msg for msg, _ in all_perf if "l1_prequential_evidence_snapshots" in msg]

    assert len(outer_logs) == 1, "Outer IPC log missing"
    if evidence_logs:
        assert len(evidence_logs) == 1, "Evidence IPC log should be 1"
        assert len(snap_logs) == 1, "Evidence snapshots log should be 1"
        ev_time = min(ev_created for ev_msg, ev_created in all_perf if "l1_evidence_ipc_collect" in ev_msg)
        snap_time = min(s_created for s_msg, s_created in all_perf if "l1_prequential_evidence_snapshots" in s_msg)
        out_time = min(o_created for o_msg, o_created in all_perf if "l1_outer_ipc_collect" in o_msg)
        assert ev_time < snap_time < out_time, (
            "Expected order: evidence -> snapshots -> outer, "
            f"got ev={ev_time:.3f} snap={snap_time:.3f} out={out_time:.3f}"
        )
    else:
        # No evidence folds: snapshots log also absent, only outer phase ran
        assert not snap_logs, "snapshots log should be absent when no evidence"
        assert len([msg for msg, _ in all_perf if "l1_evidence_phase" in msg]) == 1


# ─── Scenario 12: Thread env vars set before fork ────────────────────────────


def test_l1_nested_thread_env_vars_set(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(logging.DEBUG, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    envs = {
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    for var in envs:
        assert os.environ.get(var) == "1", f"{var} must be '1' after L1 execution, got {os.environ.get(var)}"

    mem_logs = [r.message for r in caplog.records if "[MEM]" in r.message]
    pre_fork_idx = next((i for i, m in enumerate(mem_logs) if "stage=pre_fork" in m), -1)
    assert pre_fork_idx >= 0, "pre_fork log must be present"
    assert any("stage=evidence_ipc" in m for m in mem_logs[pre_fork_idx:]) or any(
        "stage=outer_ipc" in m for m in mem_logs[pre_fork_idx:]
    ), "IPC log must appear after pre_fork log"
