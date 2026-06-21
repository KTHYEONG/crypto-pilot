"""L1 PERF 로그 구조 검증: [PERF] 접두사 일관성 + 새 타이밍 태그 존재 여부."""
from __future__ import annotations

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
            "src.domain.futures.strategy.tiered_workflow.pipeline"
            ".format_layer1_deployment_registry_table",
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

    profile_lines = [r.message for r in caplog.records if "l1_nested_fold_avg_profile" in r.message]
    assert len(profile_lines) == 1, f"Expected 1 l1_nested_fold_avg_profile log, got {len(profile_lines)}"
    log = profile_lines[0]
    assert "[PERF]" in log
    assert "edge_fit=" in log
    assert "inference=" in log
    assert "n=" in log


# ─── Scenario 2: L1-FOLD 로그에 batch/sel/eval 세분화 확인 ────────────────────

def test_l1_outer_fold_log_contains_substep_timing(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    fold_logs = [r.message for r in caplog.records if "l1_outer_fold" in r.message]
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
    assert not violations, (
        "다음 타이밍 PERF 로그가 [PERF] 접두사 없음:\n" + "\n".join(violations)
    )


# ─── Scenario 4: portfolio_constructor solve_constrained_weights PERF 로그 ────

def test_solve_constrained_weights_emits_perf_log(caplog: Any) -> None:
    import numpy as np

    from src.domain.futures.portfolio.portfolio_constructor import solve_constrained_weights

    mu = np.array([10.0, -5.0, 7.0], dtype=np.float64)
    sigma = np.eye(3, dtype=np.float64) * 0.01

    with caplog.at_level(PERF, logger="src.domain.futures.portfolio.portfolio_constructor"):
        solve_constrained_weights(
            mu, sigma,
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

def test_no_legacy_perf_tiered_prefix(
    minimal_aligned: Any, minimal_cfg: Any, empty_fold_out: Any, caplog: Any
) -> None:
    with caplog.at_level(PERF, logger="src.domain.futures.strategy.tiered_workflow"):
        _run_l1_nested(minimal_aligned, minimal_cfg, empty_fold_out)

    legacy_logs = [r.message for r in caplog.records if "perf-tiered" in r.message]
    assert not legacy_logs, (
        "구 [perf-tiered] 접두사 로그가 아직 남아있음:\n" + "\n".join(legacy_logs)
    )
