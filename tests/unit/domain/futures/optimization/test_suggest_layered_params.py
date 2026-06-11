"""Tests for TieredContext, suggest_layered_params, and objective_l1_ic.

Covers:
- TI10: L1 suggest 결과 키집합 검증
- TI11: L2 suggest 결과 키집합 검증
- TI11b: fixed 키 주입 시 suggest_* 미호출 확인
- TI13: objective_l1_ic 가 run_l2_awf (Sharpe) 미호출 확인
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import optuna
import pytest

from src.domain.futures.optimization.workflow import (
    TieredContext,
    objective_l1_ic,
    suggest_layered_params,
)

# ---------------------------------------------------------------------------
# TI10: L1 suggest 결과 키 == L1_ALPHA_SPACE 키집합
# ---------------------------------------------------------------------------


def test_suggest_layered_params_l1_keys_only_l1_space() -> None:
    """L1 suggest 결과 키 == L1_ALPHA_SPACE 키집합, L2 키 미포함."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE, L2_ALLOC_SPACE

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    result = suggest_layered_params(trial, "L1")

    assert set(result.keys()) == set(L1_ALPHA_SPACE.keys())
    assert not (set(result.keys()) & set(L2_ALLOC_SPACE.keys()))


# ---------------------------------------------------------------------------
# TI11: L2 suggest 결과 키 == L2_ALLOC_SPACE 키집합
# ---------------------------------------------------------------------------


def test_suggest_layered_params_l2_keys_only_l2_space() -> None:
    """L2 suggest 결과 키 == L2_ALLOC_SPACE 키집합, L1 키 미포함."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE, L2_ALLOC_SPACE

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    result = suggest_layered_params(trial, "L2")

    assert set(result.keys()) == set(L2_ALLOC_SPACE.keys())
    assert not (set(result.keys()) & set(L1_ALPHA_SPACE.keys()))


# ---------------------------------------------------------------------------
# TI11b: fixed 키는 suggest_* 미호출, 값 직접 주입
# ---------------------------------------------------------------------------


def test_suggest_layered_params_fixed_skips_suggest() -> None:
    """fixed 키는 trial.suggest_* 미호출, 값 그대로 주입."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE

    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    first_key = next(iter(L1_ALPHA_SPACE))
    fixed_val = 99

    result = suggest_layered_params(trial, "L1", fixed={first_key: fixed_val})

    assert result[first_key] == fixed_val


# ---------------------------------------------------------------------------
# TI13: objective_l1_ic 가 run_l2_awf(Sharpe) 미호출 (mock spy)
# ---------------------------------------------------------------------------


def test_objective_l1_ic_does_not_call_sharpe() -> None:
    """objective_l1_ic 는 run_l2_awf(Sharpe) 를 호출하지 않는다."""
    _l1_result = SimpleNamespace(
        mean_ic=0.04,
        ic_tstat=2.1,
        gate_passed=True,
        signals_per_fold=(),
        oos_stacked={},
        breadth=0.5,
        valid_coverage=0.9,
        fold_pass_ratio=0.7,
        n_valid=5,
        n_total=10,
    )

    with (
        patch(
            "src.domain.futures.strategy.tiered_workflow.run_l2_awf"
        ) as mock_l2,
        patch(
            "src.domain.futures.strategy.tiered_workflow.run_l1_cpcv",
            return_value=_l1_result,
        ),
        patch(
            "src.domain.futures.strategy.walk_forward.build_cpcv_folds",
            return_value=[MagicMock()],
        ),
        patch(
            "src.domain.futures.strategy.config.resolve_purge_and_embargo_bars",
            return_value=(5, 2),
        ),
    ):
        ctx = MagicMock()
        ctx.aligned.datetimes = list(range(200))
        ctx.fixed_l1_params = None

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        val = objective_l1_ic(trial, ctx)

    assert val == pytest.approx(0.04)
    mock_l2.assert_not_called()   # Sharpe 미참조 핵심 검증
