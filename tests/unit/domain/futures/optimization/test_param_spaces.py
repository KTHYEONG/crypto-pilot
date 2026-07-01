"""Tests for L1/L2 layer-specific Optuna parameter spaces."""
from __future__ import annotations


def test_l1_alpha_space_keys_disjoint_from_l2_alloc_space() -> None:
    """L1/L2 study 파라미터 공간 완전 분리 확인."""
    from src.domain.futures.allocation.search_space import L2_SEARCH_SPACE
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE

    overlap = set(L1_ALPHA_SPACE.keys()) & set(L2_SEARCH_SPACE.keys())
    assert overlap == set(), f"L1/L2 param overlap: {overlap}"


def test_l1_alpha_space_not_empty() -> None:
    """L1_ALPHA_SPACE가 비어있지 않음을 확인."""
    from src.domain.futures.optimization.opt_config import L1_ALPHA_SPACE

    assert len(L1_ALPHA_SPACE) >= 1


def test_l2_alloc_space_contains_k_rank() -> None:
    """K_RANK는 반드시 L2 study에 있어야 함 (allocation 결정)."""
    from src.domain.futures.allocation.search_space import L2_SEARCH_SPACE

    assert "K_RANK" in L2_SEARCH_SPACE


def test_engine_param_space_still_exists_for_backward_compat() -> None:
    """기존 ENGINE_PARAM_SPACE_FUTURES 제거되지 않았는지 확인."""
    from src.domain.futures.optimization.opt_config import ENGINE_PARAM_SPACE_FUTURES

    assert isinstance(ENGINE_PARAM_SPACE_FUTURES, dict)
    assert len(ENGINE_PARAM_SPACE_FUTURES) > 0
