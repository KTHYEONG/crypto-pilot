from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    assert_selection_replay_parity,
)


def test_assert_selection_replay_parity_accepts_matching_values() -> None:
    replay = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)
    final = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)

    assert_selection_replay_parity(
        replay_evaluation=replay,
        final_evaluation=final,
        tolerance=1e-6,
    )


def test_assert_selection_replay_parity_raises_on_metric_mismatch() -> None:
    replay = SimpleNamespace(cagr_hybrid=0.22, mdd_hybrid=0.09, fold_pass_ratio=0.67, trade_count=80)
    final = SimpleNamespace(cagr_hybrid=0.08, mdd_hybrid=0.15, fold_pass_ratio=0.34, trade_count=120)

    with pytest.raises(ValueError, match="replay/final parity"):
        assert_selection_replay_parity(
            replay_evaluation=replay,
            final_evaluation=final,
            tolerance=1e-6,
        )

