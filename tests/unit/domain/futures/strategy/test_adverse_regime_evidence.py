from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    compute_adverse_regime_evidence,
    compute_symbol_strategy_evidence,
)


def _make_cfg(**overrides: Any) -> MagicMock:
    cfg = MagicMock()
    defaults: dict[str, Any] = {
        "l1_baseline_mode": "peer_exclusive",
        "l1_qualify_by_regime": False,
        "l1_pair_min_effective_obs": 5.0,
        "l1_pair_alpha": 0.05,
        "l1_pair_power": 0.80,
        "l1_pair_mdes_multiplier": 0.5,
        "l1_pair_min_folds": 1,
        "l1_pair_min_mean_gross_bps": 0.0,
        "l1_pair_min_incremental_bps": 0.0,
        "l1_pair_min_positive_fold_ratio": 0.0,
        "l1_pair_fdr_alpha": 1.0,
        "l1_bootstrap_block_bars": 1,
        "l1_bootstrap_samples": 10,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def adverse_regime_event_frame() -> pd.DataFrame:
    bull_rows = {
        "realized_side_adjusted_gross_bps": [10.0] * 12,
        "entry_regime_code": [0] * 12,
        "decision_idx": list(range(12)),
    }
    crisis_rows = {
        "realized_side_adjusted_gross_bps": [-15.0] * 8,
        "entry_regime_code": [2] * 8,
        "decision_idx": list(range(12, 20)),
    }
    return pd.concat([pd.DataFrame(bull_rows), pd.DataFrame(crisis_rows)], ignore_index=True)


@pytest.fixture
def default_cfg() -> MagicMock:
    return _make_cfg()


# ─── Scenario 1 — Happy Path ───────────────────────────────────────────


def test_adverse_regime_detects_defeat(adverse_regime_event_frame: pd.DataFrame, default_cfg: MagicMock) -> None:
    """1.1 adverse 구간에서 방어 실패 감지: lcb < 0, defended=False."""
    lcb, n_obs, defended = compute_adverse_regime_evidence(
        adverse_regime_event_frame,
        cfg=default_cfg,
        fold_id=0,
        seed=42,
        min_bars=8,
    )
    assert n_obs == 8
    assert lcb is not None
    assert lcb < 0.0
    assert defended is False


def test_adverse_regime_short_sample_returns_undecided(
    adverse_regime_event_frame: pd.DataFrame,
    default_cfg: MagicMock,
) -> None:
    """2.2 adverse subset이 min_bars=8 미만(5행): (None, 5, True)."""
    short = adverse_regime_event_frame.iloc[:15]  # crisis = rows 12~15 → 4 rows only
    lcb, n_obs, defended = compute_adverse_regime_evidence(
        short,
        cfg=default_cfg,
        fold_id=0,
        seed=42,
        min_bars=8,
    )
    assert n_obs < 8
    assert lcb is None
    assert defended is True


def test_adverse_regime_no_entry_regime_column(default_cfg: MagicMock) -> None:
    """3.1 entry_regime_code 컬럼 없는 레거시 fixture: (None, 0, True)."""
    df = pd.DataFrame(
        {
            "realized_side_adjusted_gross_bps": [1.0, 2.0],
            "decision_idx": [0, 1],
        }
    )
    lcb, n_obs, defended = compute_adverse_regime_evidence(
        df,
        cfg=default_cfg,
        fold_id=0,
        seed=42,
    )
    assert lcb is None
    assert n_obs == 0
    assert defended is True


def test_adverse_regime_uses_entry_code_only(default_cfg: MagicMock) -> None:
    """2.1 exit_regime_code가 섞여 있어도 entry_regime_code만 참조."""
    df = pd.DataFrame(
        {
            "realized_side_adjusted_gross_bps": [10.0] * 5 + [-20.0] * 8,
            "entry_regime_code": [0] * 5 + [2] * 8,
            "exit_regime_code": [2] * 5 + [0] * 8,
            "decision_idx": list(range(13)),
        }
    )
    lcb, n_obs, defended = compute_adverse_regime_evidence(
        df,
        cfg=default_cfg,
        fold_id=0,
        seed=42,
        min_bars=8,
    )
    assert n_obs == 8
    assert lcb is not None
    assert lcb < 0.0
    assert defended is False


def test_existing_fields_unchanged_by_new_fields(
    adverse_regime_event_frame: pd.DataFrame,
    default_cfg: MagicMock,
) -> None:
    """2.5 신규 필드 3개만 추가되고 기존 필드는 동일."""
    event_results = adverse_regime_event_frame.copy()
    event_results["symbol"] = "BTCUSDT"
    event_results["strategy_id"] = "trend:fast_4h"
    event_results["activation_context"] = "all"
    event_results["fold_id"] = 0
    event_results["gross_event_bps"] = event_results["realized_side_adjusted_gross_bps"]
    event_results["uniqueness_weight"] = 1.0
    event_results["expected_holding_bars"] = 4
    evidence = compute_symbol_strategy_evidence(
        event_results=event_results,
        cfg=default_cfg,
        seed=42,
        registry_as_of_idx=100,
    )
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.adverse_regime_lcb_bps is not None
    assert ev.adverse_regime_n_obs > 0
    assert ev.adverse_regime_defended is not None
    assert isinstance(ev.hard_eligible, bool)
    assert isinstance(ev.lcb_net_bps, float)
