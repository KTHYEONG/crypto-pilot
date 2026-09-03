"""Evaluation-window end policy: canonical tz-aware UTC resolution (I-CANONICAL-WINDOW)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.mhs.params import MHS_FINAL_OOS_CUTOFF_2026H1
from src.quant.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end


# SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES
def test_SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES() -> None:
    resolved_final_oos = resolve_evaluation_end(
        None, unseal_holdout=True, ceiling=MHS_FINAL_OOS_CUTOFF_2026H1
    )
    assert resolved_final_oos == pd.Timestamp("2026-06-30 23:59:59", tz="UTC")

    resolved_default = resolve_evaluation_end(None, unseal_holdout=False)
    assert resolved_default == HOLDOUT_CUTOFF
    assert str(resolved_default) == "2025-12-31 23:59:59+00:00"

    resolved_explicit = resolve_evaluation_end("2025-12-31", unseal_holdout=False)
    assert resolved_explicit == pd.Timestamp("2025-12-31 00:00:00", tz="UTC")

    with pytest.raises(RuntimeError, match="Holdout sealed"):
        resolve_evaluation_end("2026-01-01", unseal_holdout=False)

    for result in (resolved_final_oos, resolved_default, resolved_explicit):
        assert isinstance(result, pd.Timestamp)
        assert result.tz is not None


# SCENARIO_MHS_RESOLVE_EVALUATION_END_ALWAYS_RESOLVES (unsealed ceiling guard)
def test_unsealed_path_enforces_derived_ceiling() -> None:
    with pytest.raises(RuntimeError, match="Holdout sealed") as excinfo:
        resolve_evaluation_end(
            "2026-07-15",
            unseal_holdout=True,
            ceiling=MHS_FINAL_OOS_CUTOFF_2026H1,
        )
    message = str(excinfo.value)
    assert "2026-07-15" in message
    assert "2026-06-30" in message

    within = resolve_evaluation_end(
        "2026-06-30 23:59:59", unseal_holdout=True, ceiling=MHS_FINAL_OOS_CUTOFF_2026H1
    )
    assert isinstance(within, pd.Timestamp)
    assert within.tz is not None
