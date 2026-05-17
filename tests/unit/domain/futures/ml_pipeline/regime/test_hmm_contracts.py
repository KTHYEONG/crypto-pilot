from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.features.engineering import HMM_SEMANTIC_PROB_COLUMNS
from src.domain.futures.ml_pipeline.pipeline_runner import _sorted_hmm_prob_columns
from src.domain.futures.optimization.optimizer import _hmm_columns_for_dyn_leverage


def test_canonical_hmm_order_from_semantic_columns() -> None:
    cols = [
        "hmm_prob_chop",
        "hmm_prob_bear_trend",
        "hmm_prob_crisis",
        "hmm_prob_bull_vol_up",
        "hmm_prob_bull_calm",
    ]
    df = pd.DataFrame({c: [0.2] for c in cols})

    expected = list(HMM_SEMANTIC_PROB_COLUMNS)
    assert _sorted_hmm_prob_columns(df) == expected
    assert _hmm_columns_for_dyn_leverage(df) == expected


def test_canonical_hmm_order_from_legacy_numbered_columns() -> None:
    df = pd.DataFrame(
        {
            "hmm_prob_3": [0.1],
            "hmm_prob_1": [0.2],
            "hmm_prob_0": [0.3],
            "hmm_prob_2": [0.4],
        }
    )

    expected = ["hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_prob_3"]
    assert _sorted_hmm_prob_columns(df) == expected
    assert _hmm_columns_for_dyn_leverage(df) == expected
