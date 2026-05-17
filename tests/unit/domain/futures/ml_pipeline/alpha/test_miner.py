from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

project_root = str(Path(__file__).resolve().parents[6])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.alpha.miner import MLAlphaMiner

def test_miner_labels_preparation():
    """Tests the label preparation logic in MLAlphaMiner."""
    miner = MLAlphaMiner()
    
    # Mock data
    n_rows = 100
    target = pd.Series(np.linspace(0, 1, n_rows))
    raw_returns = np.random.normal(0, 0.02, n_rows)
    atr = np.full(n_rows, 0.02)
    
    labels_long = miner._prepare_labels(target, raw_returns=raw_returns, atr_24h_pct=atr, short_oriented=False)
    labels_short = miner._prepare_labels(target, raw_returns=raw_returns, atr_24h_pct=atr, short_oriented=True)
    
    assert len(labels_long) == n_rows
    assert len(labels_short) == n_rows
    assert not np.isnan(labels_long).any()
    assert not np.isnan(labels_short).any()
    
    # Check if sum is roughly 1.0
    np.testing.assert_allclose(labels_long + labels_short, 1.0, atol=1e-6)
