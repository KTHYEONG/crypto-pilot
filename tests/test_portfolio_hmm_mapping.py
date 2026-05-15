from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.portfolio.portfolio_constructor import precompute_rebalance_weights


def _base_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bars = 3
    n_syms = 2
    close_2d = np.array(
        [
            [100.0, 101.0],
            [100.5, 101.5],
            [101.0, 102.0],
        ],
        dtype=np.float64,
    )
    xs_long = np.full((n_bars, n_syms), 1.0, dtype=np.float64)
    xs_short = np.zeros((n_bars, n_syms), dtype=np.float64)
    sigma_3d = np.repeat(np.eye(n_syms, dtype=np.float64)[None, :, :] * 1e-4, n_bars, axis=0)
    return close_2d, xs_long, xs_short, sigma_3d


def test_portfolio_uses_crisis_from_canonical_5col_order() -> None:
    close_2d, xs_long, xs_short, sigma_3d = _base_inputs()

    # Canonical 5-col: [bull_calm, bull_vol_up, bear, chop, crisis]
    # crisis=0.95 should trigger hard flat when interpreted correctly.
    hmm_probs_2d = np.array(
        [
            [0.02, 0.02, 0.01, 0.00, 0.95],
            [0.02, 0.02, 0.01, 0.00, 0.95],
            [0.02, 0.02, 0.01, 0.00, 0.95],
        ],
        dtype=np.float64,
    )
    regime_betas = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)

    w = precompute_rebalance_weights(
        close_2d,
        xs_long,
        xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=1.0,
        per_symbol_cap=1.0,
        sigma_3d=sigma_3d,
        hmm_probs_2d=hmm_probs_2d,
        regime_betas=regime_betas,
        crisis_override_thr=0.4,
    )

    assert np.allclose(w[1:], 0.0, atol=1e-12)


def test_portfolio_legacy_4col_crisis_index_still_supported() -> None:
    close_2d, xs_long, xs_short, sigma_3d = _base_inputs()

    # Legacy 4-col: [bull, bear, chop, crisis]
    hmm_probs_2d = np.array(
        [
            [0.02, 0.02, 0.01, 0.95],
            [0.02, 0.02, 0.01, 0.95],
            [0.02, 0.02, 0.01, 0.95],
        ],
        dtype=np.float64,
    )
    regime_betas = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)

    w = precompute_rebalance_weights(
        close_2d,
        xs_long,
        xs_short,
        rebalance_bars=1,
        lookback=2,
        bars_per_year=365.0 * 24.0,
        kappa=0.3,
        f_kelly_max=1.0,
        sigma_target_ann=0.2,
        gross_cap=1.0,
        per_symbol_cap=1.0,
        sigma_3d=sigma_3d,
        hmm_probs_2d=hmm_probs_2d,
        regime_betas=regime_betas,
        crisis_override_thr=0.4,
    )

    assert np.allclose(w[1:], 0.0, atol=1e-12)
