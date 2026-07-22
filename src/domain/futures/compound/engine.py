from __future__ import annotations

import logging

import numpy as np

from src.domain.futures.compound.alpha_catalog import build_canonical_alpha_catalog, compute_raw_alpha_tape
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.l1_estimator import (
    build_causal_alpha_folds,
    estimate_cross_fitted_alpha_tape,
)
from src.domain.futures.compound.simulator import simulate_compound_portfolio
from src.domain.futures.compound.validation import (
    evaluate_l2_walk_forward,
    evaluate_l3_sealed_holdout,
    slice_execution_ledger,
)

_logger = logging.getLogger(__name__)


def run_compound_engine(
    *,
    cube: MarketFeatureCube,
    holdout_manifest: SealedHoldoutManifest,
    config: CompoundEngineConfig,
) -> CompoundEngineResult:
    _logger.info("Building canonical alpha catalog")
    catalog = build_canonical_alpha_catalog()

    _logger.info("Computing raw alpha tape")
    raw_tape = compute_raw_alpha_tape(cube=cube, catalog=catalog)

    _logger.info("Building causal alpha folds")
    holdout_bar = int(np.searchsorted(cube.timestamps_ns, holdout_manifest.start_time_ns))
    if holdout_bar == 0:
        holdout_bar = cube.timestamps_ns.size - 90 * 24

    n_bars_total = cube.timestamps_ns.size
    holdout_start = max(min(holdout_bar, n_bars_total - 1), n_bars_total // 2)
    folds = build_causal_alpha_folds(
        n_bars=n_bars_total,
        fit_start=0,
        holdout_start=holdout_start,
        n_folds=config.l1.n_folds,
        purge_bars=config.l1.purge_bars,
        embargo_bars=config.l1.embargo_bars,
    )

    _logger.info("Estimating cross-fitted alpha tape")
    alpha_tape = estimate_cross_fitted_alpha_tape(
        raw=raw_tape, cube=cube, folds=folds, config=config.l1
    )

    _logger.info("Simulating compound portfolio")
    ledger = simulate_compound_portfolio(cube=cube, tape=alpha_tape, config=config)

    holdout_start_idx = int(np.searchsorted(ledger.timestamps_ns, holdout_manifest.start_time_ns))
    holdout_start_idx = max(1, min(holdout_start_idx, ledger.timestamps_ns.size - 1))
    l2_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[0]),
        end_time_ns=int(ledger.timestamps_ns[holdout_start_idx - 1]),
    )

    _logger.info("Evaluating L2 walk-forward")
    l2_eval = evaluate_l2_walk_forward(
        ledger=l2_ledger, bars_per_year=8766.0, bootstrap_seed=42
    )

    _logger.info("Evaluating L3 sealed holdout")
    prior_returns = l2_ledger.net_returns_1d[-config.l3.l2_prior_effective_days_cap:]
    holdout_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[holdout_start_idx]),
        end_time_ns=int(ledger.timestamps_ns[-1]),
    )

    l3_result = evaluate_l3_sealed_holdout(
        l2_prior_returns=prior_returns,
        holdout_ledger=holdout_ledger,
        holdout_manifest=holdout_manifest,
        config=config.l3,
    )

    return CompoundEngineResult(
        alpha_tape=alpha_tape,
        ledger=ledger,
        l2=l2_eval,
        l3=l3_result,
    )
