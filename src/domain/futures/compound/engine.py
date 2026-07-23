from __future__ import annotations

import logging

import numpy as np

from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.domain.futures.compound.alpha_catalog import (
    build_canonical_alpha_catalog,
    build_multiscale_alpha_catalog,
    compute_raw_alpha_tape,
)
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    CombinedForecast,
    CompoundEngineResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.l1_estimator import (
    build_causal_alpha_folds,
    build_causal_alpha_forecasts,
)
from src.domain.futures.compound.l1_multiscale import run_l1_multiscale
from src.domain.futures.compound.simulator import simulate_compound_portfolio, simulate_multiscale_portfolio
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
    n_bars_total = cube.timestamps_ns.size
    holdout_bar = int(np.searchsorted(cube.timestamps_ns, holdout_manifest.start_time_ns))
    if holdout_bar == 0:
        holdout_bar = n_bars_total - min(90 * 24, n_bars_total // 2)

    holdout_start = max(min(holdout_bar, n_bars_total - 1), n_bars_total // 2)

    folds = build_causal_alpha_folds(
        n_bars=n_bars_total,
        fit_start=0,
        holdout_start=holdout_start,
        n_folds=config.l1.n_folds,
        purge_bars=config.l1.purge_bars,
        embargo_bars=config.l1.embargo_bars,
    )

    _logger.info("Building causal alpha forecasts (cross-fit OOS + frozen holdout)")
    alpha_tape = build_causal_alpha_forecasts(
        raw=raw_tape,
        cube=cube,
        folds=folds,
        holdout_start_idx=holdout_start,
        config=config.l1,
    )

    _logger.info("Simulating compound portfolio")
    ledger = simulate_compound_portfolio(cube=cube, alpha_tape=alpha_tape, config=config)

    holdout_start_idx = int(np.searchsorted(ledger.timestamps_ns, holdout_manifest.start_time_ns))
    holdout_start_idx = max(1, min(holdout_start_idx, ledger.timestamps_ns.size - 1))

    l2_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[0]),
        end_time_ns=int(ledger.timestamps_ns[holdout_start_idx - 1]),
    )

    _logger.info("Evaluating L2 walk-forward (pre-holdout only)")
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

    n_estimated = int(np.sum(alpha_tape.estimated_3d))
    n_support = 0
    for t in range(alpha_tape.timestamps_ns.size):
        fc = combine_forecasts_diagnostic(alpha_tape, t, config.allocator.uncertainty_z)
        n_support += int(np.sum(fc.support_1d))
    n_nonzero = int(np.sum(np.abs(ledger.target_weights_2d) > 1e-8))

    _logger.info(
        "Diagnostics: estimated_cells=%d support_events=%d nonzero_weight_positions=%d",
        n_estimated, n_support, n_nonzero,
    )

    return CompoundEngineResult(
        alpha_tape=alpha_tape,
        ledger=ledger,
        l2=l2_eval,
        l3=l3_result,
    )


def combine_forecasts_diagnostic(
    tape: AlphaForecastTape, time_idx: int, uncertainty_z: float,
) -> CombinedForecast:
    from src.domain.futures.compound.allocator import combine_alpha_forecasts
    return combine_alpha_forecasts(tape, time_idx, uncertainty_z=uncertainty_z)


def run_multiscale_compound_engine(
    *,
    market: MarketFeatureCube,
    universe: DailyPITUniverse,
    holdout_manifest: SealedHoldoutManifest,
    config: CompoundEngineConfig,
) -> CompoundEngineResult:
    _logger.info("building multiscale alpha catalog")
    catalog = build_multiscale_alpha_catalog()

    _logger.info("running L1 multiscale causal edge proof (family/timeframe selection)")
    alpha_catalog = catalog
    handoff = run_l1_multiscale(market=market, universe=universe, catalog=alpha_catalog, config=config.l1_multiscale)

    _logger.info("simulating multiscale portfolio")
    ledger = simulate_multiscale_portfolio(market=market, universe=universe, handoff=handoff, config=config)

    holdout_start_idx = int(np.searchsorted(ledger.timestamps_ns, holdout_manifest.start_time_ns))
    holdout_start_idx = max(1, min(holdout_start_idx, ledger.timestamps_ns.size - 1))

    l2_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[0]),
        end_time_ns=int(ledger.timestamps_ns[holdout_start_idx - 1]),
    )

    _logger.info("evaluating L2 walk-forward")
    l2_eval = evaluate_l2_walk_forward(
        ledger=l2_ledger, bars_per_year=8766.0, bootstrap_seed=42,
    )

    holdout_ledger = slice_execution_ledger(
        ledger=ledger,
        start_time_ns=int(ledger.timestamps_ns[holdout_start_idx]),
        end_time_ns=int(ledger.timestamps_ns[-1]),
    )
    prior_returns = l2_ledger.net_returns_1d[-config.l3.l2_prior_effective_days_cap:]

    l3_result = evaluate_l3_sealed_holdout(
        l2_prior_returns=prior_returns,
        holdout_ledger=holdout_ledger,
        holdout_manifest=holdout_manifest,
        config=config.l3,
    )

    alpha_tape = AlphaForecastTape(
        timestamps_ns=market.timestamps_ns,
        symbols=market.symbols,
        recipe_ids=tuple(d.recipe_id for d in catalog),
        gross_mu_3d=np.zeros((market.timestamps_ns.size, len(market.symbols), len(catalog)), dtype=np.float32),
        mean_edge_var_3d=np.full((market.timestamps_ns.size, len(market.symbols), len(catalog)), 1e-4, dtype=np.float32),
        residual_var_3d=np.full((market.timestamps_ns.size, len(market.symbols), len(catalog)), 1e-4, dtype=np.float32),
        reliability_3d=np.zeros((market.timestamps_ns.size, len(market.symbols), len(catalog)), dtype=np.float32),
        estimated_3d=np.zeros((market.timestamps_ns.size, len(market.symbols), len(catalog)), dtype=np.bool_),
        valid_3d=np.zeros((market.timestamps_ns.size, len(market.symbols), len(catalog)), dtype=np.bool_),
        horizon_bars_1d=np.array([h // 24 for h in [d.horizon_hours for d in catalog]], dtype=np.int16),
        lifecycle_by_recipe=(),
        model_version="multiscale-v1",
        data_manifest_hash=market.data_manifest_hash,
        fold_manifest_hash=handoff.fold_manifest_hash,
    )

    _logger.info(
        "multiscale engine complete: recipes=%d l2_growth=%.6f l3_verdict=%s",
        len(catalog), l2_eval.annualized_log_growth, l3_result.verdict.value,
    )

    return CompoundEngineResult(
        alpha_tape=alpha_tape,
        ledger=ledger,
        l2=l2_eval,
        l3=l3_result,
    )
