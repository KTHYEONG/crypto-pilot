from __future__ import annotations

import logging

import numpy as np

from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    L3ValidationResult,
    MarketFeatureCube,
)
from src.domain.futures.compound.holdout_store import SealedHoldoutStore
from src.domain.futures.compound.l1_multiscale import run_l1_multiscale
from src.domain.futures.compound.simulator import simulate_multiscale_portfolio
from src.domain.futures.compound.validation import (
    evaluate_l2_walk_forward,
    evaluate_l3_sealed_holdout,
    slice_execution_ledger,
)

_logger = logging.getLogger(__name__)


def run_multiscale_compound_engine(
    *,
    market: MarketFeatureCube,
    universe: object,
    holdout_store: SealedHoldoutStore,
    holdout_id: str,
    config: CompoundEngineConfig,
) -> CompoundEngineResult:
    _logger.info("building multiscale alpha catalog")
    catalog = build_multiscale_alpha_catalog()

    _logger.info("running L1 multiscale causal edge proof")
    handoff = run_l1_multiscale(
        market=market,
        universe=universe,
        catalog=catalog,
        config=config.l1_multiscale,
    )

    _logger.info("simulating multiscale portfolio")
    ledger = simulate_multiscale_portfolio(
        market=market,
        universe=universe,
        handoff=handoff,
        config=config,
    )

    holdout_manifest = holdout_store.get_manifest(holdout_id)
    holdout_start_idx = int(np.searchsorted(
        ledger.timestamps_ns, holdout_manifest.start_time_ns,
    ))
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

    def evaluate_fn(manifest: object) -> L3ValidationResult:
        return evaluate_l3_sealed_holdout(
            l2_prior_returns=prior_returns,
            holdout_ledger=holdout_ledger,
            holdout_manifest=manifest,  # type: ignore[arg-type]
            config=config.l3,
        )

    l3_result = holdout_store.consume(
        holdout_id=holdout_id,
        model_version=holdout_manifest.model_version,
        data_manifest_hash=holdout_manifest.data_manifest_hash,
        strategy_spec_hash=holdout_manifest.strategy_spec_hash,
        evaluate=evaluate_fn,
        universe_state_hash=holdout_manifest.universe_state_hash,
    )

    _logger.info(
        "multiscale engine complete: recipes=%d l2_growth=%.6f l3_verdict=%s",
        len(catalog), l2_eval.annualized_log_growth, l3_result.verdict.value,
    )

    return CompoundEngineResult(
        handoff=handoff,
        ledger=ledger,
        l2=l2_eval,
        l3=l3_result,
    )
