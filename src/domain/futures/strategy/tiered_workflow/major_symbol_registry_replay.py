# src/domain/futures/strategy/tiered_workflow/major_symbol_registry_replay.py

from __future__ import annotations

import csv
import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.candidate_contracts import (
    QualifiedSignalRegistry,
    ValidatedSignalBatch,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    MAJOR_DIAG_SYMBOLS,
    MajorSymbolRegistryCensusEntry,
    MajorSymbolSleeveContributionSummary,
    compute_major_symbol_registry_census,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,
    Layer2AllocationConfig,
    Layer2Result,
    Layer3Result,
)
from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    assert_selection_replay_parity,
)
from src.domain.futures.strategy.walk_forward import WFFold

if TYPE_CHECKING:
    pass

logger = logging.getLogger("opt_main_futures")


def classify_major_symbol_registry_gap(
    *,
    symbol: str,
    family: str,
    registry_entries: Sequence[MajorSymbolRegistryCensusEntry],
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary],
    adverse_sign_mismatch_threshold: float = 0.50,
    dead_zone: float = 1e-12,
) -> Literal["admission_gap", "activation_gap", "outvoted", "no_gap"]:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Registry gap classification for a (symbol, family) pair.

    Args:
        symbol: Target symbol.
        family: Target family.
        registry_entries: Census entries from L1 registry.
        observed_sleeve_summaries: Observed sleeve contribution summaries from holdout.
        adverse_sign_mismatch_threshold: Threshold for outvoted classification.
        dead_zone: Zero-sign threshold for mismatch computation.

    Returns:
        One of ``"admission_gap"``, ``"activation_gap"``, ``"outvoted"``, ``"no_gap"``.
    """
    sym_fam_rows = [e for e in registry_entries if e.symbol == symbol and e.family == family]
    if not sym_fam_rows or not any(e.hard_eligible for e in sym_fam_rows):
        return "admission_gap"

    observed_active = any(
        getattr(s, "symbol", None) == symbol and getattr(s, "family", None) == family
        for s in observed_sleeve_summaries
    )
    if not observed_active:
        return "activation_gap"

    for s in observed_sleeve_summaries:
        if getattr(s, "symbol", None) == symbol and getattr(s, "family", None) == family:
            mismatch_pct = float(getattr(s, "regime_adverse_sign_mismatch_pct", 0.0))
            if mismatch_pct >= adverse_sign_mismatch_threshold:
                return "outvoted"
            break

    return "no_gap"


@dataclass(slots=True, frozen=True)
class MajorSymbolRegistryReplayVariant:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Replay control variant for BTC dampener."""

    name: str
    l2_intra_symbol_divergence_enabled: bool
    l2_intra_symbol_divergence_symbols: tuple[str, ...] = ("BTCUSDT",)


@dataclass(slots=True, frozen=True)
class MajorSymbolRegistryReplayResult:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Replay result row for registry adoption A/B."""

    variant: str
    seed: int
    baseline_parity: bool
    l2_cagr: float
    l3_total_return: float
    l3_cagr: float
    l3_mdd: float
    l3_sharpe: float
    l3_sortino: float
    l3_trade_count: int
    btc_mu_bullish_pct: float
    eth_mu_bullish_pct: float
    registry_census: tuple[MajorSymbolRegistryCensusEntry, ...]
    adoption_passed: bool
    blocker_reason: str


def _major_symbol_registry_replay_variants() -> tuple[MajorSymbolRegistryReplayVariant, ...]:
    """Return the baseline and treatment replay variants for BTC divergence dampener."""
    return (
        MajorSymbolRegistryReplayVariant(
            name="baseline",
            l2_intra_symbol_divergence_enabled=False,
        ),
        MajorSymbolRegistryReplayVariant(
            name="btc_divergence_dampener",
            l2_intra_symbol_divergence_enabled=True,
        ),
    )


def run_major_symbol_registry_replay(
    *,
    seed: int,
    registry: QualifiedSignalRegistry | None,
    l2_signal_batch: ValidatedSignalBatch,
    l3_signal_batch: ValidatedSignalBatch,
    aligned: AlignedMarketData,
    awf_folds: tuple[WFFold, ...],
    holdout_span: tuple[int, int],
    config: Layer2AllocationConfig,
    caps: PortfolioCaps,
    tf: str,
    deploy_leverage: float | None,
    holdout_labels: tuple[str, str] | None = None,
    baseline_l2: Layer2Result | None = None,
    baseline_l3: Layer3Result | None = None,
    regime_code_1d: NDArray[np.int8] | None = None,
    prebuilt_cache: L2SimulationCache | None = None,
    eval_memo: dict[Any, Any] | None = None,
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
) -> tuple[MajorSymbolRegistryReplayResult, ...]:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Execute the BTC divergence dampener replay.

    Args:
        seed: Random seed.
        registry: L1 deployment registry (may be None).
        l2_signal_batch: L2 signal batch.
        l3_signal_batch: L3 signal batch.
        aligned: Aligned market data.
        awf_folds: AWF simulation folds.
        holdout_span: Holdout [start, end) span.
        config: L2 allocation config.
        caps: Portfolio caps.
        tf: Timeframe.
        deploy_leverage: Deploy leverage override.
        holdout_labels: Optional holdout label pair.
        baseline_l2: Baseline L2 result for parity check.
        baseline_l3: Baseline L3 result for parity check.
        regime_code_1d: Optional regime code array.
        prebuilt_cache: Optional prebuilt simulation cache.
        eval_memo: Optional eval memo dict.
        symbols: Major diagnostic symbols.

    Returns:
        Tuple of replay result rows.
    """
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        run_l2_awf,
        run_l3_holdout,
    )

    baseline_row: MajorSymbolRegistryReplayResult | None = None
    results: list[MajorSymbolRegistryReplayResult] = []

    for variant in _major_symbol_registry_replay_variants():
        variant_cfg = dataclasses.replace(
            config,
            l2_intra_symbol_divergence_enabled=variant.l2_intra_symbol_divergence_enabled,
            l2_intra_symbol_divergence_symbols=variant.l2_intra_symbol_divergence_symbols,
        )
        l2 = run_l2_awf(
            signal_batch=l2_signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=variant_cfg,
            caps=caps,
            tf=tf,
            verbose=False,
            deploy_leverage=deploy_leverage,
            prebuilt_cache=prebuilt_cache,
            eval_memo=eval_memo,
        )
        l3 = run_l3_holdout(
            signal_batch=l3_signal_batch,
            aligned=aligned,
            holdout_span=holdout_span,
            config=variant_cfg,
            caps=caps,
            tf=tf,
            holdout_labels=holdout_labels,
            verbose=False,
            deploy_leverage=deploy_leverage,
            regime_code_1d=regime_code_1d,
        )

        _baseline_parity = True
        if variant.name == "baseline" and baseline_l2 is not None:
            _l2_parity_ok = assert_selection_replay_parity(
                replay_evaluation=l2, final_evaluation=baseline_l2, tolerance=1e-6,
            )
            _l3_parity_ok = True if baseline_l3 is None else abs(l3.cagr - baseline_l3.cagr) < 1e-6
            _baseline_parity = _l2_parity_ok and _l3_parity_ok

        _registry_census = compute_major_symbol_registry_census(
            registry=registry,
            observed_sleeve_summaries=l3.major_symbol_sleeve_diag,
            symbols=symbols,
        )

        _btc_mu = 0.0
        _eth_mu = 0.0
        for diag in l3.major_symbol_diag:
            if diag.symbol == "BTCUSDT":
                _btc_mu = diag.mu_bullish_pct
            elif diag.symbol == "ETHUSDT":
                _eth_mu = diag.mu_bullish_pct

        row = MajorSymbolRegistryReplayResult(
            variant=variant.name,
            seed=seed,
            baseline_parity=_baseline_parity,
            l2_cagr=l2.cagr_hybrid,
            l3_total_return=l3.total_return,
            l3_cagr=l3.cagr,
            l3_mdd=l3.mdd,
            l3_sharpe=l3.sharpe,
            l3_sortino=l3.sortino,
            l3_trade_count=l3.n_trades,
            btc_mu_bullish_pct=_btc_mu,
            eth_mu_bullish_pct=_eth_mu,
            registry_census=_registry_census,
            adoption_passed=False,
            blocker_reason="",
        )
        if variant.name == "baseline":
            baseline_row = row
        results.append(row)

    _replayed_baseline_parity = baseline_row.baseline_parity if baseline_row else True
    for i, r in enumerate(results):
        if r.variant != "baseline":
            results[i] = dataclasses.replace(r, baseline_parity=_replayed_baseline_parity)

    if baseline_row is not None:
        baseline_rows = [results[0]] if results else []
        candidate_rows = [r for r in results if r.variant != "baseline"]
        if candidate_rows:
            _adoption, _reason = _major_symbol_registry_replay_adoption_verdict(
                baseline_rows=baseline_rows,
                candidate_rows=candidate_rows,
            )
            for i, r in enumerate(results):
                if r.variant != "baseline":
                    results[i] = dataclasses.replace(r, adoption_passed=_adoption, blocker_reason=_reason)

    return tuple(results)


def write_major_symbol_registry_replay_csv(
    results: Sequence[MajorSymbolRegistryReplayResult],
    *,
    path: Path,
) -> None:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Persist replay rows as a compact CSV artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow((
            "variant",
            "seed",
            "baseline_parity",
            "l2_cagr",
            "l3_total_return",
            "l3_cagr",
            "l3_mdd",
            "l3_sharpe",
            "l3_sortino",
            "l3_trade_count",
            "btc_mu_bullish_pct",
            "eth_mu_bullish_pct",
            "registry_census_count",
            "adoption_passed",
            "blocker_reason",
        ))
        for r in results:
            writer.writerow((
                r.variant,
                r.seed,
                r.baseline_parity,
                r.l2_cagr,
                r.l3_total_return,
                r.l3_cagr,
                r.l3_mdd,
                r.l3_sharpe,
                r.l3_sortino,
                r.l3_trade_count,
                r.btc_mu_bullish_pct,
                r.eth_mu_bullish_pct,
                len(r.registry_census),
                r.adoption_passed,
                r.blocker_reason,
            ))


def _major_symbol_registry_replay_adoption_verdict(
    *,
    baseline_rows: Sequence[MajorSymbolRegistryReplayResult],
    candidate_rows: Sequence[MajorSymbolRegistryReplayResult],
    min_trade_ratio: float = 0.75,
) -> tuple[bool, str]:
    """Gate BTC dampener adoption on multi-seed economics.

    Args:
        baseline_rows: Baseline replay result rows (one per seed).
        candidate_rows: Candidate replay result rows (one per seed).
        min_trade_ratio: Minimum candidate/baseline trade count ratio.

    Returns:
        (adoption_passed, blocker_reason).
    """
    for r in candidate_rows:
        if not r.baseline_parity:
            return False, "baseline_parity"

    deltas: list[dict[str, float]] = []
    for c in candidate_rows:
        for b in baseline_rows:
            if c.seed == b.seed:
                deltas.append({
                    "total_return_delta": c.l3_total_return - b.l3_total_return,
                    "mdd_delta": c.l3_mdd - b.l3_mdd,
                    "trade_ratio": c.l3_trade_count / max(b.l3_trade_count, 1),
                })
                break

    if not deltas:
        return False, "no_valid_seed_pairs"

    median_total_return_delta = float(np.median([d["total_return_delta"] for d in deltas]))
    if median_total_return_delta <= 0.0:
        return False, "below_median_total_return_delta"

    median_mdd_delta = float(np.median([d["mdd_delta"] for d in deltas]))
    if median_mdd_delta >= 0.0:
        return False, "median_mdd_not_improved"

    for d in deltas:
        if d["trade_ratio"] < min_trade_ratio:
            return False, "trade_collapse"

    return True, ""


def format_major_symbol_registry_replay_table(
    results: Sequence[MajorSymbolRegistryReplayResult],
) -> str:
    """Render the major symbol registry replay scorecard (diagnostic-only)."""
    lines = ["[MAJOR-SYMBOL-REGISTRY-REPLAY] Results:"]
    header = (
        f"  {'Variant':<24} {'Seed':>5} {'Parity':>7} {'L2-CAGR':>8} "
        f"{'L3-Ret':>8} {'L3-CAGR':>8} {'L3-MDD':>8} {'L3-Sharpe':>8} "
        f"{'L3-Sortino':>10} {'Trades':>7} {'BTC-μ%':>7} {'ETH-μ%':>7} "
        f"{'Adopt':>6} {'Blocker':<20}"
    )
    lines.append(header)
    for r in results:
        _adopt = "PASS" if r.adoption_passed else "BLOCK"
        lines.append(
            f"  {r.variant:<24} {r.seed:>5} {r.baseline_parity!s:>7} {r.l2_cagr:>8.4f} "
            f"{r.l3_total_return:>8.4f} {r.l3_cagr:>8.4f} {r.l3_mdd:>8.4f} "
            f"{r.l3_sharpe:>8.4f} {r.l3_sortino:>10.4f} {r.l3_trade_count:>7} "
            f"{r.btc_mu_bullish_pct:>7.3f} {r.eth_mu_bullish_pct:>7.3f} "
            f"{_adopt:>6} {r.blocker_reason:<20}"
        )
    return "\n".join(lines)
