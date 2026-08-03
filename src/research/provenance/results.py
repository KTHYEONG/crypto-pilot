from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.research.baseline.backtest import BacktestResult
from src.research.cash_carry.contracts import CarryCostModel, CashCarrySpec
from src.research.contracts import CostModel, PortfolioSpec, StrategySpec
from src.research.provenance.ledger import (
    RUNS_LOG_PATH,
    LedgerEvent,
    append_event,
    build_evaluation_event,
    load_events,
)

if TYPE_CHECKING:
    from src.research.evaluation.metrics import Metrics
    from src.research.evaluation.promotion import CandidateIdentity, PromotionResult
    from src.research.evaluation.reliability import FoldDistributionResult, ReliabilityGateResult
    from src.research.expert_portfolio.admission_reports import LibraryAdmissionBacktestReport

_logger = logging.getLogger("ResultsLog")

_INITIAL_EQUITY_OI = 10_000.0
_INITIAL_EQUITY_TECHNICAL = 10_000.0


def _reliability_summary(gate: ReliabilityGateResult) -> dict[str, object]:
    return {
        "verdict": gate.verdict,
        "lcb90_cagr": gate.lcb90_cagr,
        "trade_count": gate.trade_count,
    }


def _promotion_summary(promotion: PromotionResult | None) -> dict[str, object] | None:
    if promotion is None:
        return None
    return {
        "status": promotion.status,
        "observation_verdict": promotion.observation_verdict,
        "fold_gate_pass": promotion.fold_gate_pass,
        "stress_verdict": promotion.stress_verdict,
        "holdout_verdict": promotion.holdout_verdict,
    }


def _git_head() -> tuple[str | None, bool]:
    """Return (short commit sha, has_uncommitted_changes); (None, False) outside a git repo."""
    # "git" resolved via PATH is the standard, portable form (not a partial-path risk here).
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False
    return sha, dirty


def _reliability_block(
    *,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None,
) -> dict[str, object]:
    return {
        "observation": _reliability_summary(observation_gate),
        "holdout": _reliability_summary(holdout_gate) if holdout_gate is not None else None,
        "fold_distribution": {
            "gate_pass": fold_distribution.gate_pass,
            "max_period_contribution": fold_distribution.max_period_contribution,
            "n_folds": fold_distribution.n_folds,
        },
        "stress_test": _reliability_summary(stress_gate),
    }


def _append_evaluation(event: LedgerEvent, log_path: Path) -> dict[str, object]:
    """Append one evaluation event via the ledger and return its flattened record."""
    appended = append_event(event, ledger_path=log_path)
    return {
        **dict(appended.payload),
        "record_type": appended.record_type,
        "schema_version": appended.schema_version,
    }


def _normalized_fingerprint(value: object) -> object:
    """JSON-normalise a fingerprint so tuples and ordering never block equality."""
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    )


def _resolve_active_parent(
    parent_registration_id: str,
    library_fingerprint: dict[str, object],
    log_path: Path,
) -> None:
    """Fail closed unless the parent registration is ACTIVE and fingerprint-matched."""
    for event in load_events(log_path):
        if event.record_type not in ("registration", "retirement"):
            continue
        if event.payload.get("registration_id") != parent_registration_id:
            continue
        if event.payload.get("status") != "ACTIVE":
            raise ValueError(f"parent registration {parent_registration_id} is not ACTIVE")
        spec_fingerprint = event.payload.get("spec_fingerprint")
        if _normalized_fingerprint(library_fingerprint) != spec_fingerprint:
            raise ValueError(
                f"parent registration {parent_registration_id} fingerprint does not match "
                "the evaluation library fingerprint"
            )
        return
    raise ValueError(f"parent registration {parent_registration_id} is absent from the ledger")


def record_run(
    *,
    spec: StrategySpec,
    costs: CostModel,
    result: BacktestResult,
    metrics: Metrics,
    start: str | None,
    end: str | None,
    initial_equity: float,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one baseline run as a ledger evaluation event.

    Never overwrites prior rows; each call appends exactly one immutable line
    capturing the frozen ``StrategySpec``/``CostModel``, the ``Metrics``
    outcome, the reliability gates, promotion, and the git commit.
    """
    git_sha, git_dirty = _git_head()
    event = build_evaluation_event(
        workflow="baseline",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        symbol=spec.symbol,
        start=start,
        end=end,
        initial_equity=initial_equity,
        spec=asdict(spec),
        costs=asdict(costs),
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def record_library_admission_backtest_run(
    report: LibraryAdmissionBacktestReport,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one contextual admission backtest evaluation to the ledger."""
    git_sha, git_dirty = _git_head()
    event = build_evaluation_event(
        workflow="library_admission_backtest",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(report.observation_metrics),
        reliability={
            "observation": asdict(report.observation_gate),
            "fold_distribution": asdict(report.observation_folds),
            "stress_test": asdict(report.stress_gate),
            "stress_folds": asdict(report.stress_folds),
        },
        promotion=asdict(report.promotion),
        proposal_id=report.proposal_id,
        expert_ids=list(report.expert_ids),
        router=asdict(report.router),
        window_start=report.window_start,
        window_end=report.window_end,
        observation_metrics=asdict(report.observation_metrics),
        stress_metrics=asdict(report.stress_metrics),
        allocation_cost_observation=report.allocation_cost_total,
        allocation_cost_stress=report.stress_allocation_cost_total,
        execution_workers=report.execution_workers,
        code_hash=report.code_hash,
        data_hashes={symbol: dict(values) for symbol, values in report.data_hashes.items()},
    )
    return _append_evaluation(event, log_path)


def record_portfolio_run(
    *,
    symbols: tuple[str, ...],
    portfolio_spec: PortfolioSpec,
    costs: CostModel,
    result: BacktestResult,
    metrics: Metrics,
    start: str | None,
    end: str | None,
    initial_equity: float,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one portfolio run as a ledger evaluation event."""
    git_sha, git_dirty = _git_head()
    event = build_evaluation_event(
        workflow="portfolio",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        kind="portfolio",
        symbols=list(symbols),
        start=start,
        end=end,
        initial_equity=initial_equity,
        portfolio_spec=asdict(portfolio_spec),
        costs=asdict(costs),
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def record_sleeve_blend_run(
    *,
    symbols: tuple[str, ...],
    candidate_kind: str,
    mdd_budget_fraction: float | None,
    leverage: float | None,
    costs: CostModel,
    result: BacktestResult,
    metrics: Metrics,
    start: str | None,
    end: str | None,
    initial_equity: float,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    universe_id: str | None = None,
    candidate_return_sources: list[str] | None = None,
    selection_window: str | None = None,
    qualification_window: str | None = None,
    leverage_schedule_hash: str | None = None,
    rejected_candidate_reasons: dict[str, str] | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one sleeve-blend run as a ledger evaluation event.

    For the causal tournament candidate the record binds the fixed universe id,
    the selected return-source identities, the discovery/qualification windows,
    the base leverage-schedule hash, and every rejected candidate reason, so the
    promotion verdict is fully auditable and never silently refitted.
    """
    git_sha, git_dirty = _git_head()
    event = build_evaluation_event(
        workflow="sleeve_blend",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        kind="sleeve_blend",
        candidate_kind=candidate_kind,
        symbols=list(symbols),
        mdd_budget_fraction=mdd_budget_fraction,
        leverage=leverage,
        universe_id=universe_id,
        candidate_return_sources=candidate_return_sources or [],
        selection_window=selection_window,
        qualification_window=qualification_window,
        leverage_schedule_hash=leverage_schedule_hash,
        rejected_candidate_reasons=rejected_candidate_reasons or {},
        start=start,
        end=end,
        initial_equity=initial_equity,
        costs=asdict(costs),
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def record_cash_carry_run(
    *,
    symbol: str,
    cash_carry_spec: CashCarrySpec,
    costs: CarryCostModel,
    result: BacktestResult,
    metrics: Metrics,
    start: str | None,
    end: str | None,
    initial_equity: float,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    candidate: CandidateIdentity | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one cash-and-carry run as a ledger evaluation event."""
    git_sha, git_dirty = _git_head()
    if candidate is not None:
        _logger.debug("cash_carry candidate provenance=%s", asdict(candidate))
    event = build_evaluation_event(
        workflow="cash_carry",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        kind="cash_carry",
        symbol=symbol,
        start=start,
        end=end,
        initial_equity=initial_equity,
        cash_carry_spec=asdict(cash_carry_spec),
        costs=asdict(costs),
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def record_oi_deleveraging_run(
    *,
    symbol: str,
    signal_delay_bars: int,
    costs: CostModel,
    result: BacktestResult,
    metrics: Metrics,
    start: str | None,
    end: str | None,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    candidate: CandidateIdentity | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one open-interest deleveraging screen run as a ledger evaluation event."""
    git_sha, git_dirty = _git_head()
    if candidate is not None:
        _logger.debug("oi_deleveraging candidate provenance=%s", asdict(candidate))
    event = build_evaluation_event(
        workflow="oi_deleveraging",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        kind="oi_deleveraging",
        symbol=symbol,
        signal_delay_bars=signal_delay_bars,
        start=start,
        end=end,
        initial_equity=_INITIAL_EQUITY_OI,
        costs=asdict(costs),
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def record_technical_expert_run(
    *,
    symbol: str,
    candidate_id: str,
    return_source: str,
    signal_delay_bars: int,
    costs: CostModel,
    result: BacktestResult,
    metrics: Metrics,
    start: str | None,
    end: str | None,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    candidate: CandidateIdentity | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one technical-expert candidate screen as a ledger evaluation event.

    The record binds the frozen candidate's return source, its immutable
    fingerprint (data hashes, costs, delay, and the catalog configuration), every
    gate outcome, and the promotion verdict. No mutable indicator parameter map
    is ever serialised.
    """
    git_sha, git_dirty = _git_head()
    if candidate is not None:
        _logger.debug("technical_expert candidate provenance=%s", asdict(candidate))
    event = build_evaluation_event(
        workflow="technical_expert",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        kind="technical_expert",
        symbol=symbol,
        candidate_id=candidate_id,
        return_source=return_source,
        signal_delay_bars=signal_delay_bars,
        start=start,
        end=end,
        initial_equity=_INITIAL_EQUITY_TECHNICAL,
        costs=asdict(costs),
        candidate_identity=asdict(candidate) if candidate is not None else None,
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def record_expert_portfolio_run(
    *,
    library_fingerprint: dict[str, object],
    allocation_cost_total: float,
    result: BacktestResult,
    metrics: Metrics,
    observation_gate: ReliabilityGateResult,
    fold_distribution: FoldDistributionResult,
    stress_gate: ReliabilityGateResult,
    holdout_gate: ReliabilityGateResult | None = None,
    promotion: PromotionResult | None = None,
    parent_registration_id: str | None = None,
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one expert-portfolio evaluation event linked to its registration.

    The evaluation binds to the immutable ``library_fingerprint``, the realised
    allocation turnover cost, the marked metrics, every gate output, and the
    promotion verdict.  When ``parent_registration_id`` is supplied it must
    resolve to an ACTIVE registration whose fingerprint matches the evaluation,
    otherwise no row is appended.  Logging is strictly append-only.
    """
    if not isinstance(library_fingerprint, dict) or not library_fingerprint:
        raise ValueError("library_fingerprint must be a non-empty dict")
    if "experts" not in library_fingerprint:
        raise ValueError("library_fingerprint must include the 'experts' library definitions")
    if not np.isfinite(float(allocation_cost_total)):
        raise ValueError(f"allocation_cost_total must be finite, got {allocation_cost_total}")
    if parent_registration_id:
        _resolve_active_parent(parent_registration_id, library_fingerprint, log_path)

    git_sha, git_dirty = _git_head()
    event = build_evaluation_event(
        workflow="expert_portfolio",
        ts=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        metrics=asdict(metrics),
        reliability=_reliability_block(
            observation_gate=observation_gate,
            fold_distribution=fold_distribution,
            stress_gate=stress_gate,
            holdout_gate=holdout_gate,
        ),
        promotion=_promotion_summary(promotion),
        parent_registration_id=parent_registration_id,
        kind="expert_portfolio",
        library_fingerprint=library_fingerprint,
        allocation_cost_total=float(allocation_cost_total),
        window="observation+holdout" if holdout_gate is not None else "observation",
    )
    return _append_evaluation(event, log_path)


def load_runs(log_path: Path = RUNS_LOG_PATH) -> pd.DataFrame:
    """Load all recorded rows as a flat DataFrame, one row per run, newest last.

    Legacy no-version rows are returned as-is; v1 events are flattened by
    merging their payload up beside the ``record_type``/``schema_version``
    markers so existing comparison columns stay available.
    """
    if not log_path.exists():
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            records.append({
                **data["payload"],
                "record_type": data.get("record_type"),
                "schema_version": data.get("schema_version"),
            })
        else:
            records.append(data)
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)
