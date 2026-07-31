from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.research.baseline.backtest import BacktestResult
from src.research.cash_carry.contracts import CarryCostModel, CashCarrySpec
from src.research.contracts import CostModel, PortfolioSpec, StrategySpec

if TYPE_CHECKING:
    from src.research.evaluation.metrics import Metrics
    from src.research.evaluation.promotion import CandidateIdentity, PromotionResult
    from src.research.evaluation.reliability import FoldDistributionResult, ReliabilityGateResult

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "results" / "runs.jsonl"
_logger = logging.getLogger("ResultsLog")


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
    """Append one backtest run as a JSONL record for longitudinal comparison.

    Never overwrites prior rows; each call appends exactly one line. Captures
    the frozen StrategySpec/CostModel (what changed) alongside Metrics (the
    outcome), the reliability gate verdicts, the promotion result, and the git
    commit (whether the code that produced it is reproducible from history).
    """
    git_sha, git_dirty = _git_head()
    record: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "symbol": spec.symbol,
        "start": start,
        "end": end,
        "initial_equity": initial_equity,
        "spec": asdict(spec),
        "costs": asdict(costs),
        "metrics": asdict(metrics),
        "reliability": {
            "observation": _reliability_summary(observation_gate),
            "holdout": _reliability_summary(holdout_gate) if holdout_gate is not None else None,
            "fold_distribution": {
                "gate_pass": fold_distribution.gate_pass,
                "max_period_contribution": fold_distribution.max_period_contribution,
                "n_folds": fold_distribution.n_folds,
            },
            "stress_test": _reliability_summary(stress_gate),
        },
        "promotion": _promotion_summary(promotion),
        "window": "observation+holdout" if holdout_gate is not None else "observation",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


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
    """Append one portfolio run as a JSONL record for longitudinal comparison.

    Logs the daily-liquid candidate pool, the frozen ``PortfolioSpec`` (what
    changed), the total-ledger ``Metrics``, the portfolio reliability gates, the
    promotion result, and the git commit. Never overwrites prior rows.
    """
    git_sha, git_dirty = _git_head()
    record: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "kind": "portfolio",
        "symbols": list(symbols),
        "start": start,
        "end": end,
        "initial_equity": initial_equity,
        "portfolio_spec": asdict(portfolio_spec),
        "costs": asdict(costs),
        "metrics": asdict(metrics),
        "reliability": {
            "observation": _reliability_summary(observation_gate),
            "holdout": _reliability_summary(holdout_gate) if holdout_gate is not None else None,
            "fold_distribution": {
                "gate_pass": fold_distribution.gate_pass,
                "max_period_contribution": fold_distribution.max_period_contribution,
                "n_folds": fold_distribution.n_folds,
            },
            "stress_test": _reliability_summary(stress_gate),
        },
        "promotion": _promotion_summary(promotion),
        "window": "observation+holdout" if holdout_gate is not None else "observation",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


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
    """Append one cash-and-carry research run as a JSONL record.

    Logs only comparison-oriented outcomes and frozen strategy parameters.
    Detailed candidate/provenance data is emitted at DEBUG level instead of
    being persisted in the longitudinal comparison log.
    """
    git_sha, git_dirty = _git_head()
    if candidate is not None:
        _logger.debug("cash_carry candidate provenance=%s", asdict(candidate))
    record: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "kind": "cash_carry",
        "symbol": symbol,
        "start": start,
        "end": end,
        "initial_equity": initial_equity,
        "cash_carry_spec": asdict(cash_carry_spec),
        "costs": asdict(costs),
        "metrics": asdict(metrics),
        "reliability": {
            "observation": _reliability_summary(observation_gate),
            "holdout": _reliability_summary(holdout_gate) if holdout_gate is not None else None,
            "fold_distribution": {
                "gate_pass": fold_distribution.gate_pass,
                "max_period_contribution": fold_distribution.max_period_contribution,
                "n_folds": fold_distribution.n_folds,
            },
            "stress_test": _reliability_summary(stress_gate),
        },
        "promotion": _promotion_summary(promotion),
        "window": "observation+holdout" if holdout_gate is not None else "observation",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_runs(log_path: Path = RUNS_LOG_PATH) -> pd.DataFrame:
    """Load all recorded runs as a flat DataFrame, one row per run, newest last."""
    if not log_path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)
