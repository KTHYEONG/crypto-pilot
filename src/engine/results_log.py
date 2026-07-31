from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.core.types import CarryCostModel, CashCarrySpec, CostModel, PortfolioSpec, StrategySpec
from src.engine.backtest import BacktestResult

if TYPE_CHECKING:
    from src.validation.candidate_promotion import CandidateIdentity, PromotionResult
    from src.validation.metrics import Metrics
    from src.validation.reliability_gate import FoldDistributionResult, ReliabilityGateResult

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "results" / "runs.jsonl"


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
            "observation": asdict(observation_gate),
            "holdout": asdict(holdout_gate) if holdout_gate is not None else None,
            "fold_distribution": asdict(fold_distribution),
            "stress_test": asdict(stress_gate),
        },
        "promotion": asdict(promotion) if promotion is not None else None,
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
            "observation": asdict(observation_gate),
            "holdout": asdict(holdout_gate) if holdout_gate is not None else None,
            "fold_distribution": asdict(fold_distribution),
            "stress_test": asdict(stress_gate),
        },
        "promotion": asdict(promotion) if promotion is not None else None,
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

    Logs the frozen ``CashCarrySpec``/``CarryCostModel``, the total-ledger
    ``Metrics``, the canonical reliability gates, the candidate identity, and
    the git commit. Never overwrites prior rows.
    """
    git_sha, git_dirty = _git_head()
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
            "observation": asdict(observation_gate),
            "holdout": asdict(holdout_gate) if holdout_gate is not None else None,
            "fold_distribution": asdict(fold_distribution),
            "stress_test": asdict(stress_gate),
        },
        "promotion": asdict(promotion) if promotion is not None else None,
        "candidate": asdict(candidate) if candidate is not None else None,
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
