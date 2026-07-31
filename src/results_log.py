from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import CostModel, StrategySpec
from src.engine import BacktestResult
from src.metrics import Metrics

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "results" / "runs.jsonl"


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
    log_path: Path = RUNS_LOG_PATH,
) -> dict[str, object]:
    """Append one backtest run as a JSONL record for longitudinal comparison.

    Never overwrites prior rows; each call appends exactly one line. Captures
    the frozen StrategySpec/CostModel (what changed) alongside Metrics (the
    outcome) and the git commit (whether the code that produced it is
    reproducible from history).
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
