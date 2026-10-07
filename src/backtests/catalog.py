"""Single cross-pipeline backtest catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def append_backtest_index(
    *, index_path: Path, kind: str, run_dir: Path, created_at: pd.Timestamp,
    evaluation_start: pd.Timestamp, evaluation_end: pd.Timestamp,
    strategy_id: str, base_cagr: float | None, base_max_drawdown: float | None,
    execution: str | None = None,
) -> None:
    """Append one headline row to the single cross-pipeline backtest catalog.

    The catalog is the only file a human or analysis script needs to read to see every
    backtest ever run (canonical, frozen research or frozen account) with headline metrics
    inline; `registry.sqlite3` and `evidence/` stay internal plumbing for fingerprint reuse
    and content-addressed detail dedup. The catalog lives in the backtests package so that
    both CLI handlers and application services can append without an application->cli edge.

    Args:
        index_path: Destination `index.jsonl`, derived by the caller from the run root it owns.
        kind: Producing pipeline identity (`"mhs"`, `"mhs_frozen"`, `"mhs_frozen_account"`).
        run_dir: Directory holding that run's own result envelope; recorded relative to
            `index_path.parent` when inside it, otherwise as an absolute path.
        created_at: UTC timestamp of index-write time.
        evaluation_start: Registered evaluation start.
        evaluation_end: Registered evaluation end.
        strategy_id: Strategy identity string for the run.
        base_cagr: Headline CAGR, or None when not cheaply available.
        base_max_drawdown: Headline max drawdown as the producing pipeline reports it, or None.
        execution: Execution mode; the key is omitted from the row when None.
    Returns:
        None after appending exactly one `sort_keys` JSON line.
    Raises:
        OSError: The catalog cannot be opened for append.
    """
    try:
        rel_run_dir = str(run_dir.relative_to(index_path.parent))
    except ValueError:
        rel_run_dir = str(run_dir)
    record = {
        "kind": kind,
        "run_dir": rel_run_dir,
        "created_at": created_at.isoformat(),
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "strategy_id": strategy_id,
        "base_cagr": base_cagr,
        "base_max_drawdown": base_max_drawdown,
    }
    if execution is not None:
        record["execution"] = execution
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
