"""Invariant scenarios for the cross-pipeline backtest catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_append_relative_run_dir_inside_catalog_root(tmp_path: Path) -> None:
    """Two appends inside the root record a sorted relative run dir without execution."""
    from src.backtests.catalog import append_backtest_index

    index_path = tmp_path / "root" / "index.jsonl"
    index_path.parent.mkdir(parents=True)
    run_dir = tmp_path / "root" / "runs" / "a"
    run_dir.mkdir(parents=True)
    now = pd.Timestamp("2026-01-01", tz="UTC")
    for _ in range(2):
        append_backtest_index(
            index_path=index_path, kind="mhs", run_dir=run_dir, created_at=now,
            evaluation_start=now, evaluation_end=now, strategy_id="s",
            base_cagr=None, base_max_drawdown=None,
        )
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert row["run_dir"] == "runs/a"
        assert "execution" not in row
        assert list(row.keys()) == sorted(row.keys())


def test_append_backtest_index_falls_back_to_absolute_path_outside_root(tmp_path: Path) -> None:
    """A run directory outside the index root records its absolute path instead of raising."""
    from src.backtests.catalog import append_backtest_index

    index_path = tmp_path / "inside" / "index.jsonl"
    index_path.parent.mkdir(parents=True)
    outside_run_dir = tmp_path / "elsewhere" / "run1"
    outside_run_dir.mkdir(parents=True)
    now = pd.Timestamp("2026-01-01", tz="UTC")
    append_backtest_index(
        index_path=index_path, kind="mhs", run_dir=outside_run_dir, created_at=now,
        evaluation_start=now, evaluation_end=now, strategy_id="s",
        base_cagr=None, base_max_drawdown=None,
    )
    row = json.loads(index_path.read_text(encoding="utf-8").strip())
    assert row["run_dir"] == str(outside_run_dir)


def test_append_execution_key_only_when_provided(tmp_path: Path) -> None:
    """The execution key appears only when an execution mode is given."""
    from src.backtests.catalog import append_backtest_index

    index_path = tmp_path / "index.jsonl"
    now = pd.Timestamp("2026-01-01", tz="UTC")
    run_dir = tmp_path / "runs" / "a"
    run_dir.mkdir(parents=True)
    append_backtest_index(
        index_path=index_path, kind="mhs", run_dir=run_dir, created_at=now,
        evaluation_start=now, evaluation_end=now, strategy_id="s",
        base_cagr=0.1, base_max_drawdown=0.05, execution="maker",
    )
    append_backtest_index(
        index_path=index_path, kind="mhs", run_dir=run_dir, created_at=now,
        evaluation_start=now, evaluation_end=now, strategy_id="s",
        base_cagr=0.1, base_max_drawdown=0.05,
    )
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["execution"] == "maker"
    assert "execution" not in rows[1]
