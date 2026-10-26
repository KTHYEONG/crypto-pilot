"""Tape-based fill calibration report for the OHLCV and paper fill rules."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.application.fill_calibration_backtest import build_backtest_intents, score_backtest
from src.application.fill_calibration_paper import score_paper
from src.common.errors import DataIntegrityError
from src.common.parquet_io import write_parquet_atomic
from src.core.resources import MhsMemoryBudget


def _utc_now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _round_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _round_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_json(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if np.isfinite(f):
            return round(f, 6)
        return None
    return value


@dataclass(frozen=True, slots=True)
class FillCalibrationRequest:
    """Typed controls for one tape-based fill calibration report."""

    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    archive_root: Path
    k_bars: tuple[int, ...] = (1, 3, 5, 10)
    touch_lookback_s: float = 60.0
    queue_ahead_fraction: float = 1.0
    bootstrap_samples: int = 2000
    seed: int = 7
    paper_run_dir: Path | None = None
    live_capture_root: Path | None = None
    data_root: Path | None = None
    memory_budget: MhsMemoryBudget | None = None

    def __post_init__(self) -> None:
        start, end = self.evaluation_start, self.evaluation_end
        for label, value in (("evaluation_start", start), ("evaluation_end", end)):
            if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
                raise ValueError(f"{label} must be a timezone-aware Timestamp, got {value!r}")
            if pd.Timestamp(value).utcoffset() != pd.Timedelta(0):
                raise ValueError(f"{label} must be UTC, got {value!r}")
        s_utc = pd.Timestamp(start).tz_convert("UTC")
        e_utc = pd.Timestamp(end).tz_convert("UTC")
        if not s_utc < e_utc:
            raise ValueError(f"require evaluation_start < evaluation_end, got {s_utc} {e_utc}")
        kb = self.k_bars
        if (
            not isinstance(kb, tuple)
            or not kb
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in kb)
        ):
            raise ValueError(f"k_bars must be a non-empty tuple of positive ints, got {kb!r}")
        ints = [int(v) for v in kb]
        if len(set(ints)) != len(ints) or tuple(sorted(ints)) != tuple(ints):
            raise ValueError(f"k_bars must be sorted unique positive ints, got {kb!r}")
        if (
            not isinstance(self.touch_lookback_s, (int, float))
            or isinstance(self.touch_lookback_s, bool)
            or not np.isfinite(float(self.touch_lookback_s))
            or not float(self.touch_lookback_s) > 0
        ):
            raise ValueError(f"touch_lookback_s must be > 0, got {self.touch_lookback_s!r}")
        if (
            not isinstance(self.bootstrap_samples, (int, np.integer))
            or isinstance(self.bootstrap_samples, bool)
            or int(self.bootstrap_samples) < 100
        ):
            raise ValueError(f"bootstrap_samples must be >= 100, got {self.bootstrap_samples!r}")
        paper_set = self.paper_run_dir is not None
        live_set = self.live_capture_root is not None
        if paper_set != live_set:
            raise ValueError("paper_run_dir and live_capture_root must be set together or both omitted")
        for label in ("archive_root", "paper_run_dir", "live_capture_root", "data_root"):
            value = getattr(self, label)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"{label} must be a Path, got {value!r}")
        from src.engine.execution.tape_fill import TapeFillSpec

        TapeFillSpec(queue_ahead_fraction=self.queue_ahead_fraction)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if int(self.touch_lookback_s * 1000) < 1:
            raise ValueError("touch_lookback_s must be at least one millisecond")


@dataclass(frozen=True, slots=True)
class FillCalibrationReport:
    """Persisted outcome of one fill calibration run."""

    backtest: dict[str, Any]
    paper: dict[str, Any] | None
    exclusions: dict[str, int]
    intents_path: Path
    report_path: Path


def _load_backtest_inputs(request: FillCalibrationRequest) -> tuple[Any, Any, Any, Any, Any, pd.DatetimeIndex]:
    from src.application.strategy_account import AccountReplayRequest, build_account_replay_inputs
    from src.common.paths import STRATEGY_BACKTESTS_DIR, VENUE_RULES_DIR
    from src.core.params import DISCOVERY_START

    replay_request = AccountReplayRequest(
        source_start=DISCOVERY_START,
        evaluation_start=request.evaluation_start,
        evaluation_end=request.evaluation_end,
        runs_root=STRATEGY_BACKTESTS_DIR,
        venue_rules_root=VENUE_RULES_DIR,
        execution="maker",
        data_root=request.data_root,
        memory_budget=request.memory_budget,
    )
    return build_account_replay_inputs(replay_request)


def run_fill_calibration(request: FillCalibrationRequest, *, output_dir: Path) -> FillCalibrationReport:
    """Score OHLCV/paper fill rules against the tape model and persist report.json + intents.parquet under output_dir; raises DataIntegrityError on malformed inputs."""
    if not isinstance(request, FillCalibrationRequest):
        raise DataIntegrityError(f"request must be FillCalibrationRequest, got {type(request).__name__}")
    if not isinstance(output_dir, Path):
        raise DataIntegrityError(f"output_dir must be a Path, got {output_dir!r}")
    if not isinstance(request.archive_root, Path):
        raise DataIntegrityError(f"archive_root must be a Path, got {request.archive_root!r}")
    stamp = _utc_now_stamp()
    target = Path(output_dir) / stamp
    if os.path.lexists(target):
        raise DataIntegrityError(f"calibration output must be fresh: {target} already exists")
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DataIntegrityError(f"calibration output must be fresh: {target} already exists") from exc
    exclusions: dict[str, int] = {}
    try:
        parts = _load_backtest_inputs(request)
    except DataIntegrityError as exc:
        raise DataIntegrityError(f"malformed calibration inputs: {exc}") from exc
    intents = build_backtest_intents(parts)
    backtest_rows, backtest_summary = score_backtest(request, intents, parts, exclusions)
    paper_rows: list[dict[str, Any]] = []
    paper_summary: dict[str, Any] | None = None
    if request.paper_run_dir is not None:
        paper_rows, paper_summary = score_paper(request, exclusions)
    all_rows = [*backtest_rows, *paper_rows]
    frame = pd.DataFrame(all_rows) if all_rows else pd.DataFrame({"cohort": pd.Series(dtype="object")})
    intents_path = target / "intents.parquet"
    try:
        intents_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_atomic(frame, intents_path, compression="zstd")
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise DataIntegrityError(f"calibration intents write failed: {exc}") from exc
    payload = {"backtest": backtest_summary, "paper": paper_summary, "exclusions": dict(sorted(exclusions.items()))}
    rounded = _round_json(payload)
    report_path = target / "report.json"
    tmp = target / ".report.json.tmp"
    try:
        tmp.write_text(json.dumps(rounded, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, report_path)
    except OSError as exc:
        raise DataIntegrityError(f"calibration report write failed: {exc}") from exc
    return FillCalibrationReport(
        backtest=backtest_summary,
        paper=paper_summary,
        exclusions=dict(sorted(exclusions.items())),
        intents_path=intents_path,
        report_path=report_path,
    )
