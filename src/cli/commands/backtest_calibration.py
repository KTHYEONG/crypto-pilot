"""``backtest fill-calibration``: tape-based calibration of the OHLCV and paper fill rules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def add_fill_calibration_command(subparsers: Any) -> None:
    """Register ``fill-calibration`` on the ``backtest`` subparsers."""
    calibration = subparsers.add_parser(
        "fill-calibration",
        help="Score OHLCV/paper fill rules against the tape model.",
        description="Score OHLCV/paper fill rules against the tape model.",
    )
    calibration.add_argument("--start", default=None, help="UTC evaluation start; date-only values are UTC.")
    calibration.add_argument("--end", default=None, help="UTC exclusive evaluation end.")
    calibration.add_argument("--archive-root", default=None, help="Public archive root holding aggTrades slices and coverage.")
    calibration.add_argument("--paper-run-dir", default=None, help="Paper run dir (VPS mirror: data/mirror/vps/state/runs/<run_id>).")
    calibration.add_argument("--live-capture-root", default=None, help="Live capture root (VPS mirror: data/mirror/vps/live_capture).")
    calibration.add_argument("--k-bars", nargs="+", type=int, default=None, help="K horizons in 3m bars.")
    calibration.add_argument("--touch-lookback-s", type=float, default=None, help="Touch proxy lookback in seconds.")
    calibration.add_argument("--queue-ahead-fraction", type=float, default=None, help="Fraction of observed queue ahead applied.")
    calibration.add_argument("--bootstrap-samples", type=int, default=None, help="Day-block bootstrap samples (>=100).")
    calibration.add_argument("--seed", type=int, default=None, help="Bootstrap seed.")
    calibration.add_argument("--output-dir", default=None, help="Calibration report root.")
    calibration.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    calibration.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    calibration.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    calibration.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    calibration.set_defaults(handler=run_fill_calibration_command)


def add_fill_parity_command(subparsers: Any) -> None:
    """Register ``fill-parity`` on the ``backtest`` subparsers."""
    parity = subparsers.add_parser(
        "fill-parity",
        help="Measure one day's paper versus tape versus backtest-rule parity.",
        description="Measure one day's paper versus tape versus backtest-rule parity.",
    )
    parity.add_argument("--day", default=None, help="Parity day YYYY-MM-DD (default: yesterday UTC).")
    parity.add_argument("--paper-run-dir", default=None, help="Paper run dir (default: resolve under data/state/runs).")
    parity.add_argument("--archive-root", default=None, help="Public archive root holding aggTrades slices.")
    parity.add_argument("--live-capture-root", default=None, help="Live capture root holding exec_depth.")
    parity.add_argument("--output", default=None, help="Parity JSONL report path.")
    parity.set_defaults(handler=run_fill_parity_command)


def _resolve_parity_run_dir(raw: str | None, day: Any) -> Path:
    import sys

    from src.common.paths import DATA_DIR

    if raw is not None:
        return Path(raw)
    runs_root = DATA_DIR / "state" / "runs"
    target = f"{day.isoformat()}.jsonl"
    matches: list[Path] = []
    if runs_root.is_dir():
        matches.extend(
            candidate for candidate in sorted(runs_root.iterdir())
            if candidate.is_dir() and (candidate / "audit" / target).exists()
        )
    if len(matches) != 1:
        print(f"require exactly one run dir with audit/{target} under {runs_root}, got {len(matches)}", file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2)
    return matches[0]


def run_fill_parity_command(args: argparse.Namespace) -> None:
    """Measure, append and gate one day's fill parity; exit 3 on a trailing breach."""
    import math
    import sys
    from datetime import date as _date

    from src.common.errors import DataIntegrityError
    from src.common.paths import DATA_DIR, LIVE_CAPTURE_DIR, PUBLIC_ARCHIVE_DIR

    raw_day = getattr(args, "day", None)
    if raw_day is None:
        day = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    else:
        try:
            day = _date.fromisoformat(str(raw_day))
        except ValueError:
            print(f"invalid --day {raw_day!r}, want YYYY-MM-DD", file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
            raise SystemExit(2) from None
    try:
        paper_run_dir = _resolve_parity_run_dir(getattr(args, "paper_run_dir", None), day)
    except SystemExit:
        raise
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2) from exc
    raw_archive = getattr(args, "archive_root", None)
    raw_live = getattr(args, "live_capture_root", None)
    raw_out = getattr(args, "output", None)
    archive_root = Path(raw_archive) if raw_archive is not None else PUBLIC_ARCHIVE_DIR
    live_root = Path(raw_live) if raw_live is not None else LIVE_CAPTURE_DIR
    output = Path(raw_out) if raw_out is not None else DATA_DIR / "state" / "exec_parity" / "fill_parity.jsonl"

    def _fmt(value: object) -> str:
        if value is None:
            return "nan"
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "nan"
        if math.isnan(number):
            return "nan"
        return f"{number}"

    try:
        from src.application.fill_parity import (
            FillParityThresholds,
            append_parity_row,
            measure_fill_parity,
            trailing_breach,
        )

        row = measure_fill_parity(day, paper_run_dir=paper_run_dir, archive_root=archive_root, live_capture_root=live_root)
        append_parity_row(row, output)
        breach = trailing_breach(output, FillParityThresholds())
    except (DataIntegrityError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2) from exc
    print(  # noqa: T201 -- parity command prints one summary line
        f"fill-parity day={row.day.isoformat()} run={row.run_id} status={row.status} "
        f"intents={row.n_intents} agreement={_fmt(row.through_agreement)} "
        f"paper_is_bps={_fmt(row.paper_is_bps)} rule_is_bps={_fmt(row.rule_is_bps)} gap_bps={_fmt(row.gap_bps)}"
    )
    if breach is not None:
        print(f"fill-parity breach={breach}")  # noqa: T201 -- breach line gates the archive job
        raise SystemExit(3)


def run_fill_calibration_command(args: argparse.Namespace) -> None:
    """Score OHLCV/paper fill rules against the tape model and persist the report."""
    import sys

    from src.application.fill_calibration import FillCalibrationRequest
    from src.application.strategy_account import AccountReplayError
    from src.cli.commands.backtest import _resolve_budget, _utc_timestamp
    from src.cli.commands.backtest_holdout import record_post_design_look
    from src.common.errors import DataIntegrityError
    from src.common.paths import EXEC_CALIBRATION_DIR, PUBLIC_ARCHIVE_DIR
    from src.core.params import PROCESS_EVALUATION_CEILING

    start = _utc_timestamp(getattr(args, "start", None), "start", pd.Timestamp("2023-01-01", tz="UTC"))
    end = _utc_timestamp(getattr(args, "end", None), "end", PROCESS_EVALUATION_CEILING)
    if not start < end:
        print(f"require start < end, got {start} {end}", file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2)
    try:
        from src.strategy.targets import FLOW_MOM_TOP20

        record_post_design_look("flow_mom_top20", "flow_mom", FLOW_MOM_TOP20, start, end, purpose="fill_calibration")
    except SystemExit:
        raise
    except (DataIntegrityError, ValueError) as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2) from exc
    raw_archive = getattr(args, "archive_root", None)
    raw_paper = getattr(args, "paper_run_dir", None)
    raw_live = getattr(args, "live_capture_root", None)
    raw_out = getattr(args, "output_dir", None)
    raw_k = getattr(args, "k_bars", None)
    raw_data_root = getattr(args, "data_root", None)
    try:
        request = FillCalibrationRequest(
            evaluation_start=start,
            evaluation_end=end,
            archive_root=Path(raw_archive) if raw_archive is not None else PUBLIC_ARCHIVE_DIR,
            k_bars=tuple(int(v) for v in raw_k) if raw_k is not None else (1, 3, 5, 10),
            touch_lookback_s=60.0 if args.touch_lookback_s is None else args.touch_lookback_s,
            queue_ahead_fraction=1.0 if args.queue_ahead_fraction is None else args.queue_ahead_fraction,
            bootstrap_samples=2000 if args.bootstrap_samples is None else args.bootstrap_samples,
            seed=7 if args.seed is None else args.seed,
            paper_run_dir=Path(raw_paper) if raw_paper is not None else None,
            live_capture_root=Path(raw_live) if raw_live is not None else None,
            data_root=Path(raw_data_root) if raw_data_root else None,
            memory_budget=_resolve_budget(args),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2) from exc
    output_dir = Path(raw_out) if raw_out is not None else EXEC_CALIBRATION_DIR
    from src.application.fill_calibration import run_fill_calibration

    try:
        report = run_fill_calibration(request, output_dir=output_dir)
    except (DataIntegrityError, AccountReplayError) as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201 -- CLI error surface is stderr
        raise SystemExit(2) from exc
    for period in sorted(report.backtest.get("periods", {})):
        node = report.backtest["periods"][period]
        rules = node.get("rules", {})
        n = int(rules.get(f"ohlcv_strict@K{request.k_bars[0]}", {}).get("n_intents", 0))
        share = node.get("marketable_anchor_share", float("nan"))
        ohlcv_fill = {k: rules.get(f"ohlcv_strict@K{k}", {}).get("fill_fraction") for k in request.k_bars}
        tape_fill = {k: rules.get(f"tape_through_feasible@K{k}", {}).get("fill_fraction") for k in request.k_bars}

        def _fmt(value: object) -> str:
            try:
                return f"{float(value)}"  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "nan"

        horizons = " ".join(f"ohlcv_fill@{k}={_fmt(ohlcv_fill[k])} tape_through_feasible_fill@{k}={_fmt(tape_fill[k])}" for k in request.k_bars)
        print(f"fill-calibration period={period} n={n} marketable_anchor_share={_fmt(share)} {horizons}")  # noqa: T201 -- calibration command prints one summary line per period
