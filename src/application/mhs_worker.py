"""Single inventory backtest worker without recursive supervision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.application.mhs_backtest import MhsBacktestRequest, execute_mhs_backtest
from src.mhs.process_backtest import ProcessInventoryBacktestError
from src.mhs.resources import MhsMemoryBudget


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute one inventory backtest without recursive supervision.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure-output", required=True)
    parser.add_argument("--targets-output", default=None)
    parser.add_argument("--rebalance-tracking-error-threshold", type=float, default=None)
    parser.add_argument("--total-tree-pss-bytes", type=int, required=True)
    parser.add_argument("--replay-tree-pss-bytes", type=int, required=True)
    parser.add_argument("--min-available-bytes", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one inventory backtest without recursive supervision.

    Args:
        argv: Explicit worker controls or process arguments.
    Returns:
        Zero only after successful inventory and requested target persistence.
    Raises:
        SystemExit: Worker arguments are invalid.
        ProcessInventoryBacktestError: Evaluation failed with preserved diagnostics.
        OSError: Requested evidence persistence failed.
    """
    args = _build_parser().parse_args(argv)
    try:
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        for label, value in (("start", start), ("end", end)):
            if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
                raise ValueError(f"{label} must be a timezone-aware ISO timestamp, got {value!r}")
        budget = MhsMemoryBudget(
            total_tree_pss_bytes=args.total_tree_pss_bytes,
            replay_tree_pss_bytes=args.replay_tree_pss_bytes,
            min_available_bytes=args.min_available_bytes,
        )
        request = MhsBacktestRequest(
            start=start,
            end=end,
            data_root=args.data_root,
            output=Path(args.output),
            failure_output=Path(args.failure_output),
            targets_output=Path(args.targets_output) if args.targets_output else None,
            tracking_error_threshold=args.rebalance_tracking_error_threshold,
            memory_budget=budget,
        )
        execute_mhs_backtest(request)
    except ProcessInventoryBacktestError:
        raise
    except ValueError as exc:
        raise SystemExit(f"invalid worker arguments: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
