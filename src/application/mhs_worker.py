"""Single inventory backtest worker without recursive supervision."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from src.application.mhs_backtest import MhsBacktestRequest, execute_mhs_backtest
from src.mhs.backtest.contracts import ProcessInventoryBacktestError
from src.mhs.resources import MhsMemoryBudget

_logger = logging.getLogger(__name__)

_PROCEDURE_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def _build_parser() -> argparse.ArgumentParser:
    """Receive the supervisor-attested source identity and forward it unchanged to the application request; the worker never invents a procedure identity."""
    parser = argparse.ArgumentParser(description="Execute one inventory backtest without recursive supervision.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--result-output", required=True)
    parser.add_argument("--targets-output", default=None)
    parser.add_argument("--rebalance-tracking-error-threshold", type=float, default=None)
    parser.add_argument("--total-tree-pss-bytes", type=int, required=True)
    parser.add_argument("--replay-tree-pss-bytes", type=int, required=True)
    parser.add_argument("--min-available-bytes", type=int, required=True)
    parser.add_argument("--evidence-root", default=None)
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--procedure-code-digest", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one inventory backtest with visible source-owned diagnostics.

    Receive the supervisor-attested source identity and forward it unchanged to
    the application request; the worker never invents a procedure identity.
    This subprocess configures its own standard logging because parent CLI
    configuration is not inherited across process execution.

    Args:
        argv: Explicit worker controls or process arguments.
    Returns:
        Zero after execution and requested evidence persistence complete;
        financial validity remains a separate report field.
    Raises:
        SystemExit: Worker arguments are invalid.
        ProcessInventoryBacktestError: Evaluation fails with preserved diagnostics.
        OSError: Requested evidence persistence fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
        digest = args.procedure_code_digest
        managed = (args.evidence_root, args.registry_path, args.run_id)
        managed_complete = all(item is not None for item in managed)
        managed_empty = all(item is None for item in managed)
        if digest is not None and (
            not isinstance(digest, str) or _PROCEDURE_DIGEST_RE.fullmatch(digest) is None
        ):
            raise ValueError(f"procedure-code-digest must be a lowercase SHA-256 hex identity, got {digest!r}")
        if managed_complete and digest is None:
            raise ValueError("managed canonical worker requires --procedure-code-digest")
        if managed_empty and digest is not None:
            raise ValueError("standalone worker must not carry --procedure-code-digest")
        if not managed_complete and not managed_empty:
            raise ValueError("managed publication context must be fully provided or all None")
        request = MhsBacktestRequest(
            start=start,
            end=end,
            data_root=args.data_root,
            result_output=Path(args.result_output),
            targets_output=Path(args.targets_output) if args.targets_output else None,
            tracking_error_threshold=args.rebalance_tracking_error_threshold,
            memory_budget=budget,
            evidence_root=Path(args.evidence_root) if args.evidence_root else None,
            registry_path=Path(args.registry_path) if args.registry_path else None,
            run_id=args.run_id,
            procedure_code_digest=digest,
        )
        _logger.info(
            "[WORKER] status=start start=%s end=%s result_output=%s",
            start.isoformat(),
            end.isoformat(),
            request.result_output,
        )
        execute_mhs_backtest(request)
        _logger.info(
            "[WORKER] status=persistence_complete result_output=%s",
            request.result_output,
        )
    except ProcessInventoryBacktestError:
        raise
    except ValueError as exc:
        raise SystemExit(f"invalid worker arguments: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
