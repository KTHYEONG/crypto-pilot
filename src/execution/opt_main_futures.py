"""Futures Optimization Execution Engine.

WARNING:
    This script consumes extreme physical memory during execution.
    To prevent Out-of-Memory (OOM) crashes under the resource-constrained WSL environment,
    this execution engine MUST NEVER be run in parallel or concurrently (single process only).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as script: python src/execution/opt_main_futures.py --phase l2
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.application.futures.runner.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
