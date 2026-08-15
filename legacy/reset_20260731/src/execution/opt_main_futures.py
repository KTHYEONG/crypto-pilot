"""Futures Optimization Execution Engine — Multiscale Causal Compound Pipeline.

Entry point: data lake → daily PIT universe → L1 → L2 → L3 path only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.application.futures.runner.cli import cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli())  # pragma: no cover
