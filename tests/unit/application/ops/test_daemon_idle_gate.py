"""The CI deploy gate and the live daemon must agree on which stages are unsafe to interrupt."""

from __future__ import annotations

import subprocess
import sys

from src.application.ops import daemon_idle_gate
from src.common import daemon_stages
from src.live import daemon_idle


def test_gate_and_daemon_share_one_busy_stage_definition() -> None:
    assert daemon_idle_gate.BUSY_STAGES is daemon_stages.BUSY_STAGES
    assert daemon_idle.BUSY_STAGES is daemon_stages.BUSY_STAGES
    assert frozenset({"refresh", "signal", "execute"}) == daemon_stages.BUSY_STAGES


def test_gate_imports_without_the_project_environment() -> None:
    """The gate runs on a bare CI runner, so importing it must not pull third-party packages."""
    code = (
        "import sys; import src.application.ops.daemon_idle_gate; "
        "raise SystemExit(1 if {'pandas', 'numpy'} & set(sys.modules) else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)  # noqa: S603
    assert completed.returncode == 0, completed.stderr
