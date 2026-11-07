"""The live golden regeneration tool never touches the reference frames without --write."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FIXTURES = Path("tests/fixtures/golden/live_decision")


def test_regen_without_write_refuses_and_leaves_fixtures() -> None:
    before = {path.name: path.read_bytes() for path in sorted(FIXTURES.glob("*.parquet"))}

    result = subprocess.run(  # noqa: S603 - fixed interpreter and repo script
        [sys.executable, "tools/regen_live_golden.py"], capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "refusing to regenerate without --write" in result.stderr
    assert {path.name: path.read_bytes() for path in sorted(FIXTURES.glob("*.parquet"))} == before
    assert set(before) == {"unit.parquet", "levered.parquet", "targets.parquet"}
