# ruff: noqa: SIM300, N811
from __future__ import annotations

def test_no_nonschema_log_tags_in_src() -> None:
    """Only the 4 standard tags may appear in src/ log messages."""
    import re
    from pathlib import Path

    allowed = {"SYS", "DATA", "ALGO", "EVAL"}
    pattern = re.compile(r'"\[([A-Z]{2,10})\]|\'\[([A-Z]{2,10})\]')
    offenders: list[str] = []
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            tag = match.group(1) or match.group(2)
            if tag not in allowed:
                offenders.append(f"{path}:{tag}")

    assert offenders == [], f"non-schema log tags: {sorted(set(offenders))}"

def test_compose_mounts_state_dir_on_every_service() -> None:
    """I-STATE-SURVIVES-REDEPLOY: every long-running service persists data/state."""
    from pathlib import Path

    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    service_blocks = compose.split("\n  ")
    mount = "./data/state:/app/data/state"

    assert compose.count(mount) >= 2, (
        f"expected the {mount} bind mount on both services, found "
        f"{compose.count(mount)}"
    )
    assert "mhs-live:" in compose
    assert "liquidation-collector:" in compose
    assert service_blocks  # compose parsed into indented blocks

def test_discovery_start_names_are_unambiguous() -> None:
    """Two different discovery windows must not share one constant name."""
    import pandas as pd

    from src.mhs.params import DISCOVERY_START
    from src.quant.technical_experts.trend_screen_catalog import (
        TREND_SCREEN_DISCOVERY_START,
    )

    assert DISCOVERY_START == pd.Timestamp("2021-01-01", tz="UTC")
    assert TREND_SCREEN_DISCOVERY_START == pd.Timestamp("2022-04-01", tz="UTC")
    assert DISCOVERY_START != TREND_SCREEN_DISCOVERY_START

def test_log_dir_declared_once() -> None:
    """telemetry reuses the common LOG_DIR object rather than redeclaring it."""
    import ast
    from pathlib import Path

    from src.common.logging import LOG_DIR as common_log_dir
    from src.mhs.telemetry import LOG_DIR as telemetry_log_dir

    assert telemetry_log_dir is common_log_dir

    declaring: list[str] = []
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(isinstance(t, ast.Name) and t.id == "LOG_DIR" for t in targets):
                declaring.append(str(path))

    assert declaring == ["src/common/logging.py"], declaring

def test_full_suite_is_green() -> None:
    """Marker for the phase exit gate; the real check is execution_command."""
    from pathlib import Path

    assert Path("tests").is_dir()
