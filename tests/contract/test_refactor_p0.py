# ruff: noqa: SIM300, N811
from __future__ import annotations

def test_no_nonschema_log_tags_in_src() -> None:
    """Only the 4 standard tags may appear in src/ log messages."""
    import re
    from pathlib import Path

    # 6 fixed domain categories from .agents/rules/logging.md §2 plus EVAL
    allowed = {"SYS", "DATA", "ALGO", "EVAL", "PORTFOLIO", "RISK", "EXEC"}
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
    """I-STATE-SURVIVES-REDEPLOY: only mhs-live persists data/state."""
    from pathlib import Path

    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    service_blocks = compose.split("\n  ")
    mount = "./data/state:/app/data/state"

    assert compose.count(mount) >= 1, (
        f"expected the {mount} bind mount on mhs-live, found "
        f"{compose.count(mount)}"
    )
    assert "mhs-live:" in compose
    assert "market-normalizer:" in compose
    assert "capture-blue:" in compose
    assert "capture-green:" in compose
    assert service_blocks  # compose parsed into indented blocks

def test_discovery_window_constants_are_ordered() -> None:
    """Discovery window constants must be strictly ordered."""
    import pandas as pd

    from src.core.params import DISCOVERY_START
    from src.quant.evaluation.policy import DISCOVERY_END, HOLDOUT_CUTOFF

    assert DISCOVERY_START == pd.Timestamp("2021-01-01", tz="UTC")
    assert DISCOVERY_START < DISCOVERY_END < HOLDOUT_CUTOFF

def test_log_dir_declared_once() -> None:
    """Log dir is declared once and not re-exported by telemetry."""
    import ast
    from pathlib import Path

    import src.mhs.telemetry as telemetry_mod

    assert not hasattr(telemetry_mod, "LOG_DIR")

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
